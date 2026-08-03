import pinocchio
import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable
from scipy.spatial.transform import Rotation as R
import logging
import os
import timeit
from DoorOpening.utils.pose_utils import wrap_to_pi

logger = logging.getLogger(__name__)

EPS_IK_CORRECT = 0.05

class CEM:
    """class implementing generic CEM solver for optimization"""

    def __init__(
        self,
        max_iterations: int,
        num_samples: int,
        num_top: int,
        tol: float,
        sigma0: np.ndarray,
    ):
        """
        max_iterations: max number of iterations
        num_samples: number of samples per iteration
        num_top: number of top samples to use for next iteration
        tol: tolerance for stopping criterion
        """
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.num_top = num_top
        self.cost_tol = tol
        self.sigma0 = sigma0

    def optimize(self, func: Callable, x0: np.ndarray):
        """optimize function func with initial guess mu=x0 and initial std=sigma0"""
        assert (
            x0.shape == self.sigma0.shape
        ), f"x0 and sigma0 must have same shape, got {x0.shape} and {self.sigma0.shape}"

        i = 0
        mu = x0
        sigma = self.sigma0

        while True:
            # Sample x
            x_arr = mu + sigma * np.random.randn(self.num_samples, x0.shape[0])

            # Compute costs
            cost_arr = np.zeros(self.num_samples)
            aux_outputs = [None for _ in range(self.num_samples)]
            for j, x in enumerate(x_arr):
                cost_arr[j], aux_outputs[j] = func(x)

            # Sort costs
            idx_sorted_arr = np.argsort(cost_arr)
            i_best = idx_sorted_arr[0]

            # Check termination
            i += 1
            if i >= self.max_iterations or np.all(sigma <= self.cost_tol / 10):
                # If we have run out of iterations or if our sigma has converged before getting close enough, per our
                # error tolerances, then the optimization failed
                success = False
                break

            if cost_arr[i_best] <= self.cost_tol:
                success = True
                break

            # Update distribution
            mu = np.mean(x_arr[idx_sorted_arr[: self.num_top], :], axis=0)
            sigma = np.std(x_arr[idx_sorted_arr[: self.num_top], :], axis=0)

        return cost_arr[i_best], aux_outputs[i_best], i, sigma, success

class PinocchioIKSolver():
    """IK solver using pinocchio which can handle end-effector constraints for optimized IK solutions"""

    # Convergence tolerance on the 6-D pose-error norm (position [m] + orientation [rad]). 1e-4
    # (~0.1 mm / 1e-4 rad) was far tighter than a damped-least-squares solve realistically floors
    # at (~1e-3), so visually-fine grasps were being flagged "not converged" and spuriously
    # triggering the random-restart fallback. 1e-2 (~1 cm / ~0.5 deg) is plenty accurate for
    # grasping/pushing and matches what looks converged. (Test tolerance EPS_IK_CORRECT=0.05.)
    EPS = 1e-2
    DT = 1e-1
    DAMP = 1e-4

    def __init__(
        self,
        urdf_path: str,
        ee_link_name: str,
        controlled_joints: List[str],
        verbose: bool = False,
        reference_joint_pos: Optional[Union[np.ndarray, Dict[str, float]]] = None,
        reference_joint_gain: float = 0.05,
    ):
        """
        urdf_path: path to urdf file
        ee_link_name: name of the end-effector link
        controlled_joints: list of joint names to control
        reference_joint_pos: optional joint configuration to bias redundant IK solutions toward.
            Dict inputs may contain any subset of model joint names.
        reference_joint_gain: null-space gain for the reference joint bias.
        """
        if verbose:
            print(f"{urdf_path=}")
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_neutral = pinocchio.neutral(self.model)

        # Cache the frame-name list once. get_frame_pose/compute_fk used to rebuild this list
        # (and linear-scan it with .index()) on every call, which is O(nframes) per lookup and
        # dominated the per-frame FK cost when replaying a full trajectory. _frame_index() below
        # preserves the exact first-occurrence semantics of the old [...].index(name) while
        # memoizing the result.
        self._frame_names = [f.name for f in self.model.frames]
        self._frame_idx_cache = {}
        # print(self._frame_names)
        self.ee_frame_idx = self._frame_index(ee_link_name)

        self.controlled_joints_by_name = {}
        self.controlled_joints = []
        self.controlled_joint_names = controlled_joints
        self.arm_joint_indices = []
        for joint in controlled_joints:
            if joint == "ignore":
                idx = -1
            else:
                jid = self.model.getJointId(joint)
                if jid >= len(self.model.idx_qs):
                    logger.error(f"{joint=} {jid=} not in model.idx_qs")
                    raise RuntimeError(f"Invalid urdf at {urdf_path=}: missing {joint=}")
                else:
                    idx = self.model.idx_qs[jid]
            self.controlled_joints.append(idx)
            self.controlled_joints_by_name[joint] = idx
            if idx >= 0 and joint not in ["base_x_joint", "base_y_joint", "base_rotation_joint"]:
                self.arm_joint_indices.append(idx)

        # arm_joint_indices come from idx_qs (configuration indices) but are used below to
        # index the frame Jacobian columns and v_full, which are velocity (nv) indexed. That
        # aliasing is only valid when idx_q == idx_v for every joint, i.e. nq == nv (no
        # free-flyer / planar-base / cos-sin-encoded continuous joint anywhere in the model).
        # Fail loudly rather than silently drive the wrong DOFs if that ever changes.
        assert self.model.nq == self.model.nv, (
            f"PinocchioIKSolver assumes nq == nv (got nq={self.model.nq}, nv={self.model.nv}); "
            "a free-flyer/planar/continuous joint would misalign arm_joint_indices between "
            "configuration space and velocity space."
        )

        self.has_joint_reference = reference_joint_pos is not None
        self.reference_joint_gain = reference_joint_gain if self.has_joint_reference else 0.0
        self.q_ref = self.q_neutral.copy()
        if reference_joint_pos is not None:
            self.q_ref = self._qmap_control2model(
                reference_joint_pos,
                ignore_missing_joints=True,
            )

        logger.info(f"{controlled_joints=}")
        for j in controlled_joints:
            idx = self.model.getJointId(j)
            idx_q = self.model.idx_qs[idx]
            logger.info(f"{j=} {idx=} {idx_q=}")

    def get_dof(self) -> int:
        """returns dof for the manipulation chain"""
        return len(self.controlled_joints)

    def get_num_controllable_joints(self) -> int:
        """returns number of controllable joints under this solver's purview"""
        return len(self.controlled_joints)

    def get_all_joint_names(self) -> List[str]:
        """Return a list of joints"""
        return [self.model.names[i + 1] for i in range(self.model.nq)]

    def _qmap_control2model(
        self, q_input: Union[np.ndarray, dict], ignore_missing_joints: bool = False
    ) -> np.ndarray:
        """returns a full joint configuration from a partial joint configuration"""
        q_out = self.q_neutral.copy()
        if isinstance(q_input, dict):
            for joint_name, value in q_input.items():
                if joint_name in self.controlled_joints_by_name:
                    q_out[self.controlled_joints_by_name[joint_name]] = value
                else:
                    jid = self.model.getJointId(joint_name)
                    if jid >= len(self.model.idx_qs):
                        if not ignore_missing_joints:
                            logger.error(f"ERROR: {joint_name=} {jid=} not in model.idx_qs")
                            raise RuntimeError(
                                f"Tried to set joint not in model.idx_qs: {joint_name=}"
                            )
                    else:
                        q_out[self.model.idx_qs[self.model.getJointId(joint_name)]] = value
        else:
            assert len(self.controlled_joints) == len(
                q_input
            ), "if not specifying by name, must match length"
            for i, joint_idx in enumerate(self.controlled_joints):
                q_out[joint_idx] = q_input[i]
        return q_out

    def _qmap_model2control(self, q_input: np.ndarray) -> np.ndarray:
        """returns a partial joint configuration from a full joint configuration"""
        q_out = np.empty(len(self.controlled_joints))
        for i, joint_idx in enumerate(self.controlled_joints):
            if joint_idx >= 0:
                q_out[i] = q_input[joint_idx]

        return q_out

    def _frame_index(self, name: str) -> int:
        """First-occurrence frame index for `name`, memoized. Equivalent to
        [f.name for f in self.model.frames].index(name) but without rebuilding/scanning the
        list on every call."""
        idx = self._frame_idx_cache.get(name)
        if idx is None:
            idx = self._frame_names.index(name)
            self._frame_idx_cache[name] = idx
        return idx

    def get_frames_pose_batch(
        self,
        config: Union[np.ndarray, dict],
        node_a_list: List[str],
        node_b: str,
        ignore_missing_joints: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pose of each frame in `node_a_list` expressed in the `node_b` frame, for a single
        configuration.

        This is the batched analogue of calling get_frame_pose() once per node_a: it runs a
        single forwardKinematics + updateFramePlacements (instead of one per body) and reuses
        cached frame indices, so replaying a whole trajectory over K key bodies costs one FK per
        frame instead of K.

        Returns:
            translations: (K, 3) float64 array
            rotations:    (K, 3, 3) float64 array
        """
        q_model = self._qmap_control2model(config, ignore_missing_joints=ignore_missing_joints)
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        pinocchio.updateFramePlacements(self.model, self.data)

        oMb_inv = self.data.oMf[self._frame_index(node_b)].inverse()

        translations = np.empty((len(node_a_list), 3))
        rotations = np.empty((len(node_a_list), 3, 3))
        for k, node_a in enumerate(node_a_list):
            bMa = oMb_inv * self.data.oMf[self._frame_index(node_a)]
            translations[k] = bMa.translation
            rotations[k] = bMa.rotation
        return translations, rotations

    def get_frame_pose(
        self,
        config: Union[np.ndarray, dict],
        node_a: str,
        node_b: str,
        ignore_missing_joints: bool = False,
    ) -> np.ndarray:
        """
        Get a transformation matrix transforming from node_a frame to node_b frame

        Args:
            config: joint values
            node_a: name of the first node
            node_b: name of the second node
            ignore_missing_joints: whether to ignore missing joints in the configuration

        Returns:
            transformation matrix from node_a to node_b
        """
        q_model = self._qmap_control2model(config, ignore_missing_joints=ignore_missing_joints)
        # print('q_model', q_model)
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        frame_idx1 = self._frame_index(node_a)
        frame_idx2 = self._frame_index(node_b)
        # print(frame_idx1)
        # print(frame_idx2)
        # print(self.model.getFrameId(node_a))
        # print(self.model.getFrameId(node_b))
        # frame_idx1 = self.model.getFrameId(node_a)
        # frame_idx2 = self.model.getFrameId(node_b)
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx1)
        placement_frame1 = self.data.oMf[frame_idx1]
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx2)
        placement_frame2 = self.data.oMf[frame_idx2]
        # print('pin 1', placement_frame1)
        # print('pin 2', placement_frame2)
        return placement_frame2.inverse() * placement_frame1

    def compute_fk(
        self, config: np.ndarray, link_name: str = None, ignore_missing_joints: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given joint values, return end-effector position and quaternion associated with it.

        Args:
            config: joint values
            link_name: name of the link to compute FK for; if None, uses the end-effector link

        Returns:
            pos: end-effector position (x, y, z)
            quat: end-effector quaternion (x, y, z, w)  # scipy convention, scalar-last
        """
        if link_name is None:
            frame_idx = self.ee_frame_idx
        else:
            try:
                frame_idx = self._frame_index(link_name)
            except ValueError:
                logger.error(f"Unknown link_name {link_name}. Defaulting to end-effector")
                frame_idx = self.ee_frame_idx
        q_model = self._qmap_control2model(config, ignore_missing_joints=ignore_missing_joints)
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx)
        pos = self.data.oMf[frame_idx].translation
        quat = R.from_matrix(self.data.oMf[frame_idx].rotation).as_quat()
        return pos.copy(), quat.copy()

    def compute_ik(
        self,
        pos_desired: Optional[np.ndarray]=None,
        quat_desired: Optional[np.ndarray]=None,
        q_init=None,
        max_iterations=1000,
        num_attempts: int = 1,
        verbose: bool = False,
        ignore_missing_joints: bool = False,
        custom_ee_frame: Optional[str] = None,
    ) -> Tuple[np.ndarray, bool, dict]:
        """given end-effector position and quaternion, return joint values.

        q_init seeds the optimization when provided. If the solver was constructed with
        reference_joint_pos, that configuration is used as the seed when q_init is absent and
        as a null-space preference for redundant arm joints.

        Two parameters are kept for compatibility with pybullet-style IK solvers:
            num_attempts: start from multiple initial configs
            max_iterations: time budget in number of steps
        """
        if custom_ee_frame is not None:
            _ee_frame_idx = self._frame_index(custom_ee_frame)
        else:
            _ee_frame_idx = self.ee_frame_idx

        assert pos_desired is not None or quat_desired is not None, "Either pos_desired or quat_desired must be provided"
        assert (pos_desired is not None and quat_desired is not None) or q_init is not None, "if pos_desired or quat_desired is not provided, q_init must be provided"
        if q_init is not None:
            pos, quat = self.compute_fk(q_init)
            pos_desired = pos if pos_desired is None else pos_desired
            quat_desired = quat if quat_desired is None else quat_desired

        # ---- Build the list of seed configurations to try ----
        if q_init is not None:
            # Attempt 0 is the caller's seed (keeps the natural, continuous pose when it solves,
            # since the loop returns the FIRST converged seed). If it FAILS to converge, fall back
            # to extra attempts from randomized arm configs (base joints kept) so a reachable
            # target isn't abandoned just because the caller's seed sat in a bad/singular basin.
            seed0 = self._qmap_control2model(q_init, ignore_missing_joints=ignore_missing_joints)
            seeds = [seed0]
            for _ in range(max(0, int(num_attempts) - 1)):
                seeds.append(self._random_arm_seed(seed0))
        else:
            base_seed = self.q_ref.copy() if self.has_joint_reference else self.q_neutral.copy()
            # attempt 0 = reference/neutral seed (previous behavior); the remaining attempts are
            # random configs sampled uniformly within the finite arm joint limits, so a target
            # that is unreachable from one (possibly singular) seed can still be found.
            seeds = [base_seed]
            for _ in range(max(0, int(num_attempts) - 1)):
                seeds.append(self._random_arm_seed(base_seed))

        desired_ee_pose = pinocchio.SE3(R.from_quat(quat_desired).as_matrix(), pos_desired)

        # Track the least-bad iterate across ALL iterations and attempts, so a non-converged
        # call returns that instead of the last (possibly wild) iterate.
        best_q = None
        best_err = None
        best_err_norm = np.inf
        best_iter = 0

        for q in seeds:
            q = q.copy()
            i = 0
            success = False
            while True:
                pinocchio.forwardKinematics(self.model, self.data, q)
                pinocchio.updateFramePlacement(self.model, self.data, _ee_frame_idx)
                dMi = desired_ee_pose.actInv(self.data.oMf[_ee_frame_idx])
                err = pinocchio.log(dMi).vector
                err_norm = np.linalg.norm(err)
                if err_norm < best_err_norm:
                    best_err_norm = err_norm
                    best_err = err
                    best_q = q.copy()
                    best_iter = i
                if verbose:
                    print(f"[pinocchio_ik_solver] iter={i}; error={err}")
                if err_norm < self.EPS:
                    success = True
                    break
                if i >= max_iterations:
                    success = False
                    break
                J = pinocchio.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    _ee_frame_idx,
                    pinocchio.ReferenceFrame.LOCAL,
                )
                J_arm = J[:, self.arm_joint_indices]
                damping = J_arm.dot(J_arm.T) + self.DAMP * np.eye(J_arm.shape[0])
                v = -J_arm.T.dot(np.linalg.solve(damping, err))
                if self.reference_joint_gain > 0.0 and len(self.arm_joint_indices) > 0:
                    J_arm_pinv = np.linalg.pinv(J_arm, rcond=1e-4)
                    nullspace = np.eye(len(self.arm_joint_indices)) - J_arm_pinv.dot(J_arm)
                    q_ref_delta = self.q_ref[self.arm_joint_indices] - q[self.arm_joint_indices]
                    q_ref_delta = (q_ref_delta + np.pi) % (2 * np.pi) - np.pi
                    v += nullspace.dot(self.reference_joint_gain * q_ref_delta)
                # v = v.clip(-0.1, 0.1)
                v_full = np.zeros(self.model.nv)
                for k, idx in enumerate(self.arm_joint_indices):
                    v_full[idx] = v[k]
                q = pinocchio.integrate(self.model, q, v_full * self.DT)
                # Clamp the integrated CONFIGURATION (never the velocity) to the model joint
                # limits every DLS step so the arm cannot walk past its limits into a
                # contorted/folded pose. Base prismatic/revolute limits are wide (+-5 m / +-1e5
                # rad) so this is a no-op on the (externally set, unmoved) base in practice.
                q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
                i += 1

            if success:
                # Return the first converged solution immediately (unchanged success behavior).
                q_control = self._qmap_model2control(q.flatten())
                debug_info = {"iter": i, "final_error": err, "attempts": len(seeds)}
                return q_control, True, debug_info

        # No attempt converged: return the least-bad iterate, not the last wild one.
        q_control = self._qmap_model2control(best_q.flatten())
        debug_info = {
            "iter": best_iter,
            "final_error": best_err,
            "attempts": len(seeds),
            "best_error_norm": best_err_norm,
        }
        return q_control, False, debug_info

    def _random_arm_seed(self, base_seed: np.ndarray) -> np.ndarray:
        """Copy of base_seed with the controlled ARM joints randomized uniformly within their
        finite position limits (base / non-arm joints are left untouched)."""
        q = base_seed.copy()
        lo = self.model.lowerPositionLimit
        hi = self.model.upperPositionLimit
        for idx in self.arm_joint_indices:
            if np.isfinite(lo[idx]) and np.isfinite(hi[idx]):
                q[idx] = np.random.uniform(lo[idx], hi[idx])
        return q

    def q_array_to_dict(self, arr: np.ndarray):
        state = {}
        assert len(arr) == len(self.controlled_joint_names)
        for i, name in enumerate(self.controlled_joint_names):
            state[name] = arr[i]
        return state


class PositionIKOptimizer():
    """
    Solver that jointly optimizes IK and best orientation to achieve desired position.
    """

    max_iterations: int = 30  # Max num of iterations for CEM
    num_samples: int = 100  # Total candidate samples for each CEM iteration
    num_top: int = 10  # Top N candidates for each CEM iteration

    def __init__(
        self,
        ik_solver,
        pos_error_tol: float,
        ori_error_range: Union[float, np.ndarray],
        pos_weight: float = 1.0,
        ori_weight: float = 0.0,
        cem_params: Optional[Dict] = None,
    ):
        self.pos_wt = pos_weight
        self.ori_wt = ori_weight

        # Initialize IK solver
        self.ik_solver = ik_solver

        # Initialize optimizer
        self.pos_error_tol = pos_error_tol
        if type(ori_error_range) is float:
            self.ori_error_range = ori_error_range * np.ones(3)
        else:
            self.ori_error_range = ori_error_range  # type: ignore

        cem_params = {} if cem_params is None else cem_params
        max_iterations = (
            cem_params["max_iterations"] if "max_iterations" in cem_params else self.max_iterations
        )
        num_samples = cem_params["num_samples"] if "num_samples" in cem_params else self.num_samples
        num_top = cem_params["num_top"] if "num_top" in cem_params else self.num_top

        self.opt = CEM(
            max_iterations=max_iterations,
            num_samples=num_samples,
            num_top=num_top,
            tol=self.pos_error_tol,
            sigma0=self.ori_error_range / 2,
        )

    def get_dof(self) -> int:
        return self.ik_solver.get_dof()

    def get_num_controllable_joints(self) -> int:
        return self.ik_solver.get_num_controllable_joints()

    def compute_ik(
        self,
        pos_desired: np.ndarray,
        quat_desired: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, bool, dict]:
        """optimization-based IK solver using CEM"""

        # Function to optimize: IK error given delta from original desired orientation
        def solve_ik(dr):
            pos = pos_desired
            quat = (R.from_rotvec(dr) * R.from_quat(quat_desired)).as_quat()

            q, _, subsolver_debug_info = self.ik_solver.compute_ik(pos, quat)
            pos_out, rot_out = self.ik_solver.compute_fk(q)

            cost_pos = np.linalg.norm(pos - pos_out)
            cost_rot = 1 - (rot_out * quat_desired).sum() ** 2  # TODO: just minimize dr?

            cost = self.pos_wt * cost_pos + self.ori_wt * cost_rot

            return cost, q

        # Optimize for IK and best orientation (x=0 -> use original desired orientation)
        cost_opt, q_result, max_iter, opt_sigma, success = self.opt.optimize(
            solve_ik, x0=np.zeros(3)
        )
        pos_out, quat_out = self.ik_solver.compute_fk(q_result)
        print(
            f"After ik optimization, cost: {cost_opt}, result: {pos_out, quat_out} vs desired: {pos_desired, quat_desired}"
        )

        debug_info = {
            "best_cost": cost_opt,
            "last_iter": max_iter,
            "opt_sigma": opt_sigma,
        }

        return q_result, success, debug_info

    def compute_fk(self, q):
        return self.ik_solver.compute_fk(q)

def test_fk_ik(urdf_file, ee_link_name, true_joint_names, initial_joint_state):
    # Create IK Solver
    try:
        urdf_path = os.path.join(os.path.dirname(__file__), urdf_file)
    except FileNotFoundError as e:
        print(e)
        assert False, "URDF file not found!"

    print("URDF path =", urdf_path)
    ik_joints_allowed_to_move = initial_joint_state.keys()

    manip_ik_solver = PinocchioIKSolver(
        urdf_path,
        ee_link_name,
        ik_joints_allowed_to_move,
    )

    all_joints = manip_ik_solver.get_all_joint_names()
    for i, j in zip(all_joints, true_joint_names):
        assert i == j, f"Joint name mismatch: {i} != {j}"

    # Test Forward Kinematics
    ee_pose = manip_ik_solver.compute_fk(initial_joint_state)
    print(f"{ee_pose=}")
    assert ee_pose is not None, "FK failed"

    # Test Inverse Kinematics
    # ee_position = np.array([-0.03, -0.4, 0.9])
    # ee_orientation = np.array([0, 0, 0, 1])
    ee_position = ee_pose[0]
    ee_orientation = ee_pose[1]
    res, success, info = manip_ik_solver.compute_ik(
        ee_position,
        ee_orientation,
        q_init=initial_joint_state,
    )
    print("Result =", res)
    print("Success =", success)
    assert success, "IK failed"

    # Test IK accuracy
    res_ee_position, res_ee_orientation = manip_ik_solver.compute_fk(res)
    ee_position_error = np.linalg.norm(res_ee_position - ee_position)
    ee_orientation_error = np.linalg.norm(res_ee_orientation - ee_orientation)
    assert ee_position_error < EPS_IK_CORRECT, "IK position error too large"
    assert ee_orientation_error < EPS_IK_CORRECT, "IK orientation error too large"

    dt_sum = 0
    # Speed test
    for i in range(1000):
        # Test Inverse Kinematics
        t0 = timeit.default_timer()
        ee_position = np.random.rand(3) * 2 - 1
        ee_position[2] += 1
        ee_orientation = np.random.rand(4)
        ee_orientation /= np.linalg.norm(ee_orientation)
        res, success, info = manip_ik_solver.compute_ik(
            ee_position,
            ee_orientation,
            q_init=initial_joint_state,
        )
        t1 = timeit.default_timer()
        dt_sum += t1 - t0
    print("Average time per IK call =", dt_sum / 1000)
    hz = 1 / (dt_sum / 1000)
    print("Average rate =", hz)
    assert hz > 100, "IK solver too slow"


def test_ik_reachable_within_limits(
    urdf_file,
    ee_link_name,
    controlled_joints,
    ready_pose,
    num_targets: int = 25,
    seed: int = 0,
):
    """Regression guard against the folded/contorted-pose bug.

    Commands a set of *reachable* EE poses (each is FK of a small random perturbation of the
    Franka ready pose, so a valid solution provably exists near the seed) and asserts that IK
    (a) reports success=True and (b) returns a configuration inside the model joint limits.
    """
    urdf_path = os.path.join(os.path.dirname(__file__), urdf_file)
    solver = PinocchioIKSolver(
        urdf_path, ee_link_name, controlled_joints, reference_joint_pos=ready_pose
    )
    lo = solver.model.lowerPositionLimit
    hi = solver.model.upperPositionLimit
    rng = np.random.default_rng(seed)

    ready = {"base_x_joint": 0.0, "base_y_joint": 0.0, "base_rotation_joint": 0.0}
    ready.update(ready_pose)

    for t in range(num_targets):
        perturbed = dict(ready)
        for j, v in ready_pose.items():
            perturbed[j] = v + float(rng.uniform(-0.3, 0.3))
        target_pos, target_quat = solver.compute_fk(perturbed)

        res, success, info = solver.compute_ik(target_pos, target_quat, q_init=ready)
        assert success, f"IK failed on reachable target #{t}: {info}"

        q_model = solver._qmap_control2model(res)
        assert np.all(q_model >= lo - 1e-6) and np.all(q_model <= hi + 1e-6), (
            f"IK result outside joint limits on target #{t}"
        )
    print(f"test_ik_reachable_within_limits PASSED ({num_targets} reachable targets)")


if __name__ == "__main__":
    _urdf = "/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/glorbot/glorbot.urdf"
    _ee = "palm_center"
    _names = ["base_x_joint", "base_y_joint", "base_rotation_joint", "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
    # Standard Franka ready pose (joint names read from the URDF).
    _ready = {"panda_joint1": 0.0, "panda_joint2": -0.785, "panda_joint3": 0.0, "panda_joint4": -2.356, "panda_joint5": 0.0, "panda_joint6": 1.571, "panda_joint7": 0.785}

    print("Testing FK and IK...")
    test_fk_ik(urdf_file=_urdf, ee_link_name=_ee, true_joint_names=_names, initial_joint_state={n: 0.0 for n in _names})
    print("Testing IK reachability + joint limits...")
    test_ik_reachable_within_limits(urdf_file=_urdf, ee_link_name=_ee, controlled_joints=_names, ready_pose=_ready)
