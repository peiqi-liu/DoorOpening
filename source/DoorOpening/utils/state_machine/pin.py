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

    EPS = 1e-4
    DT = 1e-1
    DAMP = 1e-4

    def __init__(
        self, urdf_path: str, ee_link_name: str, controlled_joints: List[str], verbose: bool = False
    ):
        """
        urdf_path: path to urdf file
        ee_link_name: name of the end-effector link
        controlled_joints: list of joint names to control
        """
        if verbose:
            print(f"{urdf_path=}")
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_neutral = pinocchio.neutral(self.model)

        # print([f.name for f in self.model.frames])
        self.ee_frame_idx = [f.name for f in self.model.frames].index(ee_link_name)

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
            if joint not in ["base_x_joint", "base_y_joint", "base_rotation_joint"]:
                self.arm_joint_indices.append(idx)

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
        frame_idx1 = [f.name for f in self.model.frames].index(node_a)
        frame_idx2 = [f.name for f in self.model.frames].index(node_b)
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
            quat: end-effector quaternion (w, x, y, z)
        """
        if link_name is None:
            frame_idx = self.ee_frame_idx
        else:
            try:
                frame_idx = [f.name for f in self.model.frames].index(link_name)
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

        Two parameters are currently unused and might be implemented in the future:
            q_init: initial configuration for the optimization to start in; especially useful for
                    arms with redundant degrees of freedom
            num_attempts: start from multiple initial configs; included for compatibility with pb
            max iterations: time budget in number of steps; included for compatibility with pb
        """
        i = 0
        if custom_ee_frame is not None:
            _ee_frame_idx = [f.name for f in self.model.frames].index(custom_ee_frame)
        else:
            _ee_frame_idx = self.ee_frame_idx
        
        assert pos_desired is not None or quat_desired is not None, "Either pos_desired or quat_desired must be provided"
        assert (pos_desired is not None and quat_desired is not None) or q_init is not None, "if pos_desired or quat_desired is not provided, q_init must be provided"
        if q_init is not None:
            pos, quat = self.compute_fk(q_init)
            pos_desired = pos if pos_desired is None else pos_desired
            quat_desired = quat if quat_desired is None else quat_desired

        if q_init is None:
            q = self.q_neutral.copy()
            if num_attempts > 1:
                raise NotImplementedError(
                    "Sampling multiple initial configs not yet supported by Pinocchio solver."
                )
        else:
            q = self._qmap_control2model(q_init, ignore_missing_joints=ignore_missing_joints)
            # Override the number of attempts
            num_attempts = 1
            if pos_desired is None or quat_desired is None:
                pos, quat = self.compute_fk(q_init)
                pos_desired = pos if pos_desired is None else pos_desired
                quat_desired = quat if quat_desired is None else quat_desired

        desired_ee_pose = pinocchio.SE3(R.from_quat(quat_desired).as_matrix(), pos_desired)
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacement(self.model, self.data, _ee_frame_idx)
            dMi = desired_ee_pose.actInv(self.data.oMf[_ee_frame_idx])
            err = pinocchio.log(dMi).vector
            if verbose:
                print(f"[pinocchio_ik_solver] iter={i}; error={err}")
            if np.linalg.norm(err) < self.EPS:
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
            v = -J_arm.T.dot(np.linalg.solve(J_arm.dot(J_arm.T) + self.DAMP * np.eye(6), err))
            # v = v.clip(-0.1, 0.1)
            v_full = np.zeros(self.model.nv)

            for k, idx in enumerate(self.arm_joint_indices):
                v_full[idx] = v[k]
            q = pinocchio.integrate(self.model, q, v_full * self.DT)
            # q[3:] = wrap_to_pi(q[3:])
            i += 1

        q_control = self._qmap_model2control(q.flatten())
        debug_info = {"iter": i, "final_error": err}

        return q_control, success, debug_info

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


if __name__ == "__main__":
    print("Testing FK and IK...")
    test_fk_ik(urdf_file="/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/glorbot/glorbot.urdf", ee_link_name="palm_center", true_joint_names=["base_x_joint", "base_y_joint", "base_rotation_joint", "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"], initial_joint_state={"base_x_joint": 0.0, "base_y_joint": 0.0, "base_rotation_joint": 0.0, "panda_joint1": 0.0, "panda_joint2": 0.0, "panda_joint3": 0.0, "panda_joint4": 0.0, "panda_joint5": 0.0, "panda_joint6": 0.0, "panda_joint7": 0.0})
