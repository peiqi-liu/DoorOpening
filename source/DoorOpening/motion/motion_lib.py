import torch
import pickle as pkl
from typing import Optional, Sequence
from DoorOpening.utils.pose_utils import normalize_to_center_frame

class ReferenceMotionManager:
    def __init__(
        self,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
        reset_from_start=False,
        env_to_file_map: Optional[list] = None,
        twist_indices: Optional[list] = None,
        step_dt: Optional[float] = None,
        frame_dt: float = 1.0 / 40.0,
    ):
        self.device = device
        self.num_envs = num_envs
        self.reset_from_start = reset_from_start
        self.frame_dt = max(float(frame_dt), 1e-6)
        self.step_dt = max(float(step_dt if step_dt is not None else self.frame_dt), 1e-6)
        self._loaded_motion_dt = None
        # Named as twist indices because we borrow the idea from Twist paper
        self.twist_indices = twist_indices

        if env_to_file_map is None:
            raise ValueError("env_to_file_map must be provided; ReferenceMotionManager now always loads the motion list.")
        self._load_motion_pkl_from_list()
        self.env_to_file_map = torch.tensor(env_to_file_map, device=self.device)
        self.frame_step = self.step_dt / self.frame_dt
        self._init_env_buffers()
        if self.twist_indices is not None:
            self._precompute_twist()

    def _extract_first_keyframe(self, key_indices, num_frames: int) -> int:
        if isinstance(key_indices, torch.Tensor):
            key_values = key_indices.flatten().tolist()
        elif key_indices is None:
            key_values = []
        else:
            key_values = list(key_indices)

        if len(key_values) >= 2:
            first_keyframe = int(key_values[1])
        elif len(key_values) == 1:
            first_keyframe = int(key_values[0])
        else:
            first_keyframe = 0

        max_idx = max(int(num_frames) - 1, 0)
        return max(0, min(first_keyframe, max_idx))

    # --------------------------------------------------
    # Load motion data (moved from Env)
    # --------------------------------------------------
    def _load_motion_pkl_from_list(self):
        from DoorOpening.assets.door.door_cfg import asset_base_folder, motion_traj_paths

        if len(motion_traj_paths) == 0:
            raise FileNotFoundError(f"No traj.pkl files found under {asset_base_folder}")

        robot_joint_pos_trajs = []
        door_trajs = []
        robot_body_pos_trajs = []
        robot_body_quat_trajs = []
        robot_joint_vel_trajs = []
        key_indices_list = []
        first_keyframes = []
        hinge_contact_masks_list = []
        robot_body_pos_vel_list = []
        door_body_pos_trajs = []
        for motion_file in motion_traj_paths:
            loaded_motion = self._load_motion_pkl(motion_file)
            if len(loaded_motion) == 10:
                (
                    robot_joint_pos_traj,
                    door_traj,
                    robot_body_pos_traj,
                    robot_body_quat_traj,
                    robot_joint_vel_traj,
                    key_indices,
                    self.num_frames,
                    hinge_contact_mask,
                    robot_body_pos_vel,
                    door_body_pos_traj,
                ) = loaded_motion
            else:
                (
                    robot_joint_pos_traj,
                    door_traj,
                    robot_body_pos_traj,
                    robot_body_quat_traj,
                    robot_joint_vel_traj,
                    key_indices,
                    self.num_frames,
                    robot_body_pos_vel,
                    door_body_pos_traj,
                ) = loaded_motion
                hinge_contact_mask = torch.zeros(self.num_frames, device=self.device)

            robot_joint_pos_trajs.append(robot_joint_pos_traj)
            door_trajs.append(door_traj)
            robot_body_pos_trajs.append(robot_body_pos_traj)
            robot_body_quat_trajs.append(robot_body_quat_traj)
            robot_joint_vel_trajs.append(robot_joint_vel_traj) 
            if isinstance(key_indices, list):
                key_indices = torch.tensor(key_indices, device=self.device)
            key_indices_list.append(key_indices)
            first_keyframes.append(self._extract_first_keyframe(key_indices, robot_joint_pos_traj.shape[0]))
            hinge_contact_masks_list.append(hinge_contact_mask)
            robot_body_pos_vel_list.append(robot_body_pos_vel)
            door_body_pos_trajs.append(door_body_pos_traj)
        # stack motions: [M, T, ...]
        self.motion_traj_paths = list(motion_traj_paths)
        self.robot_joint_pos_traj = torch.stack(robot_joint_pos_trajs, dim=0)
        self.robot_joint_vel_traj = torch.stack(robot_joint_vel_trajs, dim=0)
        self.robot_body_pos_traj = torch.stack(robot_body_pos_trajs, dim=0)
        self.robot_body_quat_traj = torch.stack(robot_body_quat_trajs, dim=0)
        self.door_traj = torch.stack(door_trajs, dim=0)
        # self.key_indices = torch.stack(key_indices_list, dim=0).to(self.device)
        # self.key_indices = self.key_indices[..., :-1] # remove the last key index
        self.key_indices = torch.arange(0, self.num_frames, 1).repeat(len(key_indices_list), 1).to(self.device).int()
        self.first_keyframe_idx = torch.tensor(first_keyframes, device=self.device, dtype=torch.float32)
        self.hinge_contact_mask = torch.stack(hinge_contact_masks_list, dim=0).to(self.device)
        self.num_motions = self.robot_joint_pos_traj.shape[0]
        self.robot_body_pos_vel = torch.stack(robot_body_pos_vel_list, dim=0).to(self.device)
        self.door_body_pos_traj = torch.stack(door_body_pos_trajs, dim=0).to(self.device)

    def _load_motion_pkl(self, motion_file: str):
        with open(motion_file, "rb") as f:
            motions = pkl.load(f)

        required_keys = [
            "robot_joint_pos_traj",
            "door_traj",
            "robot_body_pos_traj",
            "robot_body_quat_traj",
        ]
        for k in required_keys:
            assert k in motions, f"{k} not found in motion file"

        motion_dt = max(float(motions.get("sim_dt", self.frame_dt)), 1e-6)
        if self._loaded_motion_dt is None:
            self._loaded_motion_dt = motion_dt
            self.frame_dt = motion_dt
        elif abs(motion_dt - self._loaded_motion_dt) > 1e-6:
            raise ValueError(
                f"Inconsistent motion dt detected in '{motion_file}': {motion_dt} vs {self._loaded_motion_dt}."
            )

        robot_joint_pos_traj = motions["robot_joint_pos_traj"]
        door_traj = motions["door_traj"]
        robot_body_pos_traj = motions["robot_body_pos_traj"]
        robot_body_quat_traj = motions["robot_body_quat_traj"]
        robot_joint_vel_traj = motions["robot_joint_vel_traj"]
        key_indices = motions["key_indices"]
        if "hinge_contact_mask" in motions:
            hinge_contact_mask = motions["hinge_contact_mask"]
        else:
            hinge_contact_mask = None
        robot_body_pos_vel = motions["robot_body_pos_twist"]
        door_body_pos_traj = motions["door_body_pos_traj"]

        if isinstance(robot_joint_pos_traj, list):
            robot_joint_pos_traj = torch.stack(robot_joint_pos_traj, dim = 0)
        if isinstance(door_traj, list):
            door_traj = torch.stack(door_traj, dim = 0)
        if isinstance(robot_body_pos_traj, list):
            robot_body_pos_traj = torch.stack(robot_body_pos_traj, dim = 0)
        if isinstance(robot_body_quat_traj, list):
            robot_body_quat_traj = torch.stack(robot_body_quat_traj, dim = 0)
        if isinstance(robot_joint_vel_traj, list):
            robot_joint_vel_traj = torch.stack(robot_joint_vel_traj, dim = 0)
        if isinstance(robot_body_pos_vel, list):
            robot_body_pos_vel = torch.stack(robot_body_pos_vel, dim = 0)

        robot_joint_pos_traj = robot_joint_pos_traj.to(self.device).squeeze()
        door_traj = door_traj.to(self.device).squeeze()
        robot_body_pos_traj = robot_body_pos_traj.to(self.device).squeeze()
        robot_body_quat_traj = robot_body_quat_traj.to(self.device).squeeze()
        robot_joint_vel_traj = robot_joint_vel_traj.to(self.device).squeeze()
        robot_joint_vel_traj = self._finite_difference(robot_joint_pos_traj, motion_dt)
        if hinge_contact_mask is not None:
            hinge_contact_mask = hinge_contact_mask.to(self.device).squeeze()
        robot_body_pos_vel = robot_body_pos_vel.to(self.device).squeeze()
        door_body_pos_traj = door_body_pos_traj.to(self.device).squeeze()
        num_frames = robot_joint_pos_traj.shape[0]

        if hinge_contact_mask is not None:
            return robot_joint_pos_traj, door_traj, robot_body_pos_traj, robot_body_quat_traj, robot_joint_vel_traj, key_indices, num_frames, hinge_contact_mask, robot_body_pos_vel, door_body_pos_traj
        else:
            return robot_joint_pos_traj, door_traj, robot_body_pos_traj, robot_body_quat_traj, robot_joint_vel_traj, key_indices, num_frames, robot_body_pos_vel, door_body_pos_traj

    def _finite_difference(self, traj: torch.Tensor, dt: float) -> torch.Tensor:
        traj_d = torch.zeros_like(traj)
        if traj.shape[0] <= 1:
            return traj_d
        traj_d[:-1] = (traj[1:] - traj[:-1]) / max(float(dt), 1e-6)
        traj_d[-1] = traj_d[-2]
        return traj_d

    # --------------------------------------------------
    # Per-env buffers
    # --------------------------------------------------
    def _init_env_buffers(self):
        # current frame index per env
        self.frame_idx = torch.zeros(
            self.num_envs,
            device=self.device,
        ).float()

        # cached current-frame refs (avoid realloc every step)
        self.ref_robot_joint_pos = None
        self.ref_door_joint_pos = None
        self.ref_robot_body_pos = None
        self.ref_robot_body_quat = None

    def _precompute_twist(self):
        """
        Precompute the twist indices per frame.
        For each frame, add the frame index to all twist indices
        Input: twist indices: list of length 2N+1
        Output: set twist frames (robot_joint_pos_traj, door_traj, robot_body_pos_traj, robot_body_quat_traj, robot_joint_vel_traj) per frame: (traj_len, twist_len, ...) for each frame
        """
        device = self.device
        T = self.num_frames
        twist_offsets = torch.tensor(
            self.twist_indices, device=device, dtype=torch.long
        )  # (K,)
        K = twist_offsets.shape[0]

        # --------------------------------------------------
        # Compute all twist frame indices
        # --------------------------------------------------

        base_frames = torch.arange(T, device=device).unsqueeze(1)  # (T,1)

        twist_frames = base_frames + twist_offsets.unsqueeze(0)  # (T,K)
        twist_frames = twist_frames.clamp(0, T - 1)  # clamp boundary

        def precompute_multi(traj):
            """
            traj:
                (F, T, ...)
            returns:
                (F, T, K, ...)
            """

            F, T = traj.shape[:2]
            K = twist_frames.shape[1]

            # (F, T, K)
            index = twist_frames.unsqueeze(0).expand(F, -1, -1)

            # reshape index to match traj for gather
            # make index shape (F, T, K, 1, 1, ..., 1)
            extra_dims = traj.dim() - 2
            index = index.view(F, T, K, *([1] * extra_dims))

            # expand to full feature size
            index = index.expand(F, T, K, *traj.shape[2:])

            # expand traj to insert K dimension
            traj_expanded = traj.unsqueeze(2).expand(F, T, K, *traj.shape[2:])

            return torch.gather(traj_expanded, 1, index)

        self.robot_joint_pos_twist = precompute_multi(self.robot_joint_pos_traj)
        self.robot_joint_vel_twist = precompute_multi(self.robot_joint_vel_traj)
        self.door_joint_pos_twist  = precompute_multi(self.door_traj)
        self.robot_body_pos_twist  = precompute_multi(self.robot_body_pos_traj)
        self.robot_body_quat_twist = precompute_multi(self.robot_body_quat_traj)
        print("self.robot_body_pos_traj.shape: ", self.robot_body_pos_traj.shape)
        print("self.robot_body_quat_traj.shape: ", self.robot_body_quat_traj.shape)
        print("self.robot_body_pos_twist.shape: ", self.robot_body_pos_twist.shape)
        print("self.robot_body_quat_twist.shape: ", self.robot_body_quat_twist.shape)
        # self.robot_body_pos_twist, self.robot_body_quat_twist = normalize_to_center_frame(self.robot_body_pos_traj, self.robot_body_quat_traj, self.robot_body_pos_twist, self.robot_body_quat_twist)
        self.door_body_pos_twist = precompute_multi(self.door_body_pos_traj)
    # --------------------------------------------------
    # Reset logic
    # --------------------------------------------------
    def reset(self, env_ids: Sequence[int], step_count: Optional[int] = None, reset_progress_total: Optional[int] = None):
        probs = None
        if not self.reset_from_start:
            if step_count is not None and reset_progress_total is not None:
                progress = min(step_count / reset_progress_total, 1.0)
                # alpha = 0.9 - 0.7 * progress
                alpha = 1 - 0.1**(2 ** (2.0 - 4.0 * progress))
                probs = torch.tensor(
                    [(1 - alpha) * (alpha ** i) for i in range(self.key_indices.shape[1])],
                    device=self.key_indices.device
                )
                probs = probs / probs.sum()

                idx = torch.multinomial(probs, env_ids.shape[0], replacement=True)
            else:
                idx = torch.randint(
                    low=0,
                    high=self.key_indices.shape[1],
                    size=(env_ids.shape[0],),
                    device=self.key_indices.device
                )
        else:
            idx = torch.zeros(
                (env_ids.shape[0],),
            ).to(self.key_indices)
        self.frame_idx[env_ids] = self.key_indices[self.env_to_file_map[env_ids]][torch.arange(len(env_ids), device=self.device), idx].squeeze().to(self.frame_idx)
        #     self.frame_idx[env_ids] = self.frame_idx[env_ids] + torch.randint(
        #         low=-2,
        #         high=2,
        #         size=(env_ids.shape[0],),
        #         device=self.frame_idx.device
        #     )
        #     self.frame_idx[env_ids] = torch.clamp(self.frame_idx[env_ids], min=0, max=self.num_frames - 1)
        self._update_current()
        return self.frame_idx[env_ids], probs[0] if probs is not None else None

    # --------------------------------------------------
    # Step reference motion
    # --------------------------------------------------
    def step(self):
        self.frame_idx += self.frame_step
        self.frame_idx.clamp_(max=self.num_frames - 1)
        self._update_current()

    def get_before_first_keyframe_mask(self, env_ids: Optional[Sequence[int]] = None):
        first_keyframes = self.first_keyframe_idx[self.env_to_file_map]
        mask = self.frame_idx < first_keyframes.to(self.frame_idx)
        if env_ids is None:
            return mask
        return mask[env_ids]

    def _lerp(self, a, b, w):
        while w.dim() < a.dim():
            w = w.unsqueeze(-1)
        return a + w * (b - a)

    def _gather_env_traj(self, traj: torch.Tensor, frame_idx: torch.Tensor) -> torch.Tensor:
        return traj[self.env_to_file_map, frame_idx]

    def _sample_env_traj(self, traj: torch.Tensor, frame_idx: torch.Tensor) -> torch.Tensor:
        frame_idx = frame_idx.clamp(min=0.0, max=float(self.num_frames - 1))
        floor_idx = torch.floor(frame_idx).long()
        ceil_idx = torch.clamp(floor_idx + 1, max=self.num_frames - 1)
        weight = (frame_idx - floor_idx.to(frame_idx.dtype)).clamp(0.0, 1.0)
        lower = self._gather_env_traj(traj, floor_idx)
        upper = self._gather_env_traj(traj, ceil_idx)
        return self._lerp(lower, upper, weight)

    def _sample_env_quat_traj(self, traj: torch.Tensor, frame_idx: torch.Tensor) -> torch.Tensor:
        frame_idx = frame_idx.clamp(min=0.0, max=float(self.num_frames - 1))
        floor_idx = torch.floor(frame_idx).long()
        ceil_idx = torch.clamp(floor_idx + 1, max=self.num_frames - 1)
        weight = (frame_idx - floor_idx.to(frame_idx.dtype)).clamp(0.0, 1.0)
        lower = self._gather_env_traj(traj, floor_idx)
        upper = self._gather_env_traj(traj, ceil_idx)
        sign = torch.where((lower * upper).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
        blended = self._lerp(lower, upper * sign, weight)
        return blended / blended.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _update_current(self):
        idx = self.frame_idx
        floor_idx = torch.floor(idx).int().clamp(min=0, max=self.num_frames - 1)
        self.ref_robot_joint_pos = self._sample_env_traj(self.robot_joint_pos_traj, idx)
        self.ref_robot_joint_vel = self._sample_env_traj(self.robot_joint_vel_traj, idx)
        self.ref_door_joint_pos = self._sample_env_traj(self.door_traj, idx)
        self.ref_robot_body_pos = self._sample_env_traj(self.robot_body_pos_traj, idx)
        self.ref_robot_body_quat = self._sample_env_quat_traj(self.robot_body_quat_traj, idx)
        self.ref_hinge_contact_mask = self._gather_env_traj(self.hinge_contact_mask, floor_idx)
        self.ref_robot_body_pos_vel = self._sample_env_traj(self.robot_body_pos_vel, idx)
        self.ref_door_body_pos = self._sample_env_traj(self.door_body_pos_traj, idx)
        if self.twist_indices is not None:
            self.ref_robot_joint_pos_twist = self._sample_env_traj(self.robot_joint_pos_twist, idx)
            self.ref_robot_joint_vel_twist = self._sample_env_traj(self.robot_joint_vel_twist, idx)
            self.ref_door_joint_pos_twist = self._sample_env_traj(self.door_joint_pos_twist, idx)
            self.ref_robot_body_pos_twist = self._sample_env_traj(self.robot_body_pos_twist, idx)
            self.ref_robot_body_quat_twist = self._sample_env_quat_traj(self.robot_body_quat_twist, idx)
            self.ref_door_body_pos_twist = self._sample_env_traj(self.door_body_pos_twist, idx)
                
    # --------------------------------------------------
    # Getters (explicit, readable)
    # --------------------------------------------------
    def get_robot_joint_pos(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_joint_pos
        else:
            return self.ref_robot_joint_pos[env_ids]

    def get_door_joint_pos(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_door_joint_pos
        else:
            return self.ref_door_joint_pos[env_ids]

    def get_robot_body_pos(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_body_pos
        else:
            return self.ref_robot_body_pos[env_ids]

    def get_robot_body_quat(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_body_quat
        else:
            return self.ref_robot_body_quat[env_ids]

    def get_robot_joint_vel(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_joint_vel
        else:
            return self.ref_robot_joint_vel[env_ids]

    def get_robot_body_lin_vel(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_body_pos_vel[:, :, :3]
        else:
            return self.ref_robot_body_pos_vel[env_ids, :, :3]

    def get_robot_body_ang_vel(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_body_pos_vel[:, :, 3:]
        else:
            return self.ref_robot_body_pos_vel[env_ids, :, 3:]
        
    def get_door_body_pos(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_door_body_pos
        else:
            return self.ref_door_body_pos[env_ids]


    def get_robot_joint_pos_twist(self, env_ids: Optional[Sequence[int]] = None):
        assert self.ref_robot_joint_pos_twist.shape[-1] == 32 and self.ref_robot_joint_pos_twist.ndim == 3
        if env_ids is None:
            return self.ref_robot_joint_pos_twist
        else:
            return self.ref_robot_joint_pos_twist[env_ids]

    def get_door_joint_pos_twist(self, env_ids: Optional[Sequence[int]] = None):
        assert self.ref_door_joint_pos_twist.shape[-1] == 2 and self.ref_door_joint_pos_twist.ndim == 3
        if env_ids is None:
            return self.ref_door_joint_pos_twist
        else:
            return self.ref_door_joint_pos_twist[env_ids]

    def get_robot_body_pos_twist(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_body_pos_twist
        else:
            return self.ref_robot_body_pos_twist[env_ids]

    def get_robot_body_quat_twist(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_body_quat_twist
        else:
            return self.ref_robot_body_quat_twist[env_ids]

    def get_robot_joint_vel_twist(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_joint_vel_twist
        else:
            return self.ref_robot_joint_vel_twist[env_ids]

    def get_hinge_contact_mask(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_hinge_contact_mask
        else:
            return self.ref_hinge_contact_mask[env_ids]

    def get_door_body_pos_twist(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_door_body_pos_twist
        else:
            return self.ref_door_body_pos_twist[env_ids]


if __name__ == "__main__":
    from DoorOpening.assets.door.door_cfg import asset_base_folder, motion_traj_paths

    print("asset_base_folder: ", asset_base_folder)
    num_envs = 200
    device = torch.device("cpu")
    env_to_file_map = [i % len(motion_traj_paths) for i in range(num_envs)]
    twist_indices = [-50, -20, 0, 20, 50]
    ref_motion_lib = ReferenceMotionManager(num_envs=num_envs, device=device, reset_from_start = False, env_to_file_map=env_to_file_map, twist_indices=twist_indices)
    ref_motion_lib.reset(torch.arange(num_envs, device=device))
    ref_motion_lib.step()
    # print(ref_motion_lib.get_robot_joint_pos())
    # print(ref_motion_lib.get_door_joint_pos())
    # print(ref_motion_lib.get_robot_body_pos())
    # print(ref_motion_lib.get_robot_body_quat())
    # print(ref_motion_lib.get_robot_joint_vel())
    print(ref_motion_lib.get_robot_joint_pos_twist().shape)
    print(ref_motion_lib.get_door_joint_pos_twist().shape)
    print(ref_motion_lib.get_robot_body_pos_twist().shape)
    print(ref_motion_lib.get_robot_body_quat_twist().shape)
    print(ref_motion_lib.get_robot_joint_vel_twist().shape)
