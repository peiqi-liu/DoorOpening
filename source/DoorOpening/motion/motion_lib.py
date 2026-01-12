import torch
import pickle as pkl
from typing import Optional, Sequence

class ReferenceMotionManager:
    def __init__(
        self,
        motion_file: str,
        num_envs: int,
        device: torch.device,
        velocity=0.6,
        reset_from_start=False,
    ):
        self.device = device
        self.num_envs = num_envs
        self.velocity = velocity
        self.reset_from_start = reset_from_start

        self._load_motion_pkl(motion_file)
        self._init_env_buffers()

    # --------------------------------------------------
    # Load motion data (moved from Env)
    # --------------------------------------------------
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

        self.robot_joint_pos_traj = motions["robot_joint_pos_traj"]
        self.door_traj = motions["door_traj"]
        self.robot_body_pos_traj = motions["robot_body_pos_traj"]
        self.robot_body_quat_traj = motions["robot_body_quat_traj"]
        self.robot_joint_vel_traj = motions["robot_joint_vel_traj"]
        self.door_pos_traj = motions["door_pos_traj"]
        self.key_indices = motions["key_indices"]

        if isinstance(self.robot_joint_pos_traj, list):
            self.robot_joint_pos_traj = torch.stack(self.robot_joint_pos_traj, dim = 0)
        if isinstance(self.door_traj, list):
            self.door_traj = torch.stack(self.door_traj, dim = 0)
        if isinstance(self.robot_body_pos_traj, list):
            self.robot_body_pos_traj = torch.stack(self.robot_body_pos_traj, dim = 0)
        if isinstance(self.robot_body_quat_traj, list):
            self.robot_body_quat_traj = torch.stack(self.robot_body_quat_traj, dim = 0)
        if isinstance(self.robot_joint_vel_traj, list):
            self.robot_joint_vel_traj = torch.stack(self.robot_joint_vel_traj, dim = 0)
        if isinstance(self.door_pos_traj, list):
            self.door_pos_traj = torch.stack(self.door_pos_traj, dim = 0)

        self.robot_joint_pos_traj = self.robot_joint_pos_traj.to(self.device).squeeze()
        self.door_traj = self.door_traj.to(self.device).squeeze()
        self.robot_body_pos_traj = self.robot_body_pos_traj.to(self.device).squeeze()
        self.robot_body_quat_traj = self.robot_body_quat_traj.to(self.device).squeeze()
        self.robot_joint_vel_traj = self.robot_joint_vel_traj.to(self.device).squeeze()
        self.door_pos_traj = self.door_pos_traj.to(self.device).squeeze()

        self.num_frames = self.robot_joint_pos_traj.shape[0]

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

    # --------------------------------------------------
    # Reset logic
    # --------------------------------------------------
    def reset(self, env_ids: Sequence[int], step_count: Optional[int] = None, reset_progress_total: Optional[int] = None):
        if not self.reset_from_start:
            if step_count is not None and reset_progress_total is not None:
                progress = min(step_count / reset_progress_total, 1.0)
                alpha = 0.9 - 0.7 * progress  # from 0.9 → 0.2

                probs = torch.tensor(
                    [(1 - alpha) * (alpha ** i) for i in range(len(self.key_indices))],
                    device=self.key_indices.device
                )
                probs = probs / probs.sum()

                idx = torch.multinomial(probs, env_ids.shape[0], replacement=True)
            else:
                idx = torch.randint(
                    low=0,
                    high=len(self.key_indices),
                    size=(env_ids.shape[0],),
                    device=self.key_indices.device
                )
        else:
            idx = torch.zeros(
                (env_ids.shape[0],),
            ).to(self.key_indices)
        self.frame_idx[env_ids] = self.key_indices[idx].squeeze().to(self.frame_idx)
        if not self.reset_from_start:
            self.frame_idx[env_ids] = self.frame_idx[env_ids] + torch.randint(
                low=-2,
                high=2,
                size=(env_ids.shape[0],),
                device=self.frame_idx.device
            )
            self.frame_idx[env_ids] = torch.clamp(self.frame_idx[env_ids], min=0, max=self.num_frames - 1)
        self._update_current()
        return self.frame_idx[env_ids]

    # --------------------------------------------------
    # Step reference motion
    # --------------------------------------------------
    def step(self):
        self.frame_idx += self.velocity
        self.frame_idx.clamp_(max=self.num_frames - 1)
        self._update_current()

    def _lerp(self, a, b, w):
        while w.dim() < a.dim():
            w = w.unsqueeze(-1)
        return a + w * (b - a)

    def _update_current(self):
        idx = self.frame_idx
        floor_idx = torch.floor(idx).int()
        ceil_idx = torch.ceil(idx).int()
        interp_ratio = (idx - floor_idx).unsqueeze(-1)
        self.ref_robot_joint_pos = self._lerp(self.robot_joint_pos_traj[floor_idx], self.robot_joint_pos_traj[ceil_idx], interp_ratio)
        self.ref_robot_joint_vel = self._lerp(self.robot_joint_vel_traj[floor_idx], self.robot_joint_vel_traj[ceil_idx], interp_ratio)
        self.ref_door_joint_pos = self._lerp(self.door_traj[floor_idx], self.door_traj[ceil_idx], interp_ratio)
        self.ref_robot_body_pos = self._lerp(self.robot_body_pos_traj[floor_idx], self.robot_body_pos_traj[ceil_idx], interp_ratio)
        self.ref_robot_body_quat = self._lerp(self.robot_body_quat_traj[floor_idx], self.robot_body_quat_traj[ceil_idx], interp_ratio)
        self.ref_door_pos = self._lerp(self.door_pos_traj[floor_idx], self.door_pos_traj[ceil_idx], interp_ratio)

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
            return self.ref_robot_joint_vel / self.velocity
        else:
            return self.ref_robot_joint_vel[env_ids] / self.velocity

    def get_door_pos(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_door_pos
        else:
            return self.ref_door_pos[env_ids]