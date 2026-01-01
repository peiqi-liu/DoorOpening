import torch
import pickle as pkl
from typing import Optional, Sequence

class ReferenceMotionManager:
    def __init__(
        self,
        motion_file: str,
        num_envs: int,
        device: torch.device,
        reset_range=(0, 340),
        velocity=0.6,
    ):
        self.device = device
        self.num_envs = num_envs
        self.reset_lo, self.reset_hi = reset_range
        self.velocity = velocity

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

        if isinstance(self.robot_joint_pos_traj, list):
            self.robot_joint_pos_traj = torch.stack(self.robot_joint_pos_traj, dim = 0)
        if isinstance(self.door_traj, list):
            self.door_traj = torch.stack(self.door_traj, dim = 0)
        if isinstance(self.robot_body_pos_traj, list):
            self.robot_body_pos_traj = torch.stack(self.robot_body_pos_traj, dim = 0)
        if isinstance(self.robot_body_quat_traj, list):
            self.robot_body_quat_traj = torch.stack(self.robot_body_quat_traj, dim = 0)

        self.robot_joint_pos_traj = self.robot_joint_pos_traj.to(self.device).squeeze()
        self.door_traj = self.door_traj.to(self.device).squeeze()
        self.robot_body_pos_traj = self.robot_body_pos_traj.to(self.device).squeeze()
        self.robot_body_quat_traj = self.robot_body_quat_traj.to(self.device).squeeze()

        self.num_frames = self.robot_joint_pos_traj.shape[0]

        assert self.reset_hi < self.num_frames, \
            "reset_range exceeds motion length"

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
    def reset(self, env_ids: Sequence[int]):
        self.frame_idx[env_ids] = torch.randint(
            self.reset_lo,
            self.reset_hi + 1,
            (env_ids.shape[0],),
            device=self.device,
        ).float()
        self._update_current()

    # --------------------------------------------------
    # Step reference motion
    # --------------------------------------------------
    def step(self):
        # self.frame_idx += self.velocity
        self.frame_idx += 1
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
        self.ref_door_joint_pos = self._lerp(self.door_traj[floor_idx], self.door_traj[ceil_idx], interp_ratio)
        self.ref_robot_body_pos = self._lerp(self.robot_body_pos_traj[floor_idx], self.robot_body_pos_traj[ceil_idx], interp_ratio)
        self.ref_robot_body_quat = self._lerp(self.robot_body_quat_traj[floor_idx], self.robot_body_quat_traj[ceil_idx], interp_ratio)

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

if __name__ == "__main__":
    motion_file = "traj.pkl"
    num_envs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_range = (0, 30)
    motion_lib = ReferenceMotionManager(motion_file, num_envs, device, reset_range)
    motion_lib.reset(torch.tensor([1]))
    motion_lib.step()
    print(motion_lib.get_robot_joint_pos().shape)
    print(motion_lib.get_door_joint_pos().shape)
    print(motion_lib.get_robot_body_pos().shape)
    print(motion_lib.get_robot_body_quat([1, 3]).shape)