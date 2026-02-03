import torch
import pickle as pkl
from typing import Optional, Sequence

class ReferenceMotionManager:
    def __init__(
        self,
        motion_file: Optional[str] = None,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
        velocity=1.0,
        reset_from_start=False,
        env_to_file_map: Optional[list] = None,
    ):
        self.device = device
        self.num_envs = num_envs
        self.velocity = velocity
        self.reset_from_start = reset_from_start

        if motion_file is not None:
            self._load_motion_pkl_from_one_file(motion_file)
            self.one_file_loaded = True
        else:
            self._load_motion_pkl_from_list()
            self.env_to_file_map = torch.tensor(env_to_file_map, device=self.device)
            self.one_file_loaded = False
        self._init_env_buffers()

    # --------------------------------------------------
    # Load motion data (moved from Env)
    # --------------------------------------------------
    def _load_motion_pkl_from_list(self):
        from DoorOpening.assets.door.door_cfg import motion_traj_paths

        robot_joint_pos_trajs = []
        door_trajs = []
        robot_body_pos_trajs = []
        robot_body_quat_trajs = []
        door_pos_trajs = []
        key_indices_list = []
        robot_base_vel_trajs = []
        robot_palm_vel_trajs = []

        for motion_file in motion_traj_paths:
            (robot_joint_pos_traj, door_traj, robot_body_pos_traj, robot_body_quat_traj, door_pos_traj, key_indices, self.num_frames, robot_base_vel_traj, robot_palm_vel_traj) = self._load_motion_pkl(motion_file)
            robot_joint_pos_trajs.append(robot_joint_pos_traj)
            door_trajs.append(door_traj)
            robot_body_pos_trajs.append(robot_body_pos_traj)
            robot_body_quat_trajs.append(robot_body_quat_traj)
            if len(door_pos_trajs) != 0 and door_pos_traj.shape[1] != door_pos_trajs[0].shape[1]:
                door_pos_traj = door_pos_traj[:, :door_pos_trajs[0].shape[1]]
            door_pos_trajs.append(door_pos_traj)
            key_indices_list.append(key_indices)
            robot_base_vel_trajs.append(robot_base_vel_traj)
            robot_palm_vel_trajs.append(robot_palm_vel_traj)

        # stack motions: [M, T, ...]
        self.robot_joint_pos_traj = torch.stack(robot_joint_pos_trajs, dim=0)
        self.robot_body_pos_traj = torch.stack(robot_body_pos_trajs, dim=0)
        self.robot_body_quat_traj = torch.stack(robot_body_quat_trajs, dim=0)
        self.door_traj = torch.stack(door_trajs, dim=0)
        self.door_pos_traj = torch.stack(door_pos_trajs, dim=0)
        self.key_indices = torch.stack(key_indices_list, dim=0).to(self.device)
        self.robot_base_vel_traj = torch.stack(robot_base_vel_trajs, dim=0)
        self.robot_palm_vel_traj = torch.stack(robot_palm_vel_trajs, dim=0)
        self.num_motions = self.robot_joint_pos_traj.shape[0]


    def _load_motion_pkl_from_one_file(self, motion_file: str):
        (self.robot_joint_pos_traj,\
            self.door_traj, \
            self.robot_body_pos_traj, \
            self.robot_body_quat_traj, \
            self.door_pos_traj, \
            key_indices, \
            self.num_frames, \
            self.robot_base_vel_traj, \
            self.robot_palm_vel_traj)\
        = self._load_motion_pkl(motion_file)
        self.key_indices = torch.tensor(key_indices, device=self.device).unsqueeze(0)


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

        robot_joint_pos_traj = motions["robot_joint_pos_traj"]
        door_traj = motions["door_traj"]
        robot_body_pos_traj = motions["robot_body_pos_traj"]
        robot_body_quat_traj = motions["robot_body_quat_traj"]
        door_pos_traj = motions["door_pos_traj"]
        key_indices = motions["key_indices"]
        robot_base_vel_traj = motions["robot_base_vel_traj"]
        robot_palm_vel_traj = motions["robot_palm_vel_traj"]

        if isinstance(robot_joint_pos_traj, list):
            robot_joint_pos_traj = torch.stack(robot_joint_pos_traj, dim = 0)
        if isinstance(door_traj, list):
            door_traj = torch.stack(door_traj, dim = 0)
        if isinstance(robot_body_pos_traj, list):
            robot_body_pos_traj = torch.stack(robot_body_pos_traj, dim = 0)
        if isinstance(robot_body_quat_traj, list):
            robot_body_quat_traj = torch.stack(robot_body_quat_traj, dim = 0)
        if isinstance(door_pos_traj, list):
            door_pos_traj = torch.stack(door_pos_traj, dim = 0)
        if isinstance(robot_base_vel_traj, list):
            robot_base_vel_traj = torch.stack(robot_base_vel_traj, dim = 0)
        if isinstance(robot_palm_vel_traj, list):
            robot_palm_vel_traj = torch.stack(robot_palm_vel_traj, dim = 0)
        if isinstance(key_indices, list):
            key_indices = torch.tensor(key_indices)
        # key_indices = key_indices[[0, 1, -2]]
        startable_indices = torch.zeros(2).to(key_indices)
        startable_indices[0] = key_indices[0]
        startable_indices[1] = key_indices[1] * 0.7 + key_indices[0] * 0.3
        # startable_indices[2] = key_indices[-2] * 0.9 + key_indices[-1] * 0.1
        key_indices = startable_indices

        robot_joint_pos_traj = robot_joint_pos_traj.to(self.device).squeeze()
        door_traj = door_traj.to(self.device).squeeze()
        robot_body_pos_traj = robot_body_pos_traj.to(self.device).squeeze()
        robot_body_quat_traj = robot_body_quat_traj.to(self.device).squeeze()
        door_pos_traj = door_pos_traj.to(self.device).squeeze()
        robot_base_vel_traj = robot_base_vel_traj.to(self.device).squeeze()
        robot_palm_vel_traj = robot_palm_vel_traj.to(self.device).squeeze()

        num_frames = robot_joint_pos_traj.shape[0]

        return robot_joint_pos_traj, door_traj, robot_body_pos_traj, robot_body_quat_traj, door_pos_traj, key_indices, num_frames, robot_base_vel_traj, robot_palm_vel_traj

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
        if not self.one_file_loaded:
            self.frame_idx[env_ids] = self.key_indices[self.env_to_file_map[env_ids]][torch.arange(len(env_ids), device=self.device), idx].squeeze().to(self.frame_idx)
        else:
            self.frame_idx[env_ids] = self.key_indices[idx].squeeze().to(self.frame_idx)
        # if not self.reset_from_start:
        #     self.frame_idx[env_ids] = self.frame_idx[env_ids] + torch.randint(
        #         low=-2,
        #         high=2,
        #         size=(env_ids.shape[0],),
        #         device=self.frame_idx.device
        #     )
        #     self.frame_idx[env_ids] = torch.clamp(self.frame_idx[env_ids], min=0, max=self.num_frames - 1)
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
        if self.one_file_loaded:
            self.ref_robot_joint_pos = self._lerp(self.robot_joint_pos_traj[floor_idx], self.robot_joint_pos_traj[ceil_idx], interp_ratio)
            self.ref_door_joint_pos = self._lerp(self.door_traj[floor_idx], self.door_traj[ceil_idx], interp_ratio)
            self.ref_robot_body_pos = self._lerp(self.robot_body_pos_traj[floor_idx], self.robot_body_pos_traj[ceil_idx], interp_ratio)
            self.ref_robot_body_quat = self._lerp(self.robot_body_quat_traj[floor_idx], self.robot_body_quat_traj[ceil_idx], interp_ratio)
            self.ref_door_pos = self._lerp(self.door_pos_traj[floor_idx], self.door_pos_traj[ceil_idx], interp_ratio)
            self.ref_robot_base_vel = self._lerp(self.robot_base_vel_traj[floor_idx], self.robot_base_vel_traj[ceil_idx], interp_ratio)
            self.ref_robot_palm_vel = self._lerp(self.robot_palm_vel_traj[floor_idx], self.robot_palm_vel_traj[ceil_idx], interp_ratio)
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
            indices = torch.arange(len(env_ids), device=self.device)
            self.ref_robot_joint_pos = self._lerp(self.robot_joint_pos_traj[self.env_to_file_map[env_ids]][indices, floor_idx], self.robot_joint_pos_traj[self.env_to_file_map[env_ids]][indices, ceil_idx], interp_ratio)
            self.ref_door_joint_pos = self._lerp(self.door_traj[self.env_to_file_map[env_ids]][indices, floor_idx], self.door_traj[self.env_to_file_map[env_ids]][indices, ceil_idx], interp_ratio)
            self.ref_robot_body_pos = self._lerp(self.robot_body_pos_traj[self.env_to_file_map[env_ids]][indices, floor_idx], self.robot_body_pos_traj[self.env_to_file_map[env_ids]][indices, ceil_idx], interp_ratio)
            self.ref_robot_body_quat = self._lerp(self.robot_body_quat_traj[self.env_to_file_map[env_ids]][indices, floor_idx], self.robot_body_quat_traj[self.env_to_file_map[env_ids]][indices, ceil_idx], interp_ratio)
            self.ref_door_pos = self._lerp(self.door_pos_traj[self.env_to_file_map[env_ids]][indices, floor_idx], self.door_pos_traj[self.env_to_file_map[env_ids]][indices, ceil_idx], interp_ratio)
            self.ref_robot_base_vel = self._lerp(self.robot_base_vel_traj[self.env_to_file_map[env_ids]][indices, floor_idx], self.robot_base_vel_traj[self.env_to_file_map[env_ids]][indices, ceil_idx], interp_ratio)
            self.ref_robot_palm_vel = self._lerp(self.robot_palm_vel_traj[self.env_to_file_map[env_ids]][indices, floor_idx], self.robot_palm_vel_traj[self.env_to_file_map[env_ids]][indices, ceil_idx], interp_ratio)
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

    def get_door_pos(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_door_pos
        else:
            return self.ref_door_pos[env_ids]

    def get_robot_base_vel(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_base_vel
        else:
            return self.ref_robot_base_vel[env_ids]

    def get_robot_palm_vel(self, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            return self.ref_robot_palm_vel
        else:
            return self.ref_robot_palm_vel[env_ids]