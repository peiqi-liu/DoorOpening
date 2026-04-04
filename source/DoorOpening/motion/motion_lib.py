import torch
import pickle as pkl
from typing import Optional, Sequence
from DoorOpening.utils.pose_utils import normalize_to_center_frame
import os

class ReferenceMotionManager:
    def __init__(
        self,
        motion_file: Optional[str] = None,
        num_envs: int = 1,
        device: torch.device = torch.device("cpu"),
        velocity=0.6,
        reset_from_start=False,
        env_to_file_map: Optional[list] = None,
        twist_indices: Optional[list] = None,
    ):
        self.device = device
        self.num_envs = num_envs
        self.velocity = velocity
        self.reset_from_start = reset_from_start
        # Named as twist indices because we borrow the idea from Twist paper
        self.twist_indices = twist_indices

        if motion_file is not None:
            self._load_motion_pkl_from_one_file(motion_file)
            self.one_file_loaded = True
        else:
            self._load_motion_pkl_from_list()
            self.env_to_file_map = torch.tensor(env_to_file_map, device=self.device)
            self.one_file_loaded = False
        self._init_env_buffers()
        if self.twist_indices is not None:
            self._precompute_twist()

    # --------------------------------------------------
    # Load motion data (moved from Env)
    # --------------------------------------------------
    def _load_motion_pkl_from_list(self):
        # from DoorOpening.assets.door.door_cfg import motion_traj_paths
        import glob
        root_path = os.path.dirname(os.path.dirname(__file__))
        asset_base_folder = os.path.join(root_path, "assets/door/PartNetv5")
        motion_traj_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/traj.pkl"), recursive=True))

        robot_joint_pos_trajs = []
        door_trajs = []
        robot_body_pos_trajs = []
        robot_body_quat_trajs = []
        robot_joint_vel_trajs = []
        key_indices_list = []
        hinge_contact_masks_list = []
        robot_body_pos_vel_list = []
        door_body_pos_trajs = []
        for motion_file in motion_traj_paths:
            (robot_joint_pos_traj,\
                door_traj,\
                robot_body_pos_traj,\
                robot_body_quat_traj,\
                robot_joint_vel_traj,\
                key_indices,\
                self.num_frames,\
                hinge_contact_mask,\
                robot_body_pos_vel,\
                door_body_pos_traj) = self._load_motion_pkl(motion_file)
            robot_joint_pos_trajs.append(robot_joint_pos_traj)
            door_trajs.append(door_traj)
            robot_body_pos_trajs.append(robot_body_pos_traj)
            robot_body_quat_trajs.append(robot_body_quat_traj)
            robot_joint_vel_trajs.append(robot_joint_vel_traj) 
            if isinstance(key_indices, list):
                key_indices = torch.tensor(key_indices, device=self.device)
            key_indices_list.append(key_indices)
            hinge_contact_masks_list.append(hinge_contact_mask)
            robot_body_pos_vel_list.append(robot_body_pos_vel)
            door_body_pos_trajs.append(door_body_pos_traj)
        # stack motions: [M, T, ...]
        self.robot_joint_pos_traj = torch.stack(robot_joint_pos_trajs, dim=0)
        self.robot_joint_vel_traj = torch.stack(robot_joint_vel_trajs, dim=0)
        self.robot_body_pos_traj = torch.stack(robot_body_pos_trajs, dim=0)
        self.robot_body_quat_traj = torch.stack(robot_body_quat_trajs, dim=0)
        self.door_traj = torch.stack(door_trajs, dim=0)
        # self.key_indices = torch.stack(key_indices_list, dim=0).to(self.device)
        # self.key_indices = self.key_indices[..., :-1] # remove the last key index
        self.key_indices = torch.arange(0, self.num_frames, 1).repeat(len(key_indices_list), 1).to(self.device).int()
        self.hinge_contact_mask = torch.stack(hinge_contact_masks_list, dim=0).to(self.device)
        self.num_motions = self.robot_joint_pos_traj.shape[0]
        self.robot_body_pos_vel = torch.stack(robot_body_pos_vel_list, dim=0).to(self.device)
        self.door_body_pos_traj = torch.stack(door_body_pos_trajs, dim=0).to(self.device)

    def _load_motion_pkl_from_one_file(self, motion_file: str):
        (self.robot_joint_pos_traj,\
            self.door_traj, \
            self.robot_body_pos_traj, \
            self.robot_body_quat_traj, \
            self.robot_joint_vel_traj, \
            key_indices, \
            self.num_frames, \
            robot_body_pos_vel, \
            door_body_pos_traj)\
        = self._load_motion_pkl(motion_file)
        # self.key_indices = torch.tensor(key_indices, device=self.device).unsqueeze(0)
        self.key_indices = torch.arange(self.num_frames, device=self.device).unsqueeze(0)
        self.door_body_pos_traj = torch.tensor(door_body_pos_traj, device=self.device).unsqueeze(0)


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
        hinge_contact_mask = hinge_contact_mask.to(self.device).squeeze()
        robot_body_pos_vel = robot_body_pos_vel.to(self.device).squeeze()
        door_body_pos_traj = door_body_pos_traj.to(self.device).squeeze()
        num_frames = robot_joint_pos_traj.shape[0]

        if hinge_contact_mask is not None:
            return robot_joint_pos_traj, door_traj, robot_body_pos_traj, robot_body_quat_traj, robot_joint_vel_traj, key_indices, num_frames, hinge_contact_mask, robot_body_pos_vel, door_body_pos_traj
        else:
            return robot_joint_pos_traj, door_traj, robot_body_pos_traj, robot_body_quat_traj, robot_joint_vel_traj, key_indices, num_frames, robot_body_pos_vel, door_body_pos_traj

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

        # --------------------------------------------------
        # Helper for one-file case
        # --------------------------------------------------

        def precompute_single(traj):
            # traj: (T, dim)
            # output: (T, K, dim)
            return traj[twist_frames]  # advanced indexing

        # --------------------------------------------------
        # Helper for multi-file case
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Apply
        # --------------------------------------------------

        if self.one_file_loaded:

            self.robot_joint_pos_twist = precompute_single(self.robot_joint_pos_traj)
            self.robot_joint_vel_twist = precompute_single(self.robot_joint_vel_traj)
            self.door_joint_pos_twist  = precompute_single(self.door_traj)
            self.robot_body_pos_twist  = precompute_single(self.robot_body_pos_traj)
            self.robot_body_quat_twist = precompute_single(self.robot_body_quat_traj)
            # self.robot_body_pos_twist, self.robot_body_quat_twist = normalize_to_center_frame(self.robot_body_pos_traj, self.robot_body_quat_traj, self.robot_body_pos_twist, self.robot_body_quat_twist)
            self.door_body_pos_twist = precompute_single(self.door_body_pos_traj)

        else:

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
        if not self.one_file_loaded:
            # print("frame_idx", self.frame_idx.shape)
            # print("env_ids", env_ids)
            # print("self.env_to_file_map[env_ids]", self.env_to_file_map[env_ids].shape)
            self.frame_idx[env_ids] = self.key_indices[self.env_to_file_map[env_ids]][torch.arange(len(env_ids), device=self.device), idx].squeeze().to(self.frame_idx)
        else:
            self.frame_idx[env_ids] = self.key_indices[idx].squeeze().to(self.frame_idx)
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
        self.frame_idx += self.velocity
        self.frame_idx.clamp_(max=self.num_frames - 1)
        self._update_current()

    def _lerp(self, a, b, w):
        while w.dim() < a.dim():
            w = w.unsqueeze(-1)
        return a + w * (b - a)

    def _update_current(self):
        idx = self.frame_idx
        floor_idx = torch.floor(idx).int().clamp(min=0, max=self.num_frames - 1)
        if self.one_file_loaded:
            self.ref_robot_joint_pos = self.robot_joint_pos_traj[floor_idx]
            self.ref_robot_joint_vel = self.robot_joint_vel_traj[floor_idx]
            self.ref_door_joint_pos = self.door_traj[floor_idx]
            self.ref_robot_body_pos = self.robot_body_pos_traj[floor_idx]
            self.ref_robot_body_quat = self.robot_body_quat_traj[floor_idx]
            self.ref_robot_body_pos_vel = self.robot_body_pos_vel[floor_idx]
            self.ref_door_body_pos = self.door_body_pos_traj[floor_idx]
            if self.twist_indices is not None:
                self.ref_robot_joint_pos_twist = self.robot_joint_pos_twist[floor_idx]
                self.ref_robot_joint_vel_twist = self.robot_joint_vel_twist[floor_idx]
                self.ref_door_joint_pos_twist = self.door_joint_pos_twist[floor_idx]
                self.ref_robot_body_pos_twist = self.robot_body_pos_twist[floor_idx]
                self.ref_robot_body_quat_twist = self.robot_body_quat_twist[floor_idx]
                self.ref_door_body_pos_twist = self.door_body_pos_twist[floor_idx]
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
            indices = torch.arange(len(env_ids), device=self.device)
            self.ref_robot_joint_pos = self.robot_joint_pos_traj[self.env_to_file_map[env_ids]][indices, floor_idx]
            self.ref_robot_joint_vel = self.robot_joint_vel_traj[self.env_to_file_map[env_ids]][indices, floor_idx]
            self.ref_door_joint_pos = self.door_traj[self.env_to_file_map[env_ids]][indices, floor_idx]
            self.ref_robot_body_pos = self.robot_body_pos_traj[self.env_to_file_map[env_ids]][indices, floor_idx]
            self.ref_robot_body_quat = self.robot_body_quat_traj[self.env_to_file_map[env_ids]][indices, floor_idx]
            # print("self.hinge_contact_mask.shape: ", self.hinge_contact_mask.shape)
            self.ref_hinge_contact_mask = self.hinge_contact_mask[self.env_to_file_map[env_ids]][indices, floor_idx]
            # print("self.ref_hinge_contact_mask.shape: ", self.ref_hinge_contact_mask.shape)
            self.ref_robot_body_pos_vel = self.robot_body_pos_vel[self.env_to_file_map[env_ids]][indices, floor_idx]
            self.ref_door_body_pos = self.door_body_pos_traj[self.env_to_file_map[env_ids]][indices, floor_idx]
            if self.twist_indices is not None:
                self.ref_robot_joint_pos_twist = self.robot_joint_pos_twist[self.env_to_file_map[env_ids]][indices, floor_idx]
                self.ref_robot_joint_vel_twist = self.robot_joint_vel_twist[self.env_to_file_map[env_ids]][indices, floor_idx]
                self.ref_door_joint_pos_twist = self.door_joint_pos_twist[self.env_to_file_map[env_ids]][indices, floor_idx]
                self.ref_robot_body_pos_twist = self.robot_body_pos_twist[self.env_to_file_map[env_ids]][indices, floor_idx]
                self.ref_robot_body_quat_twist = self.robot_body_quat_twist[self.env_to_file_map[env_ids]][indices, floor_idx]
                self.ref_door_body_pos_twist = self.door_body_pos_twist[self.env_to_file_map[env_ids]][indices, floor_idx]
                
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
            return self.ref_robot_joint_vel_twist / self.velocity
        else:
            return self.ref_robot_joint_vel_twist[env_ids] / self.velocity

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
    import glob
    root_path = os.path.dirname(os.path.dirname(__file__))
    asset_base_folder = os.path.join(root_path, "assets/door/PartNetv5")
    print("asset_base_folder: ", asset_base_folder)
    motion_traj_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/traj.pkl"), recursive=True))
    num_envs = 200
    device = torch.device("cpu")
    velocity = 1.0
    env_to_file_map = [i % len(motion_traj_paths) for i in range(num_envs)]
    twist_indices = [-50, -20, 0, 20, 50]
    ref_motion_lib = ReferenceMotionManager(num_envs=num_envs, device=device, velocity=velocity, reset_from_start = False, env_to_file_map=env_to_file_map, twist_indices=twist_indices)
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