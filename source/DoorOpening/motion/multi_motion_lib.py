import os
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
        self.has_wrong_motions = False
        self.wrong_motion_traj_paths = []
        self.wrong_env_to_file_map = None
        self._load_motion_pkl_from_list()
        self.env_to_file_map = torch.tensor(env_to_file_map, device=self.device, dtype=torch.long)
        if self.env_to_file_map.numel() != int(self.num_envs):
            raise ValueError(
                "env_to_file_map must contain exactly one motion index per env: "
                f"got {self.env_to_file_map.numel()} for {self.num_envs} envs."
            )
        if self.env_to_file_map.numel() > 0:
            min_motion_idx = int(self.env_to_file_map.min().detach().cpu().item())
            max_motion_idx = int(self.env_to_file_map.max().detach().cpu().item())
            if min_motion_idx < 0 or max_motion_idx >= int(self.num_motions):
                raise ValueError(
                    "env_to_file_map contains an out-of-range motion index: "
                    f"min={min_motion_idx}, max={max_motion_idx}, num_motions={self.num_motions}."
                )
        self.frame_step = self.step_dt / self.frame_dt
        self._init_env_buffers()
        if self.twist_indices is not None:
            self._precompute_twist()
            self._twist_offset_to_index = {int(offset): idx for idx, offset in enumerate(self.twist_indices)}
            self._twist_plus_one_index = self._twist_offset_to_index.get(1)
        else:
            self._twist_offset_to_index = {}
            self._twist_plus_one_index = None

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

    def _normalize_key_indices(self, key_indices, num_frames: int) -> torch.Tensor:
        if isinstance(key_indices, torch.Tensor):
            key_tensor = key_indices.flatten().to(device=self.device, dtype=torch.long)
        elif key_indices is None:
            key_tensor = torch.empty(0, device=self.device, dtype=torch.long)
        else:
            key_tensor = torch.tensor(list(key_indices), device=self.device, dtype=torch.long)

        if key_tensor.numel() == 0:
            key_tensor = torch.tensor([0, max(int(num_frames) - 1, 0)], device=self.device, dtype=torch.long)
        key_tensor = key_tensor.clamp(min=0, max=max(int(num_frames) - 1, 0))
        key_tensor = torch.cummax(key_tensor, dim=0).values
        return key_tensor

    def _stack_phase_key_indices(self, key_indices_list, num_frames: int):
        normalized = [self._normalize_key_indices(key_indices, num_frames) for key_indices in key_indices_list]
        max_len = max((int(key_indices.numel()) for key_indices in normalized), default=2)
        phase_key_indices = torch.zeros(
            (len(normalized), max_len),
            device=self.device,
            dtype=torch.float32,
        )
        phase_key_counts = torch.zeros(len(normalized), device=self.device, dtype=torch.long)
        for motion_idx, key_indices in enumerate(normalized):
            count = int(key_indices.numel())
            phase_key_counts[motion_idx] = count
            phase_key_indices[motion_idx, :count] = key_indices.to(dtype=torch.float32)
            if count < max_len:
                phase_key_indices[motion_idx, count:] = key_indices[-1].to(dtype=torch.float32)
        return phase_key_indices, phase_key_counts

    # --------------------------------------------------
    # Load motion data (moved from Env)
    # --------------------------------------------------
    def _load_motion_bundle(self, motion_paths, label: str):
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
        num_frames = None
        for motion_file in motion_paths:
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

            motion_num_frames = int(robot_joint_pos_traj.shape[0])
            if num_frames is None:
                num_frames = motion_num_frames
            elif motion_num_frames != num_frames:
                raise ValueError(
                    f"Inconsistent {label} motion length in '{motion_file}': "
                    f"{motion_num_frames} vs {num_frames}."
                )

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
        if num_frames is None:
            raise FileNotFoundError(f"No {label} motion files were provided.")

        phase_key_indices, phase_key_counts = self._stack_phase_key_indices(key_indices_list, num_frames)
        return {
            "robot_joint_pos_traj": torch.stack(robot_joint_pos_trajs, dim=0),
            "robot_joint_vel_traj": torch.stack(robot_joint_vel_trajs, dim=0),
            "robot_body_pos_traj": torch.stack(robot_body_pos_trajs, dim=0),
            "robot_body_quat_traj": torch.stack(robot_body_quat_trajs, dim=0),
            "door_traj": torch.stack(door_trajs, dim=0),
            "reset_key_indices": torch.arange(0, num_frames, 1, device=self.device)
            .repeat(len(key_indices_list), 1)
            .int(),
            "phase_key_indices": phase_key_indices,
            "phase_key_counts": phase_key_counts,
            "first_keyframe_idx": torch.tensor(first_keyframes, device=self.device, dtype=torch.float32),
            "hinge_contact_mask": torch.stack(hinge_contact_masks_list, dim=0).to(self.device),
            "robot_body_pos_vel": torch.stack(robot_body_pos_vel_list, dim=0).to(self.device),
            "door_body_pos_traj": torch.stack(door_body_pos_trajs, dim=0).to(self.device),
            "num_frames": int(num_frames),
        }

    def _load_motion_pkl_from_list(self):
        from DoorOpening.assets.door.multi_door_cfg import asset_base_folder, motion_traj_paths

        if len(motion_traj_paths) == 0:
            raise FileNotFoundError(f"No traj.pkl files found under {asset_base_folder}")

        motion_bundle = self._load_motion_bundle(motion_traj_paths, label="correct")

        # stack motions: [M, T, ...]
        self.motion_traj_paths = list(motion_traj_paths)
        self.robot_joint_pos_traj = motion_bundle["robot_joint_pos_traj"]
        self.robot_joint_vel_traj = motion_bundle["robot_joint_vel_traj"]
        self.robot_body_pos_traj = motion_bundle["robot_body_pos_traj"]
        self.robot_body_quat_traj = motion_bundle["robot_body_quat_traj"]
        self.door_traj = motion_bundle["door_traj"]
        self.key_indices = motion_bundle["reset_key_indices"]
        self.phase_key_indices = motion_bundle["phase_key_indices"]
        self.phase_key_counts = motion_bundle["phase_key_counts"]
        self.first_keyframe_idx = motion_bundle["first_keyframe_idx"]
        self.hinge_contact_mask = motion_bundle["hinge_contact_mask"]
        self.num_motions = self.robot_joint_pos_traj.shape[0]
        self.num_frames = motion_bundle["num_frames"]
        self.robot_body_pos_vel = motion_bundle["robot_body_pos_vel"]
        self.door_body_pos_traj = motion_bundle["door_body_pos_traj"]

    def load_wrong_motions(
        self,
        wrong_motion_traj_paths: Optional[Sequence[str]] = None,
        wrong_traj_filename: str = "traj_wrong.pkl",
        require_all: bool = True,
    ) -> bool:
        if self.has_wrong_motions:
            return True
        if wrong_motion_traj_paths is None:
            from DoorOpening.assets.door.multi_door_cfg import motion_wrong_traj_paths

            wrong_motion_traj_paths = motion_wrong_traj_paths

        wrong_motion_traj_paths = list(wrong_motion_traj_paths)
        if len(wrong_motion_traj_paths) != int(self.num_motions):
            raise ValueError(
                "Wrong motion path count must match correct motion count: "
                f"{len(wrong_motion_traj_paths)} != {self.num_motions}."
            )

        missing_paths = [path for path in wrong_motion_traj_paths if not os.path.exists(path)]
        if missing_paths:
            preview = ", ".join(missing_paths[:5])
            suffix = "" if len(missing_paths) <= 5 else f", ... ({len(missing_paths)} missing total)"
            message = (
                "wrong_push_pull_rollout is enabled, but wrong reference trajectories are missing. "
                f"Expected '{wrong_traj_filename}' next to every active asset. Missing: {preview}{suffix}"
            )
            if require_all:
                raise FileNotFoundError(message)
            return False

        wrong_bundle = self._load_motion_bundle(wrong_motion_traj_paths, label="wrong")
        if int(wrong_bundle["num_frames"]) != int(self.num_frames):
            raise ValueError(
                "Wrong reference trajectories must have the same frame count as correct trajectories: "
                f"{wrong_bundle['num_frames']} != {self.num_frames}."
            )

        self.wrong_motion_traj_paths = wrong_motion_traj_paths
        self.wrong_robot_joint_pos_traj = wrong_bundle["robot_joint_pos_traj"]
        self.wrong_robot_joint_vel_traj = wrong_bundle["robot_joint_vel_traj"]
        self.wrong_robot_body_pos_traj = wrong_bundle["robot_body_pos_traj"]
        self.wrong_robot_body_quat_traj = wrong_bundle["robot_body_quat_traj"]
        self.wrong_door_traj = wrong_bundle["door_traj"]
        self.wrong_phase_key_indices = wrong_bundle["phase_key_indices"]
        self.wrong_phase_key_counts = wrong_bundle["phase_key_counts"]
        self.wrong_first_keyframe_idx = wrong_bundle["first_keyframe_idx"]
        self.wrong_hinge_contact_mask = wrong_bundle["hinge_contact_mask"]
        self.wrong_robot_body_pos_vel = wrong_bundle["robot_body_pos_vel"]
        self.wrong_door_body_pos_traj = wrong_bundle["door_body_pos_traj"]
        if self.twist_indices is not None:
            self._precompute_wrong_twist()
        self.has_wrong_motions = True
        return True

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

    def _precompute_twist_bundle(
        self,
        robot_joint_pos_traj: torch.Tensor,
        robot_joint_vel_traj: torch.Tensor,
        door_traj: torch.Tensor,
        robot_body_pos_traj: torch.Tensor,
        robot_body_quat_traj: torch.Tensor,
        door_body_pos_traj: torch.Tensor,
    ):
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

        return {
            "robot_joint_pos_twist": precompute_multi(robot_joint_pos_traj),
            "robot_joint_vel_twist": precompute_multi(robot_joint_vel_traj),
            "door_joint_pos_twist": precompute_multi(door_traj),
            "robot_body_pos_twist": precompute_multi(robot_body_pos_traj),
            "robot_body_quat_twist": precompute_multi(robot_body_quat_traj),
            "door_body_pos_twist": precompute_multi(door_body_pos_traj),
        }

    def _precompute_twist(self):
        twist_bundle = self._precompute_twist_bundle(
            self.robot_joint_pos_traj,
            self.robot_joint_vel_traj,
            self.door_traj,
            self.robot_body_pos_traj,
            self.robot_body_quat_traj,
            self.door_body_pos_traj,
        )
        self.robot_joint_pos_twist = twist_bundle["robot_joint_pos_twist"]
        self.robot_joint_vel_twist = twist_bundle["robot_joint_vel_twist"]
        self.door_joint_pos_twist = twist_bundle["door_joint_pos_twist"]
        self.robot_body_pos_twist = twist_bundle["robot_body_pos_twist"]
        self.robot_body_quat_twist = twist_bundle["robot_body_quat_twist"]
        print("self.robot_body_pos_traj.shape: ", self.robot_body_pos_traj.shape)
        print("self.robot_body_quat_traj.shape: ", self.robot_body_quat_traj.shape)
        print("self.robot_body_pos_twist.shape: ", self.robot_body_pos_twist.shape)
        print("self.robot_body_quat_twist.shape: ", self.robot_body_quat_twist.shape)
        # self.robot_body_pos_twist, self.robot_body_quat_twist = normalize_to_center_frame(self.robot_body_pos_traj, self.robot_body_quat_traj, self.robot_body_pos_twist, self.robot_body_quat_twist)
        self.door_body_pos_twist = twist_bundle["door_body_pos_twist"]

    def _precompute_wrong_twist(self):
        twist_bundle = self._precompute_twist_bundle(
            self.wrong_robot_joint_pos_traj,
            self.wrong_robot_joint_vel_traj,
            self.wrong_door_traj,
            self.wrong_robot_body_pos_traj,
            self.wrong_robot_body_quat_traj,
            self.wrong_door_body_pos_traj,
        )
        self.wrong_robot_joint_pos_twist = twist_bundle["robot_joint_pos_twist"]
        self.wrong_robot_joint_vel_twist = twist_bundle["robot_joint_vel_twist"]
        self.wrong_door_joint_pos_twist = twist_bundle["door_joint_pos_twist"]
        self.wrong_robot_body_pos_twist = twist_bundle["robot_body_pos_twist"]
        self.wrong_robot_body_quat_twist = twist_bundle["robot_body_quat_twist"]
        self.wrong_door_body_pos_twist = twist_bundle["door_body_pos_twist"]
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

    def set_wrong_motion_map(self, wrong_env_to_file_map: Sequence[int] | torch.Tensor):
        wrong_env_to_file_map = torch.as_tensor(wrong_env_to_file_map, device=self.device, dtype=torch.long)
        if wrong_env_to_file_map.numel() != int(self.num_envs):
            raise ValueError(
                "wrong_env_to_file_map must contain exactly one wrong motion index per env: "
                f"got {wrong_env_to_file_map.numel()} for {self.num_envs} envs."
            )
        if wrong_env_to_file_map.numel() > 0:
            min_motion_idx = int(wrong_env_to_file_map.min().detach().cpu().item())
            max_motion_idx = int(wrong_env_to_file_map.max().detach().cpu().item())
            if min_motion_idx < 0 or max_motion_idx >= int(self.num_motions):
                raise ValueError(
                    "wrong_env_to_file_map contains an out-of-range motion index: "
                    f"min={min_motion_idx}, max={max_motion_idx}, num_motions={self.num_motions}."
                )
        self.wrong_env_to_file_map = wrong_env_to_file_map

    def get_wrong_motion_indices(self, env_ids: Optional[Sequence[int]] = None) -> torch.Tensor:
        if self.wrong_env_to_file_map is None:
            raise RuntimeError("Wrong motion map has not been configured.")
        if env_ids is None:
            return self.wrong_env_to_file_map
        return self.wrong_env_to_file_map[env_ids]

    def get_current_phase(self, env_ids: Optional[Sequence[int]] = None) -> torch.Tensor:
        if env_ids is None:
            frame_idx = self.frame_idx
            motion_indices = self.env_to_file_map
        else:
            frame_idx = self.frame_idx[env_ids]
            motion_indices = self.env_to_file_map[env_ids]

        key_indices = self.phase_key_indices[motion_indices].to(frame_idx)
        key_counts = self.phase_key_counts[motion_indices]
        phase = torch.zeros_like(frame_idx, dtype=torch.float32)
        max_key_count = int(self.phase_key_indices.shape[1])
        for key_idx in range(max_key_count - 1):
            valid = key_counts > key_idx + 1
            start = key_indices[:, key_idx]
            end = key_indices[:, key_idx + 1]
            denom = (end - start).clamp_min(1.0)
            in_segment = valid & (frame_idx >= start) & (frame_idx <= end)
            segment_phase = float(key_idx) + ((frame_idx - start) / denom).clamp(0.0, 1.0)
            phase = torch.where(in_segment, segment_phase, phase)
            phase = torch.where(valid & (frame_idx > end), torch.full_like(phase, float(key_idx + 1)), phase)
        last_phase = (key_counts.to(dtype=torch.float32) - 1.0).clamp_min(0.0)
        last_key_idx = (key_counts - 1).clamp_min(0)
        last_key_frame = key_indices.gather(1, last_key_idx.unsqueeze(-1)).squeeze(-1)
        phase = torch.where(frame_idx >= last_key_frame, last_phase, phase)
        return phase

    def get_before_first_keyframe_mask(self, env_ids: Optional[Sequence[int]] = None):
        first_keyframes = self.first_keyframe_idx[self.env_to_file_map]
        mask = self.frame_idx < first_keyframes.to(self.frame_idx)
        if env_ids is None:
            return mask
        return mask[env_ids]

    def get_next_robot_joint_pos(self, env_ids: Optional[Sequence[int]] = None):
        if self._twist_plus_one_index is not None and abs(float(self.frame_step) - 1.0) < 1e-6:
            next_joint_pos = self.ref_robot_joint_pos_twist[:, self._twist_plus_one_index, :]
            if env_ids is None:
                return next_joint_pos
            return next_joint_pos[env_ids]

        next_joint_pos = self._sample_env_traj(self.robot_joint_pos_traj, self.frame_idx + self.frame_step)
        if env_ids is None:
            return next_joint_pos
        return next_joint_pos[env_ids]

    def _lerp(self, a, b, w):
        while w.dim() < a.dim():
            w = w.unsqueeze(-1)
        return a + w * (b - a)

    def _gather_env_traj(self, traj: torch.Tensor, frame_idx: torch.Tensor) -> torch.Tensor:
        return traj[self.env_to_file_map, frame_idx]

    def _gather_motion_traj(
        self,
        traj: torch.Tensor,
        motion_indices: torch.Tensor,
        frame_idx: torch.Tensor,
    ) -> torch.Tensor:
        return traj[motion_indices, frame_idx]

    def _get_reference_motion_indices(
        self,
        reference_source: str = "correct",
        env_ids: Optional[Sequence[int]] = None,
        motion_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if motion_indices is None:
            if reference_source == "wrong":
                if self.wrong_env_to_file_map is None:
                    raise RuntimeError("Wrong reference requested before wrong motion map was configured.")
                motion_indices = self.wrong_env_to_file_map
            else:
                motion_indices = self.env_to_file_map
            if env_ids is not None:
                motion_indices = motion_indices[env_ids]
            return motion_indices.to(device=self.device, dtype=torch.long)

        motion_indices = torch.as_tensor(motion_indices, device=self.device, dtype=torch.long)
        if env_ids is None:
            if motion_indices.numel() != int(self.num_envs):
                raise ValueError(
                    "motion_indices must be one index per env when env_ids is None: "
                    f"got {motion_indices.numel()} for {self.num_envs} envs."
                )
            return motion_indices

        if motion_indices.numel() == int(self.num_envs):
            return motion_indices[env_ids]
        if motion_indices.numel() == len(env_ids):
            return motion_indices
        raise ValueError(
            "motion_indices must either be one index per env or one index per requested env: "
            f"got {motion_indices.numel()} indices for {len(env_ids)} requested envs."
        )

    def _select_reference_traj(self, attr_name: str, reference_source: str = "correct") -> torch.Tensor:
        reference_source = str(reference_source).lower()
        if reference_source == "correct":
            return getattr(self, attr_name)
        if reference_source != "wrong":
            raise ValueError(f"Unsupported reference_source '{reference_source}'.")
        if not self.has_wrong_motions:
            raise RuntimeError("Wrong reference requested before wrong trajectories were loaded.")
        return getattr(self, f"wrong_{attr_name}")

    def _sample_reference_traj(
        self,
        attr_name: str,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        traj = self._select_reference_traj(attr_name, reference_source=reference_source)
        frame_idx = self.frame_idx if env_ids is None else self.frame_idx[env_ids]
        motion_indices = self._get_reference_motion_indices(
            reference_source=reference_source,
            env_ids=env_ids,
            motion_indices=motion_indices,
        )
        frame_idx = frame_idx.clamp(min=0.0, max=float(self.num_frames - 1))
        floor_idx = torch.floor(frame_idx).long()
        ceil_idx = torch.clamp(floor_idx + 1, max=self.num_frames - 1)
        weight = (frame_idx - floor_idx.to(frame_idx.dtype)).clamp(0.0, 1.0)
        lower = self._gather_motion_traj(traj, motion_indices, floor_idx)
        upper = self._gather_motion_traj(traj, motion_indices, ceil_idx)
        return self._lerp(lower, upper, weight)

    def _sample_reference_quat_traj(
        self,
        attr_name: str,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        traj = self._select_reference_traj(attr_name, reference_source=reference_source)
        frame_idx = self.frame_idx if env_ids is None else self.frame_idx[env_ids]
        motion_indices = self._get_reference_motion_indices(
            reference_source=reference_source,
            env_ids=env_ids,
            motion_indices=motion_indices,
        )
        frame_idx = frame_idx.clamp(min=0.0, max=float(self.num_frames - 1))
        floor_idx = torch.floor(frame_idx).long()
        ceil_idx = torch.clamp(floor_idx + 1, max=self.num_frames - 1)
        weight = (frame_idx - floor_idx.to(frame_idx.dtype)).clamp(0.0, 1.0)
        lower = self._gather_motion_traj(traj, motion_indices, floor_idx)
        upper = self._gather_motion_traj(traj, motion_indices, ceil_idx)
        sign = torch.where((lower * upper).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
        blended = self._lerp(lower, upper * sign, weight)
        return blended / blended.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _gather_reference_traj(
        self,
        attr_name: str,
        frame_idx: torch.Tensor,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        traj = self._select_reference_traj(attr_name, reference_source=reference_source)
        motion_indices = self._get_reference_motion_indices(
            reference_source=reference_source,
            env_ids=env_ids,
            motion_indices=motion_indices,
        )
        return self._gather_motion_traj(traj, motion_indices, frame_idx)

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
    def get_robot_joint_pos(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_joint_pos_traj",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_robot_joint_pos
        else:
            return self.ref_robot_joint_pos[env_ids]

    def get_door_joint_pos(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "door_traj",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_door_joint_pos
        else:
            return self.ref_door_joint_pos[env_ids]

    def get_robot_body_pos(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_body_pos_traj",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_robot_body_pos
        else:
            return self.ref_robot_body_pos[env_ids]

    def get_robot_body_quat(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_quat_traj(
                "robot_body_quat_traj",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_robot_body_quat
        else:
            return self.ref_robot_body_quat[env_ids]

    def get_robot_joint_vel(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_joint_vel_traj",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_robot_joint_vel
        else:
            return self.ref_robot_joint_vel[env_ids]

    def get_robot_body_lin_vel(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_body_pos_vel",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )[:, :, :3]
        if env_ids is None:
            return self.ref_robot_body_pos_vel[:, :, :3]
        else:
            return self.ref_robot_body_pos_vel[env_ids, :, :3]

    def get_robot_body_ang_vel(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_body_pos_vel",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )[:, :, 3:]
        if env_ids is None:
            return self.ref_robot_body_pos_vel[:, :, 3:]
        else:
            return self.ref_robot_body_pos_vel[env_ids, :, 3:]
        
    def get_door_body_pos(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "door_body_pos_traj",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_door_body_pos
        else:
            return self.ref_door_body_pos[env_ids]


    def get_robot_joint_pos_twist(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_joint_pos_twist",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        assert self.ref_robot_joint_pos_twist.shape[-1] == 32 and self.ref_robot_joint_pos_twist.ndim == 3
        if env_ids is None:
            return self.ref_robot_joint_pos_twist
        else:
            return self.ref_robot_joint_pos_twist[env_ids]

    def get_door_joint_pos_twist(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "door_joint_pos_twist",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        assert self.ref_door_joint_pos_twist.shape[-1] == 2 and self.ref_door_joint_pos_twist.ndim == 3
        if env_ids is None:
            return self.ref_door_joint_pos_twist
        else:
            return self.ref_door_joint_pos_twist[env_ids]

    def get_robot_body_pos_twist(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_body_pos_twist",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_robot_body_pos_twist
        else:
            return self.ref_robot_body_pos_twist[env_ids]

    def get_robot_body_quat_twist(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_quat_traj(
                "robot_body_quat_twist",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_robot_body_quat_twist
        else:
            return self.ref_robot_body_quat_twist[env_ids]

    def get_robot_joint_vel_twist(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "robot_joint_vel_twist",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_robot_joint_vel_twist
        else:
            return self.ref_robot_joint_vel_twist[env_ids]

    def get_hinge_contact_mask(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            frame_idx = self.frame_idx if env_ids is None else self.frame_idx[env_ids]
            floor_idx = torch.floor(frame_idx).long().clamp(min=0, max=self.num_frames - 1)
            return self._gather_reference_traj(
                "hinge_contact_mask",
                floor_idx,
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_hinge_contact_mask
        else:
            return self.ref_hinge_contact_mask[env_ids]

    def get_door_body_pos_twist(
        self,
        env_ids: Optional[Sequence[int]] = None,
        reference_source: str = "correct",
        motion_indices: Optional[torch.Tensor] = None,
    ):
        if reference_source != "correct" or motion_indices is not None:
            return self._sample_reference_traj(
                "door_body_pos_twist",
                env_ids=env_ids,
                reference_source=reference_source,
                motion_indices=motion_indices,
            )
        if env_ids is None:
            return self.ref_door_body_pos_twist
        else:
            return self.ref_door_body_pos_twist[env_ids]


if __name__ == "__main__":
    from DoorOpening.assets.door.multi_door_cfg import asset_base_folder, motion_traj_paths

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
