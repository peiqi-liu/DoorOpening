import inspect
import math
import os
import torch
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from DoorOpening.utils.quat_utils import quat_diff_angle, hinge_angle_diff
from DoorOpening.assets.door.multi_door_cfg import (
    DOOR_FAMILY_NAMES,
    asset_paths as door_asset_paths,
    asset_family_ids,
    board_offsets,
    configure_multi_door_assets_for_rank,
    edit_door_articulation,
    get_multi_door_asset_start_index,
    get_multi_door_env_asset_indices,
    handle_offsets,
    motion_traj_paths,
)
from DoorOpening.tasks.dooropening.dooropening_adr import DoorOpeningADR
from DoorOpening.tasks.dooropening.multi_dooropening_env_cfg import DooropeningEnvCfg
from DoorOpening.assets.glorbot.glorbot_cfg import glorbot_urdf_path
from isaaclab.sensors import Camera, ContactSensor
from DoorOpening.constants.robot_constants import FULL_JOINT_NAMES, ROBOT_KEY_BODY_NAMES
from DoorOpening.tasks.dooropening.contact_force_utils import get_filtered_contact_force_w
from DoorOpening.utils.pose_utils import world_to_local
from isaaclab.utils.math import quat_conjugate, quat_apply, quat_mul
from DoorOpening.utils.quat_utils import quat_to_6d
from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaLeapSampler
from DoorOpening.utils.viser_pt import format_iterated_record_path, prepare_pointcloud
from typing import Tuple


class DooropeningEnv(DirectRLEnv):
    cfg: DooropeningEnvCfg

    def __init__(self, cfg: DooropeningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._initialize_runtime_event_terms()
        self.early_stopping = True

        self.num_base_joints = len(self.cfg.base_joints)
        self.num_arm_joints = len(self.cfg.arm_joints)

        actuated_joints = self.cfg.base_joints + self.cfg.arm_joints + self.cfg.finger_joints
        self._robot_dof_idx, joint_names = self.robot.find_joints(actuated_joints)
        self._robot_dof_idx = torch.tensor(self._robot_dof_idx, device=self.device)
        self.ref_robot_dof_idx = torch.tensor([FULL_JOINT_NAMES.index(name) for name in joint_names], device=self.device)

        self._robot_key_body_idx, robot_key_body_names = self.robot.find_bodies(self.cfg.robot_key_bodies)
        self._robot_reset_key_body_idx, robot_reset_key_body_names = self.robot.find_bodies(self.cfg.robot_reset_key_bodies)
        self._robot_base_id_in_key_body_idx = robot_key_body_names.index(self.cfg.robot_base_body_link_name)
        self._robot_palm_id_in_key_body_idx = robot_key_body_names.index(self.cfg.robot_palm_link_name)

        self.ref_key_body_idx = [ROBOT_KEY_BODY_NAMES.index(name) for name in robot_key_body_names]
        self.ref_reset_key_body_idx = [ROBOT_KEY_BODY_NAMES.index(name) for name in robot_reset_key_body_names]

        self._robot_base_dof_idx, base_joint_names = self.robot.find_joints(self.cfg.base_joints)
        self._robot_arm_dof_idx, arm_joint_names = self.robot.find_joints(self.cfg.arm_joints)
        self._robot_finger_dof_idx, finger_joint_names = self.robot.find_joints(self.cfg.finger_joints)
        self._robot_base_dof_idx = torch.tensor(self._robot_base_dof_idx, device=self.device)
        self._robot_arm_dof_idx = torch.tensor(self._robot_arm_dof_idx, device=self.device)
        self._robot_finger_dof_idx = torch.tensor(self._robot_finger_dof_idx, device=self.device)
        base_joint_name_to_local_idx = {name: idx for idx, name in enumerate(base_joint_names)}
        self._robot_base_rot_dof_idx = torch.tensor(
            [self._robot_base_dof_idx[base_joint_name_to_local_idx["base_rotation_joint"]]], device=self.device
        )
        self._robot_base_xy_dof_idx = torch.tensor(
            [
                self._robot_base_dof_idx[base_joint_name_to_local_idx["base_x_joint"]],
                self._robot_base_dof_idx[base_joint_name_to_local_idx["base_y_joint"]],
            ],
            device=self.device,
        )
        self._robot_base_rot_local_idx = torch.tensor(
            [base_joint_name_to_local_idx["base_rotation_joint"]], device=self.device
        )
        self._robot_base_xy_local_idx = torch.tensor(
            [base_joint_name_to_local_idx["base_x_joint"], base_joint_name_to_local_idx["base_y_joint"]],
            device=self.device,
        )

        self.ref_base_joint_idx = [FULL_JOINT_NAMES.index(name) for name in base_joint_names]
        self.ref_arm_joint_idx = [FULL_JOINT_NAMES.index(name) for name in arm_joint_names]
        self.ref_finger_joint_idx = [FULL_JOINT_NAMES.index(name) for name in finger_joint_names]

        self._robot_base_link_idx, self.robot_base_link_name = self.robot.find_bodies(self.cfg.base_link_name)
        self._door_body_idx, self.door_body_names = self.door.find_bodies(self.cfg.door_body_names)
        self._door_base_link_idx, self.door_base_link_name = self.door.find_bodies(self.cfg.door_base_frame_name)
        self._door_base_link_idx = self._door_base_link_idx[0]
        self._door_joint_idx, self.door_joint_names = self.door.find_joints(self.cfg.door_joint_names)
        door_joint_name_to_idx = {name: idx for idx, name in enumerate(self.door_joint_names)}
        self._door_board_joint_idx = int(self._door_joint_idx[door_joint_name_to_idx["joint_1"]])
        self._door_hinge_joint_idx = int(self._door_joint_idx[door_joint_name_to_idx["joint_2"]])

        self._robot_base_body_link_idx, self.robot_base_body_link_name = self.robot.find_bodies(self.cfg.robot_base_body_link_name)
        self._robot_base_body_link_idx = self._robot_base_body_link_idx[0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # This is the actual environment/control step and the rate replays should follow.
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 1].to(device=self.device)

        # We are going to update this variables to control the robot
        self.robot_dof_targets = torch.zeros((self.num_envs, len(self._robot_dof_idx)), device=self.device)
        self.applied_robot_dof_targets = torch.zeros_like(self.robot_dof_targets)
        self._target_base_rot_slice = slice(0, 1)
        self._target_base_xy_slice = slice(1, self.num_base_joints)
        self._target_arm_slice = slice(self.num_base_joints, self.num_base_joints + self.num_arm_joints)
        self._target_finger_slice = slice(self.num_base_joints + self.num_arm_joints, self.robot_dof_targets.shape[1])

        # Loading all reward parameters
        self.robot_key_body_pos_scale = self.cfg.robot_key_body_pos_scale
        self.robot_body_quat_scale = self.cfg.robot_body_quat_scale
        self.door_joint_pos_scale = self.cfg.door_joint_pos_scale
        self.robot_base_joint_pos_scale = self.cfg.robot_base_joint_pos_scale
        self.robot_arm_joint_pos_scale = self.cfg.robot_arm_joint_pos_scale
        self.robot_finger_joint_pos_scale = self.cfg.robot_finger_joint_pos_scale
        self.robot_base_joint_vel_scale = self.cfg.robot_base_joint_vel_scale
        self.robot_arm_joint_vel_scale = self.cfg.robot_arm_joint_vel_scale
        self.robot_finger_joint_vel_scale = self.cfg.robot_finger_joint_vel_scale
        self.robot_body_lin_vel_scale = self.cfg.robot_body_lin_vel_scale
        self.robot_body_ang_vel_scale = self.cfg.robot_body_ang_vel_scale

        self.robot_key_body_pos_w = self.cfg.robot_key_body_pos_w
        self.robot_body_quat_w = self.cfg.robot_body_quat_w
        self.door_joint_pos_w = self.cfg.door_joint_pos_w
        self.robot_base_joint_pos_w = self.cfg.robot_base_joint_pos_w
        self.robot_arm_joint_pos_w = self.cfg.robot_arm_joint_pos_w
        self.robot_finger_joint_pos_w = self.cfg.robot_finger_joint_pos_w
        self.robot_base_joint_vel_w = self.cfg.robot_base_joint_vel_w
        self.robot_arm_joint_vel_w = self.cfg.robot_arm_joint_vel_w
        self.robot_finger_joint_vel_w = self.cfg.robot_finger_joint_vel_w
        self.hinge_contact_reward_w = self.cfg.hinge_contact_reward_w
        self.robot_body_lin_vel_w = self.cfg.robot_body_lin_vel_w
        self.robot_body_ang_vel_w = self.cfg.robot_body_ang_vel_w
        self.joint_limit_penalty_w = self.cfg.joint_limit_penalty_w
        self.joint_limit_penalty_margin_ratio = self.cfg.joint_limit_penalty_margin_ratio

        self.reset_key_body_pos_delta_min = self.cfg.reset_key_body_pos_delta_min
        self.reset_key_body_quat_delta_min = self.cfg.reset_key_body_quat_delta_min
        self.reset_key_body_pos_delta_max = self.cfg.reset_key_body_pos_delta_max
        self.reset_key_body_quat_delta_max = self.cfg.reset_key_body_quat_delta_max
        self.reset_door_joint_pos_delta_min = self.cfg.reset_door_joint_pos_delta_min
        self.reset_door_joint_pos_delta_max = self.cfg.reset_door_joint_pos_delta_max

        # self.last_actions = torch.zeros(
        #     (self.num_envs, len(self._robot_dof_idx)),
        #     device=self.device
        # )

        self.twist_indices = self.cfg.twist_indices

        self.num_door_assets = len(handle_offsets)
        self.env_asset_start_index = get_multi_door_asset_start_index(self.num_envs)
        self.env_asset_indices = get_multi_door_env_asset_indices(self.num_envs, device=self.device)
        self.env_family_ids = asset_family_ids.to(device=self.device, dtype=torch.long)[self.env_asset_indices]
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            sample_indices = self.env_asset_indices[: min(8, self.num_envs)].detach().cpu().tolist()
            sample_names = [Path(door_asset_paths[int(asset_idx)]).parent.name for asset_idx in sample_indices]
            print(
                f"[INFO][rank {rank}] Multi-door asset start index {self.env_asset_start_index}; "
                f"first env assets: {sample_names}"
            )
        self.handle_offsets = handle_offsets.to(self.device)[self.env_asset_indices]
        self.board_offsets = board_offsets.to(self.device)[self.env_asset_indices]
        self.use_motion_ref = bool(getattr(self.cfg, "use_motion_ref", True))
        env_to_file_map = self.env_asset_indices.detach().cpu().tolist()
        if self.use_motion_ref:
            from DoorOpening.motion.multi_motion_lib import ReferenceMotionManager

            self.ref_motion_lib = ReferenceMotionManager(
                num_envs=self.num_envs,
                device=self.device,
                reset_from_start=False,
                env_to_file_map=env_to_file_map,
                twist_indices=self.twist_indices,
                step_dt=self.dt,
            )
            default_trial_steps = math.ceil(
                max(float(self.ref_motion_lib.num_frames - 1), 0.0) / max(float(self.ref_motion_lib.frame_step), 1e-6)
            ) + 1
            self.success_frame_idx = float(self.ref_motion_lib.num_frames - 1)
        else:
            self.ref_motion_lib = None
            default_trial_steps = max(1, math.ceil(float(self.cfg.episode_length_s) / max(float(self.dt), 1e-6)))
            self.success_frame_idx = float("inf")
        self.prob_get_first_key_frame = None
        self.max_trial_steps = default_trial_steps * torch.ones_like(self.episode_length_buf, device=self.device)
        self.games_to_track = 100
        self.completed_successes = deque(maxlen=self.games_to_track)
        self.completed_successes_by_family = {
            family_name: deque(maxlen=self.games_to_track) for family_name in DOOR_FAMILY_NAMES
        }
        self.episode_reached_last_frame = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        torch.set_printoptions(precision=4, sci_mode=False)

        self.reset_progress_total = self.cfg.reset_progress_total
        self.adr_reset_progress_total = self.cfg.adr_reset_progress_total
        self._rlgames_env_frames = 0

        self.alive_base = self.cfg.alive_base
        self.alive_bonus = self.cfg.alive_bonus
        self.termination_penalty = self.cfg.termination_penalty

        # DEXTRAH-style split: EventTerms handle reset-time physics DR, while the env samples
        # reset/observation/controller noise from ADR.
        self.dooropening_adr = DoorOpeningADR(self.event_manager, self.cfg.adr_cfg_dict, self.cfg.adr_custom_cfg_dict)
        self._adr_enabled = bool(self.cfg.enable_adr)
        initial_adr_increments = self.cfg.starting_adr_increments if self._adr_enabled else 0
        self.dooropening_adr.set_num_increments(initial_adr_increments)

        self.robot_spawn_noise_widths = self._make_env_buffer_dict(self.cfg.adr_custom_cfg_dict["robot_spawn"].keys())
        robot_state_cfg = self.cfg.adr_custom_cfg_dict["robot_state_noise"]
        self.robot_state_noise_widths = self._make_env_buffer_dict(
            [key for key in robot_state_cfg if key.endswith("_noise")]
        )
        self.robot_state_biases = self._make_env_buffer_dict([key for key in robot_state_cfg if key.endswith("_bias")])
        self.student_joint_pos_noise_widths = self._make_env_buffer_dict(
            [
                "base_xy_joint_pos_noise",
                "base_rot_joint_pos_noise",
                "arm_joint_pos_noise",
                "finger_joint_pos_noise",
            ]
        )
        self.student_joint_pos_biases = self._make_env_buffer_dict(
            [
                "base_xy_joint_pos_bias",
                "base_rot_joint_pos_bias",
                "arm_joint_pos_bias",
                "finger_joint_pos_bias",
            ]
        )
        self._door_nominal_joint_stiffness = self.door.data.joint_stiffness.clone()
        self._door_nominal_joint_damping = self.door.data.joint_damping.clone()
        self._dr_metrics_interval = max(int(self.cfg.dr_metrics_interval), 1)
        self._log_verbose_dr_metrics = bool(self.cfg.log_verbose_dr_metrics)
        self._init_viser_pointcloud_recording()

    def _get_filtered_contact_force_w(self, sensor, expected_num_envs=None) -> torch.Tensor:
        return get_filtered_contact_force_w(sensor, expected_num_envs=expected_num_envs)

    def set_train_info(self, env_frames: int, algo=None, **kwargs):
        self._rlgames_env_frames = int(env_frames)

    def _get_curriculum_step_count(self) -> int:
        # Curriculum/reset scheduling should follow actual env progress, not logging side effects.
        return int(self.common_step_counter)

    def _init_viser_pointcloud_recording(self):
        """Initialize optional teacher-training point-cloud replay dumps."""

        self._reset_viser_pointcloud_runtime_state()

        record_cfg = dict(getattr(self.cfg, "viser_pointcloud", {}) or {})
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0 or not bool(record_cfg.get("enabled", False)):
            return

        self.viser_pointcloud_enabled = True
        self._configure_viser_pointcloud_from_cfg(record_cfg)

        self._init_viser_pointcloud_asset_samplers()
        self._init_viser_pointcloud_metadata()

        print(f"Viser .pt point-cloud capture enabled for env {self._viser_pointcloud_env_id}.")
        print(
            "Replay chunks will be written as "
            f"{format_iterated_record_path(self.viser_pointcloud_path, '<chunk_tag>')}"
        )
        print(f"Replay chunks will flush every {self._viser_pointcloud_save_interval} iterations.")
        if self._viser_pointcloud_max_frames > 0:
            print(f"Each replay chunk will keep at most {self._viser_pointcloud_max_frames} frames.")

    def _reset_viser_pointcloud_runtime_state(self):
        self.viser_pointcloud_enabled = False
        self.viser_pointcloud_path = ""
        self.viser_pointcloud_frames = []
        self.viser_pointcloud_frame_count = 0
        self.viser_pointcloud_chunk_index = 0
        self.viser_pointcloud_chunk_start_iteration = None
        self.viser_pointcloud_latest_iteration = None
        self._viser_pointcloud_metadata = {}
        self._viser_robot_sampler = None
        self._viser_door_samplers = {}
        self._viser_env_asset_idx = None

    def _configure_viser_pointcloud_from_cfg(self, record_cfg: dict):
        self._viser_pointcloud_env_id = max(0, min(int(record_cfg["env_id"]), self.num_envs - 1))
        self._viser_pointcloud_capture_interval = max(
            1,
            int(record_cfg.get("capture_interval", record_cfg.get("record_interval", 1))),
        )
        self._viser_pointcloud_save_interval = max(
            1,
            int(record_cfg.get("save_interval", record_cfg.get("raw_interval", 5000))),
        )
        self._viser_pointcloud_max_frames = max(0, int(record_cfg.get("max_frames", 2000)))
        self._viser_pointcloud_max_points = int(record_cfg.get("max_points", 6000))
        self._viser_robot_num_points = int(record_cfg.get("robot_num_points", record_cfg.get("num_points", 4096)))
        self._viser_door_num_points = int(record_cfg.get("door_num_points", record_cfg.get("num_points", 4096)))

        configured_path = Path(str(record_cfg.get("path", record_cfg.get("raw_path", "teacher_viser_replay.pt")))).expanduser()
        if not configured_path.is_absolute():
            log_dir = Path(str(getattr(self.cfg, "log_dir", os.getcwd())))
            configured_path = log_dir / configured_path
        self.viser_pointcloud_path = str(configured_path)
        raw_dir = os.path.dirname(self.viser_pointcloud_path)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)

    def _init_viser_pointcloud_asset_samplers(self):
        asset_index_by_dir = {Path(asset_path).resolve().parent: idx for idx, asset_path in enumerate(door_asset_paths)}
        if self.ref_motion_lib is None:
            self._viser_env_asset_idx = self.env_asset_indices.to(device=self.device, dtype=torch.long)
        else:
            motion_to_asset_idx = []
            for motion_path in motion_traj_paths:
                motion_dir = Path(motion_path).resolve().parent
                if motion_dir not in asset_index_by_dir:
                    raise KeyError(f"Could not map motion file '{motion_path}' to a door asset path.")
                motion_to_asset_idx.append(asset_index_by_dir[motion_dir])
            motion_to_asset_idx = torch.tensor(motion_to_asset_idx, device=self.device, dtype=torch.long)
            env_motion_idx = self.ref_motion_lib.env_to_file_map.to(device=self.device, dtype=torch.long)
            self._viser_env_asset_idx = motion_to_asset_idx[env_motion_idx]

        selected_asset_idx = int(self._viser_env_asset_idx[self._viser_pointcloud_env_id].detach().cpu().item())
        self._viser_door_samplers = {
            selected_asset_idx: FrankaLeapSampler(
                door_asset_paths[selected_asset_idx],
                device=self.device,
                num_points=self._viser_door_num_points,
            )
        }

        self._viser_robot_sampler = FrankaLeapSampler(
            glorbot_urdf_path,
            device=self.device,
            num_points=self._viser_robot_num_points,
        )
        robot_sampler_joint_names = list(self._viser_robot_sampler.robot.actuated_joint_names)
        robot_joint_ids, robot_joint_names = self.robot.find_joints(robot_sampler_joint_names)
        if len(robot_joint_ids) != len(robot_sampler_joint_names):
            raise ValueError("Could not map every robot sampler joint to the IsaacLab robot articulation.")
        self._viser_robot_sampler_joint_ids = torch.tensor(robot_joint_ids, device=self.device, dtype=torch.long)
        robot_joint_name_to_idx = {name: idx for idx, name in enumerate(robot_joint_names)}
        self._viser_robot_sampler_joint_reorder = [robot_joint_name_to_idx[name] for name in robot_sampler_joint_names]
        self._viser_robot_root_body_idx = int(self._robot_base_link_idx[0])

    def _init_viser_pointcloud_metadata(self):
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_dt = float(getattr(sim_cfg, "dt", self.cfg.sim_dt))
        env_step_dt = max(float(getattr(self, "dt", sim_dt * int(getattr(self.cfg, "decimation", 1)))), 1e-6)
        self._viser_pointcloud_metadata = {
            "format": "dooropening_viser_replay_v2",
            "capture_mode": "teacher_training_pointcloud_chunk",
            "pointcloud_frame": "world",
            "pointcloud_streams": [
                {"name": "robot", "label": "Robot", "color": (79, 195, 247), "point_size_scale": 1.0},
                {"name": "door", "label": "Door", "color": (255, 193, 7), "point_size_scale": 1.0},
            ],
            "pointcloud_source": "articulation_sampler",
            "sim_dt": sim_dt,
            "decimation": int(getattr(self.cfg, "decimation", 1)),
            "env_step_dt": env_step_dt,
            "env_step_fps": 1.0 / env_step_dt,
            "frame_dt": env_step_dt * self._viser_pointcloud_capture_interval,
            "frame_fps": 1.0 / (env_step_dt * self._viser_pointcloud_capture_interval),
            "pointcloud_sensor_dt": env_step_dt * self._viser_pointcloud_capture_interval,
            "pointcloud_sensor_fps": 1.0 / (env_step_dt * self._viser_pointcloud_capture_interval),
            "raw_cloud_config": {
                "max_points": self._viser_pointcloud_max_points,
                "robot_num_points": self._viser_robot_num_points,
                "door_num_points": self._viser_door_num_points,
                "capture_interval": self._viser_pointcloud_capture_interval,
            },
            "glorbot_urdf_path": str(glorbot_urdf_path),
            "robot_joint_names": list(getattr(self.robot, "joint_names", [])),
            "door_joint_names": list(getattr(self.door, "joint_names", [])),
        }

    def _sample_viser_robot_pointcloud_world(self, env_id: int) -> torch.Tensor:
        robot_joint_pos = self.robot.data.joint_pos[env_id : env_id + 1, self._viser_robot_sampler_joint_ids]
        robot_joint_pos = robot_joint_pos[:, self._viser_robot_sampler_joint_reorder]
        robot_local_pcd = self._viser_robot_sampler.sample(robot_joint_pos)
        robot_root_pos_w = self.robot.data.body_pos_w[env_id : env_id + 1, self._viser_robot_root_body_idx]
        robot_root_quat_w = self.robot.data.body_quat_w[env_id : env_id + 1, self._viser_robot_root_body_idx]
        quat = robot_root_quat_w.unsqueeze(1).expand(-1, robot_local_pcd.shape[1], -1)
        return (quat_apply(quat, robot_local_pcd) + robot_root_pos_w.unsqueeze(1))[0]

    def _sample_viser_door_pointcloud_world(self, env_id: int) -> torch.Tensor:
        asset_idx = int(self._viser_env_asset_idx[env_id].detach().cpu().item())
        sampler = self._viser_door_samplers.get(asset_idx)
        if sampler is None:
            sampler = FrankaLeapSampler(
                door_asset_paths[asset_idx],
                device=self.device,
                num_points=self._viser_door_num_points,
            )
            self._viser_door_samplers[asset_idx] = sampler

        door_joint_pos = self.door.data.joint_pos[env_id : env_id + 1]
        local_pcd = sampler.sample(door_joint_pos)
        door_base_pos_w = self.door.data.body_pos_w[env_id : env_id + 1, self._door_base_link_idx]
        door_base_quat_w = self.door.data.body_quat_w[env_id : env_id + 1, self._door_base_link_idx]
        quat = door_base_quat_w.unsqueeze(1).expand(-1, local_pcd.shape[1], -1)
        return (quat_apply(quat, local_pcd) + door_base_pos_w.unsqueeze(1))[0]

    def _trim_viser_pointcloud_frames(self):
        if (
            self._viser_pointcloud_max_frames > 0
            and len(self.viser_pointcloud_frames) > self._viser_pointcloud_max_frames
        ):
            self.viser_pointcloud_frames = self.viser_pointcloud_frames[-self._viser_pointcloud_max_frames :]
        self.viser_pointcloud_frame_count = len(self.viser_pointcloud_frames)
        if self.viser_pointcloud_frame_count > 0:
            self.viser_pointcloud_chunk_start_iteration = int(self.viser_pointcloud_frames[0]["iteration"])
        else:
            self.viser_pointcloud_chunk_start_iteration = None

    def _build_viser_pointcloud_payload(self, chunk_complete: bool) -> dict:
        latest_iteration = int(self.viser_pointcloud_latest_iteration or 0)
        payload = {
            **self._viser_pointcloud_metadata,
            "chunk_index": int(self.viser_pointcloud_chunk_index),
            "chunk_env_id": int(self._viser_pointcloud_env_id),
            "chunk_start_iteration": None
            if self.viser_pointcloud_chunk_start_iteration is None
            else int(self.viser_pointcloud_chunk_start_iteration),
            "chunk_end_iteration": latest_iteration,
            "chunk_complete": bool(chunk_complete),
            "chunk_frame_count": int(self.viser_pointcloud_frame_count),
            "episode_index": int(self.viser_pointcloud_chunk_index),
            "episode_env_id": int(self._viser_pointcloud_env_id),
            "episode_start_iteration": None
            if self.viser_pointcloud_chunk_start_iteration is None
            else int(self.viser_pointcloud_chunk_start_iteration),
            "episode_end_iteration": latest_iteration,
            "episode_complete": bool(chunk_complete),
            "episode_frame_count": int(self.viser_pointcloud_frame_count),
            "frames": self.viser_pointcloud_frames,
        }
        return payload

    def _flush_viser_pointcloud_recording(self, chunk_complete: bool, reason: str):
        if not self.viser_pointcloud_enabled or self.viser_pointcloud_frame_count <= 0:
            return

        latest_iteration = int(self.viser_pointcloud_latest_iteration or 0)
        record_tag = f"chunk_{self.viser_pointcloud_chunk_index:04d}_iter_{latest_iteration}"
        torch.save(
            self._build_viser_pointcloud_payload(chunk_complete=chunk_complete),
            format_iterated_record_path(self.viser_pointcloud_path, record_tag),
        )
        status = "complete" if chunk_complete else "partial"
        print(
            "Saved {} Viser .pt chunk {} for env {} with {} frames ({}).".format(
                status,
                self.viser_pointcloud_chunk_index,
                self._viser_pointcloud_env_id,
                self.viser_pointcloud_frame_count,
                reason,
            )
        )

        self.viser_pointcloud_frames = []
        self.viser_pointcloud_frame_count = 0
        self.viser_pointcloud_chunk_start_iteration = None
        self.viser_pointcloud_latest_iteration = None

    def _maybe_flush_viser_pointcloud_snapshot(self, iteration: int):
        if not self.viser_pointcloud_enabled or self.viser_pointcloud_frame_count <= 0:
            return
        if (int(iteration) + 1) % self._viser_pointcloud_save_interval != 0:
            return
        self._flush_viser_pointcloud_recording(
            chunk_complete=True,
            reason=f"save_interval {self._viser_pointcloud_save_interval} reached at iteration {int(iteration)}",
        )

    def _maybe_record_viser_pointcloud(self):
        if not self.viser_pointcloud_enabled or self._viser_env_asset_idx is None:
            return
        iteration = int(self.common_step_counter)
        if iteration % self._viser_pointcloud_capture_interval != 0:
            return

        env_id = self._viser_pointcloud_env_id
        robot_points_world = self._sample_viser_robot_pointcloud_world(env_id)
        door_points_world = self._sample_viser_door_pointcloud_world(env_id)
        robot_points_world = prepare_pointcloud(
            robot_points_world,
            max_points=self._viser_pointcloud_max_points,
        ).to(dtype=torch.float16)
        door_points_world = prepare_pointcloud(
            door_points_world,
            max_points=self._viser_pointcloud_max_points,
        ).to(dtype=torch.float16)
        door_asset_idx = int(self._viser_env_asset_idx[env_id].detach().cpu().item())

        if self.viser_pointcloud_chunk_start_iteration is None:
            self.viser_pointcloud_chunk_index += 1
            self.viser_pointcloud_chunk_start_iteration = iteration
        self.viser_pointcloud_latest_iteration = iteration

        frame_record = {
            "iteration": iteration,
            "sim_frame": int(self._rlgames_env_frames if self._rlgames_env_frames > 0 else iteration),
            "env_id": int(env_id),
            "pointcloud_source": "articulation_sampler",
            "robot_points_world": robot_points_world,
            "door_points_world": door_points_world,
            "pointclouds": {
                "robot": robot_points_world,
                "door": door_points_world,
            },
            "rlgames_env_frames": int(self._rlgames_env_frames),
            "robot_joint_pos": self.robot.data.joint_pos[env_id].detach().cpu().to(dtype=torch.float32),
            "door_joint_pos": self.door.data.joint_pos[env_id].detach().cpu().to(dtype=torch.float32),
            "robot_base_pos_w": self.robot.data.body_pos_w[env_id, self._robot_base_body_link_idx].detach().cpu().to(dtype=torch.float32),
            "robot_base_quat_w": self.robot.data.body_quat_w[env_id, self._robot_base_body_link_idx].detach().cpu().to(dtype=torch.float32),
            "door_base_pos_w": self.door.data.body_pos_w[env_id, self._door_base_link_idx].detach().cpu().to(dtype=torch.float32),
            "door_base_quat_w": self.door.data.body_quat_w[env_id, self._door_base_link_idx].detach().cpu().to(dtype=torch.float32),
            "door_asset_idx": door_asset_idx,
            "door_asset_path": str(door_asset_paths[door_asset_idx]),
        }
        self.viser_pointcloud_frames.append(frame_record)
        self._trim_viser_pointcloud_frames()
        self._maybe_flush_viser_pointcloud_snapshot(iteration)

    def _initialize_runtime_event_terms(self):
        if not self.cfg.events:
            return

        # IsaacLab's EventManager initializes class-based terms eagerly for "prestartup",
        # but reset-time class terms still need their play-time processing once physics is live.
        for mode in self.event_manager.available_modes:
            if mode == "prestartup":
                continue
            term_names = self.event_manager._mode_term_names.get(mode, [])
            term_cfgs = self.event_manager._mode_term_cfgs.get(mode, [])
            for term_name, term_cfg in zip(term_names, term_cfgs):
                if inspect.isclass(term_cfg.func):
                    self.event_manager._process_term_cfg_at_play(term_name, term_cfg)

    def _activate_door_contact_reporters(self):
        try:
            from isaaclab.sim.schemas import schemas as sim_schemas
        except ImportError:
            from isaaclab.sim.schemas.schemas import activate_contact_sensors
        else:
            activate_contact_sensors = sim_schemas.activate_contact_sensors

        for env_id in range(self.num_envs):
            activate_contact_sensors(f"/World/envs/env_{env_id}/Door", 1.0)

    def _setup_scene(self):
        configure_multi_door_assets_for_rank(self.cfg.door_cfg, int(self.cfg.scene.num_envs))
        self.robot = Articulation(self.cfg.robot_cfg)
        self.door = Articulation(self.cfg.door_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Do not clone env_0 over the other envs for heterogeneous multi-door
        # scenes. The MultiAssetSpawner has already populated each env.
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["door"] = self.door
        self._activate_door_contact_reporters()
        self.pointcloud_camera = None
        pointcloud_render_mode = str(getattr(self.cfg, "pointcloud_render_mode", "none")).lower()
        if pointcloud_render_mode not in {"none", "depth", "lidar"}:
            raise ValueError(
                "Unsupported pointcloud_render_mode "
                f"'{pointcloud_render_mode}'. Expected one of ['none', 'depth', 'lidar']."
            )
        # The render mode is the source of truth: only depth mode enables the x5 pointcloud camera.
        enable_pointcloud_camera = pointcloud_render_mode == "depth"
        if enable_pointcloud_camera:
            self.pointcloud_camera = Camera(self.cfg.pointcloud_camera_cfg)
            self.scene.sensors["pointcloud_camera"] = self.pointcloud_camera
        self.scene.sensors["contact_forces_door2"] = ContactSensor(self.cfg.contact_forces_door2)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)    

    def _make_env_buffer_dict(self, keys):
        return {key: torch.zeros((self.num_envs, 1), device=self.device) for key in keys}

    def _custom_param_upper_limit(self, group: str, name: str) -> float:
        return float(self.cfg.adr_custom_cfg_dict[group][name][1])

    def _current_custom_param(self, group: str, name: str) -> float:
        if not self._adr_enabled:
            return 0.0
        return self.dooropening_adr.get_custom_param_value(group, name)

    def _current_event_param(self, term_name: str, param_name: str):
        return self.event_manager.get_term_cfg(term_name).params[param_name]

    def _refresh_nominal_door_joint_gains(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._door_nominal_joint_stiffness.copy_(self.door.data.joint_stiffness)
            self._door_nominal_joint_damping.copy_(self.door.data.joint_damping)
            return
        self._door_nominal_joint_stiffness[env_ids] = self.door.data.joint_stiffness[env_ids]
        self._door_nominal_joint_damping[env_ids] = self.door.data.joint_damping[env_ids]

    def _compute_curriculum_progress(self, progress_total: float) -> float:
        progress_total = max(float(progress_total), 1.0)
        return min(float(self._get_curriculum_step_count()) / progress_total, 1.0)

    def _mean_completed_metric(self, values) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    def _update_last_frame_tracker(self):
        if self.ref_motion_lib is None:
            return
        current_frame_idx = torch.clamp(
            self.ref_motion_lib.frame_idx.to(device=self.device, dtype=torch.float32),
            max=self.success_frame_idx,
        )
        self.episode_reached_last_frame |= current_frame_idx >= self.success_frame_idx

    def _get_reached_last_frame_mask(self) -> torch.Tensor:
        if self.ref_motion_lib is None:
            return torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        current_frame_idx = torch.clamp(
            self.ref_motion_lib.frame_idx.to(device=self.device, dtype=torch.float32),
            max=self.success_frame_idx,
        )
        return current_frame_idx >= self.success_frame_idx

    def _update_success_metrics(self):
        self._update_last_frame_tracker()

        done_mask = torch.nonzero(self.reset_buf, as_tuple=False).squeeze(-1)
        if done_mask.numel() > 0:
            episode_successes_tensor = (
                self.episode_reached_last_frame[done_mask] | self.reset_time_outs[done_mask]
            ).to(dtype=torch.float32)
            episode_successes = episode_successes_tensor.detach().cpu().tolist()
            self.completed_successes.extend(float(value) for value in episode_successes)
            done_family_ids = self.env_family_ids[done_mask].detach().cpu().tolist()
            for family_id, value in zip(done_family_ids, episode_successes):
                family_name = DOOR_FAMILY_NAMES[int(family_id)]
                self.completed_successes_by_family[family_name].append(float(value))

        self.extras["stats/active_success_rate"] = self.episode_reached_last_frame.float().mean().item()
        success_rate = self._mean_completed_metric(self.completed_successes)
        if success_rate is not None:
            self.extras["stats/success_rate"] = success_rate

        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            family_mask = self.env_family_ids == int(family_id)
            if family_mask.any():
                self.extras[f"stats/family_active_success_rate/{family_name}"] = (
                    self.episode_reached_last_frame[family_mask].float().mean().item()
                )
            family_success_rate = self._mean_completed_metric(self.completed_successes_by_family[family_name])
            if family_success_rate is not None:
                self.extras[f"stats/family_success_rate/{family_name}"] = family_success_rate

    def _log_dr_metrics(self):
        step_count = self._get_curriculum_step_count()
        if step_count % self._dr_metrics_interval != 0:
            return

        progress = self._compute_curriculum_progress(self.adr_reset_progress_total)
        scheduled_increment = int(progress * self.cfg.num_adr_increments)
        scheduled_increment = max(self.cfg.starting_adr_increments, scheduled_increment)
        scheduled_increment = min(self.cfg.num_adr_increments, scheduled_increment)

        robot_stiffness = self._current_event_param(
            "robot_joint_stiffness_and_damping", "stiffness_distribution_params"
        )
        robot_damping = self._current_event_param(
            "robot_joint_stiffness_and_damping", "damping_distribution_params"
        )
        board_stiffness = self._current_event_param(
            "door_board_joint_stiffness_and_damping", "stiffness_distribution_params"
        )
        board_damping = self._current_event_param(
            "door_board_joint_stiffness_and_damping", "damping_distribution_params"
        )
        hinge_stiffness = self._current_event_param(
            "door_hinge_joint_stiffness_and_damping", "stiffness_distribution_params"
        )
        hinge_damping = self._current_event_param(
            "door_hinge_joint_stiffness_and_damping", "damping_distribution_params"
        )

        self.extras["dr/increment"] = float(self.dooropening_adr.increment_counter)
        self.extras["dr/robot_stiffness_min"] = float(robot_stiffness[0])
        self.extras["dr/robot_stiffness_max"] = float(robot_stiffness[1])
        self.extras["dr/robot_damping_min"] = float(robot_damping[0])
        self.extras["dr/robot_damping_max"] = float(robot_damping[1])
        self.extras["dr/door_board_stiffness_min"] = float(board_stiffness[0])
        self.extras["dr/door_board_stiffness_max"] = float(board_stiffness[1])
        self.extras["dr/door_board_damping_min"] = float(board_damping[0])
        self.extras["dr/door_board_damping_max"] = float(board_damping[1])
        self.extras["dr/door_hinge_stiffness_min"] = float(hinge_stiffness[0])
        self.extras["dr/door_hinge_stiffness_max"] = float(hinge_stiffness[1])
        self.extras["dr/door_hinge_damping_min"] = float(hinge_damping[0])
        self.extras["dr/door_hinge_damping_max"] = float(hinge_damping[1])

        self.extras["dr_limit/spawn_arm_joint_pos_noise"] = self._current_custom_param("robot_spawn", "arm_joint_pos_noise")
        self.extras["dr_limit/spawn_finger_joint_pos_noise"] = self._current_custom_param(
            "robot_spawn", "finger_joint_pos_noise"
        )
        self.extras["dr_limit/obs_arm_joint_pos_noise"] = self._current_custom_param(
            "robot_state_noise", "arm_joint_pos_noise"
        )
        self.extras["dr_limit/obs_finger_joint_pos_noise"] = self._current_custom_param(
            "robot_state_noise", "finger_joint_pos_noise"
        )
        self.extras["dr_limit/obs_arm_joint_vel_noise"] = self._current_custom_param(
            "robot_state_noise", "arm_joint_vel_noise"
        )
        self.extras["dr_limit/obs_finger_joint_vel_noise"] = self._current_custom_param(
            "robot_state_noise", "finger_joint_vel_noise"
        )
        self.extras["dr_limit/target_lag_alpha"] = self._current_custom_param("pd_targets", "target_lag_alpha")

        if not self._log_verbose_dr_metrics:
            return

        board_nominal_stiffness = self._door_nominal_joint_stiffness[:, self._door_board_joint_idx]
        board_nominal_damping = self._door_nominal_joint_damping[:, self._door_board_joint_idx]
        hinge_nominal_stiffness = self._door_nominal_joint_stiffness[:, self._door_hinge_joint_idx]
        hinge_nominal_damping = self._door_nominal_joint_damping[:, self._door_hinge_joint_idx]

        # self.extras["dr_sample/spawn_arm_joint_pos_noise_mean"] = self.robot_spawn_noise_widths["arm_joint_pos_noise"].mean().item()
        # self.extras["dr_sample/spawn_finger_joint_pos_noise_mean"] = self.robot_spawn_noise_widths[
        #     "finger_joint_pos_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_arm_joint_pos_noise_mean"] = self.robot_state_noise_widths[
        #     "arm_joint_pos_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_finger_joint_pos_noise_mean"] = self.robot_state_noise_widths[
        #     "finger_joint_pos_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_arm_joint_vel_noise_mean"] = self.robot_state_noise_widths[
        #     "arm_joint_vel_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_finger_joint_vel_noise_mean"] = self.robot_state_noise_widths[
        #     "finger_joint_vel_noise"
        # ].mean().item()
        self.extras["dr_sample/door_board_stiffness_mean"] = board_nominal_stiffness.mean().item()
        self.extras["dr_sample/door_board_stiffness_min"] = board_nominal_stiffness.min().item()
        self.extras["dr_sample/door_board_stiffness_max"] = board_nominal_stiffness.max().item()
        self.extras["dr_sample/door_board_damping_mean"] = board_nominal_damping.mean().item()
        self.extras["dr_sample/door_board_damping_min"] = board_nominal_damping.min().item()
        self.extras["dr_sample/door_board_damping_max"] = board_nominal_damping.max().item()
        self.extras["dr_sample/door_hinge_stiffness_mean"] = hinge_nominal_stiffness.mean().item()
        self.extras["dr_sample/door_hinge_stiffness_min"] = hinge_nominal_stiffness.min().item()
        self.extras["dr_sample/door_hinge_stiffness_max"] = hinge_nominal_stiffness.max().item()
        self.extras["dr_sample/door_hinge_damping_mean"] = hinge_nominal_damping.mean().item()
        self.extras["dr_sample/door_hinge_damping_min"] = hinge_nominal_damping.min().item()
        self.extras["dr_sample/door_hinge_damping_max"] = hinge_nominal_damping.max().item()

    def _update_adr_ranges(self):
        if not self._adr_enabled:
            return

        progress = self._compute_curriculum_progress(self.adr_reset_progress_total)
        target_increment = int(progress * self.cfg.num_adr_increments)
        target_increment = max(self.cfg.starting_adr_increments, target_increment)
        target_increment = min(self.cfg.num_adr_increments, target_increment)
        if target_increment != self.dooropening_adr.increment_counter:
            self.dooropening_adr.set_num_increments(target_increment)

    def _sample_reset_randomization(self, env_ids: torch.Tensor):
        num_ids = len(env_ids)

        # Sample per-env widths and biases once at reset, then reuse them during the whole episode.
        for key in self.robot_spawn_noise_widths:
            value = self._current_custom_param("robot_spawn", key)
            self.robot_spawn_noise_widths[key][env_ids, 0] = value * torch.rand(num_ids, device=self.device)

        for key in self.robot_state_noise_widths:
            value = self._current_custom_param("robot_state_noise", key)
            self.robot_state_noise_widths[key][env_ids, 0] = value * torch.rand(num_ids, device=self.device)

        for key in self.robot_state_biases:
            value = self._current_custom_param("robot_state_noise", key)
            width = value * torch.rand(num_ids, device=self.device)
            self.robot_state_biases[key][env_ids, 0] = width * (torch.rand(num_ids, device=self.device) - 0.5)

        for key in self.student_joint_pos_noise_widths:
            self.student_joint_pos_noise_widths[key][env_ids, 0] = self._custom_param_upper_limit(
                "robot_state_noise", key
            )

        for key in self.student_joint_pos_biases:
            bias_limit = self._custom_param_upper_limit("robot_state_noise", key)
            self.student_joint_pos_biases[key][env_ids, 0] = bias_limit * (
                torch.rand(num_ids, device=self.device) - 0.5
            )

    def _uniform_noise_from_buffers(
        self,
        values: torch.Tensor,
        width_buffers: dict[str, torch.Tensor],
        width_key: str,
        bias_buffers: dict[str, torch.Tensor] | None = None,
        bias_key: str | None = None,
    ):
        width = width_buffers[width_key]
        while width.dim() < values.dim():
            width = width.unsqueeze(-1)
        noise = width * 2.0 * (torch.rand_like(values) - 0.5)
        if bias_key is None:
            return values + noise
        if bias_buffers is None:
            raise RuntimeError("Bias buffers must be provided when bias_key is set.")
        bias = bias_buffers[bias_key]
        while bias.dim() < values.dim():
            bias = bias.unsqueeze(-1)
        return values + noise + bias

    def _uniform_noise_like(self, values: torch.Tensor, width_key: str, bias_key: str | None = None):
        return self._uniform_noise_from_buffers(
            values,
            width_buffers=self.robot_state_noise_widths,
            width_key=width_key,
            bias_buffers=self.robot_state_biases,
            bias_key=bias_key,
        )

    def get_student_joint_pos_obs(self, use_noise: bool = False) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos
        if not use_noise:
            return joint_pos

        student_joint_pos = joint_pos.clone()
        student_joint_pos[:, self._robot_base_rot_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_base_rot_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="base_rot_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="base_rot_joint_pos_bias",
        )
        student_joint_pos[:, self._robot_base_xy_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_base_xy_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="base_xy_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="base_xy_joint_pos_bias",
        )
        student_joint_pos[:, self._robot_arm_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_arm_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="arm_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="arm_joint_pos_bias",
        )
        student_joint_pos[:, self._robot_finger_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_finger_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="finger_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="finger_joint_pos_bias",
        )
        return student_joint_pos

    def get_student_base_joint_vel_obs(self, use_noise: bool = False) -> torch.Tensor:
        base_joint_vel = self.robot.data.joint_vel[:, self._robot_base_dof_idx]
        if not use_noise:
            return base_joint_vel

        noisy_base_joint_vel = base_joint_vel.clone()
        noisy_base_joint_vel[:, self._target_base_rot_slice] = self._uniform_noise_from_buffers(
            noisy_base_joint_vel[:, self._target_base_rot_slice],
            width_buffers=self.robot_state_noise_widths,
            width_key="base_rot_joint_vel_noise",
            bias_buffers=self.robot_state_biases,
            bias_key="base_rot_joint_vel_bias",
        )
        noisy_base_joint_vel[:, self._target_base_xy_slice] = self._uniform_noise_from_buffers(
            noisy_base_joint_vel[:, self._target_base_xy_slice],
            width_buffers=self.robot_state_noise_widths,
            width_key="base_xy_joint_vel_noise",
            bias_buffers=self.robot_state_biases,
            bias_key="base_xy_joint_vel_bias",
        )
        return noisy_base_joint_vel

    def _apply_spawn_noise(self, env_ids: torch.Tensor):
        # Reset disturbance is applied around the reference motion state, with separate scales for base, arm, and fingers.
        base_xy_noise = self.robot_spawn_noise_widths["base_xy_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), len(self._robot_base_xy_dof_idx)), device=self.device) - 1.0
        )
        base_rot_noise = self.robot_spawn_noise_widths["base_rot_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), 1), device=self.device) - 1.0
        )
        arm_noise = self.robot_spawn_noise_widths["arm_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), len(self._robot_arm_dof_idx)), device=self.device) - 1.0
        )
        finger_noise = self.robot_spawn_noise_widths["finger_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), len(self._robot_finger_dof_idx)), device=self.device) - 1.0
        )

        self.joint_pos[env_ids[:, None], self._robot_base_xy_dof_idx[None, :]] += base_xy_noise
        self.joint_pos[env_ids[:, None], self._robot_base_rot_dof_idx[None, :]] += base_rot_noise
        self.joint_pos[env_ids[:, None], self._robot_arm_dof_idx[None, :]] += arm_noise
        self.joint_pos[env_ids[:, None], self._robot_finger_dof_idx[None, :]] += finger_noise

        self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]] = torch.clamp(
            self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]],
            self.robot_dof_lower_limits[None, :],
            self.robot_dof_upper_limits[None, :],
        )

    def _get_policy_target_noise(self):
        target_noise = torch.zeros_like(self.robot_dof_targets)
        base_xy_target_noise = self._current_custom_param("pd_targets", "base_xy_target_noise")
        base_rot_target_noise = self._current_custom_param("pd_targets", "base_rot_target_noise")
        arm_target_noise = self._current_custom_param("pd_targets", "arm_target_noise")
        finger_target_noise = self._current_custom_param("pd_targets", "finger_target_noise")

        target_noise[:, self._target_base_xy_slice] = base_xy_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_base_xy_slice]) - 1.0
        )
        target_noise[:, self._target_base_rot_slice] = base_rot_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_base_rot_slice]) - 1.0
        )
        target_noise[:, self._target_arm_slice] = arm_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_arm_slice]) - 1.0
        )
        target_noise[:, self._target_finger_slice] = finger_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_finger_slice]) - 1.0
        )
        return target_noise

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scaled_actions = actions.clamp(-1.0, 1.0)
        scaled_actions[:, :self.num_base_joints] = scaled_actions[:, :self.num_base_joints] * self.cfg.base_action_scale
        scaled_actions[:, self.num_base_joints:self.num_base_joints + self.num_arm_joints] = (
            scaled_actions[:, self.num_base_joints:self.num_base_joints + self.num_arm_joints] * self.cfg.arm_action_scale
        )
        scaled_actions[:, self.num_base_joints + self.num_arm_joints:] = (
            scaled_actions[:, self.num_base_joints + self.num_arm_joints:] * self.cfg.finger_action_scale
        )
        return scaled_actions

    def _pre_physics_step(self, actions: torch.Tensor):
        # delta actions
        self.scaled_actions = self._scale_actions(actions)
        targets = self.robot_dof_targets + self.dt * self.scaled_actions
        self.scene.sensors["contact_forces_door2"].update(self.cfg.sim_dt)

        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # Advance the reference once per RL/env step. IsaacLab will call _apply_action()
        # decimation times, so stepping here keeps replay duration tied to env dt instead
        # of being multiplied by decimation again.
        if self.ref_motion_lib is not None:
            self.ref_motion_lib.step()


    def _apply_action(self):
        # self.ref_motion_lib.step()
        # joint_pos = self.robot.data.default_joint_pos.clone()
        # ref_robot_joint_pos = self.ref_motion_lib.get_robot_joint_pos().to(joint_pos)
        # joint_pos[:, self._robot_dof_idx] = ref_robot_joint_pos[:, self.ref_robot_dof_idx]
        # self.robot.write_joint_position_to_sim(joint_pos)
        # self.robot_dof_targets[:] = joint_pos[:, self._robot_dof_idx]
        # self.applied_robot_dof_targets[:] = self.robot_dof_targets
        # door_pos = self.door.data.joint_pos.clone()
        # door_pos[:] = self.ref_motion_lib.get_door_joint_pos()
        # self.door.write_joint_position_to_sim(door_pos)

        edit_door_articulation(
            self.door,
            nominal_joint_stiffness=self._door_nominal_joint_stiffness,
            nominal_joint_damping=self._door_nominal_joint_damping,
        )
        lag_alpha = self._current_custom_param("pd_targets", "target_lag_alpha")
        applied_targets = self.robot_dof_targets.clone()
        if lag_alpha > 0.0:
            applied_targets = (1.0 - lag_alpha) * applied_targets + lag_alpha * self.applied_robot_dof_targets
        # Controller-side DR is applied on the position targets after action integration.
        applied_targets += self._get_policy_target_noise()
        self.applied_robot_dof_targets[:] = torch.clamp(
            applied_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.robot.set_joint_position_target(self.applied_robot_dof_targets, joint_ids=self._robot_dof_idx)

    def _build_observations(
        self,
        record_viser: bool = True,
    ) -> dict:
        self._get_intermediate_values()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        door_to_base_link_pos = world_to_local(self.door_link_pos, self.robot_base_body_pos, self.robot_base_body_quat).reshape(self.num_envs, 1, -1)
        door_twist_in_robot_base_frame = world_to_local(
            self.ref_door_body_pos_twist,
            self.robot_base_body_pos,
            self.robot_base_body_quat,
        ).reshape(self.num_envs, 1, -1)
        robot_key_body_pos, robot_key_body_euler, base_lin_vel_local, base_ang_vel_local = self.transform_key_bodies_to_base_frame(self.robot_key_body_pos, self.robot_key_body_quat, self.robot_body_lin_vel, self.robot_body_ang_vel, self._robot_base_body_link_idx)
        key_pos_err = (self.ref_robot_key_body_pos).to(self.robot_key_body_pos) - self.robot_key_body_pos
        key_pos_err = world_to_local(key_pos_err, None, self.robot_base_body_quat)
        key_pos_err = key_pos_err.reshape(self.num_envs, 1, -1)

        clean_joint_pos = self.joint_pos[:, self._robot_dof_idx]
        clean_joint_vel = self.joint_vel[:, self._robot_dof_idx]
        policy_joint_pos = clean_joint_pos.clone()
        policy_joint_vel = clean_joint_vel.clone()

        policy_joint_pos[:, self._target_base_rot_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_base_rot_slice], "base_rot_joint_pos_noise", "base_rot_joint_pos_bias"
        )
        policy_joint_pos[:, self._target_base_xy_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_base_xy_slice], "base_xy_joint_pos_noise", "base_xy_joint_pos_bias"
        )
        policy_joint_pos[:, self._target_arm_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_arm_slice], "arm_joint_pos_noise", "arm_joint_pos_bias"
        )
        policy_joint_pos[:, self._target_finger_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_finger_slice], "finger_joint_pos_noise", "finger_joint_pos_bias"
        )

        policy_joint_vel[:, self._target_base_rot_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_base_rot_slice], "base_rot_joint_vel_noise", "base_rot_joint_vel_bias"
        )
        policy_joint_vel[:, self._target_base_xy_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_base_xy_slice], "base_xy_joint_vel_noise", "base_xy_joint_vel_bias"
        )
        policy_joint_vel[:, self._target_arm_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_arm_slice], "arm_joint_vel_noise", "arm_joint_vel_bias"
        )
        policy_joint_vel[:, self._target_finger_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_finger_slice], "finger_joint_vel_noise", "finger_joint_vel_bias"
        )

        clean_base_joint_ref_err = (
            self.ref_robot_base_joint_pos
            - clean_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop]
        ).unsqueeze(dim=1)
        clean_arm_joint_ref_err = (self.ref_robot_arm_joint_pos - clean_joint_pos[:, self._target_arm_slice]).unsqueeze(
            dim=1
        )
        policy_base_joint_ref_err = (
            self.ref_robot_base_joint_pos
            - policy_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop]
        ).unsqueeze(dim=1)
        policy_arm_joint_ref_err = (self.ref_robot_arm_joint_pos - policy_joint_pos[:, self._target_arm_slice]).unsqueeze(
            dim=1
        )

        twist_obs = torch.cat(
            (
                self.ref_robot_key_body_pos_twist.reshape(self.num_envs, 1, -1),
                self.ref_robot_key_body_quat_twist.reshape(self.num_envs, 1, -1),
                self.ref_door_joint_pos_twist.reshape(self.num_envs, 1, -1),
                (
                    self.ref_robot_base_joint_pos_twist - clean_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop].unsqueeze(dim=1)
                ).reshape(self.num_envs, 1, -1),
                (
                    self.ref_robot_arm_joint_pos_twist - clean_joint_pos[:, self._target_arm_slice].unsqueeze(dim=1)
                ).reshape(self.num_envs, 1, -1),
                door_twist_in_robot_base_frame,
            ),
            dim=-1,
        )

        policy_obs = torch.cat(
            (
                policy_joint_pos.unsqueeze(dim=1),
                policy_joint_vel.unsqueeze(dim=1),
                self.robot_dof_targets.unsqueeze(dim = 1),
                key_pos_err,
                robot_key_body_pos.reshape(self.num_envs, 1, -1),
                robot_key_body_euler.reshape(self.num_envs, 1, -1),
                base_lin_vel_local.reshape(self.num_envs, 1, -1),
                base_ang_vel_local.reshape(self.num_envs, 1, -1),
                door_to_base_link_pos,
                self.door_joint_pos[:, self._door_joint_idx].unsqueeze(dim = 1),
                self.ref_door_joint_pos[:, self._door_joint_idx].to(self.door_joint_pos).unsqueeze(dim = 1),
            ),
            dim=-1,
        )

        critic_obs = torch.cat(
            (
                clean_joint_pos.unsqueeze(dim=1),
                clean_joint_vel.unsqueeze(dim=1),
                self.robot_dof_targets.unsqueeze(dim=1),
                key_pos_err,
                robot_key_body_pos.reshape(self.num_envs, 1, -1),
                robot_key_body_euler.reshape(self.num_envs, 1, -1),
                base_lin_vel_local.reshape(self.num_envs, 1, -1),
                base_ang_vel_local.reshape(self.num_envs, 1, -1),
                door_to_base_link_pos,
                self.door_joint_pos[:, self._door_joint_idx].unsqueeze(dim = 1),
                self.ref_door_joint_pos[:, self._door_joint_idx].to(self.door_joint_pos).unsqueeze(dim = 1),
            ),
            dim=-1,
        )

        # The actor sees noisy deployment-like inputs; the critic keeps the clean privileged state.
        policy_obs = policy_obs.squeeze(1)
        critic_obs = critic_obs.squeeze(1)

        if record_viser:
            self._maybe_record_viser_pointcloud()

        observations = {"policy": policy_obs, "critic": critic_obs}
        return observations

    def _get_observations(self) -> dict:
        return self._build_observations()


    def normalize_to_base_frame(
        self, 
        base_pos: torch.Tensor,       # (N, 3) Current base position
        base_quat: torch.Tensor,      # (N, 4) Current base orientation
        current_pos: torch.Tensor,    # (N, B, 3) Current key link positions
        current_quat: torch.Tensor,   # (N, B, 4) Current key link orientations
        target_pos: torch.Tensor,     # (N, T, B, 3) Future target positions (Twist)
        target_quat: torch.Tensor     # (N, T, B, 4) Future target orientations (Twist)
    ):
        """
        Calculates the positional and rotational errors between future targets 
        and current link states, expressed in the robot's base frame.
        """
        N, T, B, _ = target_pos.shape

        # ==========================================
        # 1. POSITION ERROR (Expressed in Base Frame)
        # ==========================================
        # World-space error: P_err_world = P_target - P_current
        curr_pos_exp = current_pos.unsqueeze(1)            # (N, 1, B, 3)
        pos_err_world = target_pos - curr_pos_exp          # (N, T, B, 3)

        # Rotate the world-space error vector into the base frame
        base_quat_inv = quat_conjugate(base_quat).view(N, 1, 1, 4)
        base_quat_inv_exp = base_quat_inv.expand(N, T, B, 4)

        pos_err_base = quat_apply(
            base_quat_inv_exp.reshape(-1, 4), 
            pos_err_world.reshape(-1, 3)
        )
        pos_err_base = pos_err_base.view(N, T, B, 3)

        # ==========================================
        # 2. ORIENTATION ERROR
        # ==========================================
        # The relative rotation required to get from current_quat to target_quat.
        # Q_err = Q_current_inv * Q_target
        curr_quat_inv = quat_conjugate(current_quat).unsqueeze(1) # (N, 1, B, 4)
        curr_quat_inv_exp = curr_quat_inv.expand(N, T, B, 4)

        quat_err = quat_mul(
            curr_quat_inv_exp.reshape(-1, 4),
            target_quat.reshape(-1, 4)
        )
        quat_err = quat_err.view(N, T, B, 4)

        return pos_err_base, quat_err

    def transform_key_bodies_to_base_frame(
        self,
        robot_key_body_pos: torch.Tensor,       # (num_envs, num_bodies, 3)
        robot_key_body_quat: torch.Tensor,      # (num_envs, num_bodies, 4)
        robot_body_lin_vel: torch.Tensor,       # (num_envs, num_bodies, 3)
        robot_body_ang_vel: torch.Tensor,       # (num_envs, num_bodies, 3)
        robot_base_id_in_key_body_idx: int,     # index of tidybot_base_link
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform robot key body observations from World Frame to Robot Base Frame.
        
        Args:
            robot_key_body_pos: World frame positions of key bodies
            robot_key_body_quat: World frame orientations (quaternions, xyzw)
            robot_body_lin_vel: World frame linear velocities
            robot_body_ang_vel: World frame angular velocities
            robot_base_id_in_key_body_idx: Index of base link in key body list
        
        Returns:
            Tuple containing:
                - body_pos_rel: (num_envs, num_bodies, 3) Positions in Base Frame
                - body_euler_rel: (num_envs, num_bodies, 3) Orientations in Base Frame
                - base_lin_vel_local: (num_envs, 3) Base linear velocity in Base Frame
                - base_ang_vel_local: (num_envs, 3) Base angular velocity in Base Frame
        """
        N = robot_key_body_pos.shape[0]
        B = robot_key_body_pos.shape[1]

        # ---------------------------------------------------
        # 1. Extract base pose
        # ---------------------------------------------------
        base_pos = robot_key_body_pos[:, robot_base_id_in_key_body_idx, :]        # (N,3)
        base_quat = robot_key_body_quat[:, robot_base_id_in_key_body_idx, :]      # (N,4)

        base_quat_inv = quat_conjugate(base_quat)                        # (N,4)

        # ---------------------------------------------------
        # 2. Position transform
        # P_local = R^T (P_world - P_base)
        # ---------------------------------------------------
        pos_diff = robot_key_body_pos - base_pos.unsqueeze(1)      # (N,B,3)

        base_quat_inv_exp = base_quat_inv.unsqueeze(1).expand(-1, B, -1)
        body_pos_rel = quat_apply(base_quat_inv_exp, pos_diff)

        # ---------------------------------------------------
        # 3. Orientation transform
        # Q_local = Q_base_inv * Q_body
        # ---------------------------------------------------
        body_quat_rel = quat_mul(base_quat_inv_exp, robot_key_body_quat)

        # ---------------------------------------------------
        # 4. Base velocities (express in base frame)
        # ---------------------------------------------------
        base_lin_vel_world = robot_body_lin_vel[:, robot_base_id_in_key_body_idx, :]
        base_ang_vel_world = robot_body_ang_vel[:, robot_base_id_in_key_body_idx, :]

        base_lin_vel_local = quat_apply(base_quat_inv, base_lin_vel_world)
        base_ang_vel_local = quat_apply(base_quat_inv, base_ang_vel_world)

        # ---------------------------------------------------
        # 5. Remove base link from outputs
        # ---------------------------------------------------
        keep_mask = torch.ones(B, dtype=torch.bool, device=robot_key_body_pos.device)
        keep_mask[robot_base_id_in_key_body_idx] = False

        body_pos_rel = body_pos_rel[:, keep_mask, :]
        body_quat_rel = body_quat_rel[:, keep_mask, :]

        return (
            body_pos_rel,          # (N, B-1, 3)
            quat_to_6d(body_quat_rel),         # (N, B-1, 6)
            base_lin_vel_local,    # (N, 3)
            base_ang_vel_local,    # (N, 3)
        )

    def compute_door_keypoints(
        self,
    ) -> torch.Tensor:
        """
        Compute 5 keypoints:
            - 2 from joint 0 (handle offsets)
            - 3 from joint 1 (board offsets)

        Returns:
            keypoints_w: [B, 5, 3]
        """

        # ---- Joint 0 (handle) ----
        pos0 = self.door_link_pos[:, self.door_body_names.index("link_2"), :]         # [B, 3]
        quat0 = self.door_link_quat[:, self.door_body_names.index("link_2"), :]       # [B, 4]

        # reshape for batched quat_apply
        handle_offsets_flat = self.handle_offsets.reshape(-1, 3)           # [B*2, 3]
        quat0_rep = quat0.unsqueeze(1).repeat(1, 2, 1).reshape(-1, 4) # [B*2, 4]

        handle_rot = quat_apply(quat0_rep.float(), handle_offsets_flat.float())       # [B*2, 3]
        handle_rot = handle_rot.reshape(self.num_envs, 2, 3)

        handle_kpts = pos0.unsqueeze(1) + handle_rot                  # [B, 2, 3]

        # ---- Joint 1 (board) ----
        pos1 = self.door_link_pos[:, self.door_body_names.index("link_1"), :]
        quat1 = self.door_link_quat[:, self.door_body_names.index("link_1"), :]

        board_offsets_flat = self.board_offsets.reshape(-1, 3)             # [B*3, 3]
        quat1_rep = quat1.unsqueeze(1).repeat(1, 3, 1).reshape(-1, 4) # [B*3, 4]

        board_rot = quat_apply(quat1_rep.float(), board_offsets_flat.float())         # [B*3, 3]
        board_rot = board_rot.reshape(self.num_envs, 3, 3)

        board_kpts = pos1.unsqueeze(1) + board_rot                    # [B, 3, 3]

        # ---- Concatenate ----
        keypoints_w = torch.cat([handle_kpts, board_kpts], dim=1)     # [self.num_envs, 5, 3]

        return keypoints_w

    def get_handle_position_in_base_frame(self) -> torch.Tensor:
        """Return the door handle center position in the robot base frame.

        The handle geometry lives on door body ``link_2``. We approximate the
        handle center using the midpoint of the two precomputed handle offsets,
        then transform that point into the robot base frame.

        Returns:
            torch.Tensor: handle_pos_base with shape (num_envs, 3).
        """
        robot_base_pos_w = self.robot.data.body_pos_w[:, self._robot_base_body_link_idx]
        robot_base_quat_w = self.robot.data.body_quat_w[:, self._robot_base_body_link_idx]

        handle_body_local_idx = self.door_body_names.index("link_2")
        handle_body_idx = int(self._door_body_idx[handle_body_local_idx])
        handle_body_pos_w = self.door.data.body_pos_w[:, handle_body_idx]
        handle_body_quat_w = self.door.data.body_quat_w[:, handle_body_idx]

        handle_center_offset = self.handle_offsets.mean(dim=1)
        handle_center_pos_w = quat_apply(handle_body_quat_w.float(), handle_center_offset.float()) + handle_body_pos_w
        handle_center_pos_base = world_to_local(
            handle_center_pos_w.unsqueeze(1),
            robot_base_pos_w,
            robot_base_quat_w,
        ).squeeze(1)

        return handle_center_pos_base

    def _set_current_state_as_reference(self):
        """Populate reference tensors without loading demonstration trajectories."""

        ref_joint_pos = torch.zeros(
            (self.num_envs, len(FULL_JOINT_NAMES)),
            device=self.device,
            dtype=self.robot.data.joint_pos.dtype,
        )
        ref_joint_vel = torch.zeros_like(ref_joint_pos)
        ref_joint_pos[:, self.ref_robot_dof_idx] = self.robot.data.joint_pos[:, self._robot_dof_idx]
        ref_joint_vel[:, self.ref_robot_dof_idx] = self.robot.data.joint_vel[:, self._robot_dof_idx]

        self.ref_robot_key_body_pos = self.robot_key_body_pos
        self.ref_robot_key_body_quat = self.robot_key_body_quat
        self.ref_robot_reset_key_body_pos = self.robot_reset_key_body_pos
        self.ref_robot_joint_pos = ref_joint_pos
        self.ref_robot_base_joint_pos = ref_joint_pos[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos = ref_joint_pos[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_pos = ref_joint_pos[:, self.ref_finger_joint_idx]
        self.ref_joint_vel = ref_joint_vel
        self.ref_robot_base_joint_vel = ref_joint_vel[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_vel = ref_joint_vel[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_vel = ref_joint_vel[:, self.ref_finger_joint_idx]
        self.ref_door_joint_pos = self.door_joint_pos.clone()
        self.ref_hinge_contact_mask = torch.zeros(self.num_envs, device=self.device, dtype=self.door_joint_pos.dtype)
        self.ref_robot_body_lin_vel = self.robot_body_lin_vel
        self.ref_robot_body_ang_vel = self.robot_body_ang_vel

        twist_len = len(self.twist_indices) if self.twist_indices is not None else 0
        twist_shape = (self.num_envs, twist_len)
        self.ref_robot_key_body_pos_twist = torch.zeros(
            (*twist_shape, len(self.ref_key_body_idx), 3),
            device=self.device,
            dtype=self.robot_key_body_pos.dtype,
        )
        self.ref_robot_key_body_quat_twist = torch.zeros(
            (*twist_shape, len(self.ref_key_body_idx), 6),
            device=self.device,
            dtype=self.robot_key_body_quat.dtype,
        )
        self.ref_robot_joint_pos_twist = ref_joint_pos.unsqueeze(1).expand(-1, twist_len, -1)
        self.ref_robot_base_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_arm_joint_idx]
        self.ref_door_joint_pos_twist = self.ref_door_joint_pos.unsqueeze(1).expand(-1, twist_len, -1)
        self.ref_door_body_pos_twist = torch.zeros(
            (*twist_shape, 3),
            device=self.device,
            dtype=self.door_link_pos.dtype,
        )

    def _get_intermediate_values(self):
        self.robot_key_body_pos = self.robot.data.body_pos_w[:, self._robot_key_body_idx]\
             - self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)
        self.robot_key_body_quat = self.robot.data.body_quat_w[:, self._robot_key_body_idx]

        self.robot_base_body_pos = self.robot_key_body_pos[:, self._robot_base_id_in_key_body_idx]
        self.robot_base_body_quat = self.robot_key_body_quat[:, self._robot_base_id_in_key_body_idx]

        self.robot_palm_body_pos = self.robot_key_body_pos[:, self._robot_palm_id_in_key_body_idx]
        self.robot_palm_body_quat = self.robot_key_body_quat[:, self._robot_palm_id_in_key_body_idx]

        self.robot_reset_key_body_pos = self.robot.data.body_pos_w[:, self._robot_reset_key_body_idx]\
             - self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)

        self.robot_base_joint_pos = self.robot.data.joint_pos[:, self._robot_base_dof_idx]
        self.robot_arm_joint_pos = self.robot.data.joint_pos[:, self._robot_arm_dof_idx]
        self.robot_finger_joint_pos = self.robot.data.joint_pos[:, self._robot_finger_dof_idx]
        self.door_joint_pos = self.door.data.joint_pos
        # self.door_joint_vel = self.door.data.joint_vel
        self.robot_base_joint_vel = self.robot.data.joint_vel[:, self._robot_base_dof_idx]
        self.robot_arm_joint_vel = self.robot.data.joint_vel[:, self._robot_arm_dof_idx]
        self.robot_finger_joint_vel = self.robot.data.joint_vel[:, self._robot_finger_dof_idx]

        self.door_base_link_pos = self.door.data.body_pos_w[:, self._door_base_link_idx]
        self.door_base_link_quat = self.door.data.body_quat_w[:, self._door_base_link_idx]
        self.door_link_pos = self.door.data.body_pos_w[:, self._door_body_idx]
        self.door_link_pos -= self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)
        self.door_link_quat = self.door.data.body_quat_w[:, self._door_body_idx]
        # self.door_keypoints = self.compute_door_keypoints()
        self.robot_body_lin_vel = self.robot.data.body_link_lin_vel_w[:, self._robot_key_body_idx]
        self.robot_body_ang_vel = self.robot.data.body_link_ang_vel_w[:, self._robot_key_body_idx]

        if self.ref_motion_lib is None:
            self._set_current_state_as_reference()
            return

        self.ref_robot_key_body_pos_twist = self.ref_motion_lib.get_robot_body_pos_twist()[:, :, self.ref_key_body_idx]
        # It is a misnomer, we are actually sending euler angles as it might be more friendly to MLP
        self.ref_robot_key_body_quat_twist = self.ref_motion_lib.get_robot_body_quat_twist()[:, :, self.ref_key_body_idx]
        self.ref_robot_key_body_pos_twist, self.ref_robot_key_body_quat_twist = self.normalize_to_base_frame(
            self.robot_base_body_pos.squeeze(),
            self.robot_base_body_quat.squeeze(),
            self.robot_key_body_pos, 
            self.robot_key_body_quat, 
            self.ref_robot_key_body_pos_twist, 
            self.ref_robot_key_body_quat_twist
        )
        # self.ref_robot_key_body_pos_twist, self.ref_robot_key_body_quat_twist = self.normalize_to_base_frame(self.robot_base_body_pos, self.robot_base_body_quat, self.ref_robot_key_body_pos_twist, self.ref_robot_key_body_quat_twist)
        self.ref_robot_key_body_quat_twist = quat_to_6d(self.ref_robot_key_body_quat_twist)
        self.ref_robot_joint_pos_twist = self.ref_motion_lib.get_robot_joint_pos_twist()
        self.ref_robot_base_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_arm_joint_idx]
        self.ref_door_joint_pos_twist = self.ref_motion_lib.get_door_joint_pos_twist()

        # self.ref_robot_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_key_body_idx]
        # self.ref_robot_key_body_quat = self.ref_motion_lib.get_robot_body_quat()[:, self._robot_key_body_idx]
        # self.ref_robot_reset_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_reset_key_body_idx]
        ref_robot_body_pos = self.ref_motion_lib.get_robot_body_pos()
        self.ref_robot_key_body_pos = ref_robot_body_pos[:, self.ref_key_body_idx]
        self.ref_robot_key_body_quat = self.ref_motion_lib.get_robot_body_quat()[:, self.ref_key_body_idx]
        self.ref_robot_reset_key_body_pos = ref_robot_body_pos[:, self.ref_reset_key_body_idx]
        self.ref_robot_joint_pos = self.ref_motion_lib.get_robot_joint_pos()
        self.ref_robot_base_joint_pos = self.ref_robot_joint_pos[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos = self.ref_robot_joint_pos[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_pos = self.ref_robot_joint_pos[:, self.ref_finger_joint_idx]
        self.ref_joint_vel = self.ref_motion_lib.get_robot_joint_vel()
        self.ref_robot_base_joint_vel = self.ref_joint_vel[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_vel = self.ref_joint_vel[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_vel = self.ref_joint_vel[:, self.ref_finger_joint_idx]
        self.ref_door_joint_pos = self.ref_motion_lib.get_door_joint_pos()
        self.ref_hinge_contact_mask = self.ref_motion_lib.get_hinge_contact_mask()
        self.ref_door_body_pos_twist = self.ref_motion_lib.get_door_body_pos_twist()
        ref_motion_dt = max(float(self.ref_motion_lib.frame_dt), 1e-6)
        self.ref_robot_body_lin_vel = self.ref_motion_lib.get_robot_body_lin_vel()[:, self.ref_key_body_idx] / ref_motion_dt
        self.ref_robot_body_ang_vel = self.ref_motion_lib.get_robot_body_ang_vel()[:, self.ref_key_body_idx] / ref_motion_dt

    def _get_rewards(self) -> torch.Tensor:
        self._get_intermediate_values()
        self._log_dr_metrics()

        # key_body_pos_err, key_body_quat_err, door_err, root_pos_err, root_rot_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err, door_pos_err = compute_tracking_error(
        key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err = compute_tracking_error(
            robot_key_body_pos = self.robot_key_body_pos,
            robot_key_body_quat = self.robot_key_body_quat,
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos,
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,

            ref_robot_key_body_pos = self.ref_robot_key_body_pos,
            ref_robot_key_body_quat = self.ref_robot_key_body_quat,
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
            ref_robot_base_joint_vel = self.ref_robot_base_joint_vel,
            ref_robot_arm_joint_vel = self.ref_robot_arm_joint_vel,
            ref_robot_finger_joint_vel = self.ref_robot_finger_joint_vel,
        )

        self.extras["error/key_body_pos_err"] = math.sqrt(max(key_body_pos_err.reshape(self.num_envs, -1).mean().item(), 0.0))
        self.extras["error/key_body_quat_err"] = math.sqrt(max(key_body_quat_err.reshape(self.num_envs, -1).mean().item(), 0.0))
        self.extras["error/door_err"] = math.sqrt(max(door_err.reshape(self.num_envs, -1).mean().item(), 0.0))
        self.extras["error/base_joint_pos_err"] = math.sqrt(
            max(base_joint_pos_err.reshape(self.num_envs, -1).mean().item() / len(self.cfg.base_joints), 0.0)
        )
        self.extras["error/arm_joint_pos_err"] = math.sqrt(
            max(arm_joint_pos_err.reshape(self.num_envs, -1).mean().item() / len(self.cfg.arm_joints), 0.0)
        )
        self.extras["error/finger_joint_pos_err"] = math.sqrt(
            max(finger_joint_pos_err.reshape(self.num_envs, -1).mean().item() / len(self.cfg.finger_joints), 0.0)
        )
        # self.extras["error/base_joint_vel_err"] = base_joint_vel_err.mean()
        # self.extras["error/arm_joint_vel_err"] = arm_joint_vel_err.mean()
        # self.extras["error/finger_joint_vel_err"] = finger_joint_vel_err.mean()

        if self.prob_get_first_key_frame is not None:
            self.extras["reset/prob_get_first_key_frame"] = float(self.prob_get_first_key_frame)

        # Use filtered handle-hand force. Do not use net_forces_w here because
        # net_forces_w includes all contacts acting on Door/link_2.
        contact_forces_door2 = self._get_filtered_contact_force_w(
            self.scene.sensors["contact_forces_door2"],
            expected_num_envs=self.num_envs,
        )
        handle_force_norm = torch.linalg.vector_norm(contact_forces_door2, dim=-1)
        self.extras["stats/filtered_handle_force_norm_mean"] = float(handle_force_norm.mean().detach().cpu().item())
        self.extras["stats/filtered_handle_force_norm_max"] = float(handle_force_norm.max().detach().cpu().item())
        self.extras["stats/filtered_handle_contact_frac"] = float(
            (handle_force_norm > self.cfg.handle_contact_force_threshold).float().mean().detach().cpu().item()
        )

        deep_mimic_reward = compute_deep_mimic_rewards(
            robot_key_body_pos = self.robot_key_body_pos, 
            robot_key_body_quat = self.robot_key_body_quat, 
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos, 
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,
            robot_body_lin_vel = self.robot_body_lin_vel,
            robot_body_ang_vel = self.robot_body_ang_vel,

            ref_robot_key_body_pos = self.ref_robot_key_body_pos, 
            ref_robot_key_body_quat = self.ref_robot_key_body_quat, 
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
            ref_robot_base_joint_vel = self.ref_robot_base_joint_vel,
            ref_robot_arm_joint_vel = self.ref_robot_arm_joint_vel,
            ref_robot_finger_joint_vel = self.ref_robot_finger_joint_vel,
            ref_robot_body_lin_vel = self.ref_robot_body_lin_vel,
            ref_robot_body_ang_vel = self.ref_robot_body_ang_vel,

            robot_key_body_pos_scale = self.robot_key_body_pos_scale, 
            robot_key_body_quat_scale = self.robot_body_quat_scale,
            door_joint_pos_scale = self.door_joint_pos_scale,
            robot_base_joint_pos_scale = self.robot_base_joint_pos_scale,
            robot_arm_joint_pos_scale = self.robot_arm_joint_pos_scale,
            robot_finger_joint_pos_scale = self.robot_finger_joint_pos_scale,
            robot_base_joint_vel_scale = self.robot_base_joint_vel_scale,
            robot_arm_joint_vel_scale = self.robot_arm_joint_vel_scale,
            robot_finger_joint_vel_scale = self.robot_finger_joint_vel_scale,
            robot_body_lin_vel_scale = self.robot_body_lin_vel_scale,
            robot_body_ang_vel_scale = self.robot_body_ang_vel_scale,

            robot_key_body_pos_w = self.robot_key_body_pos_w, 
            robot_key_body_quat_w = self.robot_body_quat_w,
            door_joint_pos_w = self.door_joint_pos_w,
            robot_base_joint_pos_w = self.robot_base_joint_pos_w,
            robot_arm_joint_pos_w = self.robot_arm_joint_pos_w,
            robot_finger_joint_pos_w = self.robot_finger_joint_pos_w,
            robot_base_joint_vel_w = self.robot_base_joint_vel_w,
            robot_arm_joint_vel_w = self.robot_arm_joint_vel_w,
            robot_finger_joint_vel_w = self.robot_finger_joint_vel_w,
            robot_body_lin_vel_w = self.robot_body_lin_vel_w,
            robot_body_ang_vel_w = self.robot_body_ang_vel_w,

            contact_forces = contact_forces_door2,
            contact_force_w = self.hinge_contact_reward_w * self.ref_hinge_contact_mask,
            contact_force_threshold = self.cfg.handle_contact_force_threshold,
        )

        joint_limit_penalty, joint_limit_active_fraction = compute_joint_limit_penalty(
            joint_pos=self.robot.data.joint_pos[:, self._robot_dof_idx],
            joint_lower_limits=self.robot_dof_lower_limits,
            joint_upper_limits=self.robot_dof_upper_limits,
            soft_ratio=self.joint_limit_penalty_margin_ratio,
        )
        weighted_joint_limit_penalty = self.joint_limit_penalty_w * joint_limit_penalty
        self.extras["error/joint_limit_penalty"] = weighted_joint_limit_penalty.reshape(self.num_envs, -1).mean().item()
        self.extras["stats/joint_limit_active_fraction"] = (
            joint_limit_active_fraction.reshape(self.num_envs, -1).mean().item()
        )
        self._update_success_metrics()

        # 1. Base alive reward: small constant for remaining active
        alive_base = self.alive_base 
        
        # 2. Difficulty Bonus: Extra points for staying alive during contact
        # self.ref_hinge_contact_mask is 1.0 when grasping/pulling
        alive_bonus = self.alive_bonus * self.ref_hinge_contact_mask.squeeze()
        
        total_alive_reward = alive_base + alive_bonus

        # 3. Combine with tracking reward and termination penalty
        is_killed = self.reset_terminated
        termination_penalty = self.termination_penalty
        
        final_reward = deep_mimic_reward + total_alive_reward - weighted_joint_limit_penalty
        final_reward = torch.where(is_killed, final_reward + termination_penalty, final_reward)

        return final_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        reached_last_frame = self._get_reached_last_frame_mask()
        time_out = (self.episode_length_buf >= self.max_trial_steps - 1) | reached_last_frame
        if not self.early_stopping:
            return torch.zeros_like(time_out), time_out
        self._get_intermediate_values()
        progress = min(self._get_curriculum_step_count() / self.reset_progress_total, 1.0)
        reset_key_body_pos_delta = self.reset_key_body_pos_delta_min + (self.reset_key_body_pos_delta_max - self.reset_key_body_pos_delta_min) * progress
        reset_key_body_quat_delta = self.reset_key_body_quat_delta_min + (self.reset_key_body_quat_delta_max - self.reset_key_body_quat_delta_min) * progress
        reset_door_joint_pos_delta = self.reset_door_joint_pos_delta_min + (self.reset_door_joint_pos_delta_max - self.reset_door_joint_pos_delta_min) * progress
        self.extras["reset/reset_key_body_pos_delta"] = reset_key_body_pos_delta
        self.extras["reset/reset_key_body_quat_delta"] = reset_key_body_quat_delta
        self.extras["reset/reset_door_joint_pos_delta"] = reset_door_joint_pos_delta
        # reset_key_body_pos_delta = reset_key_body_pos_delta ** 2 * len(self.cfg.robot_reset_key_bodies)
        # reset_key_body_quat_delta = reset_key_body_quat_delta ** 2 * len(self.cfg.robot_reset_key_bodies)
        # reset_door_joint_pos_delta = reset_door_joint_pos_delta ** 2 * len(self.cfg.door_joint_names)
        reset_key_body_pos_delta = reset_key_body_pos_delta ** 2
        reset_key_body_quat_delta = reset_key_body_quat_delta ** 2
        reset_door_joint_pos_delta = reset_door_joint_pos_delta ** 2
        # key_body_pos_err, key_body_quat_err, door_err, root_pos_err, root_rot_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err, door_pos_err = compute_tracking_error(
        key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err = compute_tracking_error(
            robot_key_body_pos = self.robot_reset_key_body_pos,
            robot_key_body_quat = self.robot_key_body_quat,
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos,
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,

            ref_robot_key_body_pos = self.ref_robot_reset_key_body_pos,
            ref_robot_key_body_quat = self.ref_robot_key_body_quat,
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
            ref_robot_base_joint_vel = self.ref_robot_base_joint_vel,
            ref_robot_arm_joint_vel = self.ref_robot_arm_joint_vel,
            ref_robot_finger_joint_vel = self.ref_robot_finger_joint_vel,
        )
        return \
            (key_body_pos_err > reset_key_body_pos_delta) | \
            (key_body_quat_err > reset_key_body_quat_delta) | \
            (door_err > reset_door_joint_pos_delta), \
            time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._update_adr_ranges()
        if not self.use_motion_ref:
            self.prob_get_first_key_frame = None
            default_trial_steps = max(1, math.ceil(float(self.cfg.episode_length_s) / max(float(self.dt), 1e-6)))
            self.max_trial_steps[env_ids] = default_trial_steps
            self._sample_reset_randomization(env_ids)

            default_root_state = self.robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self.scene.env_origins[env_ids]

            self.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
            self.joint_vel[env_ids] = self.robot.data.default_joint_vel[env_ids]
            self._apply_spawn_noise(env_ids)

            self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], None, env_ids)
            self.robot.set_joint_position_target(self.joint_pos[env_ids], env_ids=env_ids)

            door_joint_pos = self.door.data.default_joint_pos[env_ids].clone()
            door_joint_vel = self.door.data.default_joint_vel[env_ids].clone()
            self.door.write_joint_state_to_sim(door_joint_pos, door_joint_vel, None, env_ids)

            self.robot_dof_targets[env_ids, :] = self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]]
            self.applied_robot_dof_targets[env_ids, :] = self.robot_dof_targets[env_ids, :]
            self.episode_reached_last_frame[env_ids] = False
            super()._reset_idx(env_ids)
            self._refresh_nominal_door_joint_gains(env_ids)
            return

        reset_frame_idx, self.prob_get_first_key_frame = self.ref_motion_lib.reset(
            env_ids,
            step_count=self._get_curriculum_step_count(),
            reset_progress_total=self.reset_progress_total,
        )
        remaining_frames = torch.clamp(
            float(self.ref_motion_lib.num_frames - 1) - reset_frame_idx.to(dtype=torch.float32),
            min=0.0,
        )
        required_steps = torch.ceil(remaining_frames / max(float(self.ref_motion_lib.frame_step), 1e-6)).long() + 1
        self.max_trial_steps[env_ids] = torch.clamp(required_steps, min=1)
        self._sample_reset_randomization(env_ids)

        deep_mimic_initial_joint_pos = self.ref_motion_lib.get_robot_joint_pos(env_ids)
        deep_mimic_initial_joint_vel = torch.zeros_like(deep_mimic_initial_joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.default_joint_vel[env_ids]
        self.joint_vel[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_vel.to(self.joint_vel)[..., self.ref_robot_dof_idx]
        self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_pos.to(self.joint_pos)[..., self.ref_robot_dof_idx]
        self._apply_spawn_noise(env_ids)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], None, env_ids)
        self.robot.set_joint_position_target(self.joint_pos[env_ids], env_ids=env_ids)

        door_joint_pos = self.ref_motion_lib.get_door_joint_pos(env_ids).to(self.door.data.joint_pos)
        door_joint_vel = self.door.data.default_joint_vel[env_ids].clone()

        self.door.write_joint_state_to_sim(door_joint_pos, door_joint_vel, None, env_ids)
        # self.door.set_joint_position_target(door_joint_pos, None, env_ids)

        # self.last_actions[env_ids] = 0.0
        self.robot_dof_targets[env_ids, :] = self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]]
        self.applied_robot_dof_targets[env_ids, :] = self.robot_dof_targets[env_ids, :]
        self.episode_reached_last_frame[env_ids] = False
        super()._reset_idx(env_ids)
        self._refresh_nominal_door_joint_gains(env_ids)

    def close(self):
        if getattr(self, "viser_pointcloud_enabled", False):
            self._flush_viser_pointcloud_recording(chunk_complete=False, reason="env close")
        return super().close()

@torch.jit.script
def compute_deep_mimic_rewards(
    robot_key_body_pos: torch.Tensor,
    robot_key_body_quat: torch.Tensor,
    door_joint_pos: torch.Tensor,
    robot_base_joint_pos: torch.Tensor,
    robot_arm_joint_pos: torch.Tensor,
    robot_finger_joint_pos: torch.Tensor,
    robot_base_joint_vel: torch.Tensor,
    robot_arm_joint_vel: torch.Tensor,
    robot_finger_joint_vel: torch.Tensor,
    robot_body_lin_vel: torch.Tensor,
    robot_body_ang_vel: torch.Tensor,

    ref_robot_key_body_pos: torch.Tensor,
    ref_robot_key_body_quat: torch.Tensor,
    ref_door_joint_pos: torch.Tensor,
    ref_robot_base_joint_pos: torch.Tensor,
    ref_robot_arm_joint_pos: torch.Tensor,
    ref_robot_finger_joint_pos: torch.Tensor,
    ref_robot_base_joint_vel: torch.Tensor,
    ref_robot_arm_joint_vel: torch.Tensor,
    ref_robot_finger_joint_vel: torch.Tensor,
    ref_robot_body_lin_vel: torch.Tensor,
    ref_robot_body_ang_vel: torch.Tensor,

    robot_key_body_pos_scale: float,
    robot_key_body_quat_scale: float,
    door_joint_pos_scale: float,
    robot_base_joint_pos_scale: float,
    robot_arm_joint_pos_scale: float,
    robot_finger_joint_pos_scale: float,
    robot_base_joint_vel_scale: float,
    robot_arm_joint_vel_scale: float,
    robot_finger_joint_vel_scale: float,
    robot_body_lin_vel_scale: float,
    robot_body_ang_vel_scale: float,

    robot_key_body_pos_w: float,
    robot_key_body_quat_w: float,
    door_joint_pos_w: float,
    robot_base_joint_pos_w: float,
    robot_arm_joint_pos_w: float,
    robot_finger_joint_pos_w: float,
    robot_base_joint_vel_w: float,
    robot_arm_joint_vel_w: float,
    robot_finger_joint_vel_w: float,
    robot_body_lin_vel_w: float,
    robot_body_ang_vel_w: float,

    contact_force_w: torch.Tensor,
    contact_forces: torch.Tensor,
    contact_force_threshold: float,
) -> torch.Tensor:
    # ----------------------------------
    # Robot body position error
    # ----------------------------------
    # [B, N, 3]
    key_body_pos_diff = ref_robot_key_body_pos - robot_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # [B, N]
    key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)

    if robot_body_lin_vel_w != 0:
        robot_body_lin_vel_diff = ref_robot_body_lin_vel - robot_body_lin_vel
        robot_body_lin_vel_err = torch.sum(robot_body_lin_vel_diff * robot_body_lin_vel_diff, dim=-1)  # [B, N]
        robot_body_lin_vel_err = torch.sum(robot_body_lin_vel_err, dim=-1)
    else:
        robot_body_lin_vel_err = None
    if robot_body_ang_vel_w != 0:
        robot_body_ang_vel_diff = ref_robot_body_ang_vel - robot_body_ang_vel
        robot_body_ang_vel_err = torch.sum(robot_body_ang_vel_diff * robot_body_ang_vel_diff, dim=-1)  # [B, N]
        robot_body_ang_vel_err = torch.sum(robot_body_ang_vel_err, dim=-1)
    else:
        robot_body_ang_vel_err = None
    # ----------------------------------
    # Robot body orientation error
    # ----------------------------------
    # [B, N]
    key_body_quat_diff = quat_diff_angle(robot_key_body_quat, ref_robot_key_body_quat)
    key_body_quat_err = torch.sum(key_body_quat_diff * key_body_quat_diff, dim=-1)  # [B]
    # ----------------------------------
    # Door joint error
    # ----------------------------------
    door_diff = hinge_angle_diff(ref_door_joint_pos, door_joint_pos)
    door_err = torch.sum(door_diff * door_diff, dim=-1)  # [B]
    # ----------------------------------
    # Robot joint position error
    # ----------------------------------
    base_joint_pos_diff = ref_robot_base_joint_pos - robot_base_joint_pos
    base_joint_pos_err = torch.sum(base_joint_pos_diff * base_joint_pos_diff, dim=-1)  # [B]
    arm_joint_pos_diff = hinge_angle_diff(ref_robot_arm_joint_pos, robot_arm_joint_pos)
    arm_joint_pos_err = torch.sum(arm_joint_pos_diff * arm_joint_pos_diff, dim=-1)  # [B]
    finger_joint_pos_diff = hinge_angle_diff(ref_robot_finger_joint_pos, robot_finger_joint_pos)
    finger_joint_pos_err = torch.sum(finger_joint_pos_diff * finger_joint_pos_diff, dim=-1)  # [B]

    base_joint_vel_diff = ref_robot_base_joint_vel - robot_base_joint_vel
    base_joint_vel_err = torch.sum(base_joint_vel_diff * base_joint_vel_diff, dim=-1)  # [B]
    arm_joint_vel_diff = ref_robot_arm_joint_vel - robot_arm_joint_vel
    arm_joint_vel_err = torch.sum(arm_joint_vel_diff * arm_joint_vel_diff, dim=-1)  # [B]
    finger_joint_vel_diff = ref_robot_finger_joint_vel - robot_finger_joint_vel
    finger_joint_vel_err = torch.sum(finger_joint_vel_diff * finger_joint_vel_diff, dim=-1)  # [B]

    # ----------------------------------
    # Exponential rewards (DeepMimic style)
    # ----------------------------------
    key_body_pos_r = torch.exp(-robot_key_body_pos_scale * key_body_pos_err)
    key_body_quat_r = torch.exp(-robot_key_body_quat_scale * key_body_quat_err)
    door_r = torch.exp(-door_joint_pos_scale * door_err)
    base_joint_pos_r = torch.exp(-robot_base_joint_pos_scale * base_joint_pos_err)
    arm_joint_pos_r = torch.exp(-robot_arm_joint_pos_scale * arm_joint_pos_err)
    finger_joint_pos_r = torch.exp(-robot_finger_joint_pos_scale * finger_joint_pos_err)
    base_joint_vel_r = torch.exp(-robot_base_joint_vel_scale * base_joint_vel_err)
    arm_joint_vel_r = torch.exp(-robot_arm_joint_vel_scale * arm_joint_vel_err)
    finger_joint_vel_r = torch.exp(-robot_finger_joint_vel_scale * finger_joint_vel_err)
    if robot_body_lin_vel_err is not None:
        robot_body_lin_vel_r = torch.exp(-robot_body_lin_vel_scale * robot_body_lin_vel_err)
    else:
        robot_body_lin_vel_r = torch.zeros_like(key_body_pos_r)
    if robot_body_ang_vel_err is not None:
        robot_body_ang_vel_r = torch.exp(-robot_body_ang_vel_scale * robot_body_ang_vel_err)
    else:
        robot_body_ang_vel_r = torch.zeros_like(key_body_pos_r)

    if contact_forces.dim() != 2 or contact_forces.size(-1) != 3:
        raise RuntimeError("Expected filtered handle contact force shape [N, 3].")
    contact_force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
    contact_reward = (contact_force_norm > contact_force_threshold).to(dtype=key_body_pos_r.dtype)
    # ----------------------------------
    # Final reward
    # ----------------------------------
    reward = robot_key_body_pos_w * key_body_pos_r\
         + robot_key_body_quat_w * key_body_quat_r\
         + door_joint_pos_w * door_r\
         + robot_base_joint_pos_w * base_joint_pos_r\
         + robot_arm_joint_pos_w * arm_joint_pos_r\
         + robot_finger_joint_pos_w * finger_joint_pos_r\
         + robot_base_joint_vel_w * base_joint_vel_r\
         + robot_arm_joint_vel_w * arm_joint_vel_r\
         + robot_finger_joint_vel_w * finger_joint_vel_r\
         + robot_body_lin_vel_w * robot_body_lin_vel_r\
         + robot_body_ang_vel_w * robot_body_ang_vel_r\
         + contact_force_w * contact_reward
    return reward

def compute_tracking_error(
    robot_key_body_pos: torch.Tensor,
    robot_key_body_quat: torch.Tensor,
    door_joint_pos: torch.Tensor,
    robot_base_joint_pos: torch.Tensor,
    robot_arm_joint_pos: torch.Tensor,
    robot_finger_joint_pos: torch.Tensor,
    robot_base_joint_vel: torch.Tensor,
    robot_arm_joint_vel: torch.Tensor,
    robot_finger_joint_vel: torch.Tensor,

    ref_robot_key_body_pos: torch.Tensor,
    ref_robot_key_body_quat: torch.Tensor,
    ref_door_joint_pos: torch.Tensor,
    ref_robot_base_joint_pos: torch.Tensor,
    ref_robot_arm_joint_pos: torch.Tensor,
    ref_robot_finger_joint_pos: torch.Tensor,
    ref_robot_base_joint_vel: torch.Tensor,
    ref_robot_arm_joint_vel: torch.Tensor,
    ref_robot_finger_joint_vel: torch.Tensor,
) -> torch.Tensor:
    # ----------------------------------
    # Robot body position error
    # ----------------------------------
    # [B, N, 3]
    key_body_pos_diff = ref_robot_key_body_pos - robot_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # [B, N]
    key_body_pos_err = torch.max(key_body_pos_err, dim=-1).values
    # ----------------------------------
    # Robot body orientation error
    # ----------------------------------
    # [B, N]
    key_body_quat_diff = quat_diff_angle(robot_key_body_quat, ref_robot_key_body_quat)
    # key_body_quat_err = torch.sum(key_body_quat_diff * key_body_quat_diff, dim=-1)  # [B]
    key_body_quat_err = torch.max(key_body_quat_diff * key_body_quat_diff, dim=-1).values
    # ----------------------------------
    # Door joint error
    # ----------------------------------
    door_diff = ref_door_joint_pos - door_joint_pos
    # door_err = torch.sum(door_diff * door_diff, dim=-1)  # [B]
    door_err = torch.max(door_diff * door_diff, dim=-1).values
    # ----------------------------------
    # Robot joint position error
    # ----------------------------------
    base_joint_pos_diff = ref_robot_base_joint_pos - robot_base_joint_pos
    base_joint_pos_err = torch.sum(base_joint_pos_diff * base_joint_pos_diff, dim=-1)  # [B]
    # root_pos_diff = ref_robot_base_joint_pos[:, :2] - robot_base_joint_pos[:, :2]
    # root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)  # [B]
    # root_rot_diff = hinge_angle_diff(ref_robot_base_joint_pos[:, 2:], robot_base_joint_pos[:, 2:])
    # root_rot_err = torch.sum(root_rot_diff * root_rot_diff, dim=-1)  # [B]
    arm_joint_pos_diff = ref_robot_arm_joint_pos - robot_arm_joint_pos
    arm_joint_pos_err = torch.sum(arm_joint_pos_diff * arm_joint_pos_diff, dim=-1)  # [B]
    finger_joint_pos_diff = ref_robot_finger_joint_pos - robot_finger_joint_pos
    finger_joint_pos_err = torch.sum(finger_joint_pos_diff * finger_joint_pos_diff, dim=-1)  # [B]
    base_joint_vel_diff = ref_robot_base_joint_vel - robot_base_joint_vel
    base_joint_vel_err = torch.sum(base_joint_vel_diff * base_joint_vel_diff, dim=-1)  # [B]
    arm_joint_vel_diff = ref_robot_arm_joint_vel - robot_arm_joint_vel
    arm_joint_vel_err = torch.sum(arm_joint_vel_diff * arm_joint_vel_diff, dim=-1)  # [B]
    finger_joint_vel_diff = ref_robot_finger_joint_vel - robot_finger_joint_vel
    finger_joint_vel_err = torch.sum(finger_joint_vel_diff * finger_joint_vel_diff, dim=-1)  # [B]
    return (key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err)

def compute_joint_limit_penalty(
    joint_pos: torch.Tensor,
    joint_lower_limits: torch.Tensor,
    joint_upper_limits: torch.Tensor,
    soft_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
    if soft_ratio <= 0.0:
        return zeros, zeros

    finite_limit_mask = torch.isfinite(joint_lower_limits) & torch.isfinite(joint_upper_limits)
    valid_limit_mask = finite_limit_mask & (joint_upper_limits > joint_lower_limits)

    joint_range = torch.where(valid_limit_mask, joint_upper_limits - joint_lower_limits, torch.ones_like(joint_lower_limits))
    soft_zone = torch.clamp(joint_range * soft_ratio, min=1e-6)

    dist_to_lower = joint_pos - joint_lower_limits.unsqueeze(0)
    dist_to_upper = joint_upper_limits.unsqueeze(0) - joint_pos
    dist_to_limit = torch.minimum(dist_to_lower, dist_to_upper)

    normalized_penalty = torch.clamp(
        (soft_zone.unsqueeze(0) - dist_to_limit) / soft_zone.unsqueeze(0),
        min=0.0,
        max=1.0,
    )
    valid_limit_mask = valid_limit_mask.unsqueeze(0)
    normalized_penalty = torch.where(valid_limit_mask, normalized_penalty, torch.zeros_like(normalized_penalty))

    num_tracked_joints = valid_limit_mask.to(joint_pos.dtype).sum(dim=-1).clamp(min=1.0)
    joint_limit_penalty = normalized_penalty.square().sum(dim=-1) / num_tracked_joints
    active_joint_fraction = (normalized_penalty > 0.0).to(joint_pos.dtype).sum(dim=-1) / num_tracked_joints
    return joint_limit_penalty, active_joint_fraction
