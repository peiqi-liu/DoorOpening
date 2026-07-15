import json
import os
import math
import pathlib
import re
import time
from collections import OrderedDict, deque
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except ImportError:
    wandb = None

from isaaclab.utils.math import quat_apply, quat_mul
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES
from DoorOpening.assets.door.multi_door_cfg import asset_family_ids as door_asset_family_ids
from DoorOpening.assets.door.multi_door_cfg import asset_paths as door_asset_paths
from DoorOpening.assets.door.multi_door_cfg import board_bboxes as door_board_bboxes
from DoorOpening.assets.door.multi_door_cfg import board_bboxes_link1 as door_board_bboxes_link1
from DoorOpening.assets.door.multi_door_cfg import door_full_bboxes as door_full_door_bboxes
from DoorOpening.assets.door.multi_door_cfg import motion_family_ids, motion_traj_paths
from DoorOpening.assets.glorbot.glorbot_cfg import glorbot_urdf_path
from DoorOpening.model.transformer import PCDTransformer, strip_prefix_from_state_dict
from DoorOpening.utils.camera_utils import (
    build_pinhole_intrinsics,
    crop_local_pcd,
    simulate_depth_cam_render_from_pose,
    simulate_lidar_render_from_pose,
)
from DoorOpening.utils.door_window_dropout import (
    apply_window_dropout_to_door_points,
    sample_random_window_hole_metadata,
)
from DoorOpening.utils.extract_pointcloud_from_articulation import (
    FrankaLeapSampler,
    build_first_visual_link_pointcloud_cache,
    compose_cached_link_pointcloud_world,
)
from DoorOpening.utils.glorbot_collision_checker import GlorbotCollisionChecker
from DoorOpening.utils.pose_utils import world_to_local
from DoorOpening.utils.wall_distractors import (
    WallDistractorParams,
    compute_wall_bbox_ordering,
    sample_wall_points_local,
)
from DoorOpening.utils.viser_pt import (
    format_iterated_record_path,
    prepare_pointcloud,
    prepare_world_points_from_local,
)

from DoorOpening.distillation._dagger_viser import ViserDebugMixin
from DoorOpening.distillation._dagger_checkpoint import CheckpointMixin
from DoorOpening.distillation._dagger_logging import LoggingMixin


def clip_teacher_obs(obs: torch.Tensor, clip_obs: float) -> torch.Tensor:
    if math.isfinite(clip_obs):
        return torch.clamp(obs, -clip_obs, clip_obs)
    return obs


class Dagger(ViserDebugMixin, CheckpointMixin, LoggingMixin):
    def __init__(self, env, config, summaries_dir, nn_dir):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_ddp = dist.is_available() and dist.is_initialized() and self.world_size > 1
        self.env = env
        self.ov_env = getattr(env, "unwrapped", getattr(env, "env", env))
        self.num_envs = self.ov_env.num_envs
        self.num_actions = int(self.ov_env.cfg.action_space)
        self.config = config

        base_action_dim = len(self.ov_env._robot_base_dof_idx)
        arm_action_dim = len(self.ov_env._robot_arm_dof_idx)
        hand_action_dim = len(self.ov_env._robot_finger_dof_idx)
        self.teacher_num_actions = base_action_dim + arm_action_dim + hand_action_dim
        self.student_joint_ids = torch.cat(
            (
                self.ov_env._robot_base_dof_idx,
                self.ov_env._robot_arm_dof_idx,
                self.ov_env._robot_finger_dof_idx,
            ),
            dim=0,
        )
        full_target_joint_ids = torch.as_tensor(self.ov_env._robot_dof_idx, device=self.device, dtype=torch.long)
        full_joint_id_to_target_pos = {
            int(joint_id): pos for pos, joint_id in enumerate(full_target_joint_ids.tolist())
        }
        self.student_target_indices_in_env = torch.as_tensor(
            [full_joint_id_to_target_pos[int(joint_id)] for joint_id in self.student_joint_ids.tolist()],
            device=self.device,
            dtype=torch.long,
        )
        self.action_component_dims = OrderedDict(
            [
                ("base", base_action_dim),
                ("arm", arm_action_dim),
                ("hand", hand_action_dim),
            ]
        )
        self.action_component_aliases = {
            "base": "base",
            "arm": "arm",
            "hand": "hand",
            "finger": "hand",
        }
        self.action_component_history_indices = self._build_action_component_history_indices()
        self.proprio_component_history_indices = self._build_proprio_component_history_indices()
        self.base_action_rot_local_idx = int(self.ov_env._robot_base_rot_local_idx[0].detach().cpu().item())
        self.base_action_xy_local_idx = [
            int(idx) for idx in self.ov_env._robot_base_xy_local_idx.detach().cpu().tolist()
        ]
        self.base_action_scale = max(float(self.ov_env.cfg.base_action_scale), 1e-6)

        self.student_cfg = self.config.get("student", {})
        self.teacher_cfg = self.config.get("teacher", {})
        self.play_policy = bool(self.config.get("play_policy", False))
        self.runtime_cfg = self.config.get("dagger", {})
        self.wall_distractor_cfg = dict(self.runtime_cfg.get("wall_distractors", {}))
        self.door_hole_aug_cfg = dict(self.runtime_cfg.get("door_hole_aug", {}))

        self.lr = float(self.runtime_cfg.get("learning_rate", 1e-4))
        self.lr_schedule = str(self.runtime_cfg.get("lr_schedule", "cosine")).lower()
        self.min_lr = float(self.runtime_cfg.get("min_learning_rate", 1e-5))
        self.weight_decay = float(self.runtime_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(self.runtime_cfg.get("grad_clip", 1.0))
        self.num_iters = int(self.runtime_cfg.get("num_iters", 1_000_000))
        self.lr_decay_iters = int(self.runtime_cfg.get("lr_decay_iters", 100_000))
        self.teacher_forcing_warmup_iters = int(self.runtime_cfg.get("teacher_forcing_warmup_iters", 0))
        self.teacher_forcing_transition_iters = int(
            self.runtime_cfg.get(
                "teacher_forcing_transition_iters",
                self.runtime_cfg.get("teacher_forcing_iters", 100_000),
            )
        )
        self.teacher_forcing_min_beta = float(self.runtime_cfg.get("teacher_forcing_min_beta", 0.0))
        self.log_interval = int(self.runtime_cfg.get("log_interval", 100))
        self.save_interval = int(self.runtime_cfg.get("save_interval", 5_000))
        self.pointcloud_source = str(self.runtime_cfg.get("pointcloud_source", "both")).lower()
        self.scene_robot_pcd_num_points = self.runtime_cfg.get(
            "scene_robot_num_points",
            self.runtime_cfg.get("robot_num_points"),
        )
        # `sampler_render` is kept as a backwards-compatible alias for older configs.
        self.depth_cam_render_cfg = self.runtime_cfg.get(
            "depth_cam_render",
            self.runtime_cfg.get("sampler_render", {}),
        )
        self.depth_cam_render_num_points = self.depth_cam_render_cfg.get("num_points")
        self.depth_cam_render_inflate_px = int(self.depth_cam_render_cfg.get("inflate_px", 2))
        self.depth_cam_render_jitter_std_m = float(self.depth_cam_render_cfg.get("jitter_std_m", 0.004))
        self.depth_cam_render_clip_mode = str(self.depth_cam_render_cfg.get("clip_mode", "post"))
        self.depth_cam_render_jitter_mode = str(self.depth_cam_render_cfg.get("jitter_mode", "xyz"))
        self.depth_cam_render_use_compile = bool(self.depth_cam_render_cfg.get("use_compile", True))
        # Spatial blur of the rendered depth image before unprojection, to emulate the RealSense
        # SDK spatial filter (handle edges blur into the door surface instead of staying mesh-crisp).
        # 0/1 = disabled. Kernel is in depth-image pixels; sigma<=0 falls back to a box blur.
        self.depth_cam_render_blur_kernel_px = int(self.depth_cam_render_cfg.get("blur_kernel_px", 0))
        self.depth_cam_render_blur_sigma_px = float(self.depth_cam_render_cfg.get("blur_sigma_px", 0.0))
        # Static occluders (door panel + wall distractors) are rasterized a second time with this much
        # larger inflate_px and composited via a per-pixel min with the main depth. Sparse occluders
        # (esp. the wall distractors) otherwise leave per-pixel z-buffer gaps at close range that let
        # farther points (background, the other side of the wall) show through. 0 disables the pass.
        self.depth_cam_render_occluder_inflate_px = int(self.depth_cam_render_cfg.get("occluder_inflate_px", 0))
        self.lidar_render_cfg = self.runtime_cfg.get("lidar_render", {})
        self.lidar_num_points = self.lidar_render_cfg.get("num_points")
        self.lidar_num_azimuth = int(self.lidar_render_cfg.get("num_azimuth", 512))
        self.lidar_num_polar = int(self.lidar_render_cfg.get("num_polar", 128))
        self.lidar_near_m = float(self.lidar_render_cfg.get("near_m", 0.1))
        self.lidar_far_m = self.lidar_render_cfg.get("far_m", 30.0)
        if self.lidar_far_m is not None:
            self.lidar_far_m = float(self.lidar_far_m)
        self.lidar_suppress_bins = int(self.lidar_render_cfg.get("suppress_bins", 2))
        self.lidar_occlusion_eps_m = float(self.lidar_render_cfg.get("occlusion_eps_m", 0.02))
        self.lidar_occlusion_eps_rel = float(self.lidar_render_cfg.get("occlusion_eps_rel", 0.01))
        self.lidar_jitter_std_m = float(self.lidar_render_cfg.get("jitter_std_m", 0.001))
        self.lidar_use_compile = bool(self.lidar_render_cfg.get("use_compile", True))
        # Lidar analog of depth_cam_render_occluder_inflate_px: bin radius for the second,
        # occluder-only pass. 0 disables it.
        self.lidar_render_occluder_fill_bins = int(self.lidar_render_cfg.get("occluder_fill_bins", 0))
        self.robot_pointcloud_filter_cfg = dict(self.runtime_cfg.get("robot_pointcloud_filter", {}))
        self.robot_pointcloud_filter_enabled = bool(self.robot_pointcloud_filter_cfg.get("enabled", True))
        self.robot_pointcloud_sdf_cutoff = float(self.robot_pointcloud_filter_cfg.get("sdf_cutoff", 0.02))
        self.robot_pointcloud_filter_max_points_per_process = int(
            self.robot_pointcloud_filter_cfg.get("max_points_per_process", 5000)
        )
        self.append_robot_model_to_policy_cloud = bool(
            self.runtime_cfg.get(
                "append_robot_model_to_policy_cloud",
                self.runtime_cfg.get("append_robot_gt_to_policy_cloud", True),
            )
        )
        self.robot_model_policy_points = self.runtime_cfg.get(
            "robot_model_policy_points",
            self.runtime_cfg.get("robot_gt_policy_points"),
        )
        # Runtime controls for optional raw point-cloud replay dumps.
        self.viser_cfg = dict(self.runtime_cfg.get("viser", {}))
        self.viser_raw_cfg = dict(self.viser_cfg.get("raw", {}))
        self.viser_raw_enabled = self.rank == 0 and bool(self.viser_raw_cfg.get("enabled", False))
        self.viser_raw_save_interval = max(
            1,
            int(self.viser_raw_cfg.get("save_interval", self.viser_cfg.get("raw_interval", 1000))),
        )
        self.viser_raw_capture_interval = max(
            1,
            int(self.viser_raw_cfg.get("capture_interval", self.viser_raw_cfg.get("frame_stride", 2))),
        )
        self.viser_raw_max_points = int(self.viser_raw_cfg.get("max_points", 12_000))
        self.viser_raw_max_frames = max(0, int(self.viser_raw_cfg.get("max_frames", 0)))
        if self.pointcloud_source not in {"sampler", "depth", "lidar", "both"}:
            raise ValueError(f"Unsupported pointcloud_source '{self.pointcloud_source}'.")
        if self.teacher_forcing_warmup_iters < 0:
            raise ValueError("teacher_forcing_warmup_iters must be non-negative.")
        if self.teacher_forcing_transition_iters < 0:
            raise ValueError("teacher_forcing_transition_iters must be non-negative.")
        if not 0.0 <= self.teacher_forcing_min_beta <= 1.0:
            raise ValueError("teacher_forcing_min_beta must be in [0, 1].")
        if self.lr_schedule not in {"linear", "cosine"}:
            raise ValueError("lr_schedule must be one of {'linear', 'cosine'}.")
        if self.lr_decay_iters < 0:
            raise ValueError("lr_decay_iters must be non-negative.")
        if self.min_lr < 0.0:
            raise ValueError("min_learning_rate must be non-negative.")
        if self.min_lr > self.lr:
            raise ValueError("min_learning_rate must be less than or equal to learning_rate.")

        self.games_to_track = 100
        self.frame = 0
        self.epoch_num = 0
        self.resume_iteration = 0
        self._resumed_from_student_ckpt = False

        self.nn_dir = nn_dir
        self.debug_pointcloud_dir = os.path.join(self.nn_dir if self.nn_dir is not None else os.getcwd(), "debug_pointclouds")
        self.wandb_cfg = self.runtime_cfg.get("wandb", self.config.get("wandb", {}))
        self.use_wandb = self.rank == 0 and bool(self.wandb_cfg.get("enabled", False))
        self.wandb_run = None
        if self.use_wandb:
            if wandb is None:
                raise ImportError("wandb logging is enabled, but the 'wandb' package is not installed.")
            self._init_wandb(summaries_dir)

        self.temporal_obs_cfg = {}
        self.observation_lag_cfg = {}
        self.temporal_derived_state_specs = OrderedDict()
        self.temporal_history_s = 0.0
        self.temporal_obs_delay_range_s = (0.0, 0.0)
        self.temporal_command_delay_range_s = (0.0, 0.0)
        self.max_temporal_history_s = 0.0
        self.temporal_dt_s = 0.0
        self.temporal_history_len = 0
        self.temporal_current_time_s = 0.0
        self.temporal_time_history = None
        self.temporal_q_history = None
        self.temporal_target_history = None
        self.temporal_base_vel_history = None
        self.temporal_aux_handle_history = None
        self.temporal_push_pull_belief_history = None
        self.proprio_temporal_enabled = False
        self.proprio_temporal_obs_key = None
        self.proprio_temporal_timestamps_ms = tuple()
        self.proprio_temporal_timestamps_s = tuple()
        self.proprio_temporal_fields = tuple()
        self.proprio_temporal_field_state_keys = OrderedDict()
        self.proprio_temporal_field_dims = OrderedDict()
        self.proprio_temporal_covered_state_keys = frozenset()
        self.temporal_aux_handle_enabled = False
        self.temporal_push_pull_belief_enabled = False
        self.observation_lag_enabled = False
        self.observation_lag_apply_to_proprio = True
        self.observation_lag_apply_to_pointcloud = False
        self.observation_lag_per_env = True
        self.observation_lag_per_timestamp = True
        self.observation_lag_clamp_to_available_history = True
        self.observation_lag_max_jitter_ms = 0
        self.observation_lag_mode = "symmetric"
        self.latest_obs_lag_enabled = 0.0
        self.latest_obs_lag_mean_ms = 0.0
        self.latest_obs_lag_min_ms = 0.0
        self.latest_obs_lag_max_ms = 0.0
        self.latest_obs_lag_effective_age_ms_by_timestamp = OrderedDict()
        self.teacher_forcing_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.interval_success_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.interval_completed_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        global_num_envs_tensor = torch.tensor([float(self.num_envs)], dtype=torch.float64, device=self.device)
        if self.use_ddp:
            dist.all_reduce(global_num_envs_tensor, op=dist.ReduceOp.SUM)
        self.global_num_envs = max(1, int(global_num_envs_tensor.item()))
        self.completed_rewards = deque(maxlen=self.games_to_track)
        self.completed_lengths = deque(maxlen=self.games_to_track)
        self.student_update_steps = 0
        self.last_local_update_batch_size = 0
        self.last_global_update_batch_size = 0
        self.train_env_mask = None
        self.validation_env_mask = None
        self.global_train_num_envs = 0
        self.global_validation_num_envs = 0
        self.latest_student_proprio_vector = None
        self.latest_aux_input_vector = None
        self.latest_aux_target_vector = None
        self.push_pull_condition_enabled = False
        self.push_pull_condition_obs_key = "push_pull_cond"
        self.push_pull_condition_source = "oracle"
        # Independent, additive left/right (handle_side) oracle one-hot conditioning. Coexists with
        # push/pull; oracle-only (no predicted source) and no temporal history input.
        self.left_right_condition_enabled = False
        self.left_right_condition_obs_key = "left_right_cond"
        self.left_right_family_one_hot = None
        self.latest_fraction_left = 0.0
        self.latest_fraction_right = 0.0
        # How aux_handle_pos is seeded at the first rollout step after each reset:
        # "zeros" -> seed the aux feedback buffer with zeros; "ground_truth" -> seed with the
        # sim handle pose. After the first step the predicted aux overwrites the buffer either way.
        self.aux_handle_init_source = "zeros"
        self.aux_handle_init_noise_m = None
        self.aux_handle_init_bias_m = None
        self.aux_handle_bias = None
        self.aux_feedback_noise_m = None
        self.push_pull_detach_predicted_condition = True
        self.push_pull_family_one_hot = None
        self.push_pull_condition_buffer = None
        self.latest_fraction_push = 0.0
        self.latest_fraction_pull = 0.0
        self.latest_push_pull_pred_entropy = 0.0
        self.latest_push_pull_pred_acc = None
        self.latest_push_pull_condition_source = "disabled"
        self.latest_push_pull_perturb_to_push_count = 0
        self.latest_push_pull_perturb_to_pull_count = 0
        self.latest_push_pull_belief_input = None
        self.latest_push_pull_belief_hist_entropy_now = 0.0
        self.latest_push_pull_belief_hist_entropy_mean = 0.0
        self._logged_temporal_state_input_keys = False
        self._timing_stats = {"sum_ms": 0.0, "count": 0}
        self.logged_env_metric_prefixes = ("dr/", "dr_limit/", "dr_sample/", "reset/")
        self.latest_env_log_metrics = {}
        self.zero_local_pcd_crop_center = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._init_teacher()
        self._init_student()
        self._init_history_buffers()
        self._init_pointcloud_assets()
        self._init_viser_debug_tools()

    def _init_teacher(self):
        self.teacher_model = None
        self.teacher_models = OrderedDict()
        self.teacher_models_by_family_id = OrderedDict()
        self.multi_teacher_enabled = False

        teachers_cfg = self.teacher_cfg.get("teachers")
        has_multi_teacher_cfg = isinstance(teachers_cfg, dict) and len(teachers_cfg) > 0
        if self.play_policy and self.teacher_cfg.get("ckpt") is None and not has_multi_teacher_cfg:
            return

        cfg_path = self.teacher_cfg.get("cfg")
        if not cfg_path:
            raise ValueError("Teacher config path is required unless play_policy=True with a student checkpoint.")

        self.teacher_network_params = self.load_yaml(cfg_path)["params"]
        self.teacher_network = self.load_networks(self.teacher_network_params)
        self.teacher_obs_type = self.teacher_cfg.get("obs_type", "policy")
        self.teacher_clip_obs = float(
            self.teacher_network_params.get("env", {}).get("clip_observations", math.inf)
        )
        self.teacher_strict_load = self.teacher_cfg.get("strict_load", True)
        self.teacher_allow_key_adjust = self.teacher_cfg.get("allow_key_adjust", True)

        teacher_model_config = {
            "actions_num": self.teacher_num_actions,
            "input_shape": (int(self.ov_env.cfg.observation_space),),
            "num_seqs": self.num_envs,
            "value_size": 1,
            "normalize_value": self.teacher_network_params["config"]["normalize_value"],
            "normalize_input": self.teacher_network_params["config"]["normalize_input"],
        }

        if has_multi_teacher_cfg:
            self.multi_teacher_enabled = True
            for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
                family_cfg = teachers_cfg.get(family_name)
                if family_cfg is None:
                    raise ValueError(
                        f"Missing multi-teacher checkpoint config for '{family_name}'. "
                        f"Expected teacher.teachers to define all families: {list(DOOR_FAMILY_NAMES)}."
                    )
                if isinstance(family_cfg, str):
                    family_ckpt = family_cfg
                elif isinstance(family_cfg, dict):
                    family_ckpt = family_cfg.get("ckpt")
                else:
                    raise TypeError(
                        f"teacher.teachers.{family_name} must be a checkpoint path or a mapping with 'ckpt'."
                    )
                if family_ckpt is None and not self.play_policy:
                    raise ValueError(f"Teacher checkpoint is required for family '{family_name}'.")
                family_model = self.teacher_network.build(teacher_model_config).to(self.device)
                if family_ckpt is not None:
                    self.set_teacher_weights(
                        family_ckpt,
                        model=family_model,
                        strict=self.teacher_strict_load,
                        allow_adjust=self.teacher_allow_key_adjust,
                    )
                self.teacher_models[family_name] = family_model
                self.teacher_models_by_family_id[family_id] = family_model

            self.teacher_model = next(iter(self.teacher_models.values()), None)
            print("Loaded multi-teacher families:", ", ".join(self.teacher_models.keys()))
            return

        self.teacher_model = self.teacher_network.build(teacher_model_config).to(self.device)

        teacher_ckpt = self.teacher_cfg.get("ckpt")
        if teacher_ckpt is None and not self.play_policy:
            raise ValueError("Teacher checkpoint is required for distillation.")
        if teacher_ckpt is not None:
            self.set_teacher_weights(
                teacher_ckpt,
                strict=self.teacher_strict_load,
                allow_adjust=self.teacher_allow_key_adjust,
            )

    def _init_student(self):
        cfg_path = self.student_cfg.get("cfg")
        if not cfg_path:
            raise ValueError("Student config path is required.")

        student_cfg_data = self.load_yaml(cfg_path) or {}
        if not isinstance(student_cfg_data, dict):
            raise ValueError(f"Student config at '{cfg_path}' must be a YAML mapping.")
        student_yaml_runtime_cfg = dict(student_cfg_data.pop("dagger", {}) or {})
        self.local_pcd_range = list(student_cfg_data.pop("local_pcd_range", [1.0, 0.35, 0.35]))
        self.local_pcd_x_direction_cutoff = student_cfg_data.pop("x_direction_cutoff", -0.5)
        self.scene_door_pcd_num_points = int(
            student_cfg_data.pop(
                "scene_door_num_points",
                student_cfg_data.pop("door_pcd_num_points", 4096),
            )
        )
        self.temporal_obs_cfg = dict(student_cfg_data.pop("temporal_obs", {}) or {})
        self.observation_lag_cfg = dict(student_cfg_data.pop("observation_lag", {}) or {})
        self.push_pull_condition_perturb_cfg = dict(student_cfg_data.pop("push_pull_condition_perturb", {}) or {})
        self.temporal_history_s = float(self.temporal_obs_cfg.get("history_s", 0.0))
        self.temporal_obs_delay_range_s = self.temporal_obs_cfg.get("obs_delay_s", [0.0, 0.0])
        self.temporal_command_delay_range_s = self.temporal_obs_cfg.get("command_delay_s", [0.0, 0.0])

        student_model_kwargs = {
            key: value
            for key, value in student_cfg_data.items()
            if not str(key).startswith("_")
        }
        self.student_model = PCDTransformer(**student_model_kwargs).to(self.device)
        self.push_pull_condition_enabled = bool(getattr(self.student_model, "push_pull_condition_enabled", False))
        self.push_pull_condition_obs_key = str(
            getattr(self.student_model, "push_pull_condition_obs_key", "push_pull_cond")
        )
        self.push_pull_condition_source = str(
            getattr(self.student_model, "push_pull_condition_source", "oracle")
        ).lower()
        self.push_pull_detach_predicted_condition = bool(
            getattr(self.student_model, "push_pull_detach_predicted_condition", True)
        )
        self.push_pull_condition_cfg = dict(getattr(self.student_model, "push_pull_condition_cfg", {}) or {})
        self.left_right_condition_enabled = bool(
            getattr(self.student_model, "left_right_condition_enabled", False)
        )
        self.left_right_condition_obs_key = str(
            getattr(self.student_model, "left_right_condition_obs_key", "left_right_cond")
        )
        self.proprio_temporal_enabled = bool(
            getattr(
                self.student_model,
                "temporal_state_enabled",
                getattr(self.student_model, "proprio_temporal_enabled", False),
            )
        )
        self.proprio_temporal_obs_key = getattr(
            self.student_model,
            "temporal_state_obs_key",
            getattr(self.student_model, "proprio_temporal_obs_key", None),
        )
        self.proprio_temporal_timestamps_ms = tuple(
            int(timestamp)
            for timestamp in getattr(
                self.student_model,
                "temporal_state_timestamps_ms",
                getattr(self.student_model, "proprio_temporal_timestamps_ms", ()),
            )
        )
        self.proprio_temporal_timestamps_s = tuple(float(timestamp) / 1000.0 for timestamp in self.proprio_temporal_timestamps_ms)
        self.proprio_temporal_fields = tuple(
            str(field)
            for field in getattr(
                self.student_model,
                "temporal_state_fields",
                getattr(self.student_model, "proprio_temporal_fields", ()),
            )
        )
        self.proprio_temporal_field_state_keys = OrderedDict(
            getattr(
                self.student_model,
                "temporal_state_field_state_keys",
                getattr(self.student_model, "proprio_temporal_field_state_keys", OrderedDict()),
            )
        )
        self.proprio_temporal_field_obs_keys = OrderedDict(
            getattr(
                self.student_model,
                "temporal_state_field_obs_keys",
                getattr(self.student_model, "proprio_temporal_field_obs_keys", OrderedDict()),
            )
        )
        self.proprio_temporal_field_dims = OrderedDict(
            (str(key), int(value))
            for key, value in getattr(
                self.student_model,
                "temporal_state_field_dims",
                getattr(self.student_model, "proprio_temporal_field_dims", OrderedDict()),
            ).items()
        )
        self.proprio_temporal_covered_state_keys = frozenset(
            getattr(
                self.student_model,
                "temporal_state_covered_state_keys",
                getattr(self.student_model, "proprio_temporal_covered_state_keys", frozenset()),
            )
        )
        self.temporal_state_uses_field_shared_encoders = bool(
            getattr(self.student_model, "temporal_state_uses_field_shared_encoders", False)
        )
        self.temporal_state_input_obs_keys = tuple(
            obs_key
            for obs_keys in self.proprio_temporal_field_obs_keys.values()
            for obs_key in obs_keys
        )
        self.mode_prediction_enabled = bool(getattr(self.student_model, "mode_prediction_enabled", False))
        self.mode_weight = float(getattr(self.student_model, "mode_weight", 0.0))
        self.num_modes = int(getattr(self.student_model, "num_modes", 4))
        self.door_joint_prediction_enabled = bool(getattr(self.student_model, "door_joint_prediction_enabled", False))
        self.door_joint_prediction_weight = float(getattr(self.student_model, "door_joint_prediction_weight", 0.0))
        self.door_joint_output_dim = int(getattr(self.student_model, "door_joint_output_dim", 0))
        if self.student_model.action_head.out_features != self.num_actions:
            raise ValueError(
                f"Student action_dim ({self.student_model.action_head.out_features}) "
                f"does not match env action dim ({self.num_actions})."
        )
        self._init_door_joint_prediction_training_state()
        self._init_prediction_training_state()
        self._init_mode_prediction_training_state()
        self._init_push_pull_condition_runtime_state(student_yaml_runtime_cfg)

        if self.use_ddp:
            self.student_model_ddp = DDP(
                self.student_model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )
        else:
            self.student_model_ddp = self.student_model
        self.optimizer = torch.optim.AdamW(
            self.student_model_ddp.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = self._build_lr_scheduler()

        student_ckpt = self.student_cfg.get("ckpt")
        if student_ckpt is not None:
            self.load_student_weights(student_ckpt)
        self._apply_optimizer_runtime_overrides()

        all_state_encoder_keys = tuple(
            key
            for key, cfg in self.student_model.state_encoders_cfg.items()
            if cfg.get("use_state", False)
        )
        self.state_encoders_keys = tuple(
            key
            for key in all_state_encoder_keys
            if key not in self.proprio_temporal_covered_state_keys
            and key != self.push_pull_condition_obs_key
            and key != self.left_right_condition_obs_key
        )
        self.pcd_encoders_keys = tuple(
            key
            for key, cfg in self.student_model.pcd_encoders_cfg.items()
            if cfg.get("use_pcd", False)
        )

        self.temporal_derived_state_specs = OrderedDict()
        for key in self.state_encoders_keys:
            spec = self._parse_temporal_derived_state_key(key)
            if spec is None:
                continue
            input_dim = int(self.student_model.state_encoders_cfg[key]["input_dim"])
            if input_dim != spec["dim"]:
                raise ValueError(
                    "{} must have input_dim={} to match the {} temporal slice.".format(
                        key,
                        spec["dim"],
                        spec["component"],
                    )
                )
            self.temporal_derived_state_specs[key] = spec
        spec_offsets = [
            float(spec["offset_s"])
            for spec in self.temporal_derived_state_specs.values()
            if spec["offset_s"] is not None
        ]
        self.max_temporal_history_s = max([0.0, self.temporal_history_s, *spec_offsets, *self.proprio_temporal_timestamps_s])
        self._init_observation_lag_state()
        if student_ckpt is not None and self.temporal_derived_state_specs and self.rank == 0:
            print(
                "Warning: student checkpoint was loaded while temporal derived inputs are active. "
                "New temporal state encoder weights may need fresh training."
            )
        if student_ckpt is not None and self.proprio_temporal_enabled and self.rank == 0:
            print(
                "Warning: student checkpoint was loaded while temporal_state_encoders are enabled. "
                "Timestamp-specific state encoders are replaced by shared per-field temporal encoders; "
                "checkpoint loading is non-strict and may leave new temporal weights to train from scratch."
            )
        if self.rank == 0 and self.proprio_temporal_enabled and self.temporal_state_input_obs_keys and not self._logged_temporal_state_input_keys:
            print("[INFO] temporal_state_encoders consume state keys:", ", ".join(self.temporal_state_input_obs_keys))
            self._logged_temporal_state_input_keys = True

        self.aux_state_specs = OrderedDict()
        self.aux_input_dim = 0
        for key in self.state_encoders_keys:
            spec = self._parse_aux_state_key(key)
            if spec is None:
                continue
            input_dim = int(self.student_model.state_encoders_cfg[key]["input_dim"])
            if input_dim != spec["dim"]:
                raise ValueError(
                    "{} must have input_dim={} to match the {} auxiliary state.".format(
                        key,
                        spec["dim"],
                        spec["name"],
                    )
                )
            spec["slice"] = slice(self.aux_input_dim, self.aux_input_dim + spec["dim"])
            self.aux_input_dim += spec["dim"]
            self.aux_state_specs[key] = spec
        self.has_aux_input = len(self.aux_state_specs) > 0
        self.has_aux_prediction = bool(getattr(self.student_model, "aux_prediction", False))
        if self.has_aux_prediction and not self.has_aux_input:
            raise ValueError("Aux prediction requires at least one enabled aux_* state encoder.")
        self.aux_feedback_to_policy = self.has_aux_input and self.has_aux_prediction and bool(
            self.runtime_cfg.get("aux_feedback_to_policy", True)
        )
        self.aux_handle_init_source = str(
            self.runtime_cfg.get("aux_handle_init_source", "zeros")
        ).lower()
        allowed_aux_handle_init = {"zeros", "ground_truth"}
        if self.aux_handle_init_source not in allowed_aux_handle_init:
            raise ValueError(
                f"dagger.aux_handle_init_source must be one of {sorted(allowed_aux_handle_init)}, "
                f"got '{self.aux_handle_init_source}'."
            )
        if self.aux_handle_init_source == "ground_truth":
            # Seeding the buffer with sim GT only matters on the first step after a reset, and only
            # if that buffer is actually fed to the policy (it is overwritten by predictions afterwards).
            if not self.has_aux_input:
                raise ValueError(
                    "dagger.aux_handle_init_source='ground_truth' requires an enabled aux_* state encoder."
                )
            if not self.aux_feedback_to_policy:
                raise ValueError(
                    "dagger.aux_handle_init_source='ground_truth' requires aux_feedback_to_policy=true; "
                    "otherwise the policy never sees the seeded handle pose."
                )
        # Perturbations on the handle-pose input (never on the aux regression target), modelling real
        # detector error (e.g. np.median over a SAM3 handle mask vs. the true handle center):
        #   noise    -> fresh per-call isotropic jitter, resampled every step (frame-to-frame noise).
        #   bias     -> a per-episode constant isotropic offset, resampled once per env reset
        #               (systematic detection offset that persists for the whole episode).
        #   feedback -> fresh per-step isotropic jitter added to the aux prediction before it is
        #               cached in aux_buffer (recurrent mode only), so the fed-back handle estimate
        #               the policy sees is never a noiseless copy of its own last prediction and
        #               error can compound across the rollout like a real drifting detector.
        # Each is a random direction x magnitude in [0, bound], so its total Euclidean error <= bound (m).
        self.aux_handle_init_noise_m = self._parse_aux_handle_offset_bound(
            self.runtime_cfg.get("aux_handle_init_noise_m", 0.0),
            "aux_handle_init_noise_m",
        )
        self.aux_handle_init_bias_m = self._parse_aux_handle_offset_bound(
            self.runtime_cfg.get("aux_handle_init_bias_m", 0.0),
            "aux_handle_init_bias_m",
        )
        self.aux_feedback_noise_m = self._parse_aux_handle_offset_bound(
            self.runtime_cfg.get("aux_feedback_noise_m", 0.0),
            "aux_feedback_noise_m",
        )
        # Aux handle input mode:
        #   "recurrent"        -> legacy: policy input = previous step's aux prediction (aux_buffer).
        #   "closed_door_base" -> policy input = the closed-door (door joint=0) handle pose expressed
        #                         in the CURRENT robot base frame + fresh per-step noise. The aux
        #                         prediction head is still trained on the ACTUAL-joint handle pose,
        #                         but its output is never fed back (non-recurrent).
        self.aux_handle_input_mode = str(
            self.runtime_cfg.get("aux_handle_input_mode", "recurrent")
        ).lower()
        allowed_aux_handle_input_modes = {"recurrent", "closed_door_base"}
        if self.aux_handle_input_mode not in allowed_aux_handle_input_modes:
            raise ValueError(
                f"dagger.aux_handle_input_mode must be one of {sorted(allowed_aux_handle_input_modes)}, "
                f"got '{self.aux_handle_input_mode}'."
            )
        if self.aux_handle_input_mode == "closed_door_base":
            if not self.has_aux_input or "aux_handle_pos" not in self.aux_state_specs:
                raise ValueError(
                    "dagger.aux_handle_input_mode='closed_door_base' requires an enabled aux_handle_pos "
                    "state encoder."
                )
            if not callable(getattr(self.ov_env, "get_closed_handle_position_in_base_frame", None)):
                raise RuntimeError(
                    "dagger.aux_handle_input_mode='closed_door_base' requires the environment to expose "
                    "get_closed_handle_position_in_base_frame()."
                )
        self.aux_buffer = None
        if self.has_aux_input:
            self.aux_buffer = torch.zeros((self.num_envs, self.aux_input_dim), dtype=torch.float32, device=self.device)
        # Per-episode constant handle-pose bias (resampled at each reset). Only the handle position
        # is perturbed, so this is a [num_envs, 3] buffer regardless of the full aux input dim.
        self.aux_handle_bias = None
        if self.has_aux_input and "aux_handle_pos" in self.aux_state_specs:
            self.aux_handle_bias = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.temporal_aux_handle_enabled = (
            self.proprio_temporal_field_state_keys.get("aux_handle_pos") == "aux_handle_pos"
        )
        self.temporal_push_pull_belief_enabled = (
            self.proprio_temporal_field_state_keys.get("push_pull_belief") == self.push_pull_condition_obs_key
        )
        if self.temporal_aux_handle_enabled and "aux_handle_pos" not in self.aux_state_specs:
            raise RuntimeError(
                "temporal_state_encoders field 'aux_handle_pos' requires the aux_handle_pos state encoder "
                "to remain configured in state_encoders_cfg."
            )
        if self.temporal_push_pull_belief_enabled and not self.push_pull_condition_enabled:
            raise RuntimeError(
                "temporal_state_encoders field 'push_pull_belief' requires push_pull_condition.enabled=true."
            )
        if student_ckpt is not None and self.has_aux_input and self.rank == 0:
            print(
                "Warning: student checkpoint was loaded while aux_handle_* inputs are active. "
                "If the auxiliary state definition changed, reusing those weights can spike loss."
            )

        local_pcd_cfg = self.student_model.pcd_encoders_cfg.get("local_pcd_t")
        self.local_pcd_points = [0, 0, 0]
        if local_pcd_cfg is not None:
            self.local_pcd_points = list(local_pcd_cfg.get("num_points", [self.scene_door_pcd_num_points, 0, 0])[:3])
        if self.robot_model_policy_points is None and len(self.local_pcd_points) >= 3:
            self.robot_model_policy_points = int(self.local_pcd_points[2])
        if self.robot_model_policy_points is None:
            self.robot_model_policy_points = 0
        self.robot_model_policy_points = max(0, int(self.robot_model_policy_points))

    def _init_mode_prediction_training_state(self):
        self.mode_prediction_loss_enabled = self.mode_prediction_enabled and self.mode_weight > 0.0
        self.mode_family_semantics = {}
        self.mode_family_direction_ids = None
        self.latest_mode_direction_acc = None
        self.latest_dir_window_acc = 0.0
        self.latest_dir_window_balanced_acc = 0.0
        self.latest_dir_window_push_acc = 0.0
        self.latest_dir_window_pull_acc = 0.0
        self.latest_dir_window_num_push_labels = 0
        self.latest_dir_window_num_pull_labels = 0
        self.latest_dir_window_num_push_preds = 0
        self.latest_dir_window_num_pull_preds = 0

    def _init_prediction_training_state(self):
        cfg = dict(self.runtime_cfg.get("prediction_training", {}) or {})
        self.direction_loss_window_start = int(cfg.get("loss_window_start", 40))
        self.direction_loss_window_end = int(cfg.get("loss_window_end", 100))
        self.direction_loss_outside_window_weight = float(cfg.get("outside_window_weight", 0.25))
        if self.direction_loss_window_start < 0:
            raise ValueError("prediction_training.loss_window_start must be non-negative.")
        if self.direction_loss_window_end < self.direction_loss_window_start:
            raise ValueError(
                "prediction_training.loss_window_end must be greater than or equal to "
                "prediction_training.loss_window_start."
            )
        if not 0.0 <= self.direction_loss_outside_window_weight <= 1.0:
            raise ValueError("prediction_training.outside_window_weight must be in [0, 1].")

    def _init_door_joint_prediction_training_state(self):
        cfg = dict(self.runtime_cfg.get("door_joint_training", {}) or {})
        # Which door joints the aux head regresses, in output order. "hinge" = joint_2 (door swing
        # angle), "latch" = joint_1 (handle/latch rotation). Targets are the raw joint angles (rad).
        joint_name_to_env_idx = {
            "hinge": int(self.ov_env._door_hinge_joint_idx),
            "latch": int(self.ov_env._door_board_joint_idx),
        }
        joint_names = list(cfg.get("joints", ["hinge", "latch"]))
        if not joint_names:
            raise ValueError("door_joint_training.joints must list at least one joint.")
        unknown = [name for name in joint_names if name not in joint_name_to_env_idx]
        if unknown:
            raise ValueError(
                f"door_joint_training.joints contains unknown joints {unknown}; "
                f"valid options are {sorted(joint_name_to_env_idx)}."
            )
        self.door_joint_prediction_joint_names = joint_names
        self.door_joint_prediction_joint_idx = torch.as_tensor(
            [joint_name_to_env_idx[name] for name in joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self.door_joint_prediction_loss_type = str(cfg.get("loss_type", "mse")).lower()
        if self.door_joint_prediction_loss_type not in {"mse", "smooth_l1"}:
            raise ValueError(
                "door_joint_training.loss_type must be one of ['mse', 'smooth_l1']."
            )
        if self.door_joint_prediction_enabled and self.door_joint_output_dim != len(joint_names):
            raise ValueError(
                f"door_joint_prediction.output_dim ({self.door_joint_output_dim}) must match the "
                f"number of door_joint_training.joints ({len(joint_names)})."
            )

        # Per-joint mean absolute error (rad), one entry per predicted joint; None until first update.
        self.latest_door_joint_abs_err = None
        self.latest_door_joint_target_mean = None

    def _init_observation_lag_state(self):
        cfg = dict(self.observation_lag_cfg or {})
        self.observation_lag_enabled = bool(cfg.get("enabled", False))
        self.observation_lag_max_jitter_ms = int(cfg.get("max_jitter_ms", 0))
        self.observation_lag_mode = str(cfg.get("mode", "symmetric")).lower()
        self.observation_lag_apply_to_proprio = bool(cfg.get("apply_to_proprio", True))
        self.observation_lag_apply_to_pointcloud = bool(cfg.get("apply_to_pointcloud", False))
        self.observation_lag_per_env = bool(cfg.get("per_env", True))
        self.observation_lag_per_timestamp = bool(cfg.get("per_timestamp", True))
        self.observation_lag_clamp_to_available_history = bool(cfg.get("clamp_to_available_history", True))

        if self.observation_lag_max_jitter_ms < 0:
            raise ValueError("observation_lag.max_jitter_ms must be non-negative.")
        if self.observation_lag_mode != "symmetric":
            raise ValueError("observation_lag.mode must be 'symmetric'.")
        if self.observation_lag_apply_to_pointcloud:
            raise NotImplementedError(
                "observation_lag.apply_to_pointcloud=true is not implemented in this patch. "
                "Set apply_to_pointcloud=false."
            )
        self._reset_observation_lag_stats()

    def _init_push_pull_condition_runtime_state(self, student_yaml_runtime_cfg):
        allowed_sources = {"oracle", "predicted"}
        if self.push_pull_condition_enabled and self.push_pull_condition_source not in allowed_sources:
            raise ValueError(
                f"push_pull_condition.source must be one of {sorted(allowed_sources)}, "
                f"got '{self.push_pull_condition_source}'."
            )
        if self.push_pull_condition_enabled and self.push_pull_condition_source == "predicted":
            if not self.mode_prediction_enabled:
                raise RuntimeError(
                    "push_pull_condition.source requires mode_prediction.enabled=true so the student can "
                    "predict push/pull logits before the action pass."
                )
            if self.num_modes != 2:
                raise RuntimeError(
                    f"push_pull_condition.source='{self.push_pull_condition_source}' requires num_modes=2, "
                    f"got {self.num_modes}."
                )

        default_perturb_cfg = {
            "enabled": False,
            "probability": 0.1,
            "wrong_class_confidence_range": [0.85, 0.95],
        }
        merged_perturb_cfg = dict(default_perturb_cfg)
        merged_perturb_cfg.update(dict(student_yaml_runtime_cfg.get("push_pull_condition_perturb", {}) or {}))
        merged_perturb_cfg.update(dict(self.runtime_cfg.get("push_pull_condition_perturb", {}) or {}))
        merged_perturb_cfg.update(dict(self.push_pull_condition_perturb_cfg or {}))
        self.push_pull_condition_perturb_cfg = merged_perturb_cfg
        self.push_pull_condition_perturb_enabled = bool(merged_perturb_cfg.get("enabled", False))
        self.push_pull_condition_perturb_probability = float(merged_perturb_cfg.get("probability", 0.1))
        wrong_confidence_min, wrong_confidence_max = merged_perturb_cfg.get(
            "wrong_class_confidence_range",
            [0.85, 0.95],
        )
        self.push_pull_condition_perturb_wrong_confidence_min = float(wrong_confidence_min)
        self.push_pull_condition_perturb_wrong_confidence_max = float(wrong_confidence_max)
        if not 0.0 <= self.push_pull_condition_perturb_probability <= 1.0:
            raise ValueError("push_pull_condition_perturb.probability must be in [0, 1].")
        if self.push_pull_condition_perturb_enabled and not self.push_pull_condition_enabled:
            raise RuntimeError(
                "push_pull_condition_perturb.enabled=true requires push_pull_condition.enabled=true."
            )
        if not 0.5 <= self.push_pull_condition_perturb_wrong_confidence_min <= self.push_pull_condition_perturb_wrong_confidence_max <= 1.0:
            raise ValueError(
                "push_pull_condition_perturb.wrong_class_confidence_range must satisfy "
                "0.5 <= min <= max <= 1.0."
            )

    def _load_family_mode_semantics(self):
        handle_side_aliases = {
            "min": "left",
            "max": "right",
            "left": "left",
            "right": "right",
        }
        semantics_by_family_name = {}
        for asset_path, family_id in zip(door_asset_paths, door_asset_family_ids.detach().cpu().tolist()):
            family_name = DOOR_FAMILY_NAMES[int(family_id)]
            if family_name in semantics_by_family_name:
                continue

            handle_side = None
            opening_direction = None
            meta_path = Path(asset_path).resolve().parent / "variant_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or {}
                except (OSError, json.JSONDecodeError):
                    meta = {}
                actual_props = meta.get("actual_properties", {})
                target_props = meta.get("target_properties", {})
                if not isinstance(actual_props, dict):
                    actual_props = {}
                if not isinstance(target_props, dict):
                    target_props = {}

                handle_side_raw = actual_props.get("handle_side") or target_props.get("handle_side")
                opening_direction_raw = actual_props.get("opening_direction") or target_props.get("opening_direction")
                if handle_side_raw is not None:
                    handle_side = handle_side_aliases.get(str(handle_side_raw).lower(), str(handle_side_raw).lower())
                if opening_direction_raw is not None:
                    opening_direction = str(opening_direction_raw).lower()

            if handle_side is None or opening_direction is None:
                raise RuntimeError(
                    f"Missing handle_side/opening_direction metadata for family '{family_name}' in {meta_path}."
                )

            semantics_by_family_name[family_name] = (handle_side, opening_direction)
        return semantics_by_family_name

    def _init_push_pull_semantics_and_targets(self):
        self.push_pull_family_one_hot = None
        self.left_right_family_one_hot = None
        self.mode_family_direction_ids = None
        self.latest_fraction_push = 0.0
        self.latest_fraction_pull = 0.0
        self.latest_fraction_left = 0.0
        self.latest_fraction_right = 0.0
        push_pull_needed = (
            self.push_pull_condition_enabled
            or self.push_pull_condition_perturb_enabled
            or self.mode_prediction_enabled
        )
        if not (push_pull_needed or self.left_right_condition_enabled):
            return

        if self.mode_prediction_enabled and self.num_modes != 2:
            raise RuntimeError(
                f"Push/pull direction prediction expects num_modes=2, got {self.num_modes}."
            )

        self.mode_family_semantics = self._load_family_mode_semantics()

        if push_pull_needed:
            direction_name_to_id = {"pull": 0, "push": 1}
            family_one_hot = torch.zeros(
                (len(DOOR_FAMILY_NAMES), 2),
                dtype=torch.float32,
                device=self.device,
            )
            if self.mode_prediction_enabled:
                self.mode_family_direction_ids = torch.full(
                    (len(DOOR_FAMILY_NAMES),),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )

            missing_semantics = []
            for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
                _, opening_direction = self.mode_family_semantics.get(family_name, (None, None))
                if opening_direction == "push":
                    family_one_hot[family_id] = torch.tensor(
                        [1.0, 0.0], dtype=torch.float32, device=self.device
                    )
                    if self.mode_family_direction_ids is not None:
                        self.mode_family_direction_ids[family_id] = direction_name_to_id[opening_direction]
                elif opening_direction == "pull":
                    family_one_hot[family_id] = torch.tensor(
                        [0.0, 1.0], dtype=torch.float32, device=self.device
                    )
                    if self.mode_family_direction_ids is not None:
                        self.mode_family_direction_ids[family_id] = direction_name_to_id[opening_direction]
                else:
                    missing_semantics.append(family_name)

            if missing_semantics:
                raise RuntimeError(
                    "Could not infer push/pull semantics for active door families: "
                    f"{missing_semantics}."
                )
            self.push_pull_family_one_hot = family_one_hot

        if self.left_right_condition_enabled:
            self.left_right_family_one_hot = self._build_family_left_right_one_hot()

    def _build_family_left_right_one_hot(self):
        # Per-family oracle one-hot from handle_side metadata: index 0 = left, index 1 = right.
        # (min -> left, max -> right via the alias map in _load_family_mode_semantics.)
        family_one_hot = torch.zeros(
            (len(DOOR_FAMILY_NAMES), 2),
            dtype=torch.float32,
            device=self.device,
        )
        missing_semantics = []
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            handle_side, _ = self.mode_family_semantics.get(family_name, (None, None))
            if handle_side == "left":
                family_one_hot[family_id, 0] = 1.0
            elif handle_side == "right":
                family_one_hot[family_id, 1] = 1.0
            else:
                missing_semantics.append(family_name)
        if missing_semantics:
            raise RuntimeError(
                "Could not infer left/right (handle_side) semantics for active door families: "
                f"{missing_semantics}."
            )
        return family_one_hot

    def _build_gt_left_right_condition(self):
        if self.left_right_family_one_hot is None:
            raise RuntimeError("Left/right condition targets are not initialized.")
        if self.env_family_ids.ndim != 1 or self.env_family_ids.shape[0] != self.num_envs:
            raise RuntimeError(
                f"Expected env_family_ids shape [{self.num_envs}], got {tuple(self.env_family_ids.shape)}."
            )
        left_right_cond = self.left_right_family_one_hot[self.env_family_ids.long()]
        return left_right_cond.to(device=self.device, dtype=torch.float32)

    def _build_gt_push_pull_condition(self):
        if self.push_pull_family_one_hot is None:
            raise RuntimeError("Push/pull condition targets are not initialized.")
        if self.env_family_ids.ndim != 1 or self.env_family_ids.shape[0] != self.num_envs:
            raise RuntimeError(
                f"Expected env_family_ids shape [{self.num_envs}], got {tuple(self.env_family_ids.shape)}."
            )

        push_pull_cond = self.push_pull_family_one_hot[self.env_family_ids.long()]
        push_pull_cond = push_pull_cond.to(device=self.device, dtype=torch.float32)
        return push_pull_cond

    def _build_initial_predicted_push_pull_condition(self, num_envs):
        num_envs = int(num_envs)
        initial_condition = torch.full(
            (num_envs, 2),
            0.5,
            dtype=torch.float32,
            device=self.device,
        )
        if initial_condition.ndim != 2 or initial_condition.shape != (num_envs, 2):
            raise RuntimeError(
                f"initial_push_pull_cond must have shape [{num_envs}, 2], got {tuple(initial_condition.shape)}."
            )
        return initial_condition

    def _seed_push_pull_condition_buffer(self, env_ids=None):
        if not self.push_pull_condition_enabled or self.push_pull_condition_source != "predicted":
            self.push_pull_condition_buffer = None
            return

        if self.push_pull_condition_buffer is None:
            self.push_pull_condition_buffer = self._build_initial_predicted_push_pull_condition(self.num_envs)

        if env_ids is None:
            self.push_pull_condition_buffer[:] = self._build_initial_predicted_push_pull_condition(self.num_envs)
            return

        if env_ids.numel() == 0:
            return
        self.push_pull_condition_buffer[env_ids] = self._build_initial_predicted_push_pull_condition(env_ids.numel())

    def _get_recurrent_push_pull_condition(self):
        if self.push_pull_condition_source == "oracle":
            # Oracle source uses ground-truth one-hot each step.
            return self._build_gt_push_pull_condition()
        if self.push_pull_condition_source == "predicted":
            if self.push_pull_condition_buffer is None:
                raise RuntimeError(
                    "Predicted push/pull conditioning requires push_pull_condition_buffer to be initialized."
                )
            return self.push_pull_condition_buffer.clone()
        raise ValueError(f"Unsupported push_pull_condition.source '{self.push_pull_condition_source}'.")

    def _update_push_pull_condition_buffer(self, mode_logits):
        if self.push_pull_condition_source != "predicted":
            return
        # Recurrent push/pull conditioning is step-to-step state, not BPTT.
        # The next-step condition must be a fixed value from the previous step.
        next_push_pull_cond = self._mode_logits_to_push_pull_condition(mode_logits)
        self.push_pull_condition_buffer = next_push_pull_cond.detach()

    def _sample_soft_wrong_push_pull_condition(self):
        gt_condition = self._build_gt_push_pull_condition()
        wrong_confidence = torch.empty(self.num_envs, dtype=torch.float32, device=self.device).uniform_(
            self.push_pull_condition_perturb_wrong_confidence_min,
            self.push_pull_condition_perturb_wrong_confidence_max,
        )
        wrong_condition = torch.empty_like(gt_condition)
        gt_is_push = gt_condition[:, 0] > gt_condition[:, 1]
        wrong_condition[gt_is_push, 0] = 1.0 - wrong_confidence[gt_is_push]
        wrong_condition[gt_is_push, 1] = wrong_confidence[gt_is_push]
        wrong_condition[~gt_is_push, 0] = wrong_confidence[~gt_is_push]
        wrong_condition[~gt_is_push, 1] = 1.0 - wrong_confidence[~gt_is_push]
        return wrong_condition

    def _build_gt_push_pull_class_ids(self):
        gt_condition = self._build_gt_push_pull_condition()
        return gt_condition[:, 0].round().to(dtype=torch.long)

    def _mode_logits_to_push_pull_condition(self, mode_logits):
        if not isinstance(mode_logits, torch.Tensor):
            raise RuntimeError("Predicted push/pull condition requires mode logits from the student.")
        if mode_logits.ndim != 2 or mode_logits.shape != (self.num_envs, 2):
            raise RuntimeError(
                "Predicted push/pull logits must have shape [num_envs, 2] with class order [pull, push]; "
                f"got {tuple(mode_logits.shape)}."
            )
        mode_probs = torch.softmax(mode_logits, dim=-1)
        push_pull_cond = torch.stack([mode_probs[:, 1], mode_probs[:, 0]], dim=-1)
        return push_pull_cond

    def _record_push_pull_prediction_metrics(self, mode_logits):
        if mode_logits is None:
            self.latest_push_pull_pred_entropy = 0.0
            self.latest_push_pull_pred_acc = None
            return
        push_pull_cond = self._mode_logits_to_push_pull_condition(mode_logits.detach())
        entropy = -(push_pull_cond * torch.log(push_pull_cond.clamp_min(1.0e-6))).sum(dim=-1)
        self.latest_push_pull_pred_entropy = float(entropy.mean().detach().cpu().item())
        gt_class_ids = self._build_gt_push_pull_class_ids()
        pred_class_ids = mode_logits.detach().argmax(dim=-1)
        self.latest_push_pull_pred_acc = float(
            (pred_class_ids == gt_class_ids).float().mean().detach().cpu().item()
        )

    def _build_push_pull_condition_from_source(self, source):
        source = str(source).lower()
        if source == "oracle":
            return self._build_gt_push_pull_condition()
        if source == "predicted":
            # Predicted source uses the recurrent condition carried over from the previous step.
            return self._get_recurrent_push_pull_condition()
        raise ValueError(f"Unsupported push_pull_condition.source '{source}'.")

    def _is_push_pull_condition_perturb_active(self):
        if not self.push_pull_condition_perturb_enabled:
            return False
        return not self.play_policy

    def _apply_push_pull_condition_perturb(self, push_pull_cond):
        perturb_active = self._is_push_pull_condition_perturb_active()
        self.latest_push_pull_perturb_to_push_count = 0
        self.latest_push_pull_perturb_to_pull_count = 0
        if not perturb_active:
            self.latest_fraction_push = float(push_pull_cond[:, 0].mean().detach().cpu().item())
            self.latest_fraction_pull = float(push_pull_cond[:, 1].mean().detach().cpu().item())
            return push_pull_cond

        # Fresh per-step, per-env Bernoulli sampling. No duration window or persistent mask is kept.
        perturb_mask = torch.rand(self.num_envs, device=self.device) < self.push_pull_condition_perturb_probability
        perturbed = push_pull_cond.clone()
        if torch.any(perturb_mask):
            replacement = self._sample_soft_wrong_push_pull_condition()
            perturbed[perturb_mask] = replacement[perturb_mask]
            gt_argmax = self._build_gt_push_pull_condition().argmax(dim=-1)
            perturbed_argmax = perturbed.argmax(dim=-1)
            if not torch.all(perturbed_argmax[perturb_mask] != gt_argmax[perturb_mask]):
                raise RuntimeError("push_pull_condition perturbation did not flip GT labels.")
            self.latest_push_pull_perturb_to_push_count = int(
                (perturbed_argmax[perturb_mask] == 0).sum().detach().cpu().item()
            )
            self.latest_push_pull_perturb_to_pull_count = int(
                (perturbed_argmax[perturb_mask] == 1).sum().detach().cpu().item()
            )

        self.latest_fraction_push = float(perturbed[:, 0].mean().detach().cpu().item())
        self.latest_fraction_pull = float(perturbed[:, 1].mean().detach().cpu().item())
        return perturbed

    def _get_rollout_step_ids(self):
        rollout_step_ids = getattr(self.ov_env, "episode_length_buf", None)
        if rollout_step_ids is None:
            rollout_step_ids = self.current_lengths
        return rollout_step_ids.to(device=self.device)

    def _get_active_rollout_mask(self, rollout_step_ids):
        max_trial_steps = getattr(self.ov_env, "max_trial_steps", None)
        if max_trial_steps is not None:
            max_trial_steps = max_trial_steps.to(device=rollout_step_ids.device, dtype=rollout_step_ids.dtype)
            return rollout_step_ids < max_trial_steps

        reset_buf = getattr(self.ov_env, "reset_buf", None)
        if reset_buf is not None:
            return ~reset_buf.to(device=rollout_step_ids.device, dtype=torch.bool)

        return torch.ones_like(rollout_step_ids, dtype=torch.bool, device=rollout_step_ids.device)

    def _align_env_tensor_to_prediction(self, tensor, prediction_leading_shape, name):
        prediction_leading_shape = torch.Size(prediction_leading_shape)
        if tensor.shape == prediction_leading_shape:
            return tensor

        if tensor.ndim == 1 and prediction_leading_shape and tensor.shape[0] == prediction_leading_shape[0]:
            view_shape = (tensor.shape[0],) + (1,) * (len(prediction_leading_shape) - 1)
            return tensor.reshape(view_shape).expand(prediction_leading_shape)

        if tensor.numel() == math.prod(prediction_leading_shape):
            return tensor.reshape(prediction_leading_shape)

        raise RuntimeError(
            f"Could not align {name} with direction logits: "
            f"{tuple(tensor.shape)} vs leading shape {tuple(prediction_leading_shape)}."
        )

    def _prepare_direction_prediction_tensors(self, mode_logits, env_mask=None):
        if mode_logits.ndim < 2:
            raise RuntimeError(f"Expected direction logits to have a class dimension, got {tuple(mode_logits.shape)}.")
        if mode_logits.shape[-1] != self.num_modes:
            raise RuntimeError(
                f"Direction logits last dim ({mode_logits.shape[-1]}) does not match num_modes ({self.num_modes})."
            )

        prediction_leading_shape = mode_logits.shape[:-1]
        env_family_ids = self.env_family_ids.long()
        rollout_step_ids = self._get_rollout_step_ids()
        active_mask = self._get_active_rollout_mask(rollout_step_ids)
        if env_mask is not None:
            env_mask = env_mask.to(device=self.device, dtype=torch.bool)
            env_family_ids = env_family_ids[env_mask]
            rollout_step_ids = rollout_step_ids[env_mask]
            active_mask = active_mask[env_mask]
        direction_target = self.mode_family_direction_ids[env_family_ids]
        direction_valid = direction_target >= 0

        tensors = {
            "direction_target": direction_target,
            "direction_valid": direction_valid,
            "rollout_step_ids": rollout_step_ids,
            "active_mask": active_mask,
        }
        aligned = {
            name: self._align_env_tensor_to_prediction(tensor, prediction_leading_shape, name)
            for name, tensor in tensors.items()
        }
        return (
            mode_logits.reshape(-1, mode_logits.shape[-1]),
            aligned["direction_target"].reshape(-1).long(),
            aligned["direction_valid"].reshape(-1).bool(),
            aligned["active_mask"].reshape(-1).bool(),
            aligned["rollout_step_ids"].reshape(-1),
            prediction_leading_shape,
        )

    def _get_direction_step_mask(self, rollout_step_ids):
        return (
            (rollout_step_ids >= self.direction_loss_window_start)
            & (rollout_step_ids <= self.direction_loss_window_end)
        )

    def _get_direction_step_weights(self, rollout_step_ids):
        return torch.where(
            self._get_direction_step_mask(rollout_step_ids),
            torch.ones_like(rollout_step_ids, dtype=torch.float32),
            torch.full_like(rollout_step_ids, self.direction_loss_outside_window_weight, dtype=torch.float32),
        )

    def _update_direction_window_metrics(
        self,
        mode_logits,
        direction_target,
        direction_valid,
        active_mask,
        rollout_step_ids,
    ):
        window_mask = direction_valid & active_mask & self._get_direction_step_mask(rollout_step_ids)
        direction_pred = mode_logits.argmax(dim=-1)

        push_label_mask = window_mask & (direction_target == 1)
        pull_label_mask = window_mask & (direction_target == 0)
        push_pred_mask = window_mask & (direction_pred == 1)
        pull_pred_mask = window_mask & (direction_pred == 0)

        num_push_labels = int(push_label_mask.sum().detach().cpu().item())
        num_pull_labels = int(pull_label_mask.sum().detach().cpu().item())
        num_push_preds = int(push_pred_mask.sum().detach().cpu().item())
        num_pull_preds = int(pull_pred_mask.sum().detach().cpu().item())

        correct_mask = direction_pred == direction_target
        correct_push = int((correct_mask & push_label_mask).sum().detach().cpu().item())
        correct_pull = int((correct_mask & pull_label_mask).sum().detach().cpu().item())
        total_labels = num_push_labels + num_pull_labels
        total_correct = correct_push + correct_pull

        push_acc = float(correct_push / num_push_labels) if num_push_labels > 0 else 0.0
        pull_acc = float(correct_pull / num_pull_labels) if num_pull_labels > 0 else 0.0
        available_class_accs = []
        if num_push_labels > 0:
            available_class_accs.append(push_acc)
        if num_pull_labels > 0:
            available_class_accs.append(pull_acc)
        balanced_acc = float(sum(available_class_accs) / len(available_class_accs)) if available_class_accs else 0.0

        self.latest_dir_window_acc = float(total_correct / total_labels) if total_labels > 0 else 0.0
        self.latest_dir_window_balanced_acc = balanced_acc
        self.latest_dir_window_push_acc = push_acc
        self.latest_dir_window_pull_acc = pull_acc
        self.latest_dir_window_num_push_labels = num_push_labels
        self.latest_dir_window_num_pull_labels = num_pull_labels
        self.latest_dir_window_num_push_preds = num_push_preds
        self.latest_dir_window_num_pull_preds = num_pull_preds
        return window_mask

    def _compute_mode_prediction_loss(self, mode_logits, env_mask=None, update_metrics=True):
        if not self.mode_prediction_loss_enabled:
            return None
        if self.mode_family_direction_ids is None:
            raise RuntimeError("Mode prediction targets are not initialized.")

        mode_logits, direction_target, direction_valid, active_mask, rollout_step_ids, _ = (
            self._prepare_direction_prediction_tensors(mode_logits, env_mask=env_mask)
        )
        if not torch.any(direction_valid):
            raise RuntimeError("No valid direction targets are available for mode prediction.")

        if update_metrics:
            self._update_direction_window_metrics(
                mode_logits,
                direction_target,
                direction_valid,
                active_mask,
                rollout_step_ids,
            )

        valid_mask = direction_valid & active_mask
        if torch.any(valid_mask):
            per_sample_loss = torch.nn.functional.cross_entropy(
                mode_logits[valid_mask],
                direction_target[valid_mask],
                reduction="none",
            )
            step_weights = self._get_direction_step_weights(rollout_step_ids[valid_mask])
            direction_loss = (per_sample_loss * step_weights).sum() / step_weights.sum().clamp_min(1.0e-6)
            if update_metrics:
                direction_pred = mode_logits[valid_mask].argmax(dim=-1)
                direction_correct = (direction_pred == direction_target[valid_mask]).float()
                self.latest_mode_direction_acc = float(direction_correct.mean().detach().cpu().item())
        else:
            direction_loss = mode_logits.mean() * 0.0
            if update_metrics:
                self.latest_mode_direction_acc = None
        return direction_loss

    def _get_door_joint_prediction_target_raw(self, env_mask=None, update_metrics=True):
        door_joint_pos = self.ov_env.door.data.joint_pos[:, self.door_joint_prediction_joint_idx]
        if env_mask is not None:
            env_mask = env_mask.to(device=door_joint_pos.device, dtype=torch.bool)
            door_joint_pos = door_joint_pos[env_mask]
        door_joint_pos = door_joint_pos.to(device=self.device, dtype=torch.float32)
        if door_joint_pos.ndim != 2 or door_joint_pos.shape[-1] != len(self.door_joint_prediction_joint_names):
            raise RuntimeError(
                f"Expected door joint target shape [N, {len(self.door_joint_prediction_joint_names)}], "
                f"got {tuple(door_joint_pos.shape)}"
            )
        if update_metrics:
            self.latest_door_joint_target_mean = door_joint_pos.mean(dim=0).detach().cpu().tolist()
        return door_joint_pos

    def _compute_door_joint_prediction_loss(self, door_joint_pred, env_mask=None, update_metrics=True):
        if not self.door_joint_prediction_enabled:
            return None

        if env_mask is not None and door_joint_pred.shape[0] == self.num_envs:
            env_mask = env_mask.to(device=self.device, dtype=torch.bool)
            door_joint_pred = door_joint_pred[env_mask]
        target = self._get_door_joint_prediction_target_raw(env_mask=env_mask, update_metrics=update_metrics)
        if door_joint_pred.ndim == 3:
            door_joint_pred = door_joint_pred[:, 0, :]
        if door_joint_pred.shape[-1] != target.shape[-1]:
            raise RuntimeError(
                f"Door joint prediction head output dim ({door_joint_pred.shape[-1]}) does not match "
                f"target dim ({target.shape[-1]})."
            )

        # Build the active-rollout mask on the FULL env tensors (max_trial_steps is num_envs-sized),
        # then slice down to the student subset. Slicing rollout_step_ids before the comparison would
        # break broadcasting against the full-size max_trial_steps.
        rollout_step_ids = self._get_rollout_step_ids()
        valid_mask = self._get_active_rollout_mask(rollout_step_ids)
        if env_mask is not None:
            env_mask = env_mask.to(device=self.device, dtype=torch.bool)
            rollout_step_ids = rollout_step_ids[env_mask]
            valid_mask = valid_mask[env_mask]
        if not torch.any(valid_mask):
            loss = door_joint_pred.mean() * 0.0
            if update_metrics:
                self.latest_door_joint_abs_err = None
            return loss
        step_weights = self._get_direction_step_weights(rollout_step_ids[valid_mask])

        if self.door_joint_prediction_loss_type == "smooth_l1":
            per_env_loss = torch.nn.functional.smooth_l1_loss(
                door_joint_pred, target, reduction="none"
            ).mean(dim=-1)
        else:
            per_env_loss = torch.nn.functional.mse_loss(
                door_joint_pred, target, reduction="none"
            ).mean(dim=-1)
        loss = (per_env_loss[valid_mask] * step_weights).sum() / step_weights.sum().clamp_min(1.0e-6)

        if update_metrics:
            abs_err = (door_joint_pred - target).abs()[valid_mask]
            self.latest_door_joint_abs_err = abs_err.mean(dim=0).detach().cpu().tolist()

        return loss

    def _build_action_component_history_indices(self):
        target_joint_ids = self.student_joint_ids.to(device=self.device, dtype=torch.long)
        joint_id_to_target_pos = {int(joint_id): pos for pos, joint_id in enumerate(target_joint_ids.tolist())}

        indices = OrderedDict()
        indices["full"] = torch.arange(self.num_actions, device=self.device, dtype=torch.long)
        indices["base"] = torch.as_tensor(
            [joint_id_to_target_pos[int(joint_id)] for joint_id in self.ov_env._robot_base_dof_idx],
            device=self.device,
            dtype=torch.long,
        )
        indices["arm"] = torch.as_tensor(
            [joint_id_to_target_pos[int(joint_id)] for joint_id in self.ov_env._robot_arm_dof_idx],
            device=self.device,
            dtype=torch.long,
        )
        indices["hand"] = torch.as_tensor(
            [joint_id_to_target_pos[int(joint_id)] for joint_id in self.ov_env._robot_finger_dof_idx],
            device=self.device,
            dtype=torch.long,
        )
        return indices

    def _build_proprio_component_history_indices(self):
        return OrderedDict(
            [
                ("full", self.student_joint_ids.to(device=self.device, dtype=torch.long)),
                ("base", torch.as_tensor(self.ov_env._robot_base_dof_idx, device=self.device, dtype=torch.long)),
                ("arm", torch.as_tensor(self.ov_env._robot_arm_dof_idx, device=self.device, dtype=torch.long)),
                ("hand", torch.as_tensor(self.ov_env._robot_finger_dof_idx, device=self.device, dtype=torch.long)),
            ]
        )

    def _parse_temporal_offset_s(self, offset_token):
        if not offset_token.endswith("ms") or not offset_token[:-2].isdigit():
            raise KeyError(f"Unsupported temporal offset '{offset_token}'. Expected '<milliseconds>ms'.")
        offset_ms = int(offset_token[:-2])
        if offset_ms <= 0:
            raise ValueError("Temporal derived state offsets must be positive.")
        return offset_ms / 1000.0

    def _parse_temporal_derived_state_key(self, key):
        if key.startswith("base_vel_"):
            offset_s = self._parse_temporal_offset_s(key[len("base_vel_"):])
            return {
                "kind": "base_vel",
                "component": "base_vel",
                "offset_s": offset_s,
                "indices": None,
                "dim": 3,
            }

        if key.startswith("q_"):
            parts = key[len("q_"):].split("_")
            if len(parts) != 2:
                return None
            component = self.action_component_aliases.get(parts[0])
            if component not in {"arm", "hand"}:
                raise KeyError(
                    f"Unsupported temporal q state key {key}. "
                    "Raw base pose history is disabled; use base_vel_<milliseconds>ms instead."
                )
            offset_s = self._parse_temporal_offset_s(parts[1])
            indices = self.proprio_component_history_indices[component]
            return {
                "kind": "q",
                "component": component,
                "offset_s": offset_s,
                "indices": indices,
                "dim": int(indices.numel()),
            }

        kind_prefixes = ("delta_target", "delta_q", "target_err")
        kind = None
        remainder = None
        for candidate in kind_prefixes:
            prefix = f"{candidate}_"
            if key.startswith(prefix):
                kind = candidate
                remainder = key[len(prefix):]
                break
        if kind is None:
            return None

        parts = remainder.split("_")
        if kind == "target_err":
            if len(parts) == 1:
                offset_s = None
            elif len(parts) == 2:
                offset_s = self._parse_temporal_offset_s(parts[1])
            else:
                raise KeyError(
                    f"Unsupported temporal state key {key}. "
                    "Expected target_err_<component> or target_err_<component>_<milliseconds>ms."
                )
        else:
            if len(parts) != 2:
                raise KeyError(
                    f"Unsupported temporal state key {key}. Expected {kind}_<component>_<milliseconds>ms."
                )
            offset_s = self._parse_temporal_offset_s(parts[1])

        component = self.action_component_aliases.get(parts[0])

        indices = self.action_component_history_indices[component]
        dim = int(indices.numel())
        return {
            "kind": kind,
            "component": component,
            "offset_s": offset_s,
            "indices": indices,
            "dim": dim,
        }

    def _parse_aux_state_key(self, key):
        if key != "aux_handle_pos":
            return None
        return {
            "name": key,
            "dim": 3,
        }

    def _validate_temporal_history_buffer_shape(self, name, value, expected_dim):
        expected_shape = (self.num_envs, self.temporal_history_len, int(expected_dim))
        if value is None:
            raise RuntimeError(f"Temporal history buffer '{name}' is not initialized.")
        if value.ndim != 3 or tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"Temporal history buffer '{name}' must have shape {expected_shape}, got {tuple(value.shape)}."
            )

    def _build_initial_temporal_push_pull_belief(self, env_ids=None):
        if env_ids is None:
            num_envs = self.num_envs
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            num_envs = int(env_ids.numel())
        if num_envs <= 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        if self.push_pull_condition_enabled and self.push_pull_condition_source == "oracle":
            gt_belief = self._build_gt_push_pull_condition()
            return gt_belief if env_ids is None else gt_belief[env_ids]
        return torch.full((num_envs, 2), 0.5, dtype=torch.float32, device=self.device)

    def _get_temporal_aux_handle_from_policy_input(self, aux_input_vector):
        if not self.temporal_aux_handle_enabled:
            return None
        if "aux_handle_pos" not in self.aux_state_specs:
            raise RuntimeError(
                "temporal_state_encoders field 'aux_handle_pos' could not find aux_handle_pos in aux_state_specs."
            )
        if aux_input_vector is None or not self.aux_feedback_to_policy:
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        value = aux_input_vector[:, self.aux_state_specs["aux_handle_pos"]["slice"]]
        return value

    def _get_temporal_aux_handle_for_history(self):
        if not self.temporal_aux_handle_enabled:
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if "aux_handle_pos" not in self.aux_state_specs:
            raise RuntimeError(
                "temporal_state_encoders field 'aux_handle_pos' could not find aux_handle_pos in aux_state_specs."
            )
        if not self.aux_feedback_to_policy:
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if self.aux_buffer is None:
            raise RuntimeError(
                "temporal_state_encoders field 'aux_handle_pos' requires aux_buffer when aux_feedback_to_policy=true."
            )
        value = self.aux_buffer[:, self.aux_state_specs["aux_handle_pos"]["slice"]].detach().clone()
        return value

    def _get_temporal_push_pull_belief_from_policy_input(self, push_pull_cond):
        if not self.temporal_push_pull_belief_enabled:
            return None
        if not self.push_pull_condition_enabled:
            raise RuntimeError(
                "temporal_state_encoders field 'push_pull_belief' requires push_pull_condition.enabled=true."
            )
        return push_pull_cond.detach().clone()

    def _get_temporal_push_pull_belief_for_history(self):
        if self.latest_push_pull_belief_input is not None:
            return self.latest_push_pull_belief_input.detach().clone()
        return self._build_initial_temporal_push_pull_belief()

    def _reset_push_pull_belief_history_metrics(self):
        self.latest_push_pull_belief_hist_entropy_now = 0.0
        self.latest_push_pull_belief_hist_entropy_mean = 0.0

    def _init_history_buffers(self):
        self.temporal_dt_s = max(float(getattr(self.ov_env, "dt", 1.0 / 15.0)), 1e-6)
        self.temporal_history_len = max(
            2,
            int(math.ceil(max(0.0, self.max_temporal_history_s) / self.temporal_dt_s)) + 4,
        )
        self.temporal_current_time_s = float(self.resume_iteration) * self.temporal_dt_s
        self.temporal_q_history_dim = int(self.ov_env.robot.data.joint_pos.shape[-1])
        self.temporal_time_history = torch.full(
            (self.num_envs, self.temporal_history_len),
            self.temporal_current_time_s,
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal_q_history = torch.zeros(
            (self.num_envs, self.temporal_history_len, self.temporal_q_history_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal_target_history = torch.zeros(
            (self.num_envs, self.temporal_history_len, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal_base_vel_history = torch.zeros(
            (self.num_envs, self.temporal_history_len, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal_aux_handle_history = torch.zeros(
            (self.num_envs, self.temporal_history_len, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal_push_pull_belief_history = torch.zeros(
            (self.num_envs, self.temporal_history_len, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self._validate_temporal_history_buffer_shape("temporal_aux_handle_history", self.temporal_aux_handle_history, 3)
        self._validate_temporal_history_buffer_shape(
            "temporal_push_pull_belief_history",
            self.temporal_push_pull_belief_history,
            2,
        )

    def _iteration_to_time_s(self, iteration):
        return float(iteration) * float(self.temporal_dt_s)

    def _get_current_time_s(self):
        return float(self.temporal_current_time_s)

    def _reset_observation_lag_stats(self):
        self.latest_obs_lag_enabled = 0.0
        self.latest_obs_lag_mean_ms = 0.0
        self.latest_obs_lag_min_ms = 0.0
        self.latest_obs_lag_max_ms = 0.0
        self.latest_obs_lag_effective_age_ms_by_timestamp = OrderedDict()

    def _is_observation_lag_active(self):
        if not self.observation_lag_enabled or not self.observation_lag_apply_to_proprio:
            return False
        return not self.play_policy

    def _merge_unique_offsets_s(self, *offset_sequences):
        merged = OrderedDict()
        for offsets in offset_sequences:
            for offset_s in offsets:
                merged[float(offset_s)] = None
        return tuple(merged.keys())

    def _sample_observation_lag_steps(self, offsets_s):
        if not offsets_s:
            raise RuntimeError("Observation lag sampling requires at least one requested offset.")
        num_offsets = len(offsets_s)
        nominal_offsets_ms = torch.as_tensor(offsets_s, dtype=torch.float32, device=self.device) * 1000.0
        sample_shape = (
            self.num_envs if self.observation_lag_per_env else 1,
            num_offsets if self.observation_lag_per_timestamp else 1,
        )
        if self.observation_lag_max_jitter_ms > 0:
            jitter_ms = torch.randint(
                low=-self.observation_lag_max_jitter_ms,
                high=self.observation_lag_max_jitter_ms + 1,
                size=sample_shape,
                device=self.device,
            ).to(torch.float32)
        else:
            jitter_ms = torch.zeros(sample_shape, dtype=torch.float32, device=self.device)
        if not self.observation_lag_per_env:
            jitter_ms = jitter_ms.expand(self.num_envs, -1)
        if not self.observation_lag_per_timestamp:
            jitter_ms = jitter_ms.expand(-1, num_offsets)

        effective_age_ms = nominal_offsets_ms.view(1, num_offsets) + jitter_ms
        max_available_age_ms = float(max(self.temporal_history_len - 1, 0)) * float(self.temporal_dt_s) * 1000.0
        if self.observation_lag_clamp_to_available_history:
            effective_age_ms = effective_age_ms.clamp(0.0, max_available_age_ms)
        else:
            effective_age_ms = effective_age_ms.clamp_min(0.0)

        dt_ms = max(float(self.temporal_dt_s) * 1000.0, 1.0e-6)
        effective_steps = torch.round(effective_age_ms / dt_ms).to(dtype=torch.long)
        effective_steps = effective_steps.clamp(0, self.temporal_history_len - 1)
        if torch.any(effective_steps < 0) or torch.any(effective_steps >= self.temporal_history_len):
            raise RuntimeError("Observation lag produced out-of-range temporal history indices.")
        effective_age_ms = effective_steps.to(dtype=torch.float32) * dt_ms
        return effective_steps, effective_age_ms

    def _record_observation_lag_stats(self, offsets_s, effective_age_ms):
        self._reset_observation_lag_stats()
        self.latest_obs_lag_enabled = 1.0
        nominal_offsets_ms = torch.as_tensor(offsets_s, dtype=torch.float32, device=effective_age_ms.device) * 1000.0
        lag_delta_ms = effective_age_ms - nominal_offsets_ms.view(1, -1)
        self.latest_obs_lag_mean_ms = float(lag_delta_ms.mean().detach().cpu().item())
        self.latest_obs_lag_min_ms = float(lag_delta_ms.min().detach().cpu().item())
        self.latest_obs_lag_max_ms = float(lag_delta_ms.max().detach().cpu().item())
        self.latest_obs_lag_effective_age_ms_by_timestamp = OrderedDict(
            (
                int(round(float(offset_s) * 1000.0)),
                float(effective_age_ms[:, idx].mean().detach().cpu().item()),
            )
            for idx, offset_s in enumerate(offsets_s)
        )

    def _get_temporal_sample_from_cache(self, sample_cache, sample_key, offset_s):
        if sample_cache is None:
            raise RuntimeError("Temporal sample cache is required.")
        offset_to_index = sample_cache["offset_to_index"]
        if float(offset_s) not in offset_to_index:
            raise RuntimeError(
                f"Temporal sample cache does not include requested offset {float(offset_s):.4f}s. "
                f"Available offsets: {list(offset_to_index.keys())}."
            )
        return sample_cache[sample_key][:, offset_to_index[float(offset_s)], :]

    def _build_temporal_sample_cache(
        self,
        q_pos,
        target_t,
        base_vel,
        offsets_s,
        apply_observation_lag=False,
        aux_handle_pos=None,
        push_pull_belief=None,
    ):
        if not offsets_s:
            return None

        offsets_s = tuple(float(offset_s) for offset_s in offsets_s)
        offset_to_index = {offset_s: idx for idx, offset_s in enumerate(offsets_s)}
        include_aux_handle = self.temporal_aux_handle_enabled
        include_push_pull_belief = self.temporal_push_pull_belief_enabled
        if include_aux_handle:
            self._validate_temporal_history_buffer_shape("temporal_aux_handle_history", self.temporal_aux_handle_history, 3)
        if include_push_pull_belief:
            self._validate_temporal_history_buffer_shape(
                "temporal_push_pull_belief_history",
                self.temporal_push_pull_belief_history,
                2,
            )
        if apply_observation_lag:
            effective_steps, effective_age_ms = self._sample_observation_lag_steps(offsets_s)
            q_samples_full = self._gather_temporal_values(self.temporal_q_history, effective_steps)
            target_samples = self._gather_temporal_values(self.temporal_target_history, effective_steps)
            base_vel_samples = self._gather_temporal_values(self.temporal_base_vel_history, effective_steps)
            aux_handle_samples = (
                self._gather_temporal_values(self.temporal_aux_handle_history, effective_steps)
                if include_aux_handle
                else None
            )
            push_pull_belief_samples = (
                self._gather_temporal_values(self.temporal_push_pull_belief_history, effective_steps)
                if include_push_pull_belief
                else None
            )
        else:
            nonzero_offsets = [offset_s for offset_s in offsets_s if abs(offset_s) > 1.0e-9]
            q_history_by_offset = self._sample_temporal_history_offsets(self.temporal_q_history, nonzero_offsets)
            target_history_by_offset = self._sample_temporal_history_offsets(self.temporal_target_history, nonzero_offsets)
            base_vel_history_by_offset = self._sample_temporal_history_offsets(self.temporal_base_vel_history, nonzero_offsets)
            aux_handle_history_by_offset = (
                self._sample_temporal_history_offsets(self.temporal_aux_handle_history, nonzero_offsets)
                if include_aux_handle
                else {}
            )
            push_pull_belief_history_by_offset = (
                self._sample_temporal_history_offsets(self.temporal_push_pull_belief_history, nonzero_offsets)
                if include_push_pull_belief
                else {}
            )
            q_samples_full = torch.stack(
                [q_pos if abs(offset_s) <= 1.0e-9 else q_history_by_offset[offset_s] for offset_s in offsets_s],
                dim=1,
            )
            target_samples = torch.stack(
                [target_t if abs(offset_s) <= 1.0e-9 else target_history_by_offset[offset_s] for offset_s in offsets_s],
                dim=1,
            )
            base_vel_samples = torch.stack(
                [base_vel if abs(offset_s) <= 1.0e-9 else base_vel_history_by_offset[offset_s] for offset_s in offsets_s],
                dim=1,
            )
            aux_handle_samples = None
            if include_aux_handle:
                aux_handle_samples = torch.stack(
                    [
                        aux_handle_pos
                        if abs(offset_s) <= 1.0e-9
                        else aux_handle_history_by_offset[offset_s]
                        for offset_s in offsets_s
                    ],
                    dim=1,
                )
            push_pull_belief_samples = None
            if include_push_pull_belief:
                push_pull_belief_samples = torch.stack(
                    [
                        push_pull_belief
                        if abs(offset_s) <= 1.0e-9
                        else push_pull_belief_history_by_offset[offset_s]
                        for offset_s in offsets_s
                    ],
                    dim=1,
                )
            effective_age_ms = (
                torch.as_tensor(offsets_s, dtype=torch.float32, device=self.device).view(1, -1) * 1000.0
            ).expand(self.num_envs, -1)

        q_samples_control = q_samples_full[:, :, self.student_joint_ids]
        target_err_samples = target_samples - q_samples_control
        if q_samples_full.ndim != 3 or target_samples.ndim != 3 or base_vel_samples.ndim != 3:
            raise RuntimeError("Temporal sample cache tensors must all be rank-3.")
        if include_aux_handle:
            if aux_handle_samples is None or aux_handle_samples.ndim != 3 or aux_handle_samples.shape[-1] != 3:
                raise RuntimeError(
                    "Temporal aux_handle_pos cache must have shape [N, T, 3], "
                    f"got {None if aux_handle_samples is None else tuple(aux_handle_samples.shape)}."
                )
        if include_push_pull_belief:
            if (
                push_pull_belief_samples is None
                or push_pull_belief_samples.ndim != 3
                or push_pull_belief_samples.shape[-1] != 2
            ):
                raise RuntimeError(
                    "Temporal push_pull_belief cache must have shape [N, T, 2], "
                    f"got {None if push_pull_belief_samples is None else tuple(push_pull_belief_samples.shape)}."
                )

        sample_cache = {
            "offsets_s": offsets_s,
            "offset_to_index": offset_to_index,
            "q_full": q_samples_full,
            "q_control": q_samples_control,
            "target": target_samples,
            "target_err": target_err_samples,
            "base_vel": base_vel_samples,
            "effective_age_ms": effective_age_ms,
        }
        if include_aux_handle:
            sample_cache["aux_handle_pos"] = aux_handle_samples
        if include_push_pull_belief:
            sample_cache["push_pull_belief"] = push_pull_belief_samples
        return sample_cache

    def _gather_temporal_values(self, value_history, indices):
        expanded_values = value_history.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
        gather_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, value_history.shape[-1])
        return torch.gather(expanded_values, dim=2, index=gather_indices).squeeze(2)

    def _sample_temporal_history_offsets(self, value_history, offsets_s):
        """
        Sample timestamped history for multiple offsets in one batched pass.
        Histories are stored newest first. If a requested time is outside the
        stored range, return the closest available value for that offset.
        """
        if not offsets_s:
            return {}
        if value_history is None or self.temporal_time_history is None:
            raise RuntimeError("Temporal history buffers are not initialized.")
        if value_history.ndim != 3 or self.temporal_time_history.ndim != 2:
            raise RuntimeError(
                "Expected value_history [N, H, D] and time_history [N, H], got "
                f"{tuple(value_history.shape)} and {tuple(self.temporal_time_history.shape)}."
            )

        num_envs, history_len, _ = value_history.shape
        offset_tensor = torch.as_tensor(offsets_s, dtype=torch.float32, device=value_history.device)
        query = self._get_current_time_s() - offset_tensor
        time_delta = torch.abs(self.temporal_time_history.unsqueeze(1) - query.view(1, -1, 1))
        nearest_idx = torch.argmin(time_delta, dim=2)
        nearest_value = self._gather_temporal_values(value_history, nearest_idx)
        if history_len < 2:
            return {float(offset): nearest_value[:, idx, :] for idx, offset in enumerate(offsets_s)}

        t_new = self.temporal_time_history[:, :-1]
        t_old = self.temporal_time_history[:, 1:]
        pair_mask = (t_new.unsqueeze(1) >= query.view(1, -1, 1)) & (
            t_old.unsqueeze(1) <= query.view(1, -1, 1)
        )
        has_pair = torch.any(pair_mask, dim=2)
        pair_idx = pair_mask.to(torch.long).argmax(dim=2)
        new_idx = pair_idx
        old_idx = pair_idx + 1

        value_new = self._gather_temporal_values(value_history, new_idx)
        value_old = self._gather_temporal_values(value_history, old_idx)
        time_new = torch.gather(self.temporal_time_history, dim=1, index=new_idx)
        time_old = torch.gather(self.temporal_time_history, dim=1, index=old_idx)
        alpha = ((query.view(1, -1) - time_old) / (time_new - time_old + 1.0e-6)).clamp(0.0, 1.0)
        alpha = alpha.unsqueeze(-1)
        interpolated = (1.0 - alpha) * value_old + alpha * value_new
        samples = torch.where(has_pair.unsqueeze(-1), interpolated, nearest_value)
        return {float(offset): samples[:, idx, :] for idx, offset in enumerate(offsets_s)}

    def _push_temporal_history(self, timestamp, q, target, base_vel, aux_handle_pos=None, push_pull_belief=None, env_ids=None):
        if self.temporal_time_history is None:
            return
        self._validate_temporal_history_buffer_shape("temporal_aux_handle_history", self.temporal_aux_handle_history, 3)
        self._validate_temporal_history_buffer_shape(
            "temporal_push_pull_belief_history",
            self.temporal_push_pull_belief_history,
            2,
        )
        if q.ndim != 2 or target.ndim != 2 or base_vel.ndim != 2:
            raise RuntimeError(
                "Expected q, target, and base_vel to be rank-2, got "
                f"{tuple(q.shape)}, {tuple(target.shape)}, and {tuple(base_vel.shape)}."
            )
        if aux_handle_pos is None:
            aux_handle_pos = self._get_temporal_aux_handle_for_history()
        if push_pull_belief is None:
            push_pull_belief = self._get_temporal_push_pull_belief_for_history()

        if env_ids is None:
            if self.temporal_history_len > 1:
                self.temporal_time_history[:, 1:] = self.temporal_time_history[:, :-1].clone()
                self.temporal_q_history[:, 1:, :] = self.temporal_q_history[:, :-1, :].clone()
                self.temporal_target_history[:, 1:, :] = self.temporal_target_history[:, :-1, :].clone()
                self.temporal_base_vel_history[:, 1:, :] = self.temporal_base_vel_history[:, :-1, :].clone()
                self.temporal_aux_handle_history[:, 1:, :] = self.temporal_aux_handle_history[:, :-1, :].clone()
                self.temporal_push_pull_belief_history[:, 1:, :] = (
                    self.temporal_push_pull_belief_history[:, :-1, :].clone()
                )
            self.temporal_time_history[:, 0] = float(timestamp)
            self.temporal_q_history[:, 0, :] = q
            self.temporal_target_history[:, 0, :] = target
            self.temporal_base_vel_history[:, 0, :] = base_vel
            self.temporal_aux_handle_history[:, 0, :] = aux_handle_pos
            self.temporal_push_pull_belief_history[:, 0, :] = push_pull_belief
            return

        if env_ids.numel() == 0:
            return
        if self.temporal_history_len > 1:
            self.temporal_time_history[env_ids, 1:] = self.temporal_time_history[env_ids, :-1].clone()
            self.temporal_q_history[env_ids, 1:, :] = self.temporal_q_history[env_ids, :-1, :].clone()
            self.temporal_target_history[env_ids, 1:, :] = self.temporal_target_history[env_ids, :-1, :].clone()
            self.temporal_base_vel_history[env_ids, 1:, :] = self.temporal_base_vel_history[
                env_ids, :-1, :
            ].clone()
            self.temporal_aux_handle_history[env_ids, 1:, :] = self.temporal_aux_handle_history[
                env_ids, :-1, :
            ].clone()
            self.temporal_push_pull_belief_history[env_ids, 1:, :] = self.temporal_push_pull_belief_history[
                env_ids, :-1, :
            ].clone()
        self.temporal_time_history[env_ids, 0] = float(timestamp)
        self.temporal_q_history[env_ids, 0, :] = q[env_ids]
        self.temporal_target_history[env_ids, 0, :] = target[env_ids]
        self.temporal_base_vel_history[env_ids, 0, :] = base_vel[env_ids]
        self.temporal_aux_handle_history[env_ids, 0, :] = aux_handle_pos[env_ids]
        self.temporal_push_pull_belief_history[env_ids, 0, :] = push_pull_belief[env_ids]

    def _seed_temporal_histories(self, env_ids=None):
        q = self._get_student_proprio_vector().detach()
        target = self._get_implemented_action_vector().detach()
        base_vel = self._get_student_base_velocity_vector().detach()
        timestamp = self._get_current_time_s()
        aux_handle = self._build_seed_temporal_aux_handle()
        push_pull_belief = self._build_initial_temporal_push_pull_belief()
        self._validate_temporal_history_buffer_shape("temporal_aux_handle_history", self.temporal_aux_handle_history, 3)
        self._validate_temporal_history_buffer_shape(
            "temporal_push_pull_belief_history",
            self.temporal_push_pull_belief_history,
            2,
        )

        if env_ids is None:
            self.temporal_time_history[:] = timestamp
            self.temporal_q_history[:] = q.unsqueeze(1).expand(-1, self.temporal_history_len, -1)
            self.temporal_target_history[:] = target.unsqueeze(1).expand(-1, self.temporal_history_len, -1)
            self.temporal_base_vel_history[:] = base_vel.unsqueeze(1).expand(-1, self.temporal_history_len, -1)
            self.temporal_aux_handle_history[:] = aux_handle.unsqueeze(1).expand(-1, self.temporal_history_len, -1)
            self.temporal_push_pull_belief_history[:] = push_pull_belief.unsqueeze(1).expand(
                -1, self.temporal_history_len, -1
            )
            return

        if env_ids.numel() == 0:
            return
        self.temporal_time_history[env_ids] = timestamp
        self.temporal_q_history[env_ids] = q[env_ids].unsqueeze(1).expand(-1, self.temporal_history_len, -1)
        self.temporal_target_history[env_ids] = target[env_ids].unsqueeze(1).expand(-1, self.temporal_history_len, -1)
        self.temporal_base_vel_history[env_ids] = base_vel[env_ids].unsqueeze(1).expand(
            -1, self.temporal_history_len, -1
        )
        self.temporal_aux_handle_history[env_ids] = aux_handle[env_ids].unsqueeze(1).expand(
            -1, self.temporal_history_len, -1
        )
        self.temporal_push_pull_belief_history[env_ids] = self._build_initial_temporal_push_pull_belief(
            env_ids
        ).unsqueeze(1).expand(-1, self.temporal_history_len, -1)

    def _get_student_proprio_vector(self):
        get_student_joint_pos_obs = getattr(self.ov_env, "get_student_joint_pos_obs", None)
        if callable(get_student_joint_pos_obs):
            q_pos = get_student_joint_pos_obs(use_noise=True)
        else:
            q_pos = self.ov_env.robot.data.joint_pos
        if q_pos.ndim != 2:
            raise RuntimeError(f"Expected joint_pos to be rank-2, got shape {tuple(q_pos.shape)}.")
        return q_pos

    def _get_base_yaw(self):
        return self.ov_env.robot.data.joint_pos[:, self.ov_env._robot_base_rot_dof_idx].squeeze(-1)

    def _env_base_vector_to_robot_frame(self, base_vector):
        if base_vector.ndim != 2 or base_vector.shape[-1] != 3:
            raise RuntimeError(f"Expected base vector shape [N, 3], got {tuple(base_vector.shape)}.")

        yaw = self._get_base_yaw().to(base_vector)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx_world = base_vector[:, self.base_action_xy_local_idx[0]]
        vy_world = base_vector[:, self.base_action_xy_local_idx[1]]
        wz_robot = base_vector[:, self.base_action_rot_local_idx]
        vx_robot = cos_yaw * vx_world + sin_yaw * vy_world
        vy_robot = -sin_yaw * vx_world + cos_yaw * vy_world
        return torch.stack((vx_robot, vy_robot, wz_robot), dim=-1)

    def _robot_base_vector_to_env_frame(self, base_vector_robot):
        if base_vector_robot.ndim != 2 or base_vector_robot.shape[-1] != 3:
            raise RuntimeError(f"Expected robot-frame base vector shape [N, 3], got {tuple(base_vector_robot.shape)}.")

        yaw = self._get_base_yaw().to(base_vector_robot)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx_robot = base_vector_robot[:, 0]
        vy_robot = base_vector_robot[:, 1]
        wz_robot = base_vector_robot[:, 2]
        vx_world = cos_yaw * vx_robot - sin_yaw * vy_robot
        vy_world = sin_yaw * vx_robot + cos_yaw * vy_robot

        base_vector_env = torch.zeros_like(base_vector_robot)
        base_vector_env[:, self.base_action_xy_local_idx[0]] = vx_world
        base_vector_env[:, self.base_action_xy_local_idx[1]] = vy_world
        base_vector_env[:, self.base_action_rot_local_idx] = wz_robot
        return base_vector_env

    def _env_actions_to_student_actions(self, env_actions):
        if env_actions.ndim != 2:
            raise RuntimeError(f"Expected env action tensor to be rank-2, got shape {tuple(env_actions.shape)}.")
        if env_actions.shape[-1] != self.num_actions:
            raise RuntimeError(
                f"Expected env action shape [N, {self.num_actions}], "
                f"got {tuple(env_actions.shape)}."
            )

        student_actions = env_actions.clone()
        # Only the base action changes frame, not magnitude: env [wz, vx_w, vy_w]
        # normalized delta-action becomes student [vx_robot, vy_robot, wz_robot] in the
        # same normalized [-1, 1] convention (robot frame). base_action_scale is applied
        # only at the boundary (env _scale_actions in sim, * real_world_base_scale at
        # deploy), so it stays a real, decoupled step-size knob instead of cancelling in
        # the integration round-trip. Arm/hand stay in the env's normalized convention.
        student_actions[:, :3] = self._env_base_vector_to_robot_frame(env_actions[:, :3])
        return student_actions

    def _student_actions_to_env_actions(self, student_actions):
        if student_actions.ndim != 2 or student_actions.shape[-1] != self.num_actions:
            raise RuntimeError(
                f"Expected student action shape [N, {self.num_actions}], got {tuple(student_actions.shape)}."
            )

        env_actions = student_actions.clone()
        # Base output is a normalized [-1, 1] robot-frame action; only rotate it into the
        # env [wz, vx_w, vy_w] frame. base_action_scale and dt are applied by the env in
        # _scale_actions/_pre_physics_step. Keep arm/hand untouched.
        env_actions[:, :3] = self._robot_base_vector_to_env_frame(student_actions[:, :3])
        return env_actions.clamp(-1.0, 1.0)

    def _get_student_base_velocity_vector(self):
        # joint_vel is already per-second in env/base-joint order [wz, vx_w, vy_w].
        # Convert it to the deployment-facing robot-frame order [vx_robot, vy_robot, wz_robot].
        get_student_base_joint_vel_obs = getattr(self.ov_env, "get_student_base_joint_vel_obs", None)
        if callable(get_student_base_joint_vel_obs):
            base_joint_vel = get_student_base_joint_vel_obs(use_noise=True)
        else:
            base_joint_vel = self.ov_env.robot.data.joint_vel[:, self.ov_env._robot_base_dof_idx]
        if base_joint_vel.ndim != 2 or base_joint_vel.shape[-1] != 3:
            raise RuntimeError(f"Expected base joint velocity shape [N, 3], got {tuple(base_joint_vel.shape)}.")
        base_vel = self._env_base_vector_to_robot_frame(base_joint_vel)
        if base_vel.ndim != 2 or base_vel.shape[-1] != 3:
            raise RuntimeError(f"Expected base velocity shape [N, 3], got {tuple(base_vel.shape)}.")
        return base_vel

    def _get_implemented_action_vector(self):
        return self._get_teacher_prev_action_vector()

    def _get_teacher_prev_action_vector(self):
        # Temporal target features use the actual joint-position targets sent to
        # the PD controller, not the normalized delta policy actions.
        pd_targets = self.ov_env.applied_robot_dof_targets
        if pd_targets.ndim != 2:
            raise RuntimeError(f"Expected PD target tensor to be rank-2, got shape {tuple(pd_targets.shape)}.")
        if pd_targets.shape[-1] == self.num_actions:
            return pd_targets
        full_robot_action_dim = int(self.ov_env._robot_dof_idx.numel())
        if pd_targets.shape[-1] == full_robot_action_dim:
            return pd_targets[:, self.student_target_indices_in_env]
        raise RuntimeError(
            f"Unexpected PD target width {pd_targets.shape[-1]}; expected {self.num_actions} or "
            f"{full_robot_action_dim}."
        )

    def _build_temporal_derived_state_values(self, q_pos, target_t, base_vel, sample_cache=None):
        if not self.temporal_derived_state_specs:
            return {}

        q_t = q_pos[:, self.student_joint_ids]
        target_err = target_t - q_t
        required_offsets = self._merge_unique_offsets_s(
            {
                float(spec["offset_s"])
                for spec in self.temporal_derived_state_specs.values()
                if spec["offset_s"] is not None
            },
            (0.0,) if self._is_observation_lag_active() else (),
        )
        if sample_cache is None:
            sample_cache = self._build_temporal_sample_cache(
                q_pos,
                target_t,
                base_vel,
                required_offsets,
                apply_observation_lag=self._is_observation_lag_active(),
            )

        values_by_key = {}
        for key, spec in self.temporal_derived_state_specs.items():
            kind = spec["kind"]
            offset_s = spec["offset_s"]
            if kind == "q":
                full_value = self._get_temporal_sample_from_cache(sample_cache, "q_full", offset_s)
            elif kind == "base_vel":
                full_value = self._get_temporal_sample_from_cache(sample_cache, "base_vel", offset_s)
            elif kind == "target_err":
                if offset_s is None:
                    if self._is_observation_lag_active():
                        full_value = self._get_temporal_sample_from_cache(sample_cache, "target_err", 0.0)
                    else:
                        full_value = target_err
                else:
                    full_value = self._get_temporal_sample_from_cache(sample_cache, "target_err", offset_s)
            elif kind == "delta_q":
                full_value = q_t - self._get_temporal_sample_from_cache(sample_cache, "q_control", offset_s)
            elif kind == "delta_target":
                full_value = target_t - self._get_temporal_sample_from_cache(sample_cache, "target", offset_s)
            else:
                raise KeyError(f"Unsupported temporal derived state kind '{kind}' for key '{key}'.")

            if spec["indices"] is None:
                values_by_key[key] = full_value
            else:
                values_by_key[key] = full_value[:, spec["indices"]]
        return values_by_key

    def _extract_proprio_temporal_field_value(self, field_name, actual_state_key, sample_cache, timestamp_s):
        if actual_state_key == "q_arm":
            full_value = self._get_temporal_sample_from_cache(sample_cache, "q_full", timestamp_s)
            value = full_value[:, self.ov_env._robot_arm_dof_idx]
        elif actual_state_key == "q_hand":
            full_value = self._get_temporal_sample_from_cache(sample_cache, "q_full", timestamp_s)
            value = full_value[:, self.ov_env._robot_finger_dof_idx]
        elif actual_state_key == "base_vel":
            value = self._get_temporal_sample_from_cache(sample_cache, "base_vel", timestamp_s)
        elif actual_state_key in {"target_err_arm", "tracking_err_arm"}:
            full_value = self._get_temporal_sample_from_cache(sample_cache, "target_err", timestamp_s)
            value = full_value[:, self.action_component_history_indices["arm"]]
        elif actual_state_key in {"target_err_hand", "tracking_err_hand"}:
            full_value = self._get_temporal_sample_from_cache(sample_cache, "target_err", timestamp_s)
            value = full_value[:, self.action_component_history_indices["hand"]]
        elif actual_state_key == "aux_handle_pos":
            value = self._get_temporal_sample_from_cache(sample_cache, "aux_handle_pos", timestamp_s)
        elif field_name == "push_pull_belief" or actual_state_key == self.push_pull_condition_obs_key:
            value = self._get_temporal_sample_from_cache(sample_cache, "push_pull_belief", timestamp_s)
        else:
            raise RuntimeError(
                f"Unsupported temporal_state_encoders field mapping '{field_name}' -> '{actual_state_key}'."
            )
        expected_dim = int(self.proprio_temporal_field_dims[field_name])
        if value.ndim != 2 or value.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Temporal field '{field_name}' at {timestamp_s:.3f}s must have shape [B, {expected_dim}], "
                f"got {tuple(value.shape)}."
            )
        return value

    def _record_push_pull_belief_history_metrics(self, sample_cache):
        self._reset_push_pull_belief_history_metrics()
        if not self.temporal_push_pull_belief_enabled:
            return
        if sample_cache is None or "push_pull_belief" not in sample_cache:
            raise RuntimeError(
                "temporal_state_encoders field 'push_pull_belief' requires push_pull_belief entries in the temporal sample cache."
            )
        belief_samples = sample_cache["push_pull_belief"]
        if belief_samples.ndim != 3 or belief_samples.shape[-1] != 2:
            raise RuntimeError(
                "push_pull_belief temporal samples must have shape [N, T, 2], "
                f"got {tuple(belief_samples.shape)}."
            )
        belief_probs = belief_samples.clamp_min(1.0e-6)
        entropy = -(belief_samples * torch.log(belief_probs)).sum(dim=-1)
        offset_to_index = sample_cache["offset_to_index"]
        idx_now = offset_to_index.get(0.0)
        if idx_now is None:
            raise RuntimeError("push_pull_belief temporal metrics require a 0.0s timestamp in the sample cache.")
        self.latest_push_pull_belief_hist_entropy_now = float(entropy[:, idx_now].mean().detach().cpu().item())
        self.latest_push_pull_belief_hist_entropy_mean = float(entropy.mean().detach().cpu().item())

    def _build_proprio_temporal_obs(self, sample_cache):
        if not self.proprio_temporal_enabled:
            return None
        if sample_cache is None:
            raise RuntimeError("temporal_state_encoders require a temporal sample cache.")

        proprio_temporal_obs = OrderedDict()
        if not self.temporal_state_uses_field_shared_encoders:
            for field_name, actual_state_key in self.proprio_temporal_field_state_keys.items():
                per_timestamp_values = [
                    self._extract_proprio_temporal_field_value(
                        field_name=field_name,
                        actual_state_key=actual_state_key,
                        sample_cache=sample_cache,
                        timestamp_s=timestamp_s,
                    )
                    for timestamp_s in self.proprio_temporal_timestamps_s
                ]
                field_tensor = torch.stack(per_timestamp_values, dim=1)
                expected_dim = int(self.proprio_temporal_field_dims[field_name])
                if field_tensor.ndim != 3 or field_tensor.shape[1] != len(self.proprio_temporal_timestamps_s):
                    raise RuntimeError(
                        f"Expected temporal field '{field_name}' to have shape "
                        f"[B, {len(self.proprio_temporal_timestamps_s)}, {expected_dim}], got {tuple(field_tensor.shape)}."
                    )
                if field_tensor.shape[-1] != expected_dim:
                    raise RuntimeError(
                        f"Temporal field '{field_name}' must have feature dim {expected_dim}, "
                        f"got {field_tensor.shape[-1]}."
                    )
                proprio_temporal_obs[field_name] = field_tensor
            return proprio_temporal_obs

        for field_name, actual_state_key in self.proprio_temporal_field_state_keys.items():
            obs_keys = self.proprio_temporal_field_obs_keys.get(field_name)
            if obs_keys is None or len(obs_keys) != len(self.proprio_temporal_timestamps_s):
                raise RuntimeError(
                    f"Temporal field '{field_name}' must expose one observation key per timestamp; got {obs_keys}."
                )
            for obs_key, timestamp_s in zip(obs_keys, self.proprio_temporal_timestamps_s):
                proprio_temporal_obs[obs_key] = self._extract_proprio_temporal_field_value(
                    field_name=field_name,
                    actual_state_key=actual_state_key,
                    sample_cache=sample_cache,
                    timestamp_s=timestamp_s,
                )
        return proprio_temporal_obs

    def _get_handle_position_base(self):
        getter = getattr(self.ov_env, "get_handle_position_in_base_frame", None)
        if callable(getter):
            return getter()
        raise RuntimeError(
            "Expected environment to expose get_handle_position_in_base_frame() "
            "for aux handle position prediction."
        )

    def _get_closed_handle_position_base(self):
        # Closed-door (door joint=0) handle position in the CURRENT robot base frame. Simulates a
        # one-shot SAM3 detection of the closed handle, re-expressed in the base frame as it moves.
        getter = getattr(self.ov_env, "get_closed_handle_position_in_base_frame", None)
        if callable(getter):
            return getter()
        raise RuntimeError(
            "Expected environment to expose get_closed_handle_position_in_base_frame() "
            "for closed_door_base aux handle input."
        )

    def _build_closed_door_aux_input_vector(self):
        # Non-recurrent aux input: the closed-door handle anchor in the current base frame with fresh
        # per-step noise. Never derived from the aux prediction.
        closed_handle_base = self._get_closed_handle_position_base()
        aux_values = OrderedDict()
        aux_values["aux_handle_pos"] = self._apply_aux_handle_init_perturbation(closed_handle_base)
        aux_input_vector = self._stack_aux_state_values(aux_values)
        if aux_input_vector is None:
            raise RuntimeError("closed_door_base aux input could not be assembled.")
        return aux_input_vector.to(device=self.device, dtype=torch.float32)

    def _get_aux_state_values(self):
        if not self.has_aux_input:
            return OrderedDict()

        handle_pos_base = self._get_handle_position_base()
        aux_state_values = OrderedDict()
        if "aux_handle_pos" in self.aux_state_specs:
            aux_state_values["aux_handle_pos"] = handle_pos_base
        return aux_state_values

    def _stack_aux_state_values(self, aux_state_values):
        if not self.has_aux_input:
            return None
        aux_vector = torch.zeros((self.num_envs, self.aux_input_dim), dtype=torch.float32, device=self.device)
        for key, spec in self.aux_state_specs.items():
            aux_vector[:, spec["slice"]] = aux_state_values[key]
        return aux_vector

    def _aux_to_2d(self, aux_tensor):
        if aux_tensor is None:
            return None
        if aux_tensor.ndim == 3:
            return aux_tensor[:, 0, :]
        return aux_tensor

    def _decode_aux_prediction(self, aux_pred):
        return self._aux_to_2d(aux_pred)

    def _get_aux_target(self, current_abs_aux):
        return current_abs_aux

    def _parse_aux_handle_offset_bound(self, raw, name):
        # Scalar bound (meters) on the total Euclidean error of a handle-pose offset (noise or bias).
        # Returns a non-negative float, or None when disabled (zero / unset).
        if raw is None:
            return None
        value = float(raw)
        if value < 0.0:
            raise ValueError(f"dagger.{name} must be non-negative.")
        if value == 0.0:
            return None
        return value

    def _sample_isotropic_offset(self, shape, bound, dtype=torch.float32):
        # Random direction (uniform on the unit sphere) x magnitude in [0, bound], so the total
        # Euclidean error of each sampled offset is at most `bound`. `shape[-1]` is the vector dim.
        direction = torch.randn(shape, dtype=dtype, device=self.device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        magnitude = torch.rand(shape[:-1] + (1,), dtype=dtype, device=self.device) * float(bound)
        return direction * magnitude

    def _resample_aux_handle_bias(self, env_ids=None):
        # Per-episode constant handle-pose bias, redrawn for the resetting envs. When the bias is
        # disabled the buffer stays zeroed so _apply_aux_handle_init_perturbation is a no-op for it.
        if self.aux_handle_bias is None:
            return
        if self.aux_handle_init_bias_m is None:
            if env_ids is None:
                self.aux_handle_bias.zero_()
            elif env_ids.numel() > 0:
                self.aux_handle_bias[env_ids] = 0.0
            return
        if env_ids is None:
            self.aux_handle_bias[:] = self._sample_isotropic_offset(
                (self.num_envs, 3), self.aux_handle_init_bias_m
            )
        elif env_ids.numel() > 0:
            self.aux_handle_bias[env_ids] = self._sample_isotropic_offset(
                (int(env_ids.numel()), 3), self.aux_handle_init_bias_m
            )

    def _apply_aux_handle_init_perturbation(self, handle_pos):
        # Add the per-episode constant bias (if any) and fresh per-call isotropic noise (if any) to a
        # handle-pose tensor. Used only for policy INPUT seeds/anchors, never for the regression target.
        if handle_pos is None:
            return handle_pos
        perturbed = handle_pos
        if self.aux_handle_bias is not None and self.aux_handle_init_bias_m is not None:
            perturbed = perturbed + self.aux_handle_bias.to(dtype=handle_pos.dtype)
        if self.aux_handle_init_noise_m is not None:
            noise = self._sample_isotropic_offset(
                handle_pos.shape, self.aux_handle_init_noise_m, dtype=handle_pos.dtype
            )
            perturbed = perturbed + noise
        return perturbed

    def _build_seed_aux_buffer_values(self):
        # Value the aux feedback buffer is reset to at the first rollout step after each reset.
        # "zeros": classic behavior. "ground_truth": the (noised) sim handle pose, fed only on that
        # first step (predictions overwrite the buffer afterwards, so all later steps stay predicted).
        if self.aux_handle_init_source == "ground_truth":
            aux_values = self._get_aux_state_values()
            if "aux_handle_pos" in aux_values:
                aux_values = OrderedDict(aux_values)
                aux_values["aux_handle_pos"] = self._apply_aux_handle_init_perturbation(
                    aux_values["aux_handle_pos"]
                )
            seed = self._stack_aux_state_values(aux_values)
            if seed is None:
                raise RuntimeError(
                    "aux_handle_init_source='ground_truth' could not build a sim aux seed vector."
                )
            return seed.to(device=self.device, dtype=torch.float32)
        return torch.zeros((self.num_envs, self.aux_input_dim), dtype=torch.float32, device=self.device)

    def _build_seed_temporal_aux_handle(self):
        # Mirror the aux feedback buffer seed for the temporal aux_handle_pos history window so the
        # first post-reset step is consistent regardless of whether the temporal field is consumed.
        if (
            self.aux_handle_init_source == "ground_truth"
            and self.has_aux_input
            and "aux_handle_pos" in self.aux_state_specs
        ):
            value = self._get_aux_state_values().get("aux_handle_pos")
            if value is not None:
                value = self._apply_aux_handle_init_perturbation(value)
                return value.to(device=self.device, dtype=torch.float32)
        return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _seed_aux_buffer(self, env_ids=None):
        if self.aux_buffer is None:
            return
        seed = self._build_seed_aux_buffer_values()
        if env_ids is None:
            self.aux_buffer[:] = seed
            return
        if env_ids.numel() == 0:
            return
        self.aux_buffer[env_ids] = seed[env_ids]

    def _init_pointcloud_assets(self):
        asset_index_by_dir = {
            Path(asset_path).resolve().parent: idx for idx, asset_path in enumerate(door_asset_paths)
        }
        ref_motion_lib = getattr(self.ov_env, "ref_motion_lib", None)
        env_asset_indices = getattr(self.ov_env, "env_asset_indices", None)
        if ref_motion_lib is None:
            if env_asset_indices is None:
                raise RuntimeError("Play mode without reference motions requires env.env_asset_indices.")
            self.motion_to_asset_idx = None
            self.motion_family_ids = None
            self.env_motion_idx = None
            self.env_asset_idx = env_asset_indices.to(device=self.device, dtype=torch.long)
            self.env_family_ids = door_asset_family_ids.to(device=self.device, dtype=torch.long)[self.env_asset_idx]
        else:
            motion_to_asset_idx = []
            for motion_path in motion_traj_paths:
                motion_dir = Path(motion_path).resolve().parent
                if motion_dir not in asset_index_by_dir:
                    raise KeyError(f"Could not map motion file '{motion_path}' to a door asset path.")
                motion_to_asset_idx.append(asset_index_by_dir[motion_dir])
            self.motion_to_asset_idx = torch.tensor(motion_to_asset_idx, device=self.device, dtype=torch.long)

            env_motion_idx = ref_motion_lib.env_to_file_map.to(device=self.device, dtype=torch.long)
            self.env_motion_idx = env_motion_idx
            self.env_asset_idx = self.motion_to_asset_idx[env_motion_idx]
            if env_asset_indices is not None:
                env_asset_indices = env_asset_indices.to(device=self.device, dtype=torch.long)
                if not torch.equal(env_asset_indices, self.env_asset_idx):
                    raise RuntimeError("Door env asset indices and reference-motion asset indices are inconsistent.")
            self.motion_family_ids = motion_family_ids.to(device=self.device, dtype=torch.long)
            self.env_family_ids = self.motion_family_ids[env_motion_idx]
        expected_asset_family_ids = door_asset_family_ids.to(device=self.device, dtype=torch.long)[self.env_asset_idx]
        if not torch.equal(expected_asset_family_ids, self.env_family_ids):
            raise RuntimeError("Door asset family ids and motion family ids are inconsistent.")
        self._init_train_validation_split()
        self._init_push_pull_semantics_and_targets()
        self.family_env_ids = {
            int(family_id): torch.nonzero(self.env_family_ids == int(family_id), as_tuple=False).squeeze(-1)
            for family_id in range(len(DOOR_FAMILY_NAMES))
        }

        local_family_counts_tensor = torch.stack(
            [
                (self.env_family_ids == int(family_id)).sum()
                for family_id in range(len(DOOR_FAMILY_NAMES))
            ]
        ).to(device=self.device, dtype=torch.float64)
        global_family_counts_tensor = local_family_counts_tensor.clone()
        if self.use_ddp:
            dist.all_reduce(global_family_counts_tensor, op=dist.ReduceOp.SUM)

        family_counts = {
            family_name: int(global_family_counts_tensor[family_id].detach().cpu().item())
            for family_id, family_name in enumerate(DOOR_FAMILY_NAMES)
        }
        print(f"[INFO] Global door family env counts: {family_counts}")
        sample = []
        sample_count = min(12, int(self.num_envs))
        for env_id in range(sample_count):
            asset_idx = int(self.env_asset_idx[env_id].detach().cpu().item())
            family_id = int(self.env_family_ids[env_id].detach().cpu().item())
            asset_name = Path(door_asset_paths[asset_idx]).parent.name
            sample.append(f"env{env_id}:{DOOR_FAMILY_NAMES[family_id]}/{asset_name}")
        print("[INFO] Rank door family sample:", ", ".join(sample))
        self.env_board_bboxes = door_board_bboxes.to(device=self.device, dtype=torch.float32)[self.env_asset_idx]
        self.env_board_bboxes_link1 = door_board_bboxes_link1.to(device=self.device, dtype=torch.float32)[
            self.env_asset_idx
        ]
        # Full door outer bbox (frame + panel + handle) per env, used ONLY for wall placement so walls
        # sit outside the whole door (the frame is wider than the link_1 panel). The link_1 panel bbox
        # stays for door-hole aug etc.
        self.env_full_door_bboxes = door_full_door_bboxes.to(device=self.device, dtype=torch.float32)[
            self.env_asset_idx
        ]
        # We do not assume a fixed asset axis convention across all door families. Instead, infer a
        # stable local frame from the full door bbox extents (smallest -> thickness, middle -> width,
        # largest -> height). Shared with the offline tooling via DoorOpening.utils.wall_distractors.
        (
            self.wall_distractor_axis_order,
            self.wall_distractor_bbox_min_ordered,
            self.wall_distractor_bbox_max_ordered,
        ) = compute_wall_bbox_ordering(self.env_full_door_bboxes)
        unique_asset_idx = sorted(set(self.env_asset_idx.detach().cpu().tolist()))
        self.door_samplers = {
            idx: FrankaLeapSampler(
                door_asset_paths[idx],
                device=self.device,
                num_points=self.scene_door_pcd_num_points,
            )
            for idx in unique_asset_idx
        }
        door_geometry_aug_cfg = self.runtime_cfg.get("door_geometry_aug", {})
        for sampler in self.door_samplers.values():
            sampler.configure_door_geometry_aug(door_geometry_aug_cfg, device=self.device)
            sampler.set_door_geometry_aug_runtime_enabled(not self.play_policy)
        self.door_sampler_env_ids = {
            idx: torch.nonzero(self.env_asset_idx == int(idx), as_tuple=False).squeeze(-1)
            for idx in unique_asset_idx
        }
        self.door_link_pointclouds = {
            idx: build_first_visual_link_pointcloud_cache(sampler, link_names=("link_1", "link_2"), device=self.device)
            for idx, sampler in self.door_samplers.items()
        }
        if self.scene_robot_pcd_num_points is None:
            self.scene_robot_pcd_num_points = self.scene_door_pcd_num_points

        self.scene_robot_pcd_num_points = int(self.scene_robot_pcd_num_points)
        self.robot_sampler = FrankaLeapSampler(
            glorbot_urdf_path,
            device=self.device,
            num_points=self.scene_robot_pcd_num_points,
        )
        robot_sampler_joint_names = list(self.robot_sampler.robot.actuated_joint_names)
        robot_joint_ids, robot_joint_names = self.ov_env.robot.find_joints(robot_sampler_joint_names)
        self.robot_sampler_joint_ids = torch.tensor(robot_joint_ids, device=self.device, dtype=torch.long)
        robot_joint_name_to_idx = {name: idx for idx, name in enumerate(robot_joint_names)}
        self.robot_sampler_joint_reorder = [robot_joint_name_to_idx[name] for name in robot_sampler_joint_names]
        self.robot_link_pointclouds = build_first_visual_link_pointcloud_cache(self.robot_sampler, device=self.device)
        self.robot_sampler_body_indices = {}
        for link_name in self.robot_link_pointclouds.keys():
            body_ids = self.ov_env.robot.find_bodies(link_name)[0]
            if len(body_ids) == 0:
                continue
            self.robot_sampler_body_indices[link_name] = int(body_ids[0])
        self.robot_collision_checker = None
        self.robot_collision_checker_base_joint_indices = []
        if self.robot_pointcloud_filter_enabled:
            self.robot_collision_checker = GlorbotCollisionChecker(
                glorbot_urdf_path,
                device=self.device,
                input_joint_names=robot_sampler_joint_names,
            )
            self.robot_collision_checker_base_joint_indices = [
                idx
                for idx, joint_name in enumerate(robot_sampler_joint_names)
                if joint_name in {"base_x_joint", "base_y_joint", "base_rotation_joint"}
            ]

        self.robot_base_body_idx = int(self.ov_env._robot_base_body_link_idx)
        self.robot_palm_body_idx = int(self.ov_env._robot_key_body_idx[self.ov_env._robot_palm_id_in_key_body_idx])
        self.door_base_body_idx = int(self.ov_env._door_base_link_idx)
        self.door_link_body_indices = {
            link_name: int(self.ov_env._door_body_idx[self.ov_env.door_body_names.index(link_name)])
            for link_name in ("link_1", "link_2")
        }
        self.door_hole_link1_body_idx = int(self.door_link_body_indices["link_1"])
        # All wall-distractor geometry knobs are parsed once into a shared params object (see
        # DoorOpening.utils.wall_distractors); the sampling itself is delegated to that module so the
        # offline tooling can reuse the exact same logic without importing Isaac.
        self.wall_distractor_params = WallDistractorParams.from_cfg(
            self.wall_distractor_cfg, self.scene_door_pcd_num_points
        )
        self.wall_distractors_enabled = self.wall_distractor_params.enabled
        self.wall_distractor_num_points = self.wall_distractor_params.num_points
        self.wall_distractor_resample_each_step = self.wall_distractor_params.resample_each_step
        self._wall_distractor_local_points = None
        if (
            self.wall_distractors_enabled
            and self.wall_distractor_num_points > 0
            and not self.wall_distractor_resample_each_step
        ):
            self._wall_distractor_local_points = torch.zeros(
                (self.num_envs, self.wall_distractor_num_points, 3),
                dtype=torch.float32,
                device=self.device,
            )

        self.door_hole_aug_enabled = bool(self.door_hole_aug_cfg.get("enabled", False))
        self.door_hole_aug_env_prob = float(self.door_hole_aug_cfg.get("env_prob", 0.35))
        self.door_hole_aug_width_range_m = tuple(
            float(v) for v in self.door_hole_aug_cfg.get("width_range_m", [0.12, 1.60])
        )
        self.door_hole_aug_height_range_m = tuple(
            float(v) for v in self.door_hole_aug_cfg.get("height_range_m", [0.18, 2.20])
        )
        self.door_hole_aug_center_height_range_m = tuple(
            float(v) for v in self.door_hole_aug_cfg.get("center_height_range_m", [0.10, 1.90])
        )
        self.door_hole_aug_side_margin_range_m = tuple(
            float(v) for v in self.door_hole_aug_cfg.get("side_margin_range_m", [0.0, 0.18])
        )
        self.door_hole_aug_surface_eps_m = float(self.door_hole_aug_cfg.get("surface_eps_m", 0.03))
        if not 0.0 <= self.door_hole_aug_env_prob <= 1.0:
            raise ValueError("door_hole_aug.env_prob must be in [0, 1].")
        if self.door_hole_aug_surface_eps_m < 0.0:
            raise ValueError("door_hole_aug.surface_eps_m must be non-negative.")
        for range_name, value_range in (
            ("width_range_m", self.door_hole_aug_width_range_m),
            ("height_range_m", self.door_hole_aug_height_range_m),
            ("center_height_range_m", self.door_hole_aug_center_height_range_m),
            ("side_margin_range_m", self.door_hole_aug_side_margin_range_m),
        ):
            if len(value_range) != 2:
                raise ValueError(f"door_hole_aug.{range_name} must contain exactly two values.")
            if float(value_range[1]) < float(value_range[0]):
                raise ValueError(f"door_hole_aug.{range_name} must satisfy min <= max.")
        self.latest_door_hole_aug_stats = {}
        if self.door_hole_aug_enabled:
            print(
                "[INFO] door_hole_aug enabled: "
                f"env_prob={self.door_hole_aug_env_prob}, "
                f"width_range_m={self.door_hole_aug_width_range_m}, "
                f"height_range_m={self.door_hole_aug_height_range_m}, "
                f"center_height_range_m={self.door_hole_aug_center_height_range_m}, "
                f"side_margin_range_m={self.door_hole_aug_side_margin_range_m}, "
                f"surface_eps_m={self.door_hole_aug_surface_eps_m}"
            )

        self.robot_camera_body_idx = None
        self.sampler_camera_spec = None
        self._camera_mount_offset_quat_wxyz = None

        if self.pointcloud_source in {"sampler", "depth", "both"}:
            self.robot_camera_body_idx = int(self.ov_env.robot.find_bodies("x5_camera_link")[0][0])
            self.sampler_camera_spec = self._build_sampler_camera_spec()
            # The real depth sensor is the `cam` prim mounted on x5_camera_link via the CameraCfg
            # offset rotation (a -45deg roll that compensates for the 45deg-tilted realsense bracket
            # on the ARX x5 wrist). The x5_camera_link frame itself is NOT the optical frame. The
            # simulated renderer must apply the same mount offset, otherwise the rendered cloud is
            # rolled 45deg vs the real camera. Read it straight from the env cfg so it stays in sync.
            camera_mount_rot = tuple(self.ov_env.cfg.pointcloud_camera_cfg.offset.rot)  # (w, x, y, z)
            self._camera_mount_offset_quat_wxyz = torch.tensor(
                camera_mount_rot, device=self.device, dtype=torch.float32
            )

        self.robot_lidar_body_idx = None
        if self.pointcloud_source in {"lidar", "both"}:
            lidar_body_name = str(getattr(self.ov_env.cfg, "pointcloud_lidar_body_name", "lidar"))
            lidar_body_ids = self.ov_env.robot.find_bodies(lidar_body_name)[0]
            if len(lidar_body_ids) == 0:
                raise ValueError(f"Could not find lidar body '{lidar_body_name}' on robot articulation.")
            self.robot_lidar_body_idx = int(lidar_body_ids[0])

        if self.depth_cam_render_num_points is None:
            self.depth_cam_render_num_points = self.scene_door_pcd_num_points
        self.depth_cam_render_num_points = int(self.depth_cam_render_num_points)

        if self.lidar_num_points is None:
            self.lidar_num_points = self.scene_door_pcd_num_points
        self.lidar_num_points = int(self.lidar_num_points)

    @staticmethod
    def _is_validation_asset_name(asset_name):
        return str(asset_name).endswith("00")

    def _init_train_validation_split(self):
        validation_flags = []
        for asset_idx in self.env_asset_idx.detach().cpu().tolist():
            asset_name = Path(door_asset_paths[int(asset_idx)]).parent.name
            validation_flags.append(self._is_validation_asset_name(asset_name))

        self.validation_env_mask = torch.tensor(validation_flags, dtype=torch.bool, device=self.device)
        self.train_env_mask = ~self.validation_env_mask
        if not torch.any(self.train_env_mask):
            raise RuntimeError("Training split is empty. Validation asset-name rule selected every env.")

        train_count_tensor = torch.tensor([int(self.train_env_mask.sum().item())], dtype=torch.float64, device=self.device)
        validation_count_tensor = torch.tensor(
            [int(self.validation_env_mask.sum().item())],
            dtype=torch.float64,
            device=self.device,
        )
        if self.use_ddp:
            dist.all_reduce(train_count_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(validation_count_tensor, op=dist.ReduceOp.SUM)
        self.global_train_num_envs = int(train_count_tensor.item())
        self.global_validation_num_envs = int(validation_count_tensor.item())

        if self.rank == 0:
            print(
                "[INFO] Train/validation env split from asset suffix rule: "
                f"train={self.global_train_num_envs}, validation={self.global_validation_num_envs}"
            )

    def _has_teacher(self):
        return self.teacher_model is not None or len(getattr(self, "teacher_models", {})) > 0

    def _iter_teacher_models(self):
        if getattr(self, "multi_teacher_enabled", False):
            return self.teacher_models.values()
        if self.teacher_model is None:
            return ()
        return (self.teacher_model,)

    def _get_teacher_actions(self, obs):
        if not self._has_teacher():
            raise RuntimeError("Teacher model is not initialized.")
        latest_targets = self._get_teacher_prev_action_vector()
        if getattr(self, "multi_teacher_enabled", False):
            teacher_actions = torch.zeros((self.num_envs, self.teacher_num_actions), dtype=torch.float32, device=self.device)
            assigned_teacher_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for family_id, family_model in self.teacher_models_by_family_id.items():
                env_ids = torch.nonzero(self.env_family_ids == int(family_id), as_tuple=False).squeeze(-1)
                if env_ids.numel() == 0:
                    continue
                assigned_teacher_mask[env_ids] = True
                batch_dict = {
                    "is_train": False,
                    "obs": clip_teacher_obs(obs[self.teacher_obs_type][env_ids], self.teacher_clip_obs),
                    "prev_actions": latest_targets[env_ids],
                }
                with torch.no_grad():
                    res_dict = family_model(batch_dict)
                family_env_actions = torch.clamp(res_dict["mus"], -1.0, 1.0)
                teacher_actions[env_ids] = family_env_actions
            if not torch.all(assigned_teacher_mask):
                missing_family_ids = sorted(
                    set(self.env_family_ids[~assigned_teacher_mask].detach().cpu().tolist())
                )
                missing_family_names = [DOOR_FAMILY_NAMES[int(family_id)] for family_id in missing_family_ids]
                raise RuntimeError(f"Missing teacher model for door families: {missing_family_names}.")
            student_teacher_actions = self._env_actions_to_student_actions(teacher_actions)
            return {
                "mus": student_teacher_actions,
                "actions": teacher_actions,
            }

        batch_dict = {
            "is_train": False,
            "obs": clip_teacher_obs(obs[self.teacher_obs_type], self.teacher_clip_obs),
            "prev_actions": latest_targets,
        }
        with torch.no_grad():
            res_dict = self.teacher_model(batch_dict)
        teacher_actions = torch.clamp(res_dict["mus"], -1.0, 1.0)
        student_teacher_actions = self._env_actions_to_student_actions(teacher_actions)
        return {
            "mus": student_teacher_actions,
            "actions": teacher_actions,
        }

    def _sync_timing_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _record_timing(self, elapsed_s):
        self._timing_stats["sum_ms"] += elapsed_s * 1000.0
        self._timing_stats["count"] += 1

    def _consume_timing_means(self):
        if self._timing_stats["count"] == 0:
            return None
        mean_ms = self._timing_stats["sum_ms"] / self._timing_stats["count"]
        self._timing_stats["sum_ms"] = 0.0
        self._timing_stats["count"] = 0
        return mean_ms

    def _build_sampler_camera_spec(self):
      camera_cfg = self.ov_env.cfg.pointcloud_camera_cfg

      # Keep the render cheap, but match the IsaacGymEnvs D435 FOV/range.
      height = int(camera_cfg.height) // 2
      width = int(camera_cfg.width) // 2

      fov_x_deg = 85.2
      fov_y_deg = 58.0
      near_m = 0.3
      far_m = 3.0

      fx = width / (2.0 * math.tan(math.radians(fov_x_deg) * 0.5))
      fy = height / (2.0 * math.tan(math.radians(fov_y_deg) * 0.5))
      cx = (width - 1.0) * 0.5
      cy = (height - 1.0) * 0.5

      intrinsics = torch.tensor(
          [
              [fx, 0.0, cx],
              [0.0, fy, cy],
              [0.0, 0.0, 1.0],
          ],
          device=self.device,
          dtype=torch.float32,
      )

      return {
          "H": height,
          "W": width,
          "intrinsics": intrinsics,
          "near_m": near_m,
          "far_m": far_m,
      }

    def _get_sampler_camera_pose(self):
        camera_link_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_camera_body_idx]
        camera_link_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_camera_body_idx]
        # Apply the same camera mount offset as the real CameraCfg sensor (the -45deg roll that
        # compensates for the 45deg-tilted realsense bracket). Composed in the link's local frame
        # (world = link ⊗ offset), matching how IsaacLab mounts the camera prim on x5_camera_link.
        # The offset pos is zero, so the optical center stays at the link origin.
        if self._camera_mount_offset_quat_wxyz is not None:
            offset = self._camera_mount_offset_quat_wxyz.unsqueeze(0).expand(camera_link_quat_w.shape[0], -1)
            camera_quat_w = quat_mul(camera_link_quat_w, offset)
        else:
            camera_quat_w = camera_link_quat_w
        return torch.cat([camera_link_pos_w, camera_quat_w[:, [1, 2, 3, 0]]], dim=-1)

    def _get_lidar_pose(self):
        lidar_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_lidar_body_idx]
        lidar_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_lidar_body_idx]
        return torch.cat([lidar_pos_w, lidar_quat_w[:, [1, 2, 3, 0]]], dim=-1)

    def _sample_wall_pointcloud_local(self, env_ids=None, num_points=None):
        if num_points is None:
            num_points = self.wall_distractor_num_points
        num_points = int(num_points)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_count = int(env_ids.numel())
        if num_points <= 0 or env_count == 0:
            return torch.zeros((env_count, 0, 3), dtype=torch.float32, device=self.device)

        return sample_wall_points_local(
            axis_order=self.wall_distractor_axis_order[env_ids],
            bbox_min_ordered=self.wall_distractor_bbox_min_ordered[env_ids],
            bbox_max_ordered=self.wall_distractor_bbox_max_ordered[env_ids],
            num_points=num_points,
            params=self.wall_distractor_params,
            device=self.device,
        )

    def _resample_wall_distractors(self, env_ids=None):
        if (
            not self.wall_distractors_enabled
            or self.wall_distractor_num_points <= 0
            or self.wall_distractor_resample_each_step
        ):
            return
        if self._wall_distractor_local_points is None:
            self._wall_distractor_local_points = torch.zeros(
                (self.num_envs, self.wall_distractor_num_points, 3),
                dtype=torch.float32,
                device=self.device,
            )
        if env_ids is None:
            self._wall_distractor_local_points[:] = self._sample_wall_pointcloud_local()
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self._wall_distractor_local_points[env_ids] = self._sample_wall_pointcloud_local(env_ids=env_ids)

    def _sample_robot_pointcloud_world_sampler(self):
        return compose_cached_link_pointcloud_world(
            link_points_by_name=self.robot_link_pointclouds,
            link_pos_w_by_name={
                link_name: self.ov_env.robot.data.body_pos_w[:, body_idx]
                for link_name, body_idx in self.robot_sampler_body_indices.items()
            },
            link_quat_w_by_name={
                link_name: self.ov_env.robot.data.body_quat_w[:, body_idx]
                for link_name, body_idx in self.robot_sampler_body_indices.items()
            },
            num_points=self.scene_robot_pcd_num_points,
        )

    def _sample_robot_pointcloud_base_sampler(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        robot_pcd_world = self._sample_robot_pointcloud_world_sampler()
        return world_to_local(robot_pcd_world, robot_base_pos_w, robot_base_quat_w)

    def _get_robot_filter_joint_pos_base_frame(self):
        robot_joint_pos = self.ov_env.robot.data.joint_pos[:, self.robot_sampler_joint_ids]
        robot_joint_pos = robot_joint_pos[:, self.robot_sampler_joint_reorder].clone()
        # The observation cloud is already in tidybot2_base_link, so zero the floating-base
        # joints before evaluating the same Glorbot sphere model in that local frame.
        if self.robot_collision_checker_base_joint_indices:
            robot_joint_pos[:, self.robot_collision_checker_base_joint_indices] = 0.0
        return robot_joint_pos

    def _filter_robot_points_base(self, pointcloud_base):
        if (
            not self.robot_pointcloud_filter_enabled
            or self.robot_collision_checker is None
            or pointcloud_base is None
            or pointcloud_base.numel() == 0
        ):
            return pointcloud_base

        return self.robot_collision_checker.filter_pointcloud_outside_spheres(
            pointclouds=pointcloud_base,
            joint_angles=self._get_robot_filter_joint_pos_base_frame(),
            sdf_cutoff=self.robot_pointcloud_sdf_cutoff,
            max_points_per_process=self.robot_pointcloud_filter_max_points_per_process,
        )

    def _door_hole_aug_active(self):
        # Window-hole simulation is a pointcloud/world-geometry transform and should
        # follow its own config regardless of whether we are training or replaying.
        return self.door_hole_aug_enabled

    def _get_link1_pose_world(self):
        link1_pos_w = self.ov_env.door.data.body_pos_w[:, self.door_hole_link1_body_idx]
        link1_quat_w = self.ov_env.door.data.body_quat_w[:, self.door_hole_link1_body_idx]
        return torch.cat([link1_pos_w, link1_quat_w], dim=-1)

    def _sample_door_hole_aug_metadata(self, link1_pose_world):
        if not self._door_hole_aug_active():
            return None
        metadata = sample_random_window_hole_metadata(
            link1_pose_world=link1_pose_world,
            board_bbox_link1=self.env_board_bboxes_link1,
            window_prob=self.door_hole_aug_env_prob,
            width_range=self.door_hole_aug_width_range_m,
            height_range=self.door_hole_aug_height_range_m,
            center_height_range=self.door_hole_aug_center_height_range_m,
            side_margin_range=self.door_hole_aug_side_margin_range_m,
        )
        enabled = metadata["enabled"].to(dtype=torch.float32)
        self.latest_door_hole_aug_stats = {
            "door_hole_aug/env_fraction": float(enabled.mean().detach().cpu()),
            "door_hole_aug/no_hole_fraction": float((1.0 - enabled).mean().detach().cpu()),
            "door_hole_aug/hole_width_mean_m": float(metadata["hole_width"].mean().detach().cpu()),
            "door_hole_aug/hole_height_mean_m": float(metadata["hole_height"].mean().detach().cpu()),
        }
        return metadata

    def _apply_door_hole_aug_to_world(self, pointcloud_world, link1_pose_world, board_bbox_link1, hole_metadata):
        if pointcloud_world is None or hole_metadata is None:
            return pointcloud_world
        filtered_points, hole_metadata = apply_window_dropout_to_door_points(
            points_world=pointcloud_world,
            link1_pose_world=link1_pose_world,
            board_bbox_link1=board_bbox_link1,
            hole_metadata=hole_metadata,
            surface_eps=self.door_hole_aug_surface_eps_m,
        )
        self.latest_door_hole_aug_stats["door_hole_aug/dropped_points_mean"] = float(
            hole_metadata["num_dropped_points"].to(dtype=torch.float32).mean().detach().cpu()
        )
        return filtered_points

    def _sample_wall_pointcloud_world(self, num_points=None):
        if not self.wall_distractors_enabled:
            return torch.zeros((self.num_envs, 0, 3), dtype=torch.float32, device=self.device)

        if num_points is None:
            num_points = self.wall_distractor_num_points
        num_points = int(num_points)
        if num_points <= 0:
            return torch.zeros((self.num_envs, 0, 3), dtype=torch.float32, device=self.device)

        door_base_pos_w = self.ov_env.door.data.body_pos_w[:, self.door_base_body_idx]
        door_base_quat_w = self.ov_env.door.data.body_quat_w[:, self.door_base_body_idx]
        if (
            self.wall_distractor_resample_each_step
            or self._wall_distractor_local_points is None
            or num_points != self.wall_distractor_num_points
        ):
            wall_points_base = self._sample_wall_pointcloud_local(num_points=num_points)
        else:
            wall_points_base = self._wall_distractor_local_points
        quat = door_base_quat_w.unsqueeze(1).expand(-1, wall_points_base.shape[1], -1)
        return quat_apply(quat, wall_points_base) + door_base_pos_w.unsqueeze(1)

    def _sample_cached_door_pointcloud_world(self, hole_metadata=None):
        door_pcd_world = torch.zeros(
            (self.num_envs, self.scene_door_pcd_num_points, 3),
            dtype=torch.float32,
            device=self.device,
        )
        link1_pose_world_all = self._get_link1_pose_world() if hole_metadata is not None else None
        for asset_idx, link_points_by_name in self.door_link_pointclouds.items():
            env_ids = self.door_sampler_env_ids.get(asset_idx)
            if env_ids is None:
                env_ids = torch.nonzero(self.env_asset_idx == asset_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            asset_pcd_world = compose_cached_link_pointcloud_world(
                link_points_by_name=link_points_by_name,
                link_pos_w_by_name={
                    link_name: self.ov_env.door.data.body_pos_w[env_ids, self.door_link_body_indices[link_name]]
                    for link_name in ("link_1", "link_2")
                    if link_name in self.door_link_body_indices
                },
                link_quat_w_by_name={
                    link_name: self.ov_env.door.data.body_quat_w[env_ids, self.door_link_body_indices[link_name]]
                    for link_name in ("link_1", "link_2")
                    if link_name in self.door_link_body_indices
                },
                num_points=self.scene_door_pcd_num_points,
            )
            if hole_metadata is not None:
                asset_pcd_world = self._apply_door_hole_aug_to_world(
                    asset_pcd_world,
                    link1_pose_world_all[env_ids],
                    self.env_board_bboxes_link1[env_ids],
                    {key: value[env_ids] for key, value in hole_metadata.items()},
                )
            door_pcd_world[env_ids] = asset_pcd_world
        return door_pcd_world

    def _sample_scene_pointcloud_world_cached(self, hole_metadata=None):
        door_pcd_world = self._sample_cached_door_pointcloud_world(hole_metadata=hole_metadata)
        robot_pcd_world = self._sample_robot_pointcloud_world_sampler()
        wall_pcd_world = self._sample_wall_pointcloud_world()
        # Occluder set = static scene geometry only (door panel + wall distractors), excluding the
        # robot: the robot's own thin links (fingers) should stay crisp, not get dilated/bloated by
        # the occluder-fill render pass.
        occluder_parts = [door_pcd_world]
        scene_parts = [door_pcd_world, robot_pcd_world]
        if wall_pcd_world.shape[1] > 0:
            occluder_parts.append(wall_pcd_world)
            scene_parts.append(wall_pcd_world)
        occluder_pcd_world = torch.cat(occluder_parts, dim=1)
        return torch.cat(scene_parts, dim=1), occluder_pcd_world

    def _render_lidar_scene_pointcloud_base(
        self, scene_pcd_world, robot_base_pos_w, robot_base_quat_w, occluder_pcd_world=None
    ):
        rendered_pcd_world, _ = simulate_lidar_render_from_pose(
            pcd=scene_pcd_world,
            lidar_pose=self._get_lidar_pose(),
            num_points=self.lidar_num_points,
            num_azimuth=self.lidar_num_azimuth,
            num_polar=self.lidar_num_polar,
            near_m=self.lidar_near_m,
            far_m=self.lidar_far_m,
            suppress_bins=self.lidar_suppress_bins,
            occlusion_eps_m=self.lidar_occlusion_eps_m,
            occlusion_eps_rel=self.lidar_occlusion_eps_rel,
            jitter_std_m=self.lidar_jitter_std_m,
            use_compile=self.lidar_use_compile,
            occluder_pcd=occluder_pcd_world,
            occluder_fill_bins=self.lidar_render_occluder_fill_bins,
        )
        return world_to_local(rendered_pcd_world, robot_base_pos_w, robot_base_quat_w)

    def _sample_scene_obs_pointcloud_base_sampler(self):
        return self._sample_scene_obs_pointcloud_base_depth()

    def _sample_scene_obs_pointcloud_base_depth(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        gt_scene_pcd_world, occluder_pcd_world = self._sample_scene_pointcloud_world_cached(
            hole_metadata=getattr(self, "_current_door_hole_aug_metadata", None)
        )
        if self.viser_raw_enabled:
            # Keep only the selected family envs on CPU so Viser debug does not retain an
            # extra full batched scene pointcloud on GPU during training/debug runs.
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(gt_scene_pcd_world)
        rendered_pcd_world, _ = simulate_depth_cam_render_from_pose(
            pcd=gt_scene_pcd_world,
            camera_pose=self._get_sampler_camera_pose(),
            num_points=self.depth_cam_render_num_points,
            inflate_px=self.depth_cam_render_inflate_px,
            jitter_std_m=self.depth_cam_render_jitter_std_m,
            cam_spec_dict=self.sampler_camera_spec,
            clip_mode=self.depth_cam_render_clip_mode,
            jitter_mode=self.depth_cam_render_jitter_mode,
            use_compile=self.depth_cam_render_use_compile,
            blur_kernel_px=self.depth_cam_render_blur_kernel_px,
            blur_sigma_px=self.depth_cam_render_blur_sigma_px,
            occluder_pcd=occluder_pcd_world,
            occluder_inflate_px=self.depth_cam_render_occluder_inflate_px,
        )
        scene_pointcloud_base = world_to_local(rendered_pcd_world, robot_base_pos_w, robot_base_quat_w)
        return scene_pointcloud_base

    def _sample_scene_obs_pointcloud_base_lidar(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        gt_scene_pcd_world, occluder_pcd_world = self._sample_scene_pointcloud_world_cached(
            hole_metadata=getattr(self, "_current_door_hole_aug_metadata", None)
        )
        if self.viser_raw_enabled:
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(gt_scene_pcd_world)

        return self._render_lidar_scene_pointcloud_base(
            gt_scene_pcd_world, robot_base_pos_w, robot_base_quat_w, occluder_pcd_world=occluder_pcd_world
        )

    def _sample_scene_obs_pointcloud_base_both(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        gt_scene_pcd_world, occluder_pcd_world = self._sample_scene_pointcloud_world_cached(
            hole_metadata=getattr(self, "_current_door_hole_aug_metadata", None)
        )
        if self.viser_raw_enabled:
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(gt_scene_pcd_world)

        rendered_depth_pcd_world, _ = simulate_depth_cam_render_from_pose(
            pcd=gt_scene_pcd_world,
            camera_pose=self._get_sampler_camera_pose(),
            num_points=self.depth_cam_render_num_points,
            inflate_px=self.depth_cam_render_inflate_px,
            jitter_std_m=self.depth_cam_render_jitter_std_m,
            cam_spec_dict=self.sampler_camera_spec,
            clip_mode=self.depth_cam_render_clip_mode,
            jitter_mode=self.depth_cam_render_jitter_mode,
            use_compile=self.depth_cam_render_use_compile,
            blur_kernel_px=self.depth_cam_render_blur_kernel_px,
            blur_sigma_px=self.depth_cam_render_blur_sigma_px,
            occluder_pcd=occluder_pcd_world,
            occluder_inflate_px=self.depth_cam_render_occluder_inflate_px,
        )
        depth_pcd_base = world_to_local(rendered_depth_pcd_world, robot_base_pos_w, robot_base_quat_w)
        lidar_pcd_base = self._render_lidar_scene_pointcloud_base(
            gt_scene_pcd_world,
            robot_base_pos_w,
            robot_base_quat_w,
            occluder_pcd_world=occluder_pcd_world,
        )
        return depth_pcd_base, lidar_pcd_base

    def _sample_scene_obs_pointcloud_base(self):
        self._viser_cached_ground_truth_pcd_world = None
        self._viser_cached_sensor_obs_pcd_base = OrderedDict()
        self.latest_door_hole_aug_stats = {}
        link1_pose_world = self._get_link1_pose_world()
        self._current_door_hole_aug_metadata = self._sample_door_hole_aug_metadata(link1_pose_world)
        try:
            if self.pointcloud_source in {"sampler", "depth"}:
                depth_pcd_base = self._filter_robot_points_base(self._sample_scene_obs_pointcloud_base_depth())
                self._viser_cached_sensor_obs_pcd_base["robot_depth_cam_obs"] = depth_pcd_base
                scene_obs_pcd_sources = [depth_pcd_base]
            elif self.pointcloud_source == "lidar":
                lidar_pcd_base = self._filter_robot_points_base(self._sample_scene_obs_pointcloud_base_lidar())
                self._viser_cached_sensor_obs_pcd_base["robot_lidar_obs"] = lidar_pcd_base
                scene_obs_pcd_sources = [lidar_pcd_base]
            else:
                depth_pcd_base, lidar_pcd_base = self._sample_scene_obs_pointcloud_base_both()
                depth_pcd_base = self._filter_robot_points_base(depth_pcd_base)
                lidar_pcd_base = self._filter_robot_points_base(lidar_pcd_base)
                self._viser_cached_sensor_obs_pcd_base["robot_lidar_obs"] = lidar_pcd_base
                self._viser_cached_sensor_obs_pcd_base["robot_depth_cam_obs"] = depth_pcd_base
                # Keep the realsense (dense, narrow FoV) and lidar (sparse, wide FoV) clouds
                # SEPARATE. Concatenating then sampling a fixed budget biases toward the denser
                # realsense and can drop the sparse-but-critical lidar handle points. Each local
                # crop instead draws an equal share from each source (see _build_local_pcd).
                scene_obs_pcd_sources = [depth_pcd_base, lidar_pcd_base]
            return scene_obs_pcd_sources
        finally:
            self._current_door_hole_aug_metadata = None

    @staticmethod
    def _split_point_budget(total, num_parts):
        # Split `total` points into `num_parts` near-equal integer chunks (extras go to the first
        # chunks), e.g. 2500 over 2 sources -> [1250, 1250], 1001 -> [501, 500].
        total = int(total)
        num_parts = max(1, int(num_parts))
        base = total // num_parts
        remainder = total - base * num_parts
        return [base + (1 if i < remainder else 0) for i in range(num_parts)]

    def _crop_local_pcd_balanced(
        self,
        scene_obs_pcd_sources,
        *,
        local_range,
        num_local_points,
        is_cylindrical,
        crop_center,
        x_direction_cutoff,
        log_name,
    ):
        # Crop each sensor source independently with an equal slice of the point budget, then
        # concatenate. Total point count equals the single-cloud crop, so the policy input size is
        # unchanged. With a single source this is identical to a plain crop_local_pcd call.
        if len(scene_obs_pcd_sources) == 1:
            crop, _ = crop_local_pcd(
                scene_obs_pcd_sources[0],
                local_range=local_range,
                num_local_points=num_local_points,
                is_cylindrical=is_cylindrical,
                crop_center=crop_center,
                x_direction_cutoff=x_direction_cutoff,
                log_name=log_name,
            )
            return crop
        per_source_counts = self._split_point_budget(num_local_points, len(scene_obs_pcd_sources))
        crops = []
        for src_idx, (src_pcd, src_count) in enumerate(zip(scene_obs_pcd_sources, per_source_counts)):
            if src_count <= 0:
                continue
            crop, _ = crop_local_pcd(
                src_pcd,
                local_range=local_range,
                num_local_points=src_count,
                is_cylindrical=is_cylindrical,
                crop_center=crop_center,
                x_direction_cutoff=x_direction_cutoff,
                log_name=f"{log_name}_src{src_idx}",
            )
            crops.append(crop)
        return torch.cat(crops, dim=1)

    def _build_local_pcd(self, scene_obs_pcd_sources, palm_pos_base, robot_pcd_base=None):
        # Accept either a single cloud or a list of per-sensor clouds (e.g. [realsense, lidar] when
        # pointcloud_source="both"). Each crop draws an equal share from every source so the denser
        # sensor cannot dominate the policy cloud.
        if torch.is_tensor(scene_obs_pcd_sources):
            scene_obs_pcd_sources = [scene_obs_pcd_sources]
        pcd_parts = []

        if self.local_pcd_points[0] > 0:
            base_crop = self._crop_local_pcd_balanced(
                scene_obs_pcd_sources,
                local_range=self.local_pcd_range[0],
                num_local_points=self.local_pcd_points[0],
                is_cylindrical=True,
                crop_center=self.zero_local_pcd_crop_center,
                x_direction_cutoff=self.local_pcd_x_direction_cutoff,
                log_name="base",
            )
            pcd_parts.append(base_crop)

        if self.local_pcd_points[1] > 0:
            palm_crop = self._crop_local_pcd_balanced(
                scene_obs_pcd_sources,
                local_range=self.local_pcd_range[1],
                num_local_points=self.local_pcd_points[1],
                is_cylindrical=False,
                crop_center=palm_pos_base,
                x_direction_cutoff=None,
                log_name="palm",
            )
            pcd_parts.append(palm_crop)

        if (
            robot_pcd_base is not None
            and robot_pcd_base.numel() > 0
            and self.robot_model_policy_points > 0
        ):
            if robot_pcd_base.shape[1] > self.robot_model_policy_points:
                sample_idx = torch.linspace(
                    0,
                    robot_pcd_base.shape[1] - 1,
                    steps=self.robot_model_policy_points,
                    device=robot_pcd_base.device,
                    dtype=torch.float32,
                ).round().to(dtype=torch.long)
                robot_pcd_base = robot_pcd_base[:, sample_idx]
            pcd_parts.append(robot_pcd_base)

        if not pcd_parts:
            raise ValueError("Student config requested local_pcd_t but no local point counts were configured.")
        return torch.cat(pcd_parts, dim=1)

    def _get_global_batch_size(self, local_batch_size):
        batch_size = torch.tensor(int(local_batch_size), dtype=torch.int64, device=self.device)
        if self.use_ddp:
            dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)
        return int(batch_size.item())

    def _build_student_obs(self, iteration=None):
        q_pos = self._get_student_proprio_vector()
        base_vel = self._get_student_base_velocity_vector()
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        palm_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_palm_body_idx].unsqueeze(1)

        self.latest_student_proprio_vector = q_pos.detach().clone()

        # Point cloud is optional: skip all (expensive) cloud sampling when the student has no pcd
        # encoders configured (state-only policy).
        has_pcd = bool(self.pcd_encoders_keys)
        if has_pcd:
            palm_pos_base = world_to_local(palm_pos_w, robot_base_pos_w, robot_base_quat_w).squeeze(1)
            scene_obs_pcd_sources = self._sample_scene_obs_pointcloud_base()
            robot_pcd_base = (
                self._sample_robot_pointcloud_base_sampler() if self.append_robot_model_to_policy_cloud else None
            )
        else:
            palm_pos_base = None
            scene_obs_pcd_sources = None
            robot_pcd_base = None
        target_t = self._get_implemented_action_vector()
        need_aux_target_vector = self.has_aux_input and (not self.play_policy and self.has_aux_prediction)
        aux_target_vector = (
            self._stack_aux_state_values(self._get_aux_state_values()) if need_aux_target_vector else None
        )
        if self.has_aux_input and self.aux_handle_input_mode == "closed_door_base":
            # Non-recurrent closed-door handle anchor (current base frame + fresh per-step noise).
            aux_input_vector = self._build_closed_door_aux_input_vector()
        elif self.has_aux_input and self.aux_feedback_to_policy:
            if self.aux_buffer is None:
                raise RuntimeError("Aux feedback requested but aux_buffer is not initialized.")
            aux_input_vector = self.aux_buffer.clone()
        elif self.has_aux_input:
            aux_input_vector = torch.zeros((self.num_envs, self.aux_input_dim), dtype=torch.float32, device=self.device)
        else:
            aux_input_vector = None
        push_pull_cond = None
        if self.push_pull_condition_enabled:
            self.latest_push_pull_condition_source = self.push_pull_condition_source
            # Oracle source uses GT one-hot. Predicted source uses the recurrent condition carried from the previous step.
            push_pull_cond = self._build_push_pull_condition_from_source(self.push_pull_condition_source)
            # Perturbation modifies only the condition input fed to the action policy, not labels or target actions.
            push_pull_cond = self._apply_push_pull_condition_perturb(push_pull_cond)
        current_aux_handle_temporal = self._get_temporal_aux_handle_from_policy_input(aux_input_vector)
        current_push_pull_belief_temporal = self._get_temporal_push_pull_belief_from_policy_input(push_pull_cond)
        self.latest_push_pull_belief_input = (
            None if current_push_pull_belief_temporal is None else current_push_pull_belief_temporal.detach().clone()
        )
        lag_active = self._is_observation_lag_active()
        required_temporal_offsets_s = self._merge_unique_offsets_s(
            self.proprio_temporal_timestamps_s,
            (
                float(spec["offset_s"])
                for spec in self.temporal_derived_state_specs.values()
                if spec["offset_s"] is not None
            ),
            (0.0,) if (lag_active or self.proprio_temporal_enabled) else (),
        )
        temporal_sample_cache = self._build_temporal_sample_cache(
            q_pos,
            target_t,
            base_vel,
            required_temporal_offsets_s,
            apply_observation_lag=lag_active,
            aux_handle_pos=current_aux_handle_temporal,
            push_pull_belief=current_push_pull_belief_temporal,
        ) if required_temporal_offsets_s else None
        if lag_active and temporal_sample_cache is not None:
            self._record_observation_lag_stats(
                temporal_sample_cache["offsets_s"],
                temporal_sample_cache["effective_age_ms"],
            )
        else:
            self._reset_observation_lag_stats()
            self.latest_obs_lag_enabled = 1.0 if lag_active else 0.0
        self._record_push_pull_belief_history_metrics(temporal_sample_cache)
        temporal_state_values = self._build_temporal_derived_state_values(
            q_pos,
            target_t,
            base_vel,
            sample_cache=temporal_sample_cache,
        )
        lagged_q_full = None
        lagged_base_vel = None
        if temporal_sample_cache is not None and 0.0 in temporal_sample_cache["offset_to_index"]:
            lagged_q_full = self._get_temporal_sample_from_cache(temporal_sample_cache, "q_full", 0.0)
            lagged_base_vel = self._get_temporal_sample_from_cache(temporal_sample_cache, "base_vel", 0.0)

        obs = OrderedDict()
        for key in self.state_encoders_keys:
            if key == "q_base":
                raise KeyError("Raw q_base is disabled for the student policy; use base_vel instead.")
            elif key == "q_arm":
                if lag_active and lagged_q_full is not None:
                    obs[key] = lagged_q_full[:, self.ov_env._robot_arm_dof_idx]
                else:
                    obs[key] = q_pos[:, self.ov_env._robot_arm_dof_idx]
            elif key == "q_hand":
                if lag_active and lagged_q_full is not None:
                    obs[key] = lagged_q_full[:, self.ov_env._robot_finger_dof_idx]
                else:
                    obs[key] = q_pos[:, self.ov_env._robot_finger_dof_idx]
            elif key == "base_vel":
                obs[key] = lagged_base_vel if lag_active and lagged_base_vel is not None else base_vel
            elif key in self.temporal_derived_state_specs:
                obs[key] = temporal_state_values[key]
            elif key in self.aux_state_specs:
                if aux_input_vector is None:
                    raise RuntimeError(f"Aux state '{key}' is enabled but aux input vector is unavailable.")
                obs[key] = aux_input_vector[:, self.aux_state_specs[key]["slice"]]
            else:
                raise KeyError(f"Unsupported student state key '{key}' in config.")

        if self.push_pull_condition_enabled:
            if push_pull_cond is None:
                raise RuntimeError(
                    f"Push/pull condition '{self.push_pull_condition_obs_key}' is enabled but unavailable."
                )
            obs[self.push_pull_condition_obs_key] = push_pull_cond

        if self.left_right_condition_enabled:
            left_right_cond = self._build_gt_left_right_condition()
            self.latest_fraction_left = float(left_right_cond[:, 0].mean().detach().cpu().item())
            self.latest_fraction_right = float(left_right_cond[:, 1].mean().detach().cpu().item())
            obs[self.left_right_condition_obs_key] = left_right_cond

        if self.proprio_temporal_enabled:
            if self.proprio_temporal_obs_key is None:
                raise RuntimeError("temporal_state_encoders are enabled but no observation key is configured.")
            obs[self.proprio_temporal_obs_key] = self._build_proprio_temporal_obs(temporal_sample_cache)

        for key in self.pcd_encoders_keys:
            if key == "local_pcd_t":
                obs[key] = self._build_local_pcd(
                    scene_obs_pcd_sources,
                    palm_pos_base,
                    robot_pcd_base=robot_pcd_base,
                )
            else:
                raise KeyError(f"Unsupported student pointcloud key '{key}' in config.")

        # Cache the latest policy-input cloud (base frame, [num_envs, N, 3]) so eval/replay can save
        # it into the compact .pt even when the full viser_raw path is not enabled.
        self._last_policy_input_pcd_base = obs.get("local_pcd_t")

        if self.viser_raw_enabled and has_pcd:
            self._viser_pending_debug_frame = {
                "iteration": iteration,
                "robot_base_pos_w": robot_base_pos_w,
                "robot_base_quat_w": robot_base_quat_w,
                "ground_truth_pcd_world": self._viser_cached_ground_truth_pcd_world,
                "sensor_obs_pcd_base_by_name": self._viser_cached_sensor_obs_pcd_base,
                "policy_input_pcd_base": obs.get("local_pcd_t"),
            }
            self._viser_cached_ground_truth_pcd_world = None
            self._viser_cached_sensor_obs_pcd_base = OrderedDict()
        elif self.viser_raw_enabled and self._is_viser_capture_iteration(iteration):
            # State-only policy (no sensor cloud fed to the policy): still show the ground-truth
            # scene cloud in Viser, but compose it from cached link clouds ONLY on capture
            # iterations -- no per-step depth/lidar rendering.
            gt_scene_pcd_world, _ = self._sample_scene_pointcloud_world_cached(
                hole_metadata=getattr(self, "_current_door_hole_aug_metadata", None)
            )
            self._viser_pending_debug_frame = {
                "iteration": iteration,
                "robot_base_pos_w": robot_base_pos_w,
                "robot_base_quat_w": robot_base_quat_w,
                "ground_truth_pcd_world": self._select_viser_ground_truth_points(gt_scene_pcd_world),
                "sensor_obs_pcd_base_by_name": None,
                "policy_input_pcd_base": None,
            }

        self.latest_aux_input_vector = None if aux_input_vector is None else aux_input_vector.detach().clone()
        self.latest_aux_target_vector = None if aux_target_vector is None else aux_target_vector.detach().clone()
        return obs

    def _student_forward(self, student_obs, iteration=None):
        base_obs = OrderedDict(student_obs)
        if not self.push_pull_condition_enabled:
            self.latest_push_pull_condition_source = "disabled"
            self.latest_push_pull_perturb_to_push_count = 0
            self.latest_push_pull_perturb_to_pull_count = 0
            student_output = self.student_model_ddp(base_obs)
            if self.mode_prediction_enabled and "mode_logits" in student_output:
                self._record_push_pull_prediction_metrics(student_output["mode_logits"])
            return student_output

        student_output = self.student_model_ddp(base_obs)
        if self.mode_prediction_enabled and "mode_logits" in student_output:
            self._record_push_pull_prediction_metrics(student_output["mode_logits"])
            # Recurrent predicted conditioning: timestep t consumes the current condition and writes the next one.
            self._update_push_pull_condition_buffer(student_output["mode_logits"])
        return student_output

    def _slice_batch_dict(self, batch_dict, env_mask):
        env_mask = env_mask.to(device=self.device, dtype=torch.bool)
        sliced = {}
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == self.num_envs:
                sliced[key] = value[env_mask]
            else:
                sliced[key] = value
        return sliced

    def _compute_student_loss(self, student_output, teacher_actions, aux_target=None, env_mask=None, update_metrics=True):
        if env_mask is not None:
            env_mask = env_mask.to(device=self.device, dtype=torch.bool)
            if not torch.any(env_mask):
                return None, None, None, None, None
            student_output = self._slice_batch_dict(student_output, env_mask)
            teacher_actions = teacher_actions[env_mask]
            if aux_target is not None:
                aux_target = aux_target[env_mask]
        target = teacher_actions
        if aux_target is not None:
            target = torch.cat([teacher_actions, aux_target], dim=-1)
        loss = self.student_model.compute_loss(student_output, target.unsqueeze(1))
        total_loss = loss["total"]
        action_loss = loss.get("action", total_loss)
        aux_loss = loss.get("aux")
        mode_loss = None
        door_joint_loss = None
        if self.mode_prediction_enabled:
            if "mode_logits" not in student_output:
                raise RuntimeError("Mode prediction is enabled, but student output does not contain 'mode_logits'.")
            mode_loss = self._compute_mode_prediction_loss(
                student_output["mode_logits"],
                env_mask=env_mask,
                update_metrics=update_metrics,
            )
            total_loss = total_loss + self.mode_weight * mode_loss
        if self.door_joint_prediction_enabled:
            if "door_joint" not in student_output:
                raise RuntimeError("Door joint prediction is enabled, but student output does not contain 'door_joint'.")
            door_joint_loss = self._compute_door_joint_prediction_loss(
                student_output["door_joint"],
                env_mask=env_mask,
                update_metrics=update_metrics,
            )
            total_loss = total_loss + self.door_joint_prediction_weight * door_joint_loss
        return total_loss, action_loss, aux_loss, mode_loss, door_joint_loss

    def _get_teacher_forcing_beta(self, iteration):
        if self.play_policy or not self._has_teacher():
            return 0.0
        if iteration < self.teacher_forcing_warmup_iters:
            return 1.0
        if self.teacher_forcing_transition_iters <= 0:
            return self.teacher_forcing_min_beta

        transition_iteration = iteration - self.teacher_forcing_warmup_iters
        if transition_iteration >= self.teacher_forcing_transition_iters:
            return self.teacher_forcing_min_beta

        progress = transition_iteration / float(self.teacher_forcing_transition_iters)
        schedule_value = 1.0 - progress
        beta = self.teacher_forcing_min_beta + (1.0 - self.teacher_forcing_min_beta) * schedule_value
        return float(max(self.teacher_forcing_min_beta, min(1.0, beta)))

    def _sample_teacher_forcing_mask(self, num_envs, beta):
        mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if num_envs <= 0 or beta <= 0.0:
            return mask
        if beta >= 1.0:
            mask.fill_(True)
            return mask

        num_teacher_envs = int(round(beta * num_envs))
        if num_teacher_envs <= 0:
            return mask
        if num_teacher_envs >= num_envs:
            mask.fill_(True)
            return mask

        selected_envs = torch.randperm(num_envs, device=self.device)[:num_teacher_envs]
        mask[selected_envs] = True
        return mask

    def _resample_teacher_forcing_env_mask(self, iteration, env_ids=None):
        beta = self._get_teacher_forcing_beta(iteration)
        if self.play_policy or not self._has_teacher():
            if env_ids is None:
                self.teacher_forcing_env_mask.zero_()
            elif env_ids.numel() > 0:
                self.teacher_forcing_env_mask[env_ids] = False
            return beta

        if env_ids is None:
            self.teacher_forcing_env_mask[:] = self._sample_teacher_forcing_mask(self.num_envs, beta)
            return beta

        if env_ids.numel() == 0:
            return beta

        self.teacher_forcing_env_mask[env_ids] = self._sample_teacher_forcing_mask(int(env_ids.numel()), beta)
        return beta

    def _get_teacher_forcing_env_fraction(self):
        if self.play_policy or not self._has_teacher():
            return 0.0
        return float(self.teacher_forcing_env_mask.float().mean().item())

    def _mix_actions(self, student_actions, teacher_actions, iteration):
        if self.play_policy or teacher_actions is None:
            return student_actions, 0.0
        beta = self._get_teacher_forcing_beta(iteration)
        teacher_mask = self.teacher_forcing_env_mask
        if not torch.any(teacher_mask):
            return student_actions, beta
        if torch.all(teacher_mask):
            return teacher_actions, beta
        step_actions = student_actions.clone()
        step_actions[teacher_mask] = teacher_actions[teacher_mask]
        return step_actions, beta

    def distill(self):
        if not self.play_policy and not self._has_teacher():
            raise RuntimeError("Teacher model must be initialized for distillation.")

        self.student_model_ddp.train(not self.play_policy)
        for teacher_model in self._iter_teacher_models():
            teacher_model.eval()

        try:
            start_iteration = int(self.resume_iteration)
            end_iteration = start_iteration + int(self.num_iters)
            obs, reset_extras = self.env.reset()
            self._update_logged_env_metrics(reset_extras)
            self.latest_student_proprio_vector = None
            self.latest_aux_input_vector = None
            self.latest_aux_target_vector = None
            self._resample_wall_distractors()
            self.temporal_current_time_s = self._iteration_to_time_s(start_iteration)
            self._resample_aux_handle_bias()
            self._seed_temporal_histories()
            self._seed_aux_buffer()
            self._seed_push_pull_condition_buffer()
            self._resample_teacher_forcing_env_mask(self.resume_iteration)

            for iteration in range(start_iteration, end_iteration):
                self._sync_timing_device()
                iteration_start_time = time.perf_counter()

                student_obs = self._build_student_obs(iteration=iteration)
                student_output = self._student_forward(student_obs, iteration=iteration)
                if self.mode_prediction_enabled and self.play_policy and iteration % 10 == 0:
                    mode_logits = student_output["mode_logits"].detach()
                    print("Iteration ", iteration, ": Direction Pred:", mode_logits.detach().cpu().tolist())
                student_actions = student_output["action"][:, 0, :]
                student_env_actions = self._student_actions_to_env_actions(student_actions)
                aux_prediction_for_replay = None
                if self.has_aux_prediction:
                    aux_prediction_for_replay = self._decode_aux_prediction(student_output["aux"].detach())
                    # Recurrent feedback only: in closed_door_base mode the aux input is the fixed
                    # closed-door anchor, so the prediction is never fed back into the buffer.
                    if self.aux_handle_input_mode != "closed_door_base":
                        aux_feedback_value = aux_prediction_for_replay
                        if self.aux_feedback_noise_m is not None:
                            aux_feedback_value = aux_feedback_value + self._sample_isotropic_offset(
                                aux_feedback_value.shape,
                                self.aux_feedback_noise_m,
                                dtype=aux_feedback_value.dtype,
                            )
                        self.aux_buffer[:] = aux_feedback_value

                if self.viser_raw_enabled and self._viser_pending_debug_frame is not None:
                    self._maybe_update_viser_debug(
                        iteration=self._viser_pending_debug_frame["iteration"],
                        robot_base_pos_w=self._viser_pending_debug_frame["robot_base_pos_w"],
                        robot_base_quat_w=self._viser_pending_debug_frame["robot_base_quat_w"],
                        ground_truth_pcd_world=self._viser_pending_debug_frame["ground_truth_pcd_world"],
                        sensor_obs_pcd_base_by_name=self._viser_pending_debug_frame["sensor_obs_pcd_base_by_name"],
                        policy_input_pcd_base=self._viser_pending_debug_frame["policy_input_pcd_base"],
                        aux_prediction=aux_prediction_for_replay,
                        aux_input=self.latest_aux_input_vector,
                    )
                    self._viser_pending_debug_frame = None

                teacher_actions = None
                train_total_loss = None
                train_action_loss = None
                train_aux_loss = None
                train_mode_loss = None
                train_door_joint_loss = None
                validation_total_loss = None
                validation_action_loss = None

                if not self.play_policy:
                    teacher_output = self._get_teacher_actions(obs)
                    teacher_actions = teacher_output["actions"]
                    aux_target = None
                    if self.has_aux_prediction:
                        if self.latest_aux_target_vector is None:
                            raise RuntimeError("Expected the latest auxiliary target vector while aux prediction is enabled.")
                        aux_target = self._get_aux_target(self.latest_aux_target_vector)
                    train_total_loss, train_action_loss, train_aux_loss, train_mode_loss, train_door_joint_loss = self._compute_student_loss(
                        student_output,
                        teacher_output["mus"],
                        aux_target=aux_target,
                        env_mask=self.train_env_mask,
                        update_metrics=True,
                    )
                    if train_total_loss is None:
                        raise RuntimeError("Training loss could not be computed because the training split is empty.")
                    validation_total_loss, validation_action_loss, _, _, _ = self._compute_student_loss(
                        student_output,
                        teacher_output["mus"],
                        aux_target=aux_target,
                        env_mask=self.validation_env_mask,
                        update_metrics=False,
                    )
                    local_batch_size = int(self.train_env_mask.sum().detach().cpu().item())
                    global_batch_size = self._get_global_batch_size(local_batch_size)
                    self.optimizer.zero_grad()
                    train_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model_ddp.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.student_update_steps += 1
                    self._apply_optimizer_runtime_overrides()
                    self.last_local_update_batch_size = local_batch_size
                    self.last_global_update_batch_size = global_batch_size

                step_actions, teacher_forcing_beta = self._mix_actions(
                    student_env_actions.detach(),
                    teacher_actions,
                    iteration,
                )
                obs, rew, out_of_reach, timed_out, step_extras = self.env.step(step_actions)
                self._update_logged_env_metrics(step_extras)
                self.temporal_current_time_s = self._iteration_to_time_s(iteration + 1)
                q_after_step = self._get_student_proprio_vector().detach().clone()
                target_after_step = self._get_implemented_action_vector().detach().clone()
                base_vel_after_step = self._get_student_base_velocity_vector().detach().clone()

                self._push_temporal_history(
                    timestamp=self.temporal_current_time_s,
                    q=q_after_step,
                    target=target_after_step,
                    base_vel=base_vel_after_step,
                )
                self.frame += self.num_envs

                self.current_rewards += rew.unsqueeze(-1)
                self.current_lengths += 1
                done_mask = torch.nonzero(out_of_reach | timed_out, as_tuple=False).squeeze(-1)
                if done_mask.numel() > 0:
                    self._update_completed_episode_metrics(done_mask, timed_out)
                    self.current_rewards[done_mask] = 0.0
                    self.current_lengths[done_mask] = 0.0
                    self._resample_wall_distractors(done_mask)
                    self._resample_aux_handle_bias(done_mask)
                    self._seed_temporal_histories(done_mask)
                    self._seed_aux_buffer(done_mask)
                    self._seed_push_pull_condition_buffer(done_mask)
                    self._resample_teacher_forcing_env_mask(iteration + 1, done_mask)

                if train_total_loss is not None:
                    self._sync_timing_device()
                    self._record_timing(time.perf_counter() - iteration_start_time)
                    self._log(
                        iteration,
                        train_total_loss,
                        train_action_loss,
                        train_aux_loss,
                        train_mode_loss,
                        train_door_joint_loss,
                        validation_total_loss,
                        validation_action_loss,
                        teacher_forcing_beta,
                    )
                else:
                    self._sync_timing_device()
                    self._record_timing(time.perf_counter() - iteration_start_time)

                if (
                    not self.play_policy
                    and iteration % self.save_interval == 0
                ):
                    if self.rank == 0:
                        ckpt_path = os.path.join(self.nn_dir, f"pcd_student_{iteration}.pt")
                        self.save(ckpt_path, iteration=iteration)
        finally:
            self._viser_pending_debug_frame = None
            self._close_viser_debug_tools()
            if not self.play_policy and self.rank == 0:
                print("=" * 10)
                print("TRAINING SUMMARY")
                print("Student Update Steps:", self.student_update_steps)
            self._finish_wandb()

    def save(self, filename, iteration=None):
        if iteration is None:
            iteration = int(self.frame // max(1, int(self.num_envs)))
        curriculum_step_count = int(getattr(self.ov_env, "common_step_counter", int(iteration)))
        checkpoint = {
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "frame": self.frame,
            "epoch": self.epoch_num,
            "iteration": int(iteration),
            "curriculum_step_count": curriculum_step_count,
            "student_update_steps": int(self.student_update_steps),
            "num_envs_at_save": int(self.num_envs),
            "config": self.config,
            "student_cfg_path": self.student_cfg.get("cfg"),
            "teacher_cfg_path": self.teacher_cfg.get("cfg"),
        }
        torch.save(checkpoint, filename)
        try:
            config_filename = str(pathlib.Path(filename).with_suffix(".yaml"))
            with open(config_filename, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)
        except Exception as exc:
            if self.rank == 0:
                print(f"Warning: failed to save checkpoint config YAML next to '{filename}': {exc}")

    def load_networks(self, params):
        builder = ModelBuilder()
        return builder.load(params)

    def load_yaml(self, cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
