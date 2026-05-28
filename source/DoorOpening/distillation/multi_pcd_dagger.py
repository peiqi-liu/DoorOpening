import json
import os
import math
import pathlib
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

from isaaclab.utils.math import quat_apply, transform_points
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES
from DoorOpening.assets.door.multi_door_cfg import asset_family_ids as door_asset_family_ids
from DoorOpening.assets.door.multi_door_cfg import asset_paths as door_asset_paths
from DoorOpening.assets.door.multi_door_cfg import board_bboxes as door_board_bboxes
from DoorOpening.assets.door.multi_door_cfg import motion_family_ids, motion_traj_paths
from DoorOpening.assets.glorbot.glorbot_cfg import glorbot_urdf_path
from DoorOpening.model.transformer import PCDTransformer, strip_prefix_from_state_dict
from DoorOpening.tasks.dooropening.contact_force_utils import get_filtered_contact_force_w
from DoorOpening.utils.camera_utils import (
    build_pinhole_intrinsics,
    crop_local_pcd,
    depth_to_pointcloud,
    simulate_depth_cam_render_from_pose,
    simulate_lidar_render_from_pose,
)
from DoorOpening.utils.extract_pointcloud_from_articulation import (
    FrankaLeapSampler,
    build_first_visual_link_pointcloud_cache,
    compose_cached_link_pointcloud_world,
)
from DoorOpening.utils.glorbot_collision_checker import GlorbotCollisionChecker
from DoorOpening.utils.pose_utils import world_to_local
from DoorOpening.utils.viser_pt import (
    format_iterated_record_path,
    prepare_pointcloud,
    prepare_world_points_from_local,
)


def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted_state_dict[key] = value
            continue

        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        key_with_orig_mod = ".".join(parts)
        if key_with_orig_mod in model_state_dict:
            adjusted_state_dict[key_with_orig_mod] = value
            continue

        key_no_orig_mod = key.replace("_orig_mod.", "")
        if key_no_orig_mod in model_state_dict:
            adjusted_state_dict[key_no_orig_mod] = value
            continue

        adjusted_state_dict[key] = value
    return adjusted_state_dict


def clip_teacher_obs(obs: torch.Tensor, clip_obs: float) -> torch.Tensor:
    if math.isfinite(clip_obs):
        return torch.clamp(obs, -clip_obs, clip_obs)
    return obs


class Dagger:
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
        if base_action_dim + arm_action_dim + hand_action_dim != self.num_actions:
            raise ValueError(
                "Action dimensions from base/arm/hand do not match env action dim: "
                f"{base_action_dim} + {arm_action_dim} + {hand_action_dim} != {self.num_actions}."
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

        self.lr = float(self.runtime_cfg.get("learning_rate", 1e-4))
        self.weight_decay = float(self.runtime_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(self.runtime_cfg.get("grad_clip", 1.0))
        self.num_iters = int(self.runtime_cfg.get("num_iters", 1_000_000))
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
        self.pointcloud_source = str(self.runtime_cfg.get("pointcloud_source", "sampler")).lower()
        self.robot_pcd_num_points = self.runtime_cfg.get("robot_num_points")
        self.sampler_render_cfg = self.runtime_cfg.get("sampler_render", {})
        self.sampler_render_inflate_px = int(self.sampler_render_cfg.get("inflate_px", 2))
        self.sampler_render_jitter_std_m = float(self.sampler_render_cfg.get("jitter_std_m", 0.004))
        self.sampler_render_clip_mode = str(self.sampler_render_cfg.get("clip_mode", "post"))
        self.sampler_render_jitter_mode = str(self.sampler_render_cfg.get("jitter_mode", "xyz"))
        self.sampler_render_use_compile = bool(self.sampler_render_cfg.get("use_compile", True))
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
        self.robot_pointcloud_filter_cfg = dict(self.runtime_cfg.get("robot_pointcloud_filter", {}))
        self.robot_pointcloud_filter_enabled = bool(self.robot_pointcloud_filter_cfg.get("enabled", True))
        self.robot_pointcloud_sdf_cutoff = float(self.robot_pointcloud_filter_cfg.get("sdf_cutoff", 0.02))
        self.robot_pointcloud_filter_max_points_per_process = int(
            self.robot_pointcloud_filter_cfg.get("max_points_per_process", 5000)
        )
        self.append_robot_gt_to_policy_cloud = bool(self.runtime_cfg.get("append_robot_gt_to_policy_cloud", True))
        self.robot_gt_policy_points = self.runtime_cfg.get("robot_gt_policy_points")
        # Runtime controls for optional raw point-cloud replay dumps.
        self.viser_cfg = dict(self.runtime_cfg.get("viser", {}))
        self.viser_raw_cfg = dict(self.viser_cfg.get("raw", {}))
        self.viser_env_id = int(self.viser_cfg.get("env_id", 0))
        self.viser_raw_enabled = self.rank == 0 and bool(self.viser_raw_cfg.get("enabled", False))
        self.viser_raw_save_interval = max(
            1,
            int(self.viser_raw_cfg.get("save_interval", self.viser_cfg.get("raw_interval", 1000))),
        )
        self.viser_raw_max_points = int(self.viser_raw_cfg.get("max_points", 12_000))
        self.viser_raw_max_frames = max(0, int(self.viser_raw_cfg.get("max_frames", 0)))
        if self.pointcloud_source not in {"sampler", "depth", "lidar"}:
            raise ValueError(f"Unsupported pointcloud_source '{self.pointcloud_source}'.")
        if self.teacher_forcing_warmup_iters < 0:
            raise ValueError("teacher_forcing_warmup_iters must be non-negative.")
        if self.teacher_forcing_transition_iters < 0:
            raise ValueError("teacher_forcing_transition_iters must be non-negative.")
        if not 0.0 <= self.teacher_forcing_min_beta <= 1.0:
            raise ValueError("teacher_forcing_min_beta must be in [0, 1].")

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
        self.completed_rewards = deque(maxlen=self.games_to_track)
        self.completed_lengths = deque(maxlen=self.games_to_track)
        self.completed_successes = deque(maxlen=self.games_to_track)
        self.completed_successes_by_family = {
            family_name: deque(maxlen=self.games_to_track)
            for family_name in DOOR_FAMILY_NAMES
        }
        self.student_update_steps = 0
        self.last_local_update_batch_size = 0
        self.last_global_update_batch_size = 0
        self.latest_student_proprio_vector = None
        self.latest_aux_input_vector = None
        self.latest_aux_target_vector = None
        self.push_pull_condition_enabled = False
        self.push_pull_condition_obs_key = "push_pull_cond"
        self.push_pull_condition_source = "oracle"
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
        self.latest_push_pull_belief_hist_delta_1500ms = 0.0
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
            "actions_num": self.num_actions,
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
        self.door_pcd_num_points = int(student_cfg_data.pop("door_pcd_num_points", 4096))
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
        self.force_prediction_enabled = bool(getattr(self.student_model, "force_prediction_enabled", False))
        self.force_prediction_weight = float(getattr(self.student_model, "force_prediction_weight", 0.0))
        self.force_output_dim = int(getattr(self.student_model, "force_output_dim", 0))
        if self.student_model.action_head.out_features != self.num_actions:
            raise ValueError(
                f"Student action_dim ({self.student_model.action_head.out_features}) "
                f"does not match env action dim ({self.num_actions})."
        )
        self._init_force_prediction_training_state()
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

        student_ckpt = self.student_cfg.get("ckpt")
        if student_ckpt is not None:
            self.load_student_weights(student_ckpt)

        all_state_encoder_keys = tuple(
            key
            for key, cfg in self.student_model.state_encoders_cfg.items()
            if cfg.get("use_state", False)
        )
        self.state_encoders_keys = tuple(
            key
            for key in all_state_encoder_keys
            if key not in self.proprio_temporal_covered_state_keys and key != self.push_pull_condition_obs_key
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
        self.aux_buffer = None
        if self.has_aux_input:
            self.aux_buffer = torch.zeros((self.num_envs, self.aux_input_dim), dtype=torch.float32, device=self.device)
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
            self.local_pcd_points = list(local_pcd_cfg.get("num_points", [self.door_pcd_num_points, 0, 0])[:3])
        if self.robot_gt_policy_points is None and len(self.local_pcd_points) >= 3:
            self.robot_gt_policy_points = int(self.local_pcd_points[2])
        if self.robot_gt_policy_points is None:
            self.robot_gt_policy_points = 0
        self.robot_gt_policy_points = max(0, int(self.robot_gt_policy_points))

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

    def _init_force_prediction_training_state(self):
        cfg = dict(self.runtime_cfg.get("force_training", {}) or {})
        self.force_prediction_sensor_name = str(cfg.get("sensor_name", "contact_forces_door2"))
        self.force_prediction_target_frame = str(cfg.get("target_frame", "base")).lower()
        if self.force_prediction_target_frame not in {"base", "world"}:
            raise ValueError(
                "force_prediction.target_frame must be one of ['base', 'world']."
            )
        self.force_prediction_loss_type = str(cfg.get("loss_type", "direction_contrastive")).lower()
        if self.force_prediction_loss_type not in {"mse", "smooth_l1", "direction_contrastive"}:
            raise ValueError(
                "force_prediction.loss_type must be one of ['mse', 'smooth_l1', 'direction_contrastive']."
            )
        self.force_prediction_contrastive_temperature = float(cfg.get("contrastive_temperature", 0.1))
        if self.force_prediction_contrastive_temperature <= 0.0:
            raise ValueError("force_prediction.contrastive_temperature must be positive.")

        self.latest_force_angle_deg = None
        self.latest_filtered_handle_force_norm_mean = 0.0
        self.latest_filtered_handle_force_norm_max = 0.0

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
        legacy_semantics = {
            "PartNetv5_plus": ("left", "pull"),
            "PartNetv8_plus": ("left", "push"),
            "PartNetv6_plus": ("right", "pull"),
            "PartNetv7_plus": ("right", "push"),
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
                fallback = legacy_semantics.get(family_name)
                if fallback is not None:
                    fallback_side, fallback_direction = fallback
                    if handle_side is None:
                        handle_side = fallback_side
                    if opening_direction is None:
                        opening_direction = fallback_direction

            semantics_by_family_name[family_name] = (handle_side, opening_direction)
        return semantics_by_family_name

    def _init_push_pull_semantics_and_targets(self):
        self.push_pull_family_one_hot = None
        self.mode_family_direction_ids = None
        self.latest_fraction_push = 0.0
        self.latest_fraction_pull = 0.0
        if not (
            self.push_pull_condition_enabled
            or self.push_pull_condition_perturb_enabled
            or self.mode_prediction_enabled
        ):
            return

        if self.mode_prediction_enabled and self.num_modes != 2:
            raise RuntimeError(
                f"Push/pull direction prediction expects num_modes=2, got {self.num_modes}."
            )

        self.mode_family_semantics = self._load_family_mode_semantics()
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

    def _get_contact_sensor_force_tensor_world(self, sensor_name):
        scene = getattr(self.ov_env, "scene", None)
        sensors = None if scene is None else getattr(scene, "sensors", None)
        if sensors is None or sensor_name not in sensors:
            raise RuntimeError(f"Contact sensor '{sensor_name}' is required but not available.")
        # force_matrix_w gives filtered contact force between Door/link_2 and robot hand links.
        # net_forces_w is intentionally not used because it is the total contact force on the handle body.
        force_world = get_filtered_contact_force_w(
            sensors[sensor_name],
            expected_num_envs=self.num_envs,
        )
        if force_world.ndim != 2 or force_world.shape[-1] != 3:
            raise RuntimeError(
                f"Expected filtered contact force shape [N, 3], got {tuple(force_world.shape)}"
            )
        force_norm = torch.linalg.vector_norm(force_world, dim=-1)
        self.latest_filtered_handle_force_norm_mean = float(force_norm.mean().detach().cpu().item())
        self.latest_filtered_handle_force_norm_max = float(force_norm.max().detach().cpu().item())
        return force_world.to(device=self.device, dtype=torch.float32)

    def _aggregate_contact_force_tensor(self, contact_forces):
        if contact_forces.ndim == 2:
            return contact_forces
        return contact_forces.reshape(contact_forces.shape[0], -1, 3).sum(dim=1)

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

    def _prepare_direction_prediction_tensors(self, mode_logits):
        if mode_logits.ndim < 2:
            raise RuntimeError(f"Expected direction logits to have a class dimension, got {tuple(mode_logits.shape)}.")
        if mode_logits.shape[-1] != self.num_modes:
            raise RuntimeError(
                f"Direction logits last dim ({mode_logits.shape[-1]}) does not match num_modes ({self.num_modes})."
            )

        prediction_leading_shape = mode_logits.shape[:-1]
        direction_target = self.mode_family_direction_ids[self.env_family_ids.long()]
        direction_valid = direction_target >= 0
        rollout_step_ids = self._get_rollout_step_ids()
        active_mask = self._get_active_rollout_mask(rollout_step_ids)

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

    def _compute_mode_prediction_loss(self, mode_logits):
        if not self.mode_prediction_loss_enabled:
            return None
        if self.mode_family_direction_ids is None:
            raise RuntimeError("Mode prediction targets are not initialized.")

        mode_logits, direction_target, direction_valid, active_mask, rollout_step_ids, _ = (
            self._prepare_direction_prediction_tensors(mode_logits)
        )
        if not torch.any(direction_valid):
            raise RuntimeError("No valid direction targets are available for mode prediction.")

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
            direction_pred = mode_logits[valid_mask].argmax(dim=-1)
            direction_correct = (direction_pred == direction_target[valid_mask]).float()
            self.latest_mode_direction_acc = float(direction_correct.mean().detach().cpu().item())
        else:
            direction_loss = mode_logits.mean() * 0.0
            self.latest_mode_direction_acc = None
        return direction_loss

    def _get_force_prediction_target_raw(self):
        contact_forces = self._get_contact_sensor_force_tensor_world(self.force_prediction_sensor_name)
        force_world = self._aggregate_contact_force_tensor(contact_forces)
        if force_world.ndim != 2 or force_world.shape[-1] != 3:
            raise RuntimeError(
                f"Expected force prediction target shape [N, 3], got {tuple(force_world.shape)}"
            )
        if self.force_prediction_target_frame == "world":
            return force_world
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        return world_to_local(force_world.unsqueeze(1), None, robot_base_quat_w).squeeze(1)

    def _normalize_force_direction(self, force_tensor):
        force_mag = torch.linalg.vector_norm(force_tensor, dim=-1, keepdim=True)
        return force_tensor / force_mag.clamp_min(1.0e-6)

    def _compute_force_prediction_loss(self, force_pred):
        if not self.force_prediction_enabled:
            return None

        force_target_raw = self._get_force_prediction_target_raw()
        if force_pred.ndim == 3:
            force_pred = force_pred[:, 0, :]
        if force_pred.shape[-1] != force_target_raw.shape[-1]:
            raise RuntimeError(
                f"Force prediction head output dim ({force_pred.shape[-1]}) does not match target dim ({force_target_raw.shape[-1]})."
            )

        rollout_step_ids = self._get_rollout_step_ids()
        active_mask = self._get_active_rollout_mask(rollout_step_ids)
        valid_mask = active_mask
        if not torch.any(valid_mask):
            loss = force_pred.mean() * 0.0
            self.latest_force_angle_deg = None
            return loss
        step_weights = self._get_direction_step_weights(rollout_step_ids[valid_mask])

        target_dir = self._normalize_force_direction(force_target_raw)
        pred_dir = self._normalize_force_direction(force_pred)

        if self.force_prediction_loss_type == "direction_contrastive":
            pred_dir_valid = pred_dir[valid_mask]
            target_dir_valid = target_dir[valid_mask]
            cosine_valid = torch.nn.functional.cosine_similarity(pred_dir_valid, target_dir_valid, dim=-1, eps=1.0e-8)

            if pred_dir_valid.shape[0] >= 2:
                logits = pred_dir_valid @ target_dir_valid.T
                logits = logits / self.force_prediction_contrastive_temperature
                labels = torch.arange(pred_dir_valid.shape[0], device=self.device)
                row_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                col_loss = torch.nn.functional.cross_entropy(logits.T, labels, reduction="none")
                weighted_row_loss = (row_loss * step_weights).sum() / step_weights.sum().clamp_min(1.0e-6)
                weighted_col_loss = (col_loss * step_weights).sum() / step_weights.sum().clamp_min(1.0e-6)
                loss = 0.5 * (weighted_row_loss + weighted_col_loss)
            else:
                loss = ((1.0 - cosine_valid) * step_weights).sum() / step_weights.sum().clamp_min(1.0e-6)

            angle_deg = torch.rad2deg(torch.acos(cosine_valid.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)))
            self.latest_force_angle_deg = float(angle_deg.mean().detach().cpu().item())
        else:
            if self.force_prediction_loss_type == "smooth_l1":
                per_env_loss = torch.nn.functional.smooth_l1_loss(
                    force_pred, force_target_raw, reduction="none"
                ).mean(dim=-1)
            else:
                per_env_loss = torch.nn.functional.mse_loss(force_pred, force_target_raw, reduction="none").mean(dim=-1)
            loss = (per_env_loss[valid_mask] * step_weights).sum() / step_weights.sum().clamp_min(1.0e-6)

            cosine_all = torch.nn.functional.cosine_similarity(pred_dir, target_dir, dim=-1, eps=1.0e-8)
            angle_deg = torch.rad2deg(torch.acos(cosine_all.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)))
            self.latest_force_angle_deg = float(angle_deg[valid_mask].mean().detach().cpu().item())

        return loss

    def _build_action_component_history_indices(self):
        target_joint_ids = torch.as_tensor(self.ov_env._robot_dof_idx, device=self.device, dtype=torch.long)
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
                ("full", torch.as_tensor(self.ov_env._robot_dof_idx, device=self.device, dtype=torch.long)),
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
        self.latest_push_pull_belief_hist_delta_1500ms = 0.0

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

        q_samples_control = q_samples_full[:, :, self.ov_env._robot_dof_idx]
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
        aux_handle = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
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
        if env_actions.ndim != 2 or env_actions.shape[-1] != self.num_actions:
            raise RuntimeError(f"Expected env action shape [N, {self.num_actions}], got {tuple(env_actions.shape)}.")

        student_actions = env_actions.clone()
        # Only the base action changes interface: env [wz, vx_w, vy_w] delta-action
        # units become student [vx_robot, vy_robot, wz_robot] velocity units.
        # Arm/hand stay in the env's normalized delta-action convention.
        student_actions[:, :3] = self._env_base_vector_to_robot_frame(
            env_actions[:, :3] * self.base_action_scale
        )
        return student_actions

    def _student_actions_to_env_actions(self, student_actions):
        if student_actions.ndim != 2 or student_actions.shape[-1] != self.num_actions:
            raise RuntimeError(
                f"Expected student action shape [N, {self.num_actions}], got {tuple(student_actions.shape)}."
            )

        env_actions = student_actions.clone()
        # The env applies dt in _pre_physics_step when integrating delta actions.
        # Keep arm/hand untouched here; only rotate/scale the base velocity command.
        env_actions[:, :3] = self._robot_base_vector_to_env_frame(student_actions[:, :3]) / self.base_action_scale
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
        # Temporal target features use the actual joint-position targets sent to
        # the PD controller, not the normalized delta policy actions.
        pd_targets = self.ov_env.applied_robot_dof_targets
        if pd_targets.ndim != 2:
            raise RuntimeError(f"Expected PD target tensor to be rank-2, got shape {tuple(pd_targets.shape)}.")
        return pd_targets

    def _build_temporal_derived_state_values(self, q_pos, target_t, base_vel, sample_cache=None):
        if not self.temporal_derived_state_specs:
            return {}

        q_t = q_pos[:, self.ov_env._robot_dof_idx]
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
        idx_1500ms = offset_to_index.get(1.5)
        if idx_1500ms is not None:
            delta = torch.linalg.vector_norm(
                belief_samples[:, idx_now, :] - belief_samples[:, idx_1500ms, :],
                dim=-1,
            )
            self.latest_push_pull_belief_hist_delta_1500ms = float(delta.mean().detach().cpu().item())

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

    def _seed_aux_buffer(self, env_ids=None):
        if self.aux_buffer is None:
            return
        if env_ids is None:
            self.aux_buffer.zero_()
            return
        if env_ids.numel() == 0:
            return
        self.aux_buffer[env_ids] = 0.0

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

        if self.rank == 0:
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
            print("[INFO] Rank 0 door family sample:", ", ".join(sample))
        self.env_board_bboxes = door_board_bboxes.to(device=self.device, dtype=torch.float32)[self.env_asset_idx]
        bbox_min = self.env_board_bboxes[:, 0]
        bbox_max = self.env_board_bboxes[:, 1]
        bbox_extent = (bbox_max - bbox_min).clamp_min(1e-4)
        self.wall_distractor_axis_order = torch.argsort(bbox_extent, dim=-1)
        self.wall_distractor_bbox_min_ordered = torch.gather(bbox_min, 1, self.wall_distractor_axis_order)
        self.wall_distractor_bbox_max_ordered = torch.gather(bbox_max, 1, self.wall_distractor_axis_order)
        unique_asset_idx = sorted(set(self.env_asset_idx.detach().cpu().tolist()))
        self.door_samplers = {
            idx: FrankaLeapSampler(door_asset_paths[idx], device=self.device, num_points=self.door_pcd_num_points)
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
        if self.robot_pcd_num_points is None:
            self.robot_pcd_num_points = self.door_pcd_num_points
        self.robot_pcd_num_points = int(self.robot_pcd_num_points)
        self.robot_sampler = FrankaLeapSampler(glorbot_urdf_path, device=self.device, num_points=self.robot_pcd_num_points)
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
        self.wall_distractors_enabled = bool(self.wall_distractor_cfg.get("enabled", True))
        self.wall_distractor_num_points = int(
            self.wall_distractor_cfg.get("num_points", max(256, self.door_pcd_num_points // 3))
        )
        self.wall_distractor_side_margin_scale_min, self.wall_distractor_side_margin_scale_max = map(
            float, self.wall_distractor_cfg.get("side_margin_scale", [0.35, 0.75])
        )
        side_margin_abs_range = self.wall_distractor_cfg.get("side_margin_m")
        if side_margin_abs_range is None:
            self.wall_distractor_side_margin_abs_min_m = None
            self.wall_distractor_side_margin_abs_max_m = None
        else:
            self.wall_distractor_side_margin_abs_min_m, self.wall_distractor_side_margin_abs_max_m = map(
                float, side_margin_abs_range
            )
        self.wall_distractor_bottom_margin_scale_min, self.wall_distractor_bottom_margin_scale_max = map(
            float, self.wall_distractor_cfg.get("bottom_margin_scale", [0.02, 0.08])
        )
        self.wall_distractor_gap_min_m, self.wall_distractor_gap_max_m = map(
            float, self.wall_distractor_cfg.get("edge_gap_m", [0.015, 0.04])
        )
        self.wall_distractor_depth_min_m, self.wall_distractor_depth_max_m = map(
            float, self.wall_distractor_cfg.get("depth_m", [0.10, 0.26])
        )
        self.wall_distractor_center_offset_min_m, self.wall_distractor_center_offset_max_m = map(
            float, self.wall_distractor_cfg.get("center_offset_m", [-0.20, 0.20])
        )
        self.wall_distractor_side_margin_min_m = float(self.wall_distractor_cfg.get("side_margin_min_m", 0.10))
        self.wall_distractor_face_jitter_m = float(self.wall_distractor_cfg.get("face_jitter_m", 0.004))
        self.wall_distractor_resample_each_step = bool(self.wall_distractor_cfg.get("resample_each_step", False))
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

        self.robot_camera_body_idx = None
        self.sampler_camera_spec = None

        if self.pointcloud_source in {"sampler", "depth"}:
            self.robot_camera_body_idx = int(self.ov_env.robot.find_bodies("x5_camera_link")[0][0])
            self.sampler_camera_spec = self._build_sampler_camera_spec()

        self.robot_lidar_body_idx = None
        if self.pointcloud_source == "lidar":
            lidar_body_name = str(getattr(self.ov_env.cfg, "pointcloud_lidar_body_name", "lidar"))
            lidar_body_ids = self.ov_env.robot.find_bodies(lidar_body_name)[0]
            if len(lidar_body_ids) == 0:
                raise ValueError(f"Could not find lidar body '{lidar_body_name}' on robot articulation.")
            self.robot_lidar_body_idx = int(lidar_body_ids[0])

        if self.lidar_num_points is None:
            self.lidar_num_points = self.door_pcd_num_points
        self.lidar_num_points = int(self.lidar_num_points)

        self.pointcloud_camera = getattr(self.ov_env, "pointcloud_camera", None)
        if self.pointcloud_source == "depth" and self.pointcloud_camera is None:
            raise ValueError("pointcloud_source='depth' requires DooropeningEnv to enable the pointcloud camera.")

    def _init_viser_debug_tools(self):
        """Initialize optional raw replay capture state used for point-cloud debugging."""
        self._viser_cached_ground_truth_pcd_world = None
        self._viser_pending_debug_frame = None
        self._viser_raw_streams = OrderedDict()

        # Keep replay outputs next to checkpoints by default so a run's artifacts stay together.
        default_record_dir = Path(self.nn_dir).parent if self.nn_dir is not None else Path(os.getcwd())
        configured_raw_path = self.viser_raw_cfg.get("path", self.viser_raw_cfg.get("raw_path"))
        if configured_raw_path is None:
            self.viser_raw_path = str(default_record_dir / "viser_replay.pt")
        else:
            configured_raw_path = Path(str(configured_raw_path))
            if not configured_raw_path.is_absolute():
                configured_raw_path = default_record_dir / configured_raw_path
            self.viser_raw_path = str(configured_raw_path)

        sim_cfg = getattr(self.ov_env.cfg, "sim", None)
        sim_dt = getattr(sim_cfg, "dt", None)
        self.viser_sim_dt = float(sim_dt) if sim_dt is not None else 1.0 / 30.0
        self.viser_env_step_dt = max(float(getattr(self.ov_env, "dt", self.viser_sim_dt)), 1e-6)
        if not self.viser_raw_enabled:
            return
        if self.viser_raw_max_points <= 0:
            raise ValueError("viser.raw.max_points must be positive when raw Viser capture is enabled.")

        raw_dir = os.path.dirname(self.viser_raw_path)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
        self._init_viser_raw_streams()

    def _get_viser_env_id(self):
        """Clamp the configured replay env index so debug capture never indexes outside the batch."""
        if self.num_envs <= 0:
            return 0
        return max(0, min(int(self.viser_env_id), self.num_envs - 1))

    def _init_viser_raw_streams(self):
        """Select one random env per door family for multi-door replay export."""
        self._viser_raw_streams = OrderedDict()
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            matching_envs = self.family_env_ids.get(int(family_id))
            if matching_envs is None:
                matching_envs = torch.nonzero(self.env_family_ids == int(family_id), as_tuple=False).squeeze(-1)
            if matching_envs.numel() == 0:
                continue
            env_offset = int(torch.randint(matching_envs.numel(), (1,), device=self.device).detach().cpu().item())
            env_id = int(matching_envs[env_offset].detach().cpu().item())
            self._viser_raw_streams[family_name] = {
                "family_id": int(family_id),
                "family_name": family_name,
                "env_id": env_id,
                "frames": [],
                "frame_count": 0,
                "chunk_index": 0,
                "latest_iteration": None,
            }

        if not self._viser_raw_streams:
            env_id = self._get_viser_env_id()
            self._viser_raw_streams["env"] = {
                "family_id": None,
                "family_name": "env",
                "env_id": env_id,
                "frames": [],
                "frame_count": 0,
                "chunk_index": 0,
                "latest_iteration": None,
            }

    def _viser_stream_env_ids(self):
        return [int(stream["env_id"]) for stream in self._viser_raw_streams.values()]

    def _select_viser_ground_truth_points(self, pointcloud_world):
        if not self.viser_raw_enabled or pointcloud_world is None:
            return None
        return {
            int(env_id): prepare_pointcloud(
                pointcloud_world,
                env_id=int(env_id),
                max_points=self.viser_raw_max_points,
            )
            for env_id in self._viser_stream_env_ids()
        }

    def _build_viser_raw_payload(self, stream):
        return {
            "format": "dooropening_viser_replay_v1",
            "frame_dt": float(self.viser_env_step_dt),
            "frames": stream["frames"],
        }

    def _trim_viser_raw_frames(self, stream):
        if self.viser_raw_max_frames > 0 and len(stream["frames"]) > self.viser_raw_max_frames:
            stream["frames"] = stream["frames"][-self.viser_raw_max_frames :]
        stream["frame_count"] = len(stream["frames"])

    def _maybe_flush_viser_raw_snapshot(self, iteration):
        if not self.viser_raw_enabled or self.rank != 0:
            return
        if self.viser_raw_save_interval <= 0:
            return
        if (int(iteration) + 1) % self.viser_raw_save_interval != 0:
            return
        for stream in self._viser_raw_streams.values():
            self._flush_viser_raw_stream(
                stream,
                chunk_complete=True,
                reason=f"save_interval {self.viser_raw_save_interval} reached at iteration {int(iteration)}",
            )

    def _flush_viser_raw_recording(self, chunk_complete, reason):
        if not self.viser_raw_enabled or self.rank != 0:
            return
        for stream in self._viser_raw_streams.values():
            self._flush_viser_raw_stream(stream, chunk_complete=chunk_complete, reason=reason)

    def _flush_viser_raw_stream(self, stream, chunk_complete, reason):
        if not self.viser_raw_enabled or self.rank != 0:
            return
        if stream["frame_count"] <= 0:
            return

        latest_iteration = int(stream["latest_iteration"])
        record_tag = f"{stream['family_name']}_chunk_{stream['chunk_index']:04d}_iter_{latest_iteration}"
        payload = self._build_viser_raw_payload(stream)
        torch.save(payload, self._format_iterated_record_path(self.viser_raw_path, record_tag))

        if self.rank == 0:
            status = "complete" if chunk_complete else "partial"
            print(
                "Saved {} Viser raw chunk {} for {} env {} with {} frames ({}).".format(
                    status,
                    stream["chunk_index"],
                    stream["family_name"],
                    stream["env_id"],
                    stream["frame_count"],
                    reason,
                )
            )

        stream["frames"] = []
        stream["frame_count"] = 0
        stream["latest_iteration"] = None

    def _prepare_viser_world_points_from_local(
        self,
        pointcloud_local,
        base_pos_w,
        base_quat_w,
        env_id,
        max_points,
        drop_zero_rows=False,
    ):
        """Convert a base-frame point cloud for one env into world coordinates for raw replay output."""
        return prepare_world_points_from_local(
            pointcloud_local,
            base_pos_w,
            base_quat_w,
            env_id=int(env_id),
            max_points=max_points,
            drop_zero_rows=drop_zero_rows,
        )

    def _prepare_viser_points(self, pointcloud, env_id, max_points, drop_zero_rows=False):
        """Extract one env's point cloud, sanitize it, and downsample it on CPU for replay export."""
        return prepare_pointcloud(
            pointcloud,
            env_id=int(env_id),
            max_points=max_points,
            drop_zero_rows=drop_zero_rows,
        )

    def _format_iterated_record_path(self, path_str, iteration):
        """Append a record-specific suffix to replay filenames so each saved capture gets its own file."""
        return format_iterated_record_path(path_str, iteration)

    def _maybe_update_viser_debug(
        self,
        iteration,
        robot_base_pos_w,
        robot_base_quat_w,
        ground_truth_pcd_world,
        robot_obs_pcd_base,
        policy_input_pcd_base,
        aux_prediction=None,
    ):
        """Append one replay frame for each selected multi-door family env."""
        if not self.viser_raw_enabled:
            return

        for stream in self._viser_raw_streams.values():
            env_id = int(stream["env_id"])
            if stream["frame_count"] == 0:
                stream["chunk_index"] += 1
            stream["latest_iteration"] = int(iteration)

            selected_ground_truth = None
            if isinstance(ground_truth_pcd_world, dict):
                selected_ground_truth = ground_truth_pcd_world.get(env_id)
            else:
                selected_ground_truth = ground_truth_pcd_world

            ground_truth_points_world = self._prepare_viser_points(
                selected_ground_truth,
                env_id,
                self.viser_raw_max_points,
            ).to(dtype=torch.float16)

            robot_obs_points_world = self._prepare_viser_world_points_from_local(
                robot_obs_pcd_base,
                robot_base_pos_w,
                robot_base_quat_w,
                env_id,
                self.viser_raw_max_points,
                drop_zero_rows=True,
            ).to(dtype=torch.float16)

            policy_input_points_world = None
            if policy_input_pcd_base is not None:
                policy_input_points_world = self._prepare_viser_world_points_from_local(
                    policy_input_pcd_base,
                    robot_base_pos_w,
                    robot_base_quat_w,
                    env_id,
                    self.viser_raw_max_points,
                    drop_zero_rows=True,
                ).to(dtype=torch.float16)

            aux_prediction_base = (
                None
                if aux_prediction is None
                else self._aux_to_2d(aux_prediction)[env_id].detach().cpu().to(dtype=torch.float32)
            )
            frame_record = {
                "pointclouds": {
                    "ground_truth": ground_truth_points_world,
                    "robot_obs": robot_obs_points_world,
                    "policy_input": policy_input_points_world,
                },
            }
            if aux_prediction_base is not None:
                frame_record["aux_prediction"] = aux_prediction_base
                frame_record["robot_base_pos_w"] = robot_base_pos_w[env_id].detach().cpu().to(dtype=torch.float32)
                frame_record["robot_base_quat_w"] = robot_base_quat_w[env_id].detach().cpu().to(dtype=torch.float32)
            stream["frames"].append(frame_record)
            self._trim_viser_raw_frames(stream)
        self._maybe_flush_viser_raw_snapshot(iteration)

    def _close_viser_debug_tools(self):
        """Flush any in-progress raw replay chunk on shutdown."""
        self._flush_viser_raw_recording(chunk_complete=False, reason="shutdown")

    def _extract_model_state(self, weights):
        if isinstance(weights, dict):
            if "model" in weights:
                return weights["model"], weights
            if "state_dict" in weights:
                return weights["state_dict"], weights
            if "model_state_dict" in weights:
                return weights["model_state_dict"], weights
        return weights, None

    def _load_checkpoint_state(self, ckpt):
        try:
            return torch_ext.load_checkpoint(ckpt)
        except Exception:
            return torch.load(ckpt, map_location="cpu")

    def set_teacher_weights(self, ckpt, model=None, strict=True, allow_adjust=True):
        if model is None:
            model = self.teacher_model
        if model is None:
            raise RuntimeError("Teacher model is not initialized.")
        weights = self._load_checkpoint_state(ckpt)
        state_dict, meta = self._extract_model_state(weights)
        if allow_adjust:
            state_dict = adjust_state_dict_keys(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=strict)
        if meta is not None and "running_mean_std" in meta:
            model.running_mean_std.load_state_dict(meta["running_mean_std"])

    def load_student_weights(self, ckpt):
        weights = torch.load(ckpt, map_location="cpu")
        state_dict, _ = self._extract_model_state(weights)
        state_dict = strip_prefix_from_state_dict(state_dict)
        self.student_model.load_state_dict(state_dict, strict=False)
        if isinstance(weights, dict):
            if "optimizer_state_dict" in weights and not self.play_policy:
                try:
                    self.optimizer.load_state_dict(weights["optimizer_state_dict"])
                except Exception as exc:
                    if self.rank == 0:
                        print(f"Warning: failed to load optimizer state from '{ckpt}': {exc}")
            if "frame" in weights:
                self.frame = int(weights["frame"])
            if "epoch" in weights:
                self.epoch_num = int(weights["epoch"])
            if "student_update_steps" in weights:
                self.student_update_steps = int(weights["student_update_steps"])

            resume_iteration = weights.get("iteration")
            if resume_iteration is None:
                saved_num_envs = int(weights.get("num_envs_at_save", self.num_envs))
                if saved_num_envs <= 0:
                    saved_num_envs = int(self.num_envs)
                resume_iteration = self.frame // max(1, saved_num_envs)
            if resume_iteration is None and not self.play_policy and "student_update_steps" in weights:
                resume_iteration = int(weights["student_update_steps"])
            self.resume_iteration = int(resume_iteration)
            self._resumed_from_student_ckpt = True

            curriculum_step_count = weights.get("curriculum_step_count")
            if curriculum_step_count is None:
                curriculum_step_count = self.resume_iteration

            # Curriculum/reset scheduling in DooropeningEnv uses common_step_counter.
            if hasattr(self.ov_env, "common_step_counter"):
                self.ov_env.common_step_counter = int(curriculum_step_count)
            if hasattr(self.ov_env, "set_train_info"):
                self.ov_env.set_train_info(int(self.frame))
            elif hasattr(self.ov_env, "_rlgames_env_frames"):
                self.ov_env._rlgames_env_frames = int(self.frame)

        print(f"Loaded student checkpoint: {ckpt}")
        if self.rank == 0 and self._resumed_from_student_ckpt:
            print(
                "Resuming student training state from checkpoint: "
                f"iteration={self.resume_iteration}, curriculum_step_count={int(curriculum_step_count)}, frame={self.frame}, "
                f"student_update_steps={self.student_update_steps}"
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
        latest_targets = self._get_implemented_action_vector()
        if getattr(self, "multi_teacher_enabled", False):
            teacher_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
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
        height = int(camera_cfg.height)
        width = int(camera_cfg.width)
        focal_length = float(camera_cfg.spawn.focal_length)
        horizontal_aperture = float(camera_cfg.spawn.horizontal_aperture)
        vertical_aperture = camera_cfg.spawn.vertical_aperture
        near_m, far_m = camera_cfg.spawn.clipping_range
        intrinsics = build_pinhole_intrinsics(
            height=height,
            width=width,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=None if vertical_aperture is None else float(vertical_aperture),
            device=self.device,
            dtype=torch.float32,
        )
        return {
            "H": int(height),
            "W": int(width),
            "intrinsics": intrinsics,
            "near_m": float(near_m),
            "far_m": float(far_m),
        }

    def _get_sampler_camera_pose(self):
        camera_link_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_camera_body_idx]
        camera_link_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_camera_body_idx]
        # The sampler render currently uses the camera-link pose directly.
        return torch.cat([camera_link_pos_w, camera_link_quat_w[:, [1, 2, 3, 0]]], dim=-1)

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

        axis_order = self.wall_distractor_axis_order[env_ids]
        bbox_min_ordered = self.wall_distractor_bbox_min_ordered[env_ids]
        bbox_max_ordered = self.wall_distractor_bbox_max_ordered[env_ids]

        thickness_min, width_min, height_min = bbox_min_ordered.unbind(dim=-1)
        thickness_max, width_max, height_max = bbox_max_ordered.unbind(dim=-1)
        thickness_extent = (thickness_max - thickness_min).clamp_min(1e-4)
        width_extent = (width_max - width_min).clamp_min(1e-4)
        height_extent = (height_max - height_min).clamp_min(1e-4)

        def rand_range(low, high, shape):
            return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(float(low), float(high))

        thickness_center = 0.5 * (thickness_min + thickness_max)

        def sample_column_surfaces():
            column_depth = thickness_extent + rand_range(
                self.wall_distractor_depth_min_m,
                self.wall_distractor_depth_max_m,
                (env_count,),
            )
            # Shift the column along the door-thickness axis so recessed and protruding
            # jamb-like distractors are both represented.
            column_center = thickness_center + rand_range(
                self.wall_distractor_center_offset_min_m,
                self.wall_distractor_center_offset_max_m,
                (env_count,),
            )
            column_depth_half = 0.5 * column_depth
            return column_center - column_depth_half, column_center + column_depth_half

        def sample_column_width():
            if self.wall_distractor_side_margin_abs_min_m is not None:
                # In column mode, the side margin range becomes the column width range.
                column_width = rand_range(
                    self.wall_distractor_side_margin_abs_min_m,
                    self.wall_distractor_side_margin_abs_max_m,
                    (env_count,),
                )
            else:
                column_width = (
                    width_extent
                    * rand_range(
                        self.wall_distractor_side_margin_scale_min,
                        self.wall_distractor_side_margin_scale_max,
                        (env_count,),
                    )
                    + self.wall_distractor_side_margin_min_m
                )
            return torch.maximum(column_width, torch.full_like(column_width, self.wall_distractor_side_margin_min_m))

        def sample_column_bounds(attach_on_right):
            edge_gap = rand_range(self.wall_distractor_gap_min_m, self.wall_distractor_gap_max_m, (env_count,))
            column_width = sample_column_width()
            bottom_margin = height_extent * rand_range(
                self.wall_distractor_bottom_margin_scale_min,
                self.wall_distractor_bottom_margin_scale_max,
                (env_count,),
            )
            column_min_surface, column_max_surface = sample_column_surfaces()
            column_inner_width = torch.where(attach_on_right, width_max + edge_gap, width_min - edge_gap)
            column_outer_width = torch.where(
                attach_on_right,
                column_inner_width + column_width,
                column_inner_width - column_width,
            )
            column_width_lo = torch.minimum(column_inner_width, column_outer_width)
            column_width_hi = torch.maximum(column_inner_width, column_outer_width)
            column_height_lo = height_min - bottom_margin
            column_height_hi = height_max
            return (
                column_min_surface,
                column_max_surface,
                column_width_lo,
                column_width_hi,
                column_height_lo,
                column_height_hi,
            )

        left_column_bounds = sample_column_bounds(
            torch.zeros((env_count,), dtype=torch.bool, device=self.device)
        )
        right_column_bounds = sample_column_bounds(
            torch.ones((env_count,), dtype=torch.bool, device=self.device)
        )
        left_column_min_surface, left_column_max_surface, left_column_width_lo, left_column_width_hi, left_column_height_lo, left_column_height_hi = left_column_bounds
        right_column_min_surface, right_column_max_surface, right_column_width_lo, right_column_width_hi, right_column_height_lo, right_column_height_hi = right_column_bounds

        wall_points_ordered = torch.empty((env_count, num_points, 3), dtype=torch.float32, device=self.device)
        face_ids = torch.randint(0, 4, (env_count, num_points), device=self.device)
        use_right_column = torch.randint(0, 2, (env_count, num_points), device=self.device, dtype=torch.int64).bool()
        if num_points >= 2:
            # Keep both jamb sides populated for each env while leaving the top lintel clear.
            use_right_column[:, 0] = False
            use_right_column[:, 1] = True

        column_min_surface = torch.where(
            use_right_column,
            right_column_min_surface.unsqueeze(1),
            left_column_min_surface.unsqueeze(1),
        )
        column_max_surface = torch.where(
            use_right_column,
            right_column_max_surface.unsqueeze(1),
            left_column_max_surface.unsqueeze(1),
        )
        column_width_lo = torch.where(
            use_right_column,
            right_column_width_lo.unsqueeze(1),
            left_column_width_lo.unsqueeze(1),
        )
        column_width_hi = torch.where(
            use_right_column,
            right_column_width_hi.unsqueeze(1),
            left_column_width_hi.unsqueeze(1),
        )
        column_height_lo = torch.where(
            use_right_column,
            right_column_height_lo.unsqueeze(1),
            left_column_height_lo.unsqueeze(1),
        )
        column_height_hi = torch.where(
            use_right_column,
            right_column_height_hi.unsqueeze(1),
            left_column_height_hi.unsqueeze(1),
        )

        wall_points_ordered[..., 0] = column_min_surface + torch.rand(
            (env_count, num_points), device=self.device
        ) * (column_max_surface - column_min_surface).clamp_min(1e-4)
        wall_points_ordered[..., 1] = column_width_lo + torch.rand(
            (env_count, num_points), device=self.device
        ) * (column_width_hi - column_width_lo).clamp_min(1e-4)
        wall_points_ordered[..., 2] = column_height_lo + torch.rand(
            (env_count, num_points), device=self.device
        ) * (column_height_hi - column_height_lo).clamp_min(1e-4)

        thickness_min_face = face_ids == 0
        thickness_max_face = face_ids == 1
        width_min_face = face_ids == 2
        width_max_face = face_ids == 3

        wall_points_ordered[..., 0] = torch.where(
            thickness_min_face,
            column_min_surface,
            wall_points_ordered[..., 0],
        )
        wall_points_ordered[..., 0] = torch.where(
            thickness_max_face,
            column_max_surface,
            wall_points_ordered[..., 0],
        )
        wall_points_ordered[..., 1] = torch.where(
            width_min_face,
            column_width_lo,
            wall_points_ordered[..., 1],
        )
        wall_points_ordered[..., 1] = torch.where(
            width_max_face,
            column_width_hi,
            wall_points_ordered[..., 1],
        )

        if self.wall_distractor_face_jitter_m > 0.0:
            thickness_face_mask = (thickness_min_face | thickness_max_face).to(torch.float32)
            width_face_mask = (width_min_face | width_max_face).to(torch.float32)
            wall_points_ordered[..., 0] += thickness_face_mask * rand_range(
                -self.wall_distractor_face_jitter_m,
                self.wall_distractor_face_jitter_m,
                (env_count, num_points),
            )
            wall_points_ordered[..., 1] += width_face_mask * rand_range(
                -self.wall_distractor_face_jitter_m,
                self.wall_distractor_face_jitter_m,
                (env_count, num_points),
            )

        wall_points_base = torch.zeros_like(wall_points_ordered)
        wall_points_base.scatter_(
            2,
            axis_order.unsqueeze(1).expand(-1, wall_points_ordered.shape[1], -1),
            wall_points_ordered,
        )
        return wall_points_base

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
            num_points=self.robot_pcd_num_points,
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

    def _sample_cached_door_pointcloud_world(self):
        door_pcd_world = torch.zeros(
            (self.num_envs, self.door_pcd_num_points, 3),
            dtype=torch.float32,
            device=self.device,
        )
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
                num_points=self.door_pcd_num_points,
            )
            door_pcd_world[env_ids] = asset_pcd_world
        return door_pcd_world

    def _sample_scene_pointcloud_world_cached(self):
        door_pcd_world = self._sample_cached_door_pointcloud_world()
        robot_pcd_world = self._sample_robot_pointcloud_world_sampler()
        scene_parts = [door_pcd_world, robot_pcd_world]
        wall_pcd_world = self._sample_wall_pointcloud_world()
        if wall_pcd_world.shape[1] > 0:
            scene_parts.append(wall_pcd_world)
        return torch.cat(scene_parts, dim=1)

    def _sample_scene_pointcloud_world_sampler(self):
        door_base_pos_w = self.ov_env.door.data.body_pos_w[:, self.door_base_body_idx]
        door_base_quat_w = self.ov_env.door.data.body_quat_w[:, self.door_base_body_idx]
        door_joint_pos = self.ov_env.door.data.joint_pos

        door_pcd_world = torch.zeros(
            (self.num_envs, self.door_pcd_num_points, 3),
            dtype=torch.float32,
            device=self.device,
        )
        for asset_idx, sampler in self.door_samplers.items():
            env_ids = self.door_sampler_env_ids.get(asset_idx)
            if env_ids is None:
                env_ids = torch.nonzero(self.env_asset_idx == asset_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            local_pcd = sampler.sample(door_joint_pos[env_ids])
            quat = door_base_quat_w[env_ids].unsqueeze(1).expand(-1, local_pcd.shape[1], -1)
            world_pcd = quat_apply(quat, local_pcd) + door_base_pos_w[env_ids].unsqueeze(1)
            door_pcd_world[env_ids] = world_pcd
        robot_pcd_world = self._sample_robot_pointcloud_world_sampler()
        scene_parts = [door_pcd_world, robot_pcd_world]
        wall_pcd_world = self._sample_wall_pointcloud_world()
        if wall_pcd_world.shape[1] > 0:
            scene_parts.append(wall_pcd_world)
        return torch.cat(scene_parts, dim=1)

    def _render_lidar_scene_pointcloud_base(self, scene_pcd_world, robot_base_pos_w, robot_base_quat_w):
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
        )
        return world_to_local(rendered_pcd_world, robot_base_pos_w, robot_base_quat_w)

    def _sample_door_pointcloud_base_sampler(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        gt_scene_pcd_world = self._sample_scene_pointcloud_world_sampler()
        if self.viser_raw_enabled:
            # Keep only the selected family envs on CPU so Viser debug does not retain an
            # extra full batched scene pointcloud on GPU during training/debug runs.
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(gt_scene_pcd_world)
        rendered_pcd_world, _ = simulate_depth_cam_render_from_pose(
            pcd=gt_scene_pcd_world,
            camera_pose=self._get_sampler_camera_pose(),
            num_points=self.door_pcd_num_points,
            inflate_px=self.sampler_render_inflate_px,
            jitter_std_m=self.sampler_render_jitter_std_m,
            cam_spec_dict=self.sampler_camera_spec,
            clip_mode=self.sampler_render_clip_mode,
            jitter_mode=self.sampler_render_jitter_mode,
            use_compile=self.sampler_render_use_compile,
        )
        scene_pointcloud_base = world_to_local(rendered_pcd_world, robot_base_pos_w, robot_base_quat_w)
        return scene_pointcloud_base

    def _sample_door_pointcloud_base_depth(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        depth = self.pointcloud_camera.data.output["distance_to_image_plane"]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth_image = depth.squeeze(-1)
        else:
            depth_image = depth

        door_pcd_camera = depth_to_pointcloud(
            depth=depth_image,
            intrinsics=self.pointcloud_camera.data.intrinsic_matrices,
            num_local_points=None,
        )

        door_pcd_world = transform_points(
            door_pcd_camera,
            pos=self.pointcloud_camera.data.pos_w,
            quat=self.pointcloud_camera.data.quat_w_ros,
        )
        scene_pcd_world = door_pcd_world
        wall_pcd_world = self._sample_wall_pointcloud_world()
        if wall_pcd_world.shape[1] > 0:
            scene_pcd_world = torch.cat([scene_pcd_world, wall_pcd_world], dim=1)
        if self.viser_raw_enabled:
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(scene_pcd_world)
        door_pcd_base = world_to_local(scene_pcd_world, robot_base_pos_w, robot_base_quat_w)
        # Filter floor points while preserving the batched layout expected by the cropper.
        floor_mask = door_pcd_base[..., 2] > 0.1
        door_pcd_base = door_pcd_base.clone()
        door_pcd_base[~floor_mask] = float("nan")
        return door_pcd_base

    def _sample_door_pointcloud_base_lidar(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        gt_scene_pcd_world = self._sample_scene_pointcloud_world_cached()
        if self.viser_raw_enabled:
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(gt_scene_pcd_world)

        return self._render_lidar_scene_pointcloud_base(gt_scene_pcd_world, robot_base_pos_w, robot_base_quat_w)

    def _sample_door_pointcloud_base(self):
        self._viser_cached_ground_truth_pcd_world = None
        if self.pointcloud_source == "sampler":
            door_pcd_base = self._sample_door_pointcloud_base_sampler()
        elif self.pointcloud_source == "depth":
            door_pcd_base = self._sample_door_pointcloud_base_depth()
        else:
            door_pcd_base = self._sample_door_pointcloud_base_lidar()
        door_pcd_base = self._filter_robot_points_base(door_pcd_base)
        return door_pcd_base

    def _build_local_pcd(self, door_pcd_base, palm_pos_base, robot_pcd_base=None):
        pcd_parts = []

        if self.local_pcd_points[0] > 0:
            base_crop, _ = crop_local_pcd(
                door_pcd_base,
                local_range=self.local_pcd_range[0],
                num_local_points=self.local_pcd_points[0],
                is_cylindrical=True,
                crop_center=self.zero_local_pcd_crop_center,
                x_direction_cutoff=self.local_pcd_x_direction_cutoff,
                log_name="base",
            )
            pcd_parts.append(base_crop)

        if self.local_pcd_points[1] > 0:
            palm_crop, _ = crop_local_pcd(
                door_pcd_base,
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
            and self.robot_gt_policy_points > 0
        ):
            if robot_pcd_base.shape[1] > self.robot_gt_policy_points:
                sample_idx = torch.linspace(
                    0,
                    robot_pcd_base.shape[1] - 1,
                    steps=self.robot_gt_policy_points,
                    device=robot_pcd_base.device,
                    dtype=torch.float32,
                ).round().to(dtype=torch.long)
                robot_pcd_base = robot_pcd_base[:, sample_idx]
            pcd_parts.append(robot_pcd_base)

        if not pcd_parts:
            raise ValueError("Student config requested local_pcd_t but no local point counts were configured.")
        return torch.cat(pcd_parts, dim=1)

    def _init_wandb(self, summaries_dir):
        summaries_path = pathlib.Path(summaries_dir).resolve()
        summaries_path.mkdir(parents=True, exist_ok=True)

        api_key = self.wandb_cfg.get("api_key") or os.getenv("WANDB_API_KEY")
        configured_project = self.wandb_cfg.get("project") or os.getenv("WANDB_PROJECT")
        entity = self.wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
        name = self.wandb_cfg.get("name") or os.getenv("WANDB_NAME")
        mode = self.wandb_cfg.get("mode") or os.getenv("WANDB_MODE")
        if mode is None:
            mode = "online" if (api_key or configured_project or entity or name) else "offline"

        if api_key and mode == "online":
            wandb.login(key=api_key)

        project = configured_project or "dooropening-pcd-dagger"
        notes = self.wandb_cfg.get("notes") or os.getenv("WANDB_NOTES")
        group = self.wandb_cfg.get("group") or os.getenv("WANDB_GROUP")
        job_type = self.wandb_cfg.get("job_type") or os.getenv("WANDB_JOB_TYPE") or "distillation"
        tags = self.wandb_cfg.get("tags") or os.getenv("WANDB_TAGS")
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        init_kwargs = {
            "project": project,
            "dir": str(summaries_path),
            "config": self.config,
            "job_type": job_type,
        }
        if entity:
            init_kwargs["entity"] = entity
        if name:
            init_kwargs["name"] = name
        if notes:
            init_kwargs["notes"] = notes
        if group:
            init_kwargs["group"] = group
        if mode:
            init_kwargs["mode"] = mode
        if tags:
            init_kwargs["tags"] = tags

        self.wandb_run = wandb.init(**init_kwargs)

    def _wandb_log(self, metrics, step):
        if not self.use_wandb or self.wandb_run is None or not metrics:
            return
        wandb.log(metrics, step=step)

    def _to_loggable_scalar(self, value):
        if isinstance(value, (bool, int, float)):
            return float(value)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().cpu().item())
        return None

    def _update_logged_env_metrics(self, extras):
        if not isinstance(extras, dict):
            return

        metrics = {}
        for key, value in extras.items():
            if not any(key.startswith(prefix) for prefix in self.logged_env_metric_prefixes):
                continue
            scalar_value = self._to_loggable_scalar(value)
            if scalar_value is None:
                continue
            metrics[key] = scalar_value

        if metrics:
            self.latest_env_log_metrics.update(metrics)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        wandb.finish()
        self.wandb_run = None

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

        palm_pos_base = world_to_local(palm_pos_w, robot_base_pos_w, robot_base_quat_w).squeeze(1)
        door_pcd_base = self._sample_door_pointcloud_base()
        robot_pcd_base = self._sample_robot_pointcloud_base_sampler() if self.append_robot_gt_to_policy_cloud else None
        target_t = self._get_implemented_action_vector()
        need_aux_target_vector = self.has_aux_input and (not self.play_policy and self.has_aux_prediction)
        aux_target_vector = (
            self._stack_aux_state_values(self._get_aux_state_values()) if need_aux_target_vector else None
        )
        if self.has_aux_input and self.aux_feedback_to_policy:
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

        if self.proprio_temporal_enabled:
            if self.proprio_temporal_obs_key is None:
                raise RuntimeError("temporal_state_encoders are enabled but no observation key is configured.")
            obs[self.proprio_temporal_obs_key] = self._build_proprio_temporal_obs(temporal_sample_cache)

        for key in self.pcd_encoders_keys:
            if key == "local_pcd_t":
                obs[key] = self._build_local_pcd(
                    door_pcd_base,
                    palm_pos_base,
                    robot_pcd_base=robot_pcd_base,
                )
            else:
                raise KeyError(f"Unsupported student pointcloud key '{key}' in config.")

        if self.viser_raw_enabled:
            self._viser_pending_debug_frame = {
                "iteration": iteration,
                "robot_base_pos_w": robot_base_pos_w,
                "robot_base_quat_w": robot_base_quat_w,
                "ground_truth_pcd_world": self._viser_cached_ground_truth_pcd_world,
                "robot_obs_pcd_base": door_pcd_base,
                "policy_input_pcd_base": obs.get("local_pcd_t"),
            }
            self._viser_cached_ground_truth_pcd_world = None

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

    def _compute_student_loss(self, student_output, teacher_actions, aux_target=None):
        target = teacher_actions
        if aux_target is not None:
            target = torch.cat([teacher_actions, aux_target], dim=-1)
        loss = self.student_model.compute_loss(student_output, target.unsqueeze(1))
        total_loss = loss["total"]
        action_loss = loss.get("action", total_loss)
        aux_loss = loss.get("aux")
        mode_loss = None
        force_loss = None
        if self.mode_prediction_enabled:
            if "mode_logits" not in student_output:
                raise RuntimeError("Mode prediction is enabled, but student output does not contain 'mode_logits'.")
            mode_loss = self._compute_mode_prediction_loss(student_output["mode_logits"])
            total_loss = total_loss + self.mode_weight * mode_loss
        if self.force_prediction_enabled:
            if "force" not in student_output:
                raise RuntimeError("Force prediction is enabled, but student output does not contain 'force'.")
            force_loss = self._compute_force_prediction_loss(student_output["force"])
            total_loss = total_loss + self.force_prediction_weight * force_loss
        return total_loss, action_loss, aux_loss, mode_loss, force_loss

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

    def _mean_completed_metric(self, values):
        if not values:
            return None
        return float(sum(values) / len(values))

    def _update_completed_episode_metrics(self, done_mask, timed_out):
        if done_mask.numel() == 0:
            return

        episode_rewards = self.current_rewards[done_mask, 0].detach().cpu().tolist()
        episode_lengths = self.current_lengths[done_mask].detach().cpu().tolist()
        episode_success_tensor = timed_out[done_mask].to(dtype=torch.float32)
        episode_successes = episode_success_tensor.detach().cpu().tolist()
        episode_family_ids = self.env_family_ids[done_mask].detach().cpu().tolist()

        self.completed_rewards.extend(float(value) for value in episode_rewards)
        self.completed_lengths.extend(float(value) for value in episode_lengths)
        self.completed_successes.extend(float(value) for value in episode_successes)
        for family_id, success_value in zip(episode_family_ids, episode_successes):
            family_name = DOOR_FAMILY_NAMES[int(family_id)]
            self.completed_successes_by_family[family_name].append(float(success_value))

    def _get_global_success_rates(self):
        num_families = len(DOOR_FAMILY_NAMES)
        stats = torch.zeros((1 + num_families, 2), dtype=torch.float64, device=self.device)
        stats[0, 0] = float(sum(self.completed_successes))
        stats[0, 1] = float(len(self.completed_successes))
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            values = self.completed_successes_by_family[family_name]
            stats[family_id + 1, 0] = float(sum(values))
            stats[family_id + 1, 1] = float(len(values))

        if self.use_ddp:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        success_rate = None
        if stats[0, 1] > 0:
            success_rate = float((stats[0, 0] / stats[0, 1]).detach().cpu().item())

        family_success_rates = {}
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            count = stats[family_id + 1, 1]
            if count > 0:
                family_success_rates[family_name] = float(
                    (stats[family_id + 1, 0] / count).detach().cpu().item()
                )
            else:
                family_success_rates[family_name] = None
        return success_rate, family_success_rates

    def _log(
        self,
        iteration,
        total_loss,
        action_loss,
        aux_loss,
        mode_loss,
        force_loss,
        teacher_forcing_beta,
    ):
        if iteration % self.log_interval != 0:
            return
        episode_reward = self._mean_completed_metric(self.completed_rewards)
        episode_length = self._mean_completed_metric(self.completed_lengths)
        env_step_dt = max(float(getattr(self.ov_env, "dt", 0.0)), 1e-6)
        episode_length_seconds = episode_length * env_step_dt if episode_length is not None else None
        success_rate, family_success_rates = self._get_global_success_rates()
        teacher_env_fraction = self._get_teacher_forcing_env_fraction()
        student_env_fraction = 1.0 - teacher_env_fraction
        iteration_time_ms = self._consume_timing_means()

        if self.rank == 0:
            print("=" * 10)
            print("ITERATION:", iteration)
            print("Total Loss:", float(total_loss.detach().cpu()))
            print("Action Loss:", float(action_loss.detach().cpu()))
            if aux_loss is not None:
                print("Aux Loss:", float(aux_loss.detach().cpu()))
            if mode_loss is not None:
                print("Direction Loss:", float(mode_loss.detach().cpu()))
            if force_loss is not None:
                print("Force Loss:", float(force_loss.detach().cpu()))
                if self.latest_force_angle_deg is not None:
                    print("Force Angle Deg:", self.latest_force_angle_deg)
            if self.latest_mode_direction_acc is not None:
                print("Direction Acc:", self.latest_mode_direction_acc)
            if self.mode_prediction_loss_enabled:
                print("Direction Window Acc:", self.latest_dir_window_acc)
                print("Direction Window Balanced Acc:", self.latest_dir_window_balanced_acc)
                print("Direction Window Push Acc:", self.latest_dir_window_push_acc)
                print("Direction Window Pull Acc:", self.latest_dir_window_pull_acc)
                print("Direction Window Num Push Labels:", self.latest_dir_window_num_push_labels)
                print("Direction Window Num Pull Labels:", self.latest_dir_window_num_pull_labels)
                print("Direction Window Num Push Preds:", self.latest_dir_window_num_push_preds)
                print("Direction Window Num Pull Preds:", self.latest_dir_window_num_pull_preds)
            if self.force_prediction_enabled:
                print("Filtered Handle Force Norm Mean:", self.latest_filtered_handle_force_norm_mean)
                print("Filtered Handle Force Norm Max:", self.latest_filtered_handle_force_norm_max)
            if self.observation_lag_enabled:
                print("Obs Lag Enabled:", bool(self.latest_obs_lag_enabled))
                print("Obs Lag Mean (ms):", self.latest_obs_lag_mean_ms)
                print("Obs Lag Min (ms):", self.latest_obs_lag_min_ms)
                print("Obs Lag Max (ms):", self.latest_obs_lag_max_ms)
            if self.push_pull_condition_enabled:
                print("Push/Pull Condition Source:", self.latest_push_pull_condition_source)
                print("Fraction Push:", self.latest_fraction_push)
                print("Fraction Pull:", self.latest_fraction_pull)
                print("Push/Pull Pred Entropy:", self.latest_push_pull_pred_entropy)
                if self.latest_push_pull_pred_acc is not None:
                    print("Push/Pull Pred Acc:", self.latest_push_pull_pred_acc)
                print("Push/Pull Perturbed To Push Count:", self.latest_push_pull_perturb_to_push_count)
                print("Push/Pull Perturbed To Pull Count:", self.latest_push_pull_perturb_to_pull_count)
            print("Temporal Aux Handle Enabled:", bool(self.temporal_aux_handle_enabled))
            print("Temporal Push/Pull Belief Enabled:", bool(self.temporal_push_pull_belief_enabled))
            if self.temporal_push_pull_belief_enabled:
                print("Push/Pull Belief Hist Entropy Now:", self.latest_push_pull_belief_hist_entropy_now)
                print("Push/Pull Belief Hist Entropy Mean:", self.latest_push_pull_belief_hist_entropy_mean)
                print("Push/Pull Belief Hist Delta 1500ms:", self.latest_push_pull_belief_hist_delta_1500ms)
            print("Teacher Forcing Beta:", teacher_forcing_beta)
            print("Teacher Rollout Env Fraction:", teacher_env_fraction)
            print("Student Rollout Env Fraction:", student_env_fraction)
            print("Student Update Steps:", self.student_update_steps)
            print("Last Local Update Batch Size:", self.last_local_update_batch_size)
            print("Last Global Update Batch Size:", self.last_global_update_batch_size)
            if episode_reward is not None:
                print("Episode Reward:", episode_reward)
            if episode_length is not None:
                print("Episode Length:", episode_length)
            if episode_length_seconds is not None:
                print("Episode Length (s):", episode_length_seconds)
            if success_rate is not None:
                print("Global Success Rate:", success_rate)
            for family_name, family_success_rate in family_success_rates.items():
                if family_success_rate is not None:
                    print(f"Global Success Rate/{family_name}:", family_success_rate)
            if iteration_time_ms is not None:
                print("Iteration Time (ms):", iteration_time_ms)
            # for key, value in sorted(self.latest_env_log_metrics.items()):
            #     print(f"{key}:", value)

        metrics = {
            "loss/total": float(total_loss.detach().cpu()),
            "loss/action": float(action_loss.detach().cpu()),
            "dist/world_size": self.world_size,
            "dist/update_steps": self.student_update_steps,
            "dist/last_local_update_batch_size": self.last_local_update_batch_size,
            "dist/last_global_update_batch_size": self.last_global_update_batch_size,
        }
        if aux_loss is not None:
            metrics["loss/aux"] = float(aux_loss.detach().cpu())
        if mode_loss is not None:
            metrics["loss/direction"] = float(mode_loss.detach().cpu())
        if force_loss is not None:
            metrics["loss/force"] = float(force_loss.detach().cpu())
            if self.latest_force_angle_deg is not None:
                metrics["stats/force_angle_deg"] = self.latest_force_angle_deg
        if self.latest_mode_direction_acc is not None:
            metrics["stats/dir_acc"] = self.latest_mode_direction_acc
        if self.mode_prediction_loss_enabled:
            metrics["stats/dir_window_acc"] = self.latest_dir_window_acc
            metrics["stats/dir_window_balanced_acc"] = self.latest_dir_window_balanced_acc
            metrics["stats/dir_window_push_acc"] = self.latest_dir_window_push_acc
            metrics["stats/dir_window_pull_acc"] = self.latest_dir_window_pull_acc
            metrics["stats/dir_window_num_push_labels"] = self.latest_dir_window_num_push_labels
            metrics["stats/dir_window_num_pull_labels"] = self.latest_dir_window_num_pull_labels
            metrics["stats/dir_window_num_push_preds"] = self.latest_dir_window_num_push_preds
            metrics["stats/dir_window_num_pull_preds"] = self.latest_dir_window_num_pull_preds
        if self.force_prediction_enabled:
            metrics["stats/filtered_handle_force_norm_mean"] = self.latest_filtered_handle_force_norm_mean
            metrics["stats/filtered_handle_force_norm_max"] = self.latest_filtered_handle_force_norm_max
        if self.observation_lag_enabled:
            metrics["timestamp/obs_lag_enabled"] = self.latest_obs_lag_enabled
            metrics["timestamp/obs_lag_mean_ms"] = self.latest_obs_lag_mean_ms
            metrics["timestamp/obs_lag_min_ms"] = self.latest_obs_lag_min_ms
            metrics["timestamp/obs_lag_max_ms"] = self.latest_obs_lag_max_ms
            for timestamp_ms, mean_age_ms in self.latest_obs_lag_effective_age_ms_by_timestamp.items():
                metrics[f"timestamp/obs_lag_effective_age_{timestamp_ms}ms"] = mean_age_ms
        metrics["stats/temporal_aux_handle_enabled"] = float(self.temporal_aux_handle_enabled)
        metrics["stats/temporal_push_pull_belief_enabled"] = float(self.temporal_push_pull_belief_enabled)
        metrics["stats/push_pull_belief_hist_entropy_now"] = self.latest_push_pull_belief_hist_entropy_now
        metrics["stats/push_pull_belief_hist_entropy_mean"] = self.latest_push_pull_belief_hist_entropy_mean
        metrics["stats/push_pull_belief_hist_delta_1500ms"] = self.latest_push_pull_belief_hist_delta_1500ms
        if self.push_pull_condition_enabled:
            metrics["stats/push_pull_condition_source"] = self.latest_push_pull_condition_source
            metrics["stats/fraction_push"] = self.latest_fraction_push
            metrics["stats/fraction_pull"] = self.latest_fraction_pull
            metrics["stats/push_pull_pred_entropy"] = self.latest_push_pull_pred_entropy
            if self.latest_push_pull_pred_acc is not None:
                metrics["stats/push_pull_pred_acc"] = self.latest_push_pull_pred_acc
            metrics["stats/push_pull_perturb_to_push_count"] = self.latest_push_pull_perturb_to_push_count
            metrics["stats/push_pull_perturb_to_pull_count"] = self.latest_push_pull_perturb_to_pull_count
        if episode_reward is not None:
            metrics["stats/episode_reward"] = episode_reward
        if episode_length is not None:
            metrics["stats/episode_length"] = episode_length
        if episode_length_seconds is not None:
            metrics["stats/episode_length_seconds"] = episode_length_seconds
        if success_rate is not None:
            metrics["success/success_rate"] = success_rate
        for family_name, family_success_rate in family_success_rates.items():
            if family_success_rate is not None:
                metrics[f"success/success_rate/{family_name}"] = family_success_rate
        if teacher_forcing_beta is not None:
            metrics["schedule/teacher_forcing_beta"] = teacher_forcing_beta
        metrics["schedule/teacher_rollout_env_fraction"] = teacher_env_fraction
        metrics["schedule/student_rollout_env_fraction"] = student_env_fraction
        if iteration_time_ms is not None:
            metrics["timing/iteration_ms"] = iteration_time_ms
        if self.latest_env_log_metrics:
            for key, value in self.latest_env_log_metrics.items():
                if key == "stats/success_rate" or key.startswith("stats/success_rate/"):
                    continue
                metrics[key] = value
        self._wandb_log(metrics, step=iteration)

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
                    self.aux_buffer[:] = aux_prediction_for_replay

                if self.viser_raw_enabled and self._viser_pending_debug_frame is not None:
                    self._maybe_update_viser_debug(
                        iteration=self._viser_pending_debug_frame["iteration"],
                        robot_base_pos_w=self._viser_pending_debug_frame["robot_base_pos_w"],
                        robot_base_quat_w=self._viser_pending_debug_frame["robot_base_quat_w"],
                        ground_truth_pcd_world=self._viser_pending_debug_frame["ground_truth_pcd_world"],
                        robot_obs_pcd_base=self._viser_pending_debug_frame["robot_obs_pcd_base"],
                        policy_input_pcd_base=self._viser_pending_debug_frame["policy_input_pcd_base"],
                        aux_prediction=aux_prediction_for_replay,
                    )
                    self._viser_pending_debug_frame = None

                teacher_actions = None
                total_loss = None
                action_loss = None
                aux_loss = None
                mode_loss = None
                force_loss = None

                if not self.play_policy:
                    teacher_output = self._get_teacher_actions(obs)
                    teacher_actions = teacher_output["actions"]
                    aux_target = None
                    if self.has_aux_prediction:
                        if self.latest_aux_target_vector is None:
                            raise RuntimeError("Expected the latest auxiliary target vector while aux prediction is enabled.")
                        aux_target = self._get_aux_target(self.latest_aux_target_vector)
                    total_loss, action_loss, aux_loss, mode_loss, force_loss = self._compute_student_loss(
                        student_output,
                        teacher_output["mus"],
                        aux_target=aux_target,
                    )
                    local_batch_size = int(student_actions.shape[0])
                    global_batch_size = self._get_global_batch_size(local_batch_size)
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model_ddp.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.student_update_steps += 1
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
                    self._seed_temporal_histories(done_mask)
                    self._seed_aux_buffer(done_mask)
                    self._seed_push_pull_condition_buffer(done_mask)
                    self._resample_teacher_forcing_env_mask(iteration + 1, done_mask)

                if total_loss is not None:
                    self._sync_timing_device()
                    self._record_timing(time.perf_counter() - iteration_start_time)
                    self._log(
                        iteration,
                        total_loss,
                        action_loss,
                        aux_loss,
                        mode_loss,
                        force_loss,
                        teacher_forcing_beta,
                    )
                else:
                    self._sync_timing_device()
                    self._record_timing(time.perf_counter() - iteration_start_time)

                if (
                    not self.play_policy
                    and self.rank == 0
                    and iteration % self.save_interval == 0
                ):
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
