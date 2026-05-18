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
from DoorOpening.utils.camera_utils import (
    build_pinhole_intrinsics,
    crop_local_pcd,
    depth_to_pointcloud,
    simulate_depth_cam_render_from_pose,
    simulate_lidar_render_from_pose,
)
from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaLeapSampler
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
        self.use_sim_body_pose_door_pcd = bool(self.runtime_cfg.get("use_sim_body_pose_door_pcd", False))
        self.timing_breakdown_interval = int(self.runtime_cfg.get("timing_breakdown_interval", 0))
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
        self.viser_show_policy_input = bool(self.viser_cfg.get("show_policy_input", True))
        self.viser_raw_enabled = self.rank == 0 and bool(self.viser_raw_cfg.get("enabled", False))
        self.viser_raw_save_interval = max(
            1,
            int(self.viser_raw_cfg.get("save_interval", self.viser_cfg.get("raw_interval", 1000))),
        )
        self.viser_raw_max_points = int(self.viser_raw_cfg.get("max_points", 12_000))
        self.viser_raw_max_frames = max(0, int(self.viser_raw_cfg.get("max_frames", 0)))
        self.viser_raw_include_ground_truth = bool(self.viser_raw_cfg.get("include_ground_truth", True))
        self.viser_raw_include_robot_obs = bool(self.viser_raw_cfg.get("include_robot_obs", True))
        self.viser_raw_include_policy_input = bool(
            self.viser_raw_cfg.get("include_policy_input", self.viser_show_policy_input)
        )
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
        self.temporal_derived_state_specs = OrderedDict()
        self.temporal_history_offsets_s = []
        self.temporal_history_interpolate = True
        self.max_temporal_history_s = 0.0
        self.temporal_dt_s = 0.0
        self.temporal_history_len = 0
        self.temporal_current_time_s = 0.0
        self.temporal_time_history = None
        self.temporal_q_history = None
        self.temporal_target_history = None
        self.temporal_base_vel_history = None
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
        self._timing_stats = {"sum_ms": 0.0, "count": 0}
        self._latest_timing_breakdown = None
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
        student_cfg_data.pop("dagger", None)
        self.local_pcd_range = list(student_cfg_data.pop("local_pcd_range", [1.0, 0.35, 0.35]))
        self.local_pcd_x_direction_cutoff = student_cfg_data.pop("x_direction_cutoff", -0.5)
        self.door_pcd_num_points = int(student_cfg_data.pop("door_pcd_num_points", 4096))
        self.temporal_obs_cfg = dict(student_cfg_data.pop("temporal_obs", {}) or {})
        self.temporal_obs_enabled = bool(self.temporal_obs_cfg.get("enabled", False))
        self.temporal_history_interpolate = bool(self.temporal_obs_cfg.get("interpolate", True))
        raw_temporal_offsets = self.temporal_obs_cfg.get("history_offsets_s", [])
        self.temporal_history_offsets_s = [float(offset) for offset in raw_temporal_offsets]
        if any(offset < 0.0 for offset in self.temporal_history_offsets_s):
            raise ValueError("temporal_obs.history_offsets_s values must be non-negative.")

        student_model_kwargs = {
            key: value
            for key, value in student_cfg_data.items()
            if not str(key).startswith("_")
        }
        self.student_model = PCDTransformer(**student_model_kwargs).to(self.device)
        if self.student_model.action_head.out_features != self.num_actions:
            raise ValueError(
                f"Student action_dim ({self.student_model.action_head.out_features}) "
                f"does not match env action dim ({self.num_actions})."
            )

        # print(self.student_model)
        # print("state_encoders_cfg", self.student_model.state_encoders_cfg)

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

        self.state_encoders_keys = tuple(
            key
            for key, cfg in self.student_model.state_encoders_cfg.items()
            if cfg.get("use_state", False)
        )
        self.pcd_encoders_keys = tuple(
            key
            for key, cfg in self.student_model.pcd_encoders_cfg.items()
            if cfg.get("use_pcd", False)
        )
        if "q_hand" not in self.state_encoders_keys:
            raise ValueError("PCDTransformer student config must include q_hand in state_encoders_cfg.")

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
        if self.temporal_derived_state_specs and not self.temporal_obs_enabled:
            raise ValueError("Temporal derived state keys require temporal_obs.enabled: true.")
        spec_offsets = [
            float(spec["offset_s"])
            for spec in self.temporal_derived_state_specs.values()
            if spec["offset_s"] is not None
        ]
        self.max_temporal_history_s = max([0.0, *self.temporal_history_offsets_s, *spec_offsets])
        if student_ckpt is not None and self.temporal_derived_state_specs and self.rank == 0:
            print(
                "Warning: student checkpoint was loaded while temporal derived inputs are active. "
                "New temporal state encoder weights may need fresh training."
            )

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
        self.aux_pregrasp_dropout_prob = float(self.runtime_cfg.get("aux_pregrasp_dropout_prob", 0.0))
        self.aux_buffer = None
        if self.has_aux_input:
            self.aux_buffer = torch.zeros((self.num_envs, self.aux_input_dim), dtype=torch.float32, device=self.device)
        if not 0.0 <= self.aux_pregrasp_dropout_prob <= 1.0:
            raise ValueError("aux_pregrasp_dropout_prob must be in [0, 1].")
        self.latest_aux_pregrasp_env_fraction = 0.0
        self.latest_aux_pregrasp_dropout_fraction = 0.0
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

    def _iteration_to_time_s(self, iteration):
        return float(iteration) * float(self.temporal_dt_s)

    def _get_current_time_s(self):
        return float(self.temporal_current_time_s)

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
        if history_len < 2 or not self.temporal_history_interpolate:
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

    def _push_temporal_history(self, timestamp, q, target, base_vel, env_ids=None):
        if self.temporal_time_history is None:
            return
        if q.ndim != 2 or target.ndim != 2 or base_vel.ndim != 2:
            raise RuntimeError(
                "Expected q, target, and base_vel to be rank-2, got "
                f"{tuple(q.shape)}, {tuple(target.shape)}, and {tuple(base_vel.shape)}."
            )

        if env_ids is None:
            if self.temporal_history_len > 1:
                self.temporal_time_history[:, 1:] = self.temporal_time_history[:, :-1].clone()
                self.temporal_q_history[:, 1:, :] = self.temporal_q_history[:, :-1, :].clone()
                self.temporal_target_history[:, 1:, :] = self.temporal_target_history[:, :-1, :].clone()
                self.temporal_base_vel_history[:, 1:, :] = self.temporal_base_vel_history[:, :-1, :].clone()
            self.temporal_time_history[:, 0] = float(timestamp)
            self.temporal_q_history[:, 0, :] = q
            self.temporal_target_history[:, 0, :] = target
            self.temporal_base_vel_history[:, 0, :] = base_vel
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
        self.temporal_time_history[env_ids, 0] = float(timestamp)
        self.temporal_q_history[env_ids, 0, :] = q[env_ids]
        self.temporal_target_history[env_ids, 0, :] = target[env_ids]
        self.temporal_base_vel_history[env_ids, 0, :] = base_vel[env_ids]

    def _seed_temporal_histories(self, env_ids=None):
        q = self._get_student_proprio_vector().detach()
        target = self._get_implemented_action_vector().detach()
        base_vel = self._get_student_base_velocity_vector().detach()
        timestamp = self._get_current_time_s()

        if env_ids is None:
            self.temporal_time_history[:] = timestamp
            self.temporal_q_history[:] = q.unsqueeze(1).expand(-1, self.temporal_history_len, -1)
            self.temporal_target_history[:] = target.unsqueeze(1).expand(-1, self.temporal_history_len, -1)
            self.temporal_base_vel_history[:] = base_vel.unsqueeze(1).expand(-1, self.temporal_history_len, -1)
            return

        if env_ids.numel() == 0:
            return
        self.temporal_time_history[env_ids] = timestamp
        self.temporal_q_history[env_ids] = q[env_ids].unsqueeze(1).expand(-1, self.temporal_history_len, -1)
        self.temporal_target_history[env_ids] = target[env_ids].unsqueeze(1).expand(-1, self.temporal_history_len, -1)
        self.temporal_base_vel_history[env_ids] = base_vel[env_ids].unsqueeze(1).expand(
            -1, self.temporal_history_len, -1
        )

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

    def _build_temporal_derived_state_values(self, q_pos, target_t):
        if not self.temporal_derived_state_specs:
            return {}

        q_t = q_pos[:, self.ov_env._robot_dof_idx]
        target_err = target_t - q_t
        required_q_offsets = sorted(
            {
                float(spec["offset_s"])
                for spec in self.temporal_derived_state_specs.values()
                if spec["kind"] in {"q", "delta_q", "target_err"} and spec["offset_s"] is not None
            }
        )
        required_target_offsets = sorted(
            {
                float(spec["offset_s"])
                for spec in self.temporal_derived_state_specs.values()
                if spec["kind"] in {"delta_target", "target_err"} and spec["offset_s"] is not None
            }
        )
        required_base_vel_offsets = sorted(
            {
                float(spec["offset_s"])
                for spec in self.temporal_derived_state_specs.values()
                if spec["kind"] == "base_vel" and spec["offset_s"] is not None
            }
        )

        delta_q_by_offset = {}
        q_history_by_offset = self._sample_temporal_history_offsets(self.temporal_q_history, required_q_offsets)
        for offset_s, q_history in q_history_by_offset.items():
            delta_q_by_offset[offset_s] = q_t - q_history[:, self.ov_env._robot_dof_idx]

        delta_target_by_offset = {}
        target_history_by_offset = self._sample_temporal_history_offsets(
            self.temporal_target_history,
            required_target_offsets,
        )
        for offset_s, target_history in target_history_by_offset.items():
            delta_target_by_offset[offset_s] = target_t - target_history

        base_vel_history_by_offset = self._sample_temporal_history_offsets(
            self.temporal_base_vel_history,
            required_base_vel_offsets,
        )

        values_by_key = {}
        for key, spec in self.temporal_derived_state_specs.items():
            kind = spec["kind"]
            offset_s = spec["offset_s"]
            if kind == "q":
                full_value = q_history_by_offset[offset_s]
            elif kind == "base_vel":
                full_value = base_vel_history_by_offset[offset_s]
            elif kind == "target_err":
                if offset_s is None:
                    full_value = target_err
                else:
                    full_value = (
                        target_history_by_offset[offset_s]
                        - q_history_by_offset[offset_s][:, self.ov_env._robot_dof_idx]
                    )
            elif kind == "delta_q":
                full_value = delta_q_by_offset[offset_s]
            elif kind == "delta_target":
                full_value = delta_target_by_offset[offset_s]
            else:
                raise KeyError(f"Unsupported temporal derived state kind '{kind}' for key '{key}'.")

            if spec["indices"] is None:
                values_by_key[key] = full_value
            else:
                values_by_key[key] = full_value[:, spec["indices"]]
        return values_by_key

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

    def _maybe_drop_aux_feedback(self, aux_input_vector):
        self.latest_aux_pregrasp_env_fraction = 0.0
        self.latest_aux_pregrasp_dropout_fraction = 0.0
        if aux_input_vector is None or not self.aux_feedback_to_policy or self.aux_pregrasp_dropout_prob <= 0.0:
            return aux_input_vector

        ref_motion_lib = getattr(self.ov_env, "ref_motion_lib", None)
        if ref_motion_lib is None:
            return aux_input_vector

        get_pregrasp_mask = getattr(ref_motion_lib, "get_before_first_keyframe_mask", None)
        if not callable(get_pregrasp_mask):
            return aux_input_vector

        pregrasp_mask = get_pregrasp_mask().to(device=self.device, dtype=torch.bool)
        self.latest_aux_pregrasp_env_fraction = float(pregrasp_mask.float().mean().item())
        if not torch.any(pregrasp_mask):
            return aux_input_vector

        drop_mask = pregrasp_mask & (
            torch.rand(aux_input_vector.shape[0], device=self.device) < self.aux_pregrasp_dropout_prob
        )
        self.latest_aux_pregrasp_dropout_fraction = float(drop_mask.float().mean().item())
        if not torch.any(drop_mask):
            return aux_input_vector

        dropped_aux_input = aux_input_vector.clone()
        # Zeroing the carried-over aux vector forces the student to recover the
        # handle estimate from the rest of the observation during pregrasp.
        dropped_aux_input[drop_mask] = 0.0
        return dropped_aux_input

    def _seed_aux_buffer(self, env_ids=None):
        if self.aux_buffer is None:
            return
        if env_ids is None:
            self.aux_buffer.zero_()
            return
        if env_ids.numel() == 0:
            return
        self.aux_buffer[env_ids] = 0.0

    def _get_cfg_range(self, cfg, key, default):
        values = cfg.get(key, default)
        if len(values) != 2:
            raise ValueError(f"Expected '{key}' to have exactly two values, got {values}.")
        low = float(values[0])
        high = float(values[1])
        if low > high:
            raise ValueError(f"Expected '{key}' to be ordered as [low, high], got {values}.")
        return low, high

    def _get_optional_cfg_range(self, cfg, key):
        values = cfg.get(key)
        if values is None:
            return None
        if len(values) != 2:
            raise ValueError(f"Expected '{key}' to have exactly two values, got {values}.")
        low = float(values[0])
        high = float(values[1])
        if low > high:
            raise ValueError(f"Expected '{key}' to be ordered as [low, high], got {values}.")
        return low, high

    def _init_pointcloud_assets(self):
        asset_index_by_dir = {
            Path(asset_path).resolve().parent: idx for idx, asset_path in enumerate(door_asset_paths)
        }
        motion_to_asset_idx = []
        for motion_path in motion_traj_paths:
            motion_dir = Path(motion_path).resolve().parent
            if motion_dir not in asset_index_by_dir:
                raise KeyError(f"Could not map motion file '{motion_path}' to a door asset path.")
            motion_to_asset_idx.append(asset_index_by_dir[motion_dir])
        self.motion_to_asset_idx = torch.tensor(motion_to_asset_idx, device=self.device, dtype=torch.long)

        env_motion_idx = self.ov_env.ref_motion_lib.env_to_file_map.to(device=self.device, dtype=torch.long)
        self.env_asset_idx = self.motion_to_asset_idx[env_motion_idx]
        env_asset_indices = getattr(self.ov_env, "env_asset_indices", None)
        if env_asset_indices is not None:
            env_asset_indices = env_asset_indices.to(device=self.device, dtype=torch.long)
            if not torch.equal(env_asset_indices, self.env_asset_idx):
                raise RuntimeError("Door env asset indices and reference-motion asset indices are inconsistent.")
        self.motion_family_ids = motion_family_ids.to(device=self.device, dtype=torch.long)
        self.env_family_ids = self.motion_family_ids[env_motion_idx]
        expected_asset_family_ids = door_asset_family_ids.to(device=self.device, dtype=torch.long)[self.env_asset_idx]
        if not torch.equal(expected_asset_family_ids, self.env_family_ids):
            raise RuntimeError("Door asset family ids and motion family ids are inconsistent.")
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
        self.door_sampler_env_ids = {
            idx: torch.nonzero(self.env_asset_idx == int(idx), as_tuple=False).squeeze(-1)
            for idx in unique_asset_idx
        }
        self.door_link_pointclouds = {
            idx: self._build_door_link_pointcloud_cache(sampler)
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
        self.robot_root_body_idx = int(self.ov_env._robot_base_link_idx[0])
        self.door_base_body_idx = int(self.ov_env._door_base_link_idx)
        self.door_link_body_indices = {
            link_name: int(self.ov_env._door_body_idx[self.ov_env.door_body_names.index(link_name)])
            for link_name in ("link_1", "link_2")
        }
        self.wall_distractors_enabled = bool(self.wall_distractor_cfg.get("enabled", True))
        self.wall_distractor_num_points = int(
            self.wall_distractor_cfg.get("num_points", max(256, self.door_pcd_num_points // 3))
        )
        self.wall_distractor_side_margin_scale_min, self.wall_distractor_side_margin_scale_max = self._get_cfg_range(
            self.wall_distractor_cfg,
            "side_margin_scale",
            [0.35, 0.75],
        )
        side_margin_abs_range = self._get_optional_cfg_range(self.wall_distractor_cfg, "side_margin_m")
        if side_margin_abs_range is None:
            self.wall_distractor_side_margin_abs_min_m = None
            self.wall_distractor_side_margin_abs_max_m = None
        else:
            self.wall_distractor_side_margin_abs_min_m, self.wall_distractor_side_margin_abs_max_m = side_margin_abs_range
        self.wall_distractor_bottom_margin_scale_min, self.wall_distractor_bottom_margin_scale_max = self._get_cfg_range(
            self.wall_distractor_cfg,
            "bottom_margin_scale",
            [0.02, 0.08],
        )
        self.wall_distractor_gap_min_m, self.wall_distractor_gap_max_m = self._get_cfg_range(
            self.wall_distractor_cfg,
            "edge_gap_m",
            [0.015, 0.04],
        )
        self.wall_distractor_depth_min_m, self.wall_distractor_depth_max_m = self._get_cfg_range(
            self.wall_distractor_cfg,
            "depth_m",
            [0.10, 0.26],
        )
        self.wall_distractor_center_offset_min_m, self.wall_distractor_center_offset_max_m = self._get_cfg_range(
            self.wall_distractor_cfg,
            "center_offset_m",
            [-0.20, 0.20],
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
        self.viser_env_step_fps = 1.0 / self.viser_env_step_dt
        self.viser_render_interval = max(int(getattr(sim_cfg, "render_interval", 1)), 1)
        self.viser_render_dt = self.viser_sim_dt * self.viser_render_interval
        pointcloud_sensor_dt = self.viser_env_step_dt
        if self.pointcloud_source == "depth":
            pointcloud_sensor_dt = getattr(
                self.ov_env.cfg,
                "pointcloud_camera_update_period",
                pointcloud_sensor_dt,
            )
        self.viser_pointcloud_sensor_dt = max(float(pointcloud_sensor_dt), 1e-6)
        self.viser_pointcloud_sensor_fps = 1.0 / self.viser_pointcloud_sensor_dt
        if not self.viser_raw_enabled:
            return

        raw_dir = os.path.dirname(self.viser_raw_path)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
        self._init_viser_raw_streams()

        if self.rank == 0:
            stream_desc = ", ".join(
                f"{name}:env{stream['env_id']}" for name, stream in self._viser_raw_streams.items()
            )
            print(f"Viser raw replay capture enabled for multi-door streams ({stream_desc}).")
            print(
                "Raw replay data will be written as "
                f"{self._format_iterated_record_path(self.viser_raw_path, '<family>_<chunk_tag>')}"
            )
            print(
                "Raw replay chunks will be written every "
                f"{self.viser_raw_save_interval} iterations."
            )
            if self.viser_raw_max_frames > 0:
                print(f"Each raw replay chunk will keep at most {self.viser_raw_max_frames} frames.")
            print(
                "Raw replay point budget: max_points={} (gt={}, obs={}, policy={}).".format(
                    self.viser_raw_max_points,
                    "off"
                    if not self.viser_raw_include_ground_truth or self.viser_raw_max_points <= 0
                    else "on",
                    "off"
                    if not self.viser_raw_include_robot_obs or self.viser_raw_max_points <= 0
                    else "on",
                    "off"
                    if not self.viser_raw_include_policy_input or self.viser_raw_max_points <= 0
                    else "on",
                )
            )
            print(
                "Replay timing: env_step_dt={:.6f}s ({:.2f} FPS), pointcloud_sensor_dt={:.6f}s ({:.2f} FPS).".format(
                    self.viser_env_step_dt,
                    self.viser_env_step_fps,
                    self.viser_pointcloud_sensor_dt,
                    self.viser_pointcloud_sensor_fps,
                )
            )

    def _get_viser_env_id(self):
        """Clamp the configured replay env index so debug capture never indexes outside the batch."""
        if self.num_envs <= 0:
            return 0
        return max(0, min(int(self.viser_env_id), self.num_envs - 1))

    def _init_viser_raw_streams(self):
        """Select one stable env per door family for multi-door replay export."""
        self._viser_raw_streams = OrderedDict()
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            matching_envs = self.family_env_ids.get(int(family_id))
            if matching_envs is None:
                matching_envs = torch.nonzero(self.env_family_ids == int(family_id), as_tuple=False).squeeze(-1)
            if matching_envs.numel() == 0:
                continue
            env_id = int(matching_envs[0].detach().cpu().item())
            self._viser_raw_streams[family_name] = {
                "family_id": int(family_id),
                "family_name": family_name,
                "env_id": env_id,
                "frames": [],
                "frame_count": 0,
                "chunk_index": 0,
                "chunk_start_iteration": None,
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
                "chunk_start_iteration": None,
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

    def _build_viser_raw_payload(self, latest_iteration, chunk_complete, stream):
        return {
            "format": "dooropening_viser_replay_v1",
            "pointcloud_frame": "world",
            "pointcloud_source": self.pointcloud_source,
            "pointcloud_render_mode": str(getattr(self.ov_env.cfg, "pointcloud_render_mode", "none")).lower(),
            "sim_dt": float(self.viser_sim_dt),
            "decimation": int(getattr(self.ov_env.cfg, "decimation", 1)),
            "render_interval": int(self.viser_render_interval),
            "render_dt": float(self.viser_render_dt),
            "env_step_dt": float(self.viser_env_step_dt),
            "env_step_fps": float(self.viser_env_step_fps),
            "frame_dt": float(self.viser_env_step_dt),
            "frame_fps": float(self.viser_env_step_fps),
            "pointcloud_sensor_dt": float(self.viser_pointcloud_sensor_dt),
            "pointcloud_sensor_fps": float(self.viser_pointcloud_sensor_fps),
            "raw_cloud_config": {
                "include_ground_truth": bool(self.viser_raw_include_ground_truth),
                "include_robot_obs": bool(self.viser_raw_include_robot_obs),
                "include_policy_input": bool(self.viser_raw_include_policy_input),
                "max_points": int(self.viser_raw_max_points),
            },
            "glorbot_urdf_path": str(glorbot_urdf_path),
            "robot_joint_names": list(getattr(self.ov_env.robot, "joint_names", [])),
            "door_joint_names": list(getattr(self.ov_env.door, "joint_names", [])),
            "capture_mode": "iteration_chunk",
            "door_family_id": stream["family_id"],
            "door_family_name": stream["family_name"],
            "chunk_index": int(stream["chunk_index"]),
            "chunk_env_id": int(stream["env_id"]),
            "chunk_start_iteration": None
            if stream["chunk_start_iteration"] is None
            else int(stream["chunk_start_iteration"]),
            "chunk_end_iteration": int(latest_iteration),
            "chunk_complete": bool(chunk_complete),
            "chunk_frame_count": int(stream["frame_count"]),
            "episode_index": int(stream["chunk_index"]),
            "episode_env_id": int(stream["env_id"]),
            "episode_start_iteration": None
            if stream["chunk_start_iteration"] is None
            else int(stream["chunk_start_iteration"]),
            "episode_end_iteration": int(latest_iteration),
            "episode_complete": bool(chunk_complete),
            "episode_frame_count": int(stream["frame_count"]),
            "frames": stream["frames"],
        }

    def _trim_viser_raw_frames(self, stream):
        if self.viser_raw_max_frames > 0 and len(stream["frames"]) > self.viser_raw_max_frames:
            stream["frames"] = stream["frames"][-self.viser_raw_max_frames :]
        stream["frame_count"] = len(stream["frames"])
        if stream["frame_count"] > 0:
            stream["chunk_start_iteration"] = int(stream["frames"][0]["iteration"])
        else:
            stream["chunk_start_iteration"] = None

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
        payload = self._build_viser_raw_payload(
            latest_iteration=latest_iteration,
            chunk_complete=chunk_complete,
            stream=stream,
        )
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
        stream["chunk_start_iteration"] = None
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
        q_pos,
        door_joint_pos,
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
            if stream["chunk_start_iteration"] is None:
                stream["chunk_index"] += 1
                stream["chunk_start_iteration"] = int(iteration)
            stream["latest_iteration"] = int(iteration)

            selected_ground_truth = None
            if isinstance(ground_truth_pcd_world, dict):
                selected_ground_truth = ground_truth_pcd_world.get(env_id)
            else:
                selected_ground_truth = ground_truth_pcd_world

            ground_truth_points_world = None
            if self.viser_raw_include_ground_truth and self.viser_raw_max_points > 0:
                ground_truth_points_world = self._prepare_viser_points(
                    selected_ground_truth,
                    env_id,
                    self.viser_raw_max_points,
                ).to(dtype=torch.float16)

            robot_obs_points_world = None
            if self.viser_raw_include_robot_obs and self.viser_raw_max_points > 0:
                robot_obs_points_world = self._prepare_viser_world_points_from_local(
                    robot_obs_pcd_base,
                    robot_base_pos_w,
                    robot_base_quat_w,
                    env_id,
                    self.viser_raw_max_points,
                    drop_zero_rows=True,
                ).to(dtype=torch.float16)

            policy_input_points_world = None
            if self.viser_raw_include_policy_input and policy_input_pcd_base is not None and self.viser_raw_max_points > 0:
                policy_input_points_world = self._prepare_viser_world_points_from_local(
                    policy_input_pcd_base,
                    robot_base_pos_w,
                    robot_base_quat_w,
                    env_id,
                    self.viser_raw_max_points,
                    drop_zero_rows=True,
                ).to(dtype=torch.float16)

            asset_idx = int(self.env_asset_idx[env_id].detach().cpu().item())
            frame_record = {
                "iteration": int(iteration),
                "sim_frame": int(self.frame),
                "env_id": int(env_id),
                "door_family_id": stream["family_id"],
                "door_family_name": stream["family_name"],
                "pointcloud_source": self.pointcloud_source,
                "ground_truth_points_world": ground_truth_points_world,
                "robot_obs_points_world": robot_obs_points_world,
                "policy_input_points_world": policy_input_points_world,
                "pointclouds": {
                    "ground_truth": ground_truth_points_world,
                    "robot_obs": robot_obs_points_world,
                    "policy_input": policy_input_points_world,
                },
                "aux_prediction": None
                if aux_prediction is None
                else self._aux_to_2d(aux_prediction)[env_id].detach().cpu().to(dtype=torch.float32),
                "robot_joint_pos": q_pos[env_id].detach().cpu().to(dtype=torch.float32),
                "door_joint_pos": door_joint_pos[env_id].detach().cpu().to(dtype=torch.float32),
                "robot_base_pos_w": self.ov_env.robot.data.body_pos_w[env_id, self.robot_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "robot_base_quat_w": self.ov_env.robot.data.body_quat_w[env_id, self.robot_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "door_base_pos_w": self.ov_env.door.data.body_pos_w[env_id, self.door_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "door_base_quat_w": self.ov_env.door.data.body_quat_w[env_id, self.door_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "door_asset_idx": asset_idx,
                "door_asset_path": str(door_asset_paths[asset_idx]),
            }
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

    def _override_actions_for_pregrasp(self, actions: torch.Tensor) -> torch.Tensor:
        override_fn = getattr(self.ov_env, "override_pregrasp_actions", None)
        if override_fn is None:
            return actions
        return override_fn(actions)

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
            teacher_actions = self._override_actions_for_pregrasp(teacher_actions)
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
        adjusted_actions = self._override_actions_for_pregrasp(torch.clamp(res_dict["mus"], -1.0, 1.0))
        student_adjusted_actions = self._env_actions_to_student_actions(adjusted_actions)
        return {
            "mus": student_adjusted_actions,
            "actions": adjusted_actions,
        }

    def _sync_timing_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _should_record_timing_breakdown(self, iteration):
        return self.timing_breakdown_interval > 0 and int(iteration) % self.timing_breakdown_interval == 0

    def _mark_timing_breakdown(self, timings, stage_start_time, stage_name):
        if timings is None:
            return stage_start_time
        self._sync_timing_device()
        now = time.perf_counter()
        timings[stage_name] = (now - stage_start_time) * 1000.0
        return now

    def _emit_timing_breakdown(self, iteration, timings):
        if not timings:
            return
        self._latest_timing_breakdown = OrderedDict(timings)
        if self.rank == 0:
            summary = ", ".join(f"{name}={elapsed_ms:.2f}" for name, elapsed_ms in timings.items())
            print(f"Timing Breakdown (ms) @ iteration {int(iteration)}: {summary}")

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
        robot_joint_pos = self.ov_env.robot.data.joint_pos[:, self.robot_sampler_joint_ids]
        robot_joint_pos = robot_joint_pos[:, self.robot_sampler_joint_reorder]
        robot_local_pcd = self.robot_sampler.sample(robot_joint_pos)
        # The URDF sampler already applies the mobile-base joints (base_x/base_y/base_rotation),
        # so these points live in the URDF root frame, not the tidybot chassis frame.
        robot_root_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_root_body_idx]
        robot_root_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_root_body_idx]
        quat = robot_root_quat_w.unsqueeze(1).expand(-1, robot_local_pcd.shape[1], -1)
        return quat_apply(quat, robot_local_pcd) + robot_root_pos_w.unsqueeze(1)

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

    def _build_door_link_pointcloud_cache(self, sampler):
        zero_joint = torch.zeros(
            (1, len(sampler.robot.actuated_joints)),
            dtype=torch.float32,
            device=self.device,
        )
        link_fk = sampler.robot.link_fk_batch(zero_joint, use_names=True)
        visual_fk = sampler.robot.visual_geometry_fk_batch(zero_joint)
        link_points = {}
        for link_name in ("link_1", "link_2"):
            link = next((candidate for candidate in sampler.links if candidate.name == link_name), None)
            if link is None:
                link_points[link_name] = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                continue

            link_to_base = link_fk[link_name]
            base_to_link = torch.linalg.inv(link_to_base)
            first_visual_geometry = link.visuals[0].geometry
            if first_visual_geometry not in visual_fk:
                link_points[link_name] = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                continue
            points = sampler.points[link_name]
            visual_to_base = visual_fk[first_visual_geometry]
            visual_to_link = torch.matmul(base_to_link, visual_to_base)
            hom_points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            link_points[link_name] = (
                torch.matmul(visual_to_link, hom_points.transpose(1, 2))[:, :3]
                .transpose(1, 2)
                .squeeze(0)
                .contiguous()
            )
        return link_points

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

            pcd_parts = []
            for link_name in ("link_1", "link_2"):
                link_points = link_points_by_name.get(link_name)
                if link_points is None or link_points.numel() == 0:
                    continue
                body_idx = self.door_link_body_indices[link_name]
                link_pos_w = self.ov_env.door.data.body_pos_w[env_ids, body_idx]
                link_quat_w = self.ov_env.door.data.body_quat_w[env_ids, body_idx]
                expanded_points = link_points.unsqueeze(0).expand(env_ids.numel(), -1, -1)
                expanded_quat = link_quat_w.unsqueeze(1).expand(-1, expanded_points.shape[1], -1)
                pcd_parts.append(quat_apply(expanded_quat, expanded_points) + link_pos_w.unsqueeze(1))

            if pcd_parts:
                asset_pcd_world = torch.cat(pcd_parts, dim=1)
                if asset_pcd_world.shape[1] != self.door_pcd_num_points:
                    sample_idx = torch.linspace(
                        0,
                        asset_pcd_world.shape[1] - 1,
                        steps=self.door_pcd_num_points,
                        device=self.device,
                        dtype=torch.float32,
                    ).round().to(dtype=torch.long)
                    asset_pcd_world = asset_pcd_world[:, sample_idx]
                door_pcd_world[env_ids] = asset_pcd_world
        return door_pcd_world

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

    def _sample_door_pointcloud_base_from_sim_body_pose(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        door_pcd_world = self._sample_cached_door_pointcloud_world()

        robot_pcd_world = self._sample_robot_pointcloud_world_sampler()
        scene_parts = [door_pcd_world, robot_pcd_world]
        wall_pcd_world = self._sample_wall_pointcloud_world()
        if wall_pcd_world.shape[1] > 0:
            scene_parts.append(wall_pcd_world)
        scene_pcd_world = torch.cat(scene_parts, dim=1)
        if self.viser_raw_enabled:
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(scene_pcd_world)
        return self._render_lidar_scene_pointcloud_base(scene_pcd_world, robot_base_pos_w, robot_base_quat_w)

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
        gt_scene_pcd_world = self._sample_scene_pointcloud_world_sampler()
        if self.viser_raw_enabled:
            self._viser_cached_ground_truth_pcd_world = self._select_viser_ground_truth_points(gt_scene_pcd_world)

        return self._render_lidar_scene_pointcloud_base(gt_scene_pcd_world, robot_base_pos_w, robot_base_quat_w)

    def _sample_door_pointcloud_base(self):
        self._viser_cached_ground_truth_pcd_world = None
        if self.pointcloud_source == "sampler":
            door_pcd_base = self._sample_door_pointcloud_base_sampler()
        elif self.pointcloud_source == "depth":
            door_pcd_base = self._sample_door_pointcloud_base_depth()
        elif self.use_sim_body_pose_door_pcd:
            door_pcd_base = self._sample_door_pointcloud_base_from_sim_body_pose()
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
        door_joint_pos = self.ov_env.door.data.joint_pos
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        palm_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_palm_body_idx].unsqueeze(1)

        self.latest_student_proprio_vector = q_pos.detach().clone()

        palm_pos_base = world_to_local(palm_pos_w, robot_base_pos_w, robot_base_quat_w).squeeze(1)
        door_pcd_base = self._sample_door_pointcloud_base()
        robot_pcd_base = self._sample_robot_pointcloud_base_sampler() if self.append_robot_gt_to_policy_cloud else None
        target_t = self._get_implemented_action_vector()
        temporal_state_values = self._build_temporal_derived_state_values(q_pos, target_t)
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
        aux_input_vector = self._maybe_drop_aux_feedback(aux_input_vector)

        obs = OrderedDict()
        for key in self.state_encoders_keys:
            if key == "q_base":
                raise KeyError("Raw q_base is disabled for the student policy; use base_vel instead.")
            elif key == "q_arm":
                obs[key] = q_pos[:, self.ov_env._robot_arm_dof_idx]
            elif key == "q_hand":
                obs[key] = q_pos[:, self.ov_env._robot_finger_dof_idx]
            elif key == "base_vel":
                obs[key] = base_vel
            elif key in self.temporal_derived_state_specs:
                obs[key] = temporal_state_values[key]
            elif key in self.aux_state_specs:
                if aux_input_vector is None:
                    raise RuntimeError(f"Aux state '{key}' is enabled but aux input vector is unavailable.")
                obs[key] = aux_input_vector[:, self.aux_state_specs[key]["slice"]]
            else:
                raise KeyError(f"Unsupported student state key '{key}' in config.")

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
                "q_pos": q_pos,
                "door_joint_pos": door_joint_pos,
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

    def _student_forward(self, student_obs):
        return self.student_model_ddp(student_obs)

    def _compute_student_loss(self, student_output, teacher_actions, aux_target=None):
        target = teacher_actions
        if aux_target is not None:
            target = torch.cat([teacher_actions, aux_target], dim=-1)
        loss = self.student_model.compute_loss(student_output, target.unsqueeze(1))
        total_loss = loss["total"]
        action_loss = loss.get("action", total_loss)
        aux_loss = loss.get("aux")
        return total_loss, action_loss, aux_loss

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

    def _log(self, iteration, total_loss, action_loss, aux_loss, teacher_forcing_beta):
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
            print("Teacher Forcing Beta:", teacher_forcing_beta)
            print("Teacher Rollout Env Fraction:", teacher_env_fraction)
            print("Student Rollout Env Fraction:", student_env_fraction)
            print("Student Update Steps:", self.student_update_steps)
            print("Last Local Update Batch Size:", self.last_local_update_batch_size)
            print("Last Global Update Batch Size:", self.last_global_update_batch_size)
            if self.aux_pregrasp_dropout_prob > 0.0:
                print("Aux Pregrasp Env Fraction:", self.latest_aux_pregrasp_env_fraction)
                print("Aux Pregrasp Dropout Fraction:", self.latest_aux_pregrasp_dropout_fraction)
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
        if episode_reward is not None:
            metrics["stats/episode_reward"] = episode_reward
        if episode_length is not None:
            metrics["stats/episode_length"] = episode_length
        if episode_length_seconds is not None:
            metrics["stats/episode_length_seconds"] = episode_length_seconds
        if success_rate is not None:
            metrics["stats/success_rate"] = success_rate
        for family_name, family_success_rate in family_success_rates.items():
            if family_success_rate is not None:
                metrics[f"stats/success_rate/{family_name}"] = family_success_rate
        if teacher_forcing_beta is not None:
            metrics["stats/teacher_forcing_beta"] = teacher_forcing_beta
        metrics["stats/teacher_rollout_env_fraction"] = teacher_env_fraction
        metrics["stats/student_rollout_env_fraction"] = student_env_fraction
        if self.aux_pregrasp_dropout_prob > 0.0:
            metrics["stats/aux_pregrasp_env_fraction"] = self.latest_aux_pregrasp_env_fraction
            metrics["stats/aux_pregrasp_dropout_fraction"] = self.latest_aux_pregrasp_dropout_fraction
        if iteration_time_ms is not None:
            metrics["timing/iteration_ms"] = iteration_time_ms
        if self._latest_timing_breakdown:
            for stage_name, elapsed_ms in self._latest_timing_breakdown.items():
                metrics[f"timing_breakdown/{stage_name}_ms"] = float(elapsed_ms)
            self._latest_timing_breakdown = None
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
            self._resample_teacher_forcing_env_mask(self.resume_iteration)

            for iteration in range(start_iteration, end_iteration):
                self._sync_timing_device()
                iteration_start_time = time.perf_counter()
                timing_breakdown = OrderedDict() if self._should_record_timing_breakdown(iteration) else None
                stage_start_time = iteration_start_time

                student_obs = self._build_student_obs(iteration=iteration)
                stage_start_time = self._mark_timing_breakdown(
                    timing_breakdown,
                    stage_start_time,
                    "build_student_obs",
                )
                student_output = self._student_forward(student_obs)
                student_actions = student_output["action"][:, 0, :]
                student_env_actions = self._student_actions_to_env_actions(student_actions)
                aux_prediction_for_replay = None
                if self.has_aux_prediction:
                    aux_prediction_for_replay = self._decode_aux_prediction(student_output["aux"].detach())
                    self.aux_buffer[:] = aux_prediction_for_replay
                stage_start_time = self._mark_timing_breakdown(
                    timing_breakdown,
                    stage_start_time,
                    "student_forward",
                )

                if self.viser_raw_enabled and self._viser_pending_debug_frame is not None:
                    self._maybe_update_viser_debug(
                        iteration=self._viser_pending_debug_frame["iteration"],
                        q_pos=self._viser_pending_debug_frame["q_pos"],
                        door_joint_pos=self._viser_pending_debug_frame["door_joint_pos"],
                        robot_base_pos_w=self._viser_pending_debug_frame["robot_base_pos_w"],
                        robot_base_quat_w=self._viser_pending_debug_frame["robot_base_quat_w"],
                        ground_truth_pcd_world=self._viser_pending_debug_frame["ground_truth_pcd_world"],
                        robot_obs_pcd_base=self._viser_pending_debug_frame["robot_obs_pcd_base"],
                        policy_input_pcd_base=self._viser_pending_debug_frame["policy_input_pcd_base"],
                        aux_prediction=aux_prediction_for_replay,
                    )
                    self._viser_pending_debug_frame = None
                stage_start_time = self._mark_timing_breakdown(
                    timing_breakdown,
                    stage_start_time,
                    "viser_raw",
                )

                teacher_actions = None
                total_loss = None
                action_loss = None
                aux_loss = None

                if not self.play_policy:
                    teacher_output = self._get_teacher_actions(obs)
                    teacher_actions = teacher_output["actions"]
                    stage_start_time = self._mark_timing_breakdown(
                        timing_breakdown,
                        stage_start_time,
                        "teacher_forward",
                    )
                    aux_target = None
                    if self.has_aux_prediction:
                        if self.latest_aux_target_vector is None:
                            raise RuntimeError("Expected the latest auxiliary target vector while aux prediction is enabled.")
                        aux_target = self._get_aux_target(self.latest_aux_target_vector)
                    total_loss, action_loss, aux_loss = self._compute_student_loss(
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
                    stage_start_time = self._mark_timing_breakdown(
                        timing_breakdown,
                        stage_start_time,
                        "student_update",
                    )

                step_actions, teacher_forcing_beta = self._mix_actions(
                    student_env_actions.detach(),
                    teacher_actions,
                    iteration,
                )
                obs, rew, out_of_reach, timed_out, step_extras = self.env.step(step_actions)
                stage_start_time = self._mark_timing_breakdown(
                    timing_breakdown,
                    stage_start_time,
                    "env_step",
                )
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
                    self._resample_teacher_forcing_env_mask(iteration + 1, done_mask)
                stage_start_time = self._mark_timing_breakdown(
                    timing_breakdown,
                    stage_start_time,
                    "post_step",
                )
                self._emit_timing_breakdown(iteration, timing_breakdown)

                if total_loss is not None:
                    self._sync_timing_device()
                    self._record_timing(time.perf_counter() - iteration_start_time)
                    self._log(iteration, total_loss, action_loss, aux_loss, teacher_forcing_beta)
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
