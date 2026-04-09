import os
import pathlib
import time
import math
from collections import OrderedDict, deque
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except ImportError:
    wandb = None

from isaaclab.utils.math import quat_apply, quat_mul, transform_points
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

from DoorOpening.assets.door.door_cfg import asset_paths as door_asset_paths
from DoorOpening.assets.door.door_cfg import motion_traj_paths
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
from DoorOpening.utils.pose_utils import world_to_local


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

        self.student_cfg = self.config.get("student", {})
        self.teacher_cfg = self.config.get("teacher", {})
        self.play_policy = bool(self.config.get("play_policy", False))
        self.runtime_cfg = self.config.get("dagger", {})

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
        self.teacher_forcing_schedule = str(self.runtime_cfg.get("teacher_forcing_schedule", "linear")).lower()
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
        # Runtime controls for live Viser inspection plus optional serializer/raw replay outputs.
        self.viser_cfg = dict(self.runtime_cfg.get("viser", {}))
        legacy_record_cfg = dict(self.viser_cfg.get("record", {}))
        self.viser_enabled = self.rank == 0 and bool(self.viser_cfg.get("enabled", False))
        self.viser_update_interval = max(1, int(self.viser_cfg.get("update_interval", 1)))
        self.viser_env_id = int(self.viser_cfg.get("env_id", self.runtime_cfg.get("debug_pointcloud_env_id", 0)))
        self.viser_show_policy_input = bool(self.viser_cfg.get("show_policy_input", True))
        self.viser_raw_interval = max(1, int(self.viser_cfg.get("raw_interval", 1000)))
        self.viser_point_size = float(self.viser_cfg.get("point_size", 0.004))
        self.viser_max_points = int(self.viser_cfg.get("max_points", 12_000))
        self.viser_serializer_cfg = dict(self.viser_cfg.get("serializer", legacy_record_cfg))
        self.viser_raw_cfg = dict(self.viser_cfg.get("raw", {}))
        self.viser_serializer_enabled = self.rank == 0 and bool(self.viser_serializer_cfg.get("enabled", False))
        self.viser_raw_enabled = self.rank == 0 and bool(
            self.viser_raw_cfg.get(
                "enabled",
                self.viser_serializer_cfg.get("save_raw", legacy_record_cfg.get("save_raw", False)),
            )
        )
        self.viser_replay_enabled = self.viser_serializer_enabled or self.viser_raw_enabled
        self.viser_replay_max_frames = int(
            self.viser_raw_cfg.get(
                "max_frames",
                self.viser_serializer_cfg.get("max_frames", legacy_record_cfg.get("max_frames", 500)),
            )
        )
        self.viser_replay_max_points = int(
            self.viser_raw_cfg.get(
                "max_points",
                self.viser_serializer_cfg.get("max_points", legacy_record_cfg.get("max_points", self.viser_max_points)),
            )
        )
        if self.pointcloud_source not in {"sampler", "depth", "lidar"}:
            raise ValueError(f"Unsupported pointcloud_source '{self.pointcloud_source}'.")
        if self.teacher_forcing_warmup_iters < 0:
            raise ValueError("teacher_forcing_warmup_iters must be non-negative.")
        if self.teacher_forcing_transition_iters < 0:
            raise ValueError("teacher_forcing_transition_iters must be non-negative.")
        if not 0.0 <= self.teacher_forcing_min_beta <= 1.0:
            raise ValueError("teacher_forcing_min_beta must be in [0, 1].")
        if self.teacher_forcing_schedule not in {"linear", "cosine"}:
            raise ValueError("teacher_forcing_schedule must be one of {'linear', 'cosine'}.")

        self.games_to_track = 100
        self.frame = 0
        self.epoch_num = 0

        self.nn_dir = nn_dir
        self.debug_pointcloud_dir = os.path.join(self.nn_dir if self.nn_dir is not None else os.getcwd(), "debug_pointclouds")
        self.wandb_cfg = self.runtime_cfg.get("wandb", self.config.get("wandb", {}))
        self.use_wandb = self.rank == 0 and bool(self.wandb_cfg.get("enabled", False))
        self.wandb_run = None
        if self.use_wandb:
            if wandb is None:
                raise ImportError("wandb logging is enabled, but the 'wandb' package is not installed.")
            self._init_wandb(summaries_dir)

        self.implemented_action_history = None
        self.student_proprio_history = None
        self.teacher_forcing_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.completed_rewards = deque(maxlen=self.games_to_track)
        self.completed_lengths = deque(maxlen=self.games_to_track)
        self.completed_successes = deque(maxlen=self.games_to_track)
        self.episode_reached_last_frame = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.student_update_steps = 0
        self.last_local_update_batch_size = 0
        self.last_global_update_batch_size = 0
        self.latest_student_proprio_vector = None
        self._timing_stats = {
            "iteration_ms": {"sum_ms": 0.0, "count": 0},
            "student_obs_ms": {"sum_ms": 0.0, "count": 0},
            "pointcloud_ms": {"sum_ms": 0.0, "count": 0},
            "env_step_ms": {"sum_ms": 0.0, "count": 0},
        }

        self._init_teacher()
        self._init_student()
        self._init_history_buffers()
        self._init_pointcloud_assets()
        self._init_viser_debug_tools()
        self.success_frame_idx = float(self.ov_env.ref_motion_lib.num_frames - 1)

    def _init_teacher(self):
        self.teacher_model = None
        if self.play_policy and self.teacher_cfg.get("ckpt") is None:
            return

        cfg_path = self.teacher_cfg.get("cfg")
        if not cfg_path:
            raise ValueError("Teacher config path is required unless play_policy=True with a student checkpoint.")

        self.teacher_network_params = self.load_yaml(cfg_path)["params"]
        self.teacher_network = self.load_networks(self.teacher_network_params)
        self.teacher_obs_type = self.teacher_cfg.get("obs_type", "policy")
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
        self.teacher_model = self.teacher_network.build(teacher_model_config).to(self.device)

        teacher_ckpt = self.teacher_cfg.get("ckpt")
        if teacher_ckpt is None and not self.play_policy:
            raise ValueError("Teacher checkpoint is required for distillation.")
        else:
            print(f"Loaded teacher checkpoint: {teacher_ckpt}")
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

        student_model_kwargs = {
            key: value
            for key, value in student_cfg_data.items()
            if not str(key).startswith("_")
        }
        self.student_model = PCDTransformer(**student_model_kwargs).to(self.device)
        if self.student_model.chunk_size != 1:
            raise ValueError("The current pointcloud DAgger loop only supports chunk_size=1.")
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

        self.action_history_state_specs = OrderedDict()
        self.max_implemented_action_history_lag = 1
        for key in self.state_encoders_keys:
            spec = self._parse_action_history_state_key(key)
            if spec is None:
                continue
            input_dim = int(self.student_model.state_encoders_cfg[key]["input_dim"])
            if input_dim != spec["dim"]:
                raise ValueError(
                    "{} must have input_dim={} to match the {} action slice.".format(
                        key,
                        spec["dim"],
                        spec["component"],
                    )
                )
            self.action_history_state_specs[key] = spec
            self.max_implemented_action_history_lag = max(self.max_implemented_action_history_lag, spec["lag"])
        if student_ckpt is not None and self.action_history_state_specs and self.rank == 0:
            print(
                "Warning: student checkpoint was loaded while prev_action_* inputs are active. "
                "If their meaning changed since the checkpoint was trained, reusing those encoder weights can spike loss."
            )

        self.proprio_history_state_specs = OrderedDict()
        self.max_student_proprio_history_lag = 1
        for key in self.state_encoders_keys:
            spec = self._parse_proprio_history_state_key(key)
            if spec is None:
                continue
            input_dim = int(self.student_model.state_encoders_cfg[key]["input_dim"])
            if input_dim != spec["dim"]:
                raise ValueError(
                    "{} must have input_dim={} to match the {} joint slice.".format(
                        key,
                        spec["dim"],
                        spec["component"],
                    )
                )
            self.proprio_history_state_specs[key] = spec
            self.max_student_proprio_history_lag = max(self.max_student_proprio_history_lag, spec["lag"])

        local_pcd_cfg = self.student_model.pcd_encoders_cfg.get("local_pcd_t")
        self.local_pcd_points = [0, 0, 0]
        print("local_pcd_cfg", local_pcd_cfg)
        # print(local_pcd_cfg.get("num_points", [self.door_pcd_num_points, 0, 0]))
        if local_pcd_cfg is not None:
            self.local_pcd_points = list(local_pcd_cfg.get("num_points", [self.door_pcd_num_points, 0, 0])[:3])
            # num_points = local_pcd_cfg.get("num_points", [self.door_pcd_num_points, 0, 0])
            # if isinstance(num_points, int):
            #     num_points = [num_points]
            # self.local_pcd_points = list(num_points[:3])
            # while len(self.local_pcd_points) < 3:
            #     self.local_pcd_points.append(0)

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

    def _parse_action_history_state_key(self, key):
        if not key.startswith("prev_action"):
            return None

        parts = key.split("_")
        if parts[:2] != ["prev", "action"]:
            raise KeyError(
                f"Unsupported action history key {key}. "
                "Expected prev_action[_base|_arm|_hand][_tK]."
            )

        component = "full"
        lag = 1
        for part in parts[2:]:
            if not part:
                continue
            if part.startswith("t") and part[1:].isdigit():
                parsed_lag = int(part[1:])
                if parsed_lag < 1:
                    raise ValueError(f"Action history lag in {key} must be >= 1.")
                lag = parsed_lag
                continue
            if part in self.action_component_aliases:
                component = self.action_component_aliases[part]
                continue
            raise KeyError(
                f"Unsupported action history key {key}. "
                "Expected prev_action[_base|_arm|_hand][_tK]."
            )

        action_indices = self.action_component_history_indices[component]
        action_dim = int(action_indices.numel())
        return {
            "component": component,
            "lag": lag,
            "indices": action_indices,
            "dim": action_dim,
        }

    def _parse_proprio_history_state_key(self, key):
        if not key.startswith("prev_q"):
            return None

        parts = key.split("_")
        if parts[:2] != ["prev", "q"]:
            raise KeyError(
                f"Unsupported proprio history key {key}. "
                "Expected prev_q[_base|_arm|_hand][_tK]."
            )

        component = "full"
        lag = 1
        for part in parts[2:]:
            if not part:
                continue
            if part.startswith("t") and part[1:].isdigit():
                parsed_lag = int(part[1:])
                if parsed_lag < 1:
                    raise ValueError(f"Proprio history lag in {key} must be >= 1.")
                lag = parsed_lag
                continue
            if part in self.action_component_aliases:
                component = self.action_component_aliases[part]
                continue
            raise KeyError(
                f"Unsupported proprio history key {key}. "
                "Expected prev_q[_base|_arm|_hand][_tK]."
            )

        proprio_indices = self.proprio_component_history_indices[component]
        proprio_dim = int(proprio_indices.numel())
        return {
            "component": component,
            "lag": lag,
            "indices": proprio_indices,
            "dim": proprio_dim,
        }

    def _init_history_buffers(self):
        self.implemented_action_history = torch.zeros(
            (self.num_envs, self.max_implemented_action_history_lag, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.student_proprio_history_dim = int(self.ov_env.robot.data.joint_pos.shape[-1])
        self.student_proprio_history = torch.zeros(
            (self.num_envs, self.max_student_proprio_history_lag, self.student_proprio_history_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def _get_history_tensor(self, history_buffer, lag, value_indices, reference_values=None):
        if history_buffer is None:
            raise RuntimeError("History buffer is not initialized.")
        if lag < 1 or lag > history_buffer.shape[1]:
            raise IndexError(
                f"Requested history lag t-{lag} but buffer only stores {history_buffer.shape[1]} steps."
            )
        values = history_buffer[:, lag - 1, value_indices]
        if reference_values is not None:
            if reference_values.ndim != 2:
                raise RuntimeError(
                    f"Expected reference_values to be rank-2, got shape {tuple(reference_values.shape)}."
                )
            values = values - reference_values[:, value_indices]
        return values

    def _push_history(self, history_buffer, values):
        if history_buffer is None:
            return
        if history_buffer.shape[1] > 1:
            history_buffer[:, 1:, :] = history_buffer[:, :-1, :].clone()
        history_buffer[:, 0, :] = values

    def _reset_history(self, history_buffer, env_ids):
        if history_buffer is None or env_ids.numel() == 0:
            return
        history_buffer[env_ids] = 0.0

    def _seed_student_histories(self, env_ids=None):
        prev_action_seed = self._get_implemented_action_vector().detach()
        proprio_seed = self._get_student_proprio_vector().detach()

        if env_ids is None:
            if self.implemented_action_history is not None:
                self.implemented_action_history[:] = prev_action_seed.unsqueeze(1).expand(
                    -1, self.implemented_action_history.shape[1], -1
                )
            if self.student_proprio_history is not None:
                self.student_proprio_history[:] = proprio_seed.unsqueeze(1).expand(
                    -1, self.student_proprio_history.shape[1], -1
                )
            return

        if env_ids.numel() == 0:
            return
        if self.implemented_action_history is not None:
            self.implemented_action_history[env_ids] = prev_action_seed[env_ids].unsqueeze(1).expand(
                -1, self.implemented_action_history.shape[1], -1
            )
        if self.student_proprio_history is not None:
            self.student_proprio_history[env_ids] = proprio_seed[env_ids].unsqueeze(1).expand(
                -1, self.student_proprio_history.shape[1], -1
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

    def _get_implemented_action_vector(self):
        # prev_action_* should reflect the actual joint-position targets sent to
        # the PD controller, not the normalized delta policy actions.
        pd_targets = getattr(self.ov_env, "applied_robot_dof_targets", None)
        if pd_targets is None:
            pd_targets = getattr(self.ov_env, "robot_dof_targets", None)
        if pd_targets is None:
            raise RuntimeError(
                "Expected the environment to expose applied_robot_dof_targets or robot_dof_targets "
                "for implemented action history features."
            )
        if pd_targets.ndim != 2:
            raise RuntimeError(f"Expected PD target tensor to be rank-2, got shape {tuple(pd_targets.shape)}.")
        return pd_targets

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
        unique_asset_idx = sorted(set(self.env_asset_idx.detach().cpu().tolist()))
        self.door_samplers = {
            idx: FrankaLeapSampler(door_asset_paths[idx], device=self.device, num_points=self.door_pcd_num_points)
            for idx in unique_asset_idx
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

        self.robot_base_body_idx = int(self.ov_env._robot_base_body_link_idx)
        self.robot_palm_body_idx = int(self.ov_env._robot_key_body_idx[self.ov_env._robot_palm_id_in_key_body_idx])
        self.robot_root_body_idx = int(self.ov_env._robot_base_link_idx[0])
        self.door_base_body_idx = int(self.ov_env._door_base_link_idx)

        self.robot_camera_body_idx = None
        self.camera_offset_pos = None
        self.camera_offset_quat_world = None
        self.sampler_camera_spec = None

        if self.pointcloud_source in {"sampler", "depth"}:
            self.robot_camera_body_idx = int(self.ov_env.robot.find_bodies("x5_camera_link")[0][0])
            camera_cfg = self.ov_env.cfg.pointcloud_camera_cfg
            self.camera_offset_pos = torch.tensor(camera_cfg.offset.pos, device=self.device, dtype=torch.float32)
            self.camera_offset_quat_world = torch.tensor(camera_cfg.offset.rot, device=self.device, dtype=torch.float32)
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
        """Create the Viser server, scene objects, and recording state used for debug playback."""
        self._viser_server = None
        self._viser_serializer = None
        self._viser_pointcloud_handles = {}
        self._viser_capture_requested = False
        self._viser_live_update_requested = False
        self._viser_replay_step_requested = False
        self._viser_capture_iteration = 0
        self._viser_capture_env_id = 0
        self._viser_cached_ground_truth_pcd_world = None
        self._viser_record_frames = []
        self._viser_record_frame_count = 0
        self._viser_record_latest_iteration = None
        self._viser_record_limit_reached = False
        self._viser_record_last_flush_frame_count = 0
        self._viser_record_last_raw_flush_frame_count = 0
        self._viser_record_episode_active = False
        self._viser_record_episode_env_id = 0
        self._viser_record_episode_index = 0
        self._viser_record_episode_start_iteration = None

        # Keep replay outputs next to checkpoints by default so a run's artifacts stay together.
        default_record_dir = Path(self.nn_dir).parent if self.nn_dir is not None else Path(os.getcwd())
        default_serializer_path = default_record_dir / "viser_replay.viser"
        configured_serializer_path = self.viser_serializer_cfg.get("path")
        if configured_serializer_path is None:
            self.viser_serializer_path = str(default_serializer_path)
        else:
            configured_serializer_path = Path(str(configured_serializer_path))
            if not configured_serializer_path.is_absolute():
                configured_serializer_path = default_record_dir / configured_serializer_path
            self.viser_serializer_path = str(configured_serializer_path)

        configured_raw_path = self.viser_raw_cfg.get("path", self.viser_raw_cfg.get("raw_path"))
        if configured_raw_path is None:
            self.viser_raw_path = str(Path(self.viser_serializer_path).with_suffix(".pt"))
        else:
            configured_raw_path = Path(str(configured_raw_path))
            if not configured_raw_path.is_absolute():
                configured_raw_path = default_record_dir / configured_raw_path
            self.viser_raw_path = str(configured_raw_path)

        sim_cfg = getattr(self.ov_env.cfg, "sim", None)
        sim_dt = getattr(sim_cfg, "dt", None)
        self.viser_serializer_frame_dt = float(
            self.viser_serializer_cfg.get("frame_dt", sim_dt if sim_dt is not None else 1.0 / 30.0)
        )

        if not (self.viser_enabled or self.viser_serializer_enabled or self.viser_raw_enabled):
            return

        if self.viser_serializer_enabled:
            serializer_dir = os.path.dirname(self.viser_serializer_path)
            if serializer_dir:
                os.makedirs(serializer_dir, exist_ok=True)
        if self.viser_raw_enabled:
            raw_dir = os.path.dirname(self.viser_raw_path)
            if raw_dir:
                os.makedirs(raw_dir, exist_ok=True)

        needs_viser_server = self.viser_enabled or self.viser_serializer_enabled
        if needs_viser_server:
            try:
                import viser
            except ImportError:
                if self.viser_enabled:
                    raise ImportError(
                        "dagger.viser.enabled=True requires the optional 'viser' package."
                    )
                if self.viser_serializer_enabled and not self.viser_raw_enabled:
                    raise ImportError(
                        "dagger.viser.serializer.enabled=True requires the optional 'viser' package."
                    )
                if self.rank == 0:
                    print("Viser is not installed; serializer/live are unavailable, saving raw replay data only.")
                return

        if needs_viser_server:
            # Start the browser-backed Viser server. It owns the GUI widgets and 3D scene below.
            self._viser_server = viser.ViserServer()
            self._viser_server.gui.configure_theme(control_width="medium")
            # Add a small origin frame so the user can tell which way the display axes point.
            self._viser_server.scene.add_frame(
                "/base_axes",
                show_axes=True,
                axes_length=0.15,
                axes_radius=0.01,
                visible=True,
            )
            # Add a reference ground plane to make motion and scale easier to read by eye.
            self._viser_server.scene.add_grid(
                "/grid",
                width=4,
                height=4,
                position=(0.0, 0.0, 0.0),
                shadow_opacity=0.1,
            )
            # Pre-create point-cloud nodes once, then update only their `.points` arrays every frame.
            self._viser_pointcloud_handles["ground_truth_points"] = self._viser_server.scene.add_point_cloud(
                "/ground_truth_points",
                points=torch.zeros((0, 3), dtype=torch.float32).cpu().numpy(),
                colors=(120, 120, 120),
                point_size=self.viser_point_size,
            )
            self._viser_pointcloud_handles["robot_obs_points"] = self._viser_server.scene.add_point_cloud(
                "/robot_obs_points",
                points=torch.zeros((0, 3), dtype=torch.float32).cpu().numpy(),
                colors=(79, 195, 247),
                point_size=self.viser_point_size,
            )
            self._viser_pointcloud_handles["policy_input_points"] = self._viser_server.scene.add_point_cloud(
                "/policy_input_points",
                points=torch.zeros((0, 3), dtype=torch.float32).cpu().numpy(),
                colors=(0, 170, 120),
                point_size=self.viser_point_size,
            )

            # Expose the currently visualized vectorized environment as a GUI control in the browser.
            env_id_handle = self._viser_server.gui.add_number(
                label="Env ID",
                initial_value=self.viser_env_id,
                min=0,
                max=self.num_envs - 1,
                step=1,
                hint="Select environment index for point-cloud playback.",
            )

            @env_id_handle.on_update
            def _(_event):
                # Mirror GUI edits back into trainer state so subsequent captures use the new env id.
                self.viser_env_id = int(env_id_handle.value)

            if self.viser_serializer_enabled:
                # This serializer captures the scene updates so they can be exported as a `.viser` replay.
                self._viser_serializer = self._viser_server.get_scene_serializer()

        if self.rank == 0:
            if self.viser_enabled:
                print(f"Viser live streaming enabled for env {self.viser_env_id}.")
                print(f"Viser live update interval: every {self.viser_update_interval} training iterations.")
            if self.viser_replay_enabled:
                print(f"Viser replay capture enabled for env {self.viser_env_id}.")
                print("Replay capture waits for that env to reset, then captures every step until the episode ends.")
            if self.viser_serializer_enabled:
                print(
                    "Serialized replays will be written as "
                    f"{self._format_iterated_record_path(self.viser_serializer_path, '<episode_tag>')}"
                )
            if self.viser_raw_enabled:
                print(
                    "Raw replay data will be written as "
                    f"{self._format_iterated_record_path(self.viser_raw_path, '<episode_tag>')}"
                )
                print(
                    "Raw replay snapshots will also be flushed every "
                    f"{self.viser_raw_interval} iterations during active replay capture."
                )

    def _get_viser_env_id(self):
        """Clamp the selected environment index so debug capture never indexes outside the batch."""
        if self.num_envs <= 0:
            return 0
        return max(0, min(int(self.viser_env_id), self.num_envs - 1))

    def _start_viser_record_episode(self, iteration, env_id):
        """Start a fresh replay buffer for the next complete episode of the selected env."""
        self._viser_record_episode_active = True
        self._viser_record_episode_env_id = int(env_id)
        self._viser_record_episode_index += 1
        self._viser_record_episode_start_iteration = int(iteration)
        self._viser_record_frames = []
        self._viser_record_frame_count = 0
        self._viser_record_latest_iteration = None
        self._viser_record_last_flush_frame_count = 0
        self._viser_record_last_raw_flush_frame_count = 0
        self._viser_record_limit_reached = False
        self._viser_cached_ground_truth_pcd_world = None
        if self.viser_serializer_enabled and self._viser_server is not None:
            # Reset the serializer so each `.viser` file contains exactly one episode.
            self._viser_serializer = self._viser_server.get_scene_serializer()
        if self.rank == 0:
            print(
                "Started Viser replay episode {} for env {} at iteration {}.".format(
                    self._viser_record_episode_index,
                    self._viser_record_episode_env_id,
                    self._viser_record_episode_start_iteration,
                )
            )

    def _finish_viser_record_episode(self, episode_complete, reason):
        """Flush the current episode replay to disk and clear the in-memory episode buffer."""
        if not self.viser_replay_enabled:
            return

        frame_count = self._viser_record_frame_count
        if frame_count > 0:
            self._flush_viser_recordings(episode_complete=episode_complete)
            if self.rank == 0:
                status = "complete" if episode_complete else "partial"
                print(
                    "Saved {} Viser episode {} for env {} with {} frames ({}).".format(
                        status,
                        self._viser_record_episode_index,
                        self._viser_record_episode_env_id,
                        frame_count,
                        reason,
                    )
                )

        self._viser_record_frames = []
        self._viser_record_frame_count = 0
        self._viser_record_latest_iteration = None
        self._viser_record_last_flush_frame_count = 0
        self._viser_record_last_raw_flush_frame_count = 0
        self._viser_record_limit_reached = False
        self._viser_record_episode_active = False
        self._viser_record_episode_start_iteration = None
        self._viser_cached_ground_truth_pcd_world = None

    def _build_viser_raw_payload(self, latest_iteration, episode_complete):
        return {
            "format": "dooropening_viser_replay_v1",
            "pointcloud_frame": "world",
            "pointcloud_source": self.pointcloud_source,
            "glorbot_urdf_path": str(glorbot_urdf_path),
            "robot_joint_names": list(getattr(self.ov_env.robot, "joint_names", [])),
            "door_joint_names": list(getattr(self.ov_env.door, "joint_names", [])),
            "episode_index": int(self._viser_record_episode_index),
            "episode_env_id": int(self._viser_record_episode_env_id),
            "episode_start_iteration": None
            if self._viser_record_episode_start_iteration is None
            else int(self._viser_record_episode_start_iteration),
            "episode_end_iteration": int(latest_iteration),
            "episode_complete": bool(episode_complete),
            "episode_frame_count": int(self._viser_record_frame_count),
            "frames": self._viser_record_frames,
        }

    def _maybe_flush_viser_raw_snapshot(self, iteration):
        if not self.viser_raw_enabled or self.rank != 0:
            return
        if self._viser_record_frame_count <= 0:
            return
        if self._viser_record_frame_count == self._viser_record_last_raw_flush_frame_count:
            return
        if self.viser_raw_interval <= 0:
            return
        if (int(iteration) + 1) % self.viser_raw_interval != 0:
            return
        latest_iteration = int(self._viser_record_latest_iteration)
        record_tag = f"episode_{self._viser_record_episode_index:04d}_iter_{latest_iteration}"
        payload = self._build_viser_raw_payload(latest_iteration=latest_iteration, episode_complete=False)
        torch.save(payload, self._format_iterated_record_path(self.viser_raw_path, record_tag))
        self._viser_record_last_raw_flush_frame_count = self._viser_record_frame_count

    def _should_capture_viser_replay_step(self, iteration):
        """Capture every step of one selected-env replay episode, starting only at an episode boundary."""
        if not self.viser_replay_enabled:
            return False

        env_id = self._get_viser_env_id()
        self._viser_capture_env_id = env_id

        if self._viser_record_episode_active and env_id != self._viser_record_episode_env_id:
            self._finish_viser_record_episode(
                episode_complete=False,
                reason=f"selected env changed from {self._viser_record_episode_env_id} to {env_id}",
            )

        if self._viser_record_episode_active:
            self._viser_capture_env_id = self._viser_record_episode_env_id
            return True

        current_length = int(self.current_lengths[env_id].detach().cpu().item())
        if current_length != 0:
            return False

        self._start_viser_record_episode(iteration, env_id)
        self._viser_capture_env_id = self._viser_record_episode_env_id
        return True

    def _should_capture_viser_frame(self, iteration):
        """Decide whether this step needs live Viser streaming, replay capture, or both."""
        self._viser_live_update_requested = (
            self.viser_enabled
            and self._viser_server is not None
            and (iteration % self.viser_update_interval == 0)
        )
        self._viser_replay_step_requested = self._should_capture_viser_replay_step(iteration)
        if not self._viser_replay_step_requested:
            self._viser_capture_env_id = self._get_viser_env_id()
        return self._viser_live_update_requested or self._viser_replay_step_requested

    def _prepare_viser_world_points_from_local(
        self,
        pointcloud_local,
        base_pos_w,
        base_quat_w,
        env_id,
        max_points,
        drop_zero_rows=False,
    ):
        """Convert a base-frame point cloud for one env into world coordinates and move it to CPU for Viser."""
        if pointcloud_local is None:
            return torch.zeros((0, 3), dtype=torch.float32)

        local_points = self._prepare_viser_points(
            pointcloud_local,
            env_id,
            max_points,
            drop_zero_rows=drop_zero_rows,
        )
        if local_points.numel() == 0:
            return local_points

        # Rotate from the robot base frame into world, then translate by the base origin.
        quat = base_quat_w[env_id].detach().to(device=local_points.device, dtype=local_points.dtype).unsqueeze(0)
        quat = quat.expand(local_points.shape[0], -1)
        pos = base_pos_w[env_id].detach().to(device=local_points.device, dtype=local_points.dtype).unsqueeze(0)
        world_points = quat_apply(quat, local_points) + pos
        return world_points.to(dtype=torch.float32).cpu()

    def _prepare_viser_points(self, pointcloud, env_id, max_points, drop_zero_rows=False):
        """Extract one env's point cloud, sanitize it, and downsample it to a Viser-friendly CPU tensor."""
        if pointcloud is None:
            return torch.zeros((0, 3), dtype=torch.float32)
        if pointcloud.ndim == 2 and pointcloud.shape[-1] == 3:
            points = pointcloud.detach()
        elif pointcloud.ndim == 3 and pointcloud.shape[-1] == 3:
            points = pointcloud[env_id].detach()
        else:
            raise ValueError(f"Expected pointcloud with shape (N, 3) or (B, N, 3), got {tuple(pointcloud.shape)}.")

        points = points.to(dtype=torch.float32, device="cpu")
        # Remove NaN/Inf rows before passing anything to the viewer.
        finite_mask = torch.isfinite(points).all(dim=-1)
        points = points[finite_mask]
        if drop_zero_rows and points.numel() > 0:
            # Policy inputs may be padded with zeros; hide those rows in the viewer.
            nonzero_mask = torch.any(points.abs() > 1e-6, dim=-1)
            points = points[nonzero_mask]
        if max_points is not None and max_points > 0 and points.shape[0] > max_points:
            # Use evenly-spaced indexing across the full cloud to avoid prefix bias (e.g., dropping later robot links).
            sample_idx = torch.linspace(0, points.shape[0] - 1, steps=max_points, dtype=torch.float32)
            sample_idx = torch.round(sample_idx).to(dtype=torch.long)
            points = points[sample_idx]
        return points

    def _format_iterated_record_path(self, path_str, iteration):
        """Append a record-specific suffix to replay filenames so each saved episode gets its own file."""
        path = Path(path_str)
        return str(path.with_name(f"{path.stem}_{iteration}{path.suffix}"))

    def _set_viser_pointcloud(self, handle_name, points_cpu):
        """Push a new numpy point array into an existing Viser scene node."""
        if self._viser_server is None:
            return
        handle = self._viser_pointcloud_handles[handle_name]
        handle.points = points_cpu.numpy()

    def _to_viser_display_frame(self, points_cpu, env_id):
        """Express world points in the robot-root frame so Viser shows motion relative to the robot body."""
        if points_cpu is None or points_cpu.numel() == 0:
            return points_cpu
        root_pos_w = self.ov_env.robot.data.body_pos_w[env_id, self.robot_root_body_idx].detach().to(device="cpu", dtype=torch.float32)
        root_quat_w = self.ov_env.robot.data.body_quat_w[env_id, self.robot_root_body_idx].detach().to(device="cpu", dtype=torch.float32)
        return world_to_local(
            points_cpu.unsqueeze(0),
            root_pos_w.unsqueeze(0),
            root_quat_w.unsqueeze(0),
        ).squeeze(0)

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
    ):
        """Stream the selected env live and/or append the current step to replay outputs."""
        if not self._viser_capture_requested:
            return

        record_active = self._viser_replay_step_requested and self._viser_record_episode_active
        env_id = (
            max(0, min(int(self._viser_record_episode_env_id), self.num_envs - 1))
            if record_active
            else max(0, min(int(self._viser_capture_env_id), self.num_envs - 1))
        )
        show_policy_input = self.viser_show_policy_input and policy_input_pcd_base is not None
        needs_scene_points = self._viser_live_update_requested or (record_active and self.viser_serializer_enabled)

        display_gt_points = None
        display_obs_points = None
        display_policy_points = None
        if needs_scene_points:
            # Ground-truth points already live in world space; the other clouds must be lifted from base space first.
            gt_points = self._prepare_viser_points(ground_truth_pcd_world, env_id, self.viser_max_points)
            obs_points = self._prepare_viser_world_points_from_local(
                robot_obs_pcd_base,
                robot_base_pos_w,
                robot_base_quat_w,
                env_id,
                self.viser_max_points,
            )
            policy_points = self._prepare_viser_world_points_from_local(
                policy_input_pcd_base,
                robot_base_pos_w,
                robot_base_quat_w,
                env_id,
                self.viser_max_points,
                drop_zero_rows=True,
            ) if show_policy_input else torch.zeros((0, 3), dtype=torch.float32)

            # Display in robot-root coordinates so the point clouds stay centered around the agent in the browser.
            display_gt_points = self._to_viser_display_frame(gt_points, env_id)
            display_obs_points = self._to_viser_display_frame(obs_points, env_id)
            display_policy_points = self._to_viser_display_frame(policy_points, env_id)

        if self._viser_live_update_requested and self._viser_server is not None:
            self._set_viser_pointcloud("ground_truth_points", display_gt_points)
            self._set_viser_pointcloud("robot_obs_points", display_obs_points)
            self._set_viser_pointcloud("policy_input_points", display_policy_points)

        if not record_active:
            return

        self._viser_record_frame_count += 1
        self._viser_record_latest_iteration = int(iteration)
        if (
            self.viser_replay_max_frames > 0
            and self._viser_record_frame_count > self.viser_replay_max_frames
            and not self._viser_record_limit_reached
        ):
            print(
                "dagger.viser replay max_frames={} was exceeded; continuing so the saved replay still contains "
                "the full episode.".format(self.viser_replay_max_frames)
            )
            self._viser_record_limit_reached = True

        if self.viser_raw_enabled:
            # Raw replay metadata stores world-frame points plus robot/door state for offline playback.
            record_world_points = lambda pts, drop_zero=False: self._prepare_viser_world_points_from_local(
                pts,
                robot_base_pos_w,
                robot_base_quat_w,
                env_id,
                self.viser_replay_max_points,
                drop_zero_rows=drop_zero,
            )
            frame_record = {
                "iteration": int(iteration),
                "sim_frame": int(self.frame),
                "env_id": int(env_id),
                "pointcloud_source": self.pointcloud_source,
                "ground_truth_points_world": self._prepare_viser_points(
                    ground_truth_pcd_world,
                    env_id,
                    self.viser_replay_max_points,
                ).to(dtype=torch.float16),
                "robot_obs_points_world": record_world_points(robot_obs_pcd_base).to(dtype=torch.float16),
                "policy_input_points_world": record_world_points(policy_input_pcd_base, drop_zero=True).to(dtype=torch.float16)
                if show_policy_input
                else None,
                "robot_joint_pos": q_pos[env_id].detach().cpu().to(dtype=torch.float32),
                "door_joint_pos": door_joint_pos[env_id].detach().cpu().to(dtype=torch.float32),
                "robot_base_pos_w": self.ov_env.robot.data.body_pos_w[env_id, self.robot_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "robot_base_quat_w": self.ov_env.robot.data.body_quat_w[env_id, self.robot_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "door_base_pos_w": self.ov_env.door.data.body_pos_w[env_id, self.door_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "door_base_quat_w": self.ov_env.door.data.body_quat_w[env_id, self.door_base_body_idx].detach().cpu().to(dtype=torch.float32),
                "door_asset_idx": int(self.env_asset_idx[env_id].detach().cpu().item()),
                "door_asset_path": str(door_asset_paths[int(self.env_asset_idx[env_id].detach().cpu().item())]),
            }
            self._viser_record_frames.append(frame_record)
            self._maybe_flush_viser_raw_snapshot(iteration)

        if self.viser_serializer_enabled and self._viser_serializer is not None:
            # Keep the serialized scene in sync with the replay timeline written at episode end.
            if not self._viser_live_update_requested:
                self._set_viser_pointcloud("ground_truth_points", display_gt_points)
                self._set_viser_pointcloud("robot_obs_points", display_obs_points)
                self._set_viser_pointcloud("policy_input_points", display_policy_points)
            self._viser_serializer.insert_sleep(self.viser_serializer_frame_dt)

    def _maybe_finish_viser_record_episode(self, done_mask):
        """Flush the replay once the selected env finishes its current episode."""
        if not self.viser_replay_enabled or not self._viser_record_episode_active or done_mask.numel() == 0:
            return

        selected_env = int(self._viser_record_episode_env_id)
        if not bool(torch.any(done_mask == selected_env).item()):
            return

        self._finish_viser_record_episode(
            episode_complete=True,
            reason=f"env {selected_env} terminated",
        )

    def _flush_viser_recordings(self, episode_complete):
        """Write the buffered selected-env episode replay to disk."""
        if not self.viser_replay_enabled or self.rank != 0:
            return

        if self._viser_record_frame_count <= 0:
            return

        if self._viser_record_frame_count == self._viser_record_last_flush_frame_count:
            return

        latest_iteration = int(self._viser_record_latest_iteration)
        record_tag = f"episode_{self._viser_record_episode_index:04d}_iter_{latest_iteration}"

        if self.viser_serializer_enabled and self._viser_serializer is not None:
            # `.viser` files contain the scene update stream and can be replayed directly in Viser tooling.
            Path(self._format_iterated_record_path(self.viser_serializer_path, record_tag)).write_bytes(
                self._viser_serializer.serialize()
            )
            if self._viser_server is not None:
                self._viser_serializer = self._viser_server.get_scene_serializer()

        if self.viser_raw_enabled:
            # The raw torch payload preserves higher-level metadata for custom offline analysis/reconstruction.
            payload = self._build_viser_raw_payload(
                latest_iteration=latest_iteration,
                episode_complete=episode_complete,
            )
            torch.save(payload, self._format_iterated_record_path(self.viser_raw_path, record_tag))
            self._viser_record_last_raw_flush_frame_count = self._viser_record_frame_count

        self._viser_record_last_flush_frame_count = self._viser_record_frame_count

    def _close_viser_debug_tools(self):
        """Flush any in-progress replay and stop the background Viser server on shutdown."""
        if self._viser_record_episode_active or self._viser_record_frame_count > 0:
            self._finish_viser_record_episode(episode_complete=False, reason="shutdown")
        if self._viser_server is not None:
            self._viser_server.stop()

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

    def set_teacher_weights(self, ckpt, strict=True, allow_adjust=True):
        weights = self._load_checkpoint_state(ckpt)
        state_dict, meta = self._extract_model_state(weights)
        if allow_adjust:
            state_dict = adjust_state_dict_keys(state_dict, self.teacher_model.state_dict())
        self.teacher_model.load_state_dict(state_dict, strict=strict)
        if meta is not None and "running_mean_std" in meta:
            self.teacher_model.running_mean_std.load_state_dict(meta["running_mean_std"])
        print(f"Loaded teacher checkpoint: {ckpt}")

    def load_student_weights(self, ckpt):
        weights = torch.load(ckpt, map_location="cpu")
        state_dict, _ = self._extract_model_state(weights)
        state_dict = strip_prefix_from_state_dict(state_dict)
        self.student_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded student checkpoint: {ckpt}")

    def _override_actions_for_pregrasp(self, actions: torch.Tensor) -> torch.Tensor:
        override_fn = getattr(self.ov_env, "override_pregrasp_actions", None)
        if override_fn is None:
            return actions
        return override_fn(actions)

    def _get_teacher_actions(self, obs):
        if self.teacher_model is None:
            raise RuntimeError("Teacher model is not initialized.")
        batch_dict = {
            "is_train": False,
            "obs": obs[self.teacher_obs_type],
            "prev_actions": self.implemented_action_history[:, 0, :],
        }
        with torch.no_grad():
            res_dict = self.teacher_model(batch_dict)
        mus = res_dict["mus"]
        adjusted_actions = self._override_actions_for_pregrasp(torch.clamp(mus, -1.0, 1.0))
        return {
            "mus": adjusted_actions,
            "actions": adjusted_actions,
        }

    def _sync_timing_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _record_timing(self, name, elapsed_s):
        stats = self._timing_stats[name]
        stats["sum_ms"] += elapsed_s * 1000.0
        stats["count"] += 1

    def _consume_timing_means(self):
        means = {}
        for name, stats in self._timing_stats.items():
            means[name] = None if stats["count"] == 0 else stats["sum_ms"] / stats["count"]
            stats["sum_ms"] = 0.0
            stats["count"] = 0
        return means

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
        offset_pos = self.camera_offset_pos.unsqueeze(0).expand(self.num_envs, -1)
        offset_quat = self.camera_offset_quat_world.unsqueeze(0).expand(self.num_envs, -1)
        camera_pos_w = quat_apply(camera_link_quat_w, offset_pos) + camera_link_pos_w
        camera_quat_w = quat_mul(camera_link_quat_w, offset_quat)
        quat_xyzw = camera_quat_w[:, [1, 2, 3, 0]]
        # return torch.cat([camera_pos_w, quat_xyzw], dim=-1)
        return torch.cat([camera_link_pos_w, camera_link_quat_w[:, [1, 2, 3, 0]]], dim=-1)

    def _get_lidar_pose(self):
        lidar_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_lidar_body_idx]
        lidar_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_lidar_body_idx]
        return torch.cat([lidar_pos_w, lidar_quat_w[:, [1, 2, 3, 0]]], dim=-1)

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
            env_ids = torch.nonzero(self.env_asset_idx == asset_idx, as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            local_pcd = sampler.sample(door_joint_pos[env_ids])
            quat = door_base_quat_w[env_ids].unsqueeze(1).expand(-1, local_pcd.shape[1], -1)
            world_pcd = quat_apply(quat, local_pcd) + door_base_pos_w[env_ids].unsqueeze(1)
            door_pcd_world[env_ids] = world_pcd
        robot_pcd_world = self._sample_robot_pointcloud_world_sampler()
        return torch.cat([door_pcd_world, robot_pcd_world], dim=1)

    def _sample_door_pointcloud_base_sampler(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        gt_scene_pcd_world = self._sample_scene_pointcloud_world_sampler()
        if self._viser_capture_requested:
            # Keep only the selected env on CPU so Viser debug does not retain an extra
            # full batched scene pointcloud on GPU during training/debug runs.
            self._viser_cached_ground_truth_pcd_world = gt_scene_pcd_world[self._viser_capture_env_id].detach().cpu()
        # self._debug_visualize_pointcloud(gt_scene_pcd_world, "gt_scene_pointcloud_before_render")
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
        # self._debug_visualize_pointcloud(rendered_pcd_world, "sampler_scene_pointcloud")
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
        door_pcd_base = world_to_local(door_pcd_world, robot_base_pos_w, robot_base_quat_w)
        # Filter floor points while preserving the batched layout expected by the cropper.
        floor_mask = door_pcd_base[..., 2] > 0.1
        door_pcd_base = door_pcd_base.clone()
        door_pcd_base[~floor_mask] = float("nan")
        # self._debug_visualize_pointcloud(door_pcd_base, "depth_door_pointcloud")
        return door_pcd_base

    def _sample_door_pointcloud_base_lidar(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        gt_scene_pcd_world = self._sample_scene_pointcloud_world_sampler()
        if self._viser_capture_requested:
            self._viser_cached_ground_truth_pcd_world = gt_scene_pcd_world[self._viser_capture_env_id].detach().cpu()

        rendered_pcd_world, _ = simulate_lidar_render_from_pose(
            pcd=gt_scene_pcd_world,
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

    def _sample_door_pointcloud_base(self):
        self._sync_timing_device()
        start_time = time.perf_counter()
        self._viser_cached_ground_truth_pcd_world = None
        if self.pointcloud_source == "sampler":
            door_pcd_base = self._sample_door_pointcloud_base_sampler()
        elif self.pointcloud_source == "depth":
            door_pcd_base = self._sample_door_pointcloud_base_depth()
        else:
            door_pcd_base = self._sample_door_pointcloud_base_lidar()
        self._sync_timing_device()
        self._record_timing("pointcloud_ms", time.perf_counter() - start_time)
        return door_pcd_base

    def _debug_visualize_pointcloud(self, pointcloud, tag):
        env_id = 0
        if env_id is None:
            return
        if pointcloud.ndim != 3:
            raise ValueError(f"Expected batched pointcloud with shape (B, N, 3), got {tuple(pointcloud.shape)}.")
        env_id = int(env_id)
        if env_id < 0 or env_id >= pointcloud.shape[0]:
            raise IndexError(f"debug_pointcloud_env_id={env_id} is out of range for batch size {pointcloud.shape[0]}.")

        import numpy as np
        import open3d as o3d

        np_points = pointcloud[env_id].detach().cpu().numpy().astype(np.float64)
        finite_mask = np.isfinite(np_points).all(axis=-1)
        np_points = np_points[finite_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(np_points))

        os.makedirs(self.debug_pointcloud_dir, exist_ok=True)
        filename = os.path.join(self.debug_pointcloud_dir, f"{tag}_env{env_id}.ply")
        print(f"Saving pointcloud to {filename}")
        o3d.io.write_point_cloud(filename, pcd)
        try:
            o3d.visualization.draw_geometries([pcd], window_name=f"{tag} env {env_id}")
        except Exception as exc:
            print(f"Skipping pointcloud visualization for {filename}: {exc}")

    def _build_local_pcd(self, door_pcd_base, palm_pos_base):
        pcd_parts = []

        if self.local_pcd_points[0] > 0:
            base_crop, _ = crop_local_pcd(
                door_pcd_base,
                local_range=self.local_pcd_range[0],
                num_local_points=self.local_pcd_points[0],
                is_cylindrical=True,
                crop_center=torch.zeros((self.num_envs, 3), device=self.device, dtype=door_pcd_base.dtype),
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

    def _build_student_obs(self):
        q_pos = self._get_student_proprio_vector()
        door_joint_pos = self.ov_env.door.data.joint_pos
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        palm_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_palm_body_idx].unsqueeze(1)

        self.latest_student_proprio_vector = q_pos.detach().clone()

        palm_pos_base = world_to_local(palm_pos_w, robot_base_pos_w, robot_base_quat_w).squeeze(1)
        door_pcd_base = self._sample_door_pointcloud_base()
        current_implemented_action = (
            self.implemented_action_history[:, 0, :] if self.implemented_action_history is not None else None
        )

        obs = OrderedDict()
        for key in self.state_encoders_keys:
            if key == "q_base":
                obs[key] = q_pos[:, self.ov_env._robot_base_dof_idx]
            elif key == "q_arm":
                obs[key] = q_pos[:, self.ov_env._robot_arm_dof_idx]
            elif key == "q_hand":
                obs[key] = q_pos[:, self.ov_env._robot_finger_dof_idx]
            elif key in self.action_history_state_specs:
                spec = self.action_history_state_specs[key]
                obs[key] = self._get_history_tensor(
                    self.implemented_action_history,
                    lag=spec["lag"],
                    value_indices=spec["indices"],
                    reference_values=current_implemented_action,
                )
            elif key in self.proprio_history_state_specs:
                spec = self.proprio_history_state_specs[key]
                obs[key] = self._get_history_tensor(
                    self.student_proprio_history,
                    lag=spec["lag"],
                    value_indices=spec["indices"],
                    reference_values=q_pos,
                )
            else:
                raise KeyError(f"Unsupported student state key '{key}' in config.")

        for key in self.pcd_encoders_keys:
            if key == "local_pcd_t":
                # self._debug_visualize_pointcloud(door_pcd_base, "door_pcd_base")
                obs[key] = self._build_local_pcd(
                    door_pcd_base,
                    palm_pos_base,
                )
                # self._debug_visualize_pointcloud(obs[key], "local_pcd_t")
            else:
                raise KeyError(f"Unsupported student pointcloud key '{key}' in config.")

        if self._viser_capture_requested:
            self._maybe_update_viser_debug(
                iteration=self._viser_capture_iteration,
                q_pos=q_pos,
                door_joint_pos=door_joint_pos,
                robot_base_pos_w=robot_base_pos_w,
                robot_base_quat_w=robot_base_quat_w,
                ground_truth_pcd_world=self._viser_cached_ground_truth_pcd_world,
                robot_obs_pcd_base=door_pcd_base,
                policy_input_pcd_base=obs.get("local_pcd_t"),
            )
            self._viser_cached_ground_truth_pcd_world = None
            self._viser_capture_requested = False

        return obs

    def _student_forward(self, student_obs):
        return self.student_model_ddp(student_obs)

    def _compute_student_loss(self, student_output, teacher_actions):
        action_loss = F.mse_loss(student_output["action"][:, 0, :], teacher_actions)
        total_loss = action_loss
        return total_loss, action_loss

    def _get_teacher_forcing_beta(self, iteration):
        if self.play_policy or self.teacher_model is None:
            return 0.0
        if iteration < self.teacher_forcing_warmup_iters:
            return 1.0
        if self.teacher_forcing_transition_iters <= 0:
            return self.teacher_forcing_min_beta

        transition_iteration = iteration - self.teacher_forcing_warmup_iters
        if transition_iteration >= self.teacher_forcing_transition_iters:
            return self.teacher_forcing_min_beta

        progress = transition_iteration / float(self.teacher_forcing_transition_iters)
        if self.teacher_forcing_schedule == "linear":
            schedule_value = 1.0 - progress
        else:
            schedule_value = 0.5 * (1.0 + math.cos(math.pi * progress))

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
        if self.play_policy or self.teacher_model is None:
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
        if self.play_policy or self.teacher_model is None:
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

    def _update_last_frame_tracker(self):
        ref_motion_lib = getattr(self.ov_env, "ref_motion_lib", None)
        if ref_motion_lib is None:
            return
        next_frame_idx = torch.clamp(
            ref_motion_lib.frame_idx.to(device=self.device, dtype=torch.float32) + float(ref_motion_lib.velocity),
            max=self.success_frame_idx,
        )
        self.episode_reached_last_frame |= next_frame_idx >= self.success_frame_idx

    def _mean_completed_metric(self, values):
        if not values:
            return None
        return float(sum(values) / len(values))

    def _update_completed_episode_metrics(self, done_mask, timed_out):
        if done_mask.numel() == 0:
            return

        episode_rewards = self.current_rewards[done_mask, 0].detach().cpu().tolist()
        episode_lengths = self.current_lengths[done_mask].detach().cpu().tolist()
        episode_successes = (
            self.episode_reached_last_frame[done_mask] | timed_out[done_mask]
        ).to(dtype=torch.float32).detach().cpu().tolist()

        self.completed_rewards.extend(float(value) for value in episode_rewards)
        self.completed_lengths.extend(float(value) for value in episode_lengths)
        self.completed_successes.extend(float(value) for value in episode_successes)

    def _log(self, iteration, total_loss, action_loss, teacher_forcing_beta):
        if iteration % self.log_interval != 0:
            return
        episode_reward = self._mean_completed_metric(self.completed_rewards)
        episode_length = self._mean_completed_metric(self.completed_lengths)
        success_rate = self._mean_completed_metric(self.completed_successes)
        teacher_env_fraction = self._get_teacher_forcing_env_fraction()
        student_env_fraction = 1.0 - teacher_env_fraction
        timing_means = self._consume_timing_means()

        if self.rank == 0:
            print("=" * 10)
            print("ITERATION:", iteration)
            print("Total Loss:", float(total_loss.detach().cpu()))
            print("Action Loss:", float(action_loss.detach().cpu()))
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
            if success_rate is not None:
                print("Success Rate:", success_rate)
            if timing_means["iteration_ms"] is not None:
                print("Iteration Time (ms):", timing_means["iteration_ms"])

        metrics = {
            "loss/total": float(total_loss.detach().cpu()),
            "loss/action": float(action_loss.detach().cpu()),
            "dist/world_size": self.world_size,
            "dist/update_steps": self.student_update_steps,
            "dist/last_local_update_batch_size": self.last_local_update_batch_size,
            "dist/last_global_update_batch_size": self.last_global_update_batch_size,
        }
        if episode_reward is not None:
            metrics["stats/episode_reward"] = episode_reward
        if episode_length is not None:
            metrics["stats/episode_length"] = episode_length
        if success_rate is not None:
            metrics["stats/success_rate"] = success_rate
        if teacher_forcing_beta is not None:
            metrics["stats/teacher_forcing_beta"] = teacher_forcing_beta
        metrics["stats/teacher_rollout_env_fraction"] = teacher_env_fraction
        metrics["stats/student_rollout_env_fraction"] = student_env_fraction
        if timing_means["iteration_ms"] is not None:
            metrics["timing/iteration_ms"] = timing_means["iteration_ms"]
        if timing_means["student_obs_ms"] is not None:
            metrics["timing/student_obs_ms"] = timing_means["student_obs_ms"]
        if timing_means["pointcloud_ms"] is not None:
            metrics["timing/pointcloud_ms"] = timing_means["pointcloud_ms"]
        if timing_means["env_step_ms"] is not None:
            metrics["timing/env_step_ms"] = timing_means["env_step_ms"]
        self._wandb_log(metrics, step=iteration)

    def distill(self):
        if not self.play_policy and self.teacher_model is None:
            raise RuntimeError("Teacher model must be initialized for distillation.")

        self.student_model_ddp.train(not self.play_policy)
        if self.teacher_model is not None:
            self.teacher_model.eval()

        try:
            obs = self.env.reset()[0]
            self.latest_student_proprio_vector = None
            self._seed_student_histories()
            self.episode_reached_last_frame.zero_()
            self._resample_teacher_forcing_env_mask(0)

            for iteration in range(self.num_iters):
                self._sync_timing_device()
                iteration_start_time = time.perf_counter()

                self._sync_timing_device()
                student_obs_start_time = time.perf_counter()
                self._viser_capture_iteration = iteration
                self._viser_capture_requested = self._should_capture_viser_frame(iteration)
                student_obs = self._build_student_obs()
                self._sync_timing_device()
                self._record_timing("student_obs_ms", time.perf_counter() - student_obs_start_time)
                student_output = self._student_forward(student_obs)
                student_actions = torch.clamp(student_output["action"][:, 0, :], -1.0, 1.0)
                # Capture q_t before stepping so prev_q features align with the next state.
                if self.latest_student_proprio_vector is None:
                    raise RuntimeError("Expected the latest student proprio vector to be captured while building obs.")
                current_q_pos = self.latest_student_proprio_vector.detach().clone()

                teacher_actions = None
                total_loss = None
                action_loss = None

                if not self.play_policy:
                    teacher_output = self._get_teacher_actions(obs)
                    teacher_actions = teacher_output["actions"]
                    total_loss, action_loss = self._compute_student_loss(student_output, teacher_output["mus"])
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
                    student_actions.detach(),
                    teacher_actions,
                    iteration,
                )
                self._update_last_frame_tracker()
                self._sync_timing_device()
                env_step_start_time = time.perf_counter()
                obs, rew, out_of_reach, timed_out, _ = self.env.step(step_actions)
                self._sync_timing_device()
                self._record_timing("env_step_ms", time.perf_counter() - env_step_start_time)
                current_pd_targets = self._get_implemented_action_vector().detach().clone()

                self._push_history(self.implemented_action_history, current_pd_targets)
                self._push_history(self.student_proprio_history, current_q_pos)
                self.frame += self.num_envs

                self.current_rewards += rew.unsqueeze(-1)
                self.current_lengths += 1
                done_mask = torch.nonzero(out_of_reach | timed_out, as_tuple=False).squeeze(-1)
                self._maybe_finish_viser_record_episode(done_mask)
                if done_mask.numel() > 0:
                    self._update_completed_episode_metrics(done_mask, timed_out)
                    self.current_rewards[done_mask] = 0.0
                    self.current_lengths[done_mask] = 0.0
                    self._seed_student_histories(done_mask)
                    self.episode_reached_last_frame[done_mask] = False
                    self._resample_teacher_forcing_env_mask(iteration + 1, done_mask)

                if total_loss is not None:
                    self._sync_timing_device()
                    self._record_timing("iteration_ms", time.perf_counter() - iteration_start_time)
                    self._log(iteration, total_loss, action_loss, teacher_forcing_beta)
                else:
                    self._sync_timing_device()
                    self._record_timing("iteration_ms", time.perf_counter() - iteration_start_time)

                if (
                    not self.play_policy
                    and self.rank == 0
                    and iteration > 0
                    and iteration % self.save_interval == 0
                ):
                    ckpt_path = os.path.join(self.nn_dir, f"pcd_student_{iteration}.pt")
                    self.save(ckpt_path)
        finally:
            self._close_viser_debug_tools()
            if not self.play_policy and self.rank == 0:
                print("=" * 10)
                print("TRAINING SUMMARY")
                print("Student Update Steps:", self.student_update_steps)
            self._finish_wandb()

    def save(self, filename):
        checkpoint = {
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "frame": self.frame,
            "epoch": self.epoch_num,
        }
        torch.save(checkpoint, filename)

    def load_networks(self, params):
        builder = ModelBuilder()
        return builder.load(params)

    def load_yaml(self, cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
