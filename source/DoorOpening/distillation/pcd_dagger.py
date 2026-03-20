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
        self.ignore_aux_debug = bool(self.runtime_cfg.get("ignore_aux_debug", False))
        self.sampler_render_cfg = self.runtime_cfg.get("sampler_render", {})
        self.sampler_render_inflate_px = int(self.sampler_render_cfg.get("inflate_px", 2))
        self.sampler_render_jitter_std_m = float(self.sampler_render_cfg.get("jitter_std_m", 0.004))
        self.sampler_render_clip_mode = str(self.sampler_render_cfg.get("clip_mode", "post"))
        self.sampler_render_jitter_mode = str(self.sampler_render_cfg.get("jitter_mode", "xyz"))
        self.sampler_render_use_compile = bool(self.sampler_render_cfg.get("use_compile", True))
        if self.pointcloud_source not in {"sampler", "depth"}:
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

        self.prev_actions_student = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.prev_actions_teacher = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.completed_rewards = deque(maxlen=self.games_to_track)
        self.completed_lengths = deque(maxlen=self.games_to_track)
        self.completed_successes = deque(maxlen=self.games_to_track)
        self.completed_timeout_successes = deque(maxlen=self.games_to_track)
        self.episode_reached_last_frame = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.latest_env_metrics = {}
        self.student_update_steps = 0
        self.last_local_update_batch_size = 0
        self.last_global_update_batch_size = 0
        self._timing_stats = {
            "iteration_ms": {"sum_ms": 0.0, "count": 0},
            "student_obs_ms": {"sum_ms": 0.0, "count": 0},
            "pointcloud_ms": {"sum_ms": 0.0, "count": 0},
            "env_step_ms": {"sum_ms": 0.0, "count": 0},
        }

        self._init_teacher()
        self._init_student()
        self._init_pointcloud_assets()
        self.success_frame_idx = float(self.ov_env.ref_motion_lib.num_frames - 1)

        self.aux_buffer = None
        if not self.ignore_aux_debug and self.student_model.aux_prediction and self.aux_total_dim > 0:
            self.aux_buffer = torch.zeros((self.num_envs, 1, self.aux_total_dim), dtype=torch.float32, device=self.device)
        if self.ignore_aux_debug and self.rank == 0:
            print("Aux debug bypass enabled: using ground-truth aux inputs and skipping aux loss/rollout.")

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

        print(self.student_model)
        print("state_encoders_cfg", self.student_model.state_encoders_cfg)

        self.student_ddp_find_unused_parameters = bool(
            self.student_model.aux_prediction and self.ignore_aux_debug
        )

        if self.use_ddp:
            self.student_model_ddp = DDP(
                self.student_model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.student_ddp_find_unused_parameters,
            )
        else:
            self.student_model_ddp = self.student_model

        if self.student_ddp_find_unused_parameters and self.rank == 0:
            print(
                "DDP unused-parameter detection enabled because aux prediction is active "
                "while ignore_aux_debug skips the aux loss."
            )
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

        self.aux_state_specs = OrderedDict()
        for key in self.state_encoders_keys:
            if key == "aux_object_state":
                input_dim = int(self.student_model.state_encoders_cfg[key]["input_dim"])
                if input_dim != 3:
                    raise ValueError("aux_object_state must have input_dim=3.")
                self.aux_state_specs[key] = {"dim": input_dim, "getter_name": "_get_aux_object_state"}
            elif key == "aux_door_joint_angle":
                input_dim = int(self.student_model.state_encoders_cfg[key]["input_dim"])
                expected_dim = len(self.ov_env._door_joint_idx)
                if input_dim != expected_dim:
                    raise ValueError(f"aux_door_joint_angle must have input_dim={expected_dim}.")
                self.aux_state_specs[key] = {"dim": input_dim, "getter_name": "_get_aux_door_joint_angle"}

        unknown_aux_keys = [
            key
            for key in self.state_encoders_keys
            if key.startswith("aux_") and key not in self.aux_state_specs
        ]
        if unknown_aux_keys:
            raise KeyError(
                f"Unsupported aux state keys in student config: {unknown_aux_keys}. "
                "Add a getter in pcd_dagger.py before enabling them."
            )

        self.aux_state_keys = tuple(self.aux_state_specs.keys())
        self.aux_total_dim = sum(spec["dim"] for spec in self.aux_state_specs.values())
        if self.student_model.aux_prediction and self.student_model.aux_output_dim != self.aux_total_dim:
            raise ValueError(
                f"Student aux output dim ({self.student_model.aux_output_dim}) does not match "
                f"configured aux state dim ({self.aux_total_dim})."
            )

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

        if self.local_pcd_points[2] > 0 and "aux_object_state" not in self.aux_state_specs:
            raise ValueError("local_pcd_t aux crop requires aux_object_state to be enabled in state_encoders_cfg.")

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

        if "link_2" in self.ov_env.door_body_names:
            handle_name_idx = self.ov_env.door_body_names.index("link_2")
        else:
            handle_name_idx = len(self.ov_env.door_body_names) - 1
        self.robot_base_body_idx = int(self.ov_env._robot_base_body_link_idx)
        self.robot_palm_body_idx = int(self.ov_env._robot_key_body_idx[self.ov_env._robot_palm_id_in_key_body_idx])
        self.robot_camera_body_idx = int(self.ov_env.robot.find_bodies("x5_camera_link")[0][0])
        self.robot_root_body_idx = int(self.ov_env._robot_base_link_idx[0])
        self.door_base_body_idx = int(self.ov_env._door_base_link_idx)
        self.door_handle_body_idx = int(self.ov_env._door_body_idx[handle_name_idx])
        self.door_aux_joint_idx = torch.as_tensor(self.ov_env._door_joint_idx, device=self.device, dtype=torch.long)
        camera_cfg = self.ov_env.cfg.pointcloud_camera_cfg
        self.camera_offset_pos = torch.tensor(camera_cfg.offset.pos, device=self.device, dtype=torch.float32)
        self.camera_offset_quat_world = torch.tensor(camera_cfg.offset.rot, device=self.device, dtype=torch.float32)
        self.pointcloud_camera = getattr(self.ov_env, "pointcloud_camera", None)
        if self.pointcloud_source == "depth" and self.pointcloud_camera is None:
            raise ValueError("pointcloud_source='depth' requires DooropeningEnv to enable the pointcloud camera.")
        self.sampler_camera_spec = self._build_sampler_camera_spec()

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

    def _aux_to_2d(self, aux_tensor):
        if aux_tensor is None:
            return None
        if aux_tensor.ndim == 3:
            return aux_tensor[:, 0, :]
        return aux_tensor

    def _flatten_aux_dict(self, aux_dict):
        if not self.aux_state_keys:
            return None
        aux_parts = []
        for key in self.aux_state_keys:
            if key not in aux_dict:
                raise KeyError(f"Missing aux state '{key}' while flattening aux dict.")
            aux_parts.append(self._aux_to_2d(aux_dict[key]))
        return torch.cat(aux_parts, dim=-1)

    def _split_aux_tensor(self, aux_tensor):
        aux_tensor_2d = self._aux_to_2d(aux_tensor)
        if aux_tensor_2d is None:
            return OrderedDict()

        aux_dict = OrderedDict()
        start_idx = 0
        for key, spec in self.aux_state_specs.items():
            end_idx = start_idx + spec["dim"]
            aux_dict[key] = aux_tensor_2d[:, start_idx:end_idx]
            start_idx = end_idx
        return aux_dict

    def _get_aux_state_dict(self):
        aux_state = OrderedDict()
        for key, spec in self.aux_state_specs.items():
            # print("key", key)
            # print("spec", spec)
            # print("getter_name", spec["getter_name"])
            aux_state[key] = getattr(self, spec["getter_name"])()
            # print("aux_state[key]", aux_state[key])
        return aux_state

    def _decode_aux_prediction(self, aux_pred, prev_abs_aux):
        prev_abs_aux = self._aux_to_2d(prev_abs_aux)
        if prev_abs_aux is None:
            return aux_pred.detach()
        if self.student_model.aux_prediction_mode == "delta":
            aux_delta = torch.clamp(aux_pred, -1.0, 1.0)
            return prev_abs_aux.unsqueeze(1) + self.student_model.aux_delta_scale * aux_delta
        return self._aux_to_2d(aux_pred).unsqueeze(1)

    def _get_aux_target(self, current_aux, prev_abs_aux):
        prev_abs_aux = self._aux_to_2d(prev_abs_aux)
        if prev_abs_aux is None:
            return current_aux
        if self.student_model.aux_prediction_mode == "delta":
            target_delta = (current_aux - prev_abs_aux) / self.student_model.aux_delta_scale
            return torch.clamp(target_delta, -1.0, 1.0)
        return current_aux

    def _get_teacher_actions(self, obs):
        if self.teacher_model is None:
            raise RuntimeError("Teacher model is not initialized.")
        batch_dict = {
            "is_train": False,
            "obs": obs[self.teacher_obs_type],
            "prev_actions": self.prev_actions_teacher,
        }
        with torch.no_grad():
            res_dict = self.teacher_model(batch_dict)
        mus = res_dict["mus"]
        return {
            "mus": mus,
            "actions": torch.clamp(mus, -1.0, 1.0),
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

    def _sample_door_pointcloud_base(self):
        self._sync_timing_device()
        start_time = time.perf_counter()
        if self.pointcloud_source == "sampler":
            door_pcd_base = self._sample_door_pointcloud_base_sampler()
        else:
            door_pcd_base = self._sample_door_pointcloud_base_depth()
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

    def _get_aux_object_state(self):
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        handle_pos_w = self.ov_env.door.data.body_pos_w[:, self.door_handle_body_idx].unsqueeze(1)
        return world_to_local(handle_pos_w, robot_base_pos_w, robot_base_quat_w).squeeze(1)

    def _get_aux_door_joint_angle(self):
        return self.ov_env.door.data.joint_pos[:, self.door_aux_joint_idx]

    def _get_aux_crop_center(self, aux_state_dict):
        if "aux_object_state" not in aux_state_dict:
            raise KeyError("aux_object_state is required to crop the auxiliary local pointcloud.")
        return aux_state_dict["aux_object_state"]

    def _build_local_pcd(self, door_pcd_base, palm_pos_base, aux_crop_center):
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

        if self.local_pcd_points[2] > 0:
            aux_crop, _ = crop_local_pcd(
                door_pcd_base,
                local_range=self.local_pcd_range[2],
                num_local_points=self.local_pcd_points[2],
                is_cylindrical=False,
                crop_center=aux_crop_center,
                x_direction_cutoff=None,
                log_name="aux",
            )
            pcd_parts.append(aux_crop)

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
        q_pos = self.ov_env.robot.data.joint_pos
        robot_base_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_base_body_idx]
        robot_base_quat_w = self.ov_env.robot.data.body_quat_w[:, self.robot_base_body_idx]
        palm_pos_w = self.ov_env.robot.data.body_pos_w[:, self.robot_palm_body_idx].unsqueeze(1)

        aux_gt_dict = self._get_aux_state_dict()
        aux_gt = self._flatten_aux_dict(aux_gt_dict)
        if self.ignore_aux_debug and aux_gt is not None:
            aux_input = torch.zeros_like(aux_gt)
        elif self.aux_buffer is not None:
            aux_input = self._aux_to_2d(self.aux_buffer)
            if torch.count_nonzero(self.aux_buffer) == 0:
                aux_input = aux_gt
        else:
            aux_input = aux_gt
        aux_input_dict = self._split_aux_tensor(aux_input)

        palm_pos_base = world_to_local(palm_pos_w, robot_base_pos_w, robot_base_quat_w).squeeze(1)
        door_pcd_base = self._sample_door_pointcloud_base()
        aux_crop_center = palm_pos_base if self.ignore_aux_debug else self._get_aux_crop_center(aux_gt_dict)

        obs = OrderedDict()
        for key in self.state_encoders_keys:
            if key == "q_base":
                obs[key] = q_pos[:, self.ov_env._robot_base_dof_idx]
            elif key == "q_arm":
                obs[key] = q_pos[:, self.ov_env._robot_arm_dof_idx]
            elif key == "q_hand":
                obs[key] = q_pos[:, self.ov_env._robot_finger_dof_idx]
            elif key == "prev_action":
                obs[key] = self.prev_actions_student
            elif key in self.aux_state_specs:
                obs[key] = aux_input_dict[key]
            else:
                raise KeyError(f"Unsupported student state key '{key}' in config.")

        for key in self.pcd_encoders_keys:
            if key == "local_pcd_t":
                # self._debug_visualize_pointcloud(door_pcd_base, "door_pcd_base")
                obs[key] = self._build_local_pcd(
                    door_pcd_base,
                    palm_pos_base,
                    aux_crop_center,
                )
                # self._debug_visualize_pointcloud(obs[key], "local_pcd_t")
            else:
                raise KeyError(f"Unsupported student pointcloud key '{key}' in config.")

        return obs, aux_gt

    def _student_forward(self, student_obs):
        return self.student_model_ddp(student_obs)

    def _compute_student_loss(self, student_output, teacher_actions, aux_gt, student_obs):
        action_loss = F.mse_loss(student_output["action"][:, 0, :], teacher_actions)
        total_loss = action_loss
        aux_loss = None
        if not self.ignore_aux_debug and self.student_model.aux_prediction and "aux" in student_output:
            prev_aux = None
            if self.aux_state_keys and all(key in student_obs for key in self.aux_state_keys):
                prev_aux = self._flatten_aux_dict(OrderedDict((key, student_obs[key]) for key in self.aux_state_keys))
            aux_target = self._get_aux_target(aux_gt, prev_aux)
            aux_loss = F.mse_loss(student_output["aux"][:, 0, :], aux_target)
            total_loss = total_loss + self.student_model.aux_weight * aux_loss
        return total_loss, action_loss, aux_loss

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

    def _mix_actions(self, student_actions, teacher_actions, iteration):
        if self.play_policy or teacher_actions is None:
            return student_actions, 0.0
        beta = self._get_teacher_forcing_beta(iteration)
        if beta <= 0.0:
            return student_actions, beta
        if beta >= 1.0:
            return teacher_actions, beta
        teacher_mask = torch.rand(self.num_envs, device=self.device) < beta
        step_actions = student_actions.clone()
        step_actions[teacher_mask] = teacher_actions[teacher_mask]
        return step_actions, beta

    def _reset_aux_buffer(self, done_mask):
        if self.aux_buffer is None or done_mask.numel() == 0:
            return
        aux_gt = self._flatten_aux_dict(self._get_aux_state_dict())
        self.aux_buffer[done_mask, 0, :] = aux_gt[done_mask]

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

    def _to_loggable_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        if isinstance(value, (int, float, bool)):
            return float(value)
        return None

    def _collect_env_metrics(self, info):
        metrics = {}
        if isinstance(info, dict):
            for key, value in info.items():
                scalar = self._to_loggable_scalar(value)
                if scalar is not None:
                    metrics[key] = scalar

        return metrics

    def _update_completed_episode_metrics(self, done_mask, timed_out):
        if done_mask.numel() == 0:
            return

        episode_rewards = self.current_rewards[done_mask, 0].detach().cpu().tolist()
        episode_lengths = self.current_lengths[done_mask].detach().cpu().tolist()
        episode_successes = (
            self.episode_reached_last_frame[done_mask] | timed_out[done_mask]
        ).to(dtype=torch.float32).detach().cpu().tolist()
        episode_timeouts = timed_out[done_mask].to(dtype=torch.float32).detach().cpu().tolist()

        self.completed_rewards.extend(float(value) for value in episode_rewards)
        self.completed_lengths.extend(float(value) for value in episode_lengths)
        self.completed_successes.extend(float(value) for value in episode_successes)
        self.completed_timeout_successes.extend(float(value) for value in episode_timeouts)

    def _log(self, iteration, total_loss, action_loss, aux_loss, teacher_forcing_beta):
        if iteration % self.log_interval != 0:
            return
        mean_reward = self.current_rewards.mean().item()
        mean_length = self.current_lengths.mean().item()
        episode_reward = self._mean_completed_metric(self.completed_rewards)
        episode_length = self._mean_completed_metric(self.completed_lengths)
        success_rate = self._mean_completed_metric(self.completed_successes)
        timeout_success_rate = self._mean_completed_metric(self.completed_timeout_successes)
        active_success_rate = self.episode_reached_last_frame.float().mean().item()
        mean_ref_frame_idx = self.ov_env.ref_motion_lib.frame_idx.float().mean().item()
        success_region = None
        if hasattr(self.ov_env, "in_success_region"):
            success_region = self.ov_env.in_success_region.float().mean().item()
        timing_means = self._consume_timing_means()

        if self.rank == 0:
            print("=" * 10)
            print("ITERATION:", iteration)
            print("Total Loss:", float(total_loss.detach().cpu()))
            print("Action Loss:", float(action_loss.detach().cpu()))
            print("Teacher Forcing Beta:", teacher_forcing_beta)
            print("Student Update Steps:", self.student_update_steps)
            print("Last Local Update Batch Size:", self.last_local_update_batch_size)
            print("Last Global Update Batch Size:", self.last_global_update_batch_size)
            if aux_loss is not None:
                print("Aux Loss:", float(aux_loss.detach().cpu()))
            # print("Mean Reward:", mean_reward)
            # print("Mean Length:", mean_length)
            # if episode_reward is not None:
            #     print("Episode Reward:", episode_reward)
            if episode_length is not None:
                print("Episode Length:", episode_length)
            if success_rate is not None:
                print("Success Rate:", success_rate)
            # print("Active Success Rate:", active_success_rate)
            if timeout_success_rate is not None:
                print("Timeout Success Rate:", timeout_success_rate)
            # print("Mean Ref Frame Idx:", mean_ref_frame_idx)
            if success_region is not None:
                print("Success Region:", success_region)
            if timing_means["iteration_ms"] is not None:
                print("Iteration Time (ms):", timing_means["iteration_ms"])
            # if timing_means["student_obs_ms"] is not None:
            #     print("Student Obs Time (ms):", timing_means["student_obs_ms"])
            # if timing_means["pointcloud_ms"] is not None:
            #     print("Pointcloud Time (ms):", timing_means["pointcloud_ms"])
            # if timing_means["env_step_ms"] is not None:
            #     print("Env Step Time (ms):", timing_means["env_step_ms"])

        metrics = {
            "loss/total": float(total_loss.detach().cpu()),
            "loss/action": float(action_loss.detach().cpu()),
            "stats/mean_reward": mean_reward,
            "stats/mean_length": mean_length,
            "stats/active_success_rate": active_success_rate,
            "stats/ref_frame_idx": mean_ref_frame_idx,
            "stats/completed_episodes": len(self.completed_lengths),
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
        if success_rate is not None:
            metrics["stats/success_rate"] = success_rate
        if timeout_success_rate is not None:
            metrics["stats/timeout_success_rate"] = timeout_success_rate
        if success_region is not None:
            metrics["stats/success_region"] = success_region
        if teacher_forcing_beta is not None:
            metrics["stats/teacher_forcing_beta"] = teacher_forcing_beta
        if timing_means["iteration_ms"] is not None:
            metrics["timing/iteration_ms"] = timing_means["iteration_ms"]
        if timing_means["student_obs_ms"] is not None:
            metrics["timing/student_obs_ms"] = timing_means["student_obs_ms"]
        if timing_means["pointcloud_ms"] is not None:
            metrics["timing/pointcloud_ms"] = timing_means["pointcloud_ms"]
        if timing_means["env_step_ms"] is not None:
            metrics["timing/env_step_ms"] = timing_means["env_step_ms"]
        metrics.update(self.latest_env_metrics)
        self._wandb_log(metrics, step=iteration)

    def distill(self):
        if not self.play_policy and self.teacher_model is None:
            raise RuntimeError("Teacher model must be initialized for distillation.")

        self.student_model_ddp.train(not self.play_policy)
        if self.teacher_model is not None:
            self.teacher_model.eval()

        try:
            obs = self.env.reset()[0]
            if self.aux_buffer is not None:
                self.aux_buffer[:, 0, :] = self._flatten_aux_dict(self._get_aux_state_dict())
            self.episode_reached_last_frame.zero_()

            for iteration in range(self.num_iters):
                self._sync_timing_device()
                iteration_start_time = time.perf_counter()

                self._sync_timing_device()
                student_obs_start_time = time.perf_counter()
                student_obs, aux_gt = self._build_student_obs()
                self._sync_timing_device()
                self._record_timing("student_obs_ms", time.perf_counter() - student_obs_start_time)
                student_output = self._student_forward(student_obs)
                student_actions = torch.clamp(student_output["action"][:, 0, :], -1.0, 1.0)

                teacher_actions = None
                total_loss = None
                action_loss = None
                aux_loss = None

                if not self.play_policy:
                    teacher_output = self._get_teacher_actions(obs)
                    teacher_actions = teacher_output["actions"]
                    total_loss, action_loss, aux_loss = self._compute_student_loss(
                        student_output, teacher_output["mus"], aux_gt, student_obs
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

                if self.student_model.aux_prediction and self.aux_buffer is not None and self.aux_state_keys:
                    prev_aux = self._flatten_aux_dict(OrderedDict((key, student_obs[key]) for key in self.aux_state_keys))
                    self.aux_buffer[:] = self._decode_aux_prediction(student_output["aux"].detach(), prev_aux)

                step_actions, teacher_forcing_beta = self._mix_actions(
                    student_actions.detach(),
                    teacher_actions,
                    iteration,
                )
                self._update_last_frame_tracker()
                self._sync_timing_device()
                env_step_start_time = time.perf_counter()
                obs, rew, out_of_reach, timed_out, info = self.env.step(step_actions)
                self._sync_timing_device()
                self._record_timing("env_step_ms", time.perf_counter() - env_step_start_time)
                self.latest_env_metrics = self._collect_env_metrics(info)

                self.prev_actions_student[:] = step_actions
                self.prev_actions_teacher[:] = step_actions
                self.frame += self.num_envs

                self.current_rewards += rew.unsqueeze(-1)
                self.current_lengths += 1
                done_mask = torch.nonzero(out_of_reach | timed_out, as_tuple=False).squeeze(-1)
                if done_mask.numel() > 0:
                    self._update_completed_episode_metrics(done_mask, timed_out)
                    self.current_rewards[done_mask] = 0.0
                    self.current_lengths[done_mask] = 0.0
                    self.prev_actions_student[done_mask] = 0.0
                    self.prev_actions_teacher[done_mask] = 0.0
                    self.episode_reached_last_frame[done_mask] = False
                    self._reset_aux_buffer(done_mask)

                if total_loss is not None:
                    self._sync_timing_device()
                    self._record_timing("iteration_ms", time.perf_counter() - iteration_start_time)
                    self._log(iteration, total_loss, action_loss, aux_loss, teacher_forcing_beta)
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
