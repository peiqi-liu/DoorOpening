# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play/evaluate DooropeningMulti with one RL-Games teacher checkpoint per door family."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import pathlib
import sys

from isaaclab.app import AppLauncher


SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_CFG = (
    SCRIPT_ROOT / "source" / "DoorOpening" / "tasks" / "dooropening" / "agents" / "pcd_transformer_dagger_cfg.yaml"
)


def _default_teacher_path(family_name: str) -> str:
    return str(SCRIPT_ROOT / "source" / "DoorOpening" / "assets" / "door" / family_name / "door_opening.pth")


# add argparse arguments
parser = argparse.ArgumentParser(description="Play/evaluate DooropeningMulti with RL-Games teacher checkpoints.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in env steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Fallback checkpoint used for every family.")
parser.add_argument(
    "--teacher_cfg",
    type=str,
    default=None,
    help="Teacher RL-Games YAML. Defaults to the hydra agent config, matching run_multi_distillation.py.",
)
parser.add_argument(
    "--teacher_obs_type",
    type=str,
    default="policy",
    choices=["policy", "critic"],
    help="Observation group fed to teacher actor models. RL-Games play uses policy.",
)
parser.add_argument("--student_cfg", type=str, default=None, help="Student DAgger YAML used to mirror env overrides.")
parser.add_argument(
    "--teacher-partnetv5",
    "--teacher_partnetv5",
    dest="teacher_partnetv5",
    type=str,
    default=None,
    help="Teacher checkpoint for PartNetv5.",
)
parser.add_argument(
    "--teacher-partnetv6",
    "--teacher_partnetv6",
    dest="teacher_partnetv6",
    type=str,
    default=None,
    help="Teacher checkpoint for PartNetv6.",
)
parser.add_argument(
    "--teacher-partnetv7",
    "--teacher_partnetv7",
    dest="teacher_partnetv7",
    type=str,
    default=None,
    help="Teacher checkpoint for PartNetv7.",
)
parser.add_argument(
    "--teacher-partnetv8",
    "--teacher_partnetv8",
    dest="teacher_partnetv8",
    type=str,
    default=None,
    help="Teacher checkpoint for PartNetv8.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_steps", type=int, default=2000, help="Stop after this many env steps. Use 0 to run forever.")
parser.add_argument("--print_interval", type=int, default=100, help="Print rollout stats every N env steps.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--train_mode",
    action="store_true",
    default=True,
    help="Use DAgger/training rollout semantics: random reference resets and early stopping.",
)
parser.add_argument(
    "--play_mode",
    action="store_false",
    dest="train_mode",
    help="Use play semantics: reset from frame 0 and disable early stopping.",
)
parser.add_argument(
    "--pregrasp_override",
    action="store_true",
    dest="pregrasp_override",
    default=False,
    help="Apply env.override_pregrasp_actions after family policy routing.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import hashlib
import os
import random
import time

import torch
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets
from DoorOpening.assets.door.multi_door_cfg import ALL_DOOR_CONFIGS as MULTI_DOOR_CONFIGS
from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES, asset_family_ids, asset_paths

import DoorOpening.tasks  # noqa: F401


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


def _resolve_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return retrieve_file_path(str(path))
    repo_path = SCRIPT_ROOT / path
    if repo_path.exists():
        return retrieve_file_path(str(repo_path))
    return retrieve_file_path(path_value)


def _resolve_teacher_paths() -> dict[str, str]:
    cli_values = {
        "PartNetv5": args_cli.teacher_partnetv5,
        "PartNetv6": args_cli.teacher_partnetv6,
        "PartNetv7": args_cli.teacher_partnetv7,
        "PartNetv8": args_cli.teacher_partnetv8,
    }
    teacher_paths = {}
    missing = []
    for family_name in DOOR_FAMILY_NAMES:
        configured = cli_values.get(family_name)
        if configured is not None:
            teacher_paths[family_name] = _resolve_path(configured)
            continue
        default_path = pathlib.Path(_default_teacher_path(family_name))
        if default_path.exists():
            teacher_paths[family_name] = str(default_path)
        else:
            missing.append(family_name)
    if missing:
        raise FileNotFoundError(
            "Missing teacher checkpoint(s) for {}. Provide --teacher-partnetvX, "
            "or use --checkpoint as a fallback for all families.".format(", ".join(missing))
        )
    return teacher_paths


def _resolve_teacher_cfg_path(path_value: str | None) -> str:
    if path_value is None:
        return str(SCRIPT_ROOT / "source" / "DoorOpening" / "tasks" / "dooropening" / "agents" / "rl_games_ppo_cfg.yaml")
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(SCRIPT_ROOT / path)


def _resolve_student_cfg_path(path_value: str | None) -> str:
    if path_value is None:
        return str(DEFAULT_STUDENT_CFG)
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(SCRIPT_ROOT / path)


def _load_student_dagger_defaults(student_cfg_path: str) -> dict:
    if not student_cfg_path or not os.path.exists(student_cfg_path):
        return {}
    with open(student_cfg_path, "r") as f:
        student_cfg = yaml.safe_load(f) or {}
    if not isinstance(student_cfg, dict):
        return {}
    dagger_cfg = student_cfg.get("dagger", {})
    return dict(dagger_cfg) if isinstance(dagger_cfg, dict) else {}


def _load_teacher_params(agent_cfg: dict) -> dict:
    if args_cli.teacher_cfg is None:
        return agent_cfg["params"]
    cfg_path = _resolve_teacher_cfg_path(args_cli.teacher_cfg)
    with open(cfg_path, "r") as f:
        cfg_data = yaml.safe_load(f) or {}
    return cfg_data["params"]


def _get_teacher_clip_obs(teacher_params: dict) -> float:
    env_params = teacher_params.get("env", {})
    return float(env_params.get("clip_observations", math.inf))


def _clip_teacher_obs(obs: torch.Tensor, clip_obs: float) -> torch.Tensor:
    if math.isfinite(clip_obs):
        return torch.clamp(obs, -clip_obs, clip_obs)
    return obs


def _extract_model_state(weights):
    if isinstance(weights, dict):
        if "model" in weights:
            return weights["model"], weights
        if "state_dict" in weights:
            return weights["state_dict"], weights
        if "model_state_dict" in weights:
            return weights["model_state_dict"], weights
    return weights, None


def _load_checkpoint_state(ckpt):
    try:
        return torch_ext.load_checkpoint(ckpt)
    except Exception:
        return torch.load(ckpt, map_location="cpu")


def _sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_teacher_weights(model, ckpt, strict=True, allow_adjust=True):
    weights = _load_checkpoint_state(ckpt)
    state_dict, meta = _extract_model_state(weights)
    if allow_adjust:
        state_dict = adjust_state_dict_keys(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=strict)
    if meta is not None and "running_mean_std" in meta:
        model.running_mean_std.load_state_dict(meta["running_mean_std"])
    print(f"Loaded teacher checkpoint: {ckpt} sha256={_sha256_file(ckpt)}")


def _build_teacher_models(agent_cfg: dict, base_env, device: torch.device) -> tuple[dict[int, torch.nn.Module], float]:
    teacher_params = _load_teacher_params(agent_cfg)
    print(f"[INFO] Teacher obs type: {args_cli.teacher_obs_type}")
    teacher_clip_obs = _get_teacher_clip_obs(teacher_params)
    print(f"[INFO] Teacher obs clip: {teacher_clip_obs}")
    teacher_network = ModelBuilder().load(teacher_params)
    teacher_model_config = {
        "actions_num": int(base_env.cfg.action_space),
        "input_shape": (int(base_env.cfg.observation_space),),
        "num_seqs": int(base_env.num_envs),
        "value_size": 1,
        "normalize_value": teacher_params["config"]["normalize_value"],
        "normalize_input": teacher_params["config"]["normalize_input"],
    }

    teacher_paths = _resolve_teacher_paths()
    teacher_models = {}
    for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
        resume_path = teacher_paths[family_name]
        print(f"[INFO] Loading {family_name} teacher checkpoint from: {resume_path}")
        print(f"[INFO] Teacher hash {family_name}: sha256={_sha256_file(resume_path)}")
        model = teacher_network.build(teacher_model_config).to(device)
        _load_teacher_weights(model, resume_path)
        model.eval()
        teacher_models[family_id] = model
    print("Loaded multi-teacher families:", ", ".join(DOOR_FAMILY_NAMES))
    return teacher_models, teacher_clip_obs


def _get_env_family_ids(base_env) -> torch.Tensor:
    if hasattr(base_env, "env_asset_indices"):
        env_asset_indices = base_env.env_asset_indices.to(device=base_env.device, dtype=torch.long)
    else:
        env_asset_indices = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long) % len(asset_paths)
    return asset_family_ids.to(device=base_env.device, dtype=torch.long)[env_asset_indices]


def _print_family_mapping(base_env, env_family_ids: torch.Tensor):
    family_counts = {}
    for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
        family_counts[family_name] = int((env_family_ids == int(family_id)).sum().detach().cpu().item())
    print("[INFO] Door family env counts:", family_counts)

    env_asset_indices = getattr(base_env, "env_asset_indices", None)
    if env_asset_indices is None:
        return
    sample = []
    sample_count = min(12, int(base_env.num_envs))
    for env_id in range(sample_count):
        asset_idx = int(env_asset_indices[env_id].detach().cpu().item())
        family_name = DOOR_FAMILY_NAMES[int(env_family_ids[env_id].detach().cpu().item())]
        asset_path = pathlib.Path(asset_paths[asset_idx])
        sample.append(f"env{env_id}:{family_name}/{asset_path.parent.name}")
    print("[INFO] Door family sample:", ", ".join(sample))


def _format_completed_rates(completed_successes_by_family: dict[str, list[float]]) -> str:
    parts = []
    for family_name in DOOR_FAMILY_NAMES:
        values = completed_successes_by_family[family_name]
        if len(values) == 0:
            parts.append(f"{family_name}=n/a")
        else:
            parts.append(f"{family_name}={sum(values) / len(values):.3f}({len(values)})")
    return ", ".join(parts)


def _format_asset_rate(asset_idx: int, values: list[float]) -> str:
    if len(values) == 0:
        rate = "n/a"
    else:
        rate = f"{sum(values) / len(values):.3f}({len(values)})"
    asset_path = pathlib.Path(asset_paths[asset_idx])
    family_name = DOOR_FAMILY_NAMES[int(asset_family_ids[asset_idx].item())]
    return f"{asset_idx}:{family_name}/{asset_path.parent.name}={rate}"


def _format_asset_rates(completed_successes_by_asset: dict[int, list[float]], limit: int = 16) -> str:
    parts = []
    for asset_idx in sorted(completed_successes_by_asset)[:limit]:
        parts.append(_format_asset_rate(asset_idx, completed_successes_by_asset[asset_idx]))
    if len(completed_successes_by_asset) > limit:
        parts.append(f"... +{len(completed_successes_by_asset) - limit} assets")
    return ", ".join(parts)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Evaluate the same multi-teacher action path used by multi_pcd_dagger.py."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    dagger_runtime_cfg = _load_student_dagger_defaults(_resolve_student_cfg_path(args_cli.student_cfg))
    if "reset_progress_total" in dagger_runtime_cfg:
        env_cfg.reset_progress_total = dagger_runtime_cfg["reset_progress_total"]
    if "adr_reset_progress_total" in dagger_runtime_cfg:
        env_cfg.adr_reset_progress_total = dagger_runtime_cfg["adr_reset_progress_total"]
    else:
        env_cfg.adr_reset_progress_total = 0.5 * float(env_cfg.reset_progress_total)
    pointcloud_source = str(dagger_runtime_cfg.get("pointcloud_source", "both")).lower()
    env_cfg.pointcloud_render_mode = "none"
    env_cfg.enable_pointcloud_camera = False
    print(f"Distillation reset_progress_total: {env_cfg.reset_progress_total}")
    print(f"Distillation adr_reset_progress_total: {env_cfg.adr_reset_progress_total}")
    print(f"Distillation pointcloud_render_mode: {env_cfg.pointcloud_render_mode}")

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", "door_opening_multi_teacher_play"))
    os.makedirs(log_root_path, exist_ok=True)
    env_cfg.log_dir = log_root_path

    device = torch.device(agent_cfg["params"]["config"]["device"])

    preconvert_shared_urdf_assets(door_configs=MULTI_DOOR_CONFIGS, verbose=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.unwrapped.ref_motion_lib.reset_from_start = True
    env.unwrapped.early_stopping = bool(args_cli.train_mode)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos", "play_multi"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during multi-teacher play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    base_env = env.unwrapped
    env_family_ids = _get_env_family_ids(base_env)
    env_family_ids_policy = env_family_ids.to(device=device)
    _print_family_mapping(base_env, env_family_ids)

    teacher_models, teacher_clip_obs = _build_teacher_models(agent_cfg, base_env, device)

    dt = env.unwrapped.step_dt
    obs, _ = env.reset()
    prev_actions = base_env.applied_robot_dof_targets.detach().clone().to(device=device)

    timestep = 0
    local_reached_success = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)
    completed_successes_by_family = {family_name: [] for family_name in DOOR_FAMILY_NAMES}
    if hasattr(base_env, "env_asset_indices"):
        env_asset_indices_policy = base_env.env_asset_indices.to(device=device, dtype=torch.long)
    else:
        env_asset_indices_policy = torch.arange(env.unwrapped.num_envs, device=device, dtype=torch.long) % len(
            asset_paths
        )
    completed_successes_by_asset = {
        int(asset_idx): [] for asset_idx in sorted(set(env_asset_indices_policy.detach().cpu().tolist()))
    }

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            current_frame_idx = torch.clamp(
                base_env.ref_motion_lib.frame_idx.to(device=device, dtype=torch.float32),
                max=float(base_env.success_frame_idx),
            )
            local_reached_success |= current_frame_idx >= float(base_env.success_frame_idx)

            actions = torch.zeros((env.unwrapped.num_envs, int(env.unwrapped.cfg.action_space)), device=device)
            for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
                env_ids = torch.nonzero(env_family_ids_policy == int(family_id), as_tuple=False).squeeze(-1)
                if env_ids.numel() == 0:
                    continue
                batch_dict = {
                    "is_train": False,
                    "obs": _clip_teacher_obs(obs[args_cli.teacher_obs_type][env_ids], teacher_clip_obs),
                    "prev_actions": prev_actions[env_ids],
                }
                res_dict = teacher_models[family_id](batch_dict)
                actions[env_ids] = torch.clamp(res_dict["mus"], -1.0, 1.0)

            override_fn = getattr(base_env, "override_pregrasp_actions", None)
            if args_cli.pregrasp_override and callable(override_fn):
                actions = override_fn(actions.to(device=base_env.device)).to(device=device)

            obs, _, terminated, truncated, _ = env.step(actions)
            dones = (terminated | truncated).to(device=device)
            prev_actions = base_env.applied_robot_dof_targets.detach().clone().to(device=device)

            done_env_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                timeout_success = torch.zeros_like(local_reached_success[done_env_ids])
                if hasattr(base_env, "reset_time_outs"):
                    timeout_success = base_env.reset_time_outs.to(device=device, dtype=torch.bool)[done_env_ids]
                done_successes = (local_reached_success[done_env_ids] | timeout_success).detach().cpu().tolist()
                done_family_ids = env_family_ids_policy[done_env_ids].detach().cpu().tolist()
                done_asset_ids = env_asset_indices_policy[done_env_ids].detach().cpu().tolist()
                for family_id, asset_idx, success in zip(done_family_ids, done_asset_ids, done_successes):
                    completed_successes_by_family[DOOR_FAMILY_NAMES[int(family_id)]].append(float(success))
                    completed_successes_by_asset[int(asset_idx)].append(float(success))
                local_reached_success[done_env_ids] = False
                prev_actions[done_env_ids] = base_env.applied_robot_dof_targets.detach().to(device=device)[done_env_ids]

        timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[RESULT] steps={timestep} completed_success={_format_completed_rates(completed_successes_by_family)}")
    print(f"[RESULT] asset_success={_format_asset_rates(completed_successes_by_asset, limit=64)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
