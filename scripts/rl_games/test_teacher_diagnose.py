# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose WHY a DooropeningMulti teacher fails on specific doors (e.g. a right-push door).

Unlike scripts/rl_games/play_multi.py (which only reports success rates), this script keeps the
DAgger-style hard *tracking-drift termination* enabled while starting every episode at reference
frame 0, then attributes each episode outcome:

    success            -> reached the reference success frame (or timed out already past it)
    kill:robot_drift   -> key-body pose error exceeded the (curriculum) threshold  [env._get_dones]
    kill:door_drift    -> door hinge-angle error exceeded the threshold
    timeout            -> ran out of steps without reaching success and without a kill

For every kill it records how far along the reference the teacher got (frame index -> phase, from
the door's traj.pkl key_indices), the door opening angle reached, and the drift magnitude vs its
threshold. That tells you at a glance whether the right-push teacher never grasps, grasps but can't
unlatch, or opens partway then drifts.

Example (isolate the family that holds the failing right-push door, 32 envs, 6 episodes each):

    python scripts/rl_games/test_teacher_diagnose.py \
        --task DooropeningMulti --num_envs 32 --episodes_per_env 6 \
        --door-families PartNetv7_plusplus \
        --teacher-partnetv7 source/DoorOpening/assets/door/PartNetv7_plusplus/door_opening.pth \
        --headless

Add --video to record, or --no-early-stopping to see the full run without drift kills.
"""

import argparse
import math
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_CFG = (
    SCRIPT_ROOT / "source" / "DoorOpening" / "tasks" / "dooropening" / "agents" / "pcd_transformer_dagger_cfg.yaml"
)


def _default_teacher_path(family_name: str) -> str:
    return str(SCRIPT_ROOT / "source" / "DoorOpening" / "assets" / "door" / family_name / "door_opening.pth")


parser = argparse.ArgumentParser(description="Diagnose DooropeningMulti teacher failures per door.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the diagnosis run.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in env steps.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Fallback checkpoint used for every family.")
parser.add_argument("--teacher_cfg", type=str, default=None, help="Teacher RL-Games YAML (defaults to hydra agent cfg).")
parser.add_argument("--teacher_obs_type", type=str, default="policy", choices=["policy", "critic"])
parser.add_argument("--student_cfg", type=str, default=None, help="Student DAgger YAML used to mirror env overrides.")
parser.add_argument("--teacher-partnetv5", "--teacher_partnetv5", dest="teacher_partnetv5", type=str, default=None)
parser.add_argument("--teacher-partnetv6", "--teacher_partnetv6", dest="teacher_partnetv6", type=str, default=None)
parser.add_argument("--teacher-partnetv7", "--teacher_partnetv7", dest="teacher_partnetv7", type=str, default=None)
parser.add_argument("--teacher-partnetv8", "--teacher_partnetv8", dest="teacher_partnetv8", type=str, default=None)
parser.add_argument(
    "--door-families",
    "--door_families",
    dest="door_families",
    type=str,
    default=None,
    help="Comma-separated door family list to load (sets DOOROPENING_MULTI_DOOR_FAMILIES). Use this to "
    "isolate the family that contains the failing right-push door.",
)
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--trace-env",
    dest="trace_env",
    type=int,
    default=0,
    help="Print a per-step drift trace for this env id (actual vs desired + drift error). -1 disables.",
)
parser.add_argument(
    "--trace-interval",
    dest="trace_interval",
    type=int,
    default=1,
    help="Print the --trace-env line every N steps (1 = every step).",
)
parser.add_argument("--episodes_per_env", type=int, default=4, help="Stop after ~this many episodes per env.")
parser.add_argument("--max_steps", type=int, default=4000, help="Hard cap on env steps (0 = run until episode budget).")
parser.add_argument("--print_interval", type=int, default=200, help="Print a running summary every N env steps.")
parser.add_argument(
    "--reset-from-start",
    dest="reset_from_start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Start every episode at reference frame 0 (default). --no-reset-from-start uses random DAgger resets.",
)
parser.add_argument(
    "--early-stopping",
    dest="early_stopping",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep the tracking-drift hard termination on (default). --no-early-stopping disables kills.",
)
parser.add_argument(
    "--threshold-progress",
    dest="threshold_progress",
    type=float,
    default=1.0,
    help="Force the drift-threshold curriculum progress in [0,1]. 1.0 = loosest/highest thresholds "
    "(default here, for testing); 0.0 = tightest early-schedule; -1 = use the real curriculum.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# The door family list is read by multi_door_cfg AT IMPORT time, so set it before any heavy imports.
if args_cli.door_families is not None:
    families = [f.strip() for f in args_cli.door_families.split(",") if f.strip()]
    if families:
        os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = ",".join(families)

if args_cli.video:
    args_cli.enable_cameras = True
else:
    os.environ["ENABLE_CAMERAS"] = "0"

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import pickle as pkl
import random
import time
from collections import Counter, defaultdict

import gymnasium as gym
import torch
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets
from DoorOpening.assets.door.multi_door_cfg import ALL_DOOR_CONFIGS as MULTI_DOOR_CONFIGS
from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES, asset_family_ids, asset_paths

try:
    from DoorOpening.constants.door_constants import DOOR_JOINT_NAMES
except Exception:
    DOOR_JOINT_NAMES = ["joint_1", "joint_2"]

import DoorOpening.tasks  # noqa: F401


# Human-readable phase names for the traj.pkl key_indices segments (see compute_waypoint.py).
# Both planners share keyframes 0..5; pull adds 6..10. Index i names the segment that STARTS at
# key_indices[i]. Unknown indices fall back to "seg{i}".
PHASE_NAMES_PUSH = ["start", "pregrasp", "grasp", "rotate-handle", "open-door", "base-forward", "through"]
PHASE_NAMES_PULL = [
    "start", "pregrasp", "grasp", "rotate-handle", "pull-open", "block-pose",
    "retract-arm", "push-panel", "push-panel2", "traverse", "traverse2",
]


def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    adjusted = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted[key] = value
            continue
        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        key_with = ".".join(parts)
        if key_with in model_state_dict:
            adjusted[key_with] = value
            continue
        key_without = key.replace("_orig_mod.", "")
        if key_without in model_state_dict:
            adjusted[key_without] = value
            continue
        adjusted[key] = value
    return adjusted


def _resolve_path(path_value):
    if path_value is None:
        return None
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return retrieve_file_path(str(path))
    repo_path = SCRIPT_ROOT / path
    if repo_path.exists():
        return retrieve_file_path(str(repo_path))
    return retrieve_file_path(path_value)


def _resolve_teacher_paths():
    cli_values = {
        "PartNetv5": args_cli.teacher_partnetv5,
        "PartNetv6": args_cli.teacher_partnetv6,
        "PartNetv7": args_cli.teacher_partnetv7,
        "PartNetv8": args_cli.teacher_partnetv8,
    }
    teacher_paths = {}
    missing = []
    for family_name in DOOR_FAMILY_NAMES:
        configured = cli_values.get(_family_base(family_name))
        if configured is None:
            configured = args_cli.checkpoint
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
            "Missing teacher checkpoint(s) for {}. Provide --teacher-partnetvX, --checkpoint, or a "
            "door_opening.pth in the family folder.".format(", ".join(missing))
        )
    return teacher_paths


def _family_base(family_name):
    # "PartNetv7_plusplus" / "PartNetv7_pro" -> "PartNetv7" so the --teacher-partnetv7 flag matches.
    for base in ("PartNetv5", "PartNetv6", "PartNetv7", "PartNetv8"):
        if str(family_name).startswith(base):
            return base
    return family_name


def _resolve_teacher_cfg_path(path_value):
    if path_value is None:
        return str(SCRIPT_ROOT / "source" / "DoorOpening" / "tasks" / "dooropening" / "agents" / "rl_games_ppo_cfg.yaml")
    path = pathlib.Path(path_value).expanduser()
    return str(path) if path.is_absolute() else str(SCRIPT_ROOT / path)


def _load_student_dagger_defaults(student_cfg_path):
    if not student_cfg_path or not os.path.exists(student_cfg_path):
        return {}
    with open(student_cfg_path, "r") as f:
        student_cfg = yaml.safe_load(f) or {}
    dagger_cfg = student_cfg.get("dagger", {}) if isinstance(student_cfg, dict) else {}
    return dict(dagger_cfg) if isinstance(dagger_cfg, dict) else {}


def _load_teacher_params(agent_cfg):
    if args_cli.teacher_cfg is None:
        return agent_cfg["params"]
    with open(_resolve_teacher_cfg_path(args_cli.teacher_cfg), "r") as f:
        return (yaml.safe_load(f) or {})["params"]


def _clip_teacher_obs(obs, clip_obs):
    return torch.clamp(obs, -clip_obs, clip_obs) if math.isfinite(clip_obs) else obs


def _load_checkpoint_state(ckpt):
    try:
        return torch_ext.load_checkpoint(ckpt)
    except Exception:
        return torch.load(ckpt, map_location="cpu")


def _extract_model_state(weights):
    if isinstance(weights, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in weights:
                return weights[key], weights
    return weights, None


def _load_teacher_weights(model, ckpt):
    weights = _load_checkpoint_state(ckpt)
    state_dict, meta = _extract_model_state(weights)
    state_dict = adjust_state_dict_keys(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=True)
    if meta is not None and "running_mean_std" in meta:
        model.running_mean_std.load_state_dict(meta["running_mean_std"])
    print(f"[INFO] Loaded teacher checkpoint: {ckpt}")


def _build_teacher_models(agent_cfg, base_env, device):
    teacher_params = _load_teacher_params(agent_cfg)
    teacher_clip_obs = float(teacher_params.get("env", {}).get("clip_observations", math.inf))
    print(f"[INFO] Teacher obs type: {args_cli.teacher_obs_type}  clip: {teacher_clip_obs}")
    teacher_network = ModelBuilder().load(teacher_params)
    model_config = {
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
        print(f"[INFO] {family_name} teacher <- {resume_path}")
        model = teacher_network.build(model_config).to(device)
        _load_teacher_weights(model, resume_path)
        model.eval()
        teacher_models[family_id] = model
    return teacher_models, teacher_clip_obs


def _get_env_family_ids(base_env):
    if hasattr(base_env, "env_asset_indices"):
        idx = base_env.env_asset_indices.to(device=base_env.device, dtype=torch.long)
    else:
        idx = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long) % len(asset_paths)
    return asset_family_ids.to(device=base_env.device, dtype=torch.long)[idx]


def _load_key_indices(asset_idx):
    traj_path = pathlib.Path(asset_paths[asset_idx]).parent / "traj.pkl"
    try:
        with open(traj_path, "rb") as f:
            data = pkl.load(f)
        key_indices = list(data.get("key_indices", []))
        direction = str(data.get("opening_direction", "pull"))
        return key_indices, direction
    except Exception:
        return [], "pull"


def _frame_to_phase(frame_idx, key_indices, direction):
    if not key_indices:
        return "?"
    names = PHASE_NAMES_PUSH if direction == "push" else PHASE_NAMES_PULL
    seg = 0
    for i, boundary in enumerate(key_indices):
        if frame_idx >= boundary:
            seg = i
    label = names[seg] if seg < len(names) else f"seg{seg}"
    return f"{label}"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    dagger_runtime_cfg = _load_student_dagger_defaults(
        args_cli.student_cfg if args_cli.student_cfg else str(DEFAULT_STUDENT_CFG)
    )
    if "reset_progress_total" in dagger_runtime_cfg:
        env_cfg.reset_progress_total = dagger_runtime_cfg["reset_progress_total"]
    if "adr_reset_progress_total" in dagger_runtime_cfg:
        env_cfg.adr_reset_progress_total = dagger_runtime_cfg["adr_reset_progress_total"]
    else:
        env_cfg.adr_reset_progress_total = 0.5 * float(env_cfg.reset_progress_total)
    env_cfg.pointcloud_render_mode = "none"
    env_cfg.enable_pointcloud_camera = False

    device = torch.device(agent_cfg["params"]["config"]["device"])

    preconvert_shared_urdf_assets(door_configs=MULTI_DOOR_CONFIGS, verbose=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped
    base_env.ref_motion_lib.reset_from_start = bool(args_cli.reset_from_start)
    base_env.early_stopping = bool(args_cli.early_stopping)
    # Force the drift-threshold curriculum (None => use the real schedule). Default 1.0 = loosest.
    base_env.drift_threshold_progress_override = (
        None if float(args_cli.threshold_progress) < 0.0 else float(args_cli.threshold_progress)
    )
    print(
        f"[INFO] Diagnosis rollout mode: reset_from_start={base_env.ref_motion_lib.reset_from_start} "
        f"early_stopping(drift-kill)={base_env.early_stopping} "
        f"drift_threshold_progress={base_env.drift_threshold_progress_override}"
    )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "rl_games", "teacher_diagnose", "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env_family_ids = _get_env_family_ids(base_env).to(device=device)
    if hasattr(base_env, "env_asset_indices"):
        env_asset_indices = base_env.env_asset_indices.to(device=device, dtype=torch.long)
    else:
        env_asset_indices = torch.arange(base_env.num_envs, device=device, dtype=torch.long) % len(asset_paths)

    # Per-asset traj metadata for phase naming.
    key_indices_by_asset = {}
    direction_by_asset = {}
    for asset_idx in sorted(set(env_asset_indices.detach().cpu().tolist())):
        key_indices_by_asset[asset_idx], direction_by_asset[asset_idx] = _load_key_indices(asset_idx)

    teacher_models, teacher_clip_obs = _build_teacher_models(agent_cfg, base_env, device)

    success_frame_idx = float(base_env.success_frame_idx)
    num_envs = base_env.num_envs
    obs, _ = env.reset()
    prev_actions = base_env.applied_robot_dof_targets.detach().clone().to(device=device)

    # Per-env running trackers (reset when an env resets).
    reached_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    max_hinge = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # Per-asset outcome accumulators.
    outcomes_by_asset = defaultdict(Counter)                # asset_idx -> {reason: count}
    progress_at_fail = defaultdict(list)                    # asset_idx -> [frame_idx reached]
    hinge_at_fail = defaultdict(list)                        # asset_idx -> [door angle reached]
    phase_at_fail = defaultdict(Counter)                     # asset_idx -> {phase: count}
    episodes_done = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _asset_label(asset_idx):
        p = pathlib.Path(asset_paths[asset_idx])
        fam = DOOR_FAMILY_NAMES[int(asset_family_ids[asset_idx].item())]
        return f"{fam}/{p.parent.name}"

    def _print_summary(step):
        print(f"\n===== teacher diagnosis @ step {step} =====")
        for asset_idx in sorted(outcomes_by_asset):
            c = outcomes_by_asset[asset_idx]
            total = sum(c.values())
            if total == 0:
                continue
            succ = c.get("success", 0)
            parts = [f"{k}={v}" for k, v in c.most_common()]
            frames = progress_at_fail[asset_idx]
            hinges = hinge_at_fail[asset_idx]
            med_frame = sorted(frames)[len(frames) // 2] if frames else float("nan")
            med_hinge = sorted(hinges)[len(hinges) // 2] if hinges else float("nan")
            phase_hist = ", ".join(f"{k}:{v}" for k, v in phase_at_fail[asset_idx].most_common())
            print(
                f"  {_asset_label(asset_idx)}: n={total} success={succ / total:.2f} "
                f"[{', '.join(parts)}] | fail@ median_frame={med_frame:.0f}/{success_frame_idx:.0f} "
                f"median_door_angle={med_hinge:.2f}rad"
                + (f" | fail_phases: {phase_hist}" if phase_hist else "")
            )

    def _print_trace(step, env_id, frame_idx_value, killed, kill_reason):
        extras = base_env.extras
        if "diag/door_joint_err_per_joint" not in extras:
            return
        # Thresholds are stored SQUARED (m^2 / rad^2); take sqrt for human-readable m / rad.
        pos_thresh = float(extras.get("fail/reset_key_body_pos_delta", float("nan"))) ** 0.5
        quat_thresh = float(extras.get("fail/reset_key_body_quat_delta", float("nan"))) ** 0.5
        door_thresh = float(extras.get("fail/reset_door_joint_pos_delta", float("nan"))) ** 0.5

        # ---- Door: actual vs desired per joint ----
        door_actual = extras["diag/door_joint_pos"][env_id].detach().cpu()
        door_ref = extras["diag/ref_door_joint_pos"][env_id].detach().cpu()
        door_err = extras["diag/door_joint_err_per_joint"][env_id].detach().cpu()
        door_parts = []
        for j in range(door_actual.numel()):
            name = DOOR_JOINT_NAMES[j] if j < len(DOOR_JOINT_NAMES) else f"door_j{j}"
            door_parts.append(
                f"{name}: actual={door_actual[j]:+.3f} desired={door_ref[j]:+.3f} "
                f"err={door_err[j].sqrt():.3f}rad"
            )

        # ---- Robot: worst-drifting key body (pos) actual vs desired ----
        pos_names = extras.get("diag/reset_key_body_names", [])
        pos_err = extras["diag/key_body_pos_err_per_body"][env_id].detach().cpu()
        wi = int(pos_err.argmax().item())
        actual_p = extras["diag/key_body_pos"][env_id, wi].detach().cpu()
        desired_p = extras["diag/ref_key_body_pos"][env_id, wi].detach().cpu()
        wname = pos_names[wi] if wi < len(pos_names) else f"body{wi}"
        robot_line = (
            f"worst_pos_body={wname} err={pos_err[wi].sqrt():.3f}m (thr={pos_thresh:.3f}m) "
            f"actual=[{actual_p[0]:+.3f},{actual_p[1]:+.3f},{actual_p[2]:+.3f}] "
            f"desired=[{desired_p[0]:+.3f},{desired_p[1]:+.3f},{desired_p[2]:+.3f}]"
        )

        # ---- Robot: worst-drifting key body (orientation) ----
        quat_names = extras.get("diag/key_body_names", [])
        quat_err = extras["diag/key_body_quat_err_per_body"][env_id].detach().cpu()
        qi = int(quat_err.argmax().item())
        qname = quat_names[qi] if qi < len(quat_names) else f"body{qi}"

        flag = f"  <<< KILL:{kill_reason}" if killed else ""
        print(
            f"[trace env{env_id} step{step} frame={frame_idx_value:.0f}/{success_frame_idx:.0f}] "
            f"DOOR {{ {' | '.join(door_parts)} }} (door_thr={door_thresh:.3f}rad) || "
            f"ROBOT {robot_line} | worst_quat_body={qname} err={quat_err[qi].sqrt():.3f}rad "
            f"(thr={quat_thresh:.3f}rad){flag}"
        )

    timestep = 0
    target_total_episodes = int(args_cli.episodes_per_env) * num_envs
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # Cache pre-step progress/hinge; on a kill this step, these describe how far it got.
            frame_idx_now = base_env.ref_motion_lib.frame_idx.to(device=device, dtype=torch.float32)
            hinge_now = base_env.door_joint_pos[:, 0].to(device=device, dtype=torch.float32)
            max_hinge = torch.maximum(max_hinge, hinge_now)
            reached_success |= frame_idx_now >= success_frame_idx

            actions = torch.zeros((num_envs, int(base_env.cfg.action_space)), device=device)
            for family_id in range(len(DOOR_FAMILY_NAMES)):
                env_ids = torch.nonzero(env_family_ids == int(family_id), as_tuple=False).squeeze(-1)
                if env_ids.numel() == 0:
                    continue
                batch = {
                    "is_train": False,
                    "obs": _clip_teacher_obs(obs[args_cli.teacher_obs_type][env_ids], teacher_clip_obs),
                    "prev_actions": prev_actions[env_ids],
                }
                res = teacher_models[family_id](batch)
                actions[env_ids] = torch.clamp(res["mus"], -1.0, 1.0)

            obs, _, terminated, truncated, _info = env.step(actions)
            terminated = terminated.to(device=device)
            truncated = truncated.to(device=device)
            dones = terminated | truncated
            prev_actions = base_env.applied_robot_dof_targets.detach().clone().to(device=device)

            # Per-env kill-reason masks the env stashed in extras during _get_dones (pre-reset).
            # Read from base_env.extras directly so gym wrappers can't strip them from the info dict.
            extras = base_env.extras
            robot_drift = extras.get("fail/robot_drift")
            door_drift = extras.get("fail/door_drift")
            timeout_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            if hasattr(base_env, "reset_time_outs"):
                timeout_success = base_env.reset_time_outs.to(device=device, dtype=torch.bool)

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            for env_id in done_ids.detach().cpu().tolist():
                asset_idx = int(env_asset_indices[env_id].item())
                is_success = bool(reached_success[env_id].item()) or bool(timeout_success[env_id].item())
                if is_success:
                    reason = "success"
                elif bool(terminated[env_id].item()):
                    r_drift = bool(robot_drift[env_id].item()) if robot_drift is not None else False
                    d_drift = bool(door_drift[env_id].item()) if door_drift is not None else False
                    if d_drift and not r_drift:
                        reason = "kill:door_drift"
                    elif r_drift and not d_drift:
                        reason = "kill:robot_drift"
                    elif r_drift and d_drift:
                        reason = "kill:both_drift"
                    else:
                        reason = "kill:other"
                else:
                    reason = "timeout"

                outcomes_by_asset[asset_idx][reason] += 1
                episodes_done[env_id] += 1
                if reason != "success":
                    reached_frame = float(frame_idx_now[env_id].item())
                    progress_at_fail[asset_idx].append(reached_frame)
                    hinge_at_fail[asset_idx].append(float(max_hinge[env_id].item()))
                    phase = _frame_to_phase(
                        reached_frame, key_indices_by_asset.get(asset_idx, []), direction_by_asset.get(asset_idx, "pull")
                    )
                    phase_at_fail[asset_idx][phase] += 1

            # Per-step drift trace for one env (actual vs desired + error), incl. the killing step.
            te = int(args_cli.trace_env)
            if te >= 0 and te < num_envs and (timestep % max(1, int(args_cli.trace_interval)) == 0 or bool(dones[te].item())):
                killed = bool(terminated[te].item())
                r_d = bool(robot_drift[te].item()) if robot_drift is not None else False
                d_d = bool(door_drift[te].item()) if door_drift is not None else False
                kill_reason = "robot_drift" if (r_d and not d_d) else "door_drift" if (d_d and not r_d) else "both" if killed else "-"
                _print_trace(timestep, te, float(frame_idx_now[te].item()), killed, kill_reason)

            # Reset per-env trackers for envs that just finished.
            if done_ids.numel() > 0:
                reached_success[done_ids] = False
                max_hinge[done_ids] = 0.0

        timestep += 1
        if args_cli.print_interval > 0 and timestep % args_cli.print_interval == 0:
            _print_summary(timestep)

        total_eps = int(episodes_done.sum().item())
        if target_total_episodes > 0 and total_eps >= target_total_episodes:
            print(f"[INFO] Reached episode budget ({total_eps} >= {target_total_episodes}).")
            break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            print(f"[INFO] Reached step cap ({timestep}).")
            break
        if args_cli.video and timestep >= args_cli.video_length:
            break
        _ = time.time() - start_time

    _print_summary(timestep)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
