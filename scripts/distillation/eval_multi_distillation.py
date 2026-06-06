"""Evaluate a distilled multi-door point-cloud policy.

The runner evaluates each vectorized env for several repeated rollouts. Envs
whose pose or door joint drifts too far from the reference motion are frozen at
the failure state; envs that reach the end of the reference motion are counted
as timed out and frozen there.
"""

import argparse
import os
import pathlib
import sys
import time
import types
from collections import OrderedDict

import yaml
from isaaclab.app import AppLauncher


SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_CFG = (
    SCRIPT_ROOT
    / "source"
    / "DoorOpening"
    / "tasks"
    / "dooropening"
    / "agents"
    / "pcd_transformer_dagger_cfg.yaml"
)


def _resolve_repo_path(path_value, default_path=None):
    if path_value is None:
        return None if default_path is None else str(default_path)
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    repo_path = SCRIPT_ROOT / path
    if repo_path.exists():
        return str(repo_path)
    return str(path)


def _resolve_checkpoint(path_value):
    if path_value is None:
        return None
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    repo_path = SCRIPT_ROOT / path
    if repo_path.exists():
        return str(repo_path)
    pretrained_path = SCRIPT_ROOT / "pretrained_ckpts" / path
    if pretrained_path.exists():
        return str(pretrained_path)
    return str(repo_path)


def _resolve_teacher_cfg_path(path_value):
    default_path = (
        SCRIPT_ROOT
        / "source"
        / "DoorOpening"
        / "tasks"
        / "dooropening"
        / "agents"
        / "rl_games_ppo_cfg.yaml"
    )
    return _resolve_repo_path(path_value, default_path)


def _load_student_dagger_defaults(student_cfg_path):
    if not student_cfg_path or not os.path.exists(student_cfg_path):
        return {}
    with open(student_cfg_path, "r", encoding="utf-8") as f:
        student_cfg = yaml.safe_load(f) or {}
    if not isinstance(student_cfg, dict):
        return {}
    dagger_cfg = student_cfg.get("dagger", {})
    return dict(dagger_cfg) if isinstance(dagger_cfg, dict) else {}


def _normalize_family_selection(family_spec):
    if family_spec is None:
        return None
    if isinstance(family_spec, str):
        family_names = [name.strip() for name in family_spec.split(",") if name.strip()]
    elif isinstance(family_spec, (list, tuple)):
        family_names = [str(name).strip() for name in family_spec if str(name).strip()]
    else:
        raise TypeError(f"Unsupported door family selection type: {type(family_spec)!r}")
    return family_names or None


def _resolve_multi_teacher_checkpoints():
    cli_values = {
        "PartNetv5": args_cli.teacher_partnetv5,
        "PartNetv5_plus": args_cli.teacher_partnetv5,
        "PartNetv5_plusplus": args_cli.teacher_partnetv5,
        "PartNetv9": args_cli.teacher_partnetv5,
        "PartNetv6": args_cli.teacher_partnetv6,
        "PartNetv6_plus": args_cli.teacher_partnetv6,
        "PartNetv6_plusplus": args_cli.teacher_partnetv6,
        "PartNetv10": args_cli.teacher_partnetv6,
        "PartNetv7": args_cli.teacher_partnetv7,
        "PartNetv7_plus": args_cli.teacher_partnetv7,
        "PartNetv7_plusplus": args_cli.teacher_partnetv7,
        "PartNetv11": args_cli.teacher_partnetv7,
        "PartNetv8": args_cli.teacher_partnetv8,
        "PartNetv8_plus": args_cli.teacher_partnetv8,
        "PartNetv8_plusplus": args_cli.teacher_partnetv8,
        "PartNetv12": args_cli.teacher_partnetv8,
    }
    any_family_cli = any(value is not None for value in cli_values.values())
    discovered_values = {}
    for family_name in DOOR_FAMILY_NAMES:
        default_family_ckpt = SCRIPT_ROOT / "source" / "DoorOpening" / "assets" / "door" / family_name / "door_opening.pth"
        configured_value = cli_values.get(family_name)
        if configured_value is not None:
            discovered_values[family_name] = _resolve_checkpoint(configured_value)
        elif default_family_ckpt.exists():
            discovered_values[family_name] = str(default_family_ckpt)

    if any_family_cli:
        missing = [family_name for family_name in DOOR_FAMILY_NAMES if family_name not in discovered_values]
        if missing:
            raise ValueError(
                "Multi-teacher evaluation needs a checkpoint for every PartNet family. "
                f"Missing: {missing}."
            )
        return discovered_values, None

    if len(discovered_values) == len(DOOR_FAMILY_NAMES):
        return discovered_values, None

    return None, _resolve_checkpoint(args_cli.teacher)


parser = argparse.ArgumentParser(description="Evaluate a distilled DooropeningMulti point-cloud policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument("--viser", action="store_true", default=False, help="Save raw Viser `.pt` replays for 8 random eval envs.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in env steps.")
parser.add_argument("--video_folder", type=str, default=None, help="Optional folder for recorded videos.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to evaluate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--student_cfg", type=str, default=None, help="Student config YAML to use.")
parser.add_argument("--student_ckpt", type=str, default=None, help="Student checkpoint to evaluate.")
parser.add_argument("--use_teacher", action="store_true", default=False, help="Roll out teacher actions instead of student actions.")
parser.add_argument("--teacher", type=str, default=None, help="Single teacher checkpoint to evaluate.")
parser.add_argument("--teacher-partnetv5", "--teacher_partnetv5", dest="teacher_partnetv5", type=str, default=None, help="Teacher checkpoint for PartNetv5.")
parser.add_argument("--teacher-partnetv6", "--teacher_partnetv6", dest="teacher_partnetv6", type=str, default=None, help="Teacher checkpoint for PartNetv6.")
parser.add_argument("--teacher-partnetv7", "--teacher_partnetv7", dest="teacher_partnetv7", type=str, default=None, help="Teacher checkpoint for PartNetv7.")
parser.add_argument("--teacher-partnetv8", "--teacher_partnetv8", dest="teacher_partnetv8", type=str, default=None, help="Teacher checkpoint for PartNetv8.")
parser.add_argument("--teacher_cfg", type=str, default=None, help="Teacher RL-Games config YAML to use.")
parser.add_argument(
    "--door-families",
    "--door_families",
    dest="door_families",
    type=str,
    default=None,
    help="Comma-separated multi-door family list to evaluate, e.g. PartNetv9,PartNetv10.",
)
parser.add_argument("--num_eval_runs", type=int, default=3, help="Number of repeated eval rollouts to run.")
parser.add_argument(
    "--pointcloud_source",
    type=str,
    choices=["sampler", "depth", "lidar"],
    default=None,
    help="Point-cloud source. Defaults to dagger.pointcloud_source in the student YAML.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Hard stop after this many env steps. Use 0 to run until every env freezes.",
)
parser.add_argument("--print_interval", type=int, default=15, help="Print rollout stats every N env steps.")
parser.add_argument(
    "--key_body_pos_threshold",
    type=float,
    default=1.2,
    help="Max robot key-body position drift limit in meters. Defaults to env reset_key_body_pos_delta_max.",
)
parser.add_argument(
    "--key_body_quat_threshold",
    type=float,
    default=None,
    help="Max robot key-body orientation drift limit in radians. Defaults to env reset_key_body_quat_delta_max.",
)
parser.add_argument(
    "--door_joint_pos_threshold",
    type=float,
    default=1.2,
    help="Max door joint drift limit in radians/meters. Defaults to env reset_door_joint_pos_delta_max.",
)
parser.add_argument(
    "--asset-preconvert-timeout-s",
    type=float,
    default=21600.0,
    help="Timeout for shared URDF preconversion.",
)
parser.add_argument(
    "--asset-preconvert-poll-interval-s",
    type=float,
    default=5.0,
    help="Polling interval while waiting for shared URDF preconversion.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

student_cfg_path = _resolve_repo_path(args_cli.student_cfg, DEFAULT_STUDENT_CFG)
student_dagger_defaults = _load_student_dagger_defaults(student_cfg_path)
selected_door_families = _normalize_family_selection(
    args_cli.door_families if args_cli.door_families is not None else student_dagger_defaults.get("door_families")
)
if selected_door_families is not None:
    os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = ",".join(selected_door_families)
    print(f"[INFO] Using multi-door families: {selected_door_families}")
pointcloud_source = str(student_dagger_defaults.get("pointcloud_source", "sampler")).lower()
if args_cli.pointcloud_source is not None:
    pointcloud_source = args_cli.pointcloud_source
if args_cli.video or pointcloud_source == "depth":
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below runs after Isaac Sim is launched."""

import gymnasium as gym
import torch

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import DoorOpening.tasks  # noqa: F401
from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets
from DoorOpening.assets.door.multi_door_cfg import ALL_DOOR_CONFIGS as MULTI_DOOR_CONFIGS
from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES, asset_family_ids, asset_paths
from DoorOpening.distillation.multi_pcd_dagger import Dagger
from DoorOpening.tasks.dooropening.multi_dooropening_env import compute_tracking_error
from DoorOpening.utils.quat_utils import quat_diff_angle


DIRECTION_NAME_TO_ID = {"pull": 0, "push": 1}
DIRECTION_ID_TO_NAME = {direction_id: name for name, direction_id in DIRECTION_NAME_TO_ID.items()}
MODE_CONTACT_SENSOR_NAME = "contact_forces_door2"
MODE_CONTACT_FORCE_THRESHOLD = 5.0


def _get_base_env(env):
    return getattr(env, "unwrapped", getattr(env, "env", env))


def _patch_no_auto_resets(base_env):
    if getattr(base_env, "_eval_no_auto_reset_patch", False):
        return

    def _get_dones_without_eval_resets(self):
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros

    base_env._get_dones = types.MethodType(_get_dones_without_eval_resets, base_env)
    base_env._eval_no_auto_reset_patch = True


class FrozenEnvState:
    """Stores and restores the state of envs that finished evaluation."""

    def __init__(self, base_env):
        self.base_env = base_env
        self.device = base_env.device
        self.mask = torch.zeros(base_env.num_envs, dtype=torch.bool, device=self.device)
        self.reset_from_env()

    def reset_from_env(self):
        base_env = self.base_env
        self.mask.zero_()
        self.robot_root_pose = base_env.robot.data.root_state_w[:, :7].detach().clone()
        self.robot_root_vel = torch.zeros_like(base_env.robot.data.root_state_w[:, 7:])
        self.robot_joint_pos = base_env.robot.data.joint_pos.detach().clone()
        self.robot_joint_vel = torch.zeros_like(base_env.robot.data.joint_vel)
        self.door_root_pose = base_env.door.data.root_state_w[:, :7].detach().clone()
        self.door_root_vel = torch.zeros_like(base_env.door.data.root_state_w[:, 7:])
        self.door_joint_pos = base_env.door.data.joint_pos.detach().clone()
        self.door_joint_vel = torch.zeros_like(base_env.door.data.joint_vel)
        self.robot_dof_targets = base_env.robot_dof_targets.detach().clone()
        self.applied_robot_dof_targets = base_env.applied_robot_dof_targets.detach().clone()
        ref_motion_lib = getattr(base_env, "ref_motion_lib", None)
        self.ref_frame_idx = None if ref_motion_lib is None else ref_motion_lib.frame_idx.detach().clone()

    def _as_env_ids(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.nonzero(self.mask, as_tuple=False).squeeze(-1)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids

    def capture(self, env_ids):
        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        base_env = self.base_env
        self.mask[env_ids] = True
        self.robot_root_pose[env_ids] = base_env.robot.data.root_state_w[env_ids, :7].detach()
        self.robot_root_vel[env_ids] = 0.0
        self.robot_joint_pos[env_ids] = base_env.robot.data.joint_pos[env_ids].detach()
        self.robot_joint_vel[env_ids] = 0.0
        self.door_root_pose[env_ids] = base_env.door.data.root_state_w[env_ids, :7].detach()
        self.door_root_vel[env_ids] = 0.0
        self.door_joint_pos[env_ids] = base_env.door.data.joint_pos[env_ids].detach()
        self.door_joint_vel[env_ids] = 0.0
        self.robot_dof_targets[env_ids] = base_env.robot_dof_targets[env_ids].detach()
        self.applied_robot_dof_targets[env_ids] = base_env.applied_robot_dof_targets[env_ids].detach()
        ref_motion_lib = getattr(base_env, "ref_motion_lib", None)
        if ref_motion_lib is not None and self.ref_frame_idx is not None:
            self.ref_frame_idx[env_ids] = ref_motion_lib.frame_idx[env_ids].detach()
        self.restore(env_ids)

    def restore(self, env_ids=None):
        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        base_env = self.base_env
        base_env.robot.write_root_pose_to_sim(self.robot_root_pose[env_ids], env_ids)
        base_env.robot.write_root_velocity_to_sim(self.robot_root_vel[env_ids], env_ids)
        base_env.robot.write_joint_state_to_sim(
            self.robot_joint_pos[env_ids],
            self.robot_joint_vel[env_ids],
            None,
            env_ids,
        )
        base_env.door.write_root_pose_to_sim(self.door_root_pose[env_ids], env_ids)
        base_env.door.write_root_velocity_to_sim(self.door_root_vel[env_ids], env_ids)
        base_env.door.write_joint_state_to_sim(
            self.door_joint_pos[env_ids],
            self.door_joint_vel[env_ids],
            None,
            env_ids,
        )
        base_env.robot_dof_targets[env_ids] = self.robot_dof_targets[env_ids]
        base_env.applied_robot_dof_targets[env_ids] = self.applied_robot_dof_targets[env_ids]
        base_env.robot.set_joint_position_target(
            self.applied_robot_dof_targets[env_ids],
            joint_ids=base_env._robot_dof_idx,
            env_ids=env_ids,
        )
        ref_motion_lib = getattr(base_env, "ref_motion_lib", None)
        if ref_motion_lib is not None and self.ref_frame_idx is not None:
            ref_motion_lib.frame_idx[env_ids] = self.ref_frame_idx[env_ids]
            ref_motion_lib._update_current()


def _install_freeze_patch(base_env, frozen_state):
    if getattr(base_env, "_eval_freeze_patch", False):
        return

    original_pre_physics_step = base_env._pre_physics_step
    original_apply_action = base_env._apply_action

    def _pre_physics_step_with_freeze(self, actions):
        if torch.any(frozen_state.mask):
            actions = actions.clone()
            actions[frozen_state.mask.to(device=actions.device)] = 0.0
        original_pre_physics_step(actions)
        frozen_state.restore()

    def _apply_action_with_freeze(self):
        original_apply_action()
        frozen_state.restore()

    base_env._pre_physics_step = types.MethodType(_pre_physics_step_with_freeze, base_env)
    base_env._apply_action = types.MethodType(_apply_action_with_freeze, base_env)
    base_env._eval_freeze_patch = True


def _resolve_drift_thresholds(base_env):
    return {
        "key_body_pos": float(
            args_cli.key_body_pos_threshold
            if args_cli.key_body_pos_threshold is not None
            else base_env.cfg.reset_key_body_pos_delta_max
        ),
        "key_body_quat": float(
            args_cli.key_body_quat_threshold
            if args_cli.key_body_quat_threshold is not None
            else base_env.cfg.reset_key_body_quat_delta_max
        ),
        "door_joint_pos": float(
            args_cli.door_joint_pos_threshold
            if args_cli.door_joint_pos_threshold is not None
            else base_env.cfg.reset_door_joint_pos_delta_max
        ),
    }


def _compute_drift(base_env, thresholds):
    base_env._get_intermediate_values()

    key_body_pos_err_sq, key_body_quat_err_sq, door_joint_pos_err_sq, *_ = compute_tracking_error(
        robot_key_body_pos=base_env.robot_reset_key_body_pos,
        robot_key_body_quat=base_env.robot_key_body_quat,
        door_joint_pos=base_env.door_joint_pos,
        robot_base_joint_pos=base_env.robot_base_joint_pos,
        robot_arm_joint_pos=base_env.robot_arm_joint_pos,
        robot_finger_joint_pos=base_env.robot_finger_joint_pos,
        robot_base_joint_vel=base_env.robot_base_joint_vel,
        robot_arm_joint_vel=base_env.robot_arm_joint_vel,
        robot_finger_joint_vel=base_env.robot_finger_joint_vel,
        ref_robot_key_body_pos=base_env.ref_robot_reset_key_body_pos,
        ref_robot_key_body_quat=base_env.ref_robot_key_body_quat,
        ref_door_joint_pos=base_env.ref_door_joint_pos,
        ref_robot_base_joint_pos=base_env.ref_robot_base_joint_pos,
        ref_robot_arm_joint_pos=base_env.ref_robot_arm_joint_pos,
        ref_robot_finger_joint_pos=base_env.ref_robot_finger_joint_pos,
        ref_robot_base_joint_vel=base_env.ref_robot_base_joint_vel,
        ref_robot_arm_joint_vel=base_env.ref_robot_arm_joint_vel,
        ref_robot_finger_joint_vel=base_env.ref_robot_finger_joint_vel,
    )

    threshold_sq = {name: float(value) * float(value) for name, value in thresholds.items()}

    # Keep printed diagnostics in human-readable units while matching the env's
    # squared-error termination criterion exactly.
    key_body_pos_err = torch.sqrt(torch.clamp_min(key_body_pos_err_sq, 0.0))
    key_body_quat_err = torch.sqrt(torch.clamp_min(key_body_quat_err_sq, 0.0))
    door_joint_pos_err = torch.sqrt(torch.clamp_min(door_joint_pos_err_sq, 0.0))

    metrics = {
        "key_body_pos": key_body_pos_err,
        "key_body_quat": key_body_quat_err,
        "door_joint_pos": door_joint_pos_err,
    }
    drifted = (
        (key_body_pos_err_sq > threshold_sq["key_body_pos"])
        | (key_body_quat_err_sq > threshold_sq["key_body_quat"])
        | (door_joint_pos_err_sq > threshold_sq["door_joint_pos"])
    )
    return drifted, metrics


def _get_env_family_ids(base_env):
    env_asset_indices = getattr(base_env, "env_asset_indices", None)
    if env_asset_indices is None:
        env_asset_indices = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long) % len(asset_paths)
    return asset_family_ids.to(device=base_env.device, dtype=torch.long)[env_asset_indices]


def _print_family_summary(base_env):
    env_family_ids = _get_env_family_ids(base_env)
    counts = {}
    for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
        counts[family_name] = int((env_family_ids == family_id).sum().detach().cpu().item())
    print("[INFO] Door family env counts:", counts)


def _empty_mode_stats():
    return {
        "count": 0,
        "dir_correct": 0,
        "dir_conf_sum": 0.0,
        "dir_pre_count": 0,
        "dir_pre_correct": 0,
        "dir_pre_conf_sum": 0.0,
        "dir_post_count": 0,
        "dir_post_correct": 0,
        "dir_post_conf_sum": 0.0,
    }


def _safe_ratio(numerator, denominator):
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _safe_mean(total, count):
    if count <= 0:
        return None
    return float(total) / float(count)


def _format_optional_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _get_measured_mode_contact_mask(dagger):
    contact_force_w = dagger._get_contact_sensor_force_tensor_world(MODE_CONTACT_SENSOR_NAME)
    contact_force_norm = torch.linalg.vector_norm(contact_force_w, dim=-1)
    return contact_force_norm > MODE_CONTACT_FORCE_THRESHOLD


class ModePredictionTracker:
    def __init__(self, device, num_envs, num_modes):
        self.device = device
        self.num_envs = int(num_envs)
        self.num_modes = int(num_modes)
        self.mode_short_names = [
            DIRECTION_ID_TO_NAME.get(mode_id, f"mode_{mode_id}") for mode_id in range(self.num_modes)
        ]
        self.reset_total()
        self.reset_run()

    def reset_total(self):
        self.total_stats = _empty_mode_stats()
        self.total_pred_counts = [0 for _ in range(self.num_modes)]

    def reset_run(self):
        self.contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.run_stats = _empty_mode_stats()
        self.run_pred_counts = [0 for _ in range(self.num_modes)]

    def print_legend(self):
        legend = ", ".join(f"{mode_id}:{label}" for mode_id, label in enumerate(self.mode_short_names))
        print("[INFO] Direction prediction labels:", legend)
        print(
            "[INFO] Mode eval reports the 2-way push/pull direction prediction continuously, "
            "split into pre-contact and post-contact phases. Periodic [MODE] logs are latest-step metrics; "
            "[MODE-RESULT] logs are cumulative."
        )

    def _accumulate_stats(
        self,
        stats,
        pred_counts,
        direction_pred,
        direction_conf,
        target,
        active,
        pre_contact,
        post_contact,
    ):
        target = target.to(device=self.device, dtype=torch.long)
        valid = active & (target >= 0) & (target < self.num_modes)
        valid_count = int(valid.sum().detach().cpu().item())
        if valid_count <= 0:
            return

        direction_correct = direction_pred == target

        stats["count"] += valid_count
        stats["dir_correct"] += int((direction_correct & valid).sum().detach().cpu().item())
        stats["dir_conf_sum"] += float(direction_conf[valid].sum().detach().cpu().item())

        direction_pre_mask = pre_contact & valid
        direction_post_mask = post_contact & valid
        direction_pre_count = int(direction_pre_mask.sum().detach().cpu().item())
        direction_post_count = int(direction_post_mask.sum().detach().cpu().item())
        stats["dir_pre_count"] += direction_pre_count
        stats["dir_post_count"] += direction_post_count
        if direction_pre_count > 0:
            stats["dir_pre_correct"] += int((direction_correct & direction_pre_mask).sum().detach().cpu().item())
            stats["dir_pre_conf_sum"] += float(direction_conf[direction_pre_mask].sum().detach().cpu().item())
        if direction_post_count > 0:
            stats["dir_post_correct"] += int((direction_correct & direction_post_mask).sum().detach().cpu().item())
            stats["dir_post_conf_sum"] += float(direction_conf[direction_post_mask].sum().detach().cpu().item())

        active_mode_preds = direction_pred[active]
        bincount = torch.bincount(active_mode_preds, minlength=self.num_modes).detach().cpu().tolist()
        for mode_id, count in enumerate(bincount):
            pred_counts[mode_id] += int(count)

    def update(self, mode_logits, target, active, current_contact):
        target = target.to(device=self.device, dtype=torch.long)
        active = active.to(device=self.device, dtype=torch.bool)
        current_contact = current_contact.to(device=self.device, dtype=torch.bool)
        if mode_logits.shape[-1] != self.num_modes:
            raise RuntimeError(
                f"Mode prediction tracker expected {self.num_modes} logits, got {mode_logits.shape[-1]}."
            )

        mode_probs = torch.softmax(mode_logits, dim=-1)
        direction_pred = mode_probs.argmax(dim=-1)
        direction_conf = mode_probs.amax(dim=-1)

        post_contact = active & (self.contact_seen | current_contact)
        pre_contact = active & ~post_contact
        self.contact_seen |= active & current_contact

        valid = active & (target >= 0) & (target < self.num_modes)
        direction_correct = direction_pred == target
        step_pre_mask = pre_contact & valid
        step_post_mask = post_contact & valid
        step_stats = _empty_mode_stats()
        step_stats["count"] = int(valid.sum().detach().cpu().item())
        step_stats["dir_correct"] = int((direction_correct & valid).sum().detach().cpu().item())
        step_stats["dir_conf_sum"] = float(direction_conf[valid].sum().detach().cpu().item())
        step_stats["dir_pre_count"] = int(step_pre_mask.sum().detach().cpu().item())
        step_stats["dir_post_count"] = int(step_post_mask.sum().detach().cpu().item())
        if step_stats["dir_pre_count"] > 0:
            step_stats["dir_pre_correct"] = int((direction_correct & step_pre_mask).sum().detach().cpu().item())
            step_stats["dir_pre_conf_sum"] = float(direction_conf[step_pre_mask].sum().detach().cpu().item())
        if step_stats["dir_post_count"] > 0:
            step_stats["dir_post_correct"] = int((direction_correct & step_post_mask).sum().detach().cpu().item())
            step_stats["dir_post_conf_sum"] = float(direction_conf[step_post_mask].sum().detach().cpu().item())

        self._accumulate_stats(
            self.run_stats,
            self.run_pred_counts,
            direction_pred,
            direction_conf,
            target,
            active,
            pre_contact,
            post_contact,
        )
        self._accumulate_stats(
            self.total_stats,
            self.total_pred_counts,
            direction_pred,
            direction_conf,
            target,
            active,
            pre_contact,
            post_contact,
        )

        step_pred_counts = [0 for _ in range(self.num_modes)]
        active_count = int(active.sum().detach().cpu().item())
        if active_count > 0:
            bincount = torch.bincount(direction_pred[active], minlength=self.num_modes).detach().cpu().tolist()
            step_pred_counts = [int(count) for count in bincount]

        return {
            "active_count": active_count,
            "valid_count": step_stats["count"],
            "pre_contact_count": step_stats["dir_pre_count"],
            "post_contact_count": step_stats["dir_post_count"],
            "step_stats": step_stats,
            "step_pred_counts": step_pred_counts,
        }

    def format_stats(self, stats):
        direction_acc = _safe_ratio(stats["dir_correct"], stats["count"])
        direction_pre_acc = _safe_ratio(stats["dir_pre_correct"], stats["dir_pre_count"])
        direction_post_acc = _safe_ratio(stats["dir_post_correct"], stats["dir_post_count"])
        direction_conf = _safe_mean(stats["dir_conf_sum"], stats["count"])
        direction_pre_conf = _safe_mean(stats["dir_pre_conf_sum"], stats["dir_pre_count"])
        direction_post_conf = _safe_mean(stats["dir_post_conf_sum"], stats["dir_post_count"])
        return (
            f"dir_acc={_format_optional_metric(direction_acc)} "
            f"dir_pre_acc={_format_optional_metric(direction_pre_acc)} "
            f"dir_post_acc={_format_optional_metric(direction_post_acc)} "
            f"dir_conf={_format_optional_metric(direction_conf)} "
            f"dir_pre_conf={_format_optional_metric(direction_pre_conf)} "
            f"dir_post_conf={_format_optional_metric(direction_post_conf)}"
        )

    def format_pred_counts(self, pred_counts):
        return ", ".join(
            f"{mode_name}={int(count)}" for mode_name, count in zip(self.mode_short_names, pred_counts)
        )


def _reset_dagger_rollout_state(dagger):
    dagger.frame = 0
    dagger.resume_iteration = 0
    dagger.latest_student_proprio_vector = None
    dagger.latest_aux_input_vector = None
    dagger.latest_aux_target_vector = None
    dagger._resample_wall_distractors()
    dagger.temporal_current_time_s = 0.0
    dagger._seed_temporal_histories()
    dagger._seed_aux_buffer()
    dagger._seed_push_pull_condition_buffer()


def _print_eval_curriculum_state(tag, dagger, base_env):
    common_step_counter = getattr(base_env, "common_step_counter", None)
    env_frames = getattr(base_env, "_rlgames_env_frames", None)
    adr_increment = None
    if hasattr(base_env, "dooropening_adr") and hasattr(base_env.dooropening_adr, "increment_counter"):
        adr_increment = int(base_env.dooropening_adr.increment_counter)
    adr_progress = None
    adr_reset_progress_total = getattr(base_env, "adr_reset_progress_total", None)
    if common_step_counter is not None and adr_reset_progress_total is not None:
        denom = max(float(adr_reset_progress_total), 1.0)
        adr_progress = min(float(common_step_counter) / denom, 1.0)
    print(
        "[INFO] "
        f"{tag}: dagger.resume_iteration={int(getattr(dagger, 'resume_iteration', 0))}, "
        f"dagger.frame={int(getattr(dagger, 'frame', 0))}, "
        f"env.common_step_counter={common_step_counter}, "
        f"env._rlgames_env_frames={env_frames}, "
        f"adr_increment={adr_increment}, "
        f"adr_progress={adr_progress}"
    )


def _force_max_adr_for_eval(base_env):
    adr_reset_progress_total = getattr(base_env, "adr_reset_progress_total", None)
    if adr_reset_progress_total is not None and hasattr(base_env, "common_step_counter"):
        target_step = max(float(adr_reset_progress_total), 1.0)
        target_step_int = int(target_step)
        if target_step_int < target_step:
            target_step_int += 1
        base_env.common_step_counter = target_step_int
    if hasattr(base_env, "dooropening_adr") and hasattr(base_env, "cfg"):
        num_adr_increments = getattr(base_env.cfg, "num_adr_increments", None)
        if num_adr_increments is not None:
            base_env.dooropening_adr.set_num_increments(int(num_adr_increments))
    if hasattr(base_env, "_rlgames_env_frames"):
        base_env._rlgames_env_frames = int(getattr(base_env, "common_step_counter", 0))


def _disable_observation_lag_for_eval(dagger):
    dagger.observation_lag_enabled = False
    dagger.observation_lag_apply_during_eval = False
    if hasattr(dagger, "_reset_observation_lag_stats"):
        dagger._reset_observation_lag_stats()


def _build_student_actions(dagger, iteration):
    student_obs = dagger._build_student_obs(iteration=iteration)
    student_output = dagger._student_forward(student_obs)
    if dagger.has_aux_prediction:
        aux_prediction = dagger._decode_aux_prediction(student_output["aux"].detach())
        dagger.aux_buffer[:] = aux_prediction
    student_actions = student_output["action"][:, 0, :]
    env_actions = dagger._student_actions_to_env_actions(student_actions)
    return env_actions, student_output


def _build_teacher_actions(dagger, obs):
    teacher_output = dagger._get_teacher_actions(obs)
    return teacher_output["actions"], None


def _configure_eval_viser_raw_streams(dagger, run_index, num_streams=8):
    if not getattr(dagger, "viser_raw_enabled", False):
        return

    num_streams = max(1, min(int(num_streams), int(dagger.num_envs)))
    selected_env_ids = torch.randperm(dagger.num_envs, device=dagger.device)[:num_streams].detach().cpu().tolist()
    streams = OrderedDict()
    selected_labels = []
    for env_id in selected_env_ids:
        family_id = int(dagger.env_family_ids[env_id].detach().cpu().item())
        family_name = DOOR_FAMILY_NAMES[family_id]
        asset_idx = int(dagger.env_asset_idx[env_id].detach().cpu().item())
        asset_name = pathlib.Path(asset_paths[asset_idx]).resolve().parent.name
        stream_name = f"run{int(run_index):02d}_{family_name}_env{env_id:03d}_{asset_name}"
        streams[stream_name] = {
            "family_id": family_id,
            "family_name": stream_name,
            "env_id": int(env_id),
            "frames": [],
            "frame_count": 0,
            "chunk_index": 0,
            "latest_iteration": None,
        }
        selected_labels.append(f"{env_id}:{asset_name}")
    dagger._viser_raw_streams = streams
    print("[INFO] Eval viser_raw envs:", ", ".join(selected_labels))


def _flush_finished_eval_viser_raw_streams(dagger, newly_finished):
    if not getattr(dagger, "viser_raw_enabled", False):
        return
    if newly_finished is None or newly_finished.numel() == 0:
        return
    if not getattr(dagger, "_viser_raw_streams", None):
        return
    finished_env_ids = set(int(env_id) for env_id in newly_finished.detach().cpu().tolist())
    completed_stream_keys = []
    for stream_name, stream in dagger._viser_raw_streams.items():
        if int(stream["env_id"]) not in finished_env_ids:
            continue
        dagger._flush_viser_raw_stream(
            stream,
            chunk_complete=True,
            reason=f"eval env {int(stream['env_id'])} finished",
        )
        completed_stream_keys.append(stream_name)
    for stream_name in completed_stream_keys:
        del dagger._viser_raw_streams[stream_name]


def _compute_eval_action_loss(dagger, obs, student_output, active_mask):
    if student_output is None or "action" not in student_output or not dagger._has_teacher():
        return None
    active_mask = active_mask.to(device=dagger.device, dtype=torch.bool)
    if not torch.any(active_mask):
        return None
    teacher_output = dagger._get_teacher_actions(obs)
    pred_actions = student_output["action"][active_mask]
    target_actions = teacher_output["mus"][active_mask].unsqueeze(1)
    return torch.nn.functional.mse_loss(pred_actions, target_actions)


def _print_progress(run_index, step, active, timed_out, drifted, metrics, action_loss=None):
    active_count = int(active.sum().detach().cpu().item())
    timeout_count = int(timed_out.sum().detach().cpu().item())
    drift_count = int(drifted.sum().detach().cpu().item())
    if torch.any(active):
        active_metrics = {
            name: float(values[active].max().detach().cpu().item())
            for name, values in metrics.items()
        }
    else:
        active_metrics = {name: 0.0 for name in metrics}
    action_loss_text = "n/a" if action_loss is None else f"{float(action_loss.detach().cpu().item()):.6f}"
    print(
        "[EVAL] "
        f"run={run_index} step={step} active={active_count} activate_rate{active_count / (active_count + drift_count)}"
        f"timed_out={timeout_count} drifted={drift_count} "
        f"max_key_pos={active_metrics['key_body_pos']:.4f} "
        f"max_key_quat={active_metrics['key_body_quat']:.4f} "
        f"max_door={active_metrics['door_joint_pos']:.4f} "
        f"action_loss={action_loss_text}"
    )


def _print_mode_progress(run_index, step, tracker, step_summary):
    if tracker is None or step_summary is None:
        return
    print(
        "[MODE] "
        f"run={run_index} step={step} "
        f"active={step_summary['active_count']} "
        f"eval_active={step_summary['valid_count']} "
        f"pre_contact={step_summary['pre_contact_count']} "
        f"post_contact={step_summary['post_contact_count']} "
        f"{tracker.format_stats(step_summary['step_stats'])} "
        f"preds={tracker.format_pred_counts(step_summary['step_pred_counts'])}"
    )


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.use_motion_ref = True

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg["params"]["seed"] = args_cli.seed

    dagger_runtime_cfg = dict(student_dagger_defaults)
    dagger_runtime_cfg.pop("wandb", None)
    dagger_runtime_cfg["pointcloud_source"] = pointcloud_source
    if "reset_progress_total" in dagger_runtime_cfg:
        env_cfg.reset_progress_total = dagger_runtime_cfg["reset_progress_total"]
    if "adr_reset_progress_total" in dagger_runtime_cfg:
        env_cfg.adr_reset_progress_total = dagger_runtime_cfg["adr_reset_progress_total"]
    else:
        env_cfg.adr_reset_progress_total = 0.5 * float(env_cfg.reset_progress_total)

    if pointcloud_source == "depth":
        env_cfg.pointcloud_render_mode = "depth"
    elif pointcloud_source == "lidar":
        env_cfg.pointcloud_render_mode = "lidar"
    else:
        env_cfg.pointcloud_render_mode = "none"
    env_cfg.enable_pointcloud_camera = env_cfg.pointcloud_render_mode == "depth"

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = os.path.join("runs", f"DoorOpening-Distillation-Eval_{timestamp}")
    nn_dir = os.path.join(experiment_dir, "nn")
    summaries_dir = os.path.join(experiment_dir, "summaries")
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    print(f"[INFO] Eval output directory: {experiment_dir}")
    print(f"[INFO] pointcloud_source={pointcloud_source}, render_mode={env_cfg.pointcloud_render_mode}")

    viser_cfg = dagger_runtime_cfg.get("viser", {})
    if not isinstance(viser_cfg, dict):
        viser_cfg = {}
    raw_cfg = viser_cfg.get("raw", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    raw_cfg["enabled"] = bool(args_cli.viser)
    raw_cfg.setdefault("path", "eval_viser_replay.pt")
    viser_cfg["raw"] = raw_cfg
    dagger_runtime_cfg["viser"] = viser_cfg

    preconvert_shared_urdf_assets(
        door_configs=MULTI_DOOR_CONFIGS,
        timeout_s=float(args_cli.asset_preconvert_timeout_s),
        poll_interval_s=float(args_cli.asset_preconvert_poll_interval_s),
        verbose=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = _get_base_env(env)
    if base_env.ref_motion_lib is None:
        raise RuntimeError("Eval requires env_cfg.use_motion_ref=True and a valid traj.pkl for each door asset.")
    base_env.ref_motion_lib.reset_from_start = True
    base_env.early_stopping = False
    _patch_no_auto_resets(base_env)
    _print_family_summary(base_env)

    if args_cli.video:
        video_folder = args_cli.video_folder or os.path.join(experiment_dir, "videos", "eval_multi_distillation")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "eval_multi_distillation",
            "disable_logger": True,
        }
        print("[INFO] Recording eval video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        base_env = _get_base_env(env)

    student_ckpt = _resolve_checkpoint(args_cli.student_ckpt)
    teacher_config = {}
    if args_cli.use_teacher:
        multi_teacher_ckpts, teacher_ckpt = _resolve_multi_teacher_checkpoints()
        if multi_teacher_ckpts is None and teacher_ckpt is None:
            raise ValueError("--use_teacher requires --teacher or per-family teacher checkpoints.")
        teacher_cfg_path = _resolve_teacher_cfg_path(args_cli.teacher_cfg)
        teacher_config = {"cfg": teacher_cfg_path, "obs_type": "policy"}
        if multi_teacher_ckpts is not None:
            teacher_config["teachers"] = {
                family_name: {"ckpt": ckpt}
                for family_name, ckpt in multi_teacher_ckpts.items()
            }
        else:
            teacher_config["ckpt"] = teacher_ckpt
        print("[INFO] Eval action source: teacher")
    else:
        if student_ckpt is None:
            raise ValueError("Student evaluation requires --student_ckpt, or use --use_teacher.")
        print("[INFO] Eval action source: student")
        multi_teacher_ckpts, teacher_ckpt = _resolve_multi_teacher_checkpoints()
        if multi_teacher_ckpts is not None or teacher_ckpt is not None:
            teacher_cfg_path = _resolve_teacher_cfg_path(args_cli.teacher_cfg)
            teacher_config = {"cfg": teacher_cfg_path, "obs_type": "policy"}
            if multi_teacher_ckpts is not None:
                teacher_config["teachers"] = {
                    family_name: {"ckpt": ckpt}
                    for family_name, ckpt in multi_teacher_ckpts.items()
                }
            else:
                teacher_config["ckpt"] = teacher_ckpt
            print("[INFO] Eval action-loss target source: teacher")
        else:
            print("[INFO] Eval action-loss target source: unavailable (teacher not configured)")

    dagger_config = {
        "student": {
            "cfg": student_cfg_path,
            "ckpt": student_ckpt,
        },
        "teacher": teacher_config,
        "play_policy": True,
        "dagger": dagger_runtime_cfg,
        "wandb": {"enabled": False},
    }
    dagger = Dagger(env, dagger_config, summaries_dir=summaries_dir, nn_dir=nn_dir)
    dagger.student_model_ddp.eval()
    if dagger._has_teacher():
        for teacher_model in dagger._iter_teacher_models():
            teacher_model.eval()
    print(f"[INFO] Observation lag for eval follows student YAML: enabled={bool(dagger.observation_lag_enabled)}")
    _print_eval_curriculum_state("Checkpoint-restored curriculum state", dagger, base_env)
    mode_direction_target = None
    if dagger.mode_prediction_enabled and not args_cli.use_teacher:
        if getattr(dagger, "mode_family_direction_ids", None) is None:
            raise RuntimeError("Mode prediction is enabled, but direction targets are not initialized.")
        mode_direction_target = dagger.mode_family_direction_ids[dagger.env_family_ids.long()]
        invalid_direction_target = (mode_direction_target < 0) | (mode_direction_target >= dagger.num_modes)
        if torch.any(invalid_direction_target):
            invalid_family_ids = sorted(
                set(dagger.env_family_ids[invalid_direction_target].detach().cpu().tolist())
            )
            raise RuntimeError(
                f"Student mode head exposes {dagger.num_modes} directions, but env families "
                f"{invalid_family_ids} map to invalid direction targets."
            )

    _force_max_adr_for_eval(base_env)
    _print_eval_curriculum_state("Eval curriculum state after forcing max ADR", dagger, base_env)

    frozen_state = FrozenEnvState(base_env)
    obs, _ = env.reset()
    _print_eval_curriculum_state("Eval curriculum state after first env.reset()", dagger, base_env)
    frozen_state.reset_from_env()
    _reset_dagger_rollout_state(dagger)
    _install_freeze_patch(base_env, frozen_state)

    thresholds = _resolve_drift_thresholds(base_env)
    print("[INFO] Drift thresholds:", thresholds)
    mode_tracker = None
    if dagger.mode_prediction_enabled and not args_cli.use_teacher:
        mode_tracker = ModePredictionTracker(base_env.device, base_env.num_envs, dagger.num_modes)
        mode_tracker.print_legend()
        print(
            "[INFO] Mode eval contact split uses measured filtered contact force: "
            f"sensor={MODE_CONTACT_SENSOR_NAME}, threshold={MODE_CONTACT_FORCE_THRESHOLD:.3f}"
        )

    total_steps = 0
    total_timeout_count = 0
    total_drift_count = 0
    total_active_count = 0
    completed_runs = 0
    num_eval_runs = max(1, int(args_cli.num_eval_runs))

    for run_index in range(1, num_eval_runs + 1):
        if not simulation_app.is_running():
            break
        if args_cli.viser:
            if run_index > 1:
                dagger._flush_viser_raw_recording(
                    chunk_complete=False,
                    reason=f"eval run {run_index - 1} ended before selected envs finished",
                )
            _configure_eval_viser_raw_streams(dagger, run_index=run_index, num_streams=8)
        if run_index > 1:
            obs, _ = env.reset()
            frozen_state.reset_from_env()
            _reset_dagger_rollout_state(dagger)
        if mode_tracker is not None:
            mode_tracker.reset_run()

        active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        timed_out = torch.zeros_like(active)
        drifted = torch.zeros_like(active)
        last_metrics = {
            "key_body_pos": torch.zeros(base_env.num_envs, dtype=torch.float32, device=base_env.device),
            "key_body_quat": torch.zeros(base_env.num_envs, dtype=torch.float32, device=base_env.device),
            "door_joint_pos": torch.zeros(base_env.num_envs, dtype=torch.float32, device=base_env.device),
        }
        last_action_loss = None

        step = 0
        while simulation_app.is_running():
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
            if not torch.any(active):
                break

            # Avoid torch.inference_mode() around Isaac env stepping. It can turn
            # simulator-owned tensors into inference tensors, then later in-place
            # simulator updates fail outside inference mode.
            with torch.no_grad():
                frozen_state.restore()
                if args_cli.use_teacher:
                    if dagger.viser_raw_enabled:
                        dagger._build_student_obs(iteration=step)
                    actions, student_output = _build_teacher_actions(dagger, obs)
                else:
                    actions, student_output = _build_student_actions(dagger, iteration=step)
                mode_step_summary = None
                mode_logits = None
                current_contact = None
                if mode_tracker is not None:
                    mode_logits = student_output.get("mode_logits")
                    if mode_logits is None:
                        raise RuntimeError(
                            "Mode prediction is enabled, but eval student output does not contain mode_logits."
                        )
                acting_mask = active.clone()
                last_action_loss = _compute_eval_action_loss(dagger, obs, student_output, acting_mask)
                actions[~active.to(device=actions.device)] = 0.0
                obs, _, _, _, _ = env.step(actions)
                frozen_state.restore()
                if mode_tracker is not None:
                    current_contact = _get_measured_mode_contact_mask(dagger)

                dagger.temporal_current_time_s = dagger._iteration_to_time_s(step + 1)
                q_after_step = dagger._get_student_proprio_vector().detach().clone()
                target_after_step = dagger._get_implemented_action_vector().detach().clone()
                base_vel_after_step = dagger._get_student_base_velocity_vector().detach().clone()
                dagger._push_temporal_history(
                    timestamp=dagger.temporal_current_time_s,
                    q=q_after_step,
                    target=target_after_step,
                    base_vel=base_vel_after_step,
                )

                if dagger.viser_raw_enabled and dagger._viser_pending_debug_frame is not None:
                    aux_prediction_for_replay = None
                    if (
                        student_output is not None
                        and dagger.has_aux_prediction
                        and isinstance(student_output, dict)
                        and "aux" in student_output
                    ):
                        aux_prediction_for_replay = dagger._decode_aux_prediction(student_output["aux"].detach())
                    dagger._maybe_update_viser_debug(
                        iteration=dagger._viser_pending_debug_frame["iteration"],
                        robot_base_pos_w=dagger._viser_pending_debug_frame["robot_base_pos_w"],
                        robot_base_quat_w=dagger._viser_pending_debug_frame["robot_base_quat_w"],
                        ground_truth_pcd_world=dagger._viser_pending_debug_frame["ground_truth_pcd_world"],
                        robot_obs_pcd_base=dagger._viser_pending_debug_frame["robot_obs_pcd_base"],
                        policy_input_pcd_base=dagger._viser_pending_debug_frame["policy_input_pcd_base"],
                        aux_prediction=aux_prediction_for_replay,
                    )
                    dagger._viser_pending_debug_frame = None

                reached_last_frame = base_env._get_reached_last_frame_mask()
                trial_timed_out = base_env.episode_length_buf >= (base_env.max_trial_steps - 1)
                drift_mask, last_metrics = _compute_drift(base_env, thresholds)
                new_timeouts = (reached_last_frame | trial_timed_out) & active
                new_drifts = drift_mask & active & ~new_timeouts
                newly_finished = new_timeouts | new_drifts
                if torch.any(newly_finished):
                    timed_out |= new_timeouts
                    drifted |= new_drifts
                    active &= ~newly_finished
                    frozen_state.capture(torch.nonzero(newly_finished, as_tuple=False).squeeze(-1))
                    _flush_finished_eval_viser_raw_streams(
                        dagger,
                        torch.nonzero(newly_finished, as_tuple=False).squeeze(-1),
                    )

                if mode_tracker is not None:
                    mode_step_summary = mode_tracker.update(
                        mode_logits=mode_logits.detach(),
                        target=mode_direction_target,
                        active=active,
                        current_contact=current_contact,
                    )

            step += 1
            if (
                args_cli.print_interval > 0
                and (step == 1 or step % args_cli.print_interval == 0 or not torch.any(active))
            ):
                _print_progress(run_index, step, active, timed_out, drifted, last_metrics, action_loss=last_action_loss)
                _print_mode_progress(run_index, step, mode_tracker, mode_step_summary)

        active_count = int(active.sum().detach().cpu().item())
        timeout_count = int(timed_out.sum().detach().cpu().item())
        drift_count = int(drifted.sum().detach().cpu().item())
        total_steps += step
        total_timeout_count += timeout_count
        total_drift_count += drift_count
        total_active_count += active_count
        completed_runs += 1
        print(
            "[RESULT] "
            f"run={run_index} steps={step} num_envs={base_env.num_envs} "
            f"timed_out_envs={timeout_count} drifted_envs={drift_count} active_envs={active_count}"
        )
        if mode_tracker is not None:
            print(
                "[MODE-RESULT] "
                f"run={run_index} {mode_tracker.format_stats(mode_tracker.run_stats)} "
                f"preds={mode_tracker.format_pred_counts(mode_tracker.run_pred_counts)}"
            )

    total_rollouts = completed_runs * int(base_env.num_envs)
    print(
        "[RESULT] "
        f"runs={completed_runs} requested_runs={num_eval_runs} "
        f"total_rollouts={total_rollouts} total_steps={total_steps} "
        f"timed_out_envs={total_timeout_count} drifted_envs={total_drift_count} "
        f"active_envs={total_active_count} "
        f"timeout_rate={total_timeout_count / max(1, total_rollouts):.4f} "
        f"drift_rate={total_drift_count / max(1, total_rollouts):.4f}"
    )
    if mode_tracker is not None:
        print(
            "[MODE-RESULT] "
            f"runs={completed_runs} {mode_tracker.format_stats(mode_tracker.total_stats)} "
            f"preds={mode_tracker.format_pred_counts(mode_tracker.total_pred_counts)}"
        )
    dagger._close_viser_debug_tools()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
