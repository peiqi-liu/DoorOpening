"""Evaluate a distilled multi-door point-cloud policy.

The runner evaluates each vectorized env for several repeated rollouts. Envs
whose pose or door joint drifts too far from the reference motion are frozen at
the failure state; envs that reach the end of the reference motion are counted
as timed out and frozen there.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import types

import yaml
from isaaclab.app import AppLauncher


SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_CFG = (
    SCRIPT_ROOT
    / "source"
    / "DoorOpening"
    / "distillation"
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
parser.add_argument(
    "--viser",
    action="store_true",
    default=False,
    help=(
        "Save replay_viser_pt.py-compatible .pt files (robot + door URDF meshes, joint angles, PD "
        "targets, actual sim door joints, policy-input points, and pointwise attribution colors) for randomly "
        "selected eval envs, one file per env, into <experiment_dir>/viser. Play any of them with "
        "scripts/replay_viser_pt.py to see the real URDF meshes."
    ),
)
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in env steps.")
parser.add_argument("--video_folder", type=str, default=None, help="Optional folder for recorded videos.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to evaluate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--panel-stiffness-range-nm-per-rad",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Eval-only full-ADR sampling range for door-panel stiffness in Nm/rad.",
)
parser.add_argument(
    "--panel-effort-limit-range-nm",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Eval-only full-ADR sampling range for the door-panel effort limit in Nm.",
)
parser.add_argument(
    "--handle-stiffness-range-nm-per-rad",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Eval-only full-ADR sampling range for handle joint_2 stiffness in Nm/rad.",
)
parser.add_argument(
    "--handle-effort-limit-range-nm",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Eval-only full-ADR sampling range for handle joint_2 effort limit in Nm.",
)
parser.add_argument(
    "--door-handle-limit-margin-range-rad",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Eval-only handle joint upper stop margin above each episode's unlatch threshold.",
)
parser.add_argument(
    "--door-solver-position-iterations",
    type=int,
    default=None,
    help="Eval-only PhysX position-solver iteration count for door articulation assets.",
)
parser.add_argument("--student_ckpt", type=str, default=None, help="Student checkpoint to evaluate.")
parser.add_argument(
    "--student_cfg",
    type=str,
    default=None,
    help="Student YAML config; defaults to the source distillation YAML.",
)
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
parser.add_argument(
    "--max-door-assets",
    type=int,
    default=0,
    help="Limit each selected door family to its first N sorted assets (0 keeps all assets).",
)
parser.add_argument(
    "--wall-raster-width-px",
    type=int,
    default=None,
    help="Override depth_cam_render.wall_raster_width_px without changing checkpoint model settings.",
)
parser.add_argument(
    "--wall-raster-height-px",
    type=int,
    default=None,
    help="Override depth_cam_render.wall_raster_height_px without changing checkpoint model settings.",
)
parser.add_argument(
    "--wall-face-jitter-m",
    type=float,
    default=None,
    help="Override wall_distractors.face_jitter_m without changing checkpoint model settings.",
)
parser.add_argument(
    "--wall-height-range-m",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Eval-only override for detached wall top heights in wall_distractors.height_range_m.",
)
parser.add_argument(
    "--flush-wall-height-range-m",
    type=float,
    nargs=2,
    default=None,
    metavar=("BOTTOM", "TOP"),
    help="Eval-only override for flush wall vertical bounds in wall_distractors.flush_height_range_m.",
)
parser.add_argument("--num_eval_runs", type=int, default=3, help="Number of repeated eval rollouts to run.")
parser.add_argument(
    "--pointcloud_source",
    type=str,
    choices=["sampler", "depth", "lidar", "both"],
    default=None,
    help="Point-cloud source. Defaults to dagger.pointcloud_source in the student YAML.",
)
parser.add_argument(
    "--base-action-scale",
    type=float,
    default=None,
    help="Eval-only override for the environment base action scale.",
)
parser.add_argument(
    "--base-control-mode",
    choices=("position", "velocity"),
    default=None,
    help="Eval-only base actuator control mode; velocity uses the deployed speed and acceleration limits.",
)
parser.add_argument(
    "--aux-handle-gt-refresh-steps",
    type=int,
    default=0,
    help=(
        "Eval-only: overwrite the recurrent aux-handle input with the current ground-truth handle pose "
        "every N policy steps; 0 disables periodic refresh."
    ),
)
parser.add_argument(
    "--arm-action-scale",
    type=float,
    default=None,
    help="Eval-only override for the environment Franka arm action scale.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Hard stop after this many env steps. Use 0 to run until every env freezes.",
)
parser.add_argument("--print_interval", type=int, default=15, help="Print rollout stats every N env steps.")
parser.add_argument(
    "--attn-viz",
    "--attn_viz",
    dest="attn_viz",
    action="store_true",
    default=False,
    help="Capture the student decoder's action-query cross-attention over the observation and, for "
    "the --viser-env-id env, print the per-modality attention and save labeled attention JPGs.",
)
parser.add_argument(
    "--attn-interval",
    "--attn_interval",
    dest="attn_interval",
    type=int,
    default=25,
    help="With --attn-viz, save one attention snapshot every N env steps (attn_0.jpg, attn_25.jpg, ...).",
)
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
parser.add_argument(
    "--save_replay",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "Save a .pt replay file to PATH containing joint_pos, joint_vel, actions, and "
        "applied_targets for every env at every step, in rollout order. "
        "Load with torch.load(path) to get a dict with keys 'runs', 'joint_names', 'num_envs'."
    ),
)
parser.add_argument(
    "--viser-env-id",
    "--viser_env_id",
    dest="viser_env_id",
    type=int,
    default=0,
    help="Env index to capture for --attn-viz.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

student_cfg_path = _resolve_repo_path(args_cli.student_cfg, DEFAULT_STUDENT_CFG)
student_dagger_defaults = _load_student_dagger_defaults(student_cfg_path)
selected_door_families = _normalize_family_selection(
    args_cli.door_families if args_cli.door_families is not None else student_dagger_defaults.get("door_families")
)
if args_cli.max_door_assets < 0:
    raise ValueError("--max-door-assets must be >= 0.")
if args_cli.max_door_assets:
    os.environ["DOOROPENING_MULTI_DOOR_ASSET_LIMIT"] = str(args_cli.max_door_assets)
    print(f"[INFO] Limiting each selected door family to {args_cli.max_door_assets} assets.")
if selected_door_families is not None:
    os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = ",".join(selected_door_families)
    print(f"[INFO] Using multi-door families: {selected_door_families}")
pointcloud_source = str(student_dagger_defaults.get("pointcloud_source", "both")).lower()
if args_cli.pointcloud_source is not None:
    pointcloud_source = args_cli.pointcloud_source
if args_cli.video:
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


DIRECTION_NAME_TO_ID = {"pull": 0, "push": 1}
DIRECTION_ID_TO_NAME = {direction_id: name for name, direction_id in DIRECTION_NAME_TO_ID.items()}
MODE_CONTACT_SENSOR_NAME = "contact_forces_door2"
MODE_CONTACT_FORCE_THRESHOLD = 5.0


def _get_base_env(env):
    return getattr(env, "unwrapped", getattr(env, "env", env))


def get_joint_velocities(base_env, joint_names):
    """Raw physical joint velocities from PhysX for the named joints -> [num_envs, len(joint_names)].

    Read straight from ``robot.data.joint_vel`` (rad/s for revolute joints, m/s for the prismatic
    base_x/base_y joints). These are the ACTUAL simulated velocities -- NOT the policy action or any
    action scale (e.g. base_action_scale) -- so they line up with what you measure on the real robot
    (base odometry / joint encoders).
    """
    all_names = list(base_env.robot.data.joint_names)
    idx = [all_names.index(n) for n in joint_names]
    return base_env.robot.data.joint_vel[:, idx]


def get_base_velocity(base_env):
    """Base planar velocity (vx, vy, wz) -> [num_envs, 3] from the base joints.

    vx, vy in m/s (base_x_joint, base_y_joint); wz in rad/s (base_rotation_joint).
    """
    return get_joint_velocities(base_env, ["base_x_joint", "base_y_joint", "base_rotation_joint"])


def get_base_lin_speed(base_env):
    """Base ground speed ||(vx, vy)|| in m/s -- directly comparable to the real robot's base odometry."""
    return torch.linalg.vector_norm(get_base_velocity(base_env)[:, :2], dim=-1)


def _build_compact_joint_layout(joint_names):
    """Return (names, indices) for the compact viser-pt joint order
    [base_x_joint, base_y_joint, base_rotation_joint, panda_joint1..7, panda_finger_joint1].

    Uses the REAL joint names so scripts/replay_viser_pt.py matches them straight onto the URDF
    (base_x/y/rotation are actuated URDF joints -> the base is driven through them).
    """
    desired = ["base_x_joint", "base_y_joint", "base_rotation_joint"] + [f"panda_joint{i}" for i in range(1, 8)]
    fingers = sorted(n for n in joint_names if str(n).startswith("panda_finger_joint"))
    desired += list(fingers)
    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
    names, indices = [], []
    for n in desired:
        if n in name_to_idx:
            names.append(n)
            indices.append(name_to_idx[n])
    return names, indices


def _patch_no_auto_resets(base_env):
    if getattr(base_env, "_eval_no_auto_reset_patch", False):
        return

    base_env._eval_original_get_dones = base_env._get_dones

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


def _compute_tracking_metrics(base_env):
    base_env._get_intermediate_values()

    key_body_pos_err_sq, key_body_quat_err_sq, door_joint_pos_err_sq, *_ = compute_tracking_error(
        robot_key_body_pos=base_env.robot_reset_key_body_pos,
        robot_key_body_quat=base_env.robot_key_body_quat,
        door_joint_pos=base_env.door_joint_pos,
        robot_base_joint_pos=base_env.robot_base_joint_pos,
        robot_arm_joint_pos=base_env.robot_arm_joint_pos,
        robot_gripper_joint_pos=base_env.robot_finger_joint_pos,
        robot_arx_joint_pos=base_env.robot_arx_joint_pos,
        robot_base_joint_vel=base_env.robot_base_joint_vel,
        robot_arm_joint_vel=base_env.robot_arm_joint_vel,
        ref_robot_key_body_pos=base_env.ref_robot_reset_key_body_pos,
        ref_robot_key_body_quat=base_env.ref_robot_key_body_quat,
        ref_door_joint_pos=base_env.ref_door_joint_pos,
        ref_robot_base_joint_pos=base_env.ref_robot_base_joint_pos,
        ref_robot_arm_joint_pos=base_env.ref_robot_arm_joint_pos,
        ref_robot_gripper_joint_pos=base_env.ref_robot_finger_joint_pos,
        ref_robot_arx_joint_pos=base_env.ref_robot_arx_joint_pos,
        ref_robot_base_joint_vel=base_env.ref_robot_base_joint_vel,
        ref_robot_arm_joint_vel=base_env.ref_robot_arm_joint_vel,
    )

    key_body_pos_err = torch.sqrt(torch.clamp_min(key_body_pos_err_sq, 0.0))
    key_body_quat_err = torch.sqrt(torch.clamp_min(key_body_quat_err_sq, 0.0))
    door_joint_pos_err = torch.sqrt(torch.clamp_min(door_joint_pos_err_sq, 0.0))

    return {
        "key_body_pos": key_body_pos_err,
        "key_body_quat": key_body_quat_err,
        "door_joint_pos": door_joint_pos_err,
    }


def _get_eval_done_status(base_env):
    original_get_dones = getattr(base_env, "_eval_original_get_dones", None)
    if original_get_dones is None:
        original_get_dones = base_env._get_dones

    original_early_stopping = bool(getattr(base_env, "early_stopping", False))
    override_state = []

    threshold_overrides = (
        ("reset_key_body_pos_delta_min", "reset_key_body_pos_delta_max", args_cli.key_body_pos_threshold),
        ("reset_key_body_quat_delta_min", "reset_key_body_quat_delta_max", args_cli.key_body_quat_threshold),
        ("reset_door_joint_pos_delta_min", "reset_door_joint_pos_delta_max", args_cli.door_joint_pos_threshold),
    )
    try:
        for min_attr, max_attr, override_value in threshold_overrides:
            if override_value is None:
                continue
            override_state.append((min_attr, getattr(base_env, min_attr), getattr(base_env.cfg, min_attr)))
            override_state.append((max_attr, getattr(base_env, max_attr), getattr(base_env.cfg, max_attr)))
            setattr(base_env, min_attr, float(override_value))
            setattr(base_env, max_attr, float(override_value))
            setattr(base_env.cfg, min_attr, float(override_value))
            setattr(base_env.cfg, max_attr, float(override_value))
        base_env.early_stopping = True
        killed, timed_out = original_get_dones()
    finally:
        for attr_name, attr_value, cfg_value in reversed(override_state):
            setattr(base_env, attr_name, attr_value)
            setattr(base_env.cfg, attr_name, cfg_value)
        base_env.early_stopping = original_early_stopping

    return killed, timed_out


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
    dagger._last_policy_point_importance = None
    dagger._last_policy_point_importance_env = -1
    dagger._last_policy_importance_pcd_base = None
    dagger._last_policy_point_importance_step = None
    dagger._last_policy_importance_base_pos_w = None
    dagger._last_policy_importance_base_quat_w = None
    dagger._resample_wall_distractors()
    # Per-rollout door augmentations are drawn once at reset (the training loop does this in run(); the
    # eval loop must too). Without this, _door_hole_aug_metadata stays None for the whole eval and the
    # window hole / glass reflection are never applied even with door_hole_aug.env_prob = 1.0.
    dagger._resample_door_frame_visibility()
    dagger._resample_door_hole_aug()
    # Diagnostic: prove the window-hole aug is actually active this eval. env_fraction should be ~1.0
    # with door_hole_aug.env_prob=1.0, and hole_width/height_mean_m should be > 0. If stats is empty or
    # env_fraction is 0, the hole metadata is not being applied (chase that); if it's populated but you
    # still see no hole in the .pt, the hole is being re-filled at render (blur/occluder) or cropped.
    print(
        f"[HOLE-DIAG] hole_enabled={dagger.door_hole_aug_enabled} "
        f"reflection_enabled={dagger.door_hole_reflection_enabled} "
        f"stats={dagger.latest_door_hole_aug_stats}"
    )
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
    gt_refresh_steps = int(getattr(dagger, "eval_aux_handle_gt_refresh_steps", 0))
    if gt_refresh_steps > 0 and iteration > 0 and iteration % gt_refresh_steps == 0:
        dagger.aux_buffer[:] = dagger._build_seed_aux_buffer_values()
    student_obs = dagger._build_student_obs(iteration=iteration)
    # _append_viser_frame serializes the exact history tensors used by this policy forward.
    dagger.latest_eval_student_obs = student_obs
    student_output = dagger._student_forward(student_obs)
    if dagger.has_aux_prediction:
        aux_prediction = dagger._decode_aux_prediction(student_output["aux"].detach())
        dagger.aux_buffer[:] = aux_prediction
    student_actions = student_output["action"][:, 0, :]
    env_actions = dagger._student_actions_to_env_actions(student_actions)
    return env_actions, student_output


def _render_attention_jpgs(attn_state, run_index=None):
    """Per-step attention TABLE (matplotlib, Agg): attn[_run<r>]_<step>.jpg where EVERY memory token is one
    cell, colored by the action-query attention. NO mean/sum aggregation. The point-cloud tokens are
    laid out as a labeled grid ('pointcloud'); every other token (proprioception at each timestamp,
    aux, target_err, conditions) is its own labeled row with its value printed."""
    out_dir = attn_state["out_dir"]
    env_id = attn_state["env_id"]
    snapshots = attn_state["snapshots"]
    if not snapshots:
        print("[ATTN] nothing captured; no jpgs written.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[ATTN] matplotlib unavailable ({exc}); skipping jpg rendering.")
        return

    mass_rows = []
    raw_token_rows = []
    for step, labels, attn, _pcd in snapshots:
        attn_np = attn.numpy() if hasattr(attn, "numpy") else np.asarray(attn, dtype=float)
        raw_token_rows.append(
            {
                "step": int(step),
                "tokens": [
                    {"index": int(i), "label": str(label), "attention": float(attn_np[i])}
                    for i, label in enumerate(labels)
                ],
            }
        )
        vmin, vmax = float(attn_np.min()), float(attn_np.max())
        pcd_idx = [i for i, lb in enumerate(labels) if lb == "pointcloud"]
        # Only report the four state terms that are part of this experiment.  Condition
        # tokens and any legacy/unused fields are deliberately excluded from the summary.
        is_requested_state = lambda lb: str(lb) in {"aux_handle_pos", "target_err_arm", "q_arm", "base_vel"} or str(lb).startswith(("aux_handle_pos ", "target_err_arm ", "q_arm ", "base_vel "))
        other_idx = [i for i, lb in enumerate(labels) if lb != "pointcloud" and is_requested_state(lb)]
        q_arm_idx = [i for i, lb in enumerate(labels) if isinstance(lb, str) and (lb == "q_arm" or lb.startswith("q_arm "))]
        base_vel_idx = [i for i, lb in enumerate(labels) if isinstance(lb, str) and (lb == "base_vel" or lb.startswith("base_vel "))]
        q_arm_mass = float(attn_np[q_arm_idx].sum()) if q_arm_idx else 0.0
        base_vel_mass = float(attn_np[base_vel_idx].sum()) if base_vel_idx else 0.0
        pointcloud_mass = float(attn_np[pcd_idx].sum()) if pcd_idx else 0.0
        mass_rows.append(
            {
                "step": int(step),
                "pointcloud": pointcloud_mass,
                "q_arm_history": q_arm_mass,
                "base_vel_history": base_vel_mass,
                "aux_handle_pos": float(attn_np[[i for i, lb in enumerate(labels) if str(lb).startswith("aux_handle_pos")]].sum()) if any(str(lb).startswith("aux_handle_pos") for lb in labels) else 0.0,
                "tracking_error": float(attn_np[[i for i, lb in enumerate(labels) if str(lb).startswith("target_err_arm")]].sum()) if any(str(lb).startswith("target_err_arm") for lb in labels) else 0.0,
                "pointcloud_token_count": len(pcd_idx),
            }
        )

        # Keep full-resolution attention data in JSON while limiting diagnostic JPGs.
        # Framewise point importance is stored separately in the replay and is not sampled.
        if len(snapshots) > 50 and int(step) % 25 != 0 and int(step) != int(snapshots[-1][0]):
            continue

        n_other = max(1, len(other_idx))
        fig = plt.figure(figsize=(13.0, max(5.0, 0.34 * n_other)))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0])

        # Point-cloud tokens: a grid, each cell = one token (order is FPS, not spatial).
        ax_pcd = fig.add_subplot(gs[0, 0])
        if pcd_idx:
            n = len(pcd_idx)
            cols = 16
            rows = int(np.ceil(n / cols))
            grid = np.full(rows * cols, np.nan)
            grid[:n] = attn_np[pcd_idx]
            grid = grid.reshape(rows, cols)
            ax_pcd.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax_pcd.set_title(f"pointcloud  ({len(pcd_idx)} tokens)")
        ax_pcd.set_xticks([])
        ax_pcd.set_yticks([])

        # Every other token: one labeled row with its attention value printed in the cell.
        ax_oth = fig.add_subplot(gs[0, 1])
        col = attn_np[other_idx].reshape(len(other_idx), 1) if other_idx else np.zeros((1, 1))
        im = ax_oth.imshow(col, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax_oth.set_xticks([])
        ax_oth.set_yticks(range(len(other_idx)))
        ax_oth.set_yticklabels([labels[i] for i in other_idx], fontsize=8)
        ax_oth.set_title("requested state terms only")
        mid = 0.5 * (vmin + vmax)
        for r, i in enumerate(other_idx):
            ax_oth.text(0, r, f"{attn_np[i]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if attn_np[i] < mid else "black")

        fig.colorbar(im, ax=[ax_pcd, ax_oth], label="action-query attention (mean over decoder layers)",
                     shrink=0.8)
        _tag = f"run{run_index}_" if run_index is not None else ""
        _rlbl = f", run {run_index}" if run_index is not None else ""
        fig.suptitle(f"per-token attention (env {env_id}{_rlbl}, step {step})")
        path = os.path.join(out_dir, f"attn_{_tag}{step}.jpg")
        fig.savefig(path, dpi=100, format="jpg")
        plt.close(fig)
    if mass_rows:
        steps = [row["step"] for row in mass_rows]
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for key, label, color in (
            ("pointcloud", "Point cloud (sum of all tokens)", "#e69f00"),
            ("aux_handle_pos", "Aux handle position", "#d55e00"),
            ("tracking_error", "Tracking error", "#cc79a7"),
            ("q_arm_history", "Franka joint", "#0072b2"),
            ("base_vel_history", "Measured base velocity", "#009e73"),
        ):
            values = [row[key] for row in mass_rows]
            ax.plot(steps, values, marker="o", linewidth=2, label=label, color=color)
            if key == "pointcloud":
                for x, y in zip(steps, values):
                    ax.annotate(f"{y:.0%}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Policy step")
        ax.set_ylabel("Summed action-query attention mass")
        ax.set_title(f"Attention mass by modality — env {env_id}{_rlbl}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        mass_path = os.path.join(out_dir, f"attn_mass_{_tag.rstrip('_') or 'eval'}.jpg")
        fig.savefig(mass_path, dpi=140, format="jpg")
        plt.close(fig)
        json_path = os.path.splitext(mass_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(mass_rows, f, indent=2)
        print(f"[ATTN] saved summed modality attention plot + values to {mass_path}")
    if raw_token_rows:
        token_path = os.path.join(out_dir, f"attn_tokens_{_tag.rstrip('_') or 'eval'}.json")
        with open(token_path, "w", encoding="utf-8") as f:
            json.dump(raw_token_rows, f, indent=2)
        print(f"[ATTN] saved all per-token weights to {token_path}")

        # Plot every non-pointcloud memory token separately over time. In particular, each
        # proprioceptive field/timestamp (e.g. q_arm 300ms) remains its own curve, not a subtotal.
        all_labels = []
        for row in raw_token_rows:
            for token in row["tokens"]:
                label = token["label"]
                if label != "pointcloud" and label not in all_labels and is_requested_state(label):
                    all_labels.append(label)
        history_fig, history_ax = plt.subplots(figsize=(12.0, 6.5))
        steps = [row["step"] for row in raw_token_rows]
        history_ax.plot(
            steps,
            [row["pointcloud"] for row in mass_rows],
            color="#e69f00",
            linewidth=2.5,
            marker="o",
            markersize=3,
            label="Point cloud (sum of all tokens)",
        )
        for label in all_labels:
            values = []
            for row in raw_token_rows:
                values.append(
                    sum(token["attention"] for token in row["tokens"] if token["label"] == label)
                )
            history_ax.plot(steps, values, linewidth=1.4, label=label)
        history_ax.set_xlabel("Policy step")
        history_ax.set_ylabel("Action-query attention weight")
        history_ax.set_title(f"Per-token history attention over rollout — env {env_id}{_rlbl}")
        history_ax.set_ylim(bottom=0.0)
        history_ax.grid(True, alpha=0.25)
        history_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        history_fig.tight_layout()
        history_path = os.path.join(out_dir, f"attn_history_{_tag.rstrip('_') or 'eval'}.jpg")
        history_fig.savefig(history_path, dpi=130, format="jpg", bbox_inches="tight")
        plt.close(history_fig)
        print(f"[ATTN] saved per-token history timeline to {history_path}")
    _run_str = f" (run {run_index})" if run_index is not None else ""
    print(f"[ATTN] saved {len(snapshots)} per-step token tables{_run_str} to {out_dir}")


def _render_query_attention_maps(attn_state, run_index=None):
    """Render separate observation-importance maps for action, aux-handle, and door-joint queries."""
    snapshots = attn_state.get("query_snapshots", [])
    if not snapshots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[ATTN] matplotlib unavailable for query maps ({exc}); skipping.")
        return

    out_dir = attn_state["out_dir"]
    env_id = attn_state["env_id"]
    tag = f"run{run_index}_" if run_index is not None else ""
    for query_name in ("action", "base", "franka", "aux_handle", "door_joint"):
        rows = []
        for step, labels, query_weights in snapshots:
            if query_name not in query_weights:
                continue
            values = query_weights[query_name].numpy()
            rows.append({
                "step": int(step),
                "tokens": [
                    {"index": int(i), "label": str(label), "attention": float(values[i])}
                    for i, label in enumerate(labels)
                ],
            })
            if len(snapshots) > 50 and int(step) % 25 != 0 and int(step) != int(snapshots[-1][0]):
                continue
            pcd_idx = [i for i, label in enumerate(labels) if label == "pointcloud"]
            other_idx = [i for i, label in enumerate(labels) if label != "pointcloud"]
            vmin, vmax = float(values.min()), float(values.max())
            fig = plt.figure(figsize=(13.0, max(5.0, 0.34 * max(1, len(other_idx)))))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0])
            ax_pcd = fig.add_subplot(gs[0, 0])
            if pcd_idx:
                cols = 16
                grid = np.full(int(np.ceil(len(pcd_idx) / cols)) * cols, np.nan)
                grid[:len(pcd_idx)] = values[pcd_idx]
                ax_pcd.imshow(grid.reshape(-1, cols), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax_pcd.set_title(f"pointcloud ({len(pcd_idx)} tokens)")
            ax_pcd.set_xticks([])
            ax_pcd.set_yticks([])
            ax_oth = fig.add_subplot(gs[0, 1])
            col = values[other_idx].reshape(len(other_idx), 1) if other_idx else np.zeros((1, 1))
            im = ax_oth.imshow(col, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax_oth.set_xticks([])
            ax_oth.set_yticks(range(len(other_idx)))
            ax_oth.set_yticklabels([labels[i] for i in other_idx], fontsize=8)
            mid = 0.5 * (vmin + vmax)
            for row_idx, token_idx in enumerate(other_idx):
                ax_oth.text(0, row_idx, f"{values[token_idx]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if values[token_idx] < mid else "black")
            fig.colorbar(im, ax=[ax_pcd, ax_oth], label=f"{query_name} query attention")
            fig.suptitle(f"{query_name} observation attention (env {env_id}, step {step})")
            fig.savefig(os.path.join(out_dir, f"attn_{query_name}_{tag}{step}.jpg"), dpi=100, format="jpg")
            plt.close(fig)

        if rows:
            json_path = os.path.join(out_dir, f"attn_{query_name}_{tag.rstrip('_') or 'eval'}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            print(f"[ATTN] saved {query_name} query maps and token weights to {json_path}")


def _render_pointcloud_source_attention(attn_state, run_index=None):
    """Plot temporal point-cloud attention split into handle, panel, and wall regions."""
    rows = attn_state.get("region_snapshots", [])
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = attn_state["out_dir"]
    env_id = attn_state["env_id"]
    tag = f"run{run_index}_" if run_index is not None else ""
    names = ("action", "base", "franka", "aux_handle", "door_joint")
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.8 * len(names)), sharex=True)
    for ax, name in zip(axes, names):
        steps = [int(step) for step, values in rows if name in values]
        for key, label, color in (("handle", "Door handle", "#d55e00"), ("panel", "Door panel", "#0072b2"), ("wall", "Wall distractors", "#009e73"), ("robot", "Robot", "#cc79a7")):
            ax.plot(steps, [float(values[name][key]) for step, values in rows if name in values], marker="o", linewidth=2, label=label, color=color)
        ax.set_ylabel(f"{name}\nattention mass")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Policy step")
    fig.suptitle(f"Point-cloud attention over rollout — env {env_id}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"attn_pointcloud_regions_{tag.rstrip('_') or 'eval'}.jpg")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump([{"step": int(step), **values} for step, values in rows], f, indent=2)
    print(f"[ATTN] saved temporal point-cloud source attention to {path}")
    count_rows = attn_state.get("point_count_snapshots", [])
    if count_rows:
        count_path = os.path.join(out_dir, f"pointcloud_point_counts_{tag.rstrip('_') or 'eval'}.json")
        with open(count_path, "w", encoding="utf-8") as f:
            json.dump([{"step": int(step), **counts} for step, counts in count_rows], f, indent=2)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        steps = [int(step) for step, _ in count_rows]
        count_keys = ("handle", "panel", "wall", "robot", "unassigned")
        count_names = ("Door handle", "Door panel", "Wall distractors", "Robot", "Unassigned")
        count_colors = ("#d55e00", "#0072b2", "#009e73", "#cc79a7", "#999999")
        ax.stackplot(steps, *[[counts.get(k, 0) for _, counts in count_rows] for k in count_keys],
                     labels=count_names, colors=count_colors, alpha=0.9)
        ax.set_xlabel("Policy step")
        ax.set_ylabel("Policy-input point count")
        ax.set_title(f"Raw policy point-cloud composition — env {env_id}\nnearest GT surface within 12 cm; unassigned shown separately")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=9)
        fig.tight_layout()
        count_plot = os.path.join(out_dir, f"pointcloud_point_counts_{tag.rstrip('_') or 'eval'}.png")
        fig.savefig(count_plot, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[ATTN] saved raw policy-cloud point counts to {count_plot}")


def _render_joint_output_attributions(attn_state, run_index=None):
    """Save shared attention and per-output gradient x activation token attributions."""
    rows = attn_state.get("output_attribution_snapshots", [])
    queries = attn_state.get("query_snapshots", [])
    regions = attn_state.get("region_snapshots", [])
    if not rows or not queries or not regions:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = attn_state["out_dir"]
    env_id = attn_state["env_id"]
    tag = f"run{run_index}_" if run_index is not None else ""
    names = [
        "Door handle points", "Door panel points", "Wall points", "Robot points", "Other cloud tokens",
        "Franka joint position", "Base velocity", "Arm target error", "Aux handle position",
        "Other state tokens",
    ]
    keys = ["handle", "panel", "wall", "robot", "other_cloud", "franka", "base_vel", "target_err_arm", "aux_handle_pos", "other_state"]
    colors = ["#e69f00", "#56b4e9", "#009e73", "#cc79a7", "#999999", "#0072b2", "#d55e00", "#666666", "#8c564b", "#7f3c8d"]

    # Raw attention is one shared door-joint query. Aggregate its per-token attention
    # into the same point-cloud and state groups used for the two output-specific maps.
    region_by_step = {int(s): v.get("door_joint", {}) for s, v in regions}
    shared_rows = []
    for step, labels, query_weights in queries:
        if "door_joint" not in query_weights or int(step) not in region_by_step:
            continue
        w = query_weights["door_joint"].detach().float().cpu()
        rr = region_by_step[int(step)]
        cloud = sum(float(w[i]) for i, label in enumerate(labels) if label == "pointcloud")
        vals = {
            "handle": float(rr.get("handle", 0.0)),
            "panel": float(rr.get("panel", 0.0)),
            "wall": float(rr.get("wall", 0.0)),
            "robot": float(rr.get("robot", 0.0)),
        }
        vals["other_cloud"] = max(0.0, cloud - vals["handle"] - vals["panel"] - vals["wall"] - vals["robot"])
        for idx, label in enumerate(labels):
            if label == "pointcloud":
                continue
            if label.startswith("q_arm"):
                key = "franka"
            elif label.startswith("base_vel"):
                key = "base_vel"
            elif label.startswith("target_err_arm"):
                key = "target_err_arm"
            elif label.startswith("aux_handle_pos"):
                key = "aux_handle_pos"
            else:
                key = "other_state"
            vals[key] = vals.get(key, 0.0) + float(w[idx])
        shared_rows.append((int(step), vals))

    output_rows = {}
    for step, values in rows:
        for output_name, groups in values.items():
            output_rows.setdefault(output_name, []).append((int(step), groups))
    attribution_json = os.path.join(out_dir, f"output_token_attributions_{tag.rstrip('_') or 'eval'}.json")
    with open(attribution_json, "w", encoding="utf-8") as f:
        json.dump(
            [{"step": int(step), "outputs": values} for step, values in rows],
            f,
            indent=2,
        )
    print(f"[ATTN] saved per-output token attributions to {attribution_json}")

    plot_specs = [("shared_attention", "Shared door-joint query attention", shared_rows, True)]
    plot_specs.extend(
        (f"{name}_output_attribution", f"{name.replace('_', ' ').title()} output attribution", series, False)
        for name, series in output_rows.items()
    )
    for suffix, title, series_rows, is_shared in plot_specs:
        if not series_rows:
            continue
        plot_steps = [s for s, _ in series_rows]
        series = {key: [float(values.get(key, 0.0)) * 100.0 for _, values in series_rows] for key in keys}
        fig, ax = plt.subplots(figsize=(15.5, 6.2))
        ax.stackplot(plot_steps, *[series[k] for k in keys], labels=names, colors=colors, alpha=0.94)
        ax.set_xlim(min(plot_steps), max(plot_steps))
        ax.set_ylim(0, 100)
        ax.set_xlabel("Policy step")
        ax.set_ylabel("Attention mass (%)" if is_shared else "Normalized |gradient × activation| (%)")
        suffix_note = "" if is_shared else "\nOutput-specific encoder-token attribution; not attention mass or causal proof"
        ax.set_title(f"{title} — env {env_id}{suffix_note}")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True, fontsize=9)
        fig.subplots_adjust(bottom=0.23, top=0.86, left=0.08, right=0.98)
        path = os.path.join(out_dir, f"output_{suffix}_{tag.rstrip('_') or 'eval'}.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[ATTN] saved output attribution figure to {path}")


def _classify_pointcloud_attention(attn, dagger, env_id):
    """Assign raw policy-cloud points and PointNet centers to nearest scene surfaces."""
    result = {}
    try:
        device = dagger.device

        def quat_apply_wxyz(quat, points):
            qv = quat[1:].expand_as(points)
            uv = torch.cross(qv, points, dim=-1)
            uuv = torch.cross(qv, uv, dim=-1)
            return points + 2.0 * (quat[0] * uv + uuv)

        robot_body_idx = int(dagger.robot_base_body_idx)
        base_pos_w = dagger.ov_env.robot.data.body_pos_w[env_id, robot_body_idx]
        base_quat_w = dagger.ov_env.robot.data.body_quat_w[env_id, robot_body_idx]
        quat_inv = torch.cat((base_quat_w[:1], -base_quat_w[1:]))

        def world_to_base(points_world):
            return quat_apply_wxyz(quat_inv, points_world - base_pos_w)

        asset_idx = int(dagger.env_asset_idx[env_id].detach().cpu().item())
        link_cache = dagger.door_link_pointclouds.get(asset_idx, {})

        def posed_link_base(link_name):
            points_local = link_cache.get(link_name)
            if points_local is None or points_local.numel() == 0:
                return torch.zeros((0, 3), dtype=torch.float32, device=device)
            body_idx = int(dagger.door_link_body_indices[link_name])
            pos_w = dagger.ov_env.door.data.body_pos_w[env_id, body_idx]
            quat_w = dagger.ov_env.door.data.body_quat_w[env_id, body_idx]
            return world_to_base(quat_apply_wxyz(quat_w, points_local.to(device)) + pos_w)

        surfaces = [(0, posed_link_base("link_2")), (1, posed_link_base("link_1"))]
        wall_world = getattr(dagger, "_wall_distractor_world_points", None)
        if wall_world is not None and wall_world.numel():
            wall = wall_world[env_id]
            if wall.shape[0] > 12000:
                wall = wall[torch.linspace(0, wall.shape[0] - 1, 12000, device=device).long()]
            surfaces.append((2, world_to_base(wall)))
        # The depth camera sees the robot itself. Include that geometry explicitly so robot
        # points cannot silently be credited to a nearby wall or door surface.
        robot_world = dagger._sample_robot_pointcloud_world_sampler()[env_id]
        surfaces.append((3, world_to_base(robot_world)))

        def point_surface_distances(points):
            class_dists, class_ids = [], []
            for region_id, surface in surfaces:
                if surface.numel() == 0:
                    continue
                # Randomized wall meshes may contain NaN padding. cdist propagates any NaN
                # row through min(), which previously made every token look like the wall.
                surface = surface[torch.isfinite(surface).all(dim=1)]
                if surface.numel() == 0:
                    continue
                if surface.shape[0] > 12000:
                    idx = torch.linspace(0, surface.shape[0] - 1, 12000, device=device).long()
                    surface = surface[idx]
                distances = torch.cdist(points.unsqueeze(0), surface.unsqueeze(0)).squeeze(0)
                class_dists.append(distances.min(dim=1).values)
                class_ids.append(region_id)
            if not class_dists:
                return torch.empty((points.shape[0], 0), device=device), class_ids
            return torch.stack(class_dists, dim=1), class_ids

        def label_points(points):
            distances, class_ids = point_surface_distances(points)
            if not class_ids:
                return torch.full((points.shape[0],), -1, dtype=torch.long, device=device)
            nearest_dist, nearest_idx = distances.min(dim=1)
            labels = torch.tensor(class_ids, dtype=torch.long, device=device)[nearest_idx]
            labels[nearest_dist > 0.12] = -1
            return labels

        # Count actual policy-input points, not latent token centers. This is an approximate
        # physical-source breakdown (nearest GT surface within 12 cm); unassigned remains explicit.
        input_pcd = getattr(dagger, "_last_policy_input_pcd_base", None)
        if input_pcd is not None and input_pcd.shape[0] > env_id:
            input_xyz = input_pcd[env_id, ..., :3].to(device=device, dtype=torch.float32)
            finite = torch.isfinite(input_xyz).all(dim=-1)
            raw_distances, class_ids = point_surface_distances(input_xyz[finite])
            nearest_dist, nearest_idx = raw_distances.min(dim=1)
            raw_labels = torch.tensor(class_ids, dtype=torch.long, device=device)[nearest_idx]
            raw_labels[nearest_dist > 0.12] = -1
            result["__point_counts__"] = torch.bincount(
                raw_labels[raw_labels >= 0], minlength=4
            ).detach().cpu()
            result["__point_counts__"] = result["__point_counts__"].tolist()
            diagnostics = {}
            for cutoff in (0.01, 0.03, 0.06, 0.12):
                accepted = nearest_dist <= cutoff
                histogram = torch.bincount(
                    torch.tensor(class_ids, dtype=torch.long, device=device)[nearest_idx[accepted]],
                    minlength=4,
                ) if bool(accepted.any()) else torch.zeros(4, dtype=torch.long, device=device)
                diagnostics[f"nearest_within_{int(cutoff * 100):02d}cm"] = {
                    name: int(histogram[i].detach().cpu()) for i, name in enumerate(("handle", "panel", "wall", "robot"))
                }
            diagnostics["nearest_distance_median"] = float(nearest_dist.median().detach().cpu())
            diagnostics["nearest_distance_p90"] = float(torch.quantile(nearest_dist, 0.9).detach().cpu())
            source_names = ("handle", "panel", "wall", "robot")
            diagnostics["median_distance_by_source_m"] = {
                source_names[region_id]: float(raw_distances[:, i].median().detach().cpu())
                for i, region_id in enumerate(class_ids)
                if region_id < len(source_names)
            }
            result["__point_counts__"] = {
                "handle": int(result["__point_counts__"][0]),
                "panel": int(result["__point_counts__"][1]),
                "wall": int(result["__point_counts__"][2]),
                "robot": int(result["__point_counts__"][3]),
                "unassigned": int((raw_labels < 0).sum().detach().cpu()),
                "total": int(finite.sum().detach().cpu()),
                "distance_diagnostics": diagnostics,
            }

        for key, (xyz, _weights) in attn.get("pcd", {}).items():
            centers = xyz.to(device=device, dtype=torch.float32)
            region = label_points(centers)
            result[key] = {"region": region.detach().cpu()}
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[ATTN] point-cloud region classification unavailable: {exc}")
        return {}


def _save_replay_pt(replay_runs, joint_names, base_env, save_replay_arg):
    """Write the accumulated per-run replay to a .pt. Called after each run for incremental saves."""
    replay_path = pathlib.Path(save_replay_arg).expanduser()
    if replay_path.is_dir():
        replay_path = replay_path / "replay.pt"
    replay_path = str(replay_path)
    os.makedirs(os.path.dirname(replay_path) or ".", exist_ok=True)
    torch.save(
        {
            "runs": replay_runs,
            "joint_names": joint_names,
            "num_envs": int(base_env.num_envs),
            "base_impedance_randomization_samples": getattr(
                base_env, "base_impedance_randomization_samples", {}
            ),
            "base_velocity_schema": {
                "frame": "robot",
                "components": ["vx", "vy", "yaw_rate"],
                "units": ["m/s", "m/s", "rad/s"],
                "measured_source": "simulated base joint velocities rotated into robot frame, before env.step",
                "policy_input_source": "velocity vector returned to the student policy (may include configured noise)",
                "commanded_source": "clamped student base action multiplied by env base_action_scale",
            },
            "base_effort_schema": {
                "components": ["base_x_force", "base_y_force", "base_yaw_torque"],
                "units": ["N", "N", "N m"],
                "source": "IsaacLab implicit-actuator applied_torque after env.step, last physics substep; clipped PD estimate, not a direct PhysX force measurement",
            },
            "control_frequency_hz": float(1.0 / max(float(getattr(base_env, "step_dt", 1.0 / 15.0)), 1e-8)),
        },
        replay_path,
    )
    print(f"[INFO] Replay saved to {replay_path} ({len(replay_runs)} run(s) so far).")


NUM_VISER_ENVS = 8


def _build_viser_meta(base_env):
    """Compact-joint layout shared by every per-env viser recorder (built once)."""
    compact_names, compact_idx = _build_compact_joint_layout(list(base_env.robot.data.joint_names))
    # applied_robot_dof_targets is stored in _robot_dof_idx order (NOT joint_names order) -> map each
    # compact joint's full-dof index to its position within _robot_dof_idx for the target.
    dof_to_target_pos = {int(d): k for k, d in enumerate(base_env._robot_dof_idx.detach().cpu().tolist())}
    target_pos = [int(dof_to_target_pos.get(int(j), -1)) for j in compact_idx]
    return {
        "compact_names": compact_names,
        "compact_idx": torch.as_tensor(compact_idx, device=base_env.device, dtype=torch.long),
        "target_idx": torch.as_tensor([max(p, 0) for p in target_pos], device=base_env.device, dtype=torch.long),
        "target_valid": torch.as_tensor([p >= 0 for p in target_pos], device=base_env.device, dtype=torch.bool),
        "frame_dt": float(getattr(base_env, "step_dt", 0.025)),
    }


def _viser_env_label(base_env, env_id):
    """(family_name, asset_name) used to name a per-env viser file."""
    asset_idx = int(base_env.env_asset_indices[env_id].item())
    family_name = DOOR_FAMILY_NAMES[int(asset_family_ids[asset_idx])]
    asset_name = pathlib.Path(asset_paths[asset_idx]).resolve().parent.name
    return family_name, asset_name


_VARIANT_META_CACHE = {}
_DOOR_SHAPE_SLICES = {}
_ROBOT_GRIPPER_SHAPE_SLICES = {}


def _load_variant_meta(asset_idx):
    """variant_meta.json shipped next to each door URDF (geometry of that door). {} if absent."""
    asset_idx = int(asset_idx)
    if asset_idx not in _VARIANT_META_CACHE:
        meta = {}
        try:
            meta_path = pathlib.Path(asset_paths[asset_idx]).resolve().parent / "variant_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
        except Exception as exc:  # geometry is diagnostic only -- never fail the eval over it
            print(f"[WARN] Could not read variant_meta.json for door asset {asset_idx}: {exc}")
        _VARIANT_META_CACHE[asset_idx] = meta
    return _VARIANT_META_CACHE[asset_idx]


def _door_geometry(asset_idx):
    """Handle/panel geometry for one door asset (all None for assets without variant_meta.json)."""
    meta = _load_variant_meta(asset_idx)
    props = meta.get("actual_properties") or meta.get("source_properties") or {}
    panel = meta.get("panel", {})
    door = meta.get("door", {})
    handle = meta.get("handle", {})
    return {
        "panel_width_m": props.get("panel_width_m", panel.get("width_m")),
        "panel_height_m": props.get("panel_height_m", panel.get("height_m")),
        "handle_height_m": props.get("handle_height_m"),
        "handle_to_edge_distance_m": props.get("handle_to_edge_distance_m"),
        "handle_lever_length_m": handle.get("lever_length_m"),
        "handle_type": handle.get("type"),
        "handle_side": props.get("handle_side", door.get("handle_side")),
        "hinge_side": door.get("hinge_side"),
        "opening_direction": props.get("opening_direction", door.get("opening_direction")),
    }


def _door_material_shape_slices(base_env):
    """Flat material-buffer shape slices for the panel (link_1) and handle (link_2).

    Same per-link shape-count walk randomize_body_material_subset uses (the stock IsaacLab parse
    asserts on these convex-decomposition doors), so the friction reported here is the friction the
    event terms actually wrote.
    """
    if not _DOOR_SHAPE_SLICES:
        _DOOR_SHAPE_SLICES["__ok__"] = False
        try:
            view = base_env.door.root_physx_view
            link_paths = view.link_paths[0]
            total = int(view.max_shapes)
            counts = [
                base_env.door._physics_sim_view.create_rigid_body_view(link_path).max_shapes
                for link_path in link_paths
            ]
            for name in ("link_1", "link_2"):
                body_ids, _ = base_env.door.find_bodies(name)
                body_id = int(body_ids[0])
                start = sum(counts[:body_id])
                end = total if body_id == len(link_paths) - 1 else start + counts[body_id]
                _DOOR_SHAPE_SLICES[name] = (max(0, min(start, total)), max(0, min(end, total)))
            _DOOR_SHAPE_SLICES["__ok__"] = True
        except Exception as exc:
            print(f"[WARN] Could not resolve door material shape slices (friction not reported): {exc}")
    return _DOOR_SHAPE_SLICES


def _robot_gripper_material_shape_slices(base_env):
    """Flat material-buffer slices for the gripper fingers and palm/hand."""
    if not _ROBOT_GRIPPER_SHAPE_SLICES:
        _ROBOT_GRIPPER_SHAPE_SLICES["__ok__"] = False
        try:
            robot = base_env.robot
            view = robot.root_physx_view
            link_paths = view.link_paths[0]
            total = int(view.max_shapes)
            counts = [
                robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes
                for link_path in link_paths
            ]
            for name in ("panda_leftfinger", "panda_rightfinger", "panda_hand"):
                body_ids, _ = robot.find_bodies(name)
                if not len(body_ids):
                    continue
                body_id = int(body_ids[0])
                start = sum(counts[:body_id])
                end = total if body_id == len(link_paths) - 1 else start + counts[body_id]
                _ROBOT_GRIPPER_SHAPE_SLICES[name] = (
                    max(0, min(start, total)), max(0, min(end, total))
                )
            _ROBOT_GRIPPER_SHAPE_SLICES["__ok__"] = bool(
                any(name in _ROBOT_GRIPPER_SHAPE_SLICES for name in
                    ("panda_leftfinger", "panda_rightfinger", "panda_hand"))
            )
        except Exception as exc:
            print(f"[WARN] Could not resolve gripper material shape slices (friction not reported): {exc}")
    return _ROBOT_GRIPPER_SHAPE_SLICES


def _door_param_snapshot(base_env, env_id):
    """Per-env door physics AS SAMPLED AT THIS RESET, plus the asset's static geometry.

    Gains come from the _door_nominal_* buffers rather than door.data, because edit_door_articulation
    rewrites the live joint effort limit every step (1e6 while latched), so the live values are not
    the sampled ones.
    """
    env_id = int(env_id)
    board_idx = base_env._door_board_joint_idx
    hinge_idx = base_env._door_hinge_joint_idx
    asset_idx = int(base_env.env_asset_indices[env_id].item())
    family_name, asset_name = _viser_env_label(base_env, env_id)

    snap = {
        "env_id": env_id,
        "family": family_name,
        "asset": asset_name,
        "panel_stiffness_nm_per_rad": float(base_env._door_nominal_joint_stiffness[env_id, board_idx]),
        "panel_damping_nms_per_rad": float(base_env._door_nominal_joint_damping[env_id, board_idx]),
        "panel_joint_friction_coeff": float(base_env.door.data.joint_friction_coeff[env_id, board_idx]),
        "panel_effort_limit_nm": float(base_env._door_panel_effort_limits[env_id, 0]),
        "handle_stiffness_nm_per_rad": float(base_env._door_nominal_joint_stiffness[env_id, hinge_idx]),
        "handle_damping_nms_per_rad": float(base_env._door_nominal_joint_damping[env_id, hinge_idx]),
        "handle_effort_limit_nm": float(base_env._door_handle_effort_limits[env_id, 0]),
        "latch_threshold_rad": float(base_env._door_latch_hinge_threshold_rad[env_id]),
        "panel_mass_kg": None,
        "panel_friction_static": None,
        "handle_friction_static": None,
        "handle_friction_dynamic": None,
        "robot_gripper_friction_static": None,
        "robot_gripper_friction_dynamic": None,
        "robot_gripper_friction_by_body": {},
    }

    try:
        body_ids, _ = base_env.door.find_bodies("link_1")
        snap["panel_mass_kg"] = float(base_env.door.root_physx_view.get_masses()[env_id, int(body_ids[0])])
    except Exception as exc:
        print(f"[WARN] Could not read door panel mass for env {env_id}: {exc}")

    slices = _door_material_shape_slices(base_env)
    if slices.get("__ok__"):
        try:
            materials = base_env.door.root_physx_view.get_material_properties()
            panel_start, panel_end = slices["link_1"]
            handle_start, handle_end = slices["link_2"]
            if panel_end > panel_start:
                snap["panel_friction_static"] = float(materials[env_id, panel_start:panel_end, 0].mean())
            if handle_end > handle_start:
                snap["handle_friction_static"] = float(materials[env_id, handle_start:handle_end, 0].mean())
                snap["handle_friction_dynamic"] = float(materials[env_id, handle_start:handle_end, 1].mean())
        except Exception as exc:
            print(f"[WARN] Could not read door material properties for env {env_id}: {exc}")

    gripper_slices = _robot_gripper_material_shape_slices(base_env)
    if gripper_slices.get("__ok__"):
        try:
            materials = base_env.robot.root_physx_view.get_material_properties()
            static_values = []
            dynamic_values = []
            for body_name in ("panda_leftfinger", "panda_rightfinger", "panda_hand"):
                shape_slice = gripper_slices.get(body_name)
                if shape_slice is None:
                    continue
                start, end = shape_slice
                if end <= start:
                    continue
                body_materials = materials[env_id, start:end]
                static_mean = float(body_materials[:, 0].mean())
                dynamic_mean = float(body_materials[:, 1].mean())
                snap["robot_gripper_friction_by_body"][body_name] = {
                    "static": static_mean,
                    "dynamic": dynamic_mean,
                    "shape_count": int(end - start),
                }
                static_values.append(body_materials[:, 0].reshape(-1))
                dynamic_values.append(body_materials[:, 1].reshape(-1))
            if static_values:
                snap["robot_gripper_friction_static"] = float(torch.cat(static_values).mean())
                snap["robot_gripper_friction_dynamic"] = float(torch.cat(dynamic_values).mean())
        except Exception as exc:
            print(f"[WARN] Could not read gripper material properties for env {env_id}: {exc}")

    snap.update(_door_geometry(asset_idx))
    return snap


# (filename key, snapshot key, format) for the compact tag appended to each viser .pt name.
DOOR_PARAM_FILENAME_FIELDS = (
    ("pk", "panel_stiffness_nm_per_rad", ".4g"),
    ("pd", "panel_damping_nms_per_rad", ".3g"),
    ("pfr", "panel_joint_friction_coeff", ".2f"),
    ("pcap", "panel_effort_limit_nm", ".3g"),
    ("pm", "panel_mass_kg", ".4g"),
    ("hk", "handle_stiffness_nm_per_rad", ".3g"),
    ("hd", "handle_damping_nms_per_rad", ".2g"),
    ("hcap", "handle_effort_limit_nm", ".3g"),
    ("hfr", "handle_friction_static", ".2f"),
    ("lat", "latch_threshold_rad", ".2f"),
    ("w", "panel_width_m", ".3f"),
    ("hh", "handle_height_m", ".3f"),
    ("he", "handle_to_edge_distance_m", ".3f"),
)

DOOR_PARAM_FILENAME_LEGEND = (
    "pk/pd=panel(joint_1) stiffness[Nm/rad]/damping[Nms/rad], pfr=panel Coulomb friction coeff, "
    "pcap=panel effort-limit cap[Nm], pm=panel mass[kg], hk/hd=handle(joint_2) stiffness/damping, "
    "hcap=handle effort limit[Nm], hfr=handle surface static friction, lat=unlatch threshold[rad], "
    "w=panel width[m], hh=handle height[m], he=handle-to-edge distance[m]"
)


def _fmt_param(value, spec):
    return "na" if value is None else format(float(value), spec)


def _door_param_filename_tag(snap):
    return "_".join(f"{key}{_fmt_param(snap.get(field), spec)}" for key, field, spec in DOOR_PARAM_FILENAME_FIELDS)


def _print_door_param_table(snapshots, tag):
    """One line per recorded env so the log carries the same numbers as the .pt filenames."""
    if not snapshots:
        return
    print(f"[INFO] {tag} -- per-env door physics + geometry ({DOOR_PARAM_FILENAME_LEGEND})")
    header = (
        f"{'env':>4} {'panel_k':>8} {'panel_d':>8} {'p_fric':>7} {'p_cap':>7} {'p_mass':>7} "
        f"{'hnd_k':>6} {'hnd_d':>6} {'h_cap':>6} {'h_fric':>7} {'latch':>6} "
        f"{'width':>6} {'h_hgt':>6} {'h_edge':>6}  asset"
    )
    print("[INFO] " + header)
    for snap in snapshots:
        print(
            "[INFO] "
            f"{snap['env_id']:>4} "
            f"{_fmt_param(snap['panel_stiffness_nm_per_rad'], '8.2f')} "
            f"{_fmt_param(snap['panel_damping_nms_per_rad'], '8.2f')} "
            f"{_fmt_param(snap['panel_joint_friction_coeff'], '7.3f')} "
            f"{_fmt_param(snap['panel_effort_limit_nm'], '7.2f')} "
            f"{_fmt_param(snap['panel_mass_kg'], '7.2f')} "
            f"{_fmt_param(snap['handle_stiffness_nm_per_rad'], '6.2f')} "
            f"{_fmt_param(snap['handle_damping_nms_per_rad'], '6.3f')} "
            f"{_fmt_param(snap['handle_effort_limit_nm'], '6.2f')} "
            f"{_fmt_param(snap['handle_friction_static'], '7.3f')} "
            f"{_fmt_param(snap['latch_threshold_rad'], '6.3f')} "
            f"{_fmt_param(snap['panel_width_m'], '6.3f')} "
            f"{_fmt_param(snap['handle_height_m'], '6.3f')} "
            f"{_fmt_param(snap['handle_to_edge_distance_m'], '6.3f')}  "
            f"{snap['family']}/{snap['asset']} ({snap['opening_direction']}, hinge={snap['hinge_side']})"
        )


def _build_viser_recorder(base_env, env_id):
    """Per-env recorder: the static door pose/URDF + an empty frame list.

    Keeps the original raw-viser design (one file per selected env) but records the
    replay_viser_pt.py URDF payload (compact joints + door URDF) so the meshes render.
    """
    env_origin = base_env.scene.env_origins[env_id].detach().clone()  # on device
    asset_idx = int(base_env.env_asset_indices[env_id].item())
    # Save the door's GLOBAL urdf path so the replay loads the real door mesh and animates it with the
    # per-frame ACTUAL sim door joints. Door is fixed-base, so store its root pose env-relative once.
    door_urdf = os.path.abspath(str(asset_paths[asset_idx]))
    door_root = base_env.door.data.root_state_w[env_id, :7].detach().clone()  # world, on device
    return {
        "env_id": int(env_id),
        "env_origin": env_origin,
        # Per-env door physics (sampled at the reset that just ran) + this asset's geometry, so the
        # rollout can be matched back to the exact door it was rolled out on.
        "door_params": _door_param_snapshot(base_env, env_id),
        "door_urdf_path": door_urdf,
        "door_joint_names": list(base_env.door.data.joint_names),
        "door_root_pos": (door_root[:3] - env_origin).detach().cpu().clone(),  # env-relative
        "door_root_quat": door_root[3:7].detach().cpu().clone(),
        "ground_truth_walls_points_world": None,
        "frames": [],
    }


def _select_viser_recorders(base_env, num_envs=NUM_VISER_ENVS):
    """Pick several random eval envs and build a recorder for each (matches the original design)."""
    n = max(1, min(int(num_envs), int(base_env.num_envs)))
    env_ids = torch.randperm(base_env.num_envs, device=base_env.device)[:n].detach().cpu().tolist()
    recorders = [_build_viser_recorder(base_env, int(env_id)) for env_id in env_ids]
    labels = [f"env{r['env_id']}:{_viser_env_label(base_env, r['env_id'])[1]}" for r in recorders]
    print("[INFO] Viser envs (URDF replay, one .pt each):", ", ".join(labels))
    _print_door_param_table([r["door_params"] for r in recorders], "Viser envs")
    return recorders


def _append_viser_frame(recorder, viser_meta, base_env, dagger, student_output):
    """Append one replay frame for the recorder's env. Call only while that env is active."""
    env_id = recorder["env_id"]
    org = recorder["env_origin"]  # device (3,)
    q = base_env.robot.data.joint_pos[env_id, viser_meta["compact_idx"]].detach().cpu().clone()
    tgt = base_env.applied_robot_dof_targets[env_id, viser_meta["target_idx"]].detach().cpu().clone()
    tgt[~viser_meta["target_valid"].cpu()] = 0.0  # joints not in the controlled set
    # Two DIFFERENT robot poses matter here, and mixing them up is what makes the meshes drift:
    #   * root_state_w  -> the articulation root, which is the URDF root link `base_link`. The replay
    #     places `/robot` here and then drives base_x/base_y/base_rotation from compact_q, so the base
    #     travel is replayed by the URDF itself. The robot does NOT start at the world origin, so this
    #     spawn/root offset must be kept -- dropping it double-counts (or drops) the base motion.
    #   * tidybot2_base_link (body_pos_w/body_quat_w at _robot_base_body_link_idx) -> the frame the
    #     policy point cloud and aux vectors are actually expressed in (this is the pose
    #     _build_student_obs feeds to world_to_local). Base-frame quantities must be un-transformed with
    #     THIS pose so they land in the world next to the door URDF, not at the root.
    rb_root = base_env.robot.data.root_state_w[env_id, :7].detach()
    base_body_idx = base_env._robot_base_body_link_idx
    rb_base_pos = base_env.robot.data.body_pos_w[env_id, base_body_idx].detach()
    rb_base_quat = base_env.robot.data.body_quat_w[env_id, base_body_idx].detach()

    def _base_to_world(points_base, base_pos_w=None, base_quat_w=None):
        """Rotate+translate base-body-frame points [N, 3] into env-relative world coords."""
        pts = points_base.detach().to(torch.float32)
        bq = (rb_base_quat if base_quat_w is None else base_quat_w).to(torch.float32)  # world wxyz
        axis = bq[1:].unsqueeze(0).expand(pts.shape[0], 3)
        uv = torch.cross(axis, pts, dim=-1)
        uuv = torch.cross(axis, uv, dim=-1)
        world = pts + 2.0 * (bq[0] * uv + uuv)
        base_pos = rb_base_pos if base_pos_w is None else base_pos_w
        return world + (base_pos - org).to(torch.float32).unsqueeze(0)

    def _pointcloud_base_to_world(points_base, base_pos_w=None, base_quat_w=None):
        """Transform a padded base-frame cloud without rendering its zero padding at the robot."""
        pts = points_base.detach().to(torch.float32)
        # local_pcd_t is fixed-size and uses exact zero rows for absent/cropped samples.
        # A literal base-frame origin is outside the configured point-cloud z crop, so it
        # cannot be a real depth return.  Preserve it as NaN for replay, rather than
        # rotating/translating it into a visible cluster that follows the base.
        # Only the two crop blocks use zero padding.  If a separate robot-model
        # block is appended, retain it even in the unlikely event it contains a
        # physical point at its base-frame origin.
        crop_point_count = min(pts.shape[0], sum(int(count) for count in dagger.local_pcd_points[:2]))
        padding = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
        padding[:crop_point_count] = (
            torch.isfinite(pts[:crop_point_count]).all(dim=-1)
            & (pts[:crop_point_count].abs().amax(dim=-1) == 0.0)
        )
        world = _base_to_world(pts, base_pos_w=base_pos_w, base_quat_w=base_quat_w)
        return torch.where(padding.unsqueeze(-1), torch.full_like(world, float("nan")), world)

    # ACTUAL sim door joints for this frame (NOT predicted); the replay poses the door URDF with these.
    dj = base_env.door.data.joint_pos[env_id].detach()
    frame = {
        "compact_q": q,
        "compact_target": tgt,
        "door_joint_pos": dj.cpu().clone(),
        "door_joint_pos_limits": base_env.door.data.joint_pos_limits[env_id].detach().cpu().clone(),
        "door_unlatch_threshold_rad": base_env._door_latch_hinge_threshold_rad[env_id]
        .detach()
        .to(torch.float32)
        .cpu()
        .clone(),
        # URDF root (`base_link`) pose; base_x/base_y/base_rotation are replayed from compact_q.
        "robot_base_pos_w": (rb_root[:3] - org).cpu().clone(),
        "robot_base_quat_w": rb_root[3:7].cpu().clone(),
        # Exact pose of the body frame used for pointcloud normalization. Keep separate from the
        # articulation-root pose above so saved pointcloud-frame conversions can be audited.
        "pointcloud_base_pos_w": (rb_base_pos - org).to(torch.float32).cpu().clone(),
        "pointcloud_base_quat_w": rb_base_quat.to(torch.float32).cpu().clone(),
    }
    # These filtered PhysX sensor values belong to this env's current observation pose.
    # Save them per frame so a replay can identify which door the fingers touched;
    # the periodic [CONTACT] log only reports mean/max across active envs.
    for frame_key, sensor_attr in (
        ("finger_panel_contact_force_norm_n", "finger_panel_contact_force_norm"),
        ("gripper_handle_contact_force_norm_n", "finger_handle_contact_force_norm"),
    ):
        force = getattr(base_env, sensor_attr, None)
        if isinstance(force, torch.Tensor) and force.shape[0] > env_id:
            frame[frame_key] = float(force[env_id].detach().item())
    # Save the normalized policy action and the exact temporal proprioception tensor fed to
    # the student on this frame: [history_time, feature] per field, including any lag/noise.
    if isinstance(student_output, dict) and isinstance(student_output.get("action"), torch.Tensor):
        action = student_output["action"]
        if action.ndim >= 3:
            action = action[env_id, 0]
        elif action.ndim == 2:
            action = action[env_id]
        frame["policy_action"] = action.detach().to(torch.float32).cpu().reshape(-1).clone()
    # Capture the live PhysX limits once per Viser stream as well as the IsaacLab-side limit buffer
    # above. This verifies that a dynamic limit override reached the simulation backend.
    if not recorder["frames"]:
        physx_joint_limits = base_env.door.root_physx_view.get_dof_limits()
        frame["door_joint_pos_limits_physx"] = (
            physx_joint_limits[env_id].detach().to(torch.float32).cpu().clone()
        )
    # Save the auxiliary door-joint head output beside the raw action for temporal
    # correlation analysis. The model predicts [hinge, latch] in radians.
    if isinstance(student_output, dict) and isinstance(student_output.get("door_joint"), torch.Tensor):
        door_joint_prediction = student_output["door_joint"]
        if door_joint_prediction.ndim >= 3:
            door_joint_prediction = door_joint_prediction[env_id, 0]
        elif door_joint_prediction.ndim == 2:
            door_joint_prediction = door_joint_prediction[env_id]
        frame["door_joint_prediction"] = (
            door_joint_prediction.detach().to(torch.float32).cpu().reshape(-1).clone()
        )

    student_obs = getattr(dagger, "latest_eval_student_obs", None)
    temporal_key = getattr(dagger, "proprio_temporal_obs_key", None)
    if isinstance(student_obs, dict) and temporal_key in student_obs:
        temporal_obs = student_obs[temporal_key]
        if isinstance(temporal_obs, dict):
            state_history = {}
            for field_name, values in temporal_obs.items():
                if isinstance(values, torch.Tensor) and values.ndim >= 3 and values.shape[0] > env_id:
                    state_history[str(field_name)] = values[env_id].detach().to(torch.float32).cpu().clone()
            # Older/shared temporal encoders expose one [batch, feature] observation key
            # per field and timestamp instead of one [batch, time, feature] field tensor.
            if not state_history:
                field_obs_keys = getattr(dagger, "proprio_temporal_field_obs_keys", {})
                for field_name in getattr(dagger, "proprio_temporal_fields", ()):
                    obs_keys = field_obs_keys.get(field_name, ())
                    samples = [
                        temporal_obs[key][env_id].detach().to(torch.float32).cpu()
                        for key in obs_keys
                        if key in temporal_obs
                        and isinstance(temporal_obs[key], torch.Tensor)
                        and temporal_obs[key].ndim == 2
                        and temporal_obs[key].shape[0] > env_id
                    ]
                    if samples:
                        state_history[str(field_name)] = torch.stack(samples, dim=0).clone()
            frame["robot_state_history"] = state_history
            frame["robot_state_history_timestamps_ms"] = [
                float(t) for t in getattr(dagger, "proprio_temporal_timestamps_ms", ())
            ]
    # Preserve the exact camera pose used to render this observation.  This lets replay diagnostics
    # distinguish a genuinely moving world surface from a camera/world reprojection error.
    sampler_camera_pose = dagger._get_sampler_camera_pose()[env_id].detach().to(torch.float32).clone()
    sampler_camera_pose[:3] -= org.to(torch.float32)
    frame["sampler_camera_pose_env_xyzw"] = sampler_camera_pose.cpu()
    # Policy-input point cloud (the local_pcd_t the student consumes), cached on the dagger in
    # _build_student_obs. Pre-transform base-body-frame -> env-relative world here so the replay just
    # displays it (the `_points_world` suffix marks an already-world cloud stream).
    pcd_base = getattr(dagger, "_last_policy_input_pcd_base", None)
    if pcd_base is not None and pcd_base.shape[0] > env_id:
        point_importance = getattr(dagger, "_last_policy_point_importance", None)
        importance_pcd_base = getattr(dagger, "_last_policy_importance_pcd_base", None)
        has_matching_importance = (
            point_importance is not None
            and importance_pcd_base is not None
            and getattr(dagger, "_last_policy_point_importance_env", -1) == env_id
            and point_importance.numel() == importance_pcd_base.shape[0]
        )
        # Importance is computed sparsely for visualization. Keep its exact source
        # cloud beside it on intervening frames; pairing cached scores with the
        # freshly sampled cloud would color unrelated points when point order changes.
        replay_pcd_base = importance_pcd_base if has_matching_importance else pcd_base[env_id]
        importance_pos = getattr(dagger, "_last_policy_importance_base_pos_w", None) if has_matching_importance else None
        importance_quat = getattr(dagger, "_last_policy_importance_base_quat_w", None) if has_matching_importance else None
        frame["policy_input_points_world"] = _pointcloud_base_to_world(
            replay_pcd_base[..., :3], base_pos_w=importance_pos, base_quat_w=importance_quat
        ).cpu().clone()
        if has_matching_importance:
            frame["policy_point_importance"] = point_importance.clone()
            frame["policy_importance_step"] = getattr(dagger, "_last_policy_point_importance_step", None)
    # RealSense-rendered depth-cam cloud (the raw render, robot hand included and self-occluding the
    # door -- this is BEFORE _filter_robot_points_base strips the robot's own points for the policy).
    # Both clouds live in the same base-body frame, so _base_to_world applies identically. NaN padding
    # is carried through and dropped by replay_viser_pt.py's finite filter.
    depth_cam_world = getattr(dagger, "_last_rendered_depth_pcd_world", None)
    if depth_cam_world is not None and depth_cam_world.shape[0] > env_id:
        # This is the direct world-frame render, before the world->robot-base transform used for policy
        # input. Recording it directly prevents replay from depending on a second quaternion rotation.
        frame["robot_depth_cam_obs_points_world"] = (
            depth_cam_world[env_id, ..., :3] - org.to(torch.float32)
        ).cpu().clone()
    else:
        depth_cam_base = getattr(dagger, "_last_rendered_depth_pcd_base", None)
        if depth_cam_base is not None and depth_cam_base.shape[0] > env_id:
            frame["robot_depth_cam_obs_points_world"] = _base_to_world(
                depth_cam_base[env_id, ..., :3]
            ).cpu().clone()
    wall_points_world = getattr(dagger, "_last_ground_truth_wall_points_world", None)
    if recorder["ground_truth_walls_points_world"] is None and wall_points_world is not None:
        if wall_points_world.shape[0] > env_id:
            recorder["ground_truth_walls_points_world"] = (
                wall_points_world[env_id].detach().to(torch.float32) - org.to(torch.float32)
            ).cpu().clone()
    # Aux handle position: the network's PREDICTED handle pos and the aux INPUT it received are both in
    # the base-body frame. Pre-transform to env-relative world here (the replay renders *_world directly)
    # so they track the point cloud instead of the root.
    if (
        getattr(dagger, "has_aux_prediction", False)
        and isinstance(student_output, dict)
        and "aux" in student_output
    ):
        aux_pred = dagger._decode_aux_prediction(student_output["aux"].detach())
        if aux_pred is not None and aux_pred.shape[0] > env_id:
            frame["aux_prediction_world"] = _base_to_world(aux_pred[env_id].reshape(1, 3))[0].cpu().clone()
    aux_in = getattr(dagger, "latest_aux_input_vector", None)
    if aux_in is not None and aux_in.shape[0] > env_id:
        frame["aux_input_world"] = _base_to_world(aux_in[env_id].reshape(1, 3))[0].cpu().clone()
    recorder["frames"].append(frame)


def _save_viser_recorders(recorders, viser_meta, base_env, out_dir, run_index):
    """Write one replay_viser_pt.py-compatible .pt per env into out_dir (called after each run)."""
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for recorder in recorders:
        if not recorder["frames"]:
            continue
        env_id = recorder["env_id"]
        family_name, asset_name = _viser_env_label(base_env, env_id)
        # Physics/geometry tag in the name so several rollouts can be told apart at a glance; the
        # same numbers (plus the ones that do not fit) live under "door_params" inside the file.
        param_tag = _door_param_filename_tag(recorder["door_params"])
        path = os.path.join(
            out_dir,
            f"run{int(run_index):02d}_{family_name}_env{env_id:03d}_{asset_name}_{param_tag}.pt",
        )
        torch.save(
            {
                "frames": recorder["frames"],
                "door_params": recorder["door_params"],
                "base_impedance": getattr(base_env, "base_impedance_randomization_samples", {}).get(env_id),
                "compact_target_joint_names": viser_meta["compact_names"],
                "frame_dt": viser_meta["frame_dt"],
                "door_urdf_path": recorder["door_urdf_path"],
                "door_joint_names": recorder["door_joint_names"],
                "door_root_pos": recorder["door_root_pos"],  # env-relative
                "door_root_quat": recorder["door_root_quat"],
                "pointcloud_streams": [
                    {
                        "name": "policy_input",
                        "key": "policy_input_points_world",
                        "label": "Policy input (importance heatmap)",
                        "color": (80, 160, 255),
                        "importance_key": "policy_point_importance",
                    },
                    {
                        "name": "ground_truth_walls",
                        "key": "ground_truth_walls_points_world",
                        "label": "Ground-truth wall points",
                        "color": (125, 125, 135),
                    }
                ],
                "static_pointclouds": {
                    "ground_truth_walls": recorder["ground_truth_walls_points_world"]
                },
            },
            path,
        )
        print(f"[INFO] Saved {len(recorder['frames'])} viser frames (env {env_id}) to {path}.")
        saved += 1
    if saved == 0:
        print("[INFO] Viser: no active frames captured this run; nothing saved.")
    else:
        print(f"[INFO] Viser filename params: {DOOR_PARAM_FILENAME_LEGEND}")


def _build_teacher_actions(dagger, obs):
    teacher_output = dagger._get_teacher_actions(obs)
    return teacher_output["actions"], None


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
    activate_denom = active_count + drift_count
    activate_rate = active_count / activate_denom if activate_denom > 0 else 0.0
    print(
        "[EVAL] "
        f"run={run_index} step={step} active={active_count} activate_rate={activate_rate:.3f} "
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
    if args_cli.door_solver_position_iterations is not None:
        iterations = int(args_cli.door_solver_position_iterations)
        if iterations < 1:
            raise ValueError("--door-solver-position-iterations must be >= 1.")
        env_cfg.door_eval_solver_position_iterations = iterations
        print(f"[INFO] Eval-only door PhysX position iterations: {iterations}")
    if args_cli.door_handle_limit_margin_range_rad is not None:
        low, high = map(float, args_cli.door_handle_limit_margin_range_rad)
        if low < 0.0 or high < low:
            raise ValueError("--door-handle-limit-margin-range-rad requires 0 <= MIN <= MAX.")
        env_cfg.door_handle_limit_margin_range_rad = (low, high)
        print(
            "[INFO] Eval-only handle joint upper stop margin above unlatch threshold: "
            f"Uniform({low:.3f}, {high:.3f}) rad"
        )
    if args_cli.panel_stiffness_range_nm_per_rad is not None:
        low, high = map(float, args_cli.panel_stiffness_range_nm_per_rad)
        if low <= 0.0 or high < low:
            raise ValueError("--panel-stiffness-range-nm-per-rad requires 0 < MIN <= MAX.")
        env_cfg.adr_cfg_dict["door_board_joint_stiffness_and_damping"][
            "stiffness_distribution_params"
        ] = (low, high)
        print(f"[INFO] Eval-only full-ADR panel stiffness range: ({low}, {high}) Nm/rad")
    if args_cli.panel_effort_limit_range_nm is not None:
        low, high = map(float, args_cli.panel_effort_limit_range_nm)
        if low < 0.0 or high < low:
            raise ValueError("--panel-effort-limit-range-nm requires 0 <= MIN <= MAX.")
        env_cfg.door_panel_effort_limit_range_nm = (low, high)
        print(f"[INFO] Eval-only full-ADR panel effort-limit range: ({low}, {high}) Nm")
    if args_cli.handle_stiffness_range_nm_per_rad is not None:
        low, high = map(float, args_cli.handle_stiffness_range_nm_per_rad)
        if low <= 0.0 or high < low:
            raise ValueError("--handle-stiffness-range-nm-per-rad requires 0 < MIN <= MAX.")
        env_cfg.adr_cfg_dict["door_hinge_joint_stiffness_and_damping"][
            "stiffness_distribution_params"
        ] = (low, high)
        print(f"[INFO] Eval-only full-ADR handle joint_2 stiffness range: ({low}, {high}) Nm/rad")
    if args_cli.handle_effort_limit_range_nm is not None:
        low, high = map(float, args_cli.handle_effort_limit_range_nm)
        if low < 0.0 or high < low:
            raise ValueError("--handle-effort-limit-range-nm requires 0 <= MIN <= MAX.")
        env_cfg.door_handle_effort_limit_range_nm = (low, high)
        env_cfg.adr_cfg_dict["door_hinge_joint_effort_limit"][
            "effort_limit_distribution_params"
        ] = (low, high)
        print(f"[INFO] Eval-only full-ADR handle joint_2 effort-limit range: ({low}, {high}) Nm")
    if args_cli.base_action_scale is not None:
        if args_cli.base_action_scale <= 0.0:
            raise ValueError("--base-action-scale must be positive")
        env_cfg.base_action_scale = args_cli.base_action_scale
        print(f"[INFO] Eval-only base_action_scale override: {env_cfg.base_action_scale}")
    if args_cli.base_control_mode is not None:
        env_cfg.base_control_mode = args_cli.base_control_mode
        print(f"[INFO] Eval-only base_control_mode override: {env_cfg.base_control_mode}")
    if args_cli.arm_action_scale is not None:
        if args_cli.arm_action_scale <= 0.0:
            raise ValueError("--arm-action-scale must be positive")
        env_cfg.arm_action_scale = args_cli.arm_action_scale
        print(f"[INFO] Eval-only arm_action_scale override: {env_cfg.arm_action_scale}")

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg["params"]["seed"] = args_cli.seed

    dagger_runtime_cfg = dict(student_dagger_defaults)
    dagger_runtime_cfg.pop("wandb", None)
    dagger_runtime_cfg["pointcloud_source"] = pointcloud_source
    depth_render_cfg = dict(dagger_runtime_cfg.get("depth_cam_render", {}))
    wall_distractor_cfg = dict(dagger_runtime_cfg.get("wall_distractors", {}))
    if args_cli.wall_raster_width_px is not None:
        if args_cli.wall_raster_width_px <= 0:
            raise ValueError("--wall-raster-width-px must be positive.")
        depth_render_cfg["wall_raster_width_px"] = args_cli.wall_raster_width_px
    if args_cli.wall_raster_height_px is not None:
        if args_cli.wall_raster_height_px <= 0:
            raise ValueError("--wall-raster-height-px must be positive.")
        depth_render_cfg["wall_raster_height_px"] = args_cli.wall_raster_height_px
    if args_cli.wall_face_jitter_m is not None:
        if args_cli.wall_face_jitter_m < 0.0:
            raise ValueError("--wall-face-jitter-m must be non-negative.")
        wall_distractor_cfg["face_jitter_m"] = args_cli.wall_face_jitter_m
    if args_cli.wall_height_range_m is not None:
        low, high = map(float, args_cli.wall_height_range_m)
        if low < 0.0 or high < low:
            raise ValueError("--wall-height-range-m requires 0 <= MIN <= MAX.")
        wall_distractor_cfg["height_range_m"] = [low, high]
    if args_cli.flush_wall_height_range_m is not None:
        low, high = map(float, args_cli.flush_wall_height_range_m)
        if low < 0.0 or high < low:
            raise ValueError("--flush-wall-height-range-m requires 0 <= BOTTOM <= TOP.")
        wall_distractor_cfg["flush_height_range_m"] = [low, high]
    if (
        args_cli.wall_raster_width_px is not None
        or args_cli.wall_raster_height_px is not None
    ):
        dagger_runtime_cfg["depth_cam_render"] = depth_render_cfg
    if (
        args_cli.wall_face_jitter_m is not None
        or args_cli.wall_height_range_m is not None
        or args_cli.flush_wall_height_range_m is not None
    ):
        dagger_runtime_cfg["wall_distractors"] = wall_distractor_cfg
    if "reset_progress_total" in dagger_runtime_cfg:
        env_cfg.reset_progress_total = dagger_runtime_cfg["reset_progress_total"]
    if "adr_reset_progress_total" in dagger_runtime_cfg:
        env_cfg.adr_reset_progress_total = dagger_runtime_cfg["adr_reset_progress_total"]
    else:
        env_cfg.adr_reset_progress_total = 0.5 * float(env_cfg.reset_progress_total)

    env_cfg.pointcloud_render_mode = "none"
    env_cfg.enable_pointcloud_camera = False

    # TEMP (eval only): run WITHOUT the gripper armature + joint friction (both -> 0). There is no
    # longer a reset-time armature event or ADR range to zero as well -- the Franka hand's armature
    # is set once, in glorbot_cfg's GRIPPER_ARMATURE -- so overriding the actuator here is enough.
    # _gripper_act = env_cfg.robot_cfg.actuators.get("panda_hand")
    # if _gripper_act is not None:
    #     _gripper_act.armature = 0.0
    #     _gripper_act.friction = 0.0
    # print("[INFO] TEMP: gripper armature + joint friction disabled (set to 0.0) for this eval run.")

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = os.path.join("runs", f"DoorOpening-Distillation-Eval_{timestamp}")
    nn_dir = os.path.join(experiment_dir, "nn")
    summaries_dir = os.path.join(experiment_dir, "summaries")
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    print(f"[INFO] Eval output directory: {experiment_dir}")
    print(f"[INFO] pointcloud_source={pointcloud_source}, render_mode={env_cfg.pointcloud_render_mode}")
    print(
        "[INFO] Wall renderer config: "
        f"raster={depth_render_cfg.get('wall_raster_width_px', 80)}x"
        f"{depth_render_cfg.get('wall_raster_height_px', 60)}, "
        f"face_jitter_m={wall_distractor_cfg.get('face_jitter_m', 0.004)}, "
        f"side_height_m={wall_distractor_cfg.get('height_range_m')}, "
        f"flush_height_m={wall_distractor_cfg.get('flush_height_range_m')}"
    )

    # Eval never streams the raw per-env viser recordings; the only viser output is the single-env
    # URDF replay written below when --viser is set. Force the dagger's raw viser recorder off.
    dagger_runtime_cfg["viser"] = {"raw": {"enabled": False}}

    preconvert_shared_urdf_assets(
        door_configs=env_cfg.door_cfg,
        timeout_s=float(args_cli.asset_preconvert_timeout_s),
        poll_interval_s=float(args_cli.asset_preconvert_poll_interval_s),
        verbose=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = _get_base_env(env)
    if args_cli.door_solver_position_iterations is not None:
        # The tensor ArticulationView does not expose solver iteration setters. Author the
        # PhysX articulation-root attributes before the first env.reset()/physics step instead.
        import omni.usd
        from pxr import PhysxSchema

        iterations = int(args_cli.door_solver_position_iterations)
        stage = omni.usd.get_context().get_stage()
        live_counts = []
        for env_id in range(base_env.num_envs):
            door_prefix = f"/World/envs/env_{env_id}/Door/"
            articulation_prims = [
                prim
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(door_prefix)
                and prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
            ]
            if not articulation_prims:
                raise RuntimeError(f"No PhysX articulation root found below {door_prefix}.")
            for prim in articulation_prims:
                api = PhysxSchema.PhysxArticulationAPI.Get(stage, prim.GetPath())
                api.CreateSolverPositionIterationCountAttr().Set(iterations)
                live_counts.append(api.GetSolverPositionIterationCountAttr().Get())
        print(f"[DOOR-SOLVER] requested={iterations} live={live_counts}")
    if base_env.ref_motion_lib is None:
        raise RuntimeError("Eval requires env_cfg.use_motion_ref=True and a valid traj.pkl for each door asset.")
    base_env.ref_motion_lib.reset_from_start = True
    base_env.early_stopping = False
    _patch_no_auto_resets(base_env)
    _print_family_summary(base_env)
    try:
        _replay_joint_names = list(base_env.robot.data.joint_names)
    except Exception:
        _replay_joint_names = None

    # --viser: record several random envs to replay_viser_pt.py-compatible .pt files (robot + door
    # URDF meshes), one file per env, into <experiment_dir>/viser. viser_meta holds the shared compact
    # joint layout; per-env recorders are (re)selected at the start of each eval run below.
    viser_meta = None
    viser_dir = None
    viser_recorders = None
    if args_cli.viser:
        viser_meta = _build_viser_meta(base_env)
        viser_dir = os.path.join(experiment_dir, "viser")
        os.makedirs(viser_dir, exist_ok=True)
        print(
            f"[INFO] Saving viser URDF replays (joint angles + PD targets + door URDF + actual sim door "
            f"joints) for up to {NUM_VISER_ENVS} envs -> {viser_dir}; joints={viser_meta['compact_names']}"
        )

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
        teacher_config = {"cfg": teacher_cfg_path, "obs_type": "critic"}
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
            teacher_config = {"cfg": teacher_cfg_path, "obs_type": "critic"}
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
    if args_cli.aux_handle_gt_refresh_steps < 0:
        raise ValueError("--aux-handle-gt-refresh-steps must be >= 0.")
    if args_cli.aux_handle_gt_refresh_steps > 0:
        if not dagger.has_aux_input or not dagger.aux_feedback_to_policy:
            raise ValueError(
                "Periodic aux-handle GT refresh requires aux inputs and recurrent aux feedback to the policy."
            )
        if dagger.aux_handle_init_source != "ground_truth":
            raise ValueError(
                "Periodic aux-handle GT refresh requires dagger.aux_handle_init_source='ground_truth'."
            )
        dagger.eval_aux_handle_gt_refresh_steps = int(args_cli.aux_handle_gt_refresh_steps)
        print(
            "[INFO] Eval periodically refreshes the recurrent aux-handle input from ground truth "
            f"every {dagger.eval_aux_handle_gt_refresh_steps} policy steps."
        )

    # Attention visualization: hook the student decoder's action-query cross-attention. Aggregated
    # for the tracked env (--viser-env-id) each step -> per-modality attention (printed) + a saved log.
    attn_state = None
    if args_cli.attn_viz or args_cli.viser:
        from DoorOpening.model.attention_capture import AttentionCapture, top_token_str

        _attn_env_id = max(0, min(int(args_cli.viser_env_id), int(dagger.num_envs) - 1))
        _attn_dir = os.path.join(experiment_dir, "attn_viz")
        os.makedirs(_attn_dir, exist_ok=True)
        attn_state = {
            "capture": AttentionCapture(dagger.student_model),
            "env_id": _attn_env_id,
            "format": top_token_str,
            # A Viser replay must pair every point-cloud frame with a fresh score map.
            # Sparse attention snapshots are useful for diagnostics, but forward-filling
            # their scores makes intermediate replay frames look framewise-scored when
            # they are not. Keep the full attribution path on every eval frame with Viser.
            "interval": 1 if args_cli.viser else max(1, int(args_cli.attn_interval)),
            "snapshots": [],  # (step, token_labels, token_attn, pcd) every `interval` steps
            "query_snapshots": [],  # (step, token_labels, {query_name: token_attn})
            "region_snapshots": [],  # (step, {query_name: {handle, panel, wall}})
            "point_count_snapshots": [],  # (step, raw policy-cloud points by nearest GT source)
            # Per-output gradient x activation scores on encoder memory. Raw attention is shared.
            "memory": {},
            "output_attribution_snapshots": [],
            # Output-specific gradients are more expensive than attention; still sample densely
            # enough to compare against the per-10-step attention/command traces.
            "output_attribution_interval": 1 if args_cli.viser else max(20, int(args_cli.attn_interval)),
            "out_dir": _attn_dir,
        }
        _encoder = getattr(dagger.student_model, "encoder", None)
        if _encoder is not None:
            def _retain_encoder_memory(_module, _inputs, _output):
                if torch.is_grad_enabled() and isinstance(_output, torch.Tensor):
                    _output.retain_grad()
                    attn_state["memory"]["tensor"] = _output
            attn_state["memory_hook"] = _encoder.register_forward_hook(_retain_encoder_memory)
        print(f"[INFO] Capturing student point attribution for env {_attn_env_id} -> {_attn_dir}")
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

    print(
        "[INFO] Drift thresholds:",
        {
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
        },
    )
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
    replay_runs = [] if args_cli.save_replay else None

    for run_index in range(1, num_eval_runs + 1):
        if not simulation_app.is_running():
            break
        if run_index > 1:
            obs, _ = env.reset()
            frozen_state.reset_from_env()
            _reset_dagger_rollout_state(dagger)
        if args_cli.viser:
            viser_recorders = _select_viser_recorders(base_env)
            if attn_state is not None:
                # Keep the user-selected Viser env as the attribution target. Recorder order is
                # otherwise random, so using the first recorder here can silently attach scores
                # to a different env (and leave the requested replay with default blue points).
                target_env_id = max(0, min(int(args_cli.viser_env_id), int(dagger.num_envs) - 1))
                target_recorder_idx = next(
                    (i for i, recorder in enumerate(viser_recorders)
                     if int(recorder["env_id"]) == target_env_id),
                    None,
                )
                if target_recorder_idx is not None:
                    viser_recorders.insert(0, viser_recorders.pop(target_recorder_idx))
                attn_state["env_id"] = target_env_id
        if mode_tracker is not None:
            mode_tracker.reset_run()
        if replay_runs is not None:
            _run_joint_pos: list = []
            _run_joint_vel: list = []
            _run_actions: list = []
            _run_applied_targets: list = []
            _run_active: list = []
            _run_base_velocity_measured_robot: list = []
            _run_base_velocity_policy_input_robot: list = []
            _run_base_velocity_commanded_robot: list = []
            _run_base_effort_applied: list = []
            _run_base_effort_computed: list = []

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
                    if args_cli.viser:
                        # Populate dagger._last_policy_input_pcd_base so viser frames still get the
                        # policy-input cloud even when the teacher supplies the actions.
                        dagger._build_student_obs(iteration=step)
                    actions, student_output = _build_teacher_actions(dagger, obs)
                else:
                    _capture_output_attribution = (
                        attn_state is not None
                        and int(step) % attn_state["output_attribution_interval"] == 0
                    )
                    if _capture_output_attribution:
                        with torch.enable_grad():
                            actions, student_output = _build_student_actions(dagger, iteration=step)
                    else:
                        actions, student_output = _build_student_actions(dagger, iteration=step)

                if attn_state is not None:
                    # Pop the attention captured by the forward inside _build_student_actions above.
                    # pop() every step to drain the captured weights; only store/print at the interval.
                    _attn = attn_state["capture"].pop(dagger.student_model, env_id=attn_state["env_id"])
                    if _attn is not None and int(step) % attn_state["interval"] == 0:
                        _memory = attn_state["memory"].get("tensor")
                        if _capture_output_attribution and isinstance(_memory, torch.Tensor):
                            _attr_env = attn_state["env_id"]
                            with torch.enable_grad():
                                _objectives = []
                                _aux_pred = student_output.get("aux") if isinstance(student_output, dict) else None
                                _action_pred = student_output.get("action") if isinstance(student_output, dict) else None
                                _joint_pred = student_output.get("door_joint") if isinstance(student_output, dict) else None
                                if isinstance(_aux_pred, torch.Tensor):
                                    _aux_value = _aux_pred[_attr_env, 0] if _aux_pred.ndim >= 3 else _aux_pred[_attr_env]
                                    _objectives.append(("aux_handle_position_norm", torch.linalg.vector_norm(_aux_value)))
                                if isinstance(_action_pred, torch.Tensor):
                                    _action_value = _action_pred[_attr_env, 0] if _action_pred.ndim >= 3 else _action_pred[_attr_env]
                                    if _action_value.numel() >= 10:
                                        _base_scale = float(base_env.cfg.base_action_scale)
                                        _arm_scale = float(base_env.cfg.arm_action_scale)
                                        _frame_dt = float(getattr(base_env, "step_dt", 1.0 / 15.0))
                                        _base_command = _action_value[:3].clamp(-1.0, 1.0) * _base_scale
                                        _arm_target_delta = (
                                            _action_value[3:10].clamp(-1.0, 1.0) * _arm_scale * _frame_dt
                                        )
                                        _objectives.append(("base_translation_speed_command", torch.linalg.vector_norm(_base_command[:2])))
                                        _objectives.append(("base_yaw_rate_command_abs", _base_command[2].abs()))
                                        _objectives.append(("franka_target_delta_norm", torch.linalg.vector_norm(_arm_target_delta)))
                                if isinstance(_joint_pred, torch.Tensor) and _joint_pred.shape[-1] >= 2:
                                    for _joint_idx, _joint_name in ((0, "door_handle_joint"), (1, "door_panel_joint")):
                                        _joint_value = (
                                            _joint_pred[_attr_env, 0, _joint_idx]
                                            if _joint_pred.ndim >= 3
                                            else _joint_pred[_attr_env, _joint_idx]
                                        )
                                        _objectives.append((_joint_name, _joint_value))
                                _attr_rows = []
                                for _objective_idx, (_objective_name, _scalar) in enumerate(_objectives):
                                    _grad = torch.autograd.grad(
                                        _scalar,
                                        _memory,
                                        retain_graph=(_objective_idx < len(_objectives) - 1),
                                        allow_unused=True,
                                    )[0]
                                    if _grad is None:
                                        _scores = torch.zeros(_memory.shape[1], device=_memory.device)
                                    else:
                                        _scores = (_grad[_attr_env] * _memory[_attr_env]).abs().sum(dim=-1)
                                        _scores = _scores / _scores.sum().clamp_min(1e-12)
                                    _attr_rows.append((_objective_name, _scores.detach().float().cpu().tolist()))
                            attn_state["memory"]["pending_rows"] = _attr_rows
                        region_masks = _classify_pointcloud_attention(_attn, dagger, attn_state["env_id"])
                        point_counts = region_masks.pop("__point_counts__", None)
                        if point_counts is not None:
                            attn_state["point_count_snapshots"].append((int(step), point_counts))
                        _attr_rows = attn_state["memory"].pop("pending_rows", None)
                        if _attr_rows is not None:
                            # Lift encoder-token attribution back onto the exact policy
                            # input points by nearest token center. Multiple output heads
                            # are averaged so one heatmap summarizes the decision.
                            if args_cli.viser and any(
                                r["env_id"] == attn_state["env_id"] for r in (viser_recorders or [])
                            ):
                                _input = getattr(dagger, "_last_policy_input_pcd_base", None)
                                if isinstance(_input, torch.Tensor) and _input.shape[0] > attn_state["env_id"]:
                                    _env = attn_state["env_id"]
                                    _input_xyz = _input[_env, ..., :3].detach().float()
                                    _pcd_positions_local = [i for i, lb in enumerate(_attn["token_labels"]) if lb == "pointcloud"]
                                    _combined = torch.zeros(len(_pcd_positions_local), dtype=torch.float32)
                                    for _, _score_list in _attr_rows:
                                        _combined += torch.as_tensor(_score_list, dtype=torch.float32)[_pcd_positions_local]
                                    _combined /= max(1, len(_attr_rows))
                                    _centers = []
                                    _center_scores = []
                                    _offset = 0
                                    for _key, (_xyz, _) in _attn.get("pcd", {}).items():
                                        _n = int(_xyz.shape[0])
                                        _centers.append(_xyz.float())
                                        _center_scores.append(_combined[_offset:_offset + _n])
                                        _offset += _n
                                    if _centers and sum(x.shape[0] for x in _centers) == _combined.numel():
                                        _centers = torch.cat(_centers, dim=0).to(_input_xyz.device)
                                        _center_scores = torch.cat(_center_scores).to(_input_xyz.device)
                                        _valid = torch.isfinite(_input_xyz).all(dim=-1)
                                        _point_scores = torch.zeros(_input_xyz.shape[0], dtype=torch.float32, device=_input_xyz.device)
                                        if bool(_valid.any()):
                                            _nearest = torch.cdist(_input_xyz[_valid].unsqueeze(0), _centers.unsqueeze(0)).squeeze(0).argmin(dim=1)
                                            _point_scores[_valid] = _center_scores[_nearest]
                                        dagger._last_policy_point_importance = _point_scores.detach().cpu()
                                        dagger._last_policy_point_importance_env = int(_env)
                                        dagger._last_policy_importance_pcd_base = _input[_env].detach().clone()
                                        dagger._last_policy_point_importance_step = int(step)
                                        _base_idx = base_env._robot_base_body_link_idx
                                        dagger._last_policy_importance_base_pos_w = (
                                            base_env.robot.data.body_pos_w[_env, _base_idx].detach().clone()
                                        )
                                        dagger._last_policy_importance_base_quat_w = (
                                            base_env.robot.data.body_quat_w[_env, _base_idx].detach().clone()
                                        )
                            _pcd_positions = [i for i, lb in enumerate(_attn["token_labels"]) if lb == "pointcloud"]
                            _grouped = {}
                            for _output_name, _scores_list in _attr_rows:
                                _scores = torch.as_tensor(_scores_list, dtype=torch.float32)
                                _groups = {"handle": 0.0, "panel": 0.0, "wall": 0.0, "robot": 0.0, "other_cloud": 0.0}
                                _offset = 0
                                for _key, (_xyz, _weights) in _attn.get("pcd", {}).items():
                                    _n = int(_xyz.shape[0])
                                    _inds = _pcd_positions[_offset:_offset + _n]
                                    _local = _scores[_inds]
                                    _masks = region_masks.get(_key, {}).get("region", torch.full((_n,), -1, dtype=torch.long))
                                    for _rid, _rname in ((0, "handle"), (1, "panel"), (2, "wall"), (3, "robot")):
                                        _groups[_rname] += float(_local[_masks == _rid].sum())
                                    _groups["other_cloud"] += float(_local[~((_masks == 0) | (_masks == 1) | (_masks == 2) | (_masks == 3))].sum())
                                    _offset += _n
                                for _i, _label in enumerate(_attn["token_labels"]):
                                    if _label == "pointcloud":
                                        continue
                                    if _label.startswith("q_arm"):
                                        _name = "franka"
                                    elif _label.startswith("base_vel"):
                                        _name = "base_vel"
                                    elif _label.startswith("target_err_arm"):
                                        _name = "target_err_arm"
                                    elif _label.startswith("aux_handle_pos"):
                                        _name = "aux_handle_pos"
                                    else:
                                        _name = "other_state"
                                    _groups[_name] = _groups.get(_name, 0.0) + float(_scores[_i])
                                _grouped[_output_name] = _groups
                            attn_state["output_attribution_snapshots"].append((int(step), _grouped))
                            attn_state["memory"].pop("tensor", None)
                        region_mass = {}
                        pcd_positions = [i for i, lb in enumerate(_attn["token_labels"]) if lb == "pointcloud"]
                        for qname, qweights in _attn.get("query_attn", {}).items():
                            vals = {"handle": 0.0, "panel": 0.0, "wall": 0.0, "robot": 0.0}
                            offset = 0
                            for key, (_xyz, _aw) in _attn.get("pcd", {}).items():
                                n = int(_xyz.shape[0])
                                if offset + n > len(pcd_positions):
                                    break
                                inds = pcd_positions[offset:offset + n]
                                w = qweights[inds].float()
                                masks = region_masks.get(key, {})
                                vals["handle"] += float(w[masks.get("region", torch.ones(n, dtype=torch.long)) == 0].sum())
                                vals["panel"] += float(w[masks.get("region", torch.ones(n, dtype=torch.long)) == 1].sum())
                                vals["wall"] += float(w[masks.get("region", torch.ones(n, dtype=torch.long)) == 2].sum())
                                vals["robot"] += float(w[masks.get("region", torch.ones(n, dtype=torch.long)) == 3].sum())
                                offset += n
                            region_mass[qname] = vals
                        attn_state["snapshots"].append(
                            (int(step), _attn["token_labels"], _attn["token_attn"], _attn["pcd"])
                        )
                        attn_state["query_snapshots"].append(
                            (int(step), _attn["token_labels"], _attn.get("query_attn", {}))
                        )
                        attn_state["region_snapshots"].append((int(step), region_mass))
                        print(
                            f"[ATTN] step {int(step):04d} env {attn_state['env_id']} | top: "
                            + attn_state["format"](_attn["token_labels"], _attn["token_attn"], top_k=8)
                        )
                        for _query_name, _query_weights in _attn.get("query_attn", {}).items():
                            print(
                                f"[ATTN][{_query_name}] step {int(step):04d} env {attn_state['env_id']} | top: "
                                + attn_state["format"](_attn["token_labels"], _query_weights, top_k=8)
                            )
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
                if viser_recorders is not None:
                    # The policy/depth point clouds were sampled from the current observation,
                    # so capture the matching robot and door pose before advancing physics.
                    # Recording after env.step paired those clouds with the next base pose and
                    # made the world-frame Viser clouds appear detached/lagged during base motion.
                    for _recorder in viser_recorders:
                        if bool(active[_recorder["env_id"]].item()):
                            _append_viser_frame(_recorder, viser_meta, base_env, dagger, student_output)
                if replay_runs is not None:
                    # Save BASE-FRAME (robot-frame) actions so deployment applies them
                    # directly (clamp + real-world scale, no world->robot rotation). The
                    # `actions` stepped into the env are env-frame [wz, vx_w, vy_w]; the base
                    # channels here are [vx_robot, vy_robot, wz]. Arm/hand channels are identical
                    # between the two conventions.
                    if args_cli.use_teacher:
                        replay_actions = dagger._env_actions_to_student_actions(actions)
                    else:
                        replay_actions = student_output["action"][:, 0, :].clone()
                    replay_actions[~active.to(device=replay_actions.device)] = 0.0
                    _run_joint_pos.append(base_env.robot.data.joint_pos.detach().cpu().clone())
                    _run_joint_vel.append(base_env.robot.data.joint_vel.detach().cpu().clone())
                    _run_actions.append(replay_actions.detach().cpu().clone())
                    _run_active.append(active.detach().cpu().clone())
                    # Store base velocity explicitly in the deployment-facing robot frame
                    # [vx, vy, yaw_rate]. Keep the true simulated measurement separate from
                    # the possibly-noisy velocity actually presented to the policy.
                    base_joint_vel = base_env.robot.data.joint_vel[:, base_env._robot_base_dof_idx]
                    measured_base_vel_robot = dagger._env_base_vector_to_robot_frame(base_joint_vel)
                    policy_base_vel_robot = dagger._get_student_base_velocity_vector()
                    commanded_base_vel_robot = replay_actions[:, :3].clamp(-1.0, 1.0) * float(
                        base_env.cfg.base_action_scale
                    )
                    _run_base_velocity_measured_robot.append(measured_base_vel_robot.detach().cpu().clone())
                    _run_base_velocity_policy_input_robot.append(policy_base_vel_robot.detach().cpu().clone())
                    _run_base_velocity_commanded_robot.append(commanded_base_vel_robot.detach().cpu().clone())
                obs, _, _, _, _ = env.step(actions)
                if replay_runs is not None:
                    # ImplicitActuator computes a clipped PD effort estimate; PhysX does
                    # not expose the exact joint-drive effort. Sample the final physics
                    # substep of this policy action, before restoring frozen env states.
                    _base_effort_idx = torch.cat(
                        (base_env._robot_base_xy_dof_idx, base_env._robot_base_rot_dof_idx)
                    )
                    _run_base_effort_applied.append(
                        base_env.robot.data.applied_torque[:, _base_effort_idx].detach().cpu().clone()
                    )
                    _run_base_effort_computed.append(
                        base_env.robot.data.computed_torque[:, _base_effort_idx].detach().cpu().clone()
                    )
                frozen_state.restore()
                if replay_runs is not None:
                    _run_applied_targets.append(base_env.applied_robot_dof_targets.detach().cpu().clone())
                if mode_tracker is not None:
                    current_contact = _get_measured_mode_contact_mask(dagger)

                dagger.temporal_current_time_s = dagger._iteration_to_time_s(step + 1)
                q_after_step = dagger._get_student_proprio_vector().detach().clone()
                target_after_step = dagger._get_implemented_action_vector().detach().clone()
                base_vel_after_step = dagger._get_student_base_velocity_vector().detach().clone()
                base_translation_after_step = dagger._get_student_base_translation_vector().detach().clone()
                dagger._push_temporal_history(
                    timestamp=dagger.temporal_current_time_s,
                    q=q_after_step,
                    target=target_after_step,
                    base_vel=base_vel_after_step,
                    base_translation=base_translation_after_step,
                )

                drift_mask, timeout_mask = _get_eval_done_status(base_env)
                last_metrics = _compute_tracking_metrics(base_env)
                new_timeouts = timeout_mask & active
                new_drifts = drift_mask & active & ~new_timeouts
                newly_finished = new_timeouts | new_drifts
                if torch.any(newly_finished):
                    timed_out |= new_timeouts
                    drifted |= new_drifts
                    active &= ~newly_finished
                    frozen_state.capture(torch.nonzero(newly_finished, as_tuple=False).squeeze(-1))

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
                # Contact-force readout over the ACTIVE envs: franka<->arx (franka arm self-collision
                # vs the arx/x5 camera arm), fingers<->panel, and fingers<->handle.
                _fa = getattr(base_env, "franka_arx_contact_force_norm", None)
                _fp = getattr(base_env, "finger_panel_contact_force_norm", None)
                _fh = getattr(base_env, "finger_handle_contact_force_norm", None)
                if _fa is not None and _fp is not None and _fh is not None and bool(active.any()):
                    _am = active.to(_fa.device)
                    _fa_a, _fp_a, _fh_a = _fa[_am], _fp[_am], _fh[_am]
                    print(
                        f"[CONTACT] run={run_index} step={step} "
                        f"franka<->arx force: mean={_fa_a.mean().item():.3f} max={_fa_a.max().item():.3f} N | "
                        f"fingers<->panel force: mean={_fp_a.mean().item():.3f} max={_fp_a.max().item():.3f} N | "
                        f"fingers<->handle force: mean={_fh_a.mean().item():.3f} max={_fh_a.max().item():.3f} N"
                    )
                # Raw base velocity (physical joint velocities; ignores base_action_scale) for
                # sim<->real comparison. lin speed = ||(base_x_vel, base_y_vel)|| in m/s.
                _base_speed = get_base_lin_speed(base_env)
                if bool(active.any()):
                    _bs = _base_speed[active]
                    _msg = (
                        f"[BASE-VEL] run={run_index} step={step} "
                        f"lin_speed_m_s mean={_bs.mean().item():.3f} max={_bs.max().item():.3f}"
                    )
                    if viser_recorders:
                        _e = viser_recorders[0]["env_id"]
                        _msg += f" viser_env[{_e}]={_base_speed[_e].item():.3f}"
                    print(_msg)

        if replay_runs is not None and _run_joint_pos:
            replay_runs.append({
                "run_index": run_index,
                "joint_pos": torch.stack(_run_joint_pos),       # [T, num_envs, num_joints]
                "joint_vel": torch.stack(_run_joint_vel),       # [T, num_envs, num_joints]
                "actions": torch.stack(_run_actions),           # [T, num_envs, action_dim]
                "applied_targets": torch.stack(_run_applied_targets),  # [T, num_envs, num_dof]
                "active_mask": torch.stack(_run_active),        # [T, num_envs] bool
                "base_velocity_measured_robot": torch.stack(_run_base_velocity_measured_robot),
                "base_velocity_policy_input_robot": torch.stack(_run_base_velocity_policy_input_robot),
                "base_velocity_commanded_robot": torch.stack(_run_base_velocity_commanded_robot),
                "base_effort_applied_estimate": torch.stack(_run_base_effort_applied),
                "base_effort_computed_estimate": torch.stack(_run_base_effort_computed),
            })

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

        # Save at the END OF EACH RUN so artifacts land immediately (no waiting for all runs to
        # finish). Attn jpgs are scoped per run (attn_run<r>_<step>.jpg) then cleared so runs never
        # overwrite each other; the .pt files are re-written with every run completed so far.
        if attn_state is not None and attn_state["snapshots"]:
            _render_attention_jpgs(attn_state, run_index=run_index)
            _render_query_attention_maps(attn_state, run_index=run_index)
            _render_pointcloud_source_attention(attn_state, run_index=run_index)
            _render_joint_output_attributions(attn_state, run_index=run_index)
            attn_state["snapshots"].clear()
            attn_state["query_snapshots"].clear()
            attn_state["region_snapshots"].clear()
            attn_state["output_attribution_snapshots"].clear()
            attn_state["point_count_snapshots"].clear()
        if args_cli.save_replay and replay_runs:
            _save_replay_pt(replay_runs, _replay_joint_names, base_env, args_cli.save_replay)
        if args_cli.viser and viser_recorders is not None:
            _save_viser_recorders(viser_recorders, viser_meta, base_env, viser_dir, run_index)

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
    # NOTE: replay/viser .pt and attn jpgs are now written incrementally at the end of each run
    # (see the per-run save block in the run loop above), so nothing is deferred to here.
    if replay_runs:
        print(
            "[INFO] Replay layout: runs[i]['joint_pos'] shape = [T, num_envs, num_joints], "
            "runs[i]['applied_targets'] shape = [T, num_envs, num_dof]. "
            "Use runs[i]['active_mask'] to filter envs that were still tracking at each step."
        )
    if attn_state is not None:
        # Final-run snapshots were already rendered + cleared per-run; just detach the capture hook.
        attn_state["capture"].remove()
        if attn_state.get("memory_hook") is not None:
            attn_state["memory_hook"].remove()

    dagger._close_viser_debug_tools()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
