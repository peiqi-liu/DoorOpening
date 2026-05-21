"""Evaluate a distilled multi-door point-cloud policy.

The runner evaluates each vectorized env once. Envs that drift too far from
their reference motion are frozen at the failure state; envs that reach the end
of the reference motion are counted as timed out and frozen there.
"""

import argparse
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


def _load_student_dagger_defaults(student_cfg_path):
    if not student_cfg_path or not os.path.exists(student_cfg_path):
        return {}
    with open(student_cfg_path, "r", encoding="utf-8") as f:
        student_cfg = yaml.safe_load(f) or {}
    if not isinstance(student_cfg, dict):
        return {}
    dagger_cfg = student_cfg.get("dagger", {})
    return dict(dagger_cfg) if isinstance(dagger_cfg, dict) else {}


parser = argparse.ArgumentParser(description="Evaluate a distilled DooropeningMulti point-cloud policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in env steps.")
parser.add_argument("--video_folder", type=str, default=None, help="Optional folder for recorded videos.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to evaluate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--student_cfg", type=str, default=None, help="Student config YAML to use.")
parser.add_argument("--student_ckpt", type=str, required=True, help="Student checkpoint to evaluate.")
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
parser.add_argument("--print_interval", type=int, default=100, help="Print rollout stats every N env steps.")
parser.add_argument(
    "--key_body_pos_threshold",
    type=float,
    default=None,
    help="Drift limit in meters. Defaults to env reset_key_body_pos_delta_max.",
)
parser.add_argument(
    "--key_body_quat_threshold",
    type=float,
    default=None,
    help="Drift limit in radians. Defaults to env reset_key_body_quat_delta_max.",
)
parser.add_argument(
    "--door_joint_pos_threshold",
    type=float,
    default=None,
    help="Door joint drift limit in radians/meters. Defaults to env reset_door_joint_pos_delta_max.",
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
    (
        key_body_pos_err,
        key_body_quat_err,
        door_err,
        *_,
    ) = compute_tracking_error(
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
    metrics = {
        "key_body_pos": torch.sqrt(torch.clamp(key_body_pos_err, min=0.0)),
        "key_body_quat": torch.sqrt(torch.clamp(key_body_quat_err, min=0.0)),
        "door_joint_pos": torch.sqrt(torch.clamp(door_err, min=0.0)),
    }
    drifted = (
        (metrics["key_body_pos"] > thresholds["key_body_pos"])
        | (metrics["key_body_quat"] > thresholds["key_body_quat"])
        | (metrics["door_joint_pos"] > thresholds["door_joint_pos"])
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


def _build_student_actions(dagger, iteration):
    student_obs = dagger._build_student_obs(iteration=iteration)
    student_output = dagger._student_forward(student_obs)
    if dagger.has_aux_prediction:
        aux_prediction = dagger._decode_aux_prediction(student_output["aux"].detach())
        dagger.aux_buffer[:] = aux_prediction
    student_actions = student_output["action"][:, 0, :]
    return dagger._student_actions_to_env_actions(student_actions)


def _print_progress(step, active, timed_out, drifted, metrics):
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
    print(
        "[EVAL] "
        f"step={step} active={active_count} timed_out={timeout_count} drifted={drift_count} "
        f"max_key_pos={active_metrics['key_body_pos']:.4f} "
        f"max_key_quat={active_metrics['key_body_quat']:.4f} "
        f"max_door={active_metrics['door_joint_pos']:.4f}"
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
    dagger_config = {
        "student": {
            "cfg": student_cfg_path,
            "ckpt": student_ckpt,
        },
        "teacher": {},
        "play_policy": True,
        "dagger": dagger_runtime_cfg,
        "wandb": {"enabled": False},
    }
    dagger = Dagger(env, dagger_config, summaries_dir=summaries_dir, nn_dir=nn_dir)
    dagger.student_model_ddp.eval()

    if hasattr(base_env, "common_step_counter"):
        base_env.common_step_counter = 0
    if hasattr(base_env, "_rlgames_env_frames"):
        base_env._rlgames_env_frames = 0

    obs, _ = env.reset()
    _reset_dagger_rollout_state(dagger)
    frozen_state = FrozenEnvState(base_env)
    _install_freeze_patch(base_env, frozen_state)

    thresholds = _resolve_drift_thresholds(base_env)
    print("[INFO] Drift thresholds:", thresholds)

    active = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    timed_out = torch.zeros_like(active)
    drifted = torch.zeros_like(active)
    last_metrics = {
        "key_body_pos": torch.zeros(base_env.num_envs, dtype=torch.float32, device=base_env.device),
        "key_body_quat": torch.zeros(base_env.num_envs, dtype=torch.float32, device=base_env.device),
        "door_joint_pos": torch.zeros(base_env.num_envs, dtype=torch.float32, device=base_env.device),
    }

    step = 0
    while simulation_app.is_running():
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break
        if not torch.any(active):
            break

        with torch.inference_mode():
            frozen_state.restore()
            actions = _build_student_actions(dagger, iteration=step)
            actions[~active.to(device=actions.device)] = 0.0
            obs, _, _, _, _ = env.step(actions)
            frozen_state.restore()

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

        step += 1
        if args_cli.print_interval > 0 and (step == 1 or step % args_cli.print_interval == 0 or not torch.any(active)):
            _print_progress(step, active, timed_out, drifted, last_metrics)

    active_count = int(active.sum().detach().cpu().item())
    timeout_count = int(timed_out.sum().detach().cpu().item())
    drift_count = int(drifted.sum().detach().cpu().item())
    print(
        "[RESULT] "
        f"steps={step} num_envs={base_env.num_envs} "
        f"timed_out_envs={timeout_count} drifted_envs={drift_count} active_envs={active_count}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
