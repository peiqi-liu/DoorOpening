"""Script to perform student-teacher distillation"""

import argparse
import os
import pathlib
import sys
import types
from distutils.util import strtobool

import yaml
from isaaclab.app import AppLauncher

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_CFG = SCRIPT_ROOT / "source" / "DoorOpening" / "tasks" / "dooropening" / "agents" / "pcd_transformer_dagger_cfg.yaml"


def _resolve_student_cfg_path(path_value):
    if path_value is None:
        return str(DEFAULT_STUDENT_CFG)
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(SCRIPT_ROOT / path_value)


def _load_student_dagger_defaults(student_cfg_path):
    if not student_cfg_path or not os.path.exists(student_cfg_path):
        return {}

    with open(student_cfg_path, "r") as f:
        student_cfg = yaml.safe_load(f) or {}

    if not isinstance(student_cfg, dict):
        return {}

    dagger_cfg = student_cfg.get("dagger", {})
    return dict(dagger_cfg) if isinstance(dagger_cfg, dict) else {}


def _get_base_env(env):
    return getattr(env, "unwrapped", getattr(env, "env", env))


def _configure_rollout_env_mode(env, play_policy):
    """Match the RL Games train/play env semantics used by the reference scripts."""
    base_env = _get_base_env(env)
    ref_motion_lib = getattr(base_env, "ref_motion_lib", None)
    if ref_motion_lib is not None:
        ref_motion_lib.reset_from_start = bool(play_policy)
    if hasattr(base_env, "early_stopping"):
        base_env.early_stopping = not bool(play_policy)
    if play_policy:
        _patch_play_mode_done_tensor(base_env)
    return base_env


def _patch_play_mode_done_tensor(base_env):
    """Normalize custom env done outputs so play mode still satisfies tensor-based reward code."""
    if not hasattr(base_env, "_get_dones") or getattr(base_env, "_play_mode_done_tensor_patch", False):
        return

    original_get_dones = base_env._get_dones

    def _get_dones_with_tensor_killed(self):
        is_killed, timed_out = original_get_dones()
        if isinstance(is_killed, bool):
            is_killed = torch.zeros_like(timed_out, dtype=torch.bool)
        return is_killed, timed_out

    base_env._get_dones = types.MethodType(_get_dones_with_tensor_killed, base_env)
    base_env._play_mode_done_tensor_patch = True


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=5000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--teacher", type=str, default=None, help="Teacher checkpoint to use")
parser.add_argument("--play_policy", action="store_true", default=False, help="Play a distilled policy.")
# parser.add_argument("--data_aug", action="store_true", default=False, help="Whether to use data augmentation for student")
parser.add_argument("--student_cfg", type=str, default=None, help="Student config YAML to use.")
parser.add_argument("--student_ckpt", type=str, default=None, help="Student checkpoint to resume or evaluate.")
parser.add_argument("--teacher_cfg", type=str, default=None, help="Teacher RL-Games config YAML to use.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=None,
    nargs="?",
    const=True,
    help="Enable or disable Weights and Biases tracking. Defaults to dagger.wandb.enabled from the student YAML.",
)
parser.add_argument(
    "--pointcloud_source",
    type=str,
    choices=["sampler", "depth", "lidar"],
    default=None,
    help="Source used to build the student pointcloud observation. Defaults to dagger.pointcloud_source in the student YAML.",
)
parser.add_argument(
    "--viser_live",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable or disable live Viser streaming. Defaults to dagger.viser.enabled in the student YAML.",
)
parser.add_argument(
    "--viser_serializer",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable or disable serialized `.viser` episode replays. Defaults to dagger.viser.serializer.enabled.",
)
parser.add_argument(
    "--viser_raw",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable or disable raw `.pt` episode replays. Defaults to dagger.viser.raw.enabled.",
)
parser.add_argument(
    "--viser_env_id",
    type=int,
    default=None,
    help="Environment index used for live/replay Viser capture. Defaults to dagger.viser.env_id.",
)
parser.add_argument(
    "--viser_update_interval",
    type=int,
    default=None,
    help="Iteration interval for live Viser streaming. Defaults to dagger.viser.update_interval.",
)
parser.add_argument(
    "--viser_serializer_path",
    type=str,
    default=None,
    help="Base output path for serialized `.viser` replays.",
)
parser.add_argument(
    "--viser_raw_path",
    type=str,
    default=None,
    help="Base output path for raw `.pt` replays.",
)
parser.add_argument(
    "--viser_raw_interval",
    type=int,
    default=None,
    help="Iteration interval for periodic raw `.pt` replay snapshots. Defaults to dagger.viser.raw_interval.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
student_cfg_path = _resolve_student_cfg_path(args_cli.student_cfg)
student_dagger_defaults = _load_student_dagger_defaults(student_cfg_path)
pointcloud_source = str(student_dagger_defaults.get("pointcloud_source", "sampler")).lower()
if args_cli.pointcloud_source is not None:
    pointcloud_source = args_cli.pointcloud_source
# enable cameras only when we need rendered outputs from the simulator
if args_cli.video or pointcloud_source == "depth":
    args_cli.enable_cameras = True


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import math
from datetime import datetime
import torch
import torch.distributed as dist

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


from DoorOpening.distillation.pcd_dagger import Dagger
from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets

import DoorOpening.tasks # noqa: F401


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """ Performs distillation. """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_distributed = args_cli.distributed or world_size > 1

    if use_distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # parse configuration
    # env_cfg = parse_env_cfg(
    #     args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    # )
    # agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if use_distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    parent_path = str(SCRIPT_ROOT)
    agent_cfg_folder = "source/DoorOpening/tasks/dooropening/agents"

    def resolve_path(path_value, default_rel_path):
        if path_value is None:
            return os.path.join(parent_path, default_rel_path)
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(parent_path, path_value)

    def resolve_checkpoint(path_value, default_rel_path=None):
        if path_value is None:
            if default_rel_path is None:
                return None
            return os.path.join(parent_path, default_rel_path)
        if os.path.isabs(path_value):
            return path_value
        repo_relative = os.path.join(parent_path, path_value)
        if os.path.exists(repo_relative):
            return repo_relative
        return os.path.join(parent_path, "pretrained_ckpts", path_value)

    student_cfg = resolve_path(
        args_cli.student_cfg,
        os.path.join(agent_cfg_folder, "pcd_transformer_dagger_cfg.yaml"),
    )
    student_dagger_cfg = _load_student_dagger_defaults(student_cfg)

    teacher_cfg = resolve_path(
        args_cli.teacher_cfg,
        os.path.join(agent_cfg_folder, "rl_games_ppo_cfg.yaml"),
    )

    dagger_runtime_cfg = dict(student_dagger_cfg)
    wandb_cfg = {}
    runtime_wandb_cfg = dagger_runtime_cfg.pop("wandb", {})
    if isinstance(runtime_wandb_cfg, dict):
        wandb_cfg.update(runtime_wandb_cfg)
    if args_cli.max_iterations is not None:
        dagger_runtime_cfg["num_iters"] = args_cli.max_iterations
    else:
        dagger_runtime_cfg.setdefault("num_iters", 100_000)
    if args_cli.pointcloud_source is not None:
        dagger_runtime_cfg["pointcloud_source"] = args_cli.pointcloud_source
    else:
        dagger_runtime_cfg.setdefault("pointcloud_source", "sampler")
    dagger_runtime_cfg["pointcloud_source"] = str(dagger_runtime_cfg["pointcloud_source"]).lower()
    if dagger_runtime_cfg["pointcloud_source"] not in {"sampler", "depth", "lidar"}:
        raise ValueError(
            "dagger.pointcloud_source must be one of ['sampler', 'depth', 'lidar'], "
            f"got '{dagger_runtime_cfg['pointcloud_source']}'."
        )

    if "reset_progress_total" in dagger_runtime_cfg:
        env_cfg.reset_progress_total = dagger_runtime_cfg["reset_progress_total"]
    if "adr_reset_progress_total" in dagger_runtime_cfg:
        env_cfg.adr_reset_progress_total = dagger_runtime_cfg["adr_reset_progress_total"]
    else:
        # Distillation default: ADR schedule progresses twice as fast as reset curriculum.
        env_cfg.adr_reset_progress_total = 0.5 * float(env_cfg.reset_progress_total)

    viser_cfg = dagger_runtime_cfg.get("viser", {})
    if not isinstance(viser_cfg, dict):
        viser_cfg = {}
    serializer_cfg = viser_cfg.get("serializer", {})
    if not isinstance(serializer_cfg, dict):
        serializer_cfg = {}
    raw_cfg = viser_cfg.get("raw", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    if args_cli.viser_live is not None:
        viser_cfg["enabled"] = args_cli.viser_live
    if args_cli.viser_update_interval is not None:
        viser_cfg["update_interval"] = max(1, int(args_cli.viser_update_interval))
    else:
        viser_cfg.setdefault("update_interval", 1)
    if args_cli.viser_env_id is not None:
        viser_cfg["env_id"] = int(args_cli.viser_env_id)
    if args_cli.viser_serializer is not None:
        serializer_cfg["enabled"] = args_cli.viser_serializer
        print(f"Viser serializer enabled: {args_cli.viser_serializer}")
    if args_cli.viser_raw is not None:
        raw_cfg["enabled"] = args_cli.viser_raw
        print(f"Viser raw enabled: {args_cli.viser_raw}")
    if args_cli.viser_serializer_path is not None:
        serializer_cfg["path"] = args_cli.viser_serializer_path
    if args_cli.viser_raw_path is not None:
        raw_cfg["path"] = args_cli.viser_raw_path
    if args_cli.viser_raw_interval is not None:
        viser_cfg["raw_interval"] = max(1, int(args_cli.viser_raw_interval))
    viser_cfg["serializer"] = serializer_cfg
    viser_cfg["raw"] = raw_cfg
    dagger_runtime_cfg["viser"] = viser_cfg

    if dagger_runtime_cfg["pointcloud_source"] == "depth":
        env_cfg.pointcloud_render_mode = "depth"
    elif dagger_runtime_cfg["pointcloud_source"] == "lidar":
        env_cfg.pointcloud_render_mode = "lidar"
    else:
        env_cfg.pointcloud_render_mode = "none"
    env_cfg.enable_pointcloud_camera = env_cfg.pointcloud_render_mode == "depth"

    # Determine teacher checkpoint path
    teacher_ckpt = None if args_cli.play_policy else resolve_checkpoint(
        args_cli.teacher,
        "pretrained_ckpts/door_opening.pth",
    )
    student_ckpt = resolve_checkpoint(args_cli.student_ckpt)

    if rank == 0:
        train_dir = "runs"
        default_project_name = "DoorOpening-Distillation"
        experiment_name = default_project_name + datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        experiment_dir = os.path.join(train_dir, experiment_name)
        nn_dir = os.path.join(experiment_dir, "nn")
        summaries_dir = os.path.join(experiment_dir, "summaries")
        default_wandb_project = default_project_name
        default_wandb_name = experiment_name

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(nn_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
    else:
        summaries_dir = None
        nn_dir = None
        default_wandb_project = None
        default_wandb_name = None

    wandb_enabled = bool(wandb_cfg.get("enabled", False)) if args_cli.track is None else args_cli.track
    wandb_project = (
        args_cli.wandb_project_name
        if args_cli.wandb_project_name is not None
        else wandb_cfg.get("project", default_wandb_project)
    )
    wandb_entity = (
        args_cli.wandb_entity
        if args_cli.wandb_entity is not None
        else wandb_cfg.get("entity")
    )
    wandb_name = (
        args_cli.wandb_name
        if args_cli.wandb_name is not None
        else wandb_cfg.get("name", default_wandb_name)
    )
    wandb_cfg["enabled"] = wandb_enabled
    if wandb_project is not None:
        wandb_cfg["project"] = wandb_project
    if wandb_entity is not None:
        wandb_cfg["entity"] = wandb_entity
    if wandb_name is not None:
        wandb_cfg["name"] = wandb_name

    if wandb_enabled and rank == 0 and wandb_entity is None:
        raise ValueError("Weights and Biases entity must be specified for tracking.")

    if rank == 0:
        print(f"Distillation reset_progress_total: {env_cfg.reset_progress_total}")
        print(f"Distillation adr_reset_progress_total: {env_cfg.adr_reset_progress_total}")

    # Serialize URDF-to-USD conversion across ranks before all workers build the same shared assets.
    preconvert_shared_urdf_assets()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ov_env = _configure_rollout_env_mode(env, args_cli.play_policy)
    if rank == 0:
        ref_motion_lib = getattr(ov_env, "ref_motion_lib", None)
        reset_from_start = getattr(ref_motion_lib, "reset_from_start", None)
        early_stopping = getattr(ov_env, "early_stopping", None)
        print(
            "[INFO] Distillation rollout mode: "
            f"{'play' if args_cli.play_policy else 'train'} "
            f"(reset_from_start={reset_from_start}, early_stopping={early_stopping})"
        )

    if args_cli.video and rank == 0:
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "distillation"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during distillation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    dagger_config = {
        "student": {
            "cfg": student_cfg,
            "ckpt": student_ckpt,
            # "data_aug": args_cli.data_aug,
        },
        "teacher": {
            "cfg": teacher_cfg,
            "ckpt": teacher_ckpt,
            "obs_type": "policy",
        },
        "play_policy": args_cli.play_policy,
        "dagger": dagger_runtime_cfg,
        "wandb": wandb_cfg,
    }

    dagger = Dagger(env, dagger_config, summaries_dir=summaries_dir, nn_dir=nn_dir)
    dagger.distill()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
