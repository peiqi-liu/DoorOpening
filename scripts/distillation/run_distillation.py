"""Script to perform student-teacher distillation"""

import argparse
import os
import pathlib
import sys
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
    choices=["sampler", "depth"],
    default=None,
    help="Source used to build the student pointcloud observation. Defaults to dagger.pointcloud_source in the student YAML.",
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

    if "reset_progress_total" in dagger_runtime_cfg:
        env_cfg.reset_progress_total = dagger_runtime_cfg["reset_progress_total"]

    viser_cfg = dagger_runtime_cfg.get("viser", {})
    if not isinstance(viser_cfg, dict):
        viser_cfg = {}
    viser_cfg["update_interval"] = int(args_cli.video_interval)
    viser_record_cfg = viser_cfg.get("record", {})
    if not isinstance(viser_record_cfg, dict):
        viser_record_cfg = {}
    viser_record_cfg["interval"] = int(args_cli.video_interval)
    viser_cfg["record"] = viser_record_cfg
    dagger_runtime_cfg["viser"] = viser_cfg

    env_cfg.enable_pointcloud_camera = dagger_runtime_cfg["pointcloud_source"] == "depth"

    # Determine teacher checkpoint path
    teacher_ckpt = None if args_cli.play_policy else resolve_checkpoint(
        args_cli.teacher,
        "pretrained_ckpts/door_opening.pth",
    )
    student_ckpt = resolve_checkpoint(args_cli.student_ckpt)

    if rank == 0:
        train_dir = "runs"
        default_project_name = "DoorOpening-Distillation"
        experiment_name = default_project_name + datetime.now().strftime("_%d-%H-%M-%S")
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ov_env = env.env
    print(ov_env)

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
