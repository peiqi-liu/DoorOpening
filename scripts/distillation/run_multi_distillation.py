"""Script to perform student-teacher distillation"""

import argparse
import os
import pathlib
import sys
import time
from datetime import datetime
from distutils.util import strtobool

# --- Hang diagnostics -------------------------------------------------------------------
# A rank can stall inside Isaac Sim init (e.g. PhysX/CUDA GPU scene setup in sim.reset())
# with no error. `kill -USR1 <pid>` dumps the Python stack of every thread of that process
# on demand (writes to stderr / the job's .err file). Passive; changes no runtime behavior.
import faulthandler as _faulthandler
import signal as _signal

try:
    _faulthandler.register(_signal.SIGUSR1, all_threads=True, chain=False)
except Exception as _exc:  # pragma: no cover - best-effort diagnostics only
    print(f"[WARN] could not register SIGUSR1 stack dumper: {_exc}")
# ----------------------------------------------------------------------------------------

import yaml
from isaaclab.app import AppLauncher

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_CFG = SCRIPT_ROOT / "source" / "DoorOpening" / "tasks" / "dooropening" / "agents" / "pcd_transformer_dagger_cfg.yaml"
DEFAULT_FINETUNE_STUDENT_CFG = (
    SCRIPT_ROOT / "source" / "DoorOpening" / "tasks" / "dooropening" / "agents" / "pcd_transformer_dagger_finetune_cfg.yaml"
)


def _resolve_student_cfg_path(path_value, default_path):
    if path_value is None:
        return str(default_path)
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


def _stagger_isaac_sim_startup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = os.environ.get("RANK", str(local_rank))
    stagger_s = float(os.environ.get("ISAAC_STARTUP_STAGGER_S", "5"))
    delay_s = max(0.0, stagger_s) * max(0, local_rank)
    if delay_s <= 0.0:
        return

    print(
        f"[INFO][rank {rank}] Staggering Isaac Sim startup by {delay_s:.1f}s "
        f"({stagger_s:.1f}s * local_rank {local_rank}).",
        flush=True,
    )
    time.sleep(delay_s)


def _get_base_env(env):
    return getattr(env, "unwrapped", getattr(env, "env", env))


def _configure_rollout_env_mode(env):
    """Match the RL Games train/play env semantics used by the reference scripts."""
    base_env = _get_base_env(env)
    ref_motion_lib = getattr(base_env, "ref_motion_lib", None)
    if ref_motion_lib is not None:
        ref_motion_lib.reset_from_start = False
    if hasattr(base_env, "early_stopping"):
        base_env.early_stopping = True
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


def _expected_viser_stream_names(pointcloud_source):
    pointcloud_source = str(pointcloud_source).lower()
    streams = ["ground_truth"]
    if pointcloud_source in {"lidar", "both"}:
        streams.append("robot_lidar_obs")
    if pointcloud_source in {"sampler", "depth", "both"}:
        streams.append("robot_depth_cam_obs")
    streams.append("policy_input")
    return streams


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=5000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--video_ranks",
    type=str,
    choices=["all", "rank0"],
    default="rank0",
    help="Which distributed ranks record videos. Use 'all' to record one rank-specific video folder per GPU.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--asset-preconvert-timeout-s",
    type=float,
    default=21600.0,
    help="Timeout for rank-0 shared URDF preconversion in distributed multi-door runs.",
)
parser.add_argument(
    "--asset-preconvert-poll-interval-s",
    type=float,
    default=5.0,
    help="Polling interval while nonzero ranks wait for shared URDF preconversion.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--teacher", type=str, default=None, help="Teacher checkpoint to use")
parser.add_argument("--teacher-partnetv5", "--teacher_partnetv5", dest="teacher_partnetv5", type=str, default=None, help="Teacher checkpoint for PartNetv5.")
parser.add_argument("--teacher-partnetv6", "--teacher_partnetv6", dest="teacher_partnetv6", type=str, default=None, help="Teacher checkpoint for PartNetv6.")
parser.add_argument("--teacher-partnetv7", "--teacher_partnetv7", dest="teacher_partnetv7", type=str, default=None, help="Teacher checkpoint for PartNetv7.")
parser.add_argument("--teacher-partnetv8", "--teacher_partnetv8", dest="teacher_partnetv8", type=str, default=None, help="Teacher checkpoint for PartNetv8.")
parser.add_argument(
    "--finetune",
    action="store_true",
    default=False,
    help="Resume supervised distillation from a student checkpoint with finetune-oriented defaults.",
)
# parser.add_argument("--data_aug", action="store_true", default=False, help="Whether to use data augmentation for student")
parser.add_argument("--student_cfg", type=str, default=None, help="Student config YAML to use.")
parser.add_argument("--student_ckpt", type=str, default=None, help="Student checkpoint to resume from.")
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
    choices=["sampler", "depth", "lidar", "both"],
    default=None,
    help="Source used to build the student pointcloud observation. Defaults to dagger.pointcloud_source in the student YAML.",
)
parser.add_argument(
    "--viser",
    "--viser-raw",
    "--viser_raw",
    dest="viser_raw",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable or disable raw `.pt` episode replays. Defaults to dagger.viser.raw.enabled.",
)
parser.add_argument(
    "--viser-raw-path",
    "--viser_raw_path",
    dest="viser_raw_path",
    type=str,
    default=None,
    help="Base output path for raw `.pt` replays.",
)
parser.add_argument(
    "--viser-raw-interval",
    "--viser_raw_interval",
    dest="viser_raw_interval",
    type=int,
    default=None,
    help="Iteration interval for periodic raw `.pt` replay snapshots. Defaults to dagger.viser.raw_interval.",
)
parser.add_argument(
    "--door-families",
    "--door_families",
    dest="door_families",
    type=str,
    default=None,
    help="Comma-separated multi-door family list to train on, e.g. PartNetv9,PartNetv10.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
default_student_cfg_path = DEFAULT_FINETUNE_STUDENT_CFG if args_cli.finetune else DEFAULT_STUDENT_CFG
student_cfg_path = _resolve_student_cfg_path(args_cli.student_cfg, default_student_cfg_path)
student_dagger_defaults = _load_student_dagger_defaults(student_cfg_path)
selected_door_families = _normalize_family_selection(
    args_cli.door_families if args_cli.door_families is not None else student_dagger_defaults.get("door_families")
)
if selected_door_families is not None:
    os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = ",".join(selected_door_families)
    print(f"[INFO] Using multi-door families: {selected_door_families}")
pointcloud_source = str(student_dagger_defaults.get("pointcloud_source", "both")).lower()
if args_cli.pointcloud_source is not None:
    pointcloud_source = args_cli.pointcloud_source
# The multi-door distillation stack now simulates depth/lidar from cached pointclouds,
# so Isaac camera rendering should stay disabled unless RGB video recording is requested.
if args_cli.video:
    args_cli.enable_cameras = True
else:
    os.environ["ENABLE_CAMERAS"] = "0"
    args_cli.enable_cameras = False


_stagger_isaac_sim_startup()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# --- Correct the distributed CPU-thread budget (prevents PhysX physics-load deadlock) ----
# IsaacLab's AppLauncher sizes the carb.tasking / USD-work(TBB) / OpenBLAS thread pools from
# os.cpu_count() // WORLD_SIZE. But os.cpu_count() reports ALL node logical cores (e.g. 128),
# ignoring the SLURM cpus-per-task cpuset. On a 15-CPU allocation that gives ~64 threads/rank
# -> ~280 threads fighting for 15 CPUs -> a priority-inversion futex deadlock inside PhysX
# force_load_physics_from_usd() (confirmed via live /proc: every worker thread in futex wait).
# Patch os.cpu_count() to report the ACTUALLY-allocated CPU count so the pools are sized right.
_ORIG_CPU_COUNT = os.cpu_count


def _allocated_cpu_count():
    # 1) Explicit allocation hints: present only on HPC/SLURM or containers (where
    #    os.cpu_count() overcounts). Absent on a normal workstation -> fall through.
    for _env in ("ISAACLAB_ALLOCATED_CPUS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        _v = os.environ.get(_env)
        if _v and _v.isdigit() and int(_v) > 0:
            return int(_v)
    # 2) Linux cpuset affinity: correct on bare-metal/non-HPC and honors cgroup/Docker limits.
    _getaffinity = getattr(os, "sched_getaffinity", None)
    if _getaffinity is not None:
        try:
            _n = len(_getaffinity(0))
            if _n > 0:
                return _n
        except OSError:
            pass
    # 3) Last resort (e.g. macOS/Windows, where sched_getaffinity is missing): full core count.
    return _ORIG_CPU_COUNT() or 1


os.cpu_count = _allocated_cpu_count
print(f"[INFO] os.cpu_count() patched: reporting {os.cpu_count()} allocated CPUs (node reports {_ORIG_CPU_COUNT()}) to size Isaac Sim thread pools.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import math
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


from DoorOpening.distillation.multi_pcd_dagger import Dagger
from DoorOpening.assets.door.multi_door_cfg import ALL_DOOR_CONFIGS as MULTI_DOOR_CONFIGS
from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES as MULTI_TEACHER_FAMILY_NAMES
from DoorOpening.assets.door.multi_door_cfg import asset_paths as MULTI_DOOR_ASSET_PATHS
from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets

import DoorOpening.tasks # noqa: F401


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """ Performs distillation. """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_distributed = args_cli.distributed or world_size > 1

    if use_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
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

    def resolve_multi_teacher_checkpoints():
        cli_values = {
            "PartNetv5": args_cli.teacher_partnetv5,
            "PartNetv5_plus": args_cli.teacher_partnetv5,
            "PartNetv5_plusplus": args_cli.teacher_partnetv5,
            "PartNetv5_pro": args_cli.teacher_partnetv5,
            "PartNetv6": args_cli.teacher_partnetv6,
            "PartNetv6_plus": args_cli.teacher_partnetv6,
            "PartNetv6_plusplus": args_cli.teacher_partnetv6,
            "PartNetv6_pro": args_cli.teacher_partnetv6,
            "PartNetv7": args_cli.teacher_partnetv7,
            "PartNetv7_plus": args_cli.teacher_partnetv7,
            "PartNetv7_plusplus": args_cli.teacher_partnetv7,
            "PartNetv7_pro": args_cli.teacher_partnetv7,
            "PartNetv8": args_cli.teacher_partnetv8,
            "PartNetv8_plus": args_cli.teacher_partnetv8,
            "PartNetv8_plusplus": args_cli.teacher_partnetv8,
            "PartNetv8_pro": args_cli.teacher_partnetv8,
        }
        any_family_cli = any(value is not None for value in cli_values.values())
        discovered_values = {}
        for family_name in MULTI_TEACHER_FAMILY_NAMES:
            default_family_ckpt = os.path.join(
                parent_path,
                "source",
                "DoorOpening",
                "assets",
                "door",
                family_name,
                "door_opening.pth",
            )
            configured_value = cli_values.get(family_name)
            if configured_value is not None:
                discovered_values[family_name] = resolve_checkpoint(configured_value)
            elif os.path.exists(default_family_ckpt):
                discovered_values[family_name] = default_family_ckpt

        if any_family_cli:
            missing = [family_name for family_name in MULTI_TEACHER_FAMILY_NAMES if family_name not in discovered_values]
            if missing:
                raise ValueError(
                    "Multi-teacher distillation needs a checkpoint for every PartNet family. "
                    f"Missing: {missing}."
                )
            return discovered_values, None

        if len(discovered_values) == len(MULTI_TEACHER_FAMILY_NAMES):
            return discovered_values, None

        teacher_ckpt = resolve_checkpoint(
            args_cli.teacher,
            "pretrained_ckpts/door_opening.pth",
        )
        return None, teacher_ckpt

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
        dagger_runtime_cfg.setdefault("pointcloud_source", "both")
    dagger_runtime_cfg["pointcloud_source"] = str(dagger_runtime_cfg["pointcloud_source"]).lower()
    if dagger_runtime_cfg["pointcloud_source"] not in {"sampler", "depth", "lidar", "both"}:
        raise ValueError(
            "dagger.pointcloud_source must be one of ['sampler', 'depth', 'lidar', 'both'], "
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
    raw_cfg = viser_cfg.get("raw", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    if args_cli.viser_raw is not None:
        raw_cfg["enabled"] = args_cli.viser_raw
        print(f"Viser raw enabled: {args_cli.viser_raw}")
    if args_cli.viser_raw_path is not None:
        raw_cfg["path"] = args_cli.viser_raw_path
    if args_cli.viser_raw_interval is not None:
        raw_cfg["save_interval"] = max(1, int(args_cli.viser_raw_interval))
    viser_cfg["raw"] = raw_cfg
    dagger_runtime_cfg["viser"] = viser_cfg
    if bool(raw_cfg.get("enabled", False)):
        expected_streams = _expected_viser_stream_names(dagger_runtime_cfg["pointcloud_source"])
        print(
            "[INFO] Multi-distillation Viser raw streams: "
            + ", ".join(expected_streams)
        )
        if raw_cfg.get("path") is not None:
            print(f"[INFO] Multi-distillation Viser raw path override: {raw_cfg['path']}")

    env_cfg.pointcloud_render_mode = "none"
    env_cfg.enable_pointcloud_camera = False

    # Determine teacher checkpoint paths. Multi mode prefers one checkpoint per PartNet folder,
    # but keeps single-teacher fallback for smoke tests and old checkpoints.
    multi_teacher_ckpts, teacher_ckpt = resolve_multi_teacher_checkpoints()
    student_ckpt = resolve_checkpoint(args_cli.student_ckpt)
    if args_cli.finetune and student_ckpt is None:
        raise ValueError("--finetune requires --student_ckpt so the student can resume from a trained checkpoint.")

    train_dir = "runs"
    default_wandb_project = "DoorOpening-Distillation-Finetune" if args_cli.finetune else "DoorOpening-Distillation"
    base_experiment_name = default_wandb_project + datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
    rank_tag = f"rank{rank:03d}_local{local_rank:03d}"
    experiment_name = f"{base_experiment_name}_{rank_tag}" if use_distributed else base_experiment_name
    default_wandb_name = base_experiment_name
    experiment_dir = os.path.join(train_dir, experiment_name)
    nn_dir = os.path.join(experiment_dir, "nn")
    summaries_dir = os.path.join(experiment_dir, "summaries")

    if rank == 0:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(nn_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
        print(f"[INFO][rank {rank}] Distillation output directory: {experiment_dir}")

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

    # Serialize URDF-to-USD conversion across ranks before all workers build the same shared assets.
    preconvert_shared_urdf_assets(
        door_configs=MULTI_DOOR_CONFIGS,
        timeout_s=float(args_cli.asset_preconvert_timeout_s),
        poll_interval_s=float(args_cli.asset_preconvert_poll_interval_s),
        verbose=True,
    )

    # create isaac environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video and (args_cli.video_ranks == "all" or rank == 0) else None,
    )
    ov_env = _configure_rollout_env_mode(env)
    env_asset_indices = getattr(ov_env, "env_asset_indices", None)
    if env_asset_indices is not None and len(env_asset_indices) > 0:
        first_asset_idx = int(env_asset_indices[0].detach().cpu().item())
        first_asset_dir = pathlib.Path(MULTI_DOOR_ASSET_PATHS[first_asset_idx]).resolve().parent
        print(
            f"[INFO][rank {rank}] First assigned door asset idx={first_asset_idx} dir={first_asset_dir}",
            flush=True,
        )
    if rank == 0:
        ref_motion_lib = getattr(ov_env, "ref_motion_lib", None)
        reset_from_start = getattr(ref_motion_lib, "reset_from_start", None)
        early_stopping = getattr(ov_env, "early_stopping", None)
        print(
            "[INFO] Distillation rollout mode: train "
            f"(reset_from_start={reset_from_start}, early_stopping={early_stopping})"
        )

    if args_cli.video and (args_cli.video_ranks == "all" or rank == 0):
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "distillation"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"distillation_{rank_tag}",
            "disable_logger": True,
        }
        print(f"[INFO][rank {rank}] Recording videos during distillation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    elif args_cli.video and rank == 0:
        print("[INFO] Video recording requested for rank0 only; nonzero ranks will not write videos.")

    teacher_config = {
        "cfg": teacher_cfg,
        "obs_type": "policy",
    }
    if multi_teacher_ckpts is not None:
        teacher_config["teachers"] = {
            family_name: {"ckpt": ckpt}
            for family_name, ckpt in multi_teacher_ckpts.items()
        }
    else:
        teacher_config["ckpt"] = teacher_ckpt

    dagger_config = {
        "student": {
            "cfg": student_cfg,
            "ckpt": student_ckpt,
            # "data_aug": args_cli.data_aug,
        },
        "teacher": teacher_config,
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
