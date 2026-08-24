# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from distutils.util import strtobool

# --- Hang diagnostics -------------------------------------------------------------------
# A rank can stall inside Isaac Sim init (e.g. PhysX/CUDA GPU scene setup in sim.reset())
# with no error. These hooks make a stuck rank self-report WHERE it is stuck:
#   * `kill -USR1 <pid>` dumps the Python stack of every thread of that process on demand.
#   * faulthandler.dump_traceback_later() (armed around gym.make() below) auto-dumps the
#     stack if sim init does not finish within HANG_DUMP_SECONDS, even while the main thread
#     is blocked in a native C call. Both write to stderr (the job's .err file).
import faulthandler as _faulthandler
import signal as _signal

_HANG_DUMP_SECONDS = int(os.environ.get("HANG_DUMP_SECONDS", "300"))
try:
    _faulthandler.register(_signal.SIGUSR1, all_threads=True, chain=False)
except Exception as _exc:  # pragma: no cover - best-effort diagnostics only
    print(f"[WARN] could not register SIGUSR1 stack dumper: {_exc}")
# ----------------------------------------------------------------------------------------

from isaaclab.app import AppLauncher


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


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument("--exp-name", type=str, default=None, help="experiment folder name (overrides full_experiment_name in agent config)")
parser.add_argument(
    "--door-families",
    "--door_families",
    "--asset-folders",
    "--asset_folders",
    dest="door_families",
    type=str,
    default=None,
    help="Comma-separated multi-door asset family folders, e.g. PartNetv5_plusplus,PartNetv6_plusplus.",
)
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
selected_door_families = _normalize_family_selection(args_cli.door_families)
if selected_door_families is not None:
    os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = ",".join(selected_door_families)
    print(f"[INFO] Using multi-door families: {selected_door_families}")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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

# Disable UJITSO multi-process collision cooking. Its on-disk datastore + cooking-worker
# subprocesses use file locking; when two distributed ranks cook collision geometry
# concurrently against the same NFS-backed cache, PhysX wedges inside
# force_load_physics_from_usd() (the proven deadlock frame). Forcing in-process cooking
# removes the multi-process + disk-lock machinery at that exact step.
try:
    import carb as _carb

    _carb_settings = _carb.settings.get_settings()
    _carb_settings.set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    _carb_settings.set_int("/persistent/physics/cooking/ujitsoCookingMaxProcessCount", 0)
    print(
        "[INFO] Disabled UJITSO multiprocess collision cooking "
        "(/physics/cooking/ujitsoCollisionCooking=False, ujitsoCookingMaxProcessCount=0)."
    )
except Exception as _exc:  # pragma: no cover - best-effort hardening only
    print(f"[WARN] could not disable UJITSO cooking settings: {_exc}")

"""Rest everything follows."""

import gymnasium as gym
import logging
import math
import random
import torch
from datetime import datetime

import rl_games.torch_runner as rl_games_torch_runner
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets

import DoorOpening.tasks # noqa: F401

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)


def _resolve_prewarm_door_configs(task_name: str):
    if task_name == "DooropeningMulti":
        from DoorOpening.assets.door.multi_door_cfg import ALL_DOOR_CONFIGS as door_configs

        return door_configs

    from DoorOpening.assets.door.door_cfg import ALL_DOOR_CONFIGS as door_configs

    return door_configs


def _configure_policy_arx_mode(env_cfg) -> None:
    env_cfg.fixed_arx_pose = True
    print(
        "[INFO] ARX joints stay in the fixed tucked pose; "
        f"policy action dim remains {env_cfg.action_space}."
    )


def _install_train_info_bridge():
    """Forward RL-Games frame/epoch counters into the unwrapped IsaacLab env."""

    def _wrapper_set_train_info(self, env_frames, *args, **kwargs):
        if hasattr(self.unwrapped, "set_train_info"):
            return self.unwrapped.set_train_info(env_frames, *args, **kwargs)

    def _gpu_env_set_train_info(self, env_frames, *args, **kwargs):
        if hasattr(self.env, "set_train_info"):
            return self.env.set_train_info(env_frames, *args, **kwargs)

    RlGamesVecEnvWrapper.set_train_info = _wrapper_set_train_info
    RlGamesGpuEnv.set_train_info = _gpu_env_set_train_info


def _install_writer_step_counter(agent):
    """Expose a stable writer-step counter and mirror RL-Games scalars to W&B when enabled.

    wandb.log() normally just enqueues to a background thread, but if the network link to
    wandb.ai stalls it can start blocking the caller. Since this hook only fires on rank 0
    (wandb.run is only non-None there), a stall here desyncs rank 0 from the other ranks'
    collectives (they keep no such dependency) and can wedge the whole distributed job for
    up to the NCCL watchdog timeout. Route the actual wandb.log() calls through a queue and
    a dedicated daemon thread so a stalled wandb sync can never block the training loop.
    """
    writer = agent.writer
    if writer is None or getattr(writer, "_dooropening_wandb_step_counter_installed", False):
        return

    try:
        import wandb
    except ImportError:
        wandb = None

    agent.dooropening_wandb_step = 0
    original_add_scalar = writer.add_scalar

    def _to_scalar(value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    wandb_log_queue = None
    if wandb is not None and wandb.run is not None:
        import queue
        import threading

        wandb_log_queue = queue.Queue(maxsize=10000)

        def _wandb_log_worker():
            while True:
                item = wandb_log_queue.get()
                if item is None:
                    break
                data, step = item
                try:
                    wandb.log(data, step=step)
                except Exception as e:
                    print(f"[WARNING] wandb.log() failed in background thread: {e}")

        threading.Thread(target=_wandb_log_worker, daemon=True, name="dooropening-wandb-log").start()

    def _counting_add_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
        current_wandb_step = agent.dooropening_wandb_step
        result = original_add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        if wandb_log_queue is not None:
            try:
                wandb_log_queue.put_nowait(({tag: _to_scalar(scalar_value)}, current_wandb_step))
            except queue.Full:
                print(
                    f"[WARNING] wandb log queue full; dropping scalar '{tag}' at step "
                    f"{current_wandb_step} (wandb sync appears stalled)."
                )
        agent.dooropening_wandb_step = current_wandb_step + 1
        return result

    writer.add_scalar = _counting_add_scalar
    writer._dooropening_wandb_step_counter_installed = True


def _bind_distributed_cuda_device(enabled: bool):
    """Select the current rank's CUDA device before RL-Games initializes NCCL."""

    if not enabled:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training was requested, but CUDA is not available.")

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    torch.cuda.set_device(local_rank)
    print(f"[INFO][rank {global_rank}] Bound torch CUDA device to cuda:{local_rank} before RL-Games init.")


def _inherit_central_value_config(agent_cfg: dict):
    """Fill central critic training defaults from the main RL-Games config."""

    config = agent_cfg["params"]["config"]
    central_cfg = config.get("central_value_config")
    if not isinstance(central_cfg, dict):
        return

    inherited_keys = (
        "minibatch_size",
        "mini_epochs",
        "learning_rate",
        "lr_schedule",
        "schedule_type",
        "kl_threshold",
        "clip_value",
        "normalize_input",
        "truncate_grads",
    )
    for key in inherited_keys:
        if key in config:
            central_cfg.setdefault(key, config[key])


def _resolve_isaaclab_env(vec_env):
    """Return the unwrapped IsaacLab env behind RL-Games' vec-env wrappers."""

    candidate = getattr(vec_env, "env", vec_env)
    return getattr(candidate, "unwrapped", candidate)


def _restore_env_curriculum_progress(agent):
    """Re-seed the env-side curriculum counter after RL-Games restores a checkpoint.

    RL-Games' `restore()` brings back `epoch_num`/`frame`, but the IsaacLab env is constructed
    fresh, so its `common_step_counter` starts at 0. That counter is what DooropeningEnv's
    `_get_curriculum_step_count()` returns, and it drives BOTH the ADR increment schedule
    (`_update_adr_ranges`, via `adr_reset_progress_total`) and the reference-motion reset /
    drift-threshold curriculum (via `reset_progress_total`). Without this, a resumed run replays
    the whole domain-randomization and reset curriculum from scratch against an already-trained
    policy. Write the counter back here — i.e. after `_restore()` but before `agent.train()`,
    whose first `env_reset()` already samples curriculum-dependent reset states in `_reset_idx`.
    """

    epoch_num = int(getattr(agent, "epoch_num", 0) or 0)
    frame = int(getattr(agent, "frame", 0) or 0)
    if epoch_num <= 0 and frame <= 0:
        # Fresh run (or a checkpoint restored with set_epoch=False) — nothing to resume.
        return

    horizon_length = int(getattr(agent, "horizon_length", 0) or 0)
    num_actors = int(getattr(agent, "num_actors", 0) or 0)
    world_size = int(getattr(agent, "world_size", 1) or 1)

    # Each epoch performs exactly `horizon_length` env.step() calls per rank, so this is exact and
    # independent of how many envs the checkpoint was trained with. `frame` counts
    # num_actors * horizon_length * world_size per epoch, so dividing it by the CURRENT num_actors
    # would misread the progress whenever the run resumes with a different --num_envs; it is only
    # used as a fallback.
    if epoch_num > 0 and horizon_length > 0:
        env_steps = epoch_num * horizon_length
    else:
        env_steps = frame // max(num_actors * world_size, 1)

    unwrapped_env = _resolve_isaaclab_env(agent.vec_env)
    if not hasattr(unwrapped_env, "common_step_counter"):
        print(
            "[WARNING] Could not resume curriculum progress: "
            f"{type(unwrapped_env).__name__} has no `common_step_counter`. "
            "ADR / reset curricula will restart from zero."
        )
        return

    unwrapped_env.common_step_counter = int(env_steps)
    if hasattr(unwrapped_env, "set_train_info"):
        unwrapped_env.set_train_info(frame)
    print(
        f"[INFO] Resumed curriculum progress from checkpoint: epoch={epoch_num}, frame={frame} -> "
        f"common_step_counter={int(env_steps)} (ADR increments and reset curriculum continue from here)."
    )


class DoorOpeningRunner(Runner):
    """Runner that installs local instrumentation before training starts."""

    def run_train(self, args):
        print("Started to train")
        _bind_distributed_cuda_device(self.params["config"].get("multi_gpu", False))
        agent = self.algo_factory.create(self.algo_name, base_name="run", params=self.params)
        _install_writer_step_counter(agent)
        rl_games_torch_runner._restore(agent, args)
        rl_games_torch_runner._override_sigma(agent, args)
        _restore_env_curriculum_progress(agent)
        agent.train()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    global_rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    use_distributed = args_cli.distributed or world_size > 1

    _bind_distributed_cuda_device(use_distributed)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _configure_policy_arx_mode(env_cfg)
    # check for invalid combination of CPU device with distributed training
    if use_distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # update agent device to match simulation device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print("resume_path: ", resume_path)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if use_distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
        log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
    else:
        log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args_cli.exp_name is not None:
        log_dir = args_cli.exp_name
    base_experiment_name = log_dir
    if use_distributed:
        rank_tag = f"rank{global_rank:03d}_local{local_rank:03d}"
        log_dir = f"{base_experiment_name}_{rank_tag}"
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = base_experiment_name if args_cli.wandb_name is None else args_cli.wandb_name
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)
    _inherit_central_value_config(agent_cfg)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"Exact experiment name requested from command line: {os.path.join(log_root_path, log_dir)}")

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # Let the env observe RL-Games frame/iteration counters without patching IsaacLab itself.
    _install_train_info_bridge()

    # Serialize URDF-to-USD conversion across ranks before all workers build the same shared assets.
    preconvert_shared_urdf_assets(
        door_configs=_resolve_prewarm_door_configs(args_cli.task),
        verbose=True,
    )

    # Serialize Isaac Sim init across local ranks so only one rank at a time runs
    # gym.make()/sim.reset() (PhysX cooking via force_load_physics_from_usd). Concurrent
    # cooking against the shared NFS cache is what deadlocks rank 0. Lower local ranks go
    # first; each higher rank waits for the previous local rank's completion sentinel. The
    # handshake is poll-based (os.path.exists) so it uses NO NFS advisory locks (the very
    # thing that deadlocks). Sentinels are scoped per job+host to avoid stale-file matches.
    import socket as _socket
    import time as _time

    _siminit_dir = os.environ.get("SIM_INIT_DIR", "/tmp/DoorOpening/.sim_init")
    _siminit_token = f"{os.environ.get('SLURM_JOB_ID', 'nojob')}.{_socket.gethostname()}"

    def _siminit_sentinel(lr):
        return os.path.join(_siminit_dir, f"siminit.{_siminit_token}.local{lr:03d}.done")

    if use_distributed and local_rank > 0:
        os.makedirs(_siminit_dir, exist_ok=True)
        _prev = _siminit_sentinel(local_rank - 1)
        _wait_timeout = int(os.environ.get("SIM_INIT_WAIT_TIMEOUT", "1800"))
        print(f"[INFO][rank {global_rank}] Serializing sim-init: waiting up to {_wait_timeout}s for local rank {local_rank - 1} sentinel ({_prev})...")
        _t0 = _time.time()
        while not os.path.exists(_prev):
            if _time.time() - _t0 > _wait_timeout:
                print(f"[WARNING][rank {global_rank}] Timed out after {_wait_timeout}s waiting for local rank {local_rank - 1} sim-init sentinel; proceeding anyway (rank may be hung — see its watchdog dump).")
                break
            _time.sleep(2.0)
        else:
            print(f"[INFO][rank {global_rank}] Local rank {local_rank - 1} finished sim-init; proceeding to gym.make().")

    # Arm the hang watchdog: if Isaac Sim init below does not finish within HANG_DUMP_SECONDS,
    # faulthandler dumps the stack of every thread to stderr (repeating) so a stuck rank shows
    # exactly where it is wedged (e.g. PhysX/CUDA GPU init in sim.reset()) instead of hanging
    # silently. Cancelled immediately after the env is created on a healthy rank.
    print(f"[INFO][rank {global_rank}] Arming hang watchdog ({_HANG_DUMP_SECONDS}s) around gym.make(); pid={os.getpid()}.")
    sys.stderr.flush()
    _faulthandler.dump_traceback_later(_HANG_DUMP_SECONDS, repeat=True, file=sys.stderr)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Env created successfully — disarm the watchdog.
    _faulthandler.cancel_dump_traceback_later()
    print(f"[INFO][rank {global_rank}] gym.make() returned; hang watchdog disarmed.")

    # Drop this local rank's completion sentinel so the next local rank may begin its sim-init.
    if use_distributed:
        try:
            os.makedirs(_siminit_dir, exist_ok=True)
            _mine = _siminit_sentinel(local_rank)
            with open(_mine, "w") as _f:
                _f.write(f"rank {global_rank} local {local_rank} pid {os.getpid()} sim-init done\n")
            print(f"[INFO][rank {global_rank}] Wrote sim-init sentinel ({_mine}).")
        except Exception as _exc:
            print(f"[WARNING][rank {global_rank}] could not write sim-init sentinel: {_exc}")

    # Barrier: wait for all ranks to finish Isaac Sim initialization before proceeding to
    # rl-games setup. Without this, a fast rank reaches the NCCL broadcast inside
    # "broadcasting parameters" while the slow rank is still in sim.reset(), causing an
    # indefinite hang or TCPStore timeout on the slow rank.
    if use_distributed and torch.distributed.is_initialized():
        print(f"[INFO][rank {global_rank}] Waiting for all ranks to finish Isaac Sim init...")
        torch.distributed.barrier()
        print(f"[INFO][rank {global_rank}] All ranks ready — proceeding to rl-games setup.")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video and global_rank == 0:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        if MultiObserver is None or PbtAlgoObserver is None:
            raise ImportError(
                "PBT is enabled in the agent config, but this installed isaaclab_rl build does not export "
                "MultiObserver/PbtAlgoObserver. Disable PBT or install a compatible Isaac Lab version."
            )
        observers = MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
        runner = DoorOpeningRunner(observers)
    else:
        runner = DoorOpeningRunner(IsaacAlgoObserver())

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        try:
            wandb.init(
                project=wandb_project,
                entity=args_cli.wandb_entity,
                name=experiment_name,
                sync_tensorboard=False,
                monitor_gym=True,
                save_code=True,
            )
            if not wandb.run.resumed:
                wandb.config.update({"env_cfg": env_cfg.to_dict()})
                wandb.config.update({"agent_cfg": agent_cfg})
        except Exception as e:
            print(f"[WARNING][rank 0] wandb.init() failed: {e}. Continuing without W&B tracking.")

    try:
        if args_cli.checkpoint is not None:
            runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
        else:
            runner.run({"train": True, "play": False, "sigma": train_sigma})
    finally:
        # Close the env (and the sim) even on an early/failed shutdown.
        env.close()


if __name__ == "__main__":
    # run the main function
    import os
    os.environ["WANDB_DISABLE_GYM"] = "true"
    main()
    # close sim app
    simulation_app.close()
