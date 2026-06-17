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
parser.add_argument(
    "--viser",
    "--viser-raw",
    "--viser-pt",
    "--viser_pt",
    dest="viser_pt",
    action="store_true",
    default=False,
    help="Save raw robot/door point-cloud .pt chunks for Viser replay during teacher training.",
)
parser.add_argument(
    "--viser-raw-path",
    "--viser-pt-path",
    "--viser_pt_path",
    dest="viser_pt_path",
    type=str,
    default=None,
    help="Output path for Viser .pt replay chunks.",
)
parser.add_argument(
    "--viser-env-id",
    "--viser-pt-env-id",
    "--viser_pt_env_id",
    dest="viser_pt_env_id",
    type=int,
    default=None,
    help="Environment index to record in Viser .pt dumps.",
)
parser.add_argument(
    "--viser-raw-interval",
    "--viser-pt-interval",
    "--viser_pt_interval",
    dest="viser_pt_interval",
    type=int,
    default=None,
    help="Environment-step interval between recorded point-cloud frames.",
)
parser.add_argument(
    "--viser-raw-save-interval",
    "--viser-pt-save-interval",
    "--viser_pt_save_interval",
    dest="viser_pt_save_interval",
    type=int,
    default=None,
    help="Iteration interval between saved Viser .pt chunks.",
)
parser.add_argument(
    "--viser-raw-max-frames",
    "--viser-pt-max-frames",
    "--viser_pt_max_frames",
    dest="viser_pt_max_frames",
    type=int,
    default=None,
    help="Maximum frames kept in each Viser .pt chunk.",
)
parser.add_argument(
    "--viser-raw-max-points",
    "--viser-pt-max-points",
    "--viser_pt_max_points",
    dest="viser_pt_max_points",
    type=int,
    default=None,
    help="Maximum exported points per cloud in each Viser .pt frame.",
)
parser.add_argument(
    "--viser-raw-robot-points",
    "--viser-pt-robot-points",
    "--viser_pt_robot_points",
    dest="viser_pt_robot_points",
    type=int,
    default=None,
    help="Robot sampler point count before export.",
)
parser.add_argument(
    "--viser-raw-door-points",
    "--viser-pt-door-points",
    "--viser_pt_door_points",
    dest="viser_pt_door_points",
    type=int,
    default=None,
    help="Door sampler point count before export.",
)
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

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
    """Expose a stable writer-step counter and mirror RL-Games scalars to W&B when enabled."""
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

    def _counting_add_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
        current_wandb_step = agent.dooropening_wandb_step
        result = original_add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        if wandb is not None and wandb.run is not None:
            wandb.log({tag: _to_scalar(scalar_value)}, step=current_wandb_step)
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


def _configure_viser_pt_recording(env_cfg, log_dir: str):
    """Apply CLI overrides for headless point-cloud replay dumps."""

    has_any_override = any(
        value is not None
        for value in (
            args_cli.viser_pt_path,
            args_cli.viser_pt_env_id,
            args_cli.viser_pt_interval,
            args_cli.viser_pt_save_interval,
            args_cli.viser_pt_max_frames,
            args_cli.viser_pt_max_points,
            args_cli.viser_pt_robot_points,
            args_cli.viser_pt_door_points,
        )
    )
    if not args_cli.viser_pt and not has_any_override:
        return
    if not hasattr(env_cfg, "viser_pointcloud"):
        logger.warning("--viser_pt was requested, but this task config has no `viser_pointcloud` field.")
        return

    record_cfg = dict(getattr(env_cfg, "viser_pointcloud", {}) or {})
    if args_cli.viser_pt:
        record_cfg["enabled"] = True
    if args_cli.viser_pt_path is not None:
        record_cfg["path"] = args_cli.viser_pt_path
    if args_cli.viser_pt_env_id is not None:
        record_cfg["env_id"] = int(args_cli.viser_pt_env_id)
    if args_cli.viser_pt_interval is not None:
        record_cfg["capture_interval"] = max(1, int(args_cli.viser_pt_interval))
    if args_cli.viser_pt_save_interval is not None:
        record_cfg["save_interval"] = max(1, int(args_cli.viser_pt_save_interval))
    if args_cli.viser_pt_max_frames is not None:
        record_cfg["max_frames"] = max(0, int(args_cli.viser_pt_max_frames))
    if args_cli.viser_pt_max_points is not None:
        record_cfg["max_points"] = int(args_cli.viser_pt_max_points)
    if args_cli.viser_pt_robot_points is not None:
        record_cfg["robot_num_points"] = int(args_cli.viser_pt_robot_points)
    if args_cli.viser_pt_door_points is not None:
        record_cfg["door_num_points"] = int(args_cli.viser_pt_door_points)

    env_cfg.viser_pointcloud = record_cfg
    print(f"[INFO] Viser .pt point-cloud recording config: {record_cfg}")
    print(f"[INFO] Relative Viser .pt paths will be resolved under: {log_dir}")


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


class DoorOpeningRunner(Runner):
    """Runner that installs local instrumentation before training starts."""

    def run_train(self, args):
        print("Started to train")
        _bind_distributed_cuda_device(self.params["config"].get("multi_gpu", False))
        agent = self.algo_factory.create(self.algo_name, base_name="run", params=self.params)
        _install_writer_step_counter(agent)
        rl_games_torch_runner._restore(agent, args)
        rl_games_torch_runner._override_sigma(agent, args)
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
    _configure_viser_pt_recording(env_cfg, env_cfg.log_dir)
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    try:
        if args_cli.checkpoint is not None:
            runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
        else:
            runner.run({"train": True, "play": False, "sigma": train_sigma})
    finally:
        # Let the env flush any pending Viser replay chunk even on early shutdown.
        env.close()


if __name__ == "__main__":
    # run the main function
    import os
    os.environ["WANDB_DISABLE_GYM"] = "true"
    main()
    # close sim app
    simulation_app.close()
