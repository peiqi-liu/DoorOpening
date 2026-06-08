# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

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
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
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
    "--enable-early-stopping",
    action="store_true",
    default=False,
    help="Enable environment early stopping in play mode using the task reset thresholds.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--track-pointcloud-camera",
    action="store_true",
    default=False,
    help="Track the live x5 pointcloud-camera depth view during evaluation.",
)
parser.add_argument(
    "--pointcloud-camera-env-id",
    type=int,
    default=0,
    help="Environment index whose pointcloud camera view should be tracked.",
)
parser.add_argument(
    "--pointcloud-camera-random-env",
    action="store_true",
    default=False,
    help="Randomly choose which environment's pointcloud camera view to track.",
)
parser.add_argument(
    "--pointcloud-camera-resample-on-reset",
    action="store_true",
    default=False,
    help="When tracking a random env, choose a new random env whenever the tracked env resets.",
)
parser.add_argument(
    "--pointcloud-camera-show-window",
    action="store_true",
    default=False,
    help="Show the tracked pointcloud camera depth stream in an Isaac UI window.",
)
parser.add_argument(
    "--pointcloud-camera-save-dir",
    type=str,
    default=None,
    help="Optional directory to save tracked pointcloud camera frames as .ppm images.",
)
parser.add_argument(
    "--pointcloud-camera-save-every",
    type=int,
    default=1,
    help="Save every Nth tracked pointcloud camera frame.",
)
parser.add_argument(
    "--pointcloud-camera-save-raw-depth",
    action="store_true",
    default=False,
    help="Also save tracked raw pointcloud camera depth arrays as .npy files.",
)
parser.add_argument(
    "--pointcloud-camera-depth-min",
    type=float,
    default=None,
    help="Minimum displayed depth in meters for tracked pointcloud camera frames.",
)
parser.add_argument(
    "--pointcloud-camera-depth-max",
    type=float,
    default=None,
    help="Maximum displayed depth in meters for tracked pointcloud camera frames.",
)
parser.add_argument(
    "--pointcloud-camera-stream",
    type=str,
    default="rgb",
    choices=("rgb", "depth"),
    help="Which pointcloud camera stream to visualize or save during evaluation.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
selected_door_families = _normalize_family_selection(args_cli.door_families)
if selected_door_families is not None:
    os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = ",".join(selected_door_families)
    print(f"[INFO] Using multi-door families: {selected_door_families}")
# always enable cameras to record video or inspect the pointcloud camera
if (
    args_cli.video
    or args_cli.track_pointcloud_camera
    or args_cli.pointcloud_camera_show_window
    or args_cli.pointcloud_camera_save_dir is not None
):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import numpy as np
import random
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
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

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    # Older Isaac Lab releases exposed this helper under isaaclab.utils.
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets

import DoorOpening.tasks  # noqa: F401


class IsaacCameraViewer:
    """Render frames inside an Isaac Kit UI window."""

    def __init__(self, title: str, width: int = 640, height: int = 480, label: str = "Pointcloud camera"):
        import omni.ui as ui

        self._ui = ui
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            title,
            width=width + 24,
            height=height + 48,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )

        blank_frame = np.zeros((height, width, 4), dtype=np.uint8)
        blank_frame[..., 3] = 255

        with self._window.frame:
            with self._ui.VStack(spacing=4):
                self._ui.Label(label, height=20)
                with self._ui.Frame(width=width, height=height):
                    self._ui.ImageWithProvider(self._provider)

        self.update_image(blank_frame)

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Unexpected image shape {image.shape}")

        if image.shape[2] == 3:
            alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=-1)

        return np.ascontiguousarray(image)

    def update_image(self, image: np.ndarray):
        rgba = self._to_rgba(np.asarray(image))
        height, width = rgba.shape[:2]
        self._provider.set_bytes_data(rgba.reshape(-1).data, [width, height])

    def close(self):
        if self._window is not None:
            self._window.visible = False


def _extract_depth_frame(camera, env_id: int) -> torch.Tensor | None:
    depth = camera.data.output.get("distance_to_image_plane")
    if depth is None:
        return None
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim != 3:
        raise ValueError(f"Expected camera depth with shape (B, H, W), got {tuple(depth.shape)}")
    return depth[env_id]


def _extract_rgb_frame(camera, env_id: int) -> torch.Tensor | None:
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return None
    if rgb.ndim != 4:
        raise ValueError(f"Expected camera rgb with shape (B, H, W, C), got {tuple(rgb.shape)}")
    return rgb[env_id]


def _colorize_depth(depth: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    valid_mask = np.isfinite(depth) & (depth > 0.0)
    rgba = np.zeros((*depth.shape, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    if not np.any(valid_mask):
        return rgba

    depth_clipped = np.clip(depth, near_m, far_m)
    denom = max(float(far_m) - float(near_m), 1e-6)
    norm = 1.0 - (depth_clipped - float(near_m)) / denom
    norm = np.clip(norm, 0.0, 1.0)

    red = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb[~valid_mask] = 0.0
    rgba[..., :3] = (rgb * 255.0).astype(np.uint8)
    return rgba


def _to_uint8_rgba(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[2] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        image = np.concatenate((image, alpha), axis=-1)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected RGBA image with shape (H, W, 4), got {image.shape}")
    return np.ascontiguousarray(image)


def _write_ppm(path: str, rgba: np.ndarray):
    rgb = np.ascontiguousarray(rgba[..., :3], dtype=np.uint8)
    height, width = rgb.shape[:2]
    with open(path, "wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(rgb.tobytes())


def _sample_tracked_env_id(num_envs: int, default_env_id: int, random_env: bool) -> int:
    if num_envs <= 0:
        raise ValueError(f"Expected num_envs > 0, got {num_envs}")
    if random_env:
        return random.randrange(num_envs)
    return max(0, min(int(default_env_id), num_envs - 1))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    pointcloud_camera_stream = str(args_cli.pointcloud_camera_stream).lower()
    track_pointcloud_camera = bool(
        args_cli.track_pointcloud_camera
        or args_cli.pointcloud_camera_show_window
        or args_cli.pointcloud_camera_save_dir is not None
    )

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # Serialize URDF-to-USD conversion across ranks before all workers build the same shared assets.
    preconvert_shared_urdf_assets()

    if track_pointcloud_camera and hasattr(env_cfg, "pointcloud_render_mode"):
        env_cfg.pointcloud_render_mode = "depth"
    if track_pointcloud_camera and hasattr(env_cfg, "pointcloud_camera_cfg"):
        # Tracked playback should feel live, so request a fresh frame every env step.
        env_cfg.pointcloud_camera_cfg.update_period = 0.0
        requested_data_types = []
        if pointcloud_camera_stream == "rgb":
            requested_data_types.append("rgb")
        if pointcloud_camera_stream == "depth":
            requested_data_types.append("distance_to_image_plane")
        env_cfg.pointcloud_camera_cfg.data_types = requested_data_types

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    play_env = env.unwrapped
    play_env.ref_motion_lib.reset_from_start = True
    play_env.early_stopping = args_cli.enable_early_stopping
    print(f"[INFO] Play early stopping enabled: {play_env.early_stopping}")

    pointcloud_camera_state = None
    if track_pointcloud_camera:
        pointcloud_camera = getattr(play_env, "pointcloud_camera", None)
        if pointcloud_camera is None:
            raise AttributeError(
                f"Environment {type(play_env).__name__} does not expose pointcloud_camera while tracking was requested."
            )
        random_env_selection = bool(args_cli.pointcloud_camera_random_env)
        tracked_env_id = _sample_tracked_env_id(
            num_envs=play_env.num_envs,
            default_env_id=int(args_cli.pointcloud_camera_env_id),
            random_env=random_env_selection,
        )
        depth_cfg = play_env.cfg.pointcloud_camera_cfg
        near_m = (
            float(args_cli.pointcloud_camera_depth_min)
            if args_cli.pointcloud_camera_depth_min is not None
            else float(depth_cfg.spawn.clipping_range[0])
        )
        far_m = (
            float(args_cli.pointcloud_camera_depth_max)
            if args_cli.pointcloud_camera_depth_max is not None
            else float(depth_cfg.spawn.clipping_range[1])
        )
        if far_m <= near_m:
            raise ValueError(f"Expected pointcloud camera depth max > min, got {far_m} <= {near_m}.")

        viewer = None
        if args_cli.pointcloud_camera_show_window:
            viewer = IsaacCameraViewer(
                title=f"Pointcloud Camera Env {tracked_env_id}",
                width=int(depth_cfg.width),
                height=int(depth_cfg.height),
                label=f"Pointcloud camera {pointcloud_camera_stream.upper()}",
            )

        save_dir = None
        if args_cli.pointcloud_camera_save_dir is not None:
            save_dir = os.path.abspath(args_cli.pointcloud_camera_save_dir)
            os.makedirs(save_dir, exist_ok=True)
            print(f"[INFO] Saving tracked pointcloud camera frames to {save_dir}")

        pointcloud_camera_state = {
            "camera": pointcloud_camera,
            "env_id": tracked_env_id,
            "random_env": random_env_selection,
            "resample_on_reset": bool(args_cli.pointcloud_camera_resample_on_reset),
            "stream": pointcloud_camera_stream,
            "near_m": near_m,
            "far_m": far_m,
            "viewer": viewer,
            "save_dir": save_dir,
            "save_every": max(int(args_cli.pointcloud_camera_save_every), 1),
            "save_raw_depth": bool(args_cli.pointcloud_camera_save_raw_depth),
            "last_frame": -1,
            "saved_frames": 0,
        }
        print(
            "[INFO] Pointcloud camera tracking enabled "
            f"(env_id={tracked_env_id}, random_env={random_env_selection}, "
            f"resample_on_reset={args_cli.pointcloud_camera_resample_on_reset}, "
            f"stream={pointcloud_camera_stream}, show_window={args_cli.pointcloud_camera_show_window}, "
            f"save_dir={save_dir})."
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                obs = agent.obs_to_torch(obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                # env stepping
                obs, _, dones, _ = env.step(actions)

                if pointcloud_camera_state is not None:
                    camera = pointcloud_camera_state["camera"]
                    tracked_env_id = pointcloud_camera_state["env_id"]
                    camera_frame = int(camera.frame[tracked_env_id].item())
                    if camera_frame != pointcloud_camera_state["last_frame"]:
                        pointcloud_camera_state["last_frame"] = camera_frame
                        stream = pointcloud_camera_state["stream"]
                        frame_rgba = None
                        save_stem = None
                        if stream == "rgb":
                            rgb_frame = _extract_rgb_frame(camera, tracked_env_id)
                            if rgb_frame is not None:
                                rgb_np = rgb_frame.detach().cpu().numpy()
                                frame_rgba = _to_uint8_rgba(rgb_np)
                                mean_rgb = frame_rgba[..., :3].mean(axis=(0, 1))
                                print(
                                    f"[CAM] frame={camera_frame:06d} env={tracked_env_id} "
                                    f"rgb_mean=[{mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f}]"
                                )
                                save_stem = f"rgb_env{tracked_env_id:02d}_frame{camera_frame:06d}"
                        else:
                            depth_frame = _extract_depth_frame(camera, tracked_env_id)
                            if depth_frame is not None:
                                depth_np = depth_frame.detach().cpu().numpy().astype(np.float32, copy=False)
                                frame_rgba = _colorize_depth(
                                    depth_np,
                                    near_m=pointcloud_camera_state["near_m"],
                                    far_m=pointcloud_camera_state["far_m"],
                                )
                                valid_mask = np.isfinite(depth_np) & (depth_np > 0.0)
                                valid_count = int(valid_mask.sum())
                                if valid_count > 0:
                                    valid_depth = depth_np[valid_mask]
                                    print(
                                        f"[CAM] frame={camera_frame:06d} env={tracked_env_id} "
                                        f"valid={valid_count}/{depth_np.size} "
                                        f"depth_range=[{valid_depth.min():.3f}, {valid_depth.max():.3f}] m"
                                    )
                                else:
                                    print(
                                        f"[CAM] frame={camera_frame:06d} env={tracked_env_id} valid=0/{depth_np.size}"
                                    )
                                save_stem = f"depth_env{tracked_env_id:02d}_frame{camera_frame:06d}"

                        if frame_rgba is not None:
                            if pointcloud_camera_state["viewer"] is not None:
                                pointcloud_camera_state["viewer"].update_image(frame_rgba)

                            save_dir = pointcloud_camera_state["save_dir"]
                            if save_dir is not None and camera_frame % pointcloud_camera_state["save_every"] == 0:
                                image_path = os.path.join(save_dir, f"{save_stem}.ppm")
                                _write_ppm(image_path, frame_rgba)
                                if pointcloud_camera_state["save_raw_depth"] and stream == "depth":
                                    np.save(
                                        os.path.join(save_dir, f"{save_stem}.npy"),
                                        depth_np,
                                    )
                                pointcloud_camera_state["saved_frames"] += 1

                # perform operations for terminated episodes
                if len(dones) > 0:
                    if (
                        pointcloud_camera_state is not None
                        and pointcloud_camera_state["random_env"]
                        and pointcloud_camera_state["resample_on_reset"]
                    ):
                        tracked_env_id = int(pointcloud_camera_state["env_id"])
                        tracked_done = bool(dones[tracked_env_id].item()) if tracked_env_id < len(dones) else False
                        if tracked_done:
                            new_env_id = _sample_tracked_env_id(
                                num_envs=play_env.num_envs,
                                default_env_id=tracked_env_id,
                                random_env=True,
                            )
                            pointcloud_camera_state["env_id"] = new_env_id
                            pointcloud_camera_state["last_frame"] = -1
                            print(f"[CAM] tracked env reset; switched pointcloud camera view to env={new_env_id}")
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0
            if args_cli.video:
                timestep += 1
                # exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if pointcloud_camera_state is not None and pointcloud_camera_state["viewer"] is not None:
            pointcloud_camera_state["viewer"].close()

    if pointcloud_camera_state is not None:
        print(
            "[CAM][SUMMARY] "
            f"last_frame={pointcloud_camera_state['last_frame']} "
            f"saved_frames={pointcloud_camera_state['saved_frames']}"
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
