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
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
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
    "--save-viser-pt",
    "--save_viser_pt",
    dest="save_viser_pt",
    type=str,
    default=None,
    help=(
        "Save a replay_viser_pt.py-compatible .pt of the robot joint angles (compact_q) + PD "
        "targets (compact_target) for one env, ordered [base_x, base_y, base_rotation, "
        "panda_1..7, finger_0..N]. Play it with scripts/replay_viser_pt.py."
    ),
)
parser.add_argument(
    "--viser-env-id",
    "--viser_env_id",
    dest="viser_env_id",
    type=int,
    default=0,
    help="Env index to record for --save-viser-pt.",
)
parser.add_argument(
    "--save-viser-num-rollouts",
    "--save_viser_num_rollouts",
    dest="save_viser_num_rollouts",
    type=int,
    default=3,
    help="Number of tracked-env rollouts to record before stopping + saving --save-viser-pt.",
)
parser.add_argument(
    "--door-audit-out",
    "--door_audit_out",
    dest="door_audit_out",
    type=str,
    default=None,
    help=(
        "Record a per-episode success/failure row for every door asset and write it as JSON. "
        "Join it with the door variant_meta.json files via scripts/tools/analyze_door_audit.py to "
        "see which door parameters separate the failures from the successes. Run with "
        "--num_envs equal to the asset count (512 per family set) so every door is covered."
    ),
)
parser.add_argument(
    "--door-audit-episodes",
    "--door_audit_episodes",
    dest="door_audit_episodes",
    type=int,
    default=1,
    help="Stop playback once EVERY env has completed at least this many episodes (with --door-audit-out).",
)
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
parser.add_argument(
    "--probe-pointcloud-camera",
    action="store_true",
    default=False,
    help=(
        "Record a same-frame comparison for one env: the multi_pcd_dagger simulated round-trip cloud "
        "and the ground-truth mesh cloud (at the synced D435 spec), plus the IsaacLab RGB image as a "
        "reference to eyeball against. Saves a .pt for scripts/replay_pointcloud_probe_viser.py."
    ),
)
parser.add_argument(
    "--probe-save-path",
    type=str,
    default="pointcloud_probe.pt",
    help="Where to write the --probe-pointcloud-camera comparison payload.",
)
parser.add_argument(
    "--probe-env-id",
    type=int,
    default=0,
    help="Env index tracked by --probe-pointcloud-camera.",
)
parser.add_argument(
    "--probe-num-frames",
    type=int,
    default=400,
    help=(
        "Capture this many frames, then save the --probe-pointcloud-camera .pt and KEEP playing the "
        "policy (capture stops, playback continues). At --probe-capture-every=1 this is a step count."
    ),
)
parser.add_argument(
    "--probe-capture-every",
    type=int,
    default=1,
    help="Record every Nth env step for --probe-pointcloud-camera.",
)
parser.add_argument(
    "--probe-door-num-points",
    type=int,
    default=4096,
    help="Points sampled on the door mesh (link_1 panel + link_2 handle) for --probe-pointcloud-camera.",
)
parser.add_argument(
    "--probe-robot-num-points",
    type=int,
    default=16384,
    help=(
        "Points sampled on the robot mesh for --probe-pointcloud-camera. Denser than the door because "
        "the robot covers more surface; raise it if background leaks through the robot (penetration)."
    ),
)
parser.add_argument(
    "--probe-render-num-points",
    type=int,
    default=0,
    help="Output size of the simulated round-trip cloud (0 = keep all: door + robot points).",
)
parser.add_argument(
    "--probe-inflate-px",
    type=int,
    default=0,
    help="z-buffer dilation (px) for the simulated round-trip render in --probe-pointcloud-camera.",
)
parser.add_argument(
    "--probe-occluder-inflate-px",
    type=int,
    default=0,
    help="Door-occluder z-buffer dilation (px) for the simulated round-trip render.",
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
    or args_cli.probe_pointcloud_camera
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
from isaaclab.utils.math import quat_apply

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


def _configure_policy_arx_mode(env_cfg) -> None:
    env_cfg.fixed_arx_pose = True
    print(
        "[INFO] ARX joints stay in the fixed tucked pose during playback; "
        f"policy action dim remains {env_cfg.action_space}."
    )


def _build_compact_joint_layout(joint_names):
    """Return (names, indices) for the compact viser-pt joint order.

    Order: [base_x_joint, base_y_joint, base_rotation_joint, panda_joint1..7, finger_joint_0..N]
    -- the human-readable order requested for the replay printout. Uses the REAL joint names so
    scripts/replay_viser_pt.py matches them straight onto the URDF (base_x/y/rotation are actuated
    URDF joints, so the base is driven through them -- no separate base pose needed).
    """
    desired = ["base_x_joint", "base_y_joint", "base_rotation_joint"] + [f"panda_joint{i}" for i in range(1, 8)]
    fingers = sorted(
        (n for n in joint_names if str(n).startswith("finger_joint_")),
        key=lambda n: int(str(n).rsplit("_", 1)[1]),
    )
    desired += list(fingers)
    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
    names, indices = [], []
    for n in desired:
        if n in name_to_idx:
            names.append(n)
            indices.append(name_to_idx[n])
    return names, indices


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
    _configure_policy_arx_mode(env_cfg)

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

    if args_cli.probe_pointcloud_camera and hasattr(env_cfg, "pointcloud_camera_cfg"):
        from DoorOpening.utils.camera_utils import REALSENSE_D435_FAR_M, REALSENSE_D435_NEAR_M

        # The point cloud comes purely from multi_pcd_dagger's renderer; the IsaacLab camera only
        # supplies a REFERENCE RGB image. Drive it to the D435 model (same half resolution, near/far
        # clipping, FOV pinned after the env is built) so the RGB roughly matches the simulated cloud's
        # view. Render mode "depth" is what instantiates the x5 camera sensor; rgb is the only stream.
        env_cfg.pointcloud_render_mode = "depth"
        env_cfg.pointcloud_camera_cfg.height = int(env_cfg.pointcloud_camera_cfg.height) // 2
        env_cfg.pointcloud_camera_cfg.width = int(env_cfg.pointcloud_camera_cfg.width) // 2
        env_cfg.pointcloud_camera_cfg.update_period = 0.0
        env_cfg.pointcloud_camera_cfg.data_types = ["rgb"]
        env_cfg.pointcloud_camera_cfg.spawn.clipping_range = (REALSENSE_D435_NEAR_M, REALSENSE_D435_FAR_M)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    play_env = env.unwrapped
    play_env.ref_motion_lib.reset_from_start = True
    play_env.early_stopping = args_cli.enable_early_stopping
    print(f"[INFO] Play early stopping enabled: {play_env.early_stopping}")

    viser_pt_state = None
    if args_cli.save_viser_pt is not None:
        from DoorOpening.assets.door.multi_door_cfg import asset_paths as _door_asset_paths

        _compact_names, _compact_idx = _build_compact_joint_layout(list(play_env.robot.data.joint_names))
        # joint_pos is in full joint_names order (use _compact_idx), but applied_robot_dof_targets
        # is stored in _robot_dof_idx order -> map each compact joint's full-dof index to its
        # position within _robot_dof_idx for the target lookup.
        _dof_to_target_pos = {int(d): k for k, d in enumerate(play_env._robot_dof_idx.detach().cpu().tolist())}
        _target_pos = [int(_dof_to_target_pos.get(int(j), -1)) for j in _compact_idx]
        _env_id = max(0, min(int(args_cli.viser_env_id), int(play_env.num_envs) - 1))
        _env_origin = play_env.scene.env_origins[_env_id].detach().clone()  # (3,) on device
        _door_asset_idx = int(play_env.env_asset_indices[_env_id].item())
        _door_urdf = str(_door_asset_paths[_door_asset_idx])
        _door_root = play_env.door.data.root_state_w[_env_id, :7].detach().clone()  # on device
        # Build the door pointcloud sampler ONCE (loads the URDF); sample_link_set() is then cheap
        # per frame (FK + resample from precomputed link meshes).
        _door_sampler = None
        try:
            from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaLeapSampler

            _door_sampler = FrankaLeapSampler(_door_urdf, device=str(play_env.device), num_points=2048)
        except Exception as _exc:  # noqa: BLE001
            print(f"[WARN] --save-viser-pt: could not build door pointcloud sampler ({_exc}); robot-only frames.")
        viser_pt_state = {
            "env_id": _env_id,
            "compact_names": _compact_names,
            "compact_idx": torch.as_tensor(_compact_idx, device=play_env.device, dtype=torch.long),
            "target_idx": torch.as_tensor([max(p, 0) for p in _target_pos], device=play_env.device, dtype=torch.long),
            "target_valid": torch.as_tensor([p >= 0 for p in _target_pos], device=play_env.device, dtype=torch.bool),
            "env_origin": _env_origin,
            "door_sampler": _door_sampler,
            "door_link_names": ["base", "link_0", "link_1", "link_2"],
            "door_root_pos_w": _door_root[:3],
            "door_root_quat_w": _door_root[3:7],
            "frames": [],
            "frame_dt": float(play_env.step_dt),
            "num_rollouts": max(1, int(args_cli.save_viser_num_rollouts)),
            "rollouts": 0,
        }
        print(
            f"[INFO] Saving viser .pt (robot joints/targets + door pointcloud, env {_env_id}) to "
            f"{os.path.abspath(args_cli.save_viser_pt)}; door={_door_urdf}"
        )

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

    door_audit_state = None
    if args_cli.door_audit_out is not None:
        from DoorOpening.assets.door.multi_door_cfg import asset_paths as _audit_asset_paths

        _n_assets = len(_audit_asset_paths)
        if play_env.num_envs < _n_assets:
            print(
                f"[AUDIT][WARN] num_envs={play_env.num_envs} < {_n_assets} door assets; only the first "
                f"{play_env.num_envs} doors of the cycle will be evaluated."
            )
        # NOTE ON SUCCESS: play_env.last_success is `episode_reached_last_frame`, and the reference
        # motion advances frame_idx on a CLOCK (multi_motion_lib: `frame_idx += frame_step`), not on
        # achievement. So it reports 1.0 for any episode that merely runs long enough -- with early
        # stopping off, that is every episode. We therefore record it only as `ref_frame_reached` and
        # derive the real outcome from physical task state accumulated over the episode:
        #   hinge_max   : how far the door actually swung open (rad)
        #   latch_max   : how far the handle actually turned (rad)
        #   dist_past_max: how far the base traversed past the door plane (m)
        # The env cfg declares success_far_push_dist / success_far_pull_dist for exactly this
        # traversal test, but nothing in the codebase reads them -- we apply them here.
        _base_x_local = list(play_env.cfg.base_joints).index("base_x_joint")
        door_audit_state = {
            "asset_paths": [str(p) for p in _audit_asset_paths],
            # env -> asset index is fixed for the whole run (rank-offset cycle), so a door's identity
            # never changes across resets and every episode row can be attributed to one door.
            "env_asset_idx": play_env.env_asset_indices.detach().cpu().tolist(),
            "rows": [],
            "episodes_per_env": [0] * int(play_env.num_envs),
            "target_episodes": max(1, int(args_cli.door_audit_episodes)),
            "board_idx": int(play_env._door_board_joint_idx),
            "hinge_idx": int(play_env._door_hinge_joint_idx),
            "base_x_dof": int(play_env._robot_base_dof_idx[_base_x_local].item()),
            "is_push": play_env._get_is_push_env().detach().cpu().clone(),
            "push_thresh": float(play_env.cfg.success_far_push_dist),
            "pull_thresh": float(play_env.cfg.success_far_pull_dist),
            # Running per-episode maxima, reset when an env is done.
            "hinge_max": torch.zeros(play_env.num_envs),
            "latch_max": torch.zeros(play_env.num_envs),
            "dist_max": torch.full((play_env.num_envs,), -1e9),
            "prev": None,
            "done": False,
        }

    def _accumulate_door_progress():
        """Track per-episode maxima of the physical task signals (call every step, pre-reset)."""
        st = door_audit_state
        d = play_env.door.data
        hinge = d.joint_pos[:, st["hinge_idx"]].abs().detach().cpu()
        latch = d.joint_pos[:, st["board_idx"]].abs().detach().cpu()
        base_x = play_env.robot.data.joint_pos[:, st["base_x_dof"]]
        dist = play_env._base_dist_past_door(base_x).detach().cpu()
        torch.maximum(st["hinge_max"], hinge, out=st["hinge_max"])
        torch.maximum(st["latch_max"], latch, out=st["latch_max"])
        torch.maximum(st["dist_max"], dist, out=st["dist_max"])
        print(
            f"[AUDIT] Per-door audit enabled: {play_env.num_envs} envs over {_n_assets} assets, "
            f"{door_audit_state['target_episodes']} episode(s) per env -> {os.path.abspath(args_cli.door_audit_out)}"
        )

    def _snapshot_door_dynamics():
        """Per-env door joint gains for the CURRENTLY running episode.

        Must be read BEFORE env.step(), because the reset inside step() redraws the ADR
        stiffness/damping/effort-limit event terms -- reading after would report the next
        episode's values against this episode's outcome.
        """
        d = play_env.door.data
        b, hh = door_audit_state["board_idx"], door_audit_state["hinge_idx"]
        eff = getattr(d, "joint_effort_limits", None)
        return {
            "board_stiffness": d.joint_stiffness[:, b].detach().cpu().clone(),
            "board_damping": d.joint_damping[:, b].detach().cpu().clone(),
            "hinge_stiffness": d.joint_stiffness[:, hh].detach().cpu().clone(),
            "hinge_damping": d.joint_damping[:, hh].detach().cpu().clone(),
            "hinge_effort_limit": (eff[:, hh].detach().cpu().clone() if eff is not None else None),
            "episode_len": play_env.episode_length_buf.detach().cpu().clone(),
        }

    def _record_door_audit(dones_tensor):
        prev = door_audit_state["prev"]
        if prev is None:
            return
        done_ids = torch.nonzero(dones_tensor.reshape(-1), as_tuple=False).squeeze(-1).detach().cpu().tolist()
        if not done_ids:
            return
        st = door_audit_state
        ref_reached = play_env.last_success.detach().cpu()
        x5_hit = getattr(play_env, "episode_x5_collided", None)
        box_hit = getattr(play_env, "episode_franka_box_collided", None)
        for env_id in done_ids:
            if st["episodes_per_env"][env_id] >= st["target_episodes"]:
                continue
            is_push = bool(st["is_push"][env_id] > 0.5)
            thresh = st["push_thresh"] if is_push else st["pull_thresh"]
            traversed = float(st["dist_max"][env_id].item()) > thresh
            row = {
                "env_id": int(env_id),
                "asset_idx": int(st["env_asset_idx"][env_id]),
                "asset_path": st["asset_paths"][st["env_asset_idx"][env_id]],
                # Real outcome: the base actually got through the doorway.
                "success": float(traversed),
                # The old clock-driven metric, kept so the two can be compared.
                "ref_frame_reached": float(ref_reached[env_id].item()),
                "hinge_max_rad": float(st["hinge_max"][env_id].item()),
                "latch_max_rad": float(st["latch_max"][env_id].item()),
                "dist_past_door_max_m": float(st["dist_max"][env_id].item()),
                "is_push": float(is_push),
                "episode_steps": int(prev["episode_len"][env_id].item()),
                "board_stiffness": float(prev["board_stiffness"][env_id].item()),
                "board_damping": float(prev["board_damping"][env_id].item()),
                "hinge_stiffness": float(prev["hinge_stiffness"][env_id].item()),
                "hinge_damping": float(prev["hinge_damping"][env_id].item()),
            }
            if prev["hinge_effort_limit"] is not None:
                row["hinge_effort_limit"] = float(prev["hinge_effort_limit"][env_id].item())
            if x5_hit is not None:
                row["x5_door_collision"] = float(x5_hit[env_id].to(torch.float32).item())
            if box_hit is not None:
                row["franka_box_collision"] = float(box_hit[env_id].to(torch.float32).item())
            # Why the episode ended. Populated only with --enable-early-stopping: without it
            # _get_dones short-circuits before computing the drift masks and everything is a timeout.
            _rd = play_env.extras.get("fail/robot_drift")
            _dd = play_env.extras.get("fail/door_drift")
            if _rd is not None and _dd is not None:
                robot_drift = bool(_rd.reshape(-1)[env_id].item())
                door_drift = bool(_dd.reshape(-1)[env_id].item())
                row["killed_robot_drift"] = float(robot_drift)
                row["killed_door_drift"] = float(door_drift)
                row["term_reason"] = (
                    "robot_drift" if robot_drift else "door_drift" if door_drift else "timeout"
                )
            st["rows"].append(row)
            st["episodes_per_env"][env_id] += 1
        # Clear the running maxima for every env that just reset, so the next episode starts clean.
        for env_id in done_ids:
            st["hinge_max"][env_id] = 0.0
            st["latch_max"][env_id] = 0.0
            st["dist_max"][env_id] = -1e9
        if all(c >= st["target_episodes"] for c in st["episodes_per_env"]):
            st["done"] = True

    def _write_door_audit():
        import json as _json

        path = os.path.abspath(args_cli.door_audit_out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = door_audit_state["rows"]
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump({"checkpoint": resume_path, "rows": rows}, fh, indent=1)
        n_ok = sum(1 for r in rows if r["success"] > 0.5)
        n_ref = sum(1 for r in rows if r.get("ref_frame_reached", 0.0) > 0.5)
        opened = sum(1 for r in rows if r.get("hinge_max_rad", 0.0) > 0.35)  # ~20 deg of swing
        print(
            f"[AUDIT] Wrote {len(rows)} episode rows to {path}\n"
            f"[AUDIT]   traversed past door : {n_ok}/{len(rows)}  <- real success\n"
            f"[AUDIT]   door swung > 20 deg : {opened}/{len(rows)}\n"
            f"[AUDIT]   reached last ref frame: {n_ref}/{len(rows)}  (clock-driven, not achievement)"
        )

    probe_state = None

    def _save_probe_payload():
        _probe_path = os.path.abspath(args_cli.probe_save_path)
        os.makedirs(os.path.dirname(_probe_path) or ".", exist_ok=True)
        torch.save(
            probe_state["probe"].payload(probe_state["frames"], frame_dt=float(play_env.step_dt)),
            _probe_path,
        )
        probe_state["saved"] = True
        print(
            f"[PROBE] Saved {len(probe_state['frames'])} pointcloud-probe frames "
            f"(sim_roundtrip + ground_truth clouds + reference rgb) to {_probe_path}"
        )

    if args_cli.probe_pointcloud_camera:
        from DoorOpening.utils.pointcloud_camera_probe import PointcloudCameraProbe

        probe_env_id = max(0, min(int(args_cli.probe_env_id), int(play_env.num_envs) - 1))
        probe = PointcloudCameraProbe(
            play_env,
            door_num_points=int(args_cli.probe_door_num_points),
            robot_num_points=int(args_cli.probe_robot_num_points),
            render_num_points=int(args_cli.probe_render_num_points) or None,
            inflate_px=int(args_cli.probe_inflate_px),
            occluder_inflate_px=int(args_cli.probe_occluder_inflate_px),
        )
        probe_state = {
            "probe": probe,
            "env_id": probe_env_id,
            "frames": [],
            "num_frames": max(1, int(args_cli.probe_num_frames)),
            "capture_every": max(1, int(args_cli.probe_capture_every)),
            "step": 0,
            "configured": False,
            "done": False,
            "saved": False,
        }
        print(
            f"[PROBE] Pointcloud camera probe enabled (env_id={probe_env_id}, "
            f"num_frames={probe_state['num_frames']}, spec {probe.sampler_camera_spec['W']}x"
            f"{probe.sampler_camera_spec['H']} @ [{probe.sampler_camera_spec['near_m']}, "
            f"{probe.sampler_camera_spec['far_m']}] m). Playback CONTINUES after saving."
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
    contact_log_step = 0
    if probe_state is not None and not probe_state["configured"]:
        # The sensor prims exist only after the first reset -> pin the D435 intrinsics now.
        probe_state["probe"].configure_isaac_camera_intrinsics()
        probe_state["configured"] = True
        print("[PROBE] IsaacLab camera intrinsics pinned to the simulated D435 spec.")
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
                # Snapshot this episode's door gains and task progress before step() resets them.
                if door_audit_state is not None:
                    door_audit_state["prev"] = _snapshot_door_dynamics()
                    _accumulate_door_progress()
                # env stepping
                obs, _, dones, _ = env.step(actions)
                if door_audit_state is not None:
                    _record_door_audit(dones)

                # Contact-force readout: franka<->arx (arm self-collision vs the arx/x5 camera arm)
                # and leap-fingers<->panel. Printed every 30 env steps to avoid flooding the console.
                contact_log_step += 1
                if contact_log_step % 30 == 0:
                    _fa = getattr(play_env, "franka_arx_contact_force_norm", None)
                    _fp = getattr(play_env, "finger_panel_contact_force_norm", None)
                    _fh = getattr(play_env, "finger_handle_contact_force_norm", None)
                    if _fa is not None and _fp is not None and _fh is not None:
                        print(
                            f"[CONTACT] step={contact_log_step} "
                            f"franka<->arx force: mean={_fa.mean().item():.3f} max={_fa.max().item():.3f} N | "
                            f"leap-fingers<->panel force: mean={_fp.mean().item():.3f} max={_fp.max().item():.3f} N | "
                            f"leap-fingers<->handle force: mean={_fh.mean().item():.3f} max={_fh.max().item():.3f} N"
                        )

                if probe_state is not None and not probe_state["done"]:
                    probe_env_id = probe_state["env_id"]
                    if probe_state["step"] % probe_state["capture_every"] == 0:
                        probe_state["frames"].append(probe_state["probe"].capture(probe_env_id))
                        if len(probe_state["frames"]) % 50 == 0:
                            print(
                                f"[PROBE] captured {len(probe_state['frames'])}/"
                                f"{probe_state['num_frames']} frames."
                            )
                    probe_state["step"] += 1
                    if len(probe_state["frames"]) >= probe_state["num_frames"]:
                        # Save once, then keep playing the policy (stop capturing further frames).
                        _save_probe_payload()
                        probe_state["done"] = True
                        print("[PROBE] Capture complete; continuing policy playback.")

                if viser_pt_state is not None and not viser_pt_state.get("done"):
                    _e = viser_pt_state["env_id"]
                    _org = viser_pt_state["env_origin"]  # device (3,)
                    _q = play_env.robot.data.joint_pos[_e, viser_pt_state["compact_idx"]].detach().cpu().clone()
                    _tgt = play_env.applied_robot_dof_targets[_e, viser_pt_state["target_idx"]].detach().cpu().clone()
                    _tgt[~viser_pt_state["target_valid"].cpu()] = 0.0  # joints not in the controlled set
                    _rb = play_env.robot.data.root_state_w[_e, :7].detach()
                    _dj = play_env.door.data.joint_pos[_e].detach()
                    _frame = {
                        "compact_q": _q,
                        "compact_target": _tgt,
                        "door_joint_pos": _dj.cpu().clone(),
                        "robot_base_pos_w": (_rb[:3] - _org).cpu().clone(),
                        "robot_base_quat_w": _rb[3:7].cpu().clone(),
                    }
                    # Deformed door pointcloud (env-relative world frame) for this frame's door joints.
                    _sampler = viser_pt_state["door_sampler"]
                    if _sampler is not None:
                        try:
                            _pb = _sampler.sample_link_set(_dj.unsqueeze(0), viser_pt_state["door_link_names"])[0]
                            _rq = viser_pt_state["door_root_quat_w"].unsqueeze(0).expand(_pb.shape[0], -1)
                            _pw = quat_apply(_rq, _pb) + viser_pt_state["door_root_pos_w"]
                            _frame["door_points_world"] = (_pw - _org).detach().cpu().clone()
                        except Exception as _exc:  # noqa: BLE001
                            if not viser_pt_state.get("_warned_sampler"):
                                print(f"[WARN] door pointcloud sampling failed ({_exc}); robot-only frames.")
                                viser_pt_state["_warned_sampler"] = True
                            viser_pt_state["door_sampler"] = None
                    viser_pt_state["frames"].append(_frame)
                    # Record N continuous rollouts: count each time the tracked env's episode ends.
                    if int(_e) < len(dones) and bool(dones[int(_e)].item()):
                        viser_pt_state["rollouts"] += 1
                        print(
                            f"[INFO] --save-viser-pt: rollout "
                            f"{viser_pt_state['rollouts']}/{viser_pt_state['num_rollouts']} captured "
                            f"({len(viser_pt_state['frames'])} frames)."
                        )
                        if viser_pt_state["rollouts"] >= viser_pt_state["num_rollouts"]:
                            viser_pt_state["done"] = True

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

            if door_audit_state is not None and door_audit_state["done"]:
                print(
                    f"[AUDIT] Every env completed {door_audit_state['target_episodes']} episode(s); "
                    "stopping playback."
                )
                break

            # Enough rollouts captured -> stop the play loop so the .pt is written by the finally block.
            if viser_pt_state is not None and viser_pt_state.get("done"):
                print(
                    f"[INFO] Captured {viser_pt_state['rollouts']} rollout(s) for --save-viser-pt; "
                    "stopping playback."
                )
                break

            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video -- BUT never cut off an in-progress
                # --save-viser-pt capture: keep stepping until the requested viser rollouts are done
                # (the viser-done break above handles stopping then). Use >= so the break still fires
                # once viser finishes even though we skipped the exact video_length step.
                viser_capturing = viser_pt_state is not None and not viser_pt_state.get("done")
                if timestep >= args_cli.video_length and not viser_capturing:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # Write whatever rows exist, even on Ctrl-C / early exit.
        if door_audit_state is not None and door_audit_state["rows"]:
            _write_door_audit()

        if viser_pt_state is not None and viser_pt_state["frames"]:
            _viser_path = os.path.abspath(args_cli.save_viser_pt)
            os.makedirs(os.path.dirname(_viser_path) or ".", exist_ok=True)
            torch.save(
                {
                    "frames": viser_pt_state["frames"],
                    "compact_target_joint_names": viser_pt_state["compact_names"],
                    "frame_dt": viser_pt_state["frame_dt"],
                },
                _viser_path,
            )
            print(f"[INFO] Saved {len(viser_pt_state['frames'])} viser frames (robot + door pointcloud) to {_viser_path}")

        # If we were interrupted before the frame target (so no save happened yet), save what we have.
        if probe_state is not None and probe_state["frames"] and not probe_state["saved"]:
            _save_probe_payload()

        if pointcloud_camera_state is not None and pointcloud_camera_state["viewer"] is not None:
            pointcloud_camera_state["viewer"].close()

        if pointcloud_camera_state is not None:
            print(
                "[CAM][SUMMARY] "
                f"last_frame={pointcloud_camera_state['last_frame']} "
                f"saved_frames={pointcloud_camera_state['saved_frames']}"
            )

        # Always close the env so resources are released even on interruption.
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
