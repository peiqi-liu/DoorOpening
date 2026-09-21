#!/usr/bin/env python3

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


parser = argparse.ArgumentParser(description="Render the DooropeningMulti pointcloud camera for one environment.")
parser.add_argument("--task", type=str, default="DooropeningMulti", help="Task name.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="Hydra agent config entry point.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to instantiate.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to inspect.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum env steps to run. 0 means until closed.")
parser.add_argument(
    "--door-families",
    "--door_families",
    dest="door_families",
    type=str,
    default=None,
    help="Comma-separated multi-door family folders, e.g. PartNetv5_plusplus,PartNetv6_plusplus.",
)
parser.add_argument(
    "--show_window",
    action="store_true",
    default=False,
    help="Show the pointcloud camera depth stream in an Isaac UI window.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="Optional directory to save colorized depth frames as .ppm images.",
)
parser.add_argument("--save_every", type=int, default=1, help="Save every Nth camera frame.")
parser.add_argument(
    "--save_raw_depth",
    action="store_true",
    default=False,
    help="Also save raw depth arrays as .npy beside the colorized images.",
)
parser.add_argument(
    "--depth_min",
    type=float,
    default=None,
    help="Minimum displayed depth in meters. Defaults to the camera near clip.",
)
parser.add_argument(
    "--depth_max",
    type=float,
    default=None,
    help="Maximum displayed depth in meters. Defaults to the camera far clip.",
)
parser.add_argument(
    "--use_motion_ref",
    action="store_true",
    default=False,
    help="Keep the reference motion library enabled. By default this script renders a static env setup.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

selected_door_families = _normalize_family_selection(args_cli.door_families)
if selected_door_families is not None:
    os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = ",".join(selected_door_families)
    print(f"[INFO] Using multi-door families: {selected_door_families}")

args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets
from DoorOpening.assets.door.multi_door_cfg import asset_family_names, asset_paths

import DoorOpening.tasks  # noqa: F401


class IsaacCameraViewer:
    """Render frames inside an Isaac Kit UI window."""

    def __init__(self, title: str, width: int = 640, height: int = 480):
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
                self._ui.Label("Pointcloud camera depth", height=20)
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


def _write_ppm(path: str, rgba: np.ndarray):
    rgb = np.ascontiguousarray(rgba[..., :3], dtype=np.uint8)
    height, width = rgb.shape[:2]
    with open(path, "wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(rgb.tobytes())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, _agent_cfg):
    env_cfg.scene.num_envs = max(int(args_cli.num_envs), 1)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.pointcloud_render_mode = "depth"
    env_cfg.enable_pointcloud_camera = True
    env_cfg.pointcloud_camera_cfg.data_types = ["distance_to_image_plane"]
    env_cfg.use_motion_ref = bool(args_cli.use_motion_ref)

    preconvert_shared_urdf_assets()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    env_id = max(0, min(int(args_cli.env_id), base_env.num_envs - 1))

    if not hasattr(base_env, "pointcloud_camera") or base_env.pointcloud_camera is None:
        raise RuntimeError("Pointcloud camera was not created. Expected pointcloud_render_mode='depth'.")

    camera = base_env.pointcloud_camera
    camera_cfg = base_env.cfg.pointcloud_camera_cfg
    near_m = float(args_cli.depth_min) if args_cli.depth_min is not None else float(camera_cfg.spawn.clipping_range[0])
    far_m = float(args_cli.depth_max) if args_cli.depth_max is not None else float(camera_cfg.spawn.clipping_range[1])
    if far_m <= near_m:
        raise ValueError(f"Expected depth_max > depth_min, got {far_m} <= {near_m}")

    if hasattr(base_env, "env_asset_indices"):
        asset_idx = int(base_env.env_asset_indices[env_id].detach().cpu().item())
        print(
            f"[INFO] Rendering env {env_id} | asset_idx={asset_idx} | family={asset_family_names[asset_idx]} | "
            f"asset={asset_paths[asset_idx]}"
        )

    viewer = None
    if args_cli.show_window:
        viewer = IsaacCameraViewer(
            title=f"Pointcloud Camera Env {env_id}",
            width=int(camera_cfg.width),
            height=int(camera_cfg.height),
        )

    save_dir = None
    if args_cli.save_dir:
        save_dir = os.path.abspath(args_cli.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"[INFO] Saving colorized depth frames to {save_dir}")

    _, _ = env.reset()
    actions = torch.zeros((base_env.num_envs, int(base_env.cfg.action_space)), device=base_env.device)

    last_camera_frame = -1
    saved_frames = 0
    step_count = 0

    try:
        while simulation_app.is_running():
            _, _, _, _, _ = env.step(actions)
            step_count += 1

            camera_frame = int(camera.frame[env_id].item())
            if camera_frame == last_camera_frame:
                if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                    break
                continue

            last_camera_frame = camera_frame
            depth_frame = _extract_depth_frame(camera, env_id)
            if depth_frame is None:
                if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                    break
                continue

            depth_np = depth_frame.detach().cpu().numpy().astype(np.float32, copy=False)
            colorized = _colorize_depth(depth_np, near_m=near_m, far_m=far_m)

            valid_mask = np.isfinite(depth_np) & (depth_np > 0.0)
            valid_count = int(valid_mask.sum())
            if valid_count > 0:
                valid_depth = depth_np[valid_mask]
                print(
                    f"[INFO] cam_frame={camera_frame:06d} env={env_id} valid={valid_count}/{depth_np.size} "
                    f"depth_range=[{valid_depth.min():.3f}, {valid_depth.max():.3f}] m"
                )
            else:
                print(f"[INFO] cam_frame={camera_frame:06d} env={env_id} valid=0/{depth_np.size}")

            if viewer is not None:
                viewer.update_image(colorized)

            if save_dir is not None and camera_frame % max(int(args_cli.save_every), 1) == 0:
                image_path = os.path.join(save_dir, f"depth_env{env_id:02d}_frame{camera_frame:06d}.ppm")
                _write_ppm(image_path, colorized)
                if args_cli.save_raw_depth:
                    np.save(
                        os.path.join(save_dir, f"depth_env{env_id:02d}_frame{camera_frame:06d}.npy"),
                        depth_np,
                    )
                saved_frames += 1

            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
    finally:
        if viewer is not None:
            viewer.close()
        print(f"[INFO] Finished after {step_count} env steps. Saved {saved_frames} frames.")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
