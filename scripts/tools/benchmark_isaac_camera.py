#!/usr/bin/env python3
"""Short end-to-end benchmark for the Isaac depth camera path."""

import argparse
import os
import statistics
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--door-families", type=str, default="PartNetv5_plus_gripper640_20260922_eval8")
parser.add_argument("--no-camera", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = args_cli.door_families
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import DoorOpening.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@hydra_task_config(args_cli.task if hasattr(args_cli, "task") else "DooropeningMulti", "rl_games_cfg_entry_point")
def main(env_cfg, _agent_cfg):
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.pointcloud_render_mode = "none" if args_cli.no_camera else "depth"
    env_cfg.enable_pointcloud_camera = not args_cli.no_camera
    env_cfg.pointcloud_camera_update_period = 0.0
    preconvert_shared_urdf_assets()
    env = gym.make("DooropeningMulti", cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    actions = torch.zeros((base_env.num_envs, int(base_env.cfg.action_space)), device=device)
    env.reset()
    for _ in range(int(args_cli.warmup)):
        env.step(actions)
    sync(device)
    samples = []
    for _ in range(int(args_cli.iters)):
        sync(device)
        t0 = time.perf_counter()
        env.step(actions)
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    env.close()
    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    label = "NO_CAMERA_BASELINE" if args_cli.no_camera else "ISAAC_CAMERA"
    print(f"{label} num_envs={args_cli.num_envs} camera=640x480 update_period=0")
    print(f"step_ms_p50={p50:.3f} step_ms_mean={statistics.fmean(samples):.3f} step_ms_p95={p95:.3f}")
    print(f"envs_per_sec={1000.0 * args_cli.num_envs / p50:.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
