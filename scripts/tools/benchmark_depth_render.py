#!/usr/bin/env python3
"""Benchmark the analytic point-cloud depth rendering used by the distillation pipeline.

Times three things across a sweep of batch sizes (parallel envs), on the real GT cloud (door mesh +
wall distractors, built the same way as render_wall_configs_viser / render_depth_roundtrip_viser):

    project      : rasterize_depth_zbuffer_from_pose        (points -> z-buffer depth image)
    backproject  : backproject_depth_to_world_from_pose     (depth image -> world points)
    roundtrip    : project + backproject                    (the full points -> depth -> points path)
    sim_render   : simulate_depth_cam_render_from_pose       (the training render, incl. jitter + the
                   fixed-N padded output; optionally torch.compile'd)

For each config it runs `--warmup` untimed iterations then `--iters` timed ones (with CUDA sync),
and reports median / mean / p95 latency plus throughput (envs/s and Mpoints/s of input cloud).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPO_ROOT / "scripts" / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from DoorOpening.utils.camera_utils import (
    backproject_depth_to_world_from_pose,
    rasterize_depth_zbuffer_from_pose,
    simulate_depth_cam_render_from_pose,
)
from render_depth_roundtrip_viser import (
    DEFAULT_DOOR_URDF,
    DEFAULT_STUDENT_CFG,
    build_camera_spec,
    load_door_asset,
    look_at_camera_pose,
)


def build_gt_cloud(args, device):
    """The GT input cloud (door mesh + wall distractors), a single (1, N, 3) tensor in world coords."""
    import math

    import yaml

    from DoorOpening.utils.wall_distractors import (
        WallDistractorParams,
        compute_wall_bbox_ordering,
        sample_wall_points_local,
    )

    cfg = yaml.safe_load(Path(args.student_cfg).read_text()) or {}
    wall_cfg = dict(cfg.get("dagger", {}).get("wall_distractors", {}))
    scene_door_num_points = int(cfg.get("scene_door_num_points", cfg.get("door_pcd_num_points", 30000)))
    board_num_points = int(args.board_num_points or scene_door_num_points)

    board_bbox, board_gt = load_door_asset(args.door, board_num_points, device)
    yaw = math.radians(args.door_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    parts = [board_gt @ R.T]
    n_walls = 0
    wall_params = WallDistractorParams.from_cfg(wall_cfg, scene_door_num_points)
    if not args.no_walls and wall_params.enabled and wall_params.num_points > 0:
        axis_order, bmin, bmax = compute_wall_bbox_ordering(board_bbox)
        walls = sample_wall_points_local(
            axis_order=axis_order, bbox_min_ordered=bmin, bbox_max_ordered=bmax,
            num_points=wall_params.num_points, params=wall_params, device=device,
        )
        parts.append(walls @ R.T)
        n_walls = wall_params.num_points
    gt = torch.cat(parts, dim=1)
    # simulate_depth_cam_render's fixed-shape compiled path can't have NaN rows; drop padding.
    finite = torch.isfinite(gt).all(dim=-1)
    gt = gt[:, finite[0]]
    return gt, board_num_points, n_walls


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return samples


def summarize(samples):
    s = sorted(samples)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return statistics.median(s), statistics.fmean(s), p95


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256], help="Parallel envs to sweep.")
    p.add_argument("--iters", type=int, default=50, help="Timed iterations per config.")
    p.add_argument("--warmup", type=int, default=10, help="Untimed warmup iterations per config.")
    p.add_argument("--student-cfg", type=Path, default=DEFAULT_STUDENT_CFG)
    p.add_argument("--door", type=Path, default=DEFAULT_DOOR_URDF)
    p.add_argument("--door-yaw-deg", type=float, default=-90.0)
    p.add_argument("--board-num-points", type=int, default=None)
    p.add_argument("--no-walls", action="store_true")
    p.add_argument("--standoff", type=float, default=1.0)
    p.add_argument("--camera-height", type=float, default=1.0)
    p.add_argument("--camera-look-z", type=float, default=1.0)
    p.add_argument("--camera-right", type=float, default=0.12)
    p.add_argument("--cam-width-px", type=int, default=320)
    p.add_argument("--cam-height-px", type=int, default=240)
    p.add_argument("--near-m", type=float, default=0.3)
    p.add_argument("--far-m", type=float, default=3.0)
    p.add_argument("--inflate-px", type=int, default=2)
    p.add_argument("--sim-num-points", type=int, default=12000, help="Fixed output points for simulate_depth_cam_render.")
    p.add_argument("--jitter-std-m", type=float, default=0.004)
    p.add_argument("--use-compile", action="store_true", help="Also benchmark the torch.compile sim_render path (slow first call).")
    p.add_argument("--skip-sim-render", action="store_true", help="Only benchmark the low-level project/backproject/roundtrip.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    gt, n_door, n_walls = build_gt_cloud(args, device)
    n_pts = gt.shape[1]
    cam_spec = build_camera_spec(args.cam_width_px, args.cam_height_px, args.near_m, args.far_m, device)
    eye = np.array([args.camera_right, -args.standoff, args.camera_height], dtype=np.float32)
    target = np.array([0.0, 0.0, args.camera_look_z], dtype=np.float32)
    cam_pose1 = torch.from_numpy(look_at_camera_pose(eye, target)).to(device)

    print(f"[INFO] device        : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[INFO] GT cloud       : {n_pts} pts  (door {n_door} + walls {n_walls})")
    print(f"[INFO] camera         : {args.cam_width_px}x{args.cam_height_px}px, range [{args.near_m}, {args.far_m}] m, inflate {args.inflate_px}px")
    print(f"[INFO] warmup/iters   : {args.warmup}/{args.iters}   batch sizes: {args.batch_sizes}")
    print()
    header = f"{'bench':<12}{'batch':>6}{'in_pts':>10}{'median_ms':>11}{'mean_ms':>10}{'p95_ms':>9}{'envs/s':>10}{'Mpts/s':>9}"
    print(header)
    print("-" * len(header))

    for B in args.batch_sizes:
        pcd = gt.expand(B, -1, -1).contiguous()
        cam_pose = cam_pose1.unsqueeze(0).expand(B, -1).contiguous()

        # precompute a depth image so backproject can be timed in isolation
        depth, intr = rasterize_depth_zbuffer_from_pose(pcd, cam_pose, cam_spec, inflate_px=args.inflate_px, clip_mode="post")

        def _project():
            return rasterize_depth_zbuffer_from_pose(pcd, cam_pose, cam_spec, inflate_px=args.inflate_px, clip_mode="post")

        def _roundtrip():
            d, i = _project()
            return backproject_depth_to_world_from_pose(d, cam_pose, i)

        benches = {
            "project": _project,
            "backproject": lambda: backproject_depth_to_world_from_pose(depth, cam_pose, intr),
            "roundtrip": _roundtrip,
        }

        if not args.skip_sim_render:
            def _sim():
                return simulate_depth_cam_render_from_pose(
                    pcd=pcd, camera_pose=cam_pose, num_points=args.sim_num_points,
                    inflate_px=args.inflate_px, jitter_std_m=args.jitter_std_m,
                    cam_spec_dict=cam_spec, clip_mode="post", jitter_mode="xyz",
                    use_compile=args.use_compile,
                )
            benches["sim_render"] = _sim

        for name, fn in benches.items():
            median, mean, p95 = summarize(time_fn(fn, args.warmup, args.iters, device))
            envs_per_s = B / (median / 1e3)
            mpts_per_s = (B * n_pts) / (median / 1e3) / 1e6
            print(f"{name:<12}{B:>6}{n_pts:>10}{median:>11.3f}{mean:>10.3f}{p95:>9.3f}{envs_per_s:>10.1f}{mpts_per_s:>9.1f}")
        print()


if __name__ == "__main__":
    main()
