#!/usr/bin/env python3
"""Compare the current depth z-buffer projection with a fused Triton prototype."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "scripts" / "tools")]
from DoorOpening.utils.camera_utils import rasterize_depth_zbuffer_from_pose
from benchmark_depth_render import build_gt_cloud
from render_depth_roundtrip_viser import DEFAULT_STUDENT_CFG, build_camera_spec, look_at_camera_pose


def timing(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), statistics.fmean(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--door", type=Path, required=True)
    p.add_argument("--student-cfg", type=Path, default=DEFAULT_STUDENT_CFG)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 400])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--board-num-points", type=int, default=None)
    p.add_argument("--no-walls", action="store_true")
    p.add_argument("--door-yaw-deg", type=float, default=-90)
    args = p.parse_args()
    device = torch.device("cuda")
    gt, _, _ = build_gt_cloud(args, device)
    spec = build_camera_spec(160, 120, 0.3, 3.0, device)
    import numpy as np
    pose = torch.from_numpy(look_at_camera_pose(np.array([0.12, -1.0, 1.0], dtype=np.float32),
                                                np.array([0, 0, 1], dtype=np.float32))).to(device)
    print(f"GPU={torch.cuda.get_device_name()} points_per_env={gt.shape[1]}", flush=True)
    for B in args.batch_sizes:
        cloud = gt.expand(B, -1, -1).contiguous()
        poses = pose.unsqueeze(0).expand(B, -1).contiguous()
        def baseline():
            return rasterize_depth_zbuffer_from_pose(cloud, poses, spec, inflate_px=0, clip_mode="post")[0]
        def candidate():
            return rasterize_depth_zbuffer_from_pose(
                cloud, poses, spec, inflate_px=0, clip_mode="post", use_triton=True,
            )[0]
        ref = baseline()
        got = candidate()
        torch.cuda.synchronize()
        same_finite = bool(torch.equal(torch.isfinite(ref), torch.isfinite(got)))
        common = torch.isfinite(ref) & torch.isfinite(got)
        max_diff = float((ref[common] - got[common]).abs().max()) if bool(common.any()) else 0.0
        bm, _ = timing(baseline, args.warmup, args.iters)
        tm, _ = timing(candidate, args.warmup, args.iters)
        print(f"B={B} torch_ms={bm:.3f} triton_ms={tm:.3f} speedup={bm/tm:.2f}x "
              f"finite_mask_equal={same_finite} max_depth_diff_m={max_diff:.7f}", flush=True)


if __name__ == "__main__":
    main()
