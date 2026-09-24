#!/usr/bin/env python3
"""Compare closed-pose wall filtering with open-door swept-volume filtering."""
from pathlib import Path
import argparse
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "scripts" / "tools")]

from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler
from DoorOpening.utils.wall_distractors import (
    WallDistractorParams,
    compute_wall_bbox_ordering,
    sample_wall_points_local,
)
import render_dex_style_wall_scene as rd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--door", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wall-points", type=int, default=120000)
    p.add_argument("--angles", type=float, nargs="+", default=[0.0, 0.4, 0.8, 1.2, 1.57])
    args = p.parse_args()
    dev = torch.device("cpu")
    sampler = FrankaGripperSampler(str(args.door), device=dev, num_points=20000)
    full_bbox, panel_bbox, *_ = rd.load_door_asset(args.door, 30000, dev)
    axis, fmin, fmax = compute_wall_bbox_ordering(full_bbox)
    pmin = torch.gather(panel_bbox[:, 0], 1, axis)
    pmax = torch.gather(panel_bbox[:, 1], 1, axis)
    cfg = {
        "enabled": True, "num_points": args.wall_points,
        "side_margin_m": [0.35, 0.80], "edge_gap_m": [0.015, 0.04],
        "depth_m": [0.10, 0.80], "center_offset_m": [-0.30, 0.30],
        "face_jitter_m": 0.004, "flush_prob": 0.7, "flush_point_fraction": 0.5,
        "flush_extent_m": [0.6, 1.6], "detached_within_flush_frac": [0.0, 1.0],
    }
    params = WallDistractorParams.from_cfg(cfg, args.wall_points)
    frames = []
    for i, angle in enumerate(args.angles):
        torch.manual_seed(17000 + i)
        walls = sample_wall_points_local(axis, fmin, fmax, args.wall_points, params, dev, pmin, pmax)[0]
        walls = walls[torch.isfinite(walls).all(-1)]
        q = torch.zeros((1, len(sampler.robot.actuated_joint_names)), device=dev)
        q[0, 0] = float(angle)  # joint_1 is the door swing joint in this asset.
        door = torch.cat([sampler.sample_link_set(q, [name])[0] for name in ("link_0", "link_1", "link_2")])
        # Current behavior: walls filtered against the closed-pose bbox by the sampler.
        current = walls
        # Proposed behavior: filter against the actual door geometry at this pose.
        lo, hi = door.min(0).values, door.max(0).values
        inside = ((walls >= lo) & (walls <= hi)).all(-1)
        swept_safe = walls[~inside]
        print(f"angle={angle:.3f} current={len(current)} swept_safe={len(swept_safe)} removed={int(inside.sum())}")
        frames.append({"pointclouds": {
            "door_open": door.to(torch.float16),
            "walls_current_closed_filter": current.to(torch.float16),
            "walls_swept_filter": swept_safe.to(torch.float16),
        }, "metadata": {"door_joint_1": float(angle), "removed_by_swept_filter": int(inside.sum())}})
    payload = {
        "format": "dooropening_viser_replay_v1",
        "pointcloud_frame": "door_base",
        "pointcloud_source": "open_door_wall_filter_stress_test",
        "pointcloud_streams": [
            {"name": "door_open", "label": "Door geometry at joint_1 angle", "color": (255, 193, 7)},
            {"name": "walls_current_closed_filter", "label": "Current walls: closed-pose filter", "color": (220, 60, 60)},
            {"name": "walls_swept_filter", "label": "Proposed walls: open-pose overlap filter", "color": (60, 180, 90)},
        ],
        "frame_dt": 0.5, "frame_fps": 2.0, "frames": frames,
        "stress_test": {"angles": args.angles, "wall_points": args.wall_points},
    }
    torch.save(payload, args.output)
    print(f"saved={args.output} frames={len(frames)}")


if __name__ == "__main__":
    main()
