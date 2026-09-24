#!/usr/bin/env python3
"""Build a compact Viser sequence of exact-training wall configurations."""
from pathlib import Path
import sys
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))
from DoorOpening.utils.urdf_utils import compute_exact_door_keypoints
from DoorOpening.utils.wall_distractors import WallDistractorParams, compute_wall_bbox_ordering, sample_wall_points_local


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--door", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=100)
    args = p.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    keypoints = compute_exact_door_keypoints(str(args.door))
    full_bbox = torch.tensor(keypoints["door_full_bbox_base"], dtype=torch.float32, device=device).unsqueeze(0)
    panel_bbox = torch.tensor(keypoints["link_1_bbox_base"], dtype=torch.float32, device=device).unsqueeze(0)
    axis_order, full_min, full_max = compute_wall_bbox_ordering(full_bbox)
    panel_min = torch.gather(panel_bbox[:, 0], 1, axis_order)
    panel_max = torch.gather(panel_bbox[:, 1], 1, axis_order)
    cfg = {
        "enabled": True, "num_points": 30000, "side_margin_m": [0.35, 0.80],
        "edge_gap_m": [0.015, 0.04], "depth_m": [0.10, 0.80],
        "center_offset_m": [-0.30, 0.30], "face_jitter_m": 0.004,
        "flush_prob": 0.7, "flush_point_fraction": 0.5,
        "flush_extent_m": [0.6, 1.6], "detached_within_flush_frac": [0.0, 1.0],
    }
    params = WallDistractorParams.from_cfg(cfg, 30000)
    base_frame = payload["frames"][0]
    frames = []
    for i in range(int(args.num_frames)):
        torch.manual_seed(17000 + i)
        walls = sample_wall_points_local(
            axis_order=axis_order, bbox_min_ordered=full_min, bbox_max_ordered=full_max,
            num_points=30000, params=params, device=device,
            flush_bbox_min_ordered=panel_min, flush_bbox_max_ordered=panel_max,
        )[0]
        clouds = dict(base_frame["pointclouds"])
        walls_f16 = walls.to(torch.float16)
        clouds["ground_truth_walls"] = walls_f16
        # Keep the diagnostic wall stream synchronized with the sampled configuration.
        # This sequence is intentionally a wall-layout sweep; the pre-rendered policy
        # cloud remains the fixed reference from the accepted initial pose.
        clouds["walls_visible"] = walls_f16
        door_gt = clouds.get("ground_truth_door")
        robot_gt = clouds.get("ground_truth_robot")
        if door_gt is not None and robot_gt is not None:
            clouds["ground_truth"] = torch.cat((door_gt, walls_f16, robot_gt), dim=0)
        frames.append({"pointclouds": clouds})
    payload["frames"] = frames
    payload["wall_sequence"] = {"num_frames": int(args.num_frames), "seed_base": 17000, "sampler": "multi_pcd_dagger.sample_wall_points_local"}
    torch.save(payload, args.output)
    print(f"saved={args.output} frames={len(frames)}")


if __name__ == "__main__":
    main()
