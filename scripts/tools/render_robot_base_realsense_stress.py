#!/usr/bin/env python3
"""Replay RealSense-style z-buffer clouds while the robot base approaches the door."""
from pathlib import Path
import argparse
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "scripts" / "tools")]
import render_dex_style_wall_scene as rd
from DoorOpening.utils.camera_utils import (
    rasterize_axis_aligned_boxes_depth_from_pose,
    rasterize_depth_zbuffer_from_pose,
    backproject_depth_to_world_from_pose,
)
from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler
from DoorOpening.utils.wall_distractors import (
    WallDistractorParams, compute_wall_bbox_ordering, move_flush_wall_behind_panel, sample_wall_points_local,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--door", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wall-points", type=int, default=60000)
    p.add_argument("--face-jitter", type=float, default=0.012)
    p.add_argument("--approaches", type=float, nargs="+", default=[0.0, 0.20, 0.40, 0.60, 0.80])
    p.add_argument("--grid", type=int, default=0, help="Use an N x N door-angle/base-approach grid.")
    p.add_argument("--wall-res", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Render only the solid wall occluder at W x H, then nearest-upsample it.")
    p.add_argument("--base-yaws", type=float, nargs="+", default=[-0.20, 0.0, 0.20])
    p.add_argument("--base-y-offsets", type=float, nargs="+", default=[0.0])
    args = p.parse_args()
    dev = torch.device("cpu")
    init = rd.load_initial_state(args.door)
    sampler = FrankaGripperSampler(str(args.door), device=dev, num_points=18000)
    full, panel, *_ = rd.load_door_asset(args.door, 30000, dev)
    axis, fmin, fmax = compute_wall_bbox_ordering(full)
    pmin = torch.gather(panel[:, 0], 1, axis); pmax = torch.gather(panel[:, 1], 1, axis)
    cfg = {"enabled": True, "num_points": args.wall_points, "side_margin_m": [0.35, 0.80],
           "edge_gap_m": [0.015, 0.04], "depth_m": [0.10, 0.80], "center_offset_m": [-0.30, 0.30],
           "face_jitter_m": float(args.face_jitter), "flush_prob": 0.7, "flush_point_fraction": 0.5,
           "flush_extent_m": [0.6, 1.6], "detached_within_flush_frac": [0.0, 1.0]}
    params = WallDistractorParams.from_cfg(cfg, args.wall_points)
    door_pos_np, door_R_np = rd._pose_from_initial_state(init, "door_world_pose")
    door_pos = torch.tensor(door_pos_np, dtype=torch.float32)
    door_R = torch.tensor(door_R_np, dtype=torch.float32)
    if args.grid > 0:
        angles = torch.linspace(0.0, 1.57, args.grid).tolist()
        approaches = torch.linspace(0.0, 0.80, args.grid).tolist()
        cases = [(float(angle), float(approach)) for angle in angles for approach in approaches]
    else:
        cases = [(0.0, float(approach), float(yaw), float(y_offset))
                 for approach in args.approaches
                 for yaw in args.base_yaws
                 for y_offset in args.base_y_offsets]
    frames = []
    penetration_rows = []
    if args.grid > 0:
        cases = [(angle, approach, yaw, y_offset)
                 for angle, approach in cases
                 for yaw in args.base_yaws
                 for y_offset in args.base_y_offsets]
    for i, (angle, approach, base_yaw, base_y_offset) in enumerate(cases):
        torch.manual_seed(24000 + i)
        walls, wall_boxes = sample_wall_points_local(
            axis, fmin, fmax, args.wall_points, params, dev, pmin, pmax, return_boxes=True
        )
        robot, camera, *_ = rd.load_ready_robot(
            dev, -0.62, init, base_approach_m=float(approach), arm_down_rad=0.15,
            base_yaw_rad=float(base_yaw), base_y_offset_m=float(base_y_offset),
        )
        cam_pos_base = (camera[:, :3] - door_pos.view(1, 1, 3))
        cam_pos_base = torch.bmm(cam_pos_base, door_R.T.view(1, 3, 3)).squeeze(1)
        walls = move_flush_wall_behind_panel(walls, axis, pmin, pmax, cam_pos_base, params)[0]
        walls = walls[torch.isfinite(walls).all(-1)] @ door_R.T + door_pos
        # Transform sampled local wall boxes to conservative world AABBs for the exact solid occluder.
        bmin, bmax = wall_boxes[:, :, 0], wall_boxes[:, :, 1]
        corners = torch.stack(torch.meshgrid(
            torch.tensor([0., 1.]), torch.tensor([0., 1.]), torch.tensor([0., 1.]), indexing="ij"
        ), dim=-1).reshape(1, 1, 8, 3)
        box_corners = bmin[:, :, None, :] + corners * (bmax - bmin)[:, :, None, :]
        box_corners = box_corners @ door_R.T + door_pos.view(1, 1, 1, 3)
        solid_bmin, solid_bmax = box_corners.amin(dim=2), box_corners.amax(dim=2)
        q = torch.zeros((1, len(sampler.robot.actuated_joint_names)), device=dev)
        q[0, 0] = float(angle)
        door_local = torch.cat([sampler.sample_link_set(q, [name])[0] for name in ("link_0", "link_1", "link_2")])
        door = door_local @ door_R.T + door_pos
        cam = rd.build_camera_spec(320, 240, 0.25, 3.0, dev)
        wall_cam = cam if args.wall_res is None else rd.build_camera_spec(
            int(args.wall_res[0]), int(args.wall_res[1]), 0.25, 3.0, dev
        )
        door_d, intr = rasterize_depth_zbuffer_from_pose(door.unsqueeze(0), camera, cam, inflate_px=2, clip_mode="post")
        wall_d, _ = rasterize_depth_zbuffer_from_pose(walls.unsqueeze(0), camera, cam, inflate_px=2, clip_mode="post")
        wall_solid_d, _ = rasterize_axis_aligned_boxes_depth_from_pose(solid_bmin, solid_bmax, camera, wall_cam)
        if args.wall_res is not None:
            # Do not nearest-copy a reduced wall depth image.  A narrow projected wall
            # column then becomes a stack of identical full-resolution columns, which is
            # the striping visible in the Viser preview.  Interpolate depth and validity
            # separately so +inf background does not contaminate neighboring wall pixels.
            wall_valid = torch.isfinite(wall_solid_d)
            wall_depth_filled = torch.where(wall_valid, wall_solid_d, torch.zeros_like(wall_solid_d))
            wall_num = F.interpolate(
                wall_depth_filled.unsqueeze(1),
                size=(int(cam["H"]), int(cam["W"])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            wall_den = F.interpolate(
                wall_valid.to(wall_solid_d.dtype).unsqueeze(1),
                size=(int(cam["H"]), int(cam["W"])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            wall_solid_d = torch.where(
                wall_den > 1e-6,
                wall_num / wall_den.clamp_min(1e-6),
                torch.full_like(wall_num, float("inf")),
            )
        robot_d, _ = rasterize_depth_zbuffer_from_pose(robot, camera, cam, inflate_px=2, clip_mode="post")
        door_mask = torch.isfinite(door_d[0])
        wall_in_door = door_mask & torch.isfinite(wall_solid_d[0]) & (wall_solid_d[0] < door_d[0])
        penetration_pixels = int(wall_in_door.sum())
        penetration_rows.append((float(angle), float(approach), float(base_yaw), float(base_y_offset), penetration_pixels))
        # Match the production source-aware renderer: behind/tied wall samples cannot leak through
        # the door, while a genuinely foreground wall remains visible.
        # Use the analytic solid-wall pass for occlusion.  The sampled wall pass is
        # retained in the replay as geometry, but must not replace the solid depth.
        wall_for_scene = torch.where(
            torch.isfinite(door_d[0]) & (wall_solid_d[0] >= door_d[0] - 0.002),
            torch.full_like(wall_solid_d[0], float("inf")), wall_solid_d[0],
        )
        wall_for_scene_2d = wall_for_scene[0] if wall_for_scene.ndim == 3 else wall_for_scene
        scene_d = torch.minimum(door_d[0], wall_for_scene_2d).unsqueeze(0)
        robot_mask = torch.isfinite(robot_d)
        hole_mask = torch.nn.functional.max_pool2d(robot_mask.to(scene_d.dtype).unsqueeze(1), 3, 1, 1).squeeze(1) > 0
        scene_d = torch.where(hole_mask, torch.full_like(scene_d, float("inf")), scene_d)
        depth = torch.minimum(scene_d, robot_d)
        rendered, valid = backproject_depth_to_world_from_pose(depth, camera, intr)
        policy = rendered[0][valid[0]].to(torch.float16)
        robot_pixels = torch.isfinite(robot_d[0]) & (robot_d[0] <= scene_d[0])
        scene_pixels = torch.isfinite(depth[0]) & ~robot_pixels
        robot_visible = rendered[0][robot_pixels].to(torch.float16)
        scene_visible = rendered[0][scene_pixels].to(torch.float16)
        print(f"angle={angle:.3f} approach={approach:.2f} policy_points={len(policy)} robot_occluding_pixels={int(robot_pixels.sum())} wall_in_door_pixels={penetration_pixels}")
        frames.append({"pointclouds": {
            "policy_input": policy, "door": door.to(torch.float16), "walls": walls.to(torch.float16),
            "robot": robot[0].to(torch.float16), "visible_robot": robot_visible,
            "visible_scene": scene_visible,
        }, "metadata": {"base_approach_m": float(approach), "door_joint_1": float(angle),
                           "robot_occluding_pixels": int(robot_pixels.sum()), "wall_in_door_pixels": penetration_pixels}})
    payload = {"format": "dooropening_viser_replay_v1", "pointcloud_frame": "world",
               "pointcloud_source": "robot_base_realsense_zbuffer_stress",
               "pointcloud_streams": [
                   {"name": "policy_input", "label": "RealSense-style camera cloud", "color": (120, 190, 230)},
                   {"name": "door", "label": "Door geometry", "color": (255, 193, 7)},
                   {"name": "walls", "label": "Wall distractors", "color": (90, 90, 220)},
                   {"name": "robot", "label": "Robot geometry", "color": (0, 170, 120)},
                   {"name": "visible_robot", "label": "Robot pixels winning z-buffer", "color": (255, 80, 40)},
                   {"name": "visible_scene", "label": "Door/wall pixels not occluded by robot", "color": (180, 180, 180)},
               ], "frame_dt": 0.5, "frame_fps": 2.0, "frames": frames,
               "stress_test": {"cases": penetration_rows}}
    torch.save(payload, args.output); print(f"saved={args.output} frames={len(frames)}")


if __name__ == "__main__":
    main()
