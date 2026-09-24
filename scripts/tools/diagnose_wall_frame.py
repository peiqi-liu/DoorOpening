#!/usr/bin/env python3
"""Deterministic frame and occlusion checks for DoorOpening wall distractors.

This is intentionally simulator-free: it audits a saved evaluation replay and exercises
the shared camera renderer with a rotated, fixed wall box from two robot/camera poses.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from DoorOpening.utils.camera_utils import (
    backproject_depth_to_world_from_pose,
    build_realsense_sampler_spec,
    rasterize_axis_aligned_boxes_depth_from_pose,
    rasterize_depth_zbuffer_from_pose,
    rasterize_oriented_boxes_depth_from_pose,
)
from DoorOpening.utils.wall_distractors import move_flush_wall_behind_panel


def _rotate_xyzw(q: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xyz, w = q[..., :3], q[..., 3:4]
    while xyz.ndim < points.ndim:
        xyz, w = xyz.unsqueeze(-2), w.unsqueeze(-2)
    uv = torch.cross(xyz.expand_as(points), points, dim=-1)
    return points + 2.0 * (w * uv + torch.cross(xyz.expand_as(points), uv, dim=-1))


def _world_to_box_local(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    return _rotate_xyzw(torch.cat([-pose[3:6], pose[6:7]]), points - pose[:3])


def _nearest_distances(points: torch.Tensor, reference: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    return torch.cat([torch.cdist(part, reference).amin(dim=1) for part in points.split(chunk)], dim=0)


def _replay_metrics(recording: Path, output_dir: Path) -> dict:
    payload = torch.load(recording, map_location="cpu", weights_only=False)
    walls = payload.get("static_pointclouds", {}).get("ground_truth_walls")
    if walls is None:
        raise ValueError(f"{recording} has no static ground_truth_walls cloud")
    walls = walls[torch.isfinite(walls).all(dim=1)].float()
    frames = payload["frames"]
    selected = sorted(set([0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]))
    per_frame = []
    wall_layers = []
    for idx in selected:
        frame = frames[idx]
        cloud = frame["robot_depth_cam_obs_points_world"].float()
        cloud = cloud[torch.isfinite(cloud).all(dim=1)]
        distance = _nearest_distances(cloud, walls)
        wall_layers.append((idx, cloud[distance < 0.05]))
        camera = frame["sampler_camera_pose_env_xyzw"].float()
        camera_local = _rotate_xyzw(torch.cat([-camera[3:6], camera[6:7]]), cloud - camera[:3])
        camera_roundtrip = _rotate_xyzw(camera[3:7], camera_local) + camera[:3]
        item = {
            "frame": idx,
            "base_pos_world": [float(x) for x in frame["pointcloud_base_pos_w"]],
            "camera_pos_world": [float(x) for x in camera[:3]],
            "finite_rendered_points": int(cloud.shape[0]),
            "camera_forward_fraction": float((camera_local[:, 0] > 0.0).float().mean()),
            "camera_world_roundtrip_max_m": float((camera_roundtrip - cloud).norm(dim=1).max()),
            "wall_returns_within_5cm": int((distance < 0.05).sum()),
            "wall_return_fraction_within_5cm": float((distance < 0.05).float().mean()),
            "wall_distance_median_m": float(distance.median()),
            "wall_distance_p95_m": float(torch.quantile(distance, 0.95)),
        }
        policy = frame.get("policy_input_points_world")
        if policy is not None:
            policy = policy.float()
            finite_policy = torch.isfinite(policy).all(dim=1)
            policy = policy[finite_policy]
            base_pos = frame["pointcloud_base_pos_w"].float()
            base_quat = frame["pointcloud_base_quat_w"].float()
            policy_local = _rotate_xyzw(torch.cat([-base_quat[1:], base_quat[:1]]), policy - base_pos)
            padding = policy_local.abs().amax(dim=1) == 0.0
            policy_roundtrip = _rotate_xyzw(torch.cat([base_quat[1:], base_quat[:1]]), policy_local) + base_pos
            item.update(
                {
                    "policy_finite_points": int(policy.shape[0]),
                    "policy_zero_padding_rendered_at_base": int(padding.sum()),
                    "policy_current_base_roundtrip_max_m": float((policy_roundtrip - policy).norm(dim=1).max()),
                }
            )
        per_frame.append(item)
    base_delta = frames[-1]["pointcloud_base_pos_w"] - frames[0]["pointcloud_base_pos_w"]
    base_positions = torch.stack([frame["pointcloud_base_pos_w"].float() for frame in frames])
    # Replay artifact: the gray layer is saved once in environment-world space;
    # each color is a wall return from a later camera pose.  All color layers
    # sit on the same gray geometry while the X markers show base travel.
    palette = ((0, 95, 210), (0, 150, 100), (235, 140, 0), (170, 65, 190), (210, 45, 60))
    overlay = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(overlay)
    xy_parts = [walls[:, :2]] + [points[:, :2] for _, points in wall_layers] + [frame["pointcloud_base_pos_w"][:2][None] for frame in frames[:: max(1, len(frames) // 40)]]
    all_xy = torch.cat(xy_parts).numpy()
    lo, hi = all_xy.min(axis=0) - 0.12, all_xy.max(axis=0) + 0.12
    scale = min(760.0 / (hi[0] - lo[0]), 560.0 / (hi[1] - lo[1]))
    def pixel(xy):
        return (70 + (float(xy[0]) - lo[0]) * scale, 630 - (float(xy[1]) - lo[1]) * scale)
    for point in walls[:: max(1, len(walls) // 7000)]:
        x0, y0 = pixel(point[:2])
        draw.point((x0, y0), fill=(165, 165, 165))
    for color, (idx, points) in zip(palette, wall_layers):
        for point in points[:: max(1, len(points) // 1400)]:
            x0, y0 = pixel(point[:2])
            draw.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill=color)
        base = frames[idx]["pointcloud_base_pos_w"]
        x0, y0 = pixel(base[:2])
        draw.line((x0 - 5, y0 - 5, x0 + 5, y0 + 5), fill=color, width=2)
        draw.line((x0 - 5, y0 + 5, x0 + 5, y0 - 5), fill=color, width=2)
    draw.text((20, 20), "Gray: static saved world-wall layer. Colors: rendered wall returns at five replay frames.", fill="black")
    draw.text((20, 45), "Colored X markers: robot-base position at the same frame.", fill="black")
    overlay.save(output_dir / "replay_world_wall_overlay.png")
    result = {
        "recording": str(recording),
        "recorded_frames": len(frames),
        "static_wall_points": int(walls.shape[0]),
        "base_displacement_first_to_last_m": float(base_delta.norm()),
        "base_max_pairwise_displacement_m": float(torch.cdist(base_positions, base_positions).max()),
        "world_wall_overlay": str(output_dir / "replay_world_wall_overlay.png"),
        "sampled_frames": per_frame,
    }
    # The replay URDF's legacy path applies virtual base joint values directly
    # below a fixed root. Compare that visual chassis path with the recorded
    # tidybot2_base_link pose that was actually used for point-cloud conversion.
    names = payload.get("compact_target_joint_names", ())
    if tuple(names[:3]) == ("base_x_joint", "base_y_joint", "base_rotation_joint"):
        physical = torch.stack([frame["pointcloud_base_pos_w"].float() for frame in frames])
        roots = torch.stack([frame["robot_base_pos_w"].float() for frame in frames])
        compact = torch.stack([frame["compact_q"].float() for frame in frames])
        legacy = roots.clone()
        legacy[:, :2] += compact[:, :2]
        error = (legacy - physical).norm(dim=1)
        canvas = Image.new("RGB", (900, 700), "white")
        draw = ImageDraw.Draw(canvas)
        xy = torch.cat([walls[:, :2], physical[:, :2], legacy[:, :2]]).numpy()
        lo, hi = xy.min(axis=0) - 0.12, xy.max(axis=0) + 0.12
        scale = min(760.0 / (hi[0] - lo[0]), 560.0 / (hi[1] - lo[1]))
        def path_pixel(point):
            return (70 + (float(point[0]) - lo[0]) * scale, 630 - (float(point[1]) - lo[1]) * scale)
        for point in walls[:: max(1, len(walls) // 7000)]:
            draw.point(path_pixel(point[:2]), fill=(170, 170, 170))
        draw.line([path_pixel(point[:2]) for point in physical[::5]], fill=(0, 150, 100), width=3)
        draw.line([path_pixel(point[:2]) for point in legacy[::5]], fill=(220, 45, 60), width=3)
        draw.text((20, 20), "Gray: static world walls. Green: recorded physical pointcloud base. Red: legacy Viser URDF chassis path.", fill="black")
        draw.text((20, 45), "Red uses virtual base joints under a fixed root and is sign-inverted from the physical chassis.", fill="black")
        path = output_dir / "replay_legacy_vs_physical_robot_path.png"
        canvas.save(path)
        result["legacy_robot_replay"] = {
            "visual": str(path),
            "chassis_error_mean_m": float(error.mean()),
            "chassis_error_max_m": float(error.max()),
            "chassis_error_at_sampled_frames_m": {
                str(idx): float(error[idx]) for idx in selected
            },
        }
    return result


def _controlled_renderer_check(output_dir: Path) -> dict:
    # A fixed thin wall box, rotated 45 degrees in the world.  The two camera poses
    # lie one metre apart along its local forward axis: a robot backing away.
    angle = math.pi / 4.0
    q = torch.tensor([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])
    box_pose = torch.tensor([[0.35, -0.20, 0.0, *q.tolist()]])
    box_min = torch.tensor([[[1.00, -0.40, 0.45]]])
    box_max = torch.tensor([[[1.15, 0.40, 1.55]]])
    camera_local = torch.tensor([[-1.00, 0.00, 1.00], [-2.00, 0.12, 1.00]])
    camera_world = _rotate_xyzw(q.expand(2, -1), camera_local) + box_pose[0, :3]
    cameras = torch.cat([camera_world, q.expand(2, -1)], dim=1)
    # Match the shared half-resolution D435 renderer (240x320), rather than
    # the reduced wall-sample raster.  The analytic fallback runs at this final
    # resolution so its silhouette cannot be reopened by depth upsampling.
    spec = build_realsense_sampler_spec(240, 320, near_m=0.1, far_m=5.0)
    mins, maxs, poses = box_min.expand(2, -1, -1), box_max.expand(2, -1, -1), box_pose.expand(2, -1)
    oriented_depth, intrinsics = rasterize_oriented_boxes_depth_from_pose(mins, maxs, poses, cameras, spec)
    reconstructed, valid = backproject_depth_to_world_from_pose(oriented_depth, cameras, intrinsics)
    residuals = []
    world_layers = []
    for idx in range(2):
        world_points = reconstructed[idx][valid[idx]]
        local = _world_to_box_local(world_points, box_pose[0])
        face_residual = torch.stack(
            [
                (local[:, axis] - box_min[0, 0, axis]).abs()
                for axis in range(3)
            ]
            + [
                (local[:, axis] - box_max[0, 0, axis]).abs()
                for axis in range(3)
            ],
            dim=1,
        ).amin(dim=1)
        residuals.append(face_residual)
        world_layers.append(world_points)

    # Legacy pass: transform the rotated OBB to its enclosing world AABB.  Its
    # mask necessarily covers empty world space at 45 degrees.
    corners_local = torch.stack(torch.meshgrid(*([torch.tensor([0.0, 1.0])] * 3), indexing="ij"), dim=-1).reshape(8, 3)
    corners_local = box_min[0, 0] + corners_local * (box_max[0, 0] - box_min[0, 0])
    corners_world = _rotate_xyzw(q.expand(8, -1), corners_local) + box_pose[0, :3]
    aabb_min, aabb_max = corners_world.amin(dim=0)[None, None], corners_world.amax(dim=0)[None, None]
    legacy_depth, _ = rasterize_axis_aligned_boxes_depth_from_pose(
        aabb_min.expand(2, -1, -1), aabb_max.expand(2, -1, -1), cameras, spec
    )
    exact_mask, legacy_mask = torch.isfinite(oriented_depth), torch.isfinite(legacy_depth)
    aabb_only = legacy_mask & ~exact_mask
    # CPU-only controlled timing (the shared implementation has the same B*K*H*W
    # memory class as the old pass; production GPU timing still needs an Isaac run).
    def _mean_ms(fn, count=8):
        fn()
        start = time.perf_counter()
        for _ in range(count):
            fn()
        return (time.perf_counter() - start) * 1000.0 / count
    oriented_ms = _mean_ms(lambda: rasterize_oriented_boxes_depth_from_pose(mins, maxs, poses, cameras, spec))
    legacy_ms = _mean_ms(lambda: rasterize_axis_aligned_boxes_depth_from_pose(
        aabb_min.expand(2, -1, -1), aabb_max.expand(2, -1, -1), cameras, spec
    ))

    # A deliberately sparse front-face cloud demonstrates the production failure
    # mode: background samples win pixels between wall points.  The exact solid OBB
    # is then composed as the nearest depth and must close every such leak.
    ys = torch.linspace(-2.5, 2.5, 321)
    zs = torch.linspace(-1.0, 3.0, 257)
    yy, zz = torch.meshgrid(ys, zs, indexing="ij")
    background_local = torch.stack([torch.full_like(yy, 2.20), yy, zz], dim=-1).reshape(-1, 3)
    sparse_y, sparse_z = torch.meshgrid(torch.linspace(-0.4, 0.4, 3), torch.linspace(0.45, 1.55, 3), indexing="ij")
    sparse_wall_local = torch.stack([torch.full_like(sparse_y, 1.0), sparse_y, sparse_z], dim=-1).reshape(-1, 3)
    sparse_and_background = torch.cat([sparse_wall_local, background_local], dim=0)
    sparse_and_background_world = _rotate_xyzw(q.expand(len(sparse_and_background), -1), sparse_and_background) + box_pose[0, :3]
    sparse_depth, _ = rasterize_depth_zbuffer_from_pose(
        sparse_and_background_world.unsqueeze(0).expand(2, -1, -1), cameras, spec, clip_mode="post"
    )
    sparse_leak = exact_mask & (~torch.isfinite(sparse_depth) | (sparse_depth > oriented_depth + 0.01))
    corrected_depth = torch.minimum(sparse_depth, oriented_depth)
    corrected_leak = sparse_leak & (~torch.isfinite(corrected_depth) | (corrected_depth > oriented_depth + 1e-5))

    # Valid geometry in front of the wall must remain visible after composition.
    front_y, front_z = torch.meshgrid(torch.linspace(-0.10, 0.10, 25), torch.linspace(0.80, 1.20, 25), indexing="ij")
    front_local = torch.stack([torch.full_like(front_y, 0.45), front_y, front_z], dim=-1).reshape(-1, 3)
    front_world = _rotate_xyzw(q.expand(len(front_local), -1), front_local) + box_pose[0, :3]
    front_depth, _ = rasterize_depth_zbuffer_from_pose(front_world.unsqueeze(0).expand(2, -1, -1), cameras, spec, clip_mode="post")
    with_front = torch.minimum(front_depth, oriented_depth)
    front_overlap = torch.isfinite(front_depth) & exact_mask & (front_depth < oriented_depth - 0.01)
    front_hidden = front_overlap & (with_front > front_depth + 1e-5)

    # Visual evidence without a plotting dependency: both reconstructed layers lie on
    # the same fixed rotated box; the red rectangle is the legacy enclosing AABB.
    canvas = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(canvas)
    all_xy = torch.cat([corners_world[:, :2], camera_world[:, :2], *(item[:, :2] for item in world_layers)]).numpy()
    lo_xy, hi_xy = all_xy.min(axis=0) - 0.15, all_xy.max(axis=0) + 0.15
    scale = min(760.0 / (hi_xy[0] - lo_xy[0]), 560.0 / (hi_xy[1] - lo_xy[1]))
    def pixel(xy):
        return (70 + (float(xy[0]) - lo_xy[0]) * scale, 630 - (float(xy[1]) - lo_xy[1]) * scale)
    for color, points in zip(((45, 105, 210), (240, 135, 25)), world_layers):
        for point in points[:: max(1, len(points) // 1800)]:
            x0, y0 = pixel(point[:2])
            draw.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill=color)
    box_xy = corners_world[:, :2].numpy()
    hull = [0, 1, 3, 2, 0]
    draw.line([pixel(box_xy[idx]) for idx in hull], fill="black", width=3)
    lo, hi = aabb_min[0, 0].numpy(), aabb_max[0, 0].numpy()
    aabb = [pixel((lo[0], lo[1])), pixel((hi[0], lo[1])), pixel((hi[0], hi[1])), pixel((lo[0], hi[1])), pixel((lo[0], lo[1]))]
    draw.line(aabb, fill="crimson", width=2)
    for color, point in zip(((45, 105, 210), (240, 135, 25)), camera_world):
        x0, y0 = pixel(point[:2])
        draw.line((x0 - 7, y0 - 7, x0 + 7, y0 + 7), fill=color, width=3)
        draw.line((x0 - 7, y0 + 7, x0 + 7, y0 - 7), fill=color, width=3)
    draw.text((20, 20), "Blue/orange: exact OBB returns from two backing-away camera poses", fill="black")
    draw.text((20, 45), "Black: fixed rotated wall; red: legacy enclosing world AABB", fill="black")
    canvas.save(output_dir / "controlled_rotated_wall_topdown.png")

    h, w = exact_mask[0].shape
    mask_image = Image.new("RGB", (w * 3, h), "white")
    exact_rgb = torch.where(exact_mask[0][..., None], torch.tensor([30, 30, 30]), torch.tensor([255, 255, 255])).byte().numpy()
    false_rgb = torch.where(aabb_only[0][..., None], torch.tensor([220, 30, 50]), torch.tensor([255, 255, 255])).byte().numpy()
    leak_rgb = torch.where(sparse_leak[0][..., None], torch.tensor([240, 160, 0]), torch.tensor([255, 255, 255])).byte().numpy()
    mask_image.paste(Image.fromarray(exact_rgb), (0, 0))
    mask_image.paste(Image.fromarray(false_rgb), (w, 0))
    mask_image.paste(Image.fromarray(leak_rgb), (w * 2, 0))
    mask_image.save(output_dir / "controlled_occlusion_masks.png")

    return {
        "camera_displacement_m": float((camera_world[1] - camera_world[0]).norm()),
        "exact_obb_pixels": [int(mask.sum()) for mask in exact_mask],
        "backprojected_face_residual_max_m": [float(item.max()) for item in residuals],
        "backprojected_face_residual_p95_m": [float(torch.quantile(item, 0.95)) for item in residuals],
        "legacy_aabb_pixels": [int(mask.sum()) for mask in legacy_mask],
        "legacy_aabb_false_opacity_pixels": [int(mask.sum()) for mask in aabb_only],
        "controlled_cpu_ms_per_render": {"exact_obb": oriented_ms, "legacy_aabb": legacy_ms},
        "sparse_zbuffer_penetration_pixels_before_fallback": [int(mask.sum()) for mask in sparse_leak],
        "sparse_zbuffer_penetration_pixels_after_fallback": [int(mask.sum()) for mask in corrected_leak],
        "front_geometry_pixels_behind_no_wall": [int(mask.sum()) for mask in front_overlap],
        "front_geometry_pixels_hidden_by_fallback": [int(mask.sum()) for mask in front_hidden],
        "penetration_check": "The sparse point-only pass leaks background through the wall. Nearest-depth composition with the exact OBB closes every leak while retaining every closer foreground pixel.",
    }


def _legacy_camera_dependent_flush_check() -> dict:
    """Exercise the removed helper to quantify its camera-dependent geometry change."""
    points = torch.zeros((2, 4, 3))
    axis_order = torch.tensor([[0, 1, 2], [0, 1, 2]])
    panel_min = torch.tensor([[-0.02, -0.5, 0.0], [-0.02, -0.5, 0.0]])
    panel_max = torch.tensor([[0.02, 0.5, 2.0], [0.02, 0.5, 2.0]])
    camera_pos = torch.tensor([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    moved = move_flush_wall_behind_panel(
        points, axis_order, panel_min, panel_max, camera_pos,
        SimpleNamespace(flush_point_fraction=0.5),
    )
    return {
        "flush_shift_with_camera_on_negative_panel_side_m": float(moved[0, -1, 0]),
        "flush_shift_with_camera_on_positive_panel_side_m": float(moved[1, -1, 0]),
        "same_authored_flush_point_world_separation_m": float((moved[0, -1] - moved[1, -1]).norm()),
        "interpretation": "This helper changed authored wall geometry based on camera side; it is removed from the shared renderer. It flips only on a panel-side crossing, so it does not explain a continuous backing-away drift by itself.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "replay": _replay_metrics(args.recording, args.output_dir),
        "controlled_rotated_wall": _controlled_renderer_check(args.output_dir),
        "legacy_camera_dependent_flush_helper": _legacy_camera_dependent_flush_check(),
    }
    path = args.output_dir / "wall_frame_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
