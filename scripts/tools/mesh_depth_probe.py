#!/usr/bin/env python3
"""Standalone mesh-raycast robot depth probe.

This bypasses Isaac Lab entirely.  It raycasts the existing robot visual mesh with
Open3D's CPU ray tracer, then compares that depth with the current sampled-point
z-buffer and composites the mesh depth with the existing procedural scene depth.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPO_ROOT / "scripts" / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from DoorOpening.utils.camera_utils import (  # noqa: E402
    _camera_basis_from_pose_x_forward,
    backproject_depth_to_world_from_pose,
    rasterize_depth_zbuffer_from_pose,
)
from benchmark_depth_render import build_gt_cloud  # noqa: E402
from render_depth_roundtrip_viser import (  # noqa: E402
    DEFAULT_STUDENT_CFG,
    build_camera_spec,
    DEFAULT_DOOR_URDF,
    load_robot_asset,
    robot_camera_pose_world,
    _load_urdf,
    _mount_offset_matrix,
    _quat_wxyz_to_matrix,
    GLORBOT_DIR,
    GLORBOT_URDF,
    FRANKA_READY_JOINT_POS,
    ROBOT_RIGHT_M,
    yaw_quat_wxyz,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--student-cfg", type=Path, default=DEFAULT_STUDENT_CFG)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "mesh_depth_probe")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--robot-points", type=int, default=30000)
    p.add_argument("--near", type=float, default=0.3)
    p.add_argument("--far", type=float, default=3.0)
    p.add_argument("--standoff", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_ready_robot_mesh():
    robot = _load_urdf(GLORBOT_URDF, {"glorbot": GLORBOT_DIR})
    names = list(robot.actuated_joint_names)
    cfg = np.zeros(len(names), dtype=np.float64)
    for i, value in enumerate(FRANKA_READY_JOINT_POS):
        name = f"panda_joint{i + 1}"
        if name in names:
            cfg[names.index(name)] = value
    robot.update_cfg(cfg)
    # The optical center lives inside the camera housing.  A real depth sensor does not
    # observe its own housing, so omit that one visual from the self-occlusion pass while
    # retaining the arm, gripper, chassis, and the other camera-support links.
    scene = robot.scene.copy()
    if "camera.STL" in scene.geometry:
        scene.delete_geometry("camera.STL")
    mesh = scene.dump(concatenate=True)
    return mesh


def robot_pose(mesh, device):
    base_pos = np.array([ROBOT_RIGHT_M, -1.0, 0.0], dtype=np.float64)
    base_R = _quat_wxyz_to_matrix(yaw_quat_wxyz(np.pi / 2.0))
    base_pos_t = torch.tensor(base_pos, dtype=torch.float32, device=device)
    base_R_t = torch.tensor(base_R, dtype=torch.float32, device=device)
    verts = np.asarray(mesh.vertices, dtype=np.float32) @ base_R.T + base_pos.astype(np.float32)

    cam_T_base = np.asarray(
        _load_urdf(GLORBOT_URDF, {"glorbot": GLORBOT_DIR}).get_transform("x5_camera_link", "base_link"),
        dtype=np.float64,
    )
    cam_np = robot_camera_pose_world(cam_T_base, base_pos, base_R)
    cam_pose = torch.from_numpy(cam_np).to(device).unsqueeze(0)
    return verts, np.asarray(mesh.faces, dtype=np.int32), cam_pose, base_pos_t, base_R_t


def mesh_depth_open3d(vertices_world, faces, camera_pose, cam_spec):
    import open3d as o3d

    device = camera_pose.device
    dtype = camera_pose.dtype
    H, W = int(cam_spec["H"]), int(cam_spec["W"])
    fx = float(cam_spec["intrinsics"][0, 0])
    fy = float(cam_spec["intrinsics"][1, 1])
    cx = float(cam_spec["intrinsics"][0, 2])
    cy = float(cam_spec["intrinsics"][1, 2])
    u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)
    forward = v_hat[0].detach().cpu().numpy()
    right = u_hat[0].detach().cpu().numpy()
    down = w_hat[0].detach().cpu().numpy()
    origin = camera_pose[0, :3].detach().cpu().numpy()

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dirs = forward[None, None, :] + ((xx - cx) / fx)[..., None] * right
    dirs += ((yy - cy) / fy)[..., None] * down
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-12)
    # The optical center is physically inside the small camera housing mesh.  Start each ray
    # 1 cm forward so Open3D does not return the degenerate t=0 self-intersection; report depth
    # relative to the original optical center below.
    ray_origins = origin[None, None, :] + 0.01 * dirs
    rays = np.concatenate([ray_origins, dirs], axis=-1).astype(np.float32)

    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(vertices_world, dtype=o3d.core.Dtype.Float32)
    tmesh.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    t0 = time.perf_counter()
    hits = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    valid = np.isfinite(hits)
    hit_world = ray_origins + hits[..., None] * dirs
    depth = np.full((H, W), np.inf, dtype=np.float32)
    depth[valid] = np.sum((hit_world[valid] - origin) * forward[None, :], axis=-1)
    depth[(depth < float(cam_spec["near_m"])) | (depth > float(cam_spec["far_m"]))] = np.inf
    return torch.from_numpy(depth).to(device=device, dtype=dtype), elapsed_ms


def world_points(depth, camera_pose, intr):
    pts, valid = backproject_depth_to_world_from_pose(depth[None], camera_pose, intr)
    return pts[0][valid[0]].detach().cpu().numpy()


def write_ply(path, points, colors):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)

    mesh = load_ready_robot_mesh()
    vertices, faces, camera_pose, base_pos, base_R = robot_pose(mesh, device)
    cam = build_camera_spec(args.width, args.height, args.near, args.far, device)
    sampled_base, _ = load_robot_asset(args.robot_points, device)
    sampled_world = sampled_base @ base_R.T + base_pos
    sampled_world = sampled_world.unsqueeze(0)

    sampled_depth, intr = rasterize_depth_zbuffer_from_pose(
        sampled_world, camera_pose, cam, inflate_px=2, clip_mode="post"
    )
    mesh_depth, ray_ms = mesh_depth_open3d(vertices, faces, camera_pose, cam)
    scene, n_door, n_walls = build_gt_cloud(
        argparse.Namespace(student_cfg=args.student_cfg, door=DEFAULT_DOOR_URDF, door_yaw_deg=-90.0,
                           board_num_points=None, no_walls=False, gt_scale=1.0), device
    )
    scene_depth, _ = rasterize_depth_zbuffer_from_pose(scene, camera_pose, cam, inflate_px=2, clip_mode="post")
    composite_depth = torch.minimum(scene_depth, mesh_depth[None])

    sampled_visible = world_points(sampled_depth[0], camera_pose, intr)
    mesh_visible = world_points(mesh_depth, camera_pose, intr)
    composite_visible = world_points(composite_depth[0], camera_pose, intr)
    composite_valid = torch.isfinite(composite_depth[0])
    robot_wins = composite_valid & torch.isfinite(mesh_depth) & (mesh_depth <= scene_depth[0])
    composite_robot_visible = world_points(torch.where(robot_wins, composite_depth[0], torch.full_like(composite_depth[0], float("inf"))), camera_pose, intr)
    composite_scene_visible = world_points(torch.where(composite_valid & ~robot_wins, composite_depth[0], torch.full_like(composite_depth[0], float("inf"))), camera_pose, intr)
    point_diff = torch.isfinite(sampled_depth[0]) & torch.isfinite(mesh_depth) & (sampled_depth[0] - mesh_depth).abs().gt(0.01)
    mesh_only = torch.isfinite(mesh_depth) & ~torch.isfinite(sampled_depth[0])
    point_coverage = torch.isfinite(sampled_depth[0]).float().mean().item()
    mesh_coverage = torch.isfinite(mesh_depth).float().mean().item()
    extra_px = int((mesh_only).sum().item())

    colors = np.tile(np.array([[0.95, 0.25, 0.15]], dtype=np.float32), (composite_visible.shape[0], 1))
    write_ply(args.output_dir / "composited_mesh_depth_pointcloud.ply", composite_visible, colors)
    colored_points = np.concatenate([composite_scene_visible, composite_robot_visible], axis=0)
    colored_colors = np.concatenate([
        np.tile(np.array([[0.62, 0.62, 0.62]], dtype=np.float32), (len(composite_scene_visible), 1)),
        np.tile(np.array([[0.05, 0.85, 1.0]], dtype=np.float32), (len(composite_robot_visible), 1)),
    ], axis=0)
    write_ply(args.output_dir / "composited_mesh_depth_colored.ply", colored_points, colored_colors)
    write_ply(args.output_dir / "mesh_robot_visible_pointcloud.ply", mesh_visible,
              np.tile(np.array([[0.10, 0.80, 1.0]], dtype=np.float32), (mesh_visible.shape[0], 1)))

    def show_depth(ax, depth, title, cmap="viridis"):
        arr = depth.detach().cpu().numpy().copy()
        arr[~np.isfinite(arr)] = np.nan
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    ax = fig.subplots(2, 3)
    show_depth(ax[0, 0], sampled_depth[0], "sampled-point robot depth")
    show_depth(ax[0, 1], mesh_depth, "triangle mesh raycast depth")
    diff = point_diff.detach().cpu().numpy()
    ax[0, 2].imshow(diff, cmap="magma")
    ax[0, 2].set_title(f">1 cm disagreement ({int(diff.sum())} px)")
    ax[0, 2].axis("off")
    ax[1, 0].scatter(sampled_visible[:, 0], sampled_visible[:, 2], s=0.2, c="#f97316")
    ax[1, 0].set_title(f"sampled visible cloud ({len(sampled_visible):,})")
    ax[1, 0].set_aspect("equal")
    ax[1, 1].scatter(mesh_visible[:, 0], mesh_visible[:, 2], s=0.2, c="#22d3ee")
    ax[1, 1].set_title(f"mesh visible cloud ({len(mesh_visible):,})")
    ax[1, 1].set_aspect("equal")
    ax[1, 2].scatter(composite_visible[:, 1], composite_visible[:, 2], s=0.2, c="#ef4444")
    if len(composite_scene_visible):
        ax[1, 2].scatter(composite_scene_visible[:, 1], composite_scene_visible[:, 2], s=0.2, c="#999999")
    if len(composite_robot_visible):
        ax[1, 2].scatter(composite_robot_visible[:, 1], composite_robot_visible[:, 2], s=0.5, c="#00c8ff")
    ax[1, 2].set_title(f"Open3D PLY colors: scene {len(composite_scene_visible):,} / robot {len(composite_robot_visible):,}")
    ax[1, 2].set_aspect("equal")
    fig.suptitle("Standalone robot mesh depth vs sampled-point depth", fontsize=15)
    fig.savefig(args.output_dir / "mesh_depth_comparison.png", dpi=160)
    plt.close(fig)

    report = args.output_dir / "report.txt"
    report.write_text(
        f"mesh_vertices={len(vertices)}\nmesh_triangles={len(faces)}\n"
        f"scene_points={scene.shape[1]} door_points={n_door} wall_points={n_walls}\n"
        f"camera={args.width}x{args.height}\nraycast_ms={ray_ms:.3f}\n"
        f"sampled_coverage={point_coverage:.6f}\nmesh_coverage={mesh_coverage:.6f}\n"
        f"mesh_only_pixels={extra_px}\nmesh_visible_points={len(mesh_visible)}\n"
        f"composited_points={len(composite_visible)}\n"
    )
    print(report.read_text(), end="")
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
