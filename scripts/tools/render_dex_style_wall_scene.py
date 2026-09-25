#!/usr/bin/env python3
"""DEX-style point-cloud sensor render with an intentionally close robot and large walls.

This is a local port of the IsaacGymEnvs/dex pipeline: surface points are sampled first,
then projected into a synthetic camera z-buffer and back-projected to a point cloud.  No
wall geometry is added to Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
TOOLS = REPO_ROOT / "scripts" / "tools"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(TOOLS))

from DoorOpening.utils.camera_utils import (  # noqa: E402
    _camera_basis_from_pose_x_forward,
    backproject_depth_to_world_from_pose,
    rasterize_depth_zbuffer_from_pose,
)
from render_depth_roundtrip_viser import (  # noqa: E402
    DEFAULT_DOOR_URDF,
    DEFAULT_STUDENT_CFG,
    FRANKA_READY_JOINT_POS,
    GLORBOT_DIR,
    GLORBOT_URDF,
    _load_urdf,
    _quat_wxyz_to_matrix,
    build_camera_spec,
    load_door_asset,
    load_robot_asset,
    robot_camera_pose_world,
    yaw_quat_wxyz,
)
from DoorOpening.utils.visual_scene_sampler import CachedVisualSceneSampler  # noqa: E402
from mock_depth_compositing import composite_robot_scene_depth  # noqa: E402
from DoorOpening.utils.wall_distractors import (  # noqa: E402
    WallDistractorParams,
    compute_wall_bbox_ordering,
    sample_wall_points_local,
)


def sample_box_surface(center, dims, num_points, device, seed):
    """Uniformly sample a cuboid surface, matching the DEX obstacle sampler's intent."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    center = torch.as_tensor(center, device=device, dtype=torch.float32)
    dims = torch.as_tensor(dims, device=device, dtype=torch.float32)
    half = dims / 2.0
    areas = torch.stack([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]])
    face = torch.multinomial(areas / areas.sum(), int(num_points), replacement=True, generator=g)
    uv = torch.rand((int(num_points), 2), device=device, generator=g)
    p = torch.empty((int(num_points), 3), device=device, dtype=torch.float32)
    p[:, 0] = (uv[:, 0] * 2.0 - 1.0) * half[0]
    p[:, 1] = (uv[:, 1] * 2.0 - 1.0) * half[1]
    p[:, 2] = (uv[:, 0] * 2.0 - 1.0) * half[2]
    p[face == 0, 0] = half[0]
    p[face == 1, 0] = -half[0]
    p[face == 2, 1] = half[1]
    p[face == 3, 1] = -half[1]
    p[face == 4, 2] = half[2]
    p[face == 5, 2] = -half[2]
    # Face ids above are six-way; choose the second axis for the area category and use a
    # second random coordinate for the remaining coordinate on each face.
    face = torch.randint(0, 6, (int(num_points),), device=device, generator=g)
    p[:, 0] = (torch.rand((int(num_points),), device=device, generator=g) * 2 - 1) * half[0]
    p[:, 1] = (torch.rand((int(num_points),), device=device, generator=g) * 2 - 1) * half[1]
    p[:, 2] = (torch.rand((int(num_points),), device=device, generator=g) * 2 - 1) * half[2]
    p[face == 0, 0] = half[0]; p[face == 1, 0] = -half[0]
    p[face == 2, 1] = half[1]; p[face == 3, 1] = -half[1]
    p[face == 4, 2] = half[2]; p[face == 5, 2] = -half[2]
    return p + center


def sample_box_volume(center, dims, num_points, device, seed):
    """Sample the occupied volume of a box, preserving the same point budget."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    center = torch.as_tensor(center, device=device, dtype=torch.float32)
    dims = torch.as_tensor(dims, device=device, dtype=torch.float32)
    half = dims / 2.0
    return center + (torch.rand((int(num_points), 3), device=device, generator=g) * 2.0 - 1.0) * half


def sample_camera_aligned_box(center, dims_forward_right_down, num_points, camera_pose, device, seed):
    """Sample a filled box whose axes are camera forward/right/down."""
    local = sample_box_volume((0.0, 0.0, 0.0), dims_forward_right_down, num_points, device, seed)
    forward, right, down = _camera_basis_from_pose_x_forward(camera_pose)
    basis = torch.cat((forward, right, down), dim=0).to(device=device, dtype=local.dtype)
    center = torch.as_tensor(center, device=device, dtype=local.dtype)
    return local @ basis + center


def sample_camera_facing_visuals(sampler, link_names, camera_pos_base, total_points, seed=101):
    """Sample only visual faces whose outward normal faces the active camera.

    A closed box has two parallel faces. Sparse point z-buffering can leak the far face
    through pixel holes in the near face, so this diagnostic samples the camera-facing
    visual surface directly. It is a test path for the solid visual asset; the production
    sampler should eventually implement the same normal/depth rule on its cached faces.
    """
    import trimesh

    rng = np.random.default_rng(int(seed))
    q = torch.zeros((1, len(sampler.robot.actuated_joint_names)), device=sampler.device, dtype=torch.float32)
    fk = sampler.robot.link_fk_batch(q, use_names=True)
    camera_pos_base = np.asarray(camera_pos_base, dtype=np.float64)
    geometries = []
    weights = []
    for link in sampler.links:
        if link.name not in link_names:
            continue
        T = fk[link.name][0].detach().cpu().numpy()
        R = T[:3, :3]
        t = T[:3, 3]
        for visual in link.visuals:
            mesh = sampler._load_visual_mesh_in_link_frame(visual)
            if mesh is None or len(mesh.faces) == 0:
                continue
            geometries.append((mesh, R, t))
            weights.append(float(mesh.bounding_box_oriented.area))
    counts = sampler._allocate_point_counts(weights, max(int(total_points) * 4, int(total_points)))
    sampled = []
    for (mesh, R, t), count in zip(geometries, counts):
        if count <= 0:
            continue
        pts, face_ids = trimesh.sample.sample_surface(mesh, int(count), seed=rng)
        normals = mesh.face_normals[face_ids]
        pts_base = pts @ R.T + t
        normals_base = normals @ R.T
        toward_camera = np.einsum("ij,ij->i", normals_base, camera_pos_base[None, :] - pts_base) > 0.0
        if np.any(toward_camera):
            sampled.append(pts_base[toward_camera])
    # The first part of candidates holds geometry tuples; collect sampled arrays separately.
    if not sampled:
        return torch.zeros((0, 3), device=sampler.device, dtype=torch.float32)
    points = np.concatenate(sampled, axis=0)
    if len(points) > int(total_points):
        keep = np.linspace(0, len(points) - 1, int(total_points)).round().astype(np.int64)
        points = points[keep]
    return torch.as_tensor(points, device=sampler.device, dtype=torch.float32)


def sample_even_visuals(sampler, link_names, total_points, seed=201):
    """Alternative comparison path using trimesh's even surface sampler."""
    import trimesh

    geometries = []
    weights = []
    q = torch.zeros((1, len(sampler.robot.actuated_joint_names)), device=sampler.device, dtype=torch.float32)
    fk = sampler.robot.link_fk_batch(q, use_names=True)
    for link in sampler.links:
        if link.name not in link_names:
            continue
        T = fk[link.name][0].detach().cpu().numpy()
        for visual in link.visuals:
            mesh = sampler._load_visual_mesh_in_link_frame(visual)
            if mesh is not None and len(mesh.faces):
                geometries.append((mesh, T))
                weights.append(float(mesh.bounding_box_oriented.area))
    counts = sampler._allocate_point_counts(weights, int(total_points))
    sampled = []
    for (mesh, T), count in zip(geometries, counts):
        if count <= 0:
            continue
        pts = trimesh.sample.sample_surface_even(mesh, int(count), radius=None)[0]
        if len(pts):
            h = np.concatenate((pts, np.ones((len(pts), 1))), axis=1)
            sampled.append((h @ T.T)[:, :3])
    if not sampled:
        return torch.zeros((0, 3), device=sampler.device, dtype=torch.float32)
    return torch.as_tensor(np.concatenate(sampled, axis=0), device=sampler.device, dtype=torch.float32)


def load_initial_state(door_urdf):
    meta_path = door_urdf.parent / "variant_meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return dict(json.load(f).get("initial_state") or {})


def _pose_from_initial_state(initial_state, key):
    pose = initial_state.get(key) or {}
    pos = np.asarray(pose.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    # Scratch-door metadata stores quaternions in wxyz order, matching IsaacLab.
    quat_wxyz = pose.get("rot", [1.0, 0.0, 0.0, 0.0])
    return pos, _quat_wxyz_to_matrix(quat_wxyz)


def load_ready_robot(device, base_y, initial_state=None, base_approach_m=0.0, arm_down_rad=0.0, base_yaw_rad=0.0, base_y_offset_m=0.0):
    initial_state = {} if initial_state is None else initial_state
    robot = _load_urdf(GLORBOT_URDF, {"glorbot": GLORBOT_DIR})
    names = list(robot.actuated_joint_names)
    cfg = np.zeros(len(names), dtype=np.float64)
    franka_q = list(initial_state.get("robot_franka_joint_pos", FRANKA_READY_JOINT_POS))
    # Panda joint 2 is the shoulder elevation in this URDF.  A small negative offset
    # lowers the forearm while preserving the rest of the recorded initial configuration.
    if len(franka_q) >= 2:
        franka_q[1] -= float(arm_down_rad)
    for i, value in enumerate(franka_q):
        name = f"panda_joint{i + 1}"
        if name in names:
            cfg[names.index(name)] = value
    robot.update_cfg(cfg)
    # `robot_world_pose` is the fixed articulation root.  When start-base sampling
    # is enabled, the actual initial robot placement is `sampled_robot_world_pose`;
    # using the root pose alone places the rendered robot away from the door.
    robot_pose_key = "sampled_robot_world_pose" if initial_state.get("sampled_robot_world_pose") else "robot_world_pose"
    if robot_pose_key in initial_state:
        base_pos, base_R = _pose_from_initial_state(initial_state, robot_pose_key)
    else:
        base_pos = np.array([0.0, float(base_y), 0.0], dtype=np.float64)
        base_R = _quat_wxyz_to_matrix(yaw_quat_wxyz(math.pi / 2.0))
    # In the environment, positive base_x_joint moves the chassis toward the door
    # (world X decreases from the default robot position at x=1).  Apply the requested
    # visualization offset in world coordinates without modifying the asset metadata.
    base_pos = np.asarray(base_pos, dtype=np.float64).copy()
    base_pos[0] -= float(base_approach_m)
    base_pos[1] += float(base_y_offset_m)
    if abs(float(base_yaw_rad)) > 1e-9:
        c, s = math.cos(float(base_yaw_rad)), math.sin(float(base_yaw_rad))
        yaw_R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        base_R = yaw_R @ base_R
    base_pos_t = torch.tensor(base_pos, dtype=torch.float32, device=device)
    base_R_t = torch.tensor(base_R, dtype=torch.float32, device=device)
    robot_points, cam_T_base = load_robot_asset(50000, device, franka_q=franka_q)
    robot_world = (robot_points @ base_R_t.T + base_pos_t).unsqueeze(0)
    camera_np = robot_camera_pose_world(cam_T_base, base_pos, base_R)
    camera_pose = torch.from_numpy(camera_np).to(device).unsqueeze(0)
    return robot_world, camera_pose, base_pos_t, base_R_t


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--student-cfg", type=Path, default=DEFAULT_STUDENT_CFG)
    p.add_argument("--door", type=Path, default=DEFAULT_DOOR_URDF)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dex_style_wall_scene")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--robot-base-y", type=float, default=-0.62)
    p.add_argument("--robot-approach-m", type=float, default=0.25,
                   help="Move the rendered robot base toward the door along world -X.")
    p.add_argument("--arm-down-rad", type=float, default=0.15,
                   help="Additional negative Panda joint-2 offset for a slightly lower arm.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wall-mode", choices=("surface", "solid"), default="surface")
    p.add_argument("--wall-points", type=int, default=30000,
                   help="Wall distractor sampling budget per frame.")
    p.add_argument("--wall-layout", choices=("camera", "training"), default="camera",
                   help="Place solid distractors behind the door in camera coordinates, or match the training sampler.")
    p.add_argument("--legacy-layout", action="store_true",
                   help="Match the earlier 8080 preview layout: legacy robot pose, preview door yaw, fixed side walls.")
    p.add_argument(
        "--ignore-initial-state",
        action="store_true",
        help="Use the legacy hand-placed robot/door pose instead of variant_meta.json initial_state.",
    )
    p.add_argument(
        "--use-urdf-visuals",
        action="store_true",
        help="Render the door's actual URDF visual primitives (including box visuals) instead of the legacy solid-panel proxy.",
    )
    p.add_argument(
        "--single-stream",
        action="store_true",
        help="Save only the final camera-visible policy cloud for unambiguous Viser inspection.",
    )
    p.add_argument(
        "--camera-facing-visuals",
        action="store_true",
        help="Sample only visual faces facing the active camera to test single-layer solid geometry.",
    )
    p.add_argument(
        "--surface-even-visuals",
        action="store_true",
        help="Use trimesh.sample_surface_even for the comparison experiment.",
    )
    p.add_argument("--num-frames", type=int, default=1,
                   help="Repeat the accepted rendered frame this many times in the Viser replay.")
    p.add_argument("--show-ground-truth-sources", action="store_true",
                   help="Include raw ground-truth door, wall, and robot partitions in the replay.")
    return p.parse_args()


def write_ply(path, points, colors):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    torch.manual_seed(7)
    initial_state = {} if args.ignore_initial_state else load_initial_state(args.door)

    # Load the actual URDF visual sources.  The default keeps the historical proxy A/B
    # behavior; --use-urdf-visuals is used for matched mesh-vs-box asset comparisons.
    full_bbox, panel_bbox, _panel_bbox_link1, _link1_pose_base, _panel_pts, _frame_pts, _handle_center = load_door_asset(
        args.door, 30000, device
    )
    from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler
    door_sampler = FrankaGripperSampler(str(args.door), device=device, num_points=16000)
    zero = torch.zeros((1, len(door_sampler.robot.actuated_joint_names)), dtype=torch.float32, device=device)
    # Offline asset pose: link_0/link_1/link_2 are already expressed in the door base frame
    # at zero joint state.  Use the original sampler for this one-time A/B reference.
    baseline_visual = door_sampler.sample_link_set(zero, ["link_0", "link_1", "link_2"])[0]
    handle = door_sampler.sample_link_set(zero, ["link_2"])[0]
    frame = door_sampler.sample_link_set(zero, ["link_0"])[0]
    # The generated door asset is already authored in the environment door frame:
    # its thin panel axis is local X, its width is local Y, and height is local Z.
    # Do not apply the old preview-only -90deg yaw here; that rotated the panel edge-on
    # to the robot camera and made the wall distractors appear to wrap around the scene.
    walls_are_world = False
    if args.legacy_layout:
        yaw = -math.pi / 2.0
        c, s = math.cos(yaw), math.sin(yaw)
        door_R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device)
    else:
        door_R = torch.eye(3, device=device, dtype=torch.float32)
    panel_min, panel_max = full_bbox[0, 0], full_bbox[0, 1]
    panel_center_local = 0.5 * (panel_min + panel_max)
    panel_dims = (panel_max - panel_min).clamp_min(0.02) + 0.006
    solid_panel = sample_box_surface(panel_center_local, panel_dims, 42000, device, 21) @ door_R.T
    cached_sampler = CachedVisualSceneSampler(
        door_sampler,
        link_names=("link_0", "link_1", "link_2"),
        replacement_link_points={"link_1": solid_panel},
    )
    identity_pos = {name: torch.zeros((1, 3), device=device) for name in cached_sampler.link_names}
    identity_quat = {name: torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device) for name in cached_sampler.link_names}
    cached_timing = cached_sampler.benchmark(identity_pos, identity_quat, repeats=20)
    baseline_visual = baseline_visual @ door_R.T
    if args.use_urdf_visuals:
        door = baseline_visual
        solid_panel = door_sampler.sample_link_set(zero, ["link_1"])[0] @ door_R.T
    else:
        door = torch.cat([solid_panel, frame @ door_R.T, handle @ door_R.T], dim=0)
    n_door = int(door.shape[0])
    mn, mx = solid_panel.min(0).values, solid_panel.max(0).values
    panel_center = (mn + mx) * 0.5
    panel_width = float(mx[0] - mn[0])
    panel_depth = max(float(mx[1] - mn[1]), 0.12)
    panel_top = float(mx[2])

    door_world_pos_np, door_world_R_np = _pose_from_initial_state(initial_state, "door_world_pose")
    door_world_pos = torch.as_tensor(door_world_pos_np, dtype=torch.float32, device=device)
    door_world_R = torch.as_tensor(door_world_R_np, dtype=torch.float32, device=device)
    # Build the distractors after loading the actual robot camera.  Their local "behind"
    # direction must be selected from the camera-to-door direction because the metadata door
    # pose may include a 180-degree yaw (the default asset does).
    robot, camera_pose, base_pos, base_R = load_ready_robot(
        device, args.robot_base_y, initial_state,
        base_approach_m=args.robot_approach_m,
        arm_down_rad=args.arm_down_rad,
    )
    if args.surface_even_visuals:
        door = sample_even_visuals(door_sampler, ("link_0", "link_1", "link_2"), 16000, seed=201)
        solid_panel = sample_even_visuals(door_sampler, ("link_1",), 12000, seed=202)
    elif args.camera_facing_visuals:
        camera_pos_base = (camera_pose[0, :3] - door_world_pos) @ door_world_R
        door = sample_camera_facing_visuals(
            door_sampler, ("link_0", "link_1", "link_2"), camera_pos_base.detach().cpu().numpy(), 16000, seed=101
        )
        solid_panel = sample_camera_facing_visuals(
            door_sampler, ("link_1",), camera_pos_base.detach().cpu().numpy(), 12000, seed=102
        )
    walls_are_world = False
    if args.legacy_layout:
        # Exact geometry used by the earlier 8080 preview.  This is intentionally a
        # visualization compatibility mode, not the environment wall sampler.
        wall_sampler = sample_box_surface if args.wall_mode == "surface" else sample_box_volume
        wall_specs = [
            ([-1.05, float(panel_center[1]), panel_top * 0.58], [0.85, 0.65, panel_top * 1.2], 28000, 11),
            ([1.05, float(panel_center[1]), panel_top * 0.58], [0.85, 0.65, panel_top * 1.2], 28000, 12),
            ([0.0, float(panel_center[1]), panel_top * 1.18], [2.95, 0.65, 0.55], 18000, 13),
        ]
        walls = torch.cat(
            [wall_sampler(center, dims, count, device, seed) for center, dims, count, seed in wall_specs], dim=0
        )
    elif args.wall_layout == "camera":
        # Visualization-safe solid distractors: the main occupied block is behind the
        # door along the active camera ray, with side returns outside the door silhouette.
        # The training sampler remains unchanged; this layout tests the geometry/occlusion
        # behavior without allowing a lateral distractor to become the foreground wall.
        forward, right, down = _camera_basis_from_pose_x_forward(camera_pose)
        door_center_local = 0.5 * (door.min(0).values + door.max(0).values)
        door_center_world = door_center_local @ door_world_R.T + door_world_pos
        rear_center = door_center_world + forward[0] * 0.80
        wall_specs = [
            (rear_center, (0.14, 2.8, 2.7), 20000, 41),
            (door_center_world + right[0] * 1.75 + forward[0] * 0.30, (0.35, 0.45, 2.5), 5000, 42),
            (door_center_world - right[0] * 1.75 + forward[0] * 0.30, (0.35, 0.45, 2.5), 5000, 43),
        ]
        walls = torch.cat([
            sample_camera_aligned_box(center, dims, count, camera_pose, device, seed)
            for center, dims, count, seed in wall_specs
        ], dim=0)
        walls_are_world = True
    else:
        # Match multi_pcd_dagger.py exactly: sample wall columns in the door-base frame from
    # the full door bbox, plus the optional flush slab from the panel bbox.  This is not a
    # synthetic rear background slab; the shared implementation places walls beside the
    # panel and allows them to protrude/recede along the panel thickness axis.
        wall_cfg = {
        "enabled": True,
        "num_points": int(args.wall_points),
        "side_margin_m": [0.35, 0.80],
        "edge_gap_m": [0.015, 0.04],
        "depth_m": [0.10, 0.80],
        "center_offset_m": [-0.30, 0.30],
        "face_jitter_m": 0.004,
        "flush_prob": 0.7,
        "flush_point_fraction": 0.5,
        "flush_extent_m": [0.6, 1.6],
        "detached_within_flush_frac": [0.0, 1.0],
        }
        wall_params = WallDistractorParams.from_cfg(wall_cfg, int(args.wall_points))
        axis_order, full_min_ord, full_max_ord = compute_wall_bbox_ordering(full_bbox)
        panel_min_ord = torch.gather(panel_bbox[:, 0], 1, axis_order)
        panel_max_ord = torch.gather(panel_bbox[:, 1], 1, axis_order)
        torch.manual_seed(17)
        walls = sample_wall_points_local(
            axis_order=axis_order,
            bbox_min_ordered=full_min_ord,
            bbox_max_ordered=full_max_ord,
            num_points=int(args.wall_points),
            params=wall_params,
            device=device,
            flush_bbox_min_ordered=panel_min_ord,
            flush_bbox_max_ordered=panel_max_ord,
        )[0]
    # The sampled door and its synthetic walls are authored in the door-base frame.
    # Apply the variant's initial world pose to both so the camera sees the same default
    # placement used by the environment.
    door = door @ door_world_R.T + door_world_pos
    baseline_visual = baseline_visual @ door_world_R.T + door_world_pos
    solid_panel = solid_panel @ door_world_R.T + door_world_pos
    if not walls_are_world:
        walls = walls @ door_world_R.T + door_world_pos
    cam = build_camera_spec(args.width, args.height, 0.25, 3.0, device)
    scene = torch.cat([door, walls], dim=0).unsqueeze(0)
    all_input = torch.cat([scene, robot], dim=1)
    _, intr = rasterize_depth_zbuffer_from_pose(all_input, camera_pose, cam, inflate_px=2, clip_mode="post")

    # Build separate depth passes so the door has priority over wall distractors inside its
    # projected silhouette.  This prevents a lateral wall column from incorrectly becoming a
    # foreground wall over the door, while the robot remains a true foreground occluder.
    scene_depth, _ = rasterize_depth_zbuffer_from_pose(scene, camera_pose, cam, inflate_px=2, clip_mode="post")
    robot_depth, _ = rasterize_depth_zbuffer_from_pose(robot, camera_pose, cam, inflate_px=2, clip_mode="post")
    panel_depth, _ = rasterize_depth_zbuffer_from_pose(
        solid_panel.unsqueeze(0), camera_pose, cam, inflate_px=2, clip_mode="post"
    )
    panel_world, _ = backproject_depth_to_world_from_pose(panel_depth, camera_pose, intr)
    # Occlusion diagnostic: within the projected door silhouette, count pixels where
    # wall depth wins over the door depth. These are the residuals that should disappear
    # if a solid wall representation closes sampling gaps behind the panel.
    door_only = door.unsqueeze(0)
    wall_only = walls.unsqueeze(0)
    door_depth, _ = rasterize_depth_zbuffer_from_pose(door_only, camera_pose, cam, inflate_px=2, clip_mode="post")
    wall_depth, _ = rasterize_depth_zbuffer_from_pose(wall_only, camera_pose, cam, inflate_px=2, clip_mode="post")
    panel_silhouette = torch.isfinite(panel_depth[0])
    wall_in_panel = panel_silhouette & torch.isfinite(wall_depth[0]) & (wall_depth[0] < panel_depth[0])
    protected_wall_depth = torch.where(wall_in_panel, torch.full_like(wall_depth, float("inf")), wall_depth)
    protected_scene_depth = torch.minimum(door_depth, protected_wall_depth)
    depth, _ = composite_robot_scene_depth(protected_scene_depth, robot_depth)
    rendered, valid = backproject_depth_to_world_from_pose(depth, camera_pose, intr)
    points = rendered[0][valid[0]].detach().cpu().numpy()
    scene_depth = protected_scene_depth
    robot_mask = torch.isfinite(robot_depth[0]) & (robot_depth[0] <= scene_depth[0])
    scene_mask = torch.isfinite(depth[0]) & ~robot_mask
    scene_pts = rendered[0][scene_mask].detach().cpu().numpy()
    robot_pts = rendered[0][robot_mask].detach().cpu().numpy()
    # Since the input is concatenated, color the final cloud by nearest source using separate passes.
    panel_visible_mask = (
        torch.isfinite(panel_depth[0])
        & (panel_depth[0] <= protected_wall_depth[0])
        & (panel_depth[0] <= robot_depth[0])
    )
    panel_visible_pts = panel_world[0][panel_visible_mask].detach().cpu()
    wall_visible_mask = (
        torch.isfinite(wall_depth[0])
        & (wall_depth[0] <= robot_depth[0])
        & (~panel_silhouette | (wall_depth[0] <= panel_depth[0]))
    )
    wall_world, _ = backproject_depth_to_world_from_pose(wall_depth, camera_pose, intr)
    wall_visible_pts = wall_world[0][wall_visible_mask].detach().cpu()
    # Measure residuals against the actual solid-panel visual, not the sparse legacy
    # mesh.  Otherwise valid wall samples can be falsely reported as leaks through
    # holes in the baseline mesh sampling.
    silhouette = panel_silhouette
    wall_wins = silhouette & torch.isfinite(wall_depth[0]) & (wall_depth[0] < panel_depth[0])
    residual_count = int(wall_wins.sum().item())
    silhouette_count = int(silhouette.sum().item())
    wall_visible_count = int((torch.isfinite(wall_depth[0]) & ~torch.isfinite(door_depth[0])).sum().item())
    residual_world, _ = backproject_depth_to_world_from_pose(wall_depth, camera_pose, intr)
    residual_pts = residual_world[0][wall_wins].detach().cpu()
    colors = np.concatenate([
        np.tile([[0.58, 0.58, 0.58]], (len(scene_pts), 1)),
        np.tile([[0.05, 0.85, 1.0]], (len(robot_pts), 1)),
    ], axis=0).astype(np.float32)
    colored = np.concatenate([scene_pts, robot_pts], axis=0)
    out_ply = args.output_dir / "dex_style_close_robot_great_walls.ply"
    write_ply(out_ply, colored, colors)

    # Camera-view plus world-space view for quick inspection.
    basis = _camera_basis_from_pose_x_forward(camera_pose)
    origin = camera_pose[0, :3]
    u, w, v = basis[0][0], basis[1][0], basis[2][0]
    def project(x):
        rel = torch.as_tensor(x, device=device) - origin
        return ((rel * u).sum(-1).cpu().numpy(), (rel * w).sum(-1).cpu().numpy())
    sx, sy = project(scene_pts); rx, ry = project(robot_pts)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax[0].scatter(sx, sy, s=0.25, c="#999999", label=f"walls ({args.wall_mode})")
    ax[0].scatter(rx, ry, s=0.8, c="#00c8ff", label="robot")
    ax[0].set_title(f"DEX-style rendered camera cloud ({args.width}×{args.height})")
    ax[0].set_xlabel("camera right (m)"); ax[0].set_ylabel("camera down (m)"); ax[0].legend(markerscale=5)
    ax[0].set_aspect("equal")
    ax[1].scatter(scene_pts[:, 0], scene_pts[:, 2], s=0.25, c="#999999")
    ax[1].scatter(robot_pts[:, 0], robot_pts[:, 2], s=0.8, c="#00c8ff")
    ax[1].set_title("world x/z projection")
    ax[1].set_aspect("equal")
    fig.savefig(args.output_dir / "dex_style_wall_scene.png", dpi=170)
    plt.close(fig)

    print(f"baseline_visual_points={baseline_visual.shape[0]} replacement_visual_points={n_door} wall_points={len(walls)} robot_input_points={robot.shape[1]}")
    print(f"door_visual_mode={'urdf_visuals' if args.use_urdf_visuals else 'legacy_solid_panel_proxy'}")
    print(f"cached_assembly_mean_ms={cached_timing['mean_ms']:.3f} repeats={cached_timing['repeats']}")
    print(
        f"initial_state_loaded={bool(initial_state)} "
        f"robot_base_pos={base_pos.detach().cpu().tolist()} "
        f"door_world_pos={door_world_pos.detach().cpu().tolist()} camera={args.width}x{args.height}"
    )
    print(f"rendered_points={len(points)} scene_visible={len(scene_pts)} robot_visible={len(robot_pts)}")
    print(f"wall_mode={args.wall_mode} panel_silhouette_pixels={silhouette_count} wall_residual_behind_door={residual_count} wall_only_visible_outside_panel={wall_visible_count}")
    print(f"wall_residual_fraction={residual_count / max(1, silhouette_count):.6f}")
    print(f"ply={out_ply}")

    payload = {
        "format": "dooropening_viser_replay_v1",
        "pointcloud_frame": "world",
        "pointcloud_source": "dex_style_point_zbuffer",
        "sampler_ab": {
            "baseline": "FrankaGripperSampler visual mesh sampling",
            "replacement": "CachedVisualSceneSampler with link_1 solid visual proxy",
            "cached_assembly_mean_ms": cached_timing["mean_ms"],
            "wall_mode": args.wall_mode,
            "panel_silhouette_pixels": silhouette_count,
            "wall_residual_behind_door": residual_count,
            "wall_residual_fraction": residual_count / max(1, silhouette_count),
        },
        "pointcloud_streams": [
            {"name": "ground_truth", "label": "Solid panel + walls + robot", "color": (120, 120, 120)},
            {"name": "solid_door_panel_visible", "label": "Camera-visible panel (one layer)", "color": (255, 193, 7)},
            {"name": "walls_visible", "label": "Camera-visible solid walls", "color": (100, 100, 255)},
            {"name": "robot_depth_cam_obs", "label": "DEX-style depth render", "color": (79, 195, 247)},
            {"name": "robot_visible", "label": "Camera-visible robot", "color": (0, 170, 120)},
        ],
        "frame_dt": 0.5,
        "frame_fps": 2.0,
        "frames": [{
            "pointclouds": {
                "ground_truth": torch.from_numpy(colored).to(dtype=torch.float16),
                "solid_door_panel_visible": panel_visible_pts.to(torch.float16),
                "walls_visible": wall_visible_pts.to(torch.float16),
                "robot_depth_cam_obs": points.astype(np.float16),
                "robot_visible": robot_pts.astype(np.float16),
            }
        }],
    }
    if args.single_stream:
        payload["pointcloud_streams"] = [
            {"name": "policy_input", "label": "Final camera-visible policy cloud", "color": (120, 190, 230)}
        ]
        payload["frames"][0]["pointclouds"] = {
            "policy_input": payload["frames"][0]["pointclouds"]["robot_depth_cam_obs"]
        }
    elif args.show_ground_truth_sources:
        payload["pointcloud_streams"].extend([
            {"name": "ground_truth_door", "label": "Ground-truth door: panel + frame + handle", "color": (255, 193, 7)},
            {"name": "ground_truth_walls", "label": "Ground-truth wall distractors", "color": (100, 100, 255)},
            {"name": "ground_truth_robot", "label": "Ground-truth robot geometry", "color": (0, 170, 120)},
        ])
        payload["frames"][0]["pointclouds"].update({
            "ground_truth_door": door.detach().cpu().to(torch.float16),
            "ground_truth_walls": walls.detach().cpu().to(torch.float16),
            "ground_truth_robot": robot[0].detach().cpu().to(torch.float16),
        })
    if args.num_frames > 1:
        frame = payload["frames"][0]
        payload["frames"] = [frame for _ in range(int(args.num_frames))]
    viser_out = args.output_dir / "dex_style_close_robot_solid_panel.pt"
    torch.save(payload, viser_out)
    print(f"viser={viser_out}")


if __name__ == "__main__":
    main()
