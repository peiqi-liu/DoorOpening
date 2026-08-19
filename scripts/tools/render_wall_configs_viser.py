#!/usr/bin/env python3
"""Sweep wall-distractor obstacle configs and render them for a viser sim-vs-render comparison.

This is a *standalone* tool (no Isaac Sim / no `carb`): the door board is approximated by a simple
cube, and only the wall distractors are re-sampled per config. For each config it produces three
overlaid point clouds, all in world coordinates, matching the raw viser replay format used by the
training pipeline (see DoorOpening.distillation._dagger_viser):

    - ground_truth       : the "sim" reference geometry  = cube board surface + wall distractor points
    - robot_depth_cam_obs : the analytic RealSense depth render of that geometry (with occluders)
    - policy_input        : the cropped cloud actually fed to the policy (base crop)

The wall-distractor sampling, camera intrinsics, depth render, and policy crop are all driven by
`pcd_transformer_dagger_cfg.yaml` so this matches what the policy sees in training.

Each viser frame is one obstacle config. Play the result with:

    python scripts/replay_viser_pt.py <out.pt>

The wall-distractor sampling is imported directly from `DoorOpening.utils.wall_distractors` (the same
module the training pipeline uses via `Dagger._sample_wall_pointcloud_local`), so there is no logic
to keep in sync.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import yaml

from isaaclab.utils.math import quat_apply
from DoorOpening.utils.camera_utils import crop_local_pcd, simulate_depth_cam_render_from_pose
from DoorOpening.utils.door_window_dropout import (
    apply_window_dropout_to_door_points,
    sample_random_window_hole_metadata,
)
from DoorOpening.utils.pose_utils import world_to_local
from DoorOpening.utils.urdf_utils import compute_exact_door_keypoints
from DoorOpening.utils.wall_distractors import (
    WallDistractorParams,
    compute_wall_bbox_ordering,
    sample_wall_points_local,
)

DEFAULT_STUDENT_CFG = (
    SOURCE_ROOT
    / "DoorOpening"
    / "tasks"
    / "dooropening"
    / "agents"
    / "pcd_transformer_dagger_cfg.yaml"
)
DEFAULT_DOOR_URDF = (
    SOURCE_ROOT / "DoorOpening" / "assets" / "door" / "v5_test" / "scratch_door__rnd_01" / "mobility.urdf"
)
GLORBOT_DIR = (SOURCE_ROOT / "DoorOpening" / "assets" / "glorbot").resolve()
GLORBOT_URDF = GLORBOT_DIR / "glorbot.urdf"
# RealSense mount offset on x5_camera_link: -45deg roll about the optical axis (matches
# POINTCLOUD_CAMERA_QUAT = quat_from_euler_xyz(-pi/4, 0, 0) in multi_dooropening_env_cfg.py).
CAMERA_MOUNT_EULER_XYZ = (-math.pi / 4.0, 0.0, 0.0)
# Franka "ready" arm pose (matches generate_randomized_doors_scratch.FRANKA_DEFAULT_JOINT_POS).
FRANKA_READY_JOINT_POS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0]


# --------------------------------------------------------------------------------------
# URDF asset loading (no Isaac; yourdfpy + trimesh do FK + mesh surface sampling)
# --------------------------------------------------------------------------------------
def _load_urdf(urdf_path, package_map=None):
    import yourdfpy

    kwargs = dict(build_scene_graph=True)
    if package_map:
        def handler(fname):
            for pkg, root in package_map.items():
                fname = fname.replace(f"package://{pkg}/", str(root) + "/")
            return fname

        kwargs["filename_handler"] = handler
    return yourdfpy.URDF.load(str(urdf_path), **kwargs)


def _sample_scene_surface(robot, num_points, device):
    """Area-weighted surface sample of the posed URDF's concatenated visual mesh (root frame)."""
    import trimesh

    scene = robot.scene
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    pts, _ = trimesh.sample.sample_surface(mesh, int(num_points))
    return torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=device)


def build_robot_link_cache(robot, num_points):
    """Sample each robot link's surface ONCE in its own frame (area-weighted to total num_points).

    Re-posing then only needs the cheap per-link FK transforms applied to these cached points, instead
    of re-concatenating + re-sampling the full ~1M-triangle robot mesh every config (~900ms -> ~12ms).
    This is the same idea as Dagger's compose_cached_link_pointcloud_world.
    """
    import trimesh

    scene = robot.scene
    nodes = list(scene.graph.nodes_geometry)
    areas = {n: float(scene.geometry[scene.graph.get(n)[1]].area) for n in nodes}
    total = sum(areas.values()) or 1.0
    cache = []
    for n in nodes:
        k = max(1, int(round(num_points * areas[n] / total)))
        geom = scene.geometry[scene.graph.get(n)[1]]
        pts, _ = trimesh.sample.sample_surface(geom, k)
        cache.append((n, np.asarray(pts, dtype=np.float64)))
    return cache


def sample_robot_points_cached(robot, cache, device):
    """Transform the cached per-link points by the CURRENT FK transforms -> base_link frame (M, 3).

    Call after robot.update_cfg(cfg). ~70x faster than re-meshing + re-sampling the whole robot.
    """
    scene = robot.scene
    parts = []
    for node, local in cache:
        T = scene.graph.get(node)[0]  # geometry -> scene root (== base_link)
        parts.append(local @ T[:3, :3].T + T[:3, 3])
    pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float64)
    return torch.as_tensor(pts, dtype=torch.float32, device=device)


def load_door_asset(urdf_path, num_points, device):
    """Real door: mesh surface points + the FULL door outer bbox (frame + panel + handle), base frame.

    The door is loaded at its default (closed) joints. Walls are placed against the full door bbox --
    the exact same `door_full_bbox_base` training now uses (via door_full_bboxes) -- so walls sit
    outside the whole door, not just the link_1 panel.
    """
    robot = _load_urdf(urdf_path)
    door_pts = _sample_scene_surface(robot, num_points, device).unsqueeze(0)  # (1, N, 3)
    kp = compute_exact_door_keypoints(str(urdf_path))
    full_bbox = kp.get("door_full_bbox_base", kp["link_1_bbox_base"])
    bbox = torch.as_tensor(full_bbox, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 2, 3)
    # Panel (link_1) bbox + pose in the link_1 LOCAL frame -- what the window-hole aug samples in
    # (multi_pcd_dagger.env_board_bboxes_link1 / _get_link1_pose_world).
    panel_bbox_link1 = torch.as_tensor(
        kp.get("link_1_bbox_link1", kp["link_1_bbox_base"]), dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 2, 3)
    link1_pose_base = torch.as_tensor(
        kp.get("link_1_pose_base", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32, device=device
    )  # (7,) [pos, quat_wxyz]
    handle_center = np.asarray(kp.get("link_2_center_base", [0.0, 0.0, 0.0]), dtype=np.float64)  # base frame
    return bbox, panel_bbox_link1, link1_pose_base, door_pts, handle_center


def load_robot_asset(num_points, device):
    """Glorbot surface points (base_link frame) + x5_camera_link transform + the joint config used.

    Returns (points, cam_T_base, joint_names, joint_cfg). The base_x/base_y/base_rotation joints stay
    at 0 (the base is placed via base_pos/base_quat), so joint_cfg is already in the base frame -- the
    same convention _get_robot_filter_joint_pos_base_frame uses for the self-point SDF filter.
    """
    robot = _load_urdf(GLORBOT_URDF, {"glorbot": GLORBOT_DIR})
    names = list(robot.actuated_joint_names)
    cfg = np.zeros(len(names), dtype=np.float64)
    for i, value in enumerate(FRANKA_READY_JOINT_POS):
        jn = f"panda_joint{i + 1}"
        if jn in names:
            cfg[names.index(jn)] = value
    robot.update_cfg(cfg)
    pts = _sample_scene_surface(robot, num_points, device)  # (M, 3) in base_link frame
    cam_T_base = np.asarray(robot.get_transform("x5_camera_link", robot.base_link), dtype=np.float64)  # 4x4
    return pts, cam_T_base, names, cfg


def _quat_wxyz_to_matrix(quat_wxyz):
    from scipy.spatial.transform import Rotation

    w, x, y, z = [float(v) for v in quat_wxyz]
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _mount_offset_matrix():
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", CAMERA_MOUNT_EULER_XYZ).as_matrix()


def robot_camera_pose_world(cam_T_base, base_pos, base_R):
    """World camera pose [pos(3), quat_xyzw(4)] = base_world @ x5_camera_link @ mount_offset.

    Mirrors Dagger._get_sampler_camera_pose: link world pose then the -45deg roll mount offset.
    """
    cam_R_world = base_R @ cam_T_base[:3, :3] @ _mount_offset_matrix()
    cam_pos_world = np.asarray(base_pos, dtype=np.float64) + base_R @ cam_T_base[:3, 3]
    quat_xyzw = rotmat_to_quat_xyzw(cam_R_world)
    return np.concatenate([cam_pos_world, quat_xyzw]).astype(np.float32)


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def sample_box_surface(bmin, bmax, num_points, device):
    """Uniformly sample `num_points` on the surface of an axis-aligned box, area-weighted per face."""
    bmin = torch.as_tensor(bmin, dtype=torch.float32, device=device)
    bmax = torch.as_tensor(bmax, dtype=torch.float32, device=device)
    ext = (bmax - bmin).clamp_min(1e-6)
    ex, ey, ez = ext.tolist()
    # face areas: two each of xy, yz, xz
    areas = torch.tensor([ex * ez, ex * ez, ey * ez, ey * ez, ex * ey, ex * ey], device=device)
    face_idx = torch.multinomial(areas / areas.sum(), num_points, replacement=True)
    uvw = torch.rand((num_points, 3), device=device)
    pts = bmin + uvw * ext
    # snap the free axis to the chosen face plane
    # faces: 0/1 = y-min/y-max, 2/3 = x-min/x-max, 4/5 = z-min/z-max
    pts[:, 1] = torch.where(face_idx == 0, bmin[1], pts[:, 1])
    pts[:, 1] = torch.where(face_idx == 1, bmax[1], pts[:, 1])
    pts[:, 0] = torch.where(face_idx == 2, bmin[0], pts[:, 0])
    pts[:, 0] = torch.where(face_idx == 3, bmax[0], pts[:, 0])
    pts[:, 2] = torch.where(face_idx == 4, bmin[2], pts[:, 2])
    pts[:, 2] = torch.where(face_idx == 5, bmax[2], pts[:, 2])
    return pts


def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def yaw_quat_wxyz(yaw_rad):
    half = 0.5 * float(yaw_rad)
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=torch.float32)


def rotmat_to_quat_xyzw(R):
    """3x3 rotation matrix (columns = local axes in world) -> quaternion (x, y, z, w)."""
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def look_at_camera_pose(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Build a camera pose [pos(3), quat_xyzw(4)] with the x-forward convention used by camera_utils.

    Local axes (columns of R): x = forward (optical axis toward target), y = image-right,
    z = image-down. This matches `_camera_basis_from_pose_x_forward` in camera_utils.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)
    forward = _normalize(target - eye)
    if abs(float(np.dot(forward, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(forward, world_up))  # image +u (right)
    down = np.cross(forward, right)  # image +v (down); == col0 x col1 -> right-handed
    R = np.stack([forward, right, down], axis=1)  # columns are the local axes
    quat_xyzw = rotmat_to_quat_xyzw(R)
    return np.concatenate([eye.astype(np.float32), quat_xyzw]).astype(np.float32)


def downsample(points, max_points):
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points[idx]


def drop_zero_rows(points):
    return points[(points.abs().sum(dim=-1) > 1e-9)]


def drop_invalid_rows(points):
    """Drop NaN/inf rows (wall density padding) and exact-zero padding rows."""
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite]
    return points[(points.abs().sum(dim=-1) > 1e-9)]


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student-cfg", type=Path, default=DEFAULT_STUDENT_CFG, help="pcd_transformer_dagger_cfg.yaml path.")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "wall_config_sweep.pt", help="Output .pt (replay_viser_pt.py compatible).")
    p.add_argument("--num-configs", type=int, default=64, help="Number of obstacle configs (= viser frames).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-points", type=int, default=8000, help="Per-cloud point cap for viser display.")
    # Door geometry: a real door URDF (default) or the simple cube approximation.
    p.add_argument("--door", type=Path, default=DEFAULT_DOOR_URDF,
                   help="Door URDF whose meshes + link_1 (panel) bbox drive the scene and wall placement.")
    p.add_argument("--cube", action="store_true",
                   help="Use the simple cube board instead of a real door URDF (legacy).")
    p.add_argument("--door-yaw-deg", type=float, default=None,
                   help="Yaw (deg about world z) applied to the door. Default: AUTO -- pick -90 or +90 so "
                   "the HANDLE faces the robot/camera (so its point cloud is visible). Pass a value to override; "
                   "ignored for --cube.")
    p.add_argument("--robot", action="store_true",
                   help="Include the glorbot robot in the scene; the camera is the robot's own "
                   "x5_camera_link (-45deg mount). The franka arm + base pose are RE-SAMPLED every "
                   "config, so you can see how different joint angles / base poses occlude the view.")
    # Per-config robot pose sampling (only with --robot). The arm moves in front of the fixed
    # base-mounted camera -> different franka angles occlude the door differently.
    p.add_argument("--arm-jitter-rad", type=float, default=0.6,
                   help="Per-config franka joint jitter (rad) around the forward 'reaching' pose. The "
                   "arm reaches toward the door (in front of the base-mounted camera) and jitters by "
                   "this much, so it occludes the view differently each config. 0 = fixed reach pose.")
    p.add_argument("--standoff-range", type=float, nargs=2, default=[0.7, 1.3], metavar=("MIN", "MAX"),
                   help="Per-config robot base distance from the door (m).")
    p.add_argument("--lateral-range", type=float, nargs=2, default=[-0.25, 0.25], metavar=("MIN", "MAX"),
                   help="Per-config robot base left/right offset along world X (m).")
    p.add_argument("--yaw-jitter-deg", type=float, default=15.0,
                   help="Per-config robot base yaw jitter (deg) around facing the door.")
    # Cube "door board" dimensions (m), only used with --cube. Defaults are the MEASURED means of the
    # real scratch_door panels (width 0.70-1.00, height 1.75-2.15, thickness 0.028-0.055); the wall
    # distractors attach to this panel's width edges, so panel size affects wall placement.
    p.add_argument("--panel-width", type=float, default=0.85)
    p.add_argument("--panel-height", type=float, default=1.97)
    p.add_argument("--panel-thickness", type=float, default=0.041)
    p.add_argument("--panel-bottom-z", type=float, default=0.0, help="World z of the panel bottom edge.")
    p.add_argument("--randomize-board", action="store_true",
                   help="Sample panel w/h/t per config from the real scratch_door ranges "
                   "(width 0.70-1.00, height 1.75-2.15, thickness 0.028-0.055 m).")
    p.add_argument("--board-num-points", type=int, default=None, help="GT board surface points (default: scene_door_num_points).")
    # Virtual robot / camera placement.
    p.add_argument("--standoff", type=float, default=1.0, help="Robot base distance from the door along -Y (m).")
    p.add_argument("--camera-height", type=float, default=1.0, help="Camera height above the floor (m).")
    p.add_argument("--camera-look-z", type=float, default=1.0, help="World z the camera aims at on the panel.")
    p.add_argument("--camera-right", type=float, default=0.12,
                   help="Lateral camera offset to the robot's RIGHT (world +X), since the real "
                   "RealSense sits on the right of the robot. Set 0 for a centered camera.")
    # Matches the training analytic RealSense spec: 640x480 // 2 = 320x240 at FOV 85.2x58 deg,
    # range 0.3-3.0 m (Dagger._build_sampler_camera_spec). This is a D435-like spec, NOT the Isaac
    # PinholeCamera optics (focal_length=8.0, clip 0.1-20 m) used by the on-robot sim sensor.
    p.add_argument("--cam-width-px", type=int, default=320)
    p.add_argument("--cam-height-px", type=int, default=240)
    # Wall vertical extent [lower, upper]. Walls start at `lower`; the upper edge is randomized per
    # config in [lower, upper] (same as training). Defaults to height_range_m from the cfg.
    p.add_argument("--wall-height-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="Override the wall height window (default: use height_range_m from the cfg).")
    p.add_argument("--use-compile", action="store_true", help="Use the torch.compile depth renderer (slower first call).")
    return p.parse_args()


def build_camera_spec(width_px, height_px, device):
    """Mirror Dagger._build_sampler_camera_spec: D435-like FOV/range with a pinhole intrinsics matrix."""
    fov_x_deg, fov_y_deg = 85.2, 58.0
    near_m, far_m = 0.3, 3.0
    fx = width_px / (2.0 * math.tan(math.radians(fov_x_deg) * 0.5))
    fy = height_px / (2.0 * math.tan(math.radians(fov_y_deg) * 0.5))
    cx = (width_px - 1.0) * 0.5
    cy = (height_px - 1.0) * 0.5
    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    return {"H": height_px, "W": width_px, "intrinsics": intrinsics, "near_m": near_m, "far_m": far_m}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    cfg = yaml.safe_load(args.student_cfg.read_text()) or {}
    wall_cfg = dict(cfg.get("dagger", {}).get("wall_distractors", {}))
    depth_cfg = dict(cfg.get("dagger", {}).get("depth_cam_render", {}))
    scene_door_num_points = int(cfg.get("scene_door_num_points", cfg.get("door_pcd_num_points", 30000)))
    local_pcd_range = list(cfg.get("local_pcd_range", [1.2, 0.25, 0.25]))
    x_direction_cutoff = float(cfg.get("x_direction_cutoff", -0.5))
    local_pcd_points = list(
        cfg.get("pcd_encoders_cfg", {}).get("local_pcd_t", {}).get("num_points", [2500, 0, 0])
    )
    base_crop_points = int(local_pcd_points[0])
    board_num_points = int(args.board_num_points or scene_door_num_points)
    dagger_cfg = dict(cfg.get("dagger", {}))
    scene_robot_num_points = int(dagger_cfg.get("scene_robot_num_points", 30000))
    robot_model_policy_points = int(dagger_cfg.get("robot_model_policy_points", 2000))
    append_robot_model = bool(dagger_cfg.get("append_robot_model_to_policy_cloud", True))

    # Real scratch_door panel ranges (measured from the training assets).
    board_width_range = (0.70, 1.00)
    board_height_range = (1.75, 2.15)
    board_thickness_range = (0.028, 0.055)

    def build_board(width, height, thickness):
        # Cube "door board" bbox in the door-base frame (== world; door base at origin, identity).
        # x = width (left/right), y = thickness (through slab), z = height (up).
        hw, ht = 0.5 * width, 0.5 * thickness
        z0 = args.panel_bottom_z
        z1 = args.panel_bottom_z + height
        bbox = torch.tensor([[[-hw, -ht, z0], [hw, ht, z1]]], dtype=torch.float32, device=device)
        gt = sample_box_surface([-hw, -ht, z0], [hw, ht, z1], board_num_points, device).unsqueeze(0)
        return bbox, gt

    # Shared wall-distractor params + sampler (same code the training pipeline uses).
    wall_params = WallDistractorParams.from_cfg(wall_cfg, scene_door_num_points)
    if args.wall_height_range is not None:
        wall_params.height_min_m, wall_params.height_max_m = map(float, args.wall_height_range)

    def sample_walls(bbox):
        axis_order, bbox_min_ordered, bbox_max_ordered = compute_wall_bbox_ordering(bbox)
        return sample_wall_points_local(
            axis_order=axis_order,
            bbox_min_ordered=bbox_min_ordered,
            bbox_max_ordered=bbox_max_ordered,
            num_points=wall_params.num_points,
            params=wall_params,
            device=device,
        )

    # --- Door geometry: real door URDF (default) or the legacy cube. Both live in the door-base
    # frame at the world origin; the panel (link_1) bbox drives wall placement. ---
    door_desc = "cube"
    handle_center = None
    panel_bbox_link1 = None
    link1_pose_base = None
    if args.cube:
        board_bbox, board_gt = build_board(args.panel_width, args.panel_height, args.panel_thickness)
        door_desc = f"cube {args.panel_width}x{args.panel_height}x{args.panel_thickness} m"
    else:
        # board_bbox is the FULL door outer bbox (frame + panel + handle) -- the same door_full_bboxes
        # training now uses for wall placement, so walls sit outside the whole door with a real gap.
        board_bbox, panel_bbox_link1, link1_pose_base, board_gt, handle_center = load_door_asset(
            args.door, board_num_points, device
        )
        door_desc = f"door urdf {args.door.parent.name}"
    wall_bbox = board_bbox

    # Orient the door in the world so its FACE points toward the robot (+Y). By default AUTO-pick the
    # yaw (-90 or +90) that puts the HANDLE on the robot-facing (-Y) side, so its cloud is visible.
    # Walls are sampled in the door-base frame and rotated the same way, so they stay glued to the door.
    if args.cube:
        door_yaw = 0.0
    elif args.door_yaw_deg is not None:
        door_yaw = math.radians(args.door_yaw_deg)
    else:
        def _handle_world_y(theta_deg):
            th = math.radians(theta_deg)
            return handle_center[0] * math.sin(th) + handle_center[1] * math.cos(th)  # world y after Rz

        door_yaw = math.radians(-90.0 if _handle_world_y(-90.0) <= _handle_world_y(90.0) else 90.0)
        door_desc += f" (auto yaw {round(math.degrees(door_yaw))}deg -> handle faces robot)"
    cos_y, sin_y = math.cos(door_yaw), math.sin(door_yaw)
    R_door = torch.tensor(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
    )

    def door_to_world(pts):  # (1, N, 3) door-base frame -> world
        return pts @ R_door.T

    board_gt = door_to_world(board_gt)

    # --- Window-hole aug: same knobs multi_pcd_dagger reads. One hole is drawn PER CONFIG (= per
    # rollout) and baked into that config's door cloud as NaN, matching the per-rollout training
    # behaviour (the hole no longer jitters step-to-step). Real door only (needs the link_1 frame). ---
    hole_cfg = dict(dagger_cfg.get("door_hole_aug", {}))
    hole_aug_enabled = bool(hole_cfg.get("enabled", False)) and (link1_pose_base is not None)
    hole_env_prob = float(hole_cfg.get("env_prob", 0.35))
    hole_width_range = tuple(float(v) for v in hole_cfg.get("width_range_m", [0.12, 1.60]))
    hole_height_range = tuple(float(v) for v in hole_cfg.get("height_range_m", [0.18, 2.20]))
    hole_center_height_range = tuple(float(v) for v in hole_cfg.get("center_height_range_m", [0.10, 1.90]))
    hole_side_margin_range = tuple(float(v) for v in hole_cfg.get("side_margin_range_m", [0.0, 0.18]))
    hole_surface_eps = float(hole_cfg.get("surface_eps_m", 0.03))
    link1_pose_world = None
    if hole_aug_enabled:
        from scipy.spatial.transform import Rotation as _R

        link1_pos_base_np = link1_pose_base[:3].cpu().numpy()
        link1_quat_base_wxyz = link1_pose_base[3:].cpu().numpy()
        R_base_link1 = _R.from_quat(
            [link1_quat_base_wxyz[1], link1_quat_base_wxyz[2], link1_quat_base_wxyz[3], link1_quat_base_wxyz[0]]
        )
        R_door_np = R_door.cpu().numpy()
        link1_pos_world_np = R_door_np @ link1_pos_base_np
        link1_quat_world_xyzw = _R.from_matrix(R_door_np @ R_base_link1.as_matrix()).as_quat()
        link1_pose_world = torch.tensor(
            [
                *link1_pos_world_np.tolist(),
                float(link1_quat_world_xyzw[3]),
                float(link1_quat_world_xyzw[0]),
                float(link1_quat_world_xyzw[1]),
                float(link1_quat_world_xyzw[2]),
            ],
            dtype=torch.float32,
            device=device,
        )

    def apply_hole(door_world):
        """Drop door-surface points inside a freshly-sampled window hole (NaN, fixed N). Same helpers
        multi_pcd_dagger uses. door_world: (1, N, 3) world; returns (1, N, 3) with holes as NaN."""
        if not hole_aug_enabled:
            return door_world
        metadata = sample_random_window_hole_metadata(
            link1_pose_world=link1_pose_world,
            board_bbox_link1=panel_bbox_link1[0],
            window_prob=hole_env_prob,
            width_range=hole_width_range,
            height_range=hole_height_range,
            center_height_range=hole_center_height_range,
            side_margin_range=hole_side_margin_range,
        )
        dropped, _ = apply_window_dropout_to_door_points(
            points_world=door_world[0],
            link1_pose_world=link1_pose_world,
            board_bbox_link1=panel_bbox_link1[0],
            hole_metadata=metadata,
            surface_eps=hole_surface_eps,
        )
        return dropped.unsqueeze(0)

    # --- Robot base: on the -Y side, +x_base points toward the door (+Y). +90deg yaw => robot faces
    # +Y, so its RIGHT-hand side is world +X (where the real RealSense sits). ---
    base_pos = torch.tensor([[0.0, -args.standoff, 0.0]], dtype=torch.float32, device=device)
    base_quat = yaw_quat_wxyz(math.pi / 2.0).to(device).unsqueeze(0)  # +90deg yaw: base +x -> world +Y
    base_R = _quat_wxyz_to_matrix(base_quat[0].detach().cpu().tolist())

    def base_to_world(pts):  # (1, N, 3) robot base frame -> world
        return quat_apply(base_quat.unsqueeze(1).expand(-1, pts.shape[1], -1), pts) + base_pos.unsqueeze(1)

    # --- Robot: loaded once; the ARM + BASE pose are re-sampled per config (in the loop), so the
    # arm occludes the camera differently each frame. Camera = the robot's own x5_camera_link. ---
    robot_obj = None
    robot_joint_names = None
    robot_link_cache = None
    panda_joint_idx = None
    panda_limits = None
    robot_collision_checker = None
    robot_pts_base = None            # (M, 3) base frame -- set per config
    robot_world = None               # (1, M, 3) world -- set per config
    robot_filter_joint_angles = None  # (1, num_joints) -- set per config
    robot_filter_cfg = dict(dagger_cfg.get("robot_pointcloud_filter", {}))
    robot_filter_enabled = bool(robot_filter_cfg.get("enabled", True))
    robot_sdf_cutoff = float(robot_filter_cfg.get("sdf_cutoff", 0.02))
    robot_filter_max_pts = int(robot_filter_cfg.get("max_points_per_process", 5000))
    if args.robot:
        robot_obj = _load_urdf(GLORBOT_URDF, {"glorbot": GLORBOT_DIR})
        robot_joint_names = list(robot_obj.actuated_joint_names)
        robot_link_cache = build_robot_link_cache(robot_obj, scene_robot_num_points)  # sample links ONCE
        panda_joint_idx = [robot_joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        panda_limits = []
        for i in range(1, 8):
            lim = robot_obj.joint_map[f"panda_joint{i}"].limit
            lo = float(lim.lower) if lim is not None and lim.lower is not None else -2.9
            hi = float(lim.upper) if lim is not None and lim.upper is not None else 2.9
            panda_limits.append((lo, hi))
        if robot_filter_enabled:
            from DoorOpening.utils.glorbot_collision_checker import GlorbotCollisionChecker

            robot_collision_checker = GlorbotCollisionChecker(
                str(GLORBOT_URDF), device, input_joint_names=robot_joint_names
            )
        cam_desc = "x5_camera_link (per-config base+arm pose, mount -45deg roll)"
    else:
        # Virtual look-at camera, shifted to the robot's right (real RealSense is right-mounted).
        eye = np.array([args.camera_right, -args.standoff, args.camera_height], dtype=np.float32)
        target = np.array([0.0, 0.0, args.camera_look_z], dtype=np.float32)
        camera_pose = torch.from_numpy(look_at_camera_pose(eye, target)).to(device).unsqueeze(0)
        cam_desc = f"virtual look-at, eye {eye.tolist()} -> {target.tolist()}"

    cam_spec = build_camera_spec(args.cam_width_px, args.cam_height_px, device)

    depth_render_kwargs = dict(
        num_points=int(depth_cfg.get("num_points", 12000)),
        inflate_px=int(depth_cfg.get("inflate_px", 2)),
        jitter_std_m=float(depth_cfg.get("jitter_std_m", 0.025)),
        cam_spec_dict=cam_spec,
        clip_mode=str(depth_cfg.get("clip_mode", "post")),
        jitter_mode=str(depth_cfg.get("jitter_mode", "xyz")),
        use_compile=bool(args.use_compile),
        blur_kernel_px=int(depth_cfg.get("blur_kernel_px", 0)),
        blur_sigma_px=float(depth_cfg.get("blur_sigma_px", 0.0)),
        occluder_inflate_px=int(depth_cfg.get("occluder_inflate_px", 0)),
    )

    print(f"[INFO] student cfg     : {args.student_cfg}")
    print(f"[INFO] wall num_points : {wall_params.num_points}  (enabled={wall_params.enabled}, density={wall_params.point_density_per_m2})")
    print(f"[INFO] door            : {door_desc}  ({board_gt.shape[1]} pts)")
    print(f"[INFO] window hole     : {'on (env_prob=' + str(hole_env_prob) + ', per-config w' + str(hole_width_range) + ' h' + str(hole_height_range) + ')' if hole_aug_enabled else 'off'}")
    print(f"[INFO] robot           : {'on (%d pts, +%d to policy; arm+base RE-SAMPLED per config)' % (scene_robot_num_points, robot_model_policy_points) if args.robot else 'off'}")
    print(f"[INFO] base crop       : {base_crop_points} pts, range {local_pcd_range[0]} m, x_cutoff {x_direction_cutoff}")
    print(f"[INFO] camera          : {args.cam_width_px}x{args.cam_height_px}px, {cam_desc}")
    print(f"[INFO] rendering {args.num_configs} wall configs on {device}...")

    frames = []
    for config_idx in range(args.num_configs):
        if args.cube and args.randomize_board:
            w = float(torch.empty(1).uniform_(*board_width_range).item())
            h = float(torch.empty(1).uniform_(*board_height_range).item())
            t = float(torch.empty(1).uniform_(*board_thickness_range).item())
            board_bbox, board_gt = build_board(w, h, t)
            wall_bbox = board_bbox

        if args.robot:
            # --- Re-sample the robot base pose + franka joint angles for THIS config. The camera is
            # base-mounted (fixed relative to the base) while the arm swings in front of it, so
            # different franka angles / base poses occlude the door differently in the render. ---
            standoff = random.uniform(*args.standoff_range)
            lateral = random.uniform(*args.lateral_range)
            yaw = math.pi / 2.0 + math.radians(random.uniform(-args.yaw_jitter_deg, args.yaw_jitter_deg))
            base_pos = torch.tensor([[lateral, -standoff, 0.0]], dtype=torch.float32, device=device)
            base_quat = yaw_quat_wxyz(yaw).to(device).unsqueeze(0)
            base_R = _quat_wxyz_to_matrix(base_quat[0].detach().cpu().tolist())
            cfg = np.zeros(len(robot_joint_names), dtype=np.float64)  # base + gripper + x5 stay at 0
            # Franka = forward "reaching" pose + per-joint jitter, clamped to limits. This keeps the
            # arm reaching toward the door (in front of the camera) while varying the exact occlusion.
            for j, k in enumerate(panda_joint_idx):
                lo, hi = panda_limits[j]
                nominal = FRANKA_READY_JOINT_POS[j] if j < len(FRANKA_READY_JOINT_POS) else 0.0
                val = nominal + random.uniform(-args.arm_jitter_rad, args.arm_jitter_rad)
                cfg[k] = min(max(val, lo), hi)
            robot_obj.update_cfg(cfg)
            robot_pts_base = sample_robot_points_cached(robot_obj, robot_link_cache, device)  # (M, 3), cached links
            robot_world = base_to_world(robot_pts_base.unsqueeze(0))
            cam_T = np.asarray(robot_obj.get_transform("x5_camera_link", robot_obj.base_link), dtype=np.float64)
            camera_pose = torch.from_numpy(
                robot_camera_pose_world(cam_T, base_pos[0].detach().cpu().numpy(), base_R)
            ).to(device).unsqueeze(0)
            robot_filter_joint_angles = torch.as_tensor(cfg, dtype=torch.float32, device=device).unsqueeze(0)

        wall_world = door_to_world(sample_walls(wall_bbox))  # (1, Nw, 3) sampled in door frame, rotated to world
        # One window hole per config (per rollout), baked into the door cloud as NaN before it feeds
        # BOTH the rendered scene and the occluder pass (so the hole shows up as missing depth).
        board_gt_holed = apply_hole(board_gt)
        # Scene = door + walls (+ robot). Occluder = door + walls only (robot excluded, matching
        # Dagger._sample_scene_pointcloud_world_cached: the robot's thin links must not be dilated).
        scene_parts = [board_gt_holed, wall_world]
        if robot_world is not None:
            scene_parts.append(robot_world)
        gt_world = torch.cat(scene_parts, dim=1)
        occluder_world = torch.cat([board_gt_holed, wall_world], dim=1)

        rendered_world, _ = simulate_depth_cam_render_from_pose(
            pcd=gt_world, camera_pose=camera_pose, occluder_pcd=occluder_world, **depth_render_kwargs
        )  # RAW depth render: includes the robot's own body AND its occlusion of the door/walls.

        rendered_base = world_to_local(rendered_world, base_pos, base_quat)
        # Filtered obs: drop the robot's own body points (Dagger._filter_robot_points_base). This is
        # what training feeds forward; the raw-vs-filtered difference IS the robot's rendering influence.
        filtered_base = rendered_base
        if robot_collision_checker is not None:
            filtered_base = robot_collision_checker.filter_pointcloud_outside_spheres(
                pointclouds=filtered_base,
                joint_angles=robot_filter_joint_angles,
                sdf_cutoff=robot_sdf_cutoff,
                max_points_per_process=robot_filter_max_pts,
            )
        # Policy input = base crop of the FILTERED cloud (Dagger._build_local_pcd) + robot-model points.
        crop_center = torch.zeros((1, 3), dtype=torch.float32, device=device)
        policy_base, _ = crop_local_pcd(
            filtered_base,
            local_range=float(local_pcd_range[0]),
            num_local_points=base_crop_points,
            is_cylindrical=True,
            crop_center=crop_center,
            x_direction_cutoff=x_direction_cutoff,
        )
        if robot_pts_base is not None and append_robot_model and robot_model_policy_points > 0:
            m = robot_pts_base.shape[0]
            idx = torch.linspace(0, m - 1, steps=min(robot_model_policy_points, m), device=device).round().long()
            policy_base = torch.cat([policy_base, robot_pts_base[idx].unsqueeze(0)], dim=1)

        gt_pts = downsample(drop_invalid_rows(gt_world[0]), args.max_points)
        raw_depth_pts = downsample(drop_invalid_rows(rendered_world[0]), args.max_points)
        policy_pts = downsample(drop_invalid_rows(base_to_world(policy_base)[0]), args.max_points)
        pointclouds = {
            "ground_truth": gt_pts.detach().cpu().to(torch.float16),
            "robot_depth_cam_obs": raw_depth_pts.detach().cpu().to(torch.float16),
            "policy_input": policy_pts.detach().cpu().to(torch.float16),
        }
        if robot_collision_checker is not None:
            filt_pts = downsample(drop_invalid_rows(base_to_world(filtered_base)[0]), args.max_points)
            pointclouds["depth_filtered"] = filt_pts.detach().cpu().to(torch.float16)
        frames.append({"pointclouds": pointclouds})
        if (config_idx + 1) % 10 == 0 or config_idx == args.num_configs - 1:
            print(
                f"  config {config_idx + 1}/{args.num_configs}: "
                f"gt={gt_pts.shape[0]} depth_raw={raw_depth_pts.shape[0]} policy={policy_pts.shape[0]}"
            )

    payload = {
        "format": "dooropening_viser_replay_v1",
        "pointcloud_frame": "world",
        "pointcloud_source": "depth",
        "pointcloud_streams": [
            {"name": "ground_truth", "label": "GT (sim: door + walls [+ robot])", "color": (120, 120, 120), "point_size_scale": 1.0},
            {"name": "robot_depth_cam_obs", "label": "Depth Render (raw, +robot self)", "color": (79, 195, 247), "point_size_scale": 1.0},
            *([{"name": "depth_filtered", "label": "Depth (robot self-filtered)", "color": (255, 140, 0), "point_size_scale": 1.0}] if args.robot else []),
            {"name": "policy_input", "label": "Policy Input", "color": (0, 170, 120), "point_size_scale": 1.0},
        ],
        "frame_dt": 0.5,
        "frame_fps": 2.0,
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"[INFO] Saved {len(frames)} configs to {args.output}")
    print(f"[INFO] Play with: python {REPO_ROOT / 'scripts' / 'replay_viser_pt.py'} {args.output}")


if __name__ == "__main__":
    main()
