#!/usr/bin/env python3
"""GT point cloud -> RealSense depth image -> point cloud round-trip, for viser replay.

This is a *standalone* tool (no Isaac Sim). The input point cloud is the SAME "ground_truth" geometry
that `render_wall_configs_viser.py` builds: the real door URDF mesh surface points plus the wall
distractor points (driven by `pcd_transformer_dagger_cfg.yaml`), oriented so the door face points at
the camera. On that cloud it demonstrates the projection round-trip:

    1. take the GT point cloud (door mesh + wall distractors),
    2. PROJECT it into a RealSense-style pinhole camera and z-buffer it to a DEPTH IMAGE
       (`rasterize_depth_zbuffer_from_pose` -> per-pixel nearest surface, occlusion-aware),
    3. BACK-PROJECT that depth image to a world point cloud
       (`backproject_depth_to_world_from_pose`, the inverse of step 2).

Only the surface visible from the camera survives the round-trip (occluded / out-of-frame geometry is
gone), and the returned points land back on the original surface -- that is the thing to eyeball.

Each viser frame re-samples the wall distractors (one config), with these overlaid clouds (world coords):

    ground_truth : the GT input cloud = door mesh + wall distractors (gray)
    reprojected  : that config's depth image back-projected to 3D (blue) -- the round-trip output

Play the result with:

    python scripts/replay_viser_pt.py <out.pt>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import yaml

from DoorOpening.utils.camera_utils import (
    backproject_depth_to_world_from_pose,
    rasterize_depth_zbuffer_from_pose,
)
from DoorOpening.utils.urdf_utils import compute_exact_door_keypoints
from DoorOpening.utils.wall_distractors import (
    WallDistractorParams,
    compute_wall_bbox_ordering,
    sample_wall_points_local,
)

DEFAULT_STUDENT_CFG = (
    SOURCE_ROOT / "DoorOpening" / "tasks" / "dooropening" / "agents" / "pcd_transformer_dagger_cfg.yaml"
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
# Robot base lateral offset along world X (0 = dead-center in front of the door, like render_wall_configs's
# nominal base). The camera moves with the base. Constant on purpose (not a per-run config).
ROBOT_RIGHT_M = 0.0


# --------------------------------------------------------------------------------------
# GT geometry (copied from render_wall_configs_viser.py: door URDF surface + wall distractors)
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

    mesh = robot.scene.dump(concatenate=True)
    pts, _ = trimesh.sample.sample_surface(mesh, int(num_points))
    return torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=device)


def load_door_asset(urdf_path, num_points, device):
    """Real door: mesh surface points + the link_1 (panel) bbox + handle center, in the door-base frame."""
    robot = _load_urdf(urdf_path)
    door_pts = _sample_scene_surface(robot, num_points, device).unsqueeze(0)  # (1, N, 3)
    kp = compute_exact_door_keypoints(str(urdf_path))
    bbox = torch.as_tensor(kp["link_1_bbox_base"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 2, 3)
    handle_center = np.asarray(kp.get("link_2_center_base", [0.0, 0.0, 0.0]), dtype=np.float64)  # base frame
    return bbox, door_pts, handle_center


def load_robot_asset(num_points, device):
    """Glorbot surface points (base_link frame) + the x5_camera_link transform in base_link frame."""
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
    return pts, cam_T_base


def _quat_wxyz_to_matrix(quat_wxyz):
    from scipy.spatial.transform import Rotation

    w, x, y, z = [float(v) for v in quat_wxyz]
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _mount_offset_matrix():
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", CAMERA_MOUNT_EULER_XYZ).as_matrix()


def yaw_quat_wxyz(yaw_rad):
    half = 0.5 * float(yaw_rad)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def robot_camera_pose_world(cam_T_base, base_pos, base_R):
    """World camera pose [pos(3), quat_xyzw(4)] = base_world @ x5_camera_link @ mount_offset.

    Mirrors Dagger._get_sampler_camera_pose / render_wall_configs_viser: link world pose then the
    -45deg roll mount offset about the optical axis.
    """
    cam_R_world = base_R @ cam_T_base[:3, :3] @ _mount_offset_matrix()
    cam_pos_world = np.asarray(base_pos, dtype=np.float64) + base_R @ cam_T_base[:3, 3]
    quat_xyzw = rotmat_to_quat_xyzw(cam_R_world)
    return np.concatenate([cam_pos_world, quat_xyzw]).astype(np.float32)


# --------------------------------------------------------------------------------------
# Camera pose helpers (x-forward convention used by camera_utils)
# --------------------------------------------------------------------------------------
def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def rotmat_to_quat_xyzw(R):
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def look_at_camera_pose(eye, target, world_up=(0.0, 0.0, 1.0)):
    """[pos(3), quat_xyzw(4)]; columns of R: x=forward (optical axis), y=image-right, z=image-down."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)
    forward = _normalize(target - eye)
    if abs(float(np.dot(forward, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(forward, world_up))
    down = np.cross(forward, right)
    R = np.stack([forward, right, down], axis=1)
    return np.concatenate([eye.astype(np.float32), rotmat_to_quat_xyzw(R)]).astype(np.float32)


def build_camera_spec(width_px, height_px, near_m, far_m, device):
    """RealSense D435-like intrinsics: FOV 85.2x58 deg, range [near, far] m (Dagger sampler spec)."""
    fov_x_deg, fov_y_deg = 85.2, 58.0
    fx = width_px / (2.0 * math.tan(math.radians(fov_x_deg) * 0.5))
    fy = height_px / (2.0 * math.tan(math.radians(fov_y_deg) * 0.5))
    cx = (width_px - 1.0) * 0.5
    cy = (height_px - 1.0) * 0.5
    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    return {"H": height_px, "W": width_px, "intrinsics": intrinsics, "near_m": near_m, "far_m": far_m}


# --------------------------------------------------------------------------------------
# The round-trip: points -> depth image -> points
# --------------------------------------------------------------------------------------
def render_batch(clouds, camera_pose, cam_spec, jitter_std_m,
                 inflate_px, clip_mode, occluder_clouds, occluder_inflate_px):
    """Batched points -> RealSense depth image -> world points for B configs at once.

    clouds: (B, N, 3), camera_pose: (B, 7). Returns (list_of_(Mi,3) world clouds, n_px_per_image).
    Rendering all B configs in one launch is what makes this fast -- the z-buffer/backproject cost is
    dominated by the B*H*W pixel work, which the GPU does in parallel instead of one config per call.

    ``inflate_px`` is the MAIN-pass z-buffer dilation (0 = plain round-trip). The anti-penetration work
    is the SECOND pass: ``occluder_clouds`` (the solid door + walls) rasterized with
    ``occluder_inflate_px`` dilation and per-pixel-min composited, so points behind a surface can't leak
    through it ("penetration") and the walls render as solid boxes. Both ``inflate_px`` and
    ``occluder_inflate_px`` are passed in from the cfg (no hardcoded numbers), matching the training
    render (render_depth_roundtrip_from_pose).
    """
    depth, intr = rasterize_depth_zbuffer_from_pose(
        clouds, camera_pose, cam_spec, inflate_px=inflate_px, clip_mode=clip_mode,
        occluder_pcd=occluder_clouds, occluder_inflate_px=occluder_inflate_px,
    )
    if jitter_std_m > 0.0:
        finite = torch.isfinite(depth)
        depth = torch.where(finite, depth + torch.randn_like(depth) * jitter_std_m, depth)
    world, valid = backproject_depth_to_world_from_pose(depth, camera_pose, intr)  # (B,H,W,3), (B,H,W)
    n_px = int(valid.shape[1] * valid.shape[2])
    out = []
    for b in range(world.shape[0]):
        pts = world[b][valid[b]]
        out.append(pts[torch.isfinite(pts).all(dim=-1)])
    return out, n_px


def downsample(points, max_points):
    if points is None or points.shape[0] <= max_points:
        return points
    return points[torch.randperm(points.shape[0], device=points.device)[:max_points]]


def drop_invalid_rows(points):
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite]
    return points[(points.abs().sum(dim=-1) > 1e-9)]


def pack(points, max_points):
    if points is None or points.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float16)
    return downsample(points, max_points).detach().cpu().to(torch.float16)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student-cfg", type=Path, default=DEFAULT_STUDENT_CFG, help="pcd_transformer_dagger_cfg.yaml path.")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "depth_roundtrip_demo.pt")
    p.add_argument("--num-configs", type=int, default=128, help="Wall-distractor configs (= viser frames).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--door", type=Path, default=DEFAULT_DOOR_URDF, help="Door URDF for the GT mesh + panel bbox.")
    p.add_argument("--door-yaw-deg", type=float, default=None,
                   help="Yaw (deg about world z) applied to the door. Default AUTO: pick -90 or +90 so the "
                   "HANDLE faces the robot/camera (world -Y), exactly like render_wall_configs_viser. Pass a "
                   "value to override.")
    p.add_argument("--board-num-points", type=int, default=None, help="GT door surface points (default: scene_door_num_points).")
    p.add_argument("--gt-scale", type=float, default=1.0,
                   help="Scale the GT input density: multiplies the door, wall AND robot point counts. "
                   "1.0 = full training density (walls stay sealed, no penetration). Lower values let "
                   "you see how the round-trip degrades with fewer points, BUT below ~0.5 the walls get "
                   "too sparse for the occluder pass to seal (gaps exceed occluder_inflate_px) and rays "
                   "start punching through -- raise --occluder-inflate-px to compensate if you do this.")
    p.add_argument("--no-walls", action="store_true", help="Door only, skip the wall distractors.")
    p.add_argument("--robot", action="store_true",
                   help="Place the glorbot robot in front of the door (base +90deg yaw, facing +Y, camera "
                   "on its right) and use its own x5_camera_link (with the -45deg mount offset) as the "
                   "camera, so the robot is IN the ray cast and self-occludes -- exactly like "
                   f"render_wall_configs_viser --robot / training. Lateral offset ROBOT_RIGHT_M={ROBOT_RIGHT_M} m.")
    # Camera placement (virtual front look-at, like render_wall_configs_viser's non-robot branch).
    p.add_argument("--standoff", type=float, default=1.0, help="Robot/camera distance from the door along -Y (m).")
    p.add_argument("--camera-height", type=float, default=1.0)
    p.add_argument("--camera-look-z", type=float, default=1.0, help="World z the camera aims at on the panel.")
    p.add_argument("--camera-right", type=float, default=0.12, help="Lateral camera offset to the robot's RIGHT (world +X).")
    p.add_argument("--cam-width-px", type=int, default=320)
    p.add_argument("--cam-height-px", type=int, default=240)
    p.add_argument("--near-m", type=float, default=0.3)
    p.add_argument("--far-m", type=float, default=3.0)
    p.add_argument("--inflate-px", type=int, default=None,
                   help="Main-pass z-buffer dilation (0 = plain round-trip). Default: read from the cfg's "
                   "dagger.depth_cam_render.inflate_px (locked to training).")
    p.add_argument("--occluder-inflate-px", type=int, default=None,
                   help="Second occluder pass dilation on the solid door+walls surfaces, per-pixel-min "
                   "composited to stop back-surface points leaking through gaps (0 disables). Default: "
                   "read from the cfg's dagger.depth_cam_render.occluder_inflate_px (locked to training).")
    p.add_argument("--jitter-std-m", type=float, default=0.0, help="Optional gaussian range noise on the depth (m).")
    p.add_argument("--batch-size", type=int, default=32, help="Configs rendered per GPU launch (higher = faster, more VRAM).")
    p.add_argument("--max-points", type=int, default=8000, help="Per-cloud point cap for viser display.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = yaml.safe_load(args.student_cfg.read_text()) or {}
    wall_cfg = dict(cfg.get("dagger", {}).get("wall_distractors", {}))
    depth_cfg = dict(cfg.get("dagger", {}).get("depth_cam_render", {}))
    scene_door_num_points = int(cfg.get("scene_door_num_points", cfg.get("door_pcd_num_points", 30000)))
    scene_robot_num_points = int(cfg.get("dagger", {}).get("scene_robot_num_points", 30000))
    # GT density: scale down BOTH door + wall points (see --gt-scale) to see how the round-trip degrades.
    board_num_points = max(1, int((args.board_num_points or scene_door_num_points) * args.gt_scale))
    # inflate_px + occluder_inflate_px default to the SAME cfg block training reads
    # (dagger.depth_cam_render), so this tool and multi_pcd_dagger's round-trip render stay in sync.
    # No hardcoded numbers -- CLI overrides win.
    inflate_px = args.inflate_px if args.inflate_px is not None else int(depth_cfg.get("inflate_px", 0))
    occluder_inflate_px = (
        args.occluder_inflate_px if args.occluder_inflate_px is not None
        else int(depth_cfg.get("occluder_inflate_px", 0))
    )
    clip_mode = str(depth_cfg.get("clip_mode", "post"))

    # --- GT door geometry (door-base frame at world origin) ---
    board_bbox, board_gt, handle_center = load_door_asset(args.door, board_num_points, device)

    # Orient the door so the HANDLE faces the robot/camera (world -Y). Default AUTO-picks -90 or +90
    # exactly like render_wall_configs_viser (mine used to hardcode -90, which faced the handle AWAY and
    # put the camera on the back of the door). Walls are sampled in the door-base frame and rotated the
    # same way, so they stay glued to the door.
    if args.door_yaw_deg is not None:
        door_yaw = math.radians(args.door_yaw_deg)
    else:
        def _handle_world_y(theta_deg):
            th = math.radians(theta_deg)
            return handle_center[0] * math.sin(th) + handle_center[1] * math.cos(th)  # world y after Rz

        door_yaw = math.radians(-90.0 if _handle_world_y(-90.0) <= _handle_world_y(90.0) else 90.0)
    cos_y, sin_y = math.cos(door_yaw), math.sin(door_yaw)
    R_door = torch.tensor([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    def door_to_world(pts):  # (1, N, 3) door-base frame -> world
        return pts @ R_door.T

    board_gt_world = door_to_world(board_gt)

    # --- Wall distractor sampler (same code the training pipeline uses) ---
    wall_params = WallDistractorParams.from_cfg(wall_cfg, scene_door_num_points)
    # Match --gt-scale: scale the wall cap AND the per-m^2 density so wall points thin out uniformly.
    wall_params.num_points = max(1, int(wall_params.num_points * args.gt_scale))
    if wall_params.point_density_per_m2 is not None:
        wall_params.point_density_per_m2 = wall_params.point_density_per_m2 * args.gt_scale

    def sample_walls():
        axis_order, bbox_min_ordered, bbox_max_ordered = compute_wall_bbox_ordering(board_bbox)
        return sample_wall_points_local(
            axis_order=axis_order,
            bbox_min_ordered=bbox_min_ordered,
            bbox_max_ordered=bbox_max_ordered,
            num_points=wall_params.num_points,
            params=wall_params,
            device=device,
        )

    # --- Robot (optional): stand the glorbot in front of the door and make its x5_camera_link the
    # camera, so the robot is part of the rasterized cloud and self-occludes (matches render_wall_configs).
    # Base at (0, -standoff, 0), +90deg yaw so base +x -> world +Y (robot faces the door). ---
    robot_world = None
    if args.robot:
        # Scale the robot too, so --gt-scale thins the WHOLE input cloud. The robot is not in the
        # occluder pass, so its surface is where reduced density actually shows up as holes.
        robot_num_points = max(1, int(scene_robot_num_points * args.gt_scale))
        robot_pts_base, cam_T_base = load_robot_asset(robot_num_points, device)  # base_link frame
        base_pos = np.array([ROBOT_RIGHT_M, -args.standoff, 0.0], dtype=np.float64)
        base_R = _quat_wxyz_to_matrix(yaw_quat_wxyz(math.pi / 2.0))
        base_R_t = torch.tensor(base_R, dtype=torch.float32, device=device)
        robot_world = (
            robot_pts_base @ base_R_t.T + torch.tensor(base_pos, dtype=torch.float32, device=device)
        ).unsqueeze(0)  # (1, M, 3) world
        cam_np = robot_camera_pose_world(cam_T_base, base_pos, base_R)
        camera_pose = torch.from_numpy(cam_np).to(device).unsqueeze(0)
        cam_desc = f"x5_camera_link @ {np.round(cam_np[:3], 3).tolist()} (mount -45deg roll)"
    else:
        # Virtual front look-at, shifted to the robot's right (matches render_wall_configs non-robot).
        eye = np.array([args.camera_right, -args.standoff, args.camera_height], dtype=np.float32)
        target = np.array([0.0, 0.0, args.camera_look_z], dtype=np.float32)
        camera_pose = torch.from_numpy(look_at_camera_pose(eye, target)).to(device).unsqueeze(0)
        cam_desc = f"virtual look-at, eye {eye.tolist()} -> {target.tolist()}"

    cam_spec = build_camera_spec(args.cam_width_px, args.cam_height_px, args.near_m, args.far_m, device)

    walls_on = (not args.no_walls) and wall_params.enabled and wall_params.num_points > 0
    print(f"[INFO] device        : {device}")
    print(f"[INFO] door           : {args.door.parent.name}  ({board_gt_world.shape[1]} pts, yaw {round(math.degrees(door_yaw))}deg -> handle faces robot)")
    print(f"[INFO] walls          : {'on (' + str(wall_params.num_points) + ' pts)' if walls_on else 'off'}")
    print(f"[INFO] robot          : {'on (' + str(robot_world.shape[1]) + ' pts, in ray cast)' if robot_world is not None else 'off'}")
    print(f"[INFO] camera         : {args.cam_width_px}x{args.cam_height_px}px RealSense, "
          f"range [{args.near_m}, {args.far_m}] m, {cam_desc}")
    print(f"[INFO] gt_scale       : {args.gt_scale}  (door {board_num_points} pts, walls {wall_params.num_points} pts cap)")
    print(f"[INFO] inflate_px     : {inflate_px}  occluder_inflate_px: {occluder_inflate_px}  (from cfg depth_cam_render)")
    print(f"[INFO] round-trip: GT points -> depth image -> points, {args.num_configs} configs")

    frames = [None] * args.num_configs
    for start in range(0, args.num_configs, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, args.num_configs)))
        # Stack this chunk of configs into one (Bc, N, 3) batch: shared door (+ robot) with per-config
        # walls. The occluder cloud (door + walls, NO robot) drives the anti-penetration second pass;
        # the robot stays out of it so its thin links aren't dilated away (matches training).
        clouds, occluders = [], []
        for _ in idxs:
            occ_parts = [board_gt_world]
            if walls_on:
                occ_parts.append(door_to_world(sample_walls()))
            occ = torch.cat(occ_parts, dim=1) if len(occ_parts) > 1 else occ_parts[0]
            occluders.append(occ)
            clouds.append(torch.cat([occ, robot_world], dim=1) if robot_world is not None else occ)
        clouds = torch.cat(clouds, dim=0)  # (Bc, N, 3)
        occluders = torch.cat(occluders, dim=0)  # (Bc, No, 3)
        cam_b = camera_pose.expand(len(idxs), -1)

        rep_list, n_px = render_batch(
            clouds, cam_b, cam_spec, args.jitter_std_m,
            inflate_px=inflate_px, clip_mode=clip_mode,
            occluder_clouds=occluders, occluder_inflate_px=occluder_inflate_px,
        )

        for j, config_idx in enumerate(idxs):
            frames[config_idx] = {
                "pointclouds": {
                    "ground_truth": pack(drop_invalid_rows(clouds[j]), args.max_points),
                    "reprojected": pack(rep_list[j], args.max_points),
                }
            }
        last = idxs[-1]
        print(f"  configs {idxs[0] + 1}-{last + 1}/{args.num_configs} (batch {len(idxs)}): "
              f"valid px≈{rep_list[-1].shape[0]}/{n_px}")

    payload = {
        "format": "dooropening_viser_replay_v1",
        "pointcloud_frame": "world",
        "pointcloud_source": "depth",
        "pointcloud_streams": [
            {"name": "ground_truth", "label": "GT (door + walls [+ robot])", "color": (120, 120, 120), "point_size_scale": 1.0},
            {"name": "reprojected", "label": "Depth round-trip", "color": (79, 195, 247), "point_size_scale": 1.4},
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
