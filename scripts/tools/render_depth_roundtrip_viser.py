#!/usr/bin/env python3
"""DEX-style RealSense mock render -> point cloud replay in Viser.

This is a *standalone* tool (no Isaac Sim). The input point cloud is the SAME "ground_truth" geometry
that `render_wall_configs_viser.py` builds: real door URDF surface points plus wall distractors,
oriented so the door face points at the camera. Door, walls, and robot are concatenated and
rasterized once into a single RealSense depth image, then back-projected. This cheaper single-pass
mock does not use a second dilated occluder pass to fill sampling gaps. Only camera-visible returns
survive.

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
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import yaml

from DoorOpening.utils.camera_utils import (
    _dilate_depth_min_pool,
    apply_depth_spatial_blur,
    backproject_depth_to_world_from_pose,
    build_depth_blur_kernel2d,
    build_realsense_sampler_spec,
    crop_local_pcd,
    drop_depth_edges,
    rasterize_depth_zbuffer_from_pose,
)
from DoorOpening.utils.door_window_dropout import (
    apply_window_dropout_to_door_points,
    reflect_robot_points_in_window,
    sample_random_window_hole_metadata,
)
from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler
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
# Slightly adjusted Franka pose used by the approved local mock preview.
FRANKA_READY_JOINT_POS = [0.05, -0.70, 0.05, -2.20, 0.05, 1.48, 0.10]
# Robot base lateral offset along world X. The camera moves with the base. Constant on purpose
# (not a per-run config); this is a mock-scene pose only, not a training reset pose.
ROBOT_RIGHT_M = 0.04


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
    """Real door in the door-base frame, split to match multi_pcd_dagger's door point cloud exactly.

    Returns ``(bbox, panel_handle_pts, frame_pts, handle_center)``:
      * ``panel_handle_pts`` -- link_1 (panel) + link_2 (handle), the geometry training ALWAYS renders.
      * ``frame_pts`` -- link_0 (casing/jamb), which training renders only when ``door_frame_aug`` is on
        and the per-env coin flip includes it. Kept separate so this preview can toggle it the same way.

    Sourced from the SAME FrankaGripperSampler cache the training/probe pipelines use (points sampled in
    each link's frame, then FK'd into the door base frame at the closed-door config), so the preview
    cloud is faithful to what the renderer actually ingests -- not a whole-mesh dump that always bakes
    the frame in.
    """
    sampler = FrankaGripperSampler(str(urdf_path), device=device, num_points=int(num_points))
    zero_joints = torch.zeros((1, len(sampler.robot.actuated_joint_names)), device=device, dtype=torch.float32)
    panel_handle_pts = sampler.sample_link_set(zero_joints, ["link_1", "link_2"])  # (1, N, 3) base frame
    frame_local = sampler.points.get("link_0")
    if frame_local is not None and frame_local.shape[1] > 0:
        frame_pts = sampler.sample_link_set(zero_joints, ["link_0"])  # (1, Nf, 3) base frame
    else:
        frame_pts = torch.zeros((1, 0, 3), dtype=torch.float32, device=device)
    kp = compute_exact_door_keypoints(str(urdf_path))
    # BOX walls use the FULL door outer bbox (frame + panel + handle), EXACTLY like training
    # (multi_door_cfg.door_full_bboxes -> multi_pcd_dagger.env_full_door_bboxes), so they sit outside
    # the whole door. The FLUSH slab uses the PANEL (link_1) bbox so it stays coplanar with the panel
    # face -- again matching training (multi_pcd_dagger.wall_distractor_panel_bbox_*). Same fallbacks.
    full_bbox = kp.get("door_full_bbox_base", kp["link_1_bbox_base"])
    bbox = torch.as_tensor(full_bbox, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 2, 3)
    panel_bbox = torch.as_tensor(kp["link_1_bbox_base"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 2, 3)
    # Panel bbox + link_1 pose in the link_1 LOCAL frame -- what the window-hole aug samples in
    # (multi_pcd_dagger.env_board_bboxes_link1 / _get_link1_pose_world). Falls back to the base bbox
    # for doors with no meta (link_1 frame ~ base frame for those).
    panel_bbox_link1 = torch.as_tensor(
        kp.get("link_1_bbox_link1", kp["link_1_bbox_base"]), dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 2, 3)
    link1_pose_base = torch.as_tensor(
        kp.get("link_1_pose_base", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32, device=device
    )  # (7,) [pos, quat_wxyz]
    handle_center = np.asarray(kp.get("link_2_center_base", [0.0, 0.0, 0.0]), dtype=np.float64)  # base frame
    return bbox, panel_bbox, panel_bbox_link1, link1_pose_base, panel_handle_pts, frame_pts, handle_center


def load_robot_asset(num_points, device, franka_q=None):
    """Source-equivalent Glorbot cached-link cloud + x5 camera transform.

    Dagger builds this same ``FrankaGripperSampler`` cache once and composes its
    link-local points from live Isaac body poses.  Offline we use the equivalent
    FK for a fixed nominal pose, rather than a separately sampled trimesh surface.
    """
    robot = _load_urdf(GLORBOT_URDF, {"glorbot": GLORBOT_DIR})
    names = list(robot.actuated_joint_names)
    cfg = np.zeros(len(names), dtype=np.float64)
    # Keep the offline camera pose identical to IsaacLab.  The environment holds the
    # non-policy x5 camera arm at these defaults; leaving them at zero points the
    # synthetic camera away from the door even when the robot base pose is correct.
    from DoorOpening.constants.robot_constants import CAMERA_JOINT_DEFAULT_VALUES

    for name, value in CAMERA_JOINT_DEFAULT_VALUES.items():
        if name in names:
            cfg[names.index(name)] = float(value)
    if franka_q is None:
        franka_q = FRANKA_READY_JOINT_POS
    for i, value in enumerate(franka_q):
        jn = f"panda_joint{i + 1}"
        if jn in names:
            cfg[names.index(jn)] = value
    robot.update_cfg(cfg)
    sampler = FrankaGripperSampler(str(GLORBOT_URDF), device=device, num_points=int(num_points))
    sampler_names = list(sampler.robot.actuated_joint_names)
    sampler_cfg = torch.zeros((1, len(sampler_names)), dtype=torch.float32, device=device)
    for index, name in enumerate(sampler_names):
        if name in names:
            sampler_cfg[0, index] = float(cfg[names.index(name)])
    # Same cached point distribution / FK path as Dagger's
    # compose_cached_link_pointcloud_world, before the base pose is applied.
    pts = sampler.sample(sampler_cfg)[0]
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
# Point-cloud utilities
# --------------------------------------------------------------------------------------
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


def pad_cloud_batch(clouds):
    """Stack variable-length ``(1, N, 3)`` clouds using NaN invalid slots.

    The per-config door-frame coin flip, window hole and reflection veil deliberately
    change the number of valid scene points.  The depth rasterizer already treats
    non-finite points as absent, so NaN padding preserves every sampled scene while
    allowing the offline Viser probe to render several configs in one batch.
    """
    if not clouds:
        raise ValueError("pad_cloud_batch needs at least one cloud")
    max_points = max(int(cloud.shape[1]) for cloud in clouds)
    batch = torch.full(
        (len(clouds), max_points, 3), float("nan"), dtype=clouds[0].dtype, device=clouds[0].device
    )
    for index, cloud in enumerate(clouds):
        batch[index, : cloud.shape[1]] = cloud[0]
    return batch


def _axial_jitter(depth, std_m):
    if std_m is None or float(std_m) <= 0.0:
        return depth
    return torch.where(torch.isfinite(depth), depth + torch.randn_like(depth) * float(std_m), depth)


def render_dex_style_depth(scene_cloud, robot_cloud, camera_pose, camera_spec, depth_cfg,
                           depth_filter="min_pool", blur_sigma_px=1.5):
    """Rasterize door, walls, and robot together once, then back-project."""
    inflate = int(depth_cfg.get("inflate_px", 0))
    clip_mode = str(depth_cfg.get("clip_mode", "post"))
    render_cloud = torch.cat((scene_cloud, robot_cloud), dim=1) if robot_cloud is not None else scene_cloud
    # All geometry competes in one z-buffer. There is no separate robot pass or
    # second, dilated occluder pass in this low-cost rendering mode.
    depth, intr = rasterize_depth_zbuffer_from_pose(
        render_cloud, camera_pose, camera_spec, inflate_px=inflate, clip_mode=clip_mode,
        occluder_pcd=None, occluder_inflate_px=0,
    )
    blur_kernel = int(depth_cfg.get("blur_kernel_px", 0))
    if blur_kernel > 1:
        if depth_filter == "gaussian":
            kernel, pad = build_depth_blur_kernel2d(
                blur_kernel, float(blur_sigma_px), depth.device, depth.dtype
            )
            depth = apply_depth_spatial_blur(depth, kernel, pad)
        elif depth_filter == "min_pool":
            depth = _dilate_depth_min_pool(depth, blur_kernel // 2)
        else:
            raise ValueError(f"Unsupported depth filter: {depth_filter}")
    depth = drop_depth_edges(depth, float(depth_cfg.get("edge_drop_m", 0.0)))
    depth = _axial_jitter(depth, depth_cfg.get("axial_jitter_std_m", 0.0))
    world, valid = backproject_depth_to_world_from_pose(depth, camera_pose, intr)
    return [world[b][valid[b] & torch.isfinite(world[b]).all(dim=-1)] for b in range(world.shape[0])], int(valid.shape[-2] * valid.shape[-1])


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
    p.add_argument("--glass-reflection", action="store_true",
                   help="Force the glass-door reflection veil ON regardless of the cfg (needs the window "
                   "hole; adds a sparse noise cloud over the opening). Uses door_hole_aug.glass_reflection knobs.")
    p.add_argument("--board-num-points", type=int, default=None, help="GT door surface points (default: scene_door_num_points).")
    p.add_argument("--gt-scale", type=float, default=1.0,
                   help="Scale the GT input density: multiplies the door, wall AND robot point counts.")
    p.add_argument("--no-walls", action="store_true", help="Door only, skip the wall distractors.")
    robot_group = p.add_mutually_exclusive_group()
    robot_group.add_argument("--robot", dest="robot", action="store_true",
                             help="Include the robot in the combined depth ray cast (the default).")
    robot_group.add_argument("--no-robot", dest="robot", action="store_false",
                             help="Render only the door/walls with a virtual camera.")
    p.set_defaults(robot=True)
    p.add_argument("--robot-lateral-offset", type=float, default=ROBOT_RIGHT_M,
                   help="Shift the robot base along world +X relative to the door (m).")
    p.add_argument("--franka-joints", type=float, nargs=7, default=None,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
                   help="Override the seven Panda arm joint angles (rad); default is FRANKA_READY_JOINT_POS.")
    # Camera placement (virtual front look-at, like render_wall_configs_viser's non-robot branch).
    p.add_argument("--standoff", type=float, default=1.0, help="Robot/camera distance from the door along -Y (m).")
    p.add_argument("--camera-height", type=float, default=1.0)
    p.add_argument("--camera-look-z", type=float, default=1.0, help="World z the camera aims at on the panel.")
    p.add_argument("--camera-right", type=float, default=0.12, help="Lateral camera offset to the robot's RIGHT (world +X).")
    p.add_argument("--cam-width-px", type=int, default=None,
                   help="Render width (px). Default: dagger.depth_cam_render.width_px in the student cfg (320 if absent).")
    p.add_argument("--cam-height-px", type=int, default=None,
                   help="Render height (px). Default: dagger.depth_cam_render.height_px in the student cfg (240 if absent).")
    p.add_argument("--near-m", type=float, default=0.3)
    p.add_argument("--far-m", type=float, default=3.0)
    p.add_argument("--inflate-px", type=int, default=None,
                   help="Main-pass z-buffer dilation (0 = plain round-trip). Default: read from the cfg's "
                   "dagger.depth_cam_render.inflate_px (locked to training).")
    p.add_argument("--jitter-std-m", type=float, default=None,
                   help="Gaussian axial range noise (m). Default: dagger.depth_cam_render.axial_jitter_std_m.")
    p.add_argument("--blur-kernel-px", type=int, default=None,
                   help="Depth-filter window/kernel size (px). Default: read from the cfg's "
                   "dagger.depth_cam_render.blur_kernel_px. <=1 disables.")
    p.add_argument("--depth-filter", choices=("auto", "min_pool", "gaussian"), default="auto",
                   help="Depth-image filter. 'auto' uses Gaussian when blur_sigma_px exists in the cfg "
                   "(as in older saved configs), otherwise nearest-return min pooling.")
    p.add_argument("--blur-sigma-px", type=float, default=None,
                   help="Gaussian sigma in pixels. Defaults to cfg blur_sigma_px, or 1.5 when explicitly "
                   "selecting Gaussian without a sigma in the cfg.")
    p.add_argument("--edge-drop-m", type=float, default=None,
                   help="Drop SCENE pixels on a depth discontinuity > this (m) to remove the blur's "
                   "flying-pixel smears on wall/door edges. Default: cfg dagger.depth_cam_render.edge_drop_m.")
    p.add_argument("--batch-size", type=int, default=32, help="Configs rendered per GPU launch (higher = faster, more VRAM).")
    p.add_argument("--max-points", type=int, default=8000, help="Per-cloud point cap for viser display.")
    p.add_argument("--compare-policy-prefilter", action="store_true",
                   help="Also render only source points inside the policy's base-frame crop volume; saves both results and times both paths.")
    p.add_argument("--reflection-demo", action="store_true",
                   help="Create a compact Viser replay sweeping robot arm poses and door hinge angles through the live reflection implementation.")
    return p.parse_args()


def run_reflection_demo(args, cfg, device):
    """Build a small replay that directly validates live robot reflections.

    Each frame uses a fresh robot FK pose and a different door hinge angle. The reflected stream is
    generated by reflect_robot_points_in_window, not by the legacy random blob generator.
    """
    from scipy.spatial.transform import Rotation as _R

    # Keep the same scene composition and renderer as training, with a smaller offline point budget
    # so the multi-state Viser replay remains practical on CPU.
    door_sampler = FrankaGripperSampler(str(args.door), device=device, num_points=4000)
    door_names = list(door_sampler.robot.actuated_joint_names)
    door_q = torch.zeros((1, len(door_names)), dtype=torch.float32, device=device)
    joint_1_idx = door_names.index("joint_1") if "joint_1" in door_names else None
    joint_2_idx = door_names.index("joint_2") if "joint_2" in door_names else None
    kp = compute_exact_door_keypoints(str(args.door))
    board_bbox_link1 = torch.tensor(kp["link_1_bbox_link1"], dtype=torch.float32, device=device).unsqueeze(0)
    # Sweep the robot's floating base-x position relative to the door. This is the important
    # diagnostic: the reflected cloud should translate with the robot and disappear when it moves
    # outside the window opening.
    base_x_variants = [-0.50, 0.0, 0.50]
    robot_q = FRANKA_READY_JOINT_POS
    hinge_angles = [0.0, 0.70]
    latch_angles = [0.0, 0.35]
    frames = []
    base_pos = np.array([args.robot_lateral_offset, -args.standoff, 0.0], dtype=np.float64)
    base_R = _quat_wxyz_to_matrix(yaw_quat_wxyz(math.pi / 2.0))
    base_R_t = torch.tensor(base_R, dtype=torch.float32, device=device)
    base_pos_t = torch.tensor(base_pos, dtype=torch.float32, device=device)
    # The door asset's closed panel normal points along the door-base x axis. Rotate the door -90deg
    # about world z so its front face points toward the robot at y < 0. This matches the normal
    # handle-facing orientation used by the full preview path below.
    door_yaw = -math.pi / 2.0
    c, s = math.cos(door_yaw), math.sin(door_yaw)
    R_door = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R_door_t = torch.tensor(R_door, dtype=torch.float32, device=device)
    full_bbox = torch.tensor(kp.get("door_full_bbox_base", kp["link_1_bbox_base"]), dtype=torch.float32, device=device).unsqueeze(0)
    wall_cfg = dict(cfg.get("dagger", {}).get("wall_distractors", {}))
    wall_params = WallDistractorParams.from_cfg(
        wall_cfg, int(cfg.get("scene_door_num_points", cfg.get("door_pcd_num_points", 30000)))
    )
    wall_params.num_points = max(1, int(wall_params.num_points * min(float(args.gt_scale), 1.0)))
    wall_axis_order, wall_bbox_min_ordered, wall_bbox_max_ordered = compute_wall_bbox_ordering(full_bbox)
    panel_bbox_base = torch.tensor(kp["link_1_bbox_base"], dtype=torch.float32, device=device).unsqueeze(0)
    panel_min_ordered = torch.gather(panel_bbox_base[:, 0], 1, wall_axis_order)
    panel_max_ordered = torch.gather(panel_bbox_base[:, 1], 1, wall_axis_order)
    hole_cfg = dict(cfg.get("dagger", {}).get("door_hole_aug", {}))
    hole_width_range = tuple(float(v) for v in hole_cfg.get("width_range_m", [0.12, 1.60]))
    hole_height_range = tuple(float(v) for v in hole_cfg.get("height_range_m", [0.18, 2.20]))
    hole_center_height_range = tuple(float(v) for v in hole_cfg.get("center_height_range_m", [0.10, 1.90]))
    hole_side_margin_range = tuple(float(v) for v in hole_cfg.get("side_margin_range_m", [0.0, 0.18]))
    hole_surface_eps = float(hole_cfg.get("surface_eps_m", 0.03))
    robot_pts_base, cam_T_base = load_robot_asset(max(1, int(args.gt_scale * 6000)), device, franka_q=robot_q)
    _, cam_T_base_small = load_robot_asset(100, device, franka_q=robot_q)

    for hinge in hinge_angles:
        for variant_idx, (latch, base_x) in enumerate(
            ( (latch_value, x_value) for latch_value in latch_angles for x_value in base_x_variants )
        ):
            base_pos_variant = np.array([base_x, -args.standoff, 0.0], dtype=np.float64)
            base_pos_variant_t = torch.tensor(base_pos_variant, dtype=torch.float32, device=device)
            q = door_q.clone()
            if joint_1_idx is not None:
                q[0, joint_1_idx] = latch
            if joint_2_idx is not None:
                q[0, joint_2_idx] = hinge
            door_base_pts = door_sampler.sample_link_set(q, ["link_1", "link_2"])
            frame_base_pts = door_sampler.sample_link_set(q, ["link_0"])
            link1_tf = door_sampler.robot.link_fk_batch(q, use_names=True)["link_1"][0].detach().cpu().numpy()
            link1_pos_world_np = R_door @ link1_tf[:3, 3]
            link1_rot_world_np = R_door @ link1_tf[:3, :3]
            link1_pos_base = torch.tensor(link1_pos_world_np, dtype=torch.float32, device=device)
            link1_quat_xyzw = _R.from_matrix(link1_rot_world_np).as_quat()
            link1_quat_base = torch.tensor(
                [link1_quat_xyzw[3], link1_quat_xyzw[0], link1_quat_xyzw[1], link1_quat_xyzw[2]],
                dtype=torch.float32, device=device,
            )
            link1_pose_world = torch.cat([
                link1_pos_base.view(1, 3),
                link1_quat_base.view(1, 4),
            ], dim=-1)
            # Door yaw is zero in this compact diagnostic scene; rotate the robot base only.
            robot_world = (robot_pts_base @ base_R_t.T + base_pos_variant_t).unsqueeze(0)
            link1_z_world = torch.tensor(link1_rot_world_np[:, 2], dtype=torch.float32, device=device)
            front_sign = torch.where(
                torch.dot(base_pos_variant_t - link1_pose_world[0, :3], link1_z_world) >= 0.0,
                torch.ones(1, device=device),
                -torch.ones(1, device=device),
            )
            # Use the same sampled hole generator as training, but force a hole in every demo frame
            # so the actual opening, dropout, and reflected cloud are visible and comparable.
            hole_metadata = sample_random_window_hole_metadata(
                link1_pose_world=link1_pose_world,
                board_bbox_link1=board_bbox_link1,
                window_prob=1.0,
                width_range=hole_width_range,
                height_range=hole_height_range,
                center_height_range=hole_center_height_range,
                side_margin_range=hole_side_margin_range,
            )
            hole_metadata["reflection_enabled"] = torch.ones(1, dtype=torch.bool, device=device)
            reflected = reflect_robot_points_in_window(
                robot_world, link1_pose_world, board_bbox_link1, hole_metadata,
                front_sign, num_points=400, density_range=(1.0, 1.0),
            )
            door_world = door_base_pts @ R_door_t.T
            frame_world = frame_base_pts @ R_door_t.T
            door_world, hole_metadata = apply_window_dropout_to_door_points(
                points_world=door_world,
                link1_pose_world=link1_pose_world,
                board_bbox_link1=board_bbox_link1,
                hole_metadata=hole_metadata,
                surface_eps=hole_surface_eps,
            )
            wall_local = sample_wall_points_local(
                axis_order=wall_axis_order,
                bbox_min_ordered=wall_bbox_min_ordered,
                bbox_max_ordered=wall_bbox_max_ordered,
                num_points=wall_params.num_points,
                params=wall_params,
                device=device,
                flush_bbox_min_ordered=panel_min_ordered,
                flush_bbox_max_ordered=panel_max_ordered,
            )
            wall_world = wall_local @ R_door_t.T
            scene = torch.cat([door_world, frame_world, wall_world, reflected], dim=1)
            cam_np = robot_camera_pose_world(cam_T_base_small, base_pos_variant, base_R)
            camera_pose = torch.from_numpy(cam_np).to(device).unsqueeze(0)
            depth_cfg = dict(cfg.get("dagger", {}).get("depth_cam_render", {}))
            cam_spec = build_realsense_sampler_spec(240, 320, device=device)
            depth_returns, _ = render_dex_style_depth(scene, robot_world, camera_pose, cam_spec, depth_cfg)
            # Render the sampled window boundary at the front face so the opening and the reflected
            # robot can be inspected independently in Viser.
            hole_bbox = hole_metadata["hole_bbox_link1"]
            hole_min_frame, hole_max_frame = hole_bbox[:, :3], hole_bbox[:, 3:]
            bmin, bmax = board_bbox_link1[:, 0], board_bbox_link1[:, 1]
            front_z = torch.where(front_sign[0] > 0, bmax[0, 2], bmin[0, 2])
            window_local = torch.stack([
                torch.stack([hole_min_frame[0, 0], hole_min_frame[0, 1], front_z]),
                torch.stack([hole_max_frame[0, 0], hole_min_frame[0, 1], front_z]),
                torch.stack([hole_max_frame[0, 0], hole_max_frame[0, 1], front_z]),
                torch.stack([hole_min_frame[0, 0], hole_max_frame[0, 1], front_z]),
            ], dim=0)
            window_local = torch.cat([window_local, window_local[:1]], dim=0)
            window_world = (
                window_local @ torch.tensor(link1_rot_world_np, dtype=torch.float32, device=device).T
                + link1_pose_world[0, :3]
            )
            frames.append({
                "pointclouds": {
                    "door": pack(drop_invalid_rows(door_world[0]), args.max_points),
                    "door_frame": pack(drop_invalid_rows(frame_world[0]), args.max_points),
                    "walls": pack(drop_invalid_rows(wall_world[0]), args.max_points),
                    "robot": pack(drop_invalid_rows(robot_world[0]), args.max_points),
                    "reflected_robot": pack(drop_invalid_rows(reflected[0]), args.max_points),
                    "window": pack(window_world, args.max_points),
                    "depth_return": pack(depth_returns[0], args.max_points),
                },
                "label": f"latch={latch:.2f}, hinge={hinge:.2f}, base_x={base_x:+.2f} m",
                "door_joint_pos": [float(latch), float(hinge)],
                "robot_base_pos_w": base_pos_variant.tolist(),
            })
    payload = {
        "format": "dooropening_viser_replay_v1",
        "pointcloud_frame": "world",
        "pointcloud_source": "live_robot_reflection_demo",
        "frame_dt": 1.0,
        "frame_fps": 1.0,
        "pointcloud_streams": [
            {"name": "door", "label": "Door", "color": (180, 180, 180), "point_size_scale": 1.0},
            {"name": "door_frame", "label": "Door frame", "color": (255, 170, 40), "point_size_scale": 1.2},
            {"name": "walls", "label": "Wall distractors", "color": (120, 120, 135), "point_size_scale": 0.9},
            {"name": "robot", "label": "Robot", "color": (255, 140, 0), "point_size_scale": 1.4},
            {"name": "reflected_robot", "label": "Reflected robot", "color": (50, 220, 255), "point_size_scale": 1.5},
            {"name": "window", "label": "Window boundary", "color": (255, 230, 40), "point_size_scale": 3.0},
            {"name": "depth_return", "label": "Depth return", "color": (100, 255, 100), "point_size_scale": 1.2},
        ],
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"[INFO] Saved reflection demo: {len(frames)} frames to {args.output}")
    print(f"[INFO] Swept hinge angles: {hinge_angles}; base_x positions: {base_x_variants}")
    print(f"[INFO] Play with: python {REPO_ROOT / 'scripts' / 'replay_viser_pt.py'} {args.output} --start-paused")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = yaml.safe_load(args.student_cfg.read_text()) or {}
    if args.reflection_demo:
        run_reflection_demo(args, cfg, device)
        return
    wall_cfg = dict(cfg.get("dagger", {}).get("wall_distractors", {}))
    frame_cfg = dict(cfg.get("dagger", {}).get("door_frame_aug", {}))
    hole_cfg = dict(cfg.get("dagger", {}).get("door_hole_aug", {}))
    depth_cfg = dict(cfg.get("dagger", {}).get("depth_cam_render", {}))
    # Use the same camera resolution/noise values as training unless explicitly overridden.
    args.cam_width_px = int(args.cam_width_px or depth_cfg.get("width_px", 320))
    args.cam_height_px = int(args.cam_height_px or depth_cfg.get("height_px", 240))
    axial_jitter_std_m = (
        float(args.jitter_std_m)
        if args.jitter_std_m is not None
        else float(depth_cfg.get("axial_jitter_std_m", 0.0))
    )
    depth_cfg["axial_jitter_std_m"] = axial_jitter_std_m
    # Policy-input crop knobs, read from the SAME student cfg multi_pcd_dagger uses (_build_local_pcd ->
    # crop_local_pcd base cylindrical crop). Height bounds [0.55, 1.5] are crop_local_pcd's own defaults.
    student_cfg = dict(cfg.get("student", {}))
    policy_crop_range_cfg = student_cfg.get("local_pcd_range", cfg.get("local_pcd_range", [1.0, 0.35, 0.35]))
    policy_crop_range = float(policy_crop_range_cfg[0])
    policy_x_cutoff = float(student_cfg.get("x_direction_cutoff", cfg.get("x_direction_cutoff", -0.5)))
    local_pcd_cfg = dict(cfg.get("pcd_encoders_cfg", {}).get("local_pcd_t", {}))
    local_point_counts = list(local_pcd_cfg.get("num_points", [12000, 0, 0]))
    policy_base_points = int(local_point_counts[0]) if local_point_counts else 12000
    scene_door_num_points = int(cfg.get("scene_door_num_points", cfg.get("door_pcd_num_points", 30000)))
    scene_robot_num_points = int(cfg.get("dagger", {}).get("scene_robot_num_points", 30000))
    # GT density: scale down BOTH door + wall points (see --gt-scale) to see how the round-trip degrades.
    board_num_points = max(1, int((args.board_num_points or scene_door_num_points) * args.gt_scale))
    # Main z-buffer dilation defaults to the cfg block training reads. This mock
    # intentionally skips the second dilated occluder pass for a single rasterization.
    inflate_px = args.inflate_px if args.inflate_px is not None else int(depth_cfg.get("inflate_px", 0))
    clip_mode = str(depth_cfg.get("clip_mode", "post"))
    # Depth blur defaults to the same cfg block training reads; CLI overrides win.
    blur_kernel_px = args.blur_kernel_px if args.blur_kernel_px is not None else int(depth_cfg.get("blur_kernel_px", 0))
    depth_filter = args.depth_filter
    if depth_filter == "auto":
        depth_filter = "gaussian" if "blur_sigma_px" in depth_cfg else "min_pool"
    blur_sigma_px = (
        float(args.blur_sigma_px)
        if args.blur_sigma_px is not None
        else float(depth_cfg.get("blur_sigma_px", 1.5))
    )
    edge_drop_m = args.edge_drop_m if args.edge_drop_m is not None else float(depth_cfg.get("edge_drop_m", 0.0))
    # The source-aware mock renderer reads these resolved values from the depth cfg.
    # Propagate the already-supported CLI overrides so replay experiments really apply.
    depth_cfg["blur_kernel_px"] = blur_kernel_px
    depth_cfg["blur_sigma_px"] = blur_sigma_px
    depth_cfg["edge_drop_m"] = edge_drop_m

    # --- GT door geometry (door-base frame at world origin) ---
    board_bbox, panel_bbox, panel_bbox_link1, link1_pose_base, board_gt, frame_gt, handle_center = load_door_asset(
        args.door, board_num_points, device
    )

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
    frame_gt_world = door_to_world(frame_gt)

    # link_1 (panel) pose in WORLD = door-yaw rotation (origin, no translation) composed with the
    # static link_1-in-base pose. The window-hole aug transforms world door points into this frame.
    from scipy.spatial.transform import Rotation as _R

    link1_pos_base_np = link1_pose_base[:3].cpu().numpy()
    link1_quat_base_wxyz = link1_pose_base[3:].cpu().numpy()
    R_base_link1 = _R.from_quat(
        [link1_quat_base_wxyz[1], link1_quat_base_wxyz[2], link1_quat_base_wxyz[3], link1_quat_base_wxyz[0]]
    )
    R_door_np = R_door.cpu().numpy()
    R_world_link1_np = R_door_np @ R_base_link1.as_matrix()
    link1_pos_world_np = R_door_np @ link1_pos_base_np
    link1_quat_world_xyzw = _R.from_matrix(R_world_link1_np).as_quat()
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
    # link_1 LOCAL -> world transform for the reflection veil (points_link1 @ R.T + pos).
    R_world_link1_t = torch.tensor(R_world_link1_np, dtype=torch.float32, device=device)
    link1_pos_world_t = torch.tensor(link1_pos_world_np, dtype=torch.float32, device=device)

    # --- Door-frame (link_0) aug: same knobs multi_pcd_dagger reads. When on, the frame is included
    # in a per-config fraction (env_prob) of frames, exactly like the per-env episode coin flip. ---
    frame_aug_enabled = bool(frame_cfg.get("enabled", False)) and frame_gt_world.shape[1] > 0
    frame_env_prob = float(frame_cfg.get("env_prob", 0.5))

    # --- Window-hole aug: same knobs multi_pcd_dagger reads. One hole is drawn PER CONFIG (= per
    # rollout) and applied to that config's panel+handle cloud, matching the per-rollout training
    # behaviour (the hole no longer jitters step-to-step). ---
    hole_aug_enabled = bool(hole_cfg.get("enabled", False))
    hole_env_prob = float(hole_cfg.get("env_prob", 0.35))
    hole_width_range = tuple(float(v) for v in hole_cfg.get("width_range_m", [0.12, 1.60]))
    hole_height_range = tuple(float(v) for v in hole_cfg.get("height_range_m", [0.18, 2.20]))
    hole_center_height_range = tuple(float(v) for v in hole_cfg.get("center_height_range_m", [0.10, 1.90]))
    hole_side_margin_range = tuple(float(v) for v in hole_cfg.get("side_margin_range_m", [0.0, 0.18]))
    hole_surface_eps = float(hole_cfg.get("surface_eps_m", 0.03))

    # --- Glass-door reflection: on a fraction of the hole configs, reflect the current robot cloud
    # across the window plane and clip it to the opening. ---
    reflection_cfg = dict(hole_cfg.get("glass_reflection", {}))
    reflection_enabled = hole_aug_enabled and (args.glass_reflection or bool(reflection_cfg.get("enabled", False)))
    reflection_prob = float(reflection_cfg.get("prob", 0.5))
    reflection_num_points = int(reflection_cfg.get("num_points", 400))
    reflection_density_range = tuple(float(v) for v in reflection_cfg.get("density_range", [1.0, 1.0]))
    # Which link_1 thickness (z) direction faces the camera, so the veil goes BEHIND the glass. Filled in
    # once camera_pose is known (below); read late by make_holed_door at render time.
    reflection_front_sign = None

    def make_holed_door(panel_handle_world):
        """One window hole per config: drop panel+handle points inside it (NaN, fixed N) and, when glass
        reflection is on, also return a sparse world veil of reflection points over the SAME window.
        Returns (holed (1, N, 3), reflection_world (1, P, 3) or None)."""
        if not hole_aug_enabled:
            return panel_handle_world, None
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
            points_world=panel_handle_world[0],
            link1_pose_world=link1_pose_world,
            board_bbox_link1=panel_bbox_link1[0],
            hole_metadata=metadata,
            surface_eps=hole_surface_eps,
        )
        holed = dropped.unsqueeze(0)
        reflection_world = None
        if reflection_enabled and reflection_num_points > 0 and robot_world is not None:
            metadata["reflection_enabled"] = metadata["enabled"] & (
                torch.rand((), device=device) < reflection_prob
            )
            reflection_world = reflect_robot_points_in_window(
                robot_points_world=robot_world,
                link1_pose_world=link1_pose_world,
                board_bbox_link1=panel_bbox_link1[0],
                hole_metadata=metadata,
                front_sign=reflection_front_sign,
                num_points=reflection_num_points,
                density_range=reflection_density_range,
            )
        return holed, reflection_world

    # --- Wall distractor sampler (same code the training pipeline uses) ---
    wall_params = WallDistractorParams.from_cfg(wall_cfg, scene_door_num_points)
    # Match --gt-scale: scale the wall cap AND the per-m^2 density so wall points thin out uniformly.
    wall_params.num_points = max(1, int(wall_params.num_points * args.gt_scale))
    if wall_params.point_density_per_m2 is not None:
        wall_params.point_density_per_m2 = wall_params.point_density_per_m2 * args.gt_scale
    axis_order, bbox_min_ordered, bbox_max_ordered = compute_wall_bbox_ordering(board_bbox)
    # Flush slab is driven by the PANEL bbox, reordered by the SAME axis order (matches training).
    panel_bbox_min_ordered = torch.gather(panel_bbox[:, 0], 1, axis_order)
    panel_bbox_max_ordered = torch.gather(panel_bbox[:, 1], 1, axis_order)

    def sample_walls():
        return sample_wall_points_local(
            axis_order=axis_order,
            bbox_min_ordered=bbox_min_ordered,
            bbox_max_ordered=bbox_max_ordered,
            num_points=wall_params.num_points,
            params=wall_params,
            device=device,
            flush_bbox_min_ordered=panel_bbox_min_ordered,
            flush_bbox_max_ordered=panel_bbox_max_ordered,
        )

    # --- Robot (optional): stand the glorbot in front of the door and make its x5_camera_link the
    # camera, so the robot is part of the rasterized cloud and self-occludes (matches render_wall_configs).
    # Base at (0, -standoff, 0), +90deg yaw so base +x -> world +Y (robot faces the door). ---
    # Robot base pose in world (base at (ROBOT_RIGHT_M, -standoff, 0), +90deg yaw so base +x -> world +Y).
    # Computed unconditionally: it also defines the frame for the policy-input crop below, so that crop
    # matches multi_pcd_dagger (which crops in the robot base frame) whether or not the robot is drawn.
    base_pos = np.array([args.robot_lateral_offset, -args.standoff, 0.0], dtype=np.float64)
    base_R = _quat_wxyz_to_matrix(yaw_quat_wxyz(math.pi / 2.0))
    base_R_t = torch.tensor(base_R, dtype=torch.float32, device=device)
    base_pos_t = torch.tensor(base_pos, dtype=torch.float32, device=device)

    robot_world = None
    if args.robot:
        # Scale the robot too, so --gt-scale thins the WHOLE input cloud. It joins the
        # scene before the single z-buffer pass, so its sampling density affects visibility.
        robot_num_points = max(1, int(scene_robot_num_points * args.gt_scale))
        robot_pts_base, cam_T_base = load_robot_asset(
            robot_num_points, device, franka_q=args.franka_joints
        )  # base_link frame
        robot_world = (robot_pts_base @ base_R_t.T + base_pos_t).unsqueeze(0)  # (1, M, 3) world
        cam_np = robot_camera_pose_world(cam_T_base, base_pos, base_R)
        camera_pose = torch.from_numpy(cam_np).to(device).unsqueeze(0)
        cam_desc = f"x5_camera_link @ {np.round(cam_np[:3], 3).tolist()} (mount -45deg roll)"
    else:
        # Virtual front look-at, shifted to the robot's right (matches render_wall_configs non-robot).
        eye = np.array([args.camera_right, -args.standoff, args.camera_height], dtype=np.float32)
        target = np.array([0.0, 0.0, args.camera_look_z], dtype=np.float32)
        camera_pose = torch.from_numpy(look_at_camera_pose(eye, target)).to(device).unsqueeze(0)
        cam_desc = f"virtual look-at, eye {eye.tolist()} -> {target.tolist()}"

    # One shared full-resolution spec keeps door and wall depth in exact pixel alignment.
    fov_kwargs = {}
    if depth_cfg.get("fov_x_deg") is not None:
        fov_kwargs["fov_x_deg"] = float(depth_cfg["fov_x_deg"])
    if depth_cfg.get("fov_y_deg") is not None:
        fov_kwargs["fov_y_deg"] = float(depth_cfg["fov_y_deg"])
    cam_spec = build_realsense_sampler_spec(
        args.cam_height_px, args.cam_width_px, device=device, **fov_kwargs
    )

    # Now that the camera pose is known, resolve which panel face points at it, so the reflection veil is
    # placed BEHIND the glass (link_1 z column of R_world_link1 is the panel normal in world).
    if reflection_enabled:
        panel_normal_world = R_world_link1_np[:, 2]
        cam_to_panel = camera_pose[0, :3].cpu().numpy() - link1_pos_world_np
        reflection_front_sign = float(np.sign(float(np.dot(cam_to_panel, panel_normal_world))) or 1.0)

    walls_on = (not args.no_walls) and wall_params.enabled and wall_params.num_points > 0
    print(f"[INFO] device        : {device}")
    print(f"[INFO] door           : {args.door.parent.name}  ({board_gt_world.shape[1]} pts, yaw {round(math.degrees(door_yaw))}deg -> handle faces robot)")
    print(f"[INFO] walls          : {'on (' + str(wall_params.num_points) + ' pts)' if walls_on else 'off'}")
    print(f"[INFO] door frame     : {'on (link_0, env_prob=' + str(frame_env_prob) + ', ' + str(frame_gt_world.shape[1]) + ' pts)' if frame_aug_enabled else 'off (panel+handle only, matches training default)'}")
    print(f"[INFO] window hole    : {'on (env_prob=' + str(hole_env_prob) + ', per-config w' + str(hole_width_range) + ' h' + str(hole_height_range) + ')' if hole_aug_enabled else 'off'}")
    print(f"[INFO] glass reflect  : {'on (prob=' + str(reflection_prob) + ', ' + str(reflection_num_points) + ' reflected robot pts, density=' + str(reflection_density_range) + ', front_sign=' + str(reflection_front_sign) + ')' if reflection_enabled else 'off'}")
    print(f"[INFO] robot          : {'on (' + str(robot_world.shape[1]) + ' pts, in ray cast)' if robot_world is not None else 'off'}")
    if robot_world is not None:
        joint_values = args.franka_joints if args.franka_joints is not None else FRANKA_READY_JOINT_POS
        print(f"[INFO] robot pose     : base={base_pos.tolist()}, Panda joints={joint_values}")
    print(f"[INFO] camera         : {args.cam_width_px}x{args.cam_height_px}px RealSense, "
          f"range [{args.near_m}, {args.far_m}] m, {cam_desc}")
    print(f"[INFO] gt_scale       : {args.gt_scale}  (door {board_num_points} pts, walls {wall_params.num_points} pts cap)")
    print(f"[INFO] rasterization  : one combined scene+robot z-buffer pass (inflate_px={inflate_px}; no occluder pass)")
    filter_info = (f"Gaussian ({blur_kernel_px}px, sigma={blur_sigma_px:g}px)"
                   if depth_filter == "gaussian" else f"min pooling ({blur_kernel_px}px window)")
    print(f"[INFO] depth filter   : {filter_info}")
    print(f"[INFO] edge drop      : {edge_drop_m} m  ({'ON -> scene edge smears dropped' if edge_drop_m>0 else 'off'})")
    print(f"[INFO] policy input   : {policy_base_points} sampled base-crop points -> z[0.55,1.5], "
          f"cyl r<{policy_crop_range}, x>{policy_x_cutoff} "
          f"(multi_pcd_dagger._build_local_pcd / crop_local_pcd)")
    print(f"[INFO] round-trip: GT points -> depth image -> points, {args.num_configs} configs")

    def policy_input_from_world(cloud_world):
        """Apply multi_pcd_dagger's EXACT policy-input crop to a world cloud, return the surviving points
        (world frame). Crop happens in the robot base frame (z = height above floor), so tall walls stay
        and boxes whose top < 0.55 vanish -- what the policy actually receives."""
        if cloud_world is None or cloud_world.shape[0] == 0:
            return cloud_world
        base_pts = (cloud_world - base_pos_t) @ base_R_t  # world -> robot base frame
        cropped, _ = crop_local_pcd(
            base_pts.unsqueeze(0),
            local_range=policy_crop_range,
            num_local_points=min(policy_base_points, base_pts.shape[0]),
            is_cylindrical=True,
            crop_center=torch.zeros((1, 3), device=device, dtype=torch.float32),
            x_direction_cutoff=policy_x_cutoff,
            log_name="policy",
        )
        cropped = cropped[0]
        valid = cropped.abs().sum(-1) > 1e-9  # crop_local_pcd zero-pads; base-frame origin => padding
        return cropped[valid] @ base_R_t.T + base_pos_t  # base -> world

    def prefilter_to_policy_volume(cloud_world):
        """Experimental: remove points outside the configured base cylindrical crop BEFORE ray casting.

        This is intentionally only the base crop (the current cfg allocates no palm-crop points).
        It is not assumed equivalent: out-of-crop geometry can still occlude in-crop returns.
        """
        if cloud_world is None or cloud_world.shape[1] == 0:
            return cloud_world
        base_pts = (cloud_world - base_pos_t) @ base_R_t
        finite = torch.isfinite(base_pts).all(dim=-1)
        keep = (
            finite
            & (base_pts[..., 2] >= 0.55)
            & (base_pts[..., 2] <= 1.5)
            & (torch.linalg.vector_norm(base_pts[..., :2], dim=-1) < policy_crop_range)
            & (base_pts[..., 0] > policy_x_cutoff)
        )
        return cloud_world[keep].unsqueeze(0)

    def sync_render_device():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    frames = [None] * args.num_configs
    for start in range(0, args.num_configs, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, args.num_configs)))
        # Stack this chunk of configs into one (Bc, N, 3) SCENE batch: shared door with per-config walls.
        # The robot is kept SEPARATE (crisp pass) so blur/jitter never touch it. The occluder cloud
        # (door + walls, NO robot) drives the anti-penetration second pass; the robot stays out of it so
        # its thin links aren't dilated away (matches training).
        scene_clouds, gt_clouds = [], []
        for _ in idxs:
            # One window hole per config (per rollout), drawn once and baked into the panel cloud (NaN);
            # its glass reflection veil (if on) is a sparse world cloud over the same window. The veil
            # joins the scene AND the occluder set (matching multi_pcd_dagger: a dark glass door reflects
            # instead of showing through, so the reflection surface occludes the room behind it).
            holed_board, reflection_world = make_holed_door(board_gt_world)
            door_parts = [holed_board]
            if reflection_world is not None:
                door_parts.append(reflection_world)
            if frame_aug_enabled and torch.rand((), device=device).item() < frame_env_prob:
                door_parts.append(frame_gt_world)
            door_cloud = torch.cat(door_parts, dim=1)
            # Source composes fixed per-link caches and then takes exactly scene_door_num_points.
            # Repeating the standalone cache here preserves the visible geometry / dropout pattern
            # while reproducing that fixed source boundary for the source-aware split.
            if door_cloud.shape[1] != scene_door_num_points:
                index = torch.linspace(0, door_cloud.shape[1] - 1, scene_door_num_points, device=device).round().long()
                door_cloud = door_cloud[:, index]
            if walls_on:
                wall_local = sample_walls()
                wall_cloud = door_to_world(wall_local)
            else:
                wall_cloud = torch.full((1, 0, 3), float("nan"), device=device)
            scene = torch.cat([door_cloud, wall_cloud], dim=1)
            scene_clouds.append(scene)
            gt_clouds.append(torch.cat([scene, robot_world], dim=1) if robot_world is not None else scene)
        # Config-local augments vary valid point counts; NaN padding keeps one batched render.
        scene_clouds = pad_cloud_batch(scene_clouds)
        robot_b = robot_world.expand(len(idxs), -1, -1) if robot_world is not None else None
        cam_b = camera_pose.expand(len(idxs), -1)

        # Exclude one-time CUDA/context initialization from the comparison. Warm both full and
        # prefiltered paths on identical geometry before timing the first batch.
        if args.compare_policy_prefilter and start == 0 and device.type == "cuda":
            warm_scene = scene_clouds[:1]
            warm_robot = robot_b[:1] if robot_b is not None else None
            warm_cam = cam_b[:1]
            render_dex_style_depth(warm_scene, warm_robot, warm_cam, cam_spec, depth_cfg,
                                   depth_filter=depth_filter, blur_sigma_px=blur_sigma_px)
            warm_scene_filtered = prefilter_to_policy_volume(warm_scene[0:1])
            if warm_robot is not None:
                warm_robot_filtered = prefilter_to_policy_volume(warm_robot).expand(1, -1, -1)
            else:
                warm_robot_filtered = None
            render_dex_style_depth(warm_scene_filtered, warm_robot_filtered, warm_cam, cam_spec, depth_cfg,
                                   depth_filter=depth_filter, blur_sigma_px=blur_sigma_px)
            sync_render_device()

        sync_render_device()
        full_start = time.perf_counter()
        rep_list, n_px = render_dex_style_depth(
            scene_clouds, robot_b, cam_b, cam_spec, depth_cfg,
            depth_filter=depth_filter, blur_sigma_px=blur_sigma_px,
        )
        sync_render_device()
        full_render_ms = (time.perf_counter() - full_start) * 1000.0

        prefiltered_rep_list = None
        prefiltered_scene_counts = []
        prefiltered_robot = None
        prefiltered_render_ms = None
        if args.compare_policy_prefilter:
            sync_render_device()
            cropped_start = time.perf_counter()
            prefiltered_scenes = []
            for batch_idx in range(scene_clouds.shape[0]):
                filtered = prefilter_to_policy_volume(scene_clouds[batch_idx:batch_idx + 1])
                prefiltered_scene_counts.append(int(filtered.shape[1]))
                prefiltered_scenes.append(filtered)
            prefiltered_scene_batch = pad_cloud_batch(prefiltered_scenes)
            if robot_world is not None:
                prefiltered_robot = prefilter_to_policy_volume(robot_world)
                prefiltered_robot_b = prefiltered_robot.expand(len(idxs), -1, -1)
            else:
                prefiltered_robot_b = None
            prefiltered_rep_list, _ = render_dex_style_depth(
                prefiltered_scene_batch, prefiltered_robot_b, cam_b, cam_spec, depth_cfg,
                depth_filter=depth_filter, blur_sigma_px=blur_sigma_px,
            )
            sync_render_device()
            prefiltered_render_ms = (time.perf_counter() - cropped_start) * 1000.0

        for j, config_idx in enumerate(idxs):
            frames[config_idx] = {
                "pointclouds": {
                    "ground_truth": pack(drop_invalid_rows(gt_clouds[j]), args.max_points),
                    "reprojected": pack(rep_list[j], args.max_points),
                    "policy_input": pack(policy_input_from_world(rep_list[j]), args.max_points),
                    # Keep the robot as a separately toggleable layer.  It is also
                    # included in ground_truth and the ray cast above; this stream is
                    # purely for inspecting exactly which geometry self-occludes.
                    **({"robot": pack(robot_world[0], args.max_points)} if robot_world is not None else {}),
                    **({
                        "prefiltered_reprojected": pack(prefiltered_rep_list[j], args.max_points),
                        "prefiltered_policy_input": pack(policy_input_from_world(prefiltered_rep_list[j]), args.max_points),
                    } if prefiltered_rep_list is not None else {}),
                }
            }
        last = idxs[-1]
        print(f"  configs {idxs[0] + 1}-{last + 1}/{args.num_configs} (batch {len(idxs)}): "
              f"valid px≈{rep_list[-1].shape[0]}/{n_px}; full render {full_render_ms / len(idxs):.2f} ms/config")
        if prefiltered_rep_list is not None:
            original_points = int(scene_clouds.shape[1]) + (int(robot_b.shape[1]) if robot_b is not None else 0)
            cropped_points = max(prefiltered_scene_counts, default=0) + (int(prefiltered_robot.shape[1]) if prefiltered_robot is not None else 0)
            print(f"    prefilter: max {cropped_points:,}/{original_points:,} source pts; "
                  f"render {prefiltered_render_ms / len(idxs):.2f} ms/config "
                  f"({full_render_ms / max(prefiltered_render_ms, 1e-9):.2f}x full/prefiltered)")

    payload = {
        "format": "dooropening_viser_replay_v1",
        "pointcloud_frame": "world",
        "pointcloud_source": "single_zbuffer_scene_plus_robot",
        "depth_filter": depth_filter,
        "robot_base_pos_w": base_pos.tolist() if robot_world is not None else None,
        "franka_joint_pos": (args.franka_joints if args.franka_joints is not None else FRANKA_READY_JOINT_POS)
        if robot_world is not None else None,
        "pointcloud_streams": [
            {"name": "ground_truth", "label": "GT (door + walls [+ robot])", "color": (120, 120, 120), "point_size_scale": 1.0},
            *([{"name": "robot", "label": "Robot (ray-cast geometry)", "color": (255, 140, 0), "point_size_scale": 1.4}] if args.robot else []),
            {"name": "reprojected", "label": "Single-pass depth (combined scene + robot)", "color": (79, 195, 247), "point_size_scale": 1.4},
            {"name": "policy_input", "label": "Policy input (cropped)", "color": (124, 240, 130), "point_size_scale": 1.8},
            *([{"name": "prefiltered_reprojected", "label": "Early policy-volume render", "color": (240, 100, 220), "point_size_scale": 1.4},
               {"name": "prefiltered_policy_input", "label": "Early-render policy input", "color": (255, 220, 80), "point_size_scale": 1.8}]
              if args.compare_policy_prefilter else []),
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
