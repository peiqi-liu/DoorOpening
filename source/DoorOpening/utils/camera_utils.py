import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# --- On-robot depth camera model (Intel RealSense D435 depth stream) --------------------------
# These are the single source of truth for the simulated depth-camera intrinsics/range used by the
# distillation pointcloud renderer (see multi_pcd_dagger._build_sampler_camera_spec) AND for driving
# the real IsaacLab camera to matching intrinsics in scripts/rl_games/play.py. Keep them here so the
# two never drift apart.
REALSENSE_D435_FOV_X_DEG = 85.0
REALSENSE_D435_FOV_Y_DEG = 58.0
REALSENSE_D435_NEAR_M = 0.3
REALSENSE_D435_FAR_M = 3.0


def build_realsense_sampler_spec(
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    fov_x_deg: float = REALSENSE_D435_FOV_X_DEG,
    fov_y_deg: float = REALSENSE_D435_FOV_Y_DEG,
    near_m: float = REALSENSE_D435_NEAR_M,
    far_m: float = REALSENSE_D435_FAR_M,
) -> Dict:
    """Build the D435 depth-camera ``cam_spec_dict`` ({H, W, intrinsics, near_m, far_m}).

    ``height``/``width`` are the ACTUAL render resolution (already halved, if the caller halves the
    sensor resolution). The intrinsics are a centred pinhole derived from the requested horizontal /
    vertical field of view, exactly matching the historical inline math in
    ``multi_pcd_dagger._build_sampler_camera_spec`` so the two stay in sync.
    """
    height = int(height)
    width = int(width)
    fx = width / (2.0 * math.tan(math.radians(fov_x_deg) * 0.5))
    fy = height / (2.0 * math.tan(math.radians(fov_y_deg) * 0.5))
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5

    intrinsics = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    return {
        "H": height,
        "W": width,
        "intrinsics": intrinsics,
        "near_m": float(near_m),
        "far_m": float(far_m),
    }


def build_pinhole_intrinsics(
    height: int,
    width: int,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float | None = None,
    device=None,
    dtype=torch.float32,
):
    """Build a 3x3 pinhole intrinsics matrix from Isaac camera parameters."""
    if vertical_aperture is None:
        vertical_aperture = horizontal_aperture * float(height) / float(width)

    fx = float(width) * float(focal_length) / float(horizontal_aperture)
    fy = float(height) * float(focal_length) / float(vertical_aperture)
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5

    intrinsics = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return intrinsics


def depth_to_pointcloud(
    depth,
    intrinsics=None,
    fx=383.0,
    fy=383.0,
    cx=320.0,
    cy=240.0,
    debug=False,
    local_range=1.0,
    num_local_points=1000,
    is_cylindrical=False,
    x_direction_cutoff=-0.5,
):
    """
    Convert a depth image to a point cloud.

    Args:
        depth: (..., H, W) or (..., H, W, 1)
        intrinsics: Optional camera intrinsic matrix of shape (3, 3) or (..., 3, 3).
            When provided, ``fx``, ``fy``, ``cx`` and ``cy`` are read from it.

    Returns:
        (..., N, 3), where N is ``num_local_points`` when cropping is enabled,
        otherwise ``H * W``.
    """

    if not isinstance(depth, torch.Tensor):
        depth = torch.from_numpy(depth)

    if depth.ndim < 2:
        raise ValueError(f"Expected depth to have at least 2 dims, got shape {tuple(depth.shape)}")

    if depth.ndim >= 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    if depth.ndim == 2:
        batch_shape = ()
        depth = depth.unsqueeze(0)
    else:
        batch_shape = tuple(depth.shape[:-2])

    H, W = depth.shape[-2:]
    depth = depth.reshape(-1, H, W)
    device = depth.device
    dtype = depth.dtype

    if intrinsics is not None:
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.as_tensor(intrinsics, device=device, dtype=dtype)
        else:
            intrinsics = intrinsics.to(device=device, dtype=dtype)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected intrinsics to have shape (3, 3) or (B, 3, 3), got {tuple(intrinsics.shape)}"
            )
        if intrinsics.shape[0] == 1 and depth.shape[0] != 1:
            intrinsics = intrinsics.expand(depth.shape[0], -1, -1)
        elif intrinsics.shape[0] != depth.shape[0]:
            raise ValueError(
                f"Depth batch size {depth.shape[0]} does not match intrinsics batch size {intrinsics.shape[0]}"
            )
        fx = intrinsics[:, 0, 0].view(-1, 1, 1)
        fy = intrinsics[:, 1, 1].view(-1, 1, 1)
        cx = intrinsics[:, 0, 2].view(-1, 1, 1)
        cy = intrinsics[:, 1, 2].view(-1, 1, 1)
    else:
        fx = torch.full((depth.shape[0], 1, 1), float(fx), device=device, dtype=dtype)
        fy = torch.full((depth.shape[0], 1, 1), float(fy), device=device, dtype=dtype)
        cx = torch.full((depth.shape[0], 1, 1), float(cx), device=device, dtype=dtype)
        cy = torch.full((depth.shape[0], 1, 1), float(cy), device=device, dtype=dtype)

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    uu = uu.unsqueeze(0).expand(depth.shape[0], -1, -1)
    vv = vv.unsqueeze(0).expand(depth.shape[0], -1, -1)

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points = torch.stack((x, y, z), dim=-1).reshape(depth.shape[0], H * W, 3)
    valid_mask = torch.isfinite(depth).reshape(depth.shape[0], H * W) & (depth.reshape(depth.shape[0], H * W) > 0.0)
    points[~valid_mask] = float("nan")

    if num_local_points is not None:
        points, _ = crop_local_pcd(
            points,
            local_range=local_range,
            num_local_points=num_local_points,
            is_cylindrical=is_cylindrical,
            x_direction_cutoff=x_direction_cutoff,
        )

    if debug:
        import numpy as np
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        np_points = points.detach().cpu().numpy().astype(np.float64).reshape(-1, 3)
        np_points[:, 1] *= -1
        np_points[:, 2] *= -1
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(np_points))

        o3d.visualization.draw_geometries([pcd], window_name="Open3D Point Cloud Visualization")
        o3d.io.write_point_cloud("pointcloud.ply", pcd)

    if batch_shape:
        return points.reshape(*batch_shape, points.shape[-2], 3)
    return points.squeeze(0)


def shuffle_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """
    Randomize ordering of points in a point cloud.
    
    Args:
        pcd: (B, N, 3) tensor
    Returns:
        shuffled: (B, N, 3) tensor with points randomly permuted
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # Random permutations (different per batch)
    idx = torch.argsort(torch.rand(B, N, device=device), dim=-1)  # (B, N)

    # Build batch index for gather
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)

    # Gather shuffled points
    return pcd[batch_idx, idx]  # (B, N, 3)

def crop_local_pcd(
    pcd: torch.Tensor,
    local_range: torch.float,
    num_local_points: torch.int,
    is_cylindrical: bool = False,
    crop_center: torch.Tensor = None,
    x_direction_cutoff: torch.float = -0.5,
    max_height_m: float | None = 1.5,
    min_height_m: float | None = 0.55,
    log_name: str = "",
):
    """
    Crop the point cloud to a local region around the origin with 0 padding.
    Args:
        pcd: (B, N, 3) tensor
        local_range: float, the radius of the local region
        crop_center: (B, 3)
    """
    B, N, _ = pcd.shape
    device = pcd.device

    if crop_center is None:
        crop_center = torch.zeros((B, 3), device=pcd.device, dtype=pcd.dtype)

    # Remove overhead and underfloor points in the input frame before local cropping.
    if max_height_m is not None:
        pcd = pcd.clone()
        height_mask = torch.isfinite(pcd[..., 2]) & (pcd[..., 2] <= float(max_height_m))
        pcd[~height_mask] = float("nan")

    if min_height_m is not None:
        pcd = pcd.clone()
        height_mask = torch.isfinite(pcd[..., 2]) & (pcd[..., 2] >= float(min_height_m))
        pcd[~height_mask] = float("nan")

    crop_center_expanded = crop_center.unsqueeze(1)
    pcd_centered = pcd - crop_center_expanded

    # get local pcd
    masked_pcds = shuffle_pcd(pcd_centered)
    if is_cylindrical:
        dist = torch.norm(masked_pcds[..., :2], dim=-1)
    else:
        dist = torch.norm(masked_pcds, dim=-1)
    valid_mask = dist < local_range  # nan < X is false, so NaN points are filtered out here
    masked_pcds[~valid_mask] = float("nan")

    if x_direction_cutoff is not None:
        x_dir_mask = masked_pcds[:, :, 0] > x_direction_cutoff
        valid_mask = valid_mask & x_dir_mask
        masked_pcds[~x_dir_mask] = float("nan")

    # sort to get all the valid points
    is_valid = valid_mask.int()
    sort_idx = torch.argsort(is_valid, dim=-1, descending=True)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    sorted_pcds = masked_pcds[batch_idx, sort_idx]  # (B, N, 3)
    pcd_local_nan_padding = sorted_pcds[:, :num_local_points]
    sorted_valid_mask = valid_mask[batch_idx, sort_idx][:, :num_local_points]

    avg_num_valid_points = is_valid.sum() / B
    min_num_valid_points = is_valid.sum(dim=-1).min()

    crop_type = "cylindrical" if is_cylindrical else "spherical"
    logs = {
        f"{log_name}_local_{crop_type}_crop/avg_num_valid_points": avg_num_valid_points.item(),
        f"{log_name}_local_{crop_type}_crop/min_num_valid_points": min_num_valid_points.item(),
    }

    # replace nan values as 0s
    local_pcd_zero_padding = torch.nan_to_num(pcd_local_nan_padding, nan=0.0)

    # Shift only valid points back to the original frame. Keep padding at zero instead of fabricating
    # points at the crop center.
    cropped_pcd = local_pcd_zero_padding + crop_center_expanded * sorted_valid_mask.unsqueeze(-1).to(pcd.dtype)

    return cropped_pcd, logs


_uv_cache: dict = {}
_intr_cache: dict = {}
_compiled_cache: dict = {}


def _get_uv_base(H: int, W: int, device, dtype):
    key = (H, W, device, dtype)
    if key not in _uv_cache:
        u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(1, H, W)
        v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(1, H, W)
        _uv_cache[key] = (u, v)
    return _uv_cache[key]


def _get_render_intrinsics(cam_spec_dict: Dict, device, dtype):
    H = int(cam_spec_dict["H"])
    W = int(cam_spec_dict["W"])
    intrinsics = cam_spec_dict.get("intrinsics")
    if intrinsics is None:
        raise ValueError("cam_spec_dict must provide an 'intrinsics' matrix.")

    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.as_tensor(intrinsics, device=device, dtype=dtype)
    else:
        intrinsics = intrinsics.to(device=device, dtype=dtype)

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3, 3), got {tuple(intrinsics.shape)}")

    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    cx = float(intrinsics[0, 2].item())
    cy = float(intrinsics[1, 2].item())

    key = (H, W, fx, fy, cx, cy, device, dtype)
    if key not in _intr_cache:
        _intr_cache[key] = (
            torch.tensor(fx, device=device, dtype=dtype),
            torch.tensor(fy, device=device, dtype=dtype),
            torch.tensor(cx, device=device, dtype=dtype),
            torch.tensor(cy, device=device, dtype=dtype),
        )
    return _intr_cache[key]


def _camera_basis_from_pose_x_forward(
    camera_pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert camera_pose.shape[-1] == 7, "camera_pose must have last dim = 7 (pos + quat_xyzw)"

    q = camera_pose[..., 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    qx, qy, qz, qw = q.unbind(-1)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R00 = 1.0 - 2.0 * (yy + zz)
    R01 = 2.0 * (xy - wz)
    R02 = 2.0 * (xz + wy)
    R10 = 2.0 * (xy + wz)
    R11 = 1.0 - 2.0 * (xx + zz)
    R12 = 2.0 * (yz - wx)
    R20 = 2.0 * (xz - wy)
    R21 = 2.0 * (yz + wx)
    R22 = 1.0 - 2.0 * (xx + yy)

    v_hat = torch.stack([R00, R10, R20], dim=-1)
    u_hat = torch.stack([R01, R11, R21], dim=-1)
    w_hat = torch.stack([R02, R12, R22], dim=-1)

    v_hat = v_hat / v_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u_hat = u_hat / u_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w_hat = w_hat / w_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return u_hat, w_hat, v_hat


def _dilate_depth_min_pool(depth: torch.Tensor, inflate_px: int) -> torch.Tensor:
    """Pixel-space nearest-neighbor fill: each pixel becomes the min depth within a
    (2*inflate_px+1) window, closing small gaps left by sparse point coverage."""
    if inflate_px <= 0:
        return depth
    neg = -depth.unsqueeze(1)
    neg = F.pad(neg, (inflate_px, inflate_px, inflate_px, inflate_px), mode="constant", value=float("-inf"))
    k = 2 * inflate_px + 1
    pooled_neg = F.max_pool2d(neg, kernel_size=(k, k), stride=1)
    return (-pooled_neg).squeeze(1)


def _zbuffer_scatter_raw_depth(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,
    cam_spec_dict: Dict,
    clip_mode: str,
) -> torch.Tensor:
    """Per-pixel min-depth z-buffer scatter, before dilation and before near/far clipping."""
    B, _, _ = pcd.shape
    H = int(cam_spec_dict["H"])
    W = int(cam_spec_dict["W"])
    near_m = float(cam_spec_dict["near_m"])
    far_m = cam_spec_dict["far_m"]
    far_val = float("inf") if far_m is None else float(far_m)
    device, dtype = pcd.device, pcd.dtype
    finite_input = torch.isfinite(pcd).all(dim=-1)
    pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)

    fx, fy, cx, cy = _get_render_intrinsics(cam_spec_dict, device, dtype)
    cam_pos = camera_pose[:, 0:3]
    u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)

    rel = pcd - cam_pos[:, None, :]
    x = (rel * u_hat[:, None, :]).sum(-1)
    y = (rel * w_hat[:, None, :]).sum(-1)
    z = (rel * v_hat[:, None, :]).sum(-1)

    invz = 1.0 / z.clamp_min(1e-12)
    u_pix = fx * (x * invz) + cx
    v_pix = fy * (y * invz) + cy

    in_front = z > 0
    inside = finite_input & (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front
    if clip_mode == "pre":
        inside = inside & (z >= near_m) & (z <= far_val)

    iu = u_pix.floor().clamp(0, W - 1).long()
    iv = v_pix.floor().clamp(0, H - 1).long()
    pix = iv * W + iu

    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("Tensor.scatter_reduce_ not found. Need PyTorch >= 1.12.")

    inf = torch.full((), float("inf"), device=device, dtype=dtype)
    z_masked = torch.where(inside, z, inf)
    K = H * W
    min_depth = torch.full((B, K), float("inf"), device=device, dtype=dtype)
    min_depth.scatter_reduce_(dim=1, index=pix, src=z_masked, reduce="amin", include_self=True)
    return min_depth.view(B, H, W)


@torch.no_grad()
def rasterize_depth_zbuffer_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,
    cam_spec_dict: Dict,
    inflate_px: int = 0,
    clip_mode: str = "post",
    occluder_pcd: Optional[torch.Tensor] = None,
    occluder_inflate_px: int = 0,
):
    """
    occluder_pcd / occluder_inflate_px: an optional second point set (e.g. wall distractors)
    rasterized in a separate pass with a much larger inflate_px and composited in via a per-pixel
    min with the main depth. This closes gaps that a sparse occluder leaves in the fine-detail
    z-buffer (which needs a small inflate_px to keep thin features like the door handle crisp)
    without needing to blur the whole scene to fill them.
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    near_m = float(cam_spec_dict["near_m"])
    far_m = cam_spec_dict["far_m"]
    far_val = float("inf") if far_m is None else float(far_m)
    device, dtype = pcd.device, pcd.dtype
    intr = _get_render_intrinsics(cam_spec_dict, device, dtype)

    depth = _zbuffer_scatter_raw_depth(pcd, camera_pose, cam_spec_dict, clip_mode)
    depth = _dilate_depth_min_pool(depth, inflate_px)

    if occluder_pcd is not None and occluder_inflate_px > 0 and occluder_pcd.shape[1] > 0:
        occluder_depth = _zbuffer_scatter_raw_depth(occluder_pcd, camera_pose, cam_spec_dict, clip_mode)
        occluder_depth = _dilate_depth_min_pool(occluder_depth, occluder_inflate_px)
        depth = torch.minimum(depth, occluder_depth)

    inf = torch.full((), float("inf"), device=device, dtype=dtype)
    depth = torch.where(depth >= near_m, depth, inf)
    depth = torch.where(depth <= far_val, depth, inf)
    return depth, intr


@torch.no_grad()
def backproject_depth_to_world_from_pose(
    depth: torch.Tensor,
    camera_pose: torch.Tensor,
    intrinsics: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
):
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    fx, fy, cx, cy = intrinsics

    cam_pos = camera_pose[:, 0:3]
    u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)
    u_base, v_base = _get_uv_base(H, W, device, dtype)
    u = u_base.expand(B, H, W)
    v = v_base.expand(B, H, W)

    valid = torch.isfinite(depth)
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    nan = torch.full((), float("nan"), device=device, dtype=dtype)
    x = torch.where(valid, x, nan)
    y = torch.where(valid, y, nan)
    z = torch.where(valid, z, nan)

    uB = u_hat.view(B, 1, 1, 3)
    wB = w_hat.view(B, 1, 1, 3)
    vB = v_hat.view(B, 1, 1, 3)
    oB = cam_pos.view(B, 1, 1, 3)
    pcd_world = oB + x[..., None] * uB + y[..., None] * wB + z[..., None] * vB
    return pcd_world, valid


@torch.no_grad()
def build_depth_blur_kernel2d(kernel_px: int, sigma_px: float, device, dtype):
    """Separable Gaussian (box if sigma<=0) blur kernel for a depth image, returned as ((1,1,k,k), pad)."""
    k = int(kernel_px)
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    if sigma_px and sigma_px > 0.0:
        ker1d = torch.exp(-0.5 * (coords / float(sigma_px)) ** 2)
    else:
        ker1d = torch.ones(k, device=device, dtype=dtype)
    ker1d = ker1d / ker1d.sum()
    ker2d = (ker1d[:, None] * ker1d[None, :]).reshape(1, 1, k, k)
    return ker2d, k // 2


@torch.no_grad()
def apply_depth_spatial_blur(depth: torch.Tensor, kernel2d: torch.Tensor, pad: int):
    """Edge-bleeding normalized blur of a depth image (invalid pixels are +inf).

    Mimics the RealSense SDK spatial filter: each valid pixel becomes a weighted average of its
    valid neighbours, so the handle silhouette rounds off and depth bleeds across the handle/door
    boundary (spatially-correlated smoothing, unlike per-point scalar jitter). Pixels with no valid
    neighbour, and originally-invalid pixels, are left untouched.
    """
    valid = torch.isfinite(depth)
    w = valid.to(depth.dtype)
    depth_filled = torch.where(valid, depth, torch.zeros_like(depth))
    num = F.conv2d((depth_filled * w).unsqueeze(1), kernel2d, padding=pad).squeeze(1)
    den = F.conv2d(w.unsqueeze(1), kernel2d, padding=pad).squeeze(1)
    blurred = num / den.clamp_min(1e-6)
    return torch.where(valid & (den > 1e-6), blurred, depth)


@torch.no_grad()
def drop_depth_edges(depth: torch.Tensor, grad_thresh_m: float):
    """Invalidate depth pixels sitting on a depth discontinuity (a real depth-camera artifact).

    For each valid pixel, take the max abs depth difference to its 4 neighbours (ignoring invalid
    neighbours); if it exceeds ``grad_thresh_m`` the pixel straddles an edge and is set to +inf. This
    removes the smeared "flying pixel" points the spatial blur leaves along wall / door silhouettes --
    a real depth camera drops edge returns rather than bleeding them. ``grad_thresh_m <= 0`` is a no-op.
    """
    if grad_thresh_m is None or float(grad_thresh_m) <= 0.0:
        return depth
    valid = torch.isfinite(depth)
    inf = torch.full((), float("inf"), device=depth.device, dtype=depth.dtype)
    grad = torch.zeros_like(depth)

    # Non-wrapping neighbour diffs (image borders must NOT wrap onto the opposite edge -- that would
    # falsely drop the border pixels). Each interior diff is written to BOTH straddling pixels.
    dh = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    vh = valid[..., :, 1:] & valid[..., :, :-1]
    dh = torch.where(vh, dh, torch.zeros_like(dh))
    grad[..., :, :-1] = torch.maximum(grad[..., :, :-1], dh)
    grad[..., :, 1:] = torch.maximum(grad[..., :, 1:], dh)

    dv = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    vv = valid[..., 1:, :] & valid[..., :-1, :]
    dv = torch.where(vv, dv, torch.zeros_like(dv))
    grad[..., :-1, :] = torch.maximum(grad[..., :-1, :], dv)
    grad[..., 1:, :] = torch.maximum(grad[..., 1:, :], dv)

    edge = valid & (grad > float(grad_thresh_m))
    return torch.where(edge, inf, depth)


@torch.no_grad()
def render_points_to_world_grid_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,
    cam_spec_dict: Dict,
    inflate_px: int = 0,
    clip_mode: str = "post",
    jitter_std_m: float = 0.0,
    jitter_mode: str = "xyz",
    blur_kernel_px: int = 0,
    blur_sigma_px: float = 0.0,
    occluder_pcd: Optional[torch.Tensor] = None,
    occluder_inflate_px: int = 0,
):
    depth, intr = rasterize_depth_zbuffer_from_pose(
        pcd,
        camera_pose,
        cam_spec_dict=cam_spec_dict,
        inflate_px=inflate_px,
        clip_mode=clip_mode,
        occluder_pcd=occluder_pcd,
        occluder_inflate_px=occluder_inflate_px,
    )
    # Spatial blur of the depth image before unprojection (mimics the RealSense SDK spatial filter).
    if int(blur_kernel_px) > 1:
        kernel2d, pad = build_depth_blur_kernel2d(blur_kernel_px, blur_sigma_px, depth.device, depth.dtype)
        depth = apply_depth_spatial_blur(depth, kernel2d, pad)

    # Axial (along-ray) jitter perturbs DEPTH before unprojection, so points stay on their pixel rays
    # and the lateral silhouette (thin features like the handle) stays crisp -- the RealSense noise
    # model is dominantly axial. isotropic 'xyz' / 'tangent' jitter is applied in world space below.
    axial_jitter = jitter_std_m > 0.0 and jitter_mode.lower() in ("axial", "ray", "depth")
    if axial_jitter:
        finite = torch.isfinite(depth)
        depth = torch.where(finite, depth + torch.randn_like(depth) * float(jitter_std_m), depth)

    pcd_world, valid = backproject_depth_to_world_from_pose(depth, camera_pose, intr)

    if jitter_std_m > 0.0 and not axial_jitter:
        if jitter_mode.lower() == "xyz":
            noise = torch.randn_like(pcd_world) * float(jitter_std_m)
            pcd_world = torch.where(valid[..., None], pcd_world + noise, pcd_world)
        elif jitter_mode.lower() == "tangent":
            u_hat, w_hat, _ = _camera_basis_from_pose_x_forward(camera_pose)
            B, H_, W_ = depth.shape
            uB = u_hat.view(B, 1, 1, 3)
            wB = w_hat.view(B, 1, 1, 3)
            eps_u = torch.randn(B, H_, W_, 1, device=pcd.device, dtype=pcd.dtype) * float(jitter_std_m)
            eps_w = torch.randn(B, H_, W_, 1, device=pcd.device, dtype=pcd.dtype) * float(jitter_std_m)
            jitter = eps_u * uB + eps_w * wB
            pcd_world = torch.where(valid[..., None], pcd_world + jitter, pcd_world)
        else:
            raise ValueError(f"Unknown jitter_mode '{jitter_mode}'. Use 'axial', 'tangent', or 'xyz'.")

    return depth, pcd_world, valid


def get_compiled_renderer_fixed_shapes(
    cam_spec_dict: Dict,
    inflate_px: int = 0,
    clip_mode: str = "post",
    jitter_mode: str = "xyz",
    compile_mode: str = "max-autotune",
    blur_kernel_px: int = 0,
    blur_sigma_px: float = 0.0,
    occluder_inflate_px: int = 0,
):
    """
    occluder_inflate_px > 0: the returned wrapper additionally accepts an `occluder_pcd` tensor
    (e.g. wall distractor points), rasterized in a second pass with this (typically much larger)
    inflate_px and composited into the main depth via a per-pixel min. See
    rasterize_depth_zbuffer_from_pose for why this is needed on top of the main inflate_px.
    """
    axial_jitter = jitter_mode.lower() in ("axial", "ray", "depth")
    if not axial_jitter and jitter_mode.lower() != "xyz":
        raise ValueError("Compiled renderer supports jitter_mode='xyz' or 'axial' only.")

    H = int(cam_spec_dict["H"])
    W = int(cam_spec_dict["W"])
    near_m = float(cam_spec_dict["near_m"])
    far_m = cam_spec_dict["far_m"]
    far_val = float("inf") if far_m is None else float(far_m)
    clip_pre = clip_mode == "pre"
    has_occluder = int(occluder_inflate_px) > 0
    intrinsics = cam_spec_dict.get("intrinsics")
    if intrinsics is None:
        raise ValueError("cam_spec_dict must provide an 'intrinsics' matrix.")
    if isinstance(intrinsics, torch.Tensor):
        intrinsics_key = tuple(float(v) for v in intrinsics.detach().cpu().reshape(-1).tolist())
    else:
        intrinsics_key = tuple(float(v) for v in torch.as_tensor(intrinsics).reshape(-1).tolist())

    def _get_or_build(device, dtype):
        key = (H, W, near_m, far_val, int(inflate_px), clip_mode, jitter_mode, compile_mode,
               int(blur_kernel_px), float(blur_sigma_px), int(occluder_inflate_px), intrinsics_key,
               device, dtype)
        if key in _compiled_cache:
            return _compiled_cache[key]

        fx, fy, cx, cy = _get_render_intrinsics(cam_spec_dict, device, dtype)
        u_base, v_base = _get_uv_base(H, W, device, dtype)
        PAD = (int(inflate_px), int(inflate_px), int(inflate_px), int(inflate_px))
        KERNEL = (int(2 * inflate_px + 1), int(2 * inflate_px + 1))
        OCC_PAD = (int(occluder_inflate_px),) * 4
        OCC_KERNEL = (int(2 * occluder_inflate_px + 1), int(2 * occluder_inflate_px + 1))
        if int(blur_kernel_px) > 1:
            BLUR_KERNEL2D, BLUR_PAD = build_depth_blur_kernel2d(blur_kernel_px, blur_sigma_px, device, dtype)
        else:
            BLUR_KERNEL2D, BLUR_PAD = None, 0

        def _rasterize(points, cam_pos, u_hat, w_hat, v_hat, pad, kernel, infl):
            finite_input = torch.isfinite(points).all(dim=-1)
            points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

            rel = points - cam_pos[:, None, :]
            x = (rel * u_hat[:, None, :]).sum(-1)
            y = (rel * w_hat[:, None, :]).sum(-1)
            z = (rel * v_hat[:, None, :]).sum(-1)

            invz = 1.0 / z.clamp_min(1e-12)
            u_pix = fx * (x * invz) + cx
            v_pix = fy * (y * invz) + cy

            in_front = z > 0
            inside = finite_input & (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front
            if clip_pre:
                inside = inside & (z >= near_m) & (z <= far_val)

            iu = u_pix.floor().clamp(0, W - 1).long()
            iv = v_pix.floor().clamp(0, H - 1).long()
            pix = iv * W + iu

            inf_local = torch.full((), float("inf"), device=points.device, dtype=points.dtype)
            z_masked = torch.where(inside, z, inf_local)
            K = H * W
            Bc = points.shape[0]
            min_depth = torch.full((Bc, K), float("inf"), device=points.device, dtype=points.dtype)
            min_depth.scatter_reduce_(dim=1, index=pix, src=z_masked, reduce="amin", include_self=True)
            d = min_depth.view(Bc, H, W)

            if infl > 0:
                neg = -d.unsqueeze(1)
                neg = F.pad(neg, pad, mode="constant", value=float("-inf"))
                pooled_neg = F.max_pool2d(neg, kernel_size=kernel, stride=1)
                d = (-pooled_neg).squeeze(1)
            return d

        def _backproject_and_jitter(depth, cam_pos, u_hat, w_hat, v_hat, jitter_std, B, pcd_device, pcd_dtype):
            u = u_base.expand(B, H, W)
            v = v_base.expand(B, H, W)
            valid = torch.isfinite(depth)
            zz = depth
            if axial_jitter:
                # Perturb depth ALONG the camera ray only (recompute x,y from the noisy depth). This
                # keeps each point on its pixel ray, so the lateral silhouette stays crisp and thin
                # features (e.g. the door handle) are not smeared -- matching the RealSense depth-noise
                # model, which is dominantly axial rather than isotropic.
                zz = zz + torch.where(valid, torch.randn_like(zz) * jitter_std, torch.zeros_like(zz))
            xx = (u - cx) / fx * zz
            yy = (v - cy) / fy * zz

            nan = torch.full((), float("nan"), device=pcd_device, dtype=pcd_dtype)
            xx = torch.where(valid, xx, nan)
            yy = torch.where(valid, yy, nan)
            zz = torch.where(valid, zz, nan)

            uB = u_hat.view(B, 1, 1, 3)
            wB = w_hat.view(B, 1, 1, 3)
            vB = v_hat.view(B, 1, 1, 3)
            oB = cam_pos.view(B, 1, 1, 3)
            pcd_world = oB + xx[..., None] * uB + yy[..., None] * wB + zz[..., None] * vB

            if not axial_jitter:
                noise = torch.randn_like(pcd_world) * jitter_std
                pcd_world = torch.where(valid[..., None], pcd_world + noise, pcd_world)
            return pcd_world, valid

        if has_occluder:

            @torch.no_grad()
            def _compiled_fn(
                pcd: torch.Tensor,
                camera_pose: torch.Tensor,
                jitter_std: torch.Tensor,
                occluder_pcd: torch.Tensor,
            ):
                cam_pos = camera_pose[:, 0:3]
                u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)
                B = pcd.shape[0]

                depth = _rasterize(pcd, cam_pos, u_hat, w_hat, v_hat, PAD, KERNEL, inflate_px)
                occluder_depth = _rasterize(
                    occluder_pcd, cam_pos, u_hat, w_hat, v_hat, OCC_PAD, OCC_KERNEL, occluder_inflate_px
                )
                depth = torch.minimum(depth, occluder_depth)

                inf = torch.full((), float("inf"), device=pcd.device, dtype=pcd.dtype)
                depth = torch.where(depth >= near_m, depth, inf)
                depth = torch.where(depth <= far_val, depth, inf)

                if BLUR_KERNEL2D is not None:
                    depth = apply_depth_spatial_blur(depth, BLUR_KERNEL2D, BLUR_PAD)

                pcd_world, valid = _backproject_and_jitter(
                    depth, cam_pos, u_hat, w_hat, v_hat, jitter_std, B, pcd.device, pcd.dtype
                )
                return depth, pcd_world, valid

        else:

            @torch.no_grad()
            def _compiled_fn(pcd: torch.Tensor, camera_pose: torch.Tensor, jitter_std: torch.Tensor):
                cam_pos = camera_pose[:, 0:3]
                u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)
                B = pcd.shape[0]

                depth = _rasterize(pcd, cam_pos, u_hat, w_hat, v_hat, PAD, KERNEL, inflate_px)

                inf = torch.full((), float("inf"), device=pcd.device, dtype=pcd.dtype)
                depth = torch.where(depth >= near_m, depth, inf)
                depth = torch.where(depth <= far_val, depth, inf)

                if BLUR_KERNEL2D is not None:
                    depth = apply_depth_spatial_blur(depth, BLUR_KERNEL2D, BLUR_PAD)

                pcd_world, valid = _backproject_and_jitter(
                    depth, cam_pos, u_hat, w_hat, v_hat, jitter_std, B, pcd.device, pcd.dtype
                )
                return depth, pcd_world, valid

        compiled = torch.compile(_compiled_fn, mode=compile_mode, dynamic=False)
        _compiled_cache[key] = compiled
        return compiled

    def wrapper(
        pcd: torch.Tensor,
        camera_pose: torch.Tensor,
        jitter_std_m: float,
        occluder_pcd: Optional[torch.Tensor] = None,
    ):
        compiled = _get_or_build(pcd.device, pcd.dtype)
        jitter_std = torch.tensor(float(jitter_std_m), device=pcd.device, dtype=pcd.dtype)
        if has_occluder:
            if occluder_pcd is None:
                raise ValueError("occluder_inflate_px > 0 requires occluder_pcd to be provided.")
            return compiled(pcd, camera_pose, jitter_std, occluder_pcd)
        return compiled(pcd, camera_pose, jitter_std)

    return wrapper


@torch.no_grad()
def simulate_depth_cam_render_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,
    num_points: int,
    inflate_px: int = 2,
    jitter_std_m: float = 0.004,
    cam_spec_dict: Optional[Dict] = None,
    clip_mode: str = "post",
    jitter_mode: str = "xyz",
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
    blur_kernel_px: int = 0,
    blur_sigma_px: float = 0.0,
    occluder_pcd: Optional[torch.Tensor] = None,
    occluder_inflate_px: int = 0,
):
    """
    occluder_pcd / occluder_inflate_px: optional second point set (e.g. wall distractors) rasterized
    with a much larger inflate_px and composited via a per-pixel min with the main depth, so sparse
    occluders reliably block rays behind them without needing to dilate (and blur) the whole scene.
    """
    if cam_spec_dict is None:
        raise ValueError("cam_spec_dict must be provided and must contain H/W/intrinsics/near_m/far_m.")

    batch_size = pcd.shape[0]
    device = pcd.device

    if use_compile:
        renderer = get_compiled_renderer_fixed_shapes(
            cam_spec_dict=cam_spec_dict,
            inflate_px=inflate_px,
            clip_mode=clip_mode,
            jitter_mode=jitter_mode,
            compile_mode=compile_mode,
            blur_kernel_px=blur_kernel_px,
            blur_sigma_px=blur_sigma_px,
            occluder_inflate_px=occluder_inflate_px,
        )
        if occluder_inflate_px > 0:
            _, pcd_world, _ = renderer(pcd, camera_pose, jitter_std_m, occluder_pcd)
        else:
            _, pcd_world, _ = renderer(pcd, camera_pose, jitter_std_m)
    else:
        _, pcd_world, _ = render_points_to_world_grid_from_pose(
            pcd,
            camera_pose,
            cam_spec_dict=cam_spec_dict,
            inflate_px=inflate_px,
            clip_mode=clip_mode,
            jitter_std_m=jitter_std_m,
            jitter_mode=jitter_mode,
            blur_kernel_px=blur_kernel_px,
            blur_sigma_px=blur_sigma_px,
            occluder_pcd=occluder_pcd,
            occluder_inflate_px=occluder_inflate_px,
        )

    rendered_pcd = pcd_world.view(batch_size, -1, 3)
    num_total_points = rendered_pcd.shape[1]
    rendered_pcd = shuffle_pcd(rendered_pcd)

    nan_mask = torch.isnan(rendered_pcd).any(dim=-1)
    sort_key = nan_mask.int()
    sort_idx = torch.argsort(sort_key, dim=-1)
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_total_points)
    sorted_pcds = rendered_pcd[batch_idx, sort_idx]

    avg_num_valid_points = num_total_points - nan_mask.sum().float() / float(batch_size)
    min_num_valid_points = (num_total_points - nan_mask.sum(dim=-1)).min().float()
    logs = {
        "sim_depth_cam_render/avg_num_valid_points": float(avg_num_valid_points.item()),
        "sim_depth_cam_render/min_num_valid_points": float(min_num_valid_points.item()),
    }

    pcd_nan_padding = sorted_pcds[:, :num_points]
    return pcd_nan_padding, logs


def render_depth_roundtrip_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,
    num_points: int,
    cam_spec_dict: Optional[Dict] = None,
    inflate_px: int = 0,
    clip_mode: str = "post",
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
    occluder_pcd: Optional[torch.Tensor] = None,
    occluder_inflate_px: int = 0,
    blur_kernel_px: int = 0,
    blur_sigma_px: float = 0.0,
):
    """True projection round-trip: 3D points -> 2D depth image -> 3D points.

    Two steps, nothing else (this does NOT go through the simulate_depth_cam_render_from_pose ray-cast
    wrapper -- it calls the same two primitives ``scripts/tools/render_depth_roundtrip_viser.py`` uses):

        PROJECT     : ``rasterize_depth_zbuffer_from_pose`` z-buffers the 3D points into a 2D depth
                      image (nearest surface per pixel), plus the ``occluder_pcd`` anti-penetration
                      pass (``occluder_inflate_px``) so nothing leaks through the door/walls.
        BACK-PROJECT: ``backproject_depth_to_world_from_pose`` unprojects every valid pixel back to a
                      3D world point.

    ``blur_kernel_px`` / ``blur_sigma_px`` optionally apply the RealSense-style edge-bleeding spatial
    blur (``apply_depth_spatial_blur``) to the depth image before back-projection: each valid pixel is
    replaced by a (Gaussian if sigma>0, else box) weighted average of its valid neighbours. Flat
    surfaces are unchanged, but thin features (the door handle) bleed into the door/plate behind them,
    so a crisp lever renders as an indistinct bump -- matching what a real depth camera returns.
    ``blur_kernel_px <= 1`` disables it (default), so this is a no-op unless the cfg turns it on.
    ``inflate_px`` is the main-pass z-buffer dilation (0 = plain round-trip: each point -> one pixel ->
    back). The variable-length valid points are then shuffled and NaN-padded to a fixed ``num_points``
    so the policy gets a constant-shape tensor. ``use_compile`` runs the exact same project+back-project
    (and blur) fused via torch.compile for speed. All knobs come from the caller (cfg), none hardcoded.
    """
    if cam_spec_dict is None:
        raise ValueError("cam_spec_dict must be provided and must contain H/W/intrinsics/near_m/far_m.")
    batch_size = pcd.shape[0]
    device = pcd.device

    if use_compile:
        # Same two ops (rasterize z-buffer -> back-project), fused into one compiled kernel. Jitter is
        # still compiled out (std 0); the optional spatial blur is baked in when blur_kernel_px > 1.
        renderer = get_compiled_renderer_fixed_shapes(
            cam_spec_dict=cam_spec_dict,
            inflate_px=inflate_px,
            clip_mode=clip_mode,
            jitter_mode="xyz",
            compile_mode=compile_mode,
            blur_kernel_px=blur_kernel_px,
            blur_sigma_px=blur_sigma_px,
            occluder_inflate_px=occluder_inflate_px,
        )
        if occluder_inflate_px > 0:
            _, pcd_world, _ = renderer(pcd, camera_pose, 0.0, occluder_pcd)
        else:
            _, pcd_world, _ = renderer(pcd, camera_pose, 0.0)
    else:
        # PROJECT 3D -> 2D depth image ...
        depth, intr = rasterize_depth_zbuffer_from_pose(
            pcd,
            camera_pose,
            cam_spec_dict=cam_spec_dict,
            inflate_px=inflate_px,
            clip_mode=clip_mode,
            occluder_pcd=occluder_pcd,
            occluder_inflate_px=occluder_inflate_px,
        )
        # ... optional RealSense-style edge-bleeding blur (thin features smear into a bump) ...
        if int(blur_kernel_px) > 1:
            kernel2d, pad = build_depth_blur_kernel2d(blur_kernel_px, blur_sigma_px, depth.device, depth.dtype)
            depth = apply_depth_spatial_blur(depth, kernel2d, pad)
        # ... BACK-PROJECT 2D -> 3D world points.
        pcd_world, _ = backproject_depth_to_world_from_pose(depth, camera_pose, intr)

    # Fixed-N packing: flatten the pixels, shuffle, push NaN (invalid pixels) to the end, keep num_points.
    rendered_pcd = shuffle_pcd(pcd_world.view(batch_size, -1, 3))
    num_total_points = rendered_pcd.shape[1]
    nan_mask = torch.isnan(rendered_pcd).any(dim=-1)
    sort_idx = torch.argsort(nan_mask.int(), dim=-1)
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_total_points)
    sorted_pcds = rendered_pcd[batch_idx, sort_idx]

    avg_num_valid_points = num_total_points - nan_mask.sum().float() / float(batch_size)
    logs = {"depth_roundtrip/avg_num_valid_points": float(avg_num_valid_points.item())}
    return sorted_pcds[:, :num_points], logs


_lidar_dir_cache: dict = {}
_lidar_compiled_cache: dict = {}


def rotmat_from_quat_xyzw(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    q: (...,4) in xyzw
    Returns: (...,3,3) rotation matrix mapping local(frame) -> world.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    qx, qy, qz, qw = q.unbind(-1)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R00 = 1.0 - 2.0 * (yy + zz)
    R01 = 2.0 * (xy - wz)
    R02 = 2.0 * (xz + wy)

    R10 = 2.0 * (xy + wz)
    R11 = 1.0 - 2.0 * (xx + zz)
    R12 = 2.0 * (yz - wx)

    R20 = 2.0 * (xz - wy)
    R21 = 2.0 * (yz + wx)
    R22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1),
        ],
        dim=-2,
    )


def _get_lidar_dir_flat(Hp: int, Wp: int, device, dtype) -> torch.Tensor:
    """
    Returns dir_flat: (K,3) LiDAR-frame unit directions for bin centers.
    Hemisphere:
      theta in [0, pi/2] from +Z (0 = +Z, pi/2 = XY plane)
      phi   in [-pi, pi) around +Z
    """
    key = (int(Hp), int(Wp), device, dtype)
    if key in _lidar_dir_cache:
        return _lidar_dir_cache[key]

    iu_c = (torch.arange(Wp, device=device, dtype=dtype) + 0.5) / float(Wp)
    iv_c = (torch.arange(Hp, device=device, dtype=dtype) + 0.5) / float(Hp)

    phi = iu_c * (2.0 * math.pi) - math.pi
    theta = iv_c * (0.5 * math.pi)

    theta2d = theta[:, None]
    phi2d = phi[None, :]

    sin_t = torch.sin(theta2d)
    cos_t = torch.cos(theta2d)
    cos_p = torch.cos(phi2d)
    sin_p = torch.sin(phi2d)

    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t.expand_as(x)

    dir_hw3 = torch.stack([x, y, z], dim=-1)
    dir_flat = dir_hw3.reshape(Hp * Wp, 3).contiguous()
    _lidar_dir_cache[key] = dir_flat
    return dir_flat


def _lidar_neighbor_min_pool(range_img: torch.Tensor, radius: int) -> torch.Tensor:
    """Min range within a (2*radius+1) bin neighborhood, wrapping circularly in azimuth and
    edge-repeating in polar (same padding convention as the suppress_bins cleanup below). Used both
    to fill gaps (occluder dilation) and to find a comparison neighbor (suppress_bins cleanup)."""
    if radius <= 0:
        return range_img
    s = int(radius)
    k = int(2 * s + 1)
    neg = -range_img.unsqueeze(1)
    neg = torch.cat([neg[..., -s:], neg, neg[..., :s]], dim=-1)
    top = neg[:, :, 0:1, :].expand(-1, -1, s, -1)
    bot = neg[:, :, -1:, :].expand(-1, -1, s, -1)
    neg = torch.cat([top, neg, bot], dim=-2)
    pooled = F.max_pool2d(neg, kernel_size=(k, k), stride=1)
    return (-pooled).squeeze(1)


def _lidar_scatter_raw_range(
    pcd: torch.Tensor,
    R: torch.Tensor,
    Rt: torch.Tensor,
    o: torch.Tensor,
    num_azimuth: int,
    num_polar: int,
    near_m: float,
    far_val: float,
) -> torch.Tensor:
    """Per-bin min-range scatter, before any suppress/fill neighborhood pooling."""
    B = pcd.shape[0]
    device, dtype = pcd.device, pcd.dtype
    finite_input = torch.isfinite(pcd).all(dim=-1)
    pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)

    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    rel_w = pcd - o[:, None, :]
    rel_l = torch.matmul(Rt[:, None, :, :], rel_w[..., None]).squeeze(-1)
    x, y, z = rel_l[..., 0], rel_l[..., 1], rel_l[..., 2]

    r = torch.sqrt((x * x + y * y + z * z).clamp_min(1e-24))

    inside = finite_input & (z > 0.0)
    if near_m is not None and near_m > 0.0:
        inside = inside & (r >= float(near_m))
    inside = inside & (r <= far_val)

    phi = torch.atan2(y, x)
    u = (phi + math.pi) / (2.0 * math.pi)
    u = u - torch.floor(u)

    zr = (z / r.clamp_min(1e-12)).clamp(0.0, 1.0)
    theta = torch.acos(zr)
    v = theta / (0.5 * math.pi)

    iu = torch.floor(u * float(Wp)).clamp(0, Wp - 1).long()
    iv = torch.floor(v * float(Hp)).clamp(0, Hp - 1).long()
    pix = iv * Wp + iu

    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("Tensor.scatter_reduce_ not found. Need PyTorch >= 1.12 (PyTorch 2.x is fine).")

    inf = torch.full((), float("inf"), device=device, dtype=dtype)
    r_masked = torch.where(inside, r, inf)

    range_flat = torch.full((B, K), float("inf"), device=device, dtype=dtype)
    range_flat.scatter_reduce_(dim=1, index=pix, src=r_masked, reduce="amin", include_self=True)
    return range_flat.view(B, Hp, Wp)


@torch.no_grad()
def render_lidar_bins_to_world_from_pose_fast(
    pcd: torch.Tensor,
    lidar_pose: torch.Tensor,
    num_azimuth: int,
    num_polar: int,
    near_m: float,
    far_m: Optional[float],
    suppress_bins: int,
    occlusion_eps_m: float,
    occlusion_eps_rel: float,
    jitter_std_m: float,
    occluder_pcd: Optional[torch.Tensor] = None,
    occluder_fill_bins: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pts_w:      (B,K,3) world-frame points with NaNs for empty bins
      valid_flat: (B,K) bool

    occluder_pcd / occluder_fill_bins: an optional second point set (e.g. wall distractors) scanned
    into its own range image and neighbor-min-filled over a (typically much larger) bin radius, then
    composited into the main range image via a per-bin min. This is the lidar analog of
    rasterize_depth_zbuffer_from_pose's occluder_pcd/occluder_inflate_px: it closes gaps a sparse
    occluder leaves in the fine per-bin scatter without touching suppress_bins' edge-cleanup role.
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    assert lidar_pose.ndim == 2 and lidar_pose.shape[-1] == 7

    device, dtype = pcd.device, pcd.dtype
    far_val = float("inf") if far_m is None else float(far_m)

    o = lidar_pose[:, 0:3]
    q = lidar_pose[:, 3:7]
    R = rotmat_from_quat_xyzw(q).to(dtype=dtype)
    Rt = R.transpose(-1, -2)

    range_img = _lidar_scatter_raw_range(pcd, R, Rt, o, num_azimuth, num_polar, near_m, far_val)

    if occluder_pcd is not None and occluder_fill_bins > 0 and occluder_pcd.shape[1] > 0:
        occluder_range = _lidar_scatter_raw_range(occluder_pcd, R, Rt, o, num_azimuth, num_polar, near_m, far_val)
        occluder_range = _lidar_neighbor_min_pool(occluder_range, int(occluder_fill_bins))
        range_img = torch.minimum(range_img, occluder_range)

    if suppress_bins > 0:
        neighbor_min = _lidar_neighbor_min_pool(range_img, int(suppress_bins))
        eps = float(occlusion_eps_m) + float(occlusion_eps_rel) * neighbor_min.clamp_min(0.0)
        inf = torch.full((), float("inf"), device=device, dtype=dtype)
        suppress = torch.isfinite(range_img) & torch.isfinite(neighbor_min) & (range_img > neighbor_min + eps)
        range_img = torch.where(suppress, inf, range_img)

    Hp = int(num_polar)
    Wp = int(num_azimuth)
    range_flat = range_img.view(range_img.shape[0], Hp * Wp)
    valid_flat = torch.isfinite(range_flat)

    dir_flat = _get_lidar_dir_flat(Hp, Wp, device, dtype)
    pts_l = dir_flat.unsqueeze(0) * range_flat.unsqueeze(-1)
    pts_w = torch.matmul(R, pts_l.transpose(1, 2)).transpose(1, 2) + o[:, None, :]

    nan = torch.full((), float("nan"), device=device, dtype=dtype)
    pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w, nan)

    if jitter_std_m > 0.0:
        noise = torch.randn_like(pts_w) * float(jitter_std_m)
        pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w + noise, pts_w)

    return pts_w, valid_flat


def get_compiled_lidar_renderer_fixed_shapes(
    num_azimuth: int,
    num_polar: int,
    near_m: float,
    far_m: Optional[float],
    suppress_bins: int,
    occlusion_eps_m: float,
    occlusion_eps_rel: float,
    compile_mode: str = "max-autotune",
    occluder_fill_bins: int = 0,
):
    """
    Returns compiled callable:
        fn(pcd: (B,N,3), lidar_pose: (B,7), jitter_std: scalar tensor[, occluder_pcd: (B,M,3)])
            -> (pts_w: (B,K,3), valid_flat: (B,K))
    The occluder_pcd argument is only accepted (and required) when occluder_fill_bins > 0; see
    render_lidar_bins_to_world_from_pose_fast for what it does.

    IMPORTANT: FIXED shapes (B and N do not change), and all params passed here remain constant.
    """
    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    far_val = float("inf") if far_m is None else float(far_m)
    do_suppress = int(suppress_bins) > 0
    s = int(suppress_bins)
    k = int(2 * s + 1)
    has_occluder = int(occluder_fill_bins) > 0
    occ_s = int(occluder_fill_bins)
    occ_k = int(2 * occ_s + 1)

    def _get_or_build(device, dtype):
        key = (
            Hp,
            Wp,
            float(near_m),
            float(far_val),
            int(suppress_bins),
            float(occlusion_eps_m),
            float(occlusion_eps_rel),
            compile_mode,
            int(occluder_fill_bins),
            device,
            dtype,
        )
        if key in _lidar_compiled_cache:
            return _lidar_compiled_cache[key]

        dir_flat = _get_lidar_dir_flat(Hp, Wp, device, dtype)

        def _scatter_range(points, R, Rt, o):
            finite_input = torch.isfinite(points).all(dim=-1)
            points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
            Bc = points.shape[0]

            rel_w = points - o[:, None, :]
            rel_l = torch.matmul(Rt[:, None, :, :], rel_w[..., None]).squeeze(-1)
            x = rel_l[..., 0]
            y = rel_l[..., 1]
            z = rel_l[..., 2]

            r = torch.sqrt((x * x + y * y + z * z).clamp_min(1e-24))

            inside = finite_input & (z > 0.0)
            if near_m is not None and near_m > 0.0:
                inside = inside & (r >= float(near_m))
            inside = inside & (r <= float(far_val))

            phi = torch.atan2(y, x)
            u = (phi + math.pi) / (2.0 * math.pi)
            u = u - torch.floor(u)

            zr = (z / r.clamp_min(1e-12)).clamp(0.0, 1.0)
            theta = torch.acos(zr)
            v = theta / (0.5 * math.pi)

            iu = torch.floor(u * float(Wp)).clamp(0, Wp - 1).long()
            iv = torch.floor(v * float(Hp)).clamp(0, Hp - 1).long()
            pix = iv * Wp + iu

            inf_local = torch.full((), float("inf"), device=points.device, dtype=points.dtype)
            r_masked = torch.where(inside, r, inf_local)

            range_flat = torch.full((Bc, K), float("inf"), device=points.device, dtype=points.dtype)
            range_flat.scatter_reduce_(dim=1, index=pix, src=r_masked, reduce="amin", include_self=True)
            return range_flat.view(Bc, Hp, Wp)

        def _neighbor_min_pool(range_img, radius, kernel):
            if radius <= 0:
                return range_img
            neg = -range_img.unsqueeze(1)
            neg = torch.cat([neg[..., -radius:], neg, neg[..., :radius]], dim=-1)
            top = neg[:, :, 0:1, :].expand(-1, -1, radius, -1)
            bot = neg[:, :, -1:, :].expand(-1, -1, radius, -1)
            neg = torch.cat([top, neg, bot], dim=-2)
            pooled = F.max_pool2d(neg, kernel_size=kernel, stride=1)
            return (-pooled).squeeze(1)

        def _finalize(range_img, R, o, jitter_std, pcd_device, pcd_dtype):
            inf = torch.full((), float("inf"), device=pcd_device, dtype=pcd_dtype)
            if do_suppress:
                neighbor_min = _neighbor_min_pool(range_img, s, (k, k))
                eps = float(occlusion_eps_m) + float(occlusion_eps_rel) * neighbor_min.clamp_min(0.0)
                suppress = torch.isfinite(range_img) & torch.isfinite(neighbor_min) & (range_img > neighbor_min + eps)
                range_img = torch.where(suppress, inf, range_img)
            range_flat = range_img.reshape(range_img.shape[0], K)

            valid_flat = torch.isfinite(range_flat)

            pts_l = dir_flat.unsqueeze(0) * range_flat.unsqueeze(-1)
            pts_w = torch.matmul(R, pts_l.transpose(1, 2)).transpose(1, 2) + o[:, None, :]

            nan = torch.full((), float("nan"), device=pcd_device, dtype=pcd_dtype)
            pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w, nan)

            noise = torch.randn_like(pts_w) * jitter_std
            pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w + noise, pts_w)
            return pts_w, valid_flat

        if has_occluder:

            @torch.no_grad()
            def _compiled_fn(
                pcd: torch.Tensor,
                lidar_pose: torch.Tensor,
                jitter_std: torch.Tensor,
                occluder_pcd: torch.Tensor,
            ):
                o = lidar_pose[:, 0:3]
                q = lidar_pose[:, 3:7]
                R = rotmat_from_quat_xyzw(q).to(dtype=pcd.dtype)
                Rt = R.transpose(-1, -2)

                range_img = _scatter_range(pcd, R, Rt, o)
                occluder_range = _scatter_range(occluder_pcd, R, Rt, o)
                occluder_range = _neighbor_min_pool(occluder_range, occ_s, (occ_k, occ_k))
                range_img = torch.minimum(range_img, occluder_range)

                return _finalize(range_img, R, o, jitter_std, pcd.device, pcd.dtype)

        else:

            @torch.no_grad()
            def _compiled_fn(pcd: torch.Tensor, lidar_pose: torch.Tensor, jitter_std: torch.Tensor):
                o = lidar_pose[:, 0:3]
                q = lidar_pose[:, 3:7]
                R = rotmat_from_quat_xyzw(q).to(dtype=pcd.dtype)
                Rt = R.transpose(-1, -2)

                range_img = _scatter_range(pcd, R, Rt, o)

                return _finalize(range_img, R, o, jitter_std, pcd.device, pcd.dtype)

        compiled = torch.compile(_compiled_fn, mode=compile_mode, dynamic=False)
        _lidar_compiled_cache[key] = compiled
        return compiled

    def wrapper(
        pcd: torch.Tensor,
        lidar_pose: torch.Tensor,
        jitter_std_m: float,
        occluder_pcd: Optional[torch.Tensor] = None,
    ):
        fn = _get_or_build(pcd.device, pcd.dtype)
        jitter_std = torch.tensor(float(jitter_std_m), device=pcd.device, dtype=pcd.dtype)
        if has_occluder:
            if occluder_pcd is None:
                raise ValueError("occluder_fill_bins > 0 requires occluder_pcd to be provided.")
            return fn(pcd, lidar_pose, jitter_std, occluder_pcd)
        return fn(pcd, lidar_pose, jitter_std)

    return wrapper


@torch.no_grad()
def simulate_lidar_render_from_pose(
    pcd: torch.Tensor,
    lidar_pose: torch.Tensor,
    num_points: int = 10000,
    num_azimuth: int = 512,
    num_polar: int = 512,
    near_m: float = 0.1,
    far_m: Optional[float] = 30.0,
    suppress_bins: int = 2,
    occlusion_eps_m: float = 0.02,
    occlusion_eps_rel: float = 0.01,
    jitter_std_m: float = 0.0,
    shuffle: bool = True,
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
    occluder_pcd: Optional[torch.Tensor] = None,
    occluder_fill_bins: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      lidar_pcd: (B, num_points, 3) valid-first, NaN padded
      logs: dict

    occluder_pcd / occluder_fill_bins: optional second point set (e.g. wall distractors) filled over
    a much larger bin radius and composited via a per-bin min with the main range image, so sparse
    occluders reliably block returns behind them. See render_lidar_bins_to_world_from_pose_fast.
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    assert lidar_pose.ndim == 2 and lidar_pose.shape[-1] == 7

    B = pcd.shape[0]
    device = pcd.device

    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    if use_compile:
        renderer = get_compiled_lidar_renderer_fixed_shapes(
            num_azimuth=Wp,
            num_polar=Hp,
            near_m=float(near_m),
            far_m=None if far_m is None else float(far_m),
            suppress_bins=int(suppress_bins),
            occlusion_eps_m=float(occlusion_eps_m),
            occlusion_eps_rel=float(occlusion_eps_rel),
            compile_mode=compile_mode,
            occluder_fill_bins=occluder_fill_bins,
        )
        if occluder_fill_bins > 0:
            pts_w_flat, valid_flat = renderer(pcd, lidar_pose, jitter_std_m, occluder_pcd)
        else:
            pts_w_flat, valid_flat = renderer(pcd, lidar_pose, jitter_std_m)
    else:
        pts_w_flat, valid_flat = render_lidar_bins_to_world_from_pose_fast(
            pcd=pcd,
            lidar_pose=lidar_pose,
            num_azimuth=Wp,
            num_polar=Hp,
            near_m=float(near_m),
            far_m=None if far_m is None else float(far_m),
            suppress_bins=int(suppress_bins),
            occlusion_eps_m=float(occlusion_eps_m),
            occlusion_eps_rel=float(occlusion_eps_rel),
            jitter_std_m=float(jitter_std_m),
            occluder_pcd=occluder_pcd,
            occluder_fill_bins=occluder_fill_bins,
        )

    select_count = min(int(num_points), int(K))
    finite_mask = torch.isfinite(pts_w_flat).all(dim=-1)
    valid_select_mask = valid_flat & finite_mask
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, select_count)
    if shuffle:
        scores = torch.rand((B, K), device=device, dtype=pts_w_flat.dtype)
        scores = torch.where(valid_select_mask, scores + 1.0, scores - 1.0)
        scores = torch.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        select_idx = torch.topk(scores, k=select_count, dim=-1, sorted=True).indices
    else:
        select_idx = torch.argsort((~valid_select_mask).int(), dim=-1)[:, :select_count]
    lidar_pcd = pts_w_flat[batch_idx, select_idx]
    lidar_pcd = torch.where(
        torch.isfinite(lidar_pcd).all(dim=-1, keepdim=True),
        lidar_pcd,
        torch.full_like(lidar_pcd, float("nan")),
    )
    if select_count < int(num_points):
        pad = torch.full((B, int(num_points) - select_count, 3), float("nan"), dtype=lidar_pcd.dtype, device=device)
        lidar_pcd = torch.cat([lidar_pcd, pad], dim=1)

    num_valid_per_batch = valid_flat.sum(dim=-1)
    logs: Dict[str, float] = {
        "sim_lidar_render/avg_num_valid_points": float(num_valid_per_batch.float().mean().item()),
        "sim_lidar_render/min_num_valid_points": float(num_valid_per_batch.min().item()),
        "sim_lidar_render/num_rays": float(K),
        "sim_lidar_render/suppress_bins": float(suppress_bins),
    }

    return lidar_pcd, logs
