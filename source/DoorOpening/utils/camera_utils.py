import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


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


@torch.no_grad()
def rasterize_depth_zbuffer_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,
    cam_spec_dict: Dict,
    inflate_px: int = 0,
    clip_mode: str = "post",
):
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
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
    intr = (fx, fy, cx, cy)

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
    depth = min_depth.view(B, H, W)

    if inflate_px > 0:
        neg = -depth.unsqueeze(1)
        neg = F.pad(
            neg,
            (inflate_px, inflate_px, inflate_px, inflate_px),
            mode="constant",
            value=float("-inf"),
        )
        k = 2 * inflate_px + 1
        pooled_neg = F.max_pool2d(neg, kernel_size=(k, k), stride=1)
        depth = (-pooled_neg).squeeze(1)

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
):
    depth, intr = rasterize_depth_zbuffer_from_pose(
        pcd,
        camera_pose,
        cam_spec_dict=cam_spec_dict,
        inflate_px=inflate_px,
        clip_mode=clip_mode,
    )
    # Spatial blur of the depth image before unprojection (mimics the RealSense SDK spatial filter).
    if int(blur_kernel_px) > 1:
        kernel2d, pad = build_depth_blur_kernel2d(blur_kernel_px, blur_sigma_px, depth.device, depth.dtype)
        depth = apply_depth_spatial_blur(depth, kernel2d, pad)
    pcd_world, valid = backproject_depth_to_world_from_pose(depth, camera_pose, intr)

    if jitter_std_m > 0.0:
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
            raise ValueError(f"Unknown jitter_mode '{jitter_mode}'. Use 'tangent' or 'xyz'.")

    return depth, pcd_world, valid


def get_compiled_renderer_fixed_shapes(
    cam_spec_dict: Dict,
    inflate_px: int = 0,
    clip_mode: str = "post",
    jitter_mode: str = "xyz",
    compile_mode: str = "max-autotune",
    blur_kernel_px: int = 0,
    blur_sigma_px: float = 0.0,
):
    if jitter_mode.lower() != "xyz":
        raise ValueError("Compiled renderer supports jitter_mode='xyz' only.")

    H = int(cam_spec_dict["H"])
    W = int(cam_spec_dict["W"])
    near_m = float(cam_spec_dict["near_m"])
    far_m = cam_spec_dict["far_m"]
    far_val = float("inf") if far_m is None else float(far_m)
    clip_pre = clip_mode == "pre"
    intrinsics = cam_spec_dict.get("intrinsics")
    if intrinsics is None:
        raise ValueError("cam_spec_dict must provide an 'intrinsics' matrix.")
    if isinstance(intrinsics, torch.Tensor):
        intrinsics_key = tuple(float(v) for v in intrinsics.detach().cpu().reshape(-1).tolist())
    else:
        intrinsics_key = tuple(float(v) for v in torch.as_tensor(intrinsics).reshape(-1).tolist())

    def _get_or_build(device, dtype):
        key = (H, W, near_m, far_val, int(inflate_px), clip_mode, jitter_mode, compile_mode,
               int(blur_kernel_px), float(blur_sigma_px), intrinsics_key, device, dtype)
        if key in _compiled_cache:
            return _compiled_cache[key]

        fx, fy, cx, cy = _get_render_intrinsics(cam_spec_dict, device, dtype)
        u_base, v_base = _get_uv_base(H, W, device, dtype)
        PAD = (int(inflate_px), int(inflate_px), int(inflate_px), int(inflate_px))
        k = int(2 * inflate_px + 1)
        KERNEL = (k, k)
        if int(blur_kernel_px) > 1:
            BLUR_KERNEL2D, BLUR_PAD = build_depth_blur_kernel2d(blur_kernel_px, blur_sigma_px, device, dtype)
        else:
            BLUR_KERNEL2D, BLUR_PAD = None, 0

        @torch.no_grad()
        def _compiled_fn(pcd: torch.Tensor, camera_pose: torch.Tensor, jitter_std: torch.Tensor):
            finite_input = torch.isfinite(pcd).all(dim=-1)
            pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)
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
            if clip_pre:
                inside = inside & (z >= near_m) & (z <= far_val)

            iu = u_pix.floor().clamp(0, W - 1).long()
            iv = v_pix.floor().clamp(0, H - 1).long()
            pix = iv * W + iu

            inf = torch.full((), float("inf"), device=pcd.device, dtype=pcd.dtype)
            z_masked = torch.where(inside, z, inf)
            K = H * W
            B = pcd.shape[0]
            min_depth = torch.full((B, K), float("inf"), device=pcd.device, dtype=pcd.dtype)
            min_depth.scatter_reduce_(dim=1, index=pix, src=z_masked, reduce="amin", include_self=True)
            depth = min_depth.view(B, H, W)

            if inflate_px > 0:
                neg = -depth.unsqueeze(1)
                neg = F.pad(neg, PAD, mode="constant", value=float("-inf"))
                pooled_neg = F.max_pool2d(neg, kernel_size=KERNEL, stride=1)
                depth = (-pooled_neg).squeeze(1)

            depth = torch.where(depth >= near_m, depth, inf)
            depth = torch.where(depth <= far_val, depth, inf)

            if BLUR_KERNEL2D is not None:
                depth = apply_depth_spatial_blur(depth, BLUR_KERNEL2D, BLUR_PAD)

            u = u_base.expand(B, H, W)
            v = v_base.expand(B, H, W)
            valid = torch.isfinite(depth)
            zz = depth
            xx = (u - cx) / fx * zz
            yy = (v - cy) / fy * zz

            nan = torch.full((), float("nan"), device=pcd.device, dtype=pcd.dtype)
            xx = torch.where(valid, xx, nan)
            yy = torch.where(valid, yy, nan)
            zz = torch.where(valid, zz, nan)

            uB = u_hat.view(B, 1, 1, 3)
            wB = w_hat.view(B, 1, 1, 3)
            vB = v_hat.view(B, 1, 1, 3)
            oB = cam_pos.view(B, 1, 1, 3)
            pcd_world = oB + xx[..., None] * uB + yy[..., None] * wB + zz[..., None] * vB

            noise = torch.randn_like(pcd_world) * jitter_std
            pcd_world = torch.where(valid[..., None], pcd_world + noise, pcd_world)
            return depth, pcd_world, valid

        compiled = torch.compile(_compiled_fn, mode=compile_mode, dynamic=False)
        _compiled_cache[key] = compiled
        return compiled

    def wrapper(pcd: torch.Tensor, camera_pose: torch.Tensor, jitter_std_m: float):
        compiled = _get_or_build(pcd.device, pcd.dtype)
        jitter_std = torch.tensor(float(jitter_std_m), device=pcd.device, dtype=pcd.dtype)
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
):
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
        )
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pts_w:      (B,K,3) world-frame points with NaNs for empty bins
      valid_flat: (B,K) bool
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    assert lidar_pose.ndim == 2 and lidar_pose.shape[-1] == 7

    B, _, _ = pcd.shape
    device, dtype = pcd.device, pcd.dtype
    finite_input = torch.isfinite(pcd).all(dim=-1)
    pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)

    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    o = lidar_pose[:, 0:3]
    q = lidar_pose[:, 3:7]
    R = rotmat_from_quat_xyzw(q).to(dtype=dtype)
    Rt = R.transpose(-1, -2)

    rel_w = pcd - o[:, None, :]
    rel_l = torch.matmul(Rt[:, None, :, :], rel_w[..., None]).squeeze(-1)
    x, y, z = rel_l[..., 0], rel_l[..., 1], rel_l[..., 2]

    r = torch.sqrt((x * x + y * y + z * z).clamp_min(1e-24))

    inside = finite_input & (z > 0.0)
    if near_m is not None and near_m > 0.0:
        inside = inside & (r >= float(near_m))
    far_val = float("inf") if far_m is None else float(far_m)
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
    range_img = range_flat.view(B, Hp, Wp)

    if suppress_bins > 0:
        s = int(suppress_bins)
        k = int(2 * s + 1)

        neg = -range_img.unsqueeze(1)
        neg = torch.cat([neg[..., -s:], neg, neg[..., :s]], dim=-1)

        top = neg[:, :, 0:1, :].expand(-1, -1, s, -1)
        bot = neg[:, :, -1:, :].expand(-1, -1, s, -1)
        neg = torch.cat([top, neg, bot], dim=-2)

        pooled = F.max_pool2d(neg, kernel_size=(k, k), stride=1)
        neighbor_min = (-pooled).squeeze(1)

        eps = float(occlusion_eps_m) + float(occlusion_eps_rel) * neighbor_min.clamp_min(0.0)
        suppress = torch.isfinite(range_img) & torch.isfinite(neighbor_min) & (range_img > neighbor_min + eps)

        range_img = torch.where(suppress, inf, range_img)
        range_flat = range_img.view(B, K)

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
):
    """
    Returns compiled callable:
        fn(pcd: (B,N,3), lidar_pose: (B,7), jitter_std: scalar tensor) -> (pts_w: (B,K,3), valid_flat: (B,K))

    IMPORTANT: FIXED shapes (B and N do not change), and all params passed here remain constant.
    """
    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    far_val = float("inf") if far_m is None else float(far_m)
    do_suppress = int(suppress_bins) > 0
    s = int(suppress_bins)
    k = int(2 * s + 1)

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
            device,
            dtype,
        )
        if key in _lidar_compiled_cache:
            return _lidar_compiled_cache[key]

        dir_flat = _get_lidar_dir_flat(Hp, Wp, device, dtype)

        @torch.no_grad()
        def _compiled_fn(pcd: torch.Tensor, lidar_pose: torch.Tensor, jitter_std: torch.Tensor):
            finite_input = torch.isfinite(pcd).all(dim=-1)
            pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)
            B = pcd.shape[0]

            o = lidar_pose[:, 0:3]
            q = lidar_pose[:, 3:7]
            R = rotmat_from_quat_xyzw(q).to(dtype=pcd.dtype)
            Rt = R.transpose(-1, -2)

            rel_w = pcd - o[:, None, :]
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

            inf = torch.full((), float("inf"), device=pcd.device, dtype=pcd.dtype)
            r_masked = torch.where(inside, r, inf)

            range_flat = torch.full((B, K), float("inf"), device=pcd.device, dtype=pcd.dtype)
            range_flat.scatter_reduce_(dim=1, index=pix, src=r_masked, reduce="amin", include_self=True)
            range_img = range_flat.view(B, Hp, Wp)

            if do_suppress:
                neg = -range_img.unsqueeze(1)

                neg = torch.cat([neg[..., -s:], neg, neg[..., :s]], dim=-1)

                top = neg[:, :, 0:1, :].expand(-1, -1, s, -1)
                bot = neg[:, :, -1:, :].expand(-1, -1, s, -1)
                neg = torch.cat([top, neg, bot], dim=-2)

                pooled = F.max_pool2d(neg, kernel_size=(k, k), stride=1)
                neighbor_min = (-pooled).squeeze(1)

                eps = float(occlusion_eps_m) + float(occlusion_eps_rel) * neighbor_min.clamp_min(0.0)
                suppress = torch.isfinite(range_img) & torch.isfinite(neighbor_min) & (range_img > neighbor_min + eps)
                range_img = torch.where(suppress, inf, range_img)
                range_flat = range_img.view(B, K)

            valid_flat = torch.isfinite(range_flat)

            pts_l = dir_flat.unsqueeze(0) * range_flat.unsqueeze(-1)
            pts_w = torch.matmul(R, pts_l.transpose(1, 2)).transpose(1, 2) + o[:, None, :]

            nan = torch.full((), float("nan"), device=pcd.device, dtype=pcd.dtype)
            pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w, nan)

            noise = torch.randn_like(pts_w) * jitter_std
            pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w + noise, pts_w)

            return pts_w, valid_flat

        compiled = torch.compile(_compiled_fn, mode=compile_mode, dynamic=False)
        _lidar_compiled_cache[key] = compiled
        return compiled

    def wrapper(pcd: torch.Tensor, lidar_pose: torch.Tensor, jitter_std_m: float):
        fn = _get_or_build(pcd.device, pcd.dtype)
        jitter_std = torch.tensor(float(jitter_std_m), device=pcd.device, dtype=pcd.dtype)
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      lidar_pcd: (B, num_points, 3) valid-first, NaN padded
      logs: dict
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
        )
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
