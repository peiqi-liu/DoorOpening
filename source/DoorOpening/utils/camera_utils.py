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
    inside = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front
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
def render_points_to_world_grid_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,
    cam_spec_dict: Dict,
    inflate_px: int = 0,
    clip_mode: str = "post",
    jitter_std_m: float = 0.0,
    jitter_mode: str = "xyz",
):
    depth, intr = rasterize_depth_zbuffer_from_pose(
        pcd,
        camera_pose,
        cam_spec_dict=cam_spec_dict,
        inflate_px=inflate_px,
        clip_mode=clip_mode,
    )
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
        key = (H, W, near_m, far_val, int(inflate_px), clip_mode, jitter_mode, compile_mode, intrinsics_key, device, dtype)
        if key in _compiled_cache:
            return _compiled_cache[key]

        fx, fy, cx, cy = _get_render_intrinsics(cam_spec_dict, device, dtype)
        u_base, v_base = _get_uv_base(H, W, device, dtype)
        PAD = (int(inflate_px), int(inflate_px), int(inflate_px), int(inflate_px))
        k = int(2 * inflate_px + 1)
        KERNEL = (k, k)

        @torch.no_grad()
        def _compiled_fn(pcd: torch.Tensor, camera_pose: torch.Tensor, jitter_std: torch.Tensor):
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
            inside = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front
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
