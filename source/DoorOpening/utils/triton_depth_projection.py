"""Optional fused point-cloud projection for the distillation depth renderer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _project_depth(
    points, poses, right, down, forward, depth_out, split_out,
    num_points: tl.constexpr, height: tl.constexpr, width: tl.constexpr,
    fx: tl.constexpr, fy: tl.constexpr, cx: tl.constexpr, cy: tl.constexpr,
    near: tl.constexpr, far: tl.constexpr, clip_pre: tl.constexpr,
    split_index: tl.constexpr, has_split: tl.constexpr, block: tl.constexpr,
):
    env = tl.program_id(0)
    point = tl.program_id(1) * block + tl.arange(0, block)
    in_range = point < num_points
    offset = (env * num_points + point) * 3
    px = tl.load(points + offset, in_range, 0)
    py = tl.load(points + offset + 1, in_range, 0)
    pz = tl.load(points + offset + 2, in_range, 0)
    rx = px - tl.load(poses + env * 7)
    ry = py - tl.load(poses + env * 7 + 1)
    rz = pz - tl.load(poses + env * 7 + 2)
    depth = rx * tl.load(forward + env * 3) + ry * tl.load(forward + env * 3 + 1) + rz * tl.load(forward + env * 3 + 2)
    horizontal = rx * tl.load(right + env * 3) + ry * tl.load(right + env * 3 + 1) + rz * tl.load(right + env * 3 + 2)
    vertical = rx * tl.load(down + env * 3) + ry * tl.load(down + env * 3 + 1) + rz * tl.load(down + env * 3 + 2)
    inv_depth = 1.0 / tl.maximum(depth, 1.0e-12)
    col = fx * horizontal * inv_depth + cx
    row = fy * vertical * inv_depth + cy
    finite = (tl.abs(px) < float("inf")) & (tl.abs(py) < float("inf")) & (tl.abs(pz) < float("inf"))
    valid = in_range & finite & (depth > 0) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
    if clip_pre:
        valid = valid & (depth >= near) & (depth <= far)
    pixel = env * height * width + tl.floor(row).to(tl.int32) * width + tl.floor(col).to(tl.int32)
    if has_split:
        tl.atomic_min(depth_out + pixel, depth, valid & (point < split_index), sem="relaxed")
        tl.atomic_min(split_out + pixel, depth, valid & (point >= split_index), sem="relaxed")
    else:
        tl.atomic_min(depth_out + pixel, depth, valid, sem="relaxed")


@torch.no_grad()
def project_depth_triton(pcd, camera_pose, cam_spec_dict, clip_mode="post", source_split_idx=None):
    """Return the same raw z-buffer as ``_zbuffer_scatter_raw_depth``.

    The caller keeps dilation and post-projection range clipping in the existing
    PyTorch path. Inputs must be contiguous float32 CUDA tensors.
    """
    if pcd.device.type != "cuda" or pcd.dtype != torch.float32 or camera_pose.device != pcd.device:
        raise ValueError("Triton depth projection requires float32 CUDA points and poses on the same device")
    if pcd.ndim != 3 or pcd.shape[-1] != 3 or camera_pose.shape != (pcd.shape[0], 7):
        raise ValueError("Expected point clouds [B, N, 3] and camera poses [B, 7]")
    if clip_mode not in {"pre", "post"}:
        raise ValueError("clip_mode must be 'pre' or 'post'")
    if source_split_idx is not None and not 0 <= int(source_split_idx) <= pcd.shape[1]:
        raise ValueError("source_split_idx must lie within the point cloud")

    # Import lazily to keep Triton optional for non-distillation callers.
    from DoorOpening.utils.camera_utils import _camera_basis_from_pose_x_forward, _get_render_intrinsics

    pcd = pcd.contiguous()
    camera_pose = camera_pose.contiguous()
    batch, num_points, _ = pcd.shape
    height, width = int(cam_spec_dict["H"]), int(cam_spec_dict["W"])
    near = float(cam_spec_dict["near_m"])
    far = float("inf") if cam_spec_dict["far_m"] is None else float(cam_spec_dict["far_m"])
    fx, fy, cx, cy = (float(value) for value in _get_render_intrinsics(cam_spec_dict, pcd.device, pcd.dtype))
    right, down, forward = _camera_basis_from_pose_x_forward(camera_pose)
    first = torch.full((batch, height, width), float("inf"), device=pcd.device, dtype=pcd.dtype)
    second = torch.full_like(first, float("inf")) if source_split_idx is not None else first
    if num_points:
        _project_depth[(batch, triton.cdiv(num_points, 256))](
            pcd, camera_pose, right, down, forward, first, second,
            num_points, height, width, fx, fy, cx, cy, near, far,
            clip_mode == "pre", 0 if source_split_idx is None else int(source_split_idx),
            source_split_idx is not None, 256,
            enable_fp_fusion=False,
        )
    return (first, second) if source_split_idx is not None else first
