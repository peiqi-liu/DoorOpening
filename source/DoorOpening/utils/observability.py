from typing import Any

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul


def _to_batched_tensor(value: Any, batch_size: int, trailing_shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    """Convert ``value`` to a tensor with leading batch dimension ``batch_size``."""
    if value is None:
        raise ValueError("Expected a tensor-like value but got None.")

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)

    if tensor.ndim == len(trailing_shape):
        tensor = tensor.unsqueeze(0)

    expected_shape = (batch_size, *trailing_shape)
    if tensor.shape == expected_shape:
        return tensor
    if tensor.shape == (1, *trailing_shape):
        return tensor.expand(expected_shape)

    raise ValueError(f"Expected shape {(1, *trailing_shape)} or {expected_shape}, got {tuple(tensor.shape)}")


def compose_world_pose(
    parent_pos_w: torch.Tensor,
    parent_quat_w: torch.Tensor,
    local_pos: torch.Tensor | None = None,
    local_quat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a child pose in world coordinates from a parent pose and local offset."""
    batch_size = parent_pos_w.shape[0]
    device = parent_pos_w.device
    dtype = parent_pos_w.dtype

    if local_pos is None:
        local_pos = torch.zeros((1, 3), device=device, dtype=dtype)
    if local_quat is None:
        local_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)

    local_pos = _to_batched_tensor(local_pos, batch_size, (3,), device, dtype)
    local_quat = _to_batched_tensor(local_quat, batch_size, (4,), device, dtype)

    child_pos_w = parent_pos_w + quat_apply(parent_quat_w, local_pos)
    child_quat_w = quat_mul(parent_quat_w, local_quat)
    return child_pos_w, child_quat_w


def make_handle_observation_points(handle_offsets_local: torch.Tensor, include_center: bool = True) -> torch.Tensor:
    """Return handle observation points from the precomputed local handle offsets."""
    if handle_offsets_local.ndim != 3 or handle_offsets_local.shape[-1] != 3:
        raise ValueError(
            "Expected handle_offsets_local to have shape [B, N, 3], "
            f"got {tuple(handle_offsets_local.shape)}"
        )

    if not include_center:
        return handle_offsets_local

    handle_center = handle_offsets_local.mean(dim=1, keepdim=True)
    return torch.cat([handle_offsets_local, handle_center], dim=1)


def world_points_to_camera(points_w: torch.Tensor, camera_pos_w: torch.Tensor, camera_quat_w: torch.Tensor) -> torch.Tensor:
    """Transform world-frame points into the camera frame."""
    if points_w.ndim != 3 or points_w.shape[-1] != 3:
        raise ValueError(f"Expected points_w to have shape [B, N, 3], got {tuple(points_w.shape)}")

    batch_size, num_points, _ = points_w.shape
    camera_pos_w = _to_batched_tensor(camera_pos_w, batch_size, (3,), points_w.device, points_w.dtype)
    camera_quat_w = _to_batched_tensor(camera_quat_w, batch_size, (4,), points_w.device, points_w.dtype)

    rel_points_w = points_w - camera_pos_w.unsqueeze(1)
    camera_quat_inv = quat_conjugate(camera_quat_w).unsqueeze(1).expand(-1, num_points, -1)
    return quat_apply(camera_quat_inv, rel_points_w)


def project_camera_points(points_c: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-frame points onto the image plane."""
    if points_c.ndim != 3 or points_c.shape[-1] != 3:
        raise ValueError(f"Expected points_c to have shape [B, N, 3], got {tuple(points_c.shape)}")

    batch_size = points_c.shape[0]
    intrinsics = _to_batched_tensor(intrinsics, batch_size, (3, 3), points_c.device, points_c.dtype)

    fx = intrinsics[:, 0, 0].unsqueeze(1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)

    depth = points_c[..., 2]
    safe_depth = torch.where(
        depth.abs() > torch.finfo(points_c.dtype).eps,
        depth,
        torch.full_like(depth, torch.finfo(points_c.dtype).eps),
    )

    u = fx * (points_c[..., 0] / safe_depth) + cx
    v = fy * (points_c[..., 1] / safe_depth) + cy
    return torch.stack((u, v), dim=-1), depth


def evaluate_points_observability(
    points_w: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    near_m: float = 0.0,
    far_m: float | None = None,
    depth_image: torch.Tensor | None = None,
    occlusion_tolerance_m: float = 0.03,
    pixel_margin: float = 0.0,
    min_visible_points: int = 1,
    require_all_points: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate whether world-frame target points are visible from a pinhole camera."""
    batch_size = points_w.shape[0]
    device = points_w.device
    dtype = points_w.dtype

    points_c = world_points_to_camera(points_w, camera_pos_w, camera_quat_w)
    pixel_coords, depth = project_camera_points(points_c, intrinsics)

    inside_depth = depth > float(near_m)
    if far_m is not None:
        inside_depth = inside_depth & (depth < float(far_m))

    inside_image = (
        (pixel_coords[..., 0] >= -float(pixel_margin))
        & (pixel_coords[..., 0] <= float(image_width - 1) + float(pixel_margin))
        & (pixel_coords[..., 1] >= -float(pixel_margin))
        & (pixel_coords[..., 1] <= float(image_height - 1) + float(pixel_margin))
    )
    inside_frustum = inside_depth & inside_image

    depth_match = torch.ones_like(inside_frustum, dtype=torch.bool, device=device)
    sampled_depth = torch.full_like(depth, float("nan"))

    if depth_image is not None:
        depth_image = depth_image if isinstance(depth_image, torch.Tensor) else torch.as_tensor(depth_image)
        depth_image = depth_image.to(device=device, dtype=dtype)
        if depth_image.ndim == 4 and depth_image.shape[-1] == 1:
            depth_image = depth_image.squeeze(-1)
        if depth_image.ndim == 2:
            depth_image = depth_image.unsqueeze(0)
        if depth_image.shape != (batch_size, image_height, image_width):
            raise ValueError(
                "Expected depth_image to have shape "
                f"({batch_size}, {image_height}, {image_width}), got {tuple(depth_image.shape)}"
            )

        pixel_u = pixel_coords[..., 0].round().long().clamp_(0, image_width - 1)
        pixel_v = pixel_coords[..., 1].round().long().clamp_(0, image_height - 1)
        batch_ids = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(pixel_u)
        sampled_depth = depth_image[batch_ids, pixel_v, pixel_u]
        valid_sampled_depth = torch.isfinite(sampled_depth) & (sampled_depth > 0.0)
        depth_match = valid_sampled_depth & (depth <= sampled_depth + float(occlusion_tolerance_m))

    visible_point_mask = inside_frustum & depth_match
    visible_point_count = visible_point_mask.sum(dim=1)
    visible_mask = visible_point_mask.all(dim=1) if require_all_points else (visible_point_count >= int(min_visible_points))

    occluded_mask = inside_frustum & ~depth_match if depth_image is not None else torch.zeros_like(inside_frustum)

    return {
        "visible_mask": visible_mask,
        "visible_point_mask": visible_point_mask,
        "visible_point_count": visible_point_count,
        "points_w": points_w,
        "points_c": points_c,
        "pixel_coords": pixel_coords,
        "depth": depth,
        "inside_frustum_mask": inside_frustum,
        "inside_image_mask": inside_image,
        "occluded_mask": occluded_mask,
        "sampled_depth": sampled_depth,
    }


def evaluate_link_points_observability(
    link_pos_w: torch.Tensor,
    link_quat_w: torch.Tensor,
    local_points: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    near_m: float = 0.0,
    far_m: float | None = None,
    depth_image: torch.Tensor | None = None,
    occlusion_tolerance_m: float = 0.03,
    pixel_margin: float = 0.0,
    min_visible_points: int = 1,
    require_all_points: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate visibility of points expressed in a link-local frame."""
    if link_pos_w.ndim != 2 or link_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected link_pos_w to have shape [B, 3], got {tuple(link_pos_w.shape)}")
    if link_quat_w.ndim != 2 or link_quat_w.shape[-1] != 4:
        raise ValueError(f"Expected link_quat_w to have shape [B, 4], got {tuple(link_quat_w.shape)}")
    if local_points.ndim != 3 or local_points.shape[-1] != 3:
        raise ValueError(f"Expected local_points to have shape [B, N, 3], got {tuple(local_points.shape)}")

    batch_size, num_points, _ = local_points.shape
    if link_pos_w.shape[0] != batch_size or link_quat_w.shape[0] != batch_size:
        raise ValueError(
            "Batch size mismatch between link pose and local points: "
            f"link_pos={link_pos_w.shape[0]}, link_quat={link_quat_w.shape[0]}, local_points={batch_size}"
        )

    link_quat_expanded = link_quat_w.unsqueeze(1).expand(-1, num_points, -1)
    points_w = quat_apply(link_quat_expanded, local_points) + link_pos_w.unsqueeze(1)
    return evaluate_points_observability(
        points_w=points_w,
        camera_pos_w=camera_pos_w,
        camera_quat_w=camera_quat_w,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
        near_m=near_m,
        far_m=far_m,
        depth_image=depth_image,
        occlusion_tolerance_m=occlusion_tolerance_m,
        pixel_margin=pixel_margin,
        min_visible_points=min_visible_points,
        require_all_points=require_all_points,
    )
