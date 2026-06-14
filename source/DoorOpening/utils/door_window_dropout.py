import torch


def _as_batched_points(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if points.ndim == 2 and points.shape[-1] == 3:
        return points.unsqueeze(0), True
    if points.ndim == 3 and points.shape[-1] == 3:
        return points, False
    raise ValueError(f"Expected points with shape (N, 3) or (B, N, 3), got {tuple(points.shape)}")


def _as_batched_pose(pose: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if pose.ndim == 1 and pose.shape[0] == 7:
        return pose.unsqueeze(0), True
    if pose.ndim == 2 and pose.shape[-1] == 7:
        return pose, False
    raise ValueError(f"Expected pose with shape (7,) or (B, 7), got {tuple(pose.shape)}")


def _as_batched_bbox(board_bbox_link1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if board_bbox_link1.ndim == 1 and board_bbox_link1.shape[0] == 6:
        bbox = board_bbox_link1.view(1, 2, 3)
        return bbox[:, 0], bbox[:, 1], True
    if board_bbox_link1.ndim == 2 and board_bbox_link1.shape == (2, 3):
        bbox = board_bbox_link1.unsqueeze(0)
        return bbox[:, 0], bbox[:, 1], True
    if board_bbox_link1.ndim == 2 and board_bbox_link1.shape[-1] == 6:
        bbox = board_bbox_link1.view(-1, 2, 3)
        return bbox[:, 0], bbox[:, 1], False
    if board_bbox_link1.ndim == 3 and board_bbox_link1.shape[-2:] == (2, 3):
        return board_bbox_link1[:, 0], board_bbox_link1[:, 1], False
    raise ValueError(
        "Expected board_bbox_link1 with shape (6,), (2, 3), (B, 6), or (B, 2, 3), "
        f"got {tuple(board_bbox_link1.shape)}"
    )


def _rand_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
    rng: torch.Generator | None,
) -> torch.Tensor:
    if rng is None:
        return torch.empty(shape, device=device, dtype=dtype).uniform_(float(low), float(high))
    sample = torch.empty(shape, dtype=dtype).uniform_(float(low), float(high), generator=rng)
    return sample.to(device=device)


def _rand_bool(shape: tuple[int, ...], prob: float, device: torch.device, rng: torch.Generator | None) -> torch.Tensor:
    if prob <= 0.0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    if prob >= 1.0:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if rng is None:
        return torch.rand(shape, device=device) < float(prob)
    sample = torch.rand(shape, generator=rng)
    return sample.to(device=device) < float(prob)


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[..., 1:]
    quat_w = quat[..., :1]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)


def _transform_points_to_link1_local(points_frame: torch.Tensor, link1_pose_frame: torch.Tensor) -> torch.Tensor:
    link1_pos = link1_pose_frame[:, :3]
    link1_quat = link1_pose_frame[:, 3:7]
    points_rel = points_frame - link1_pos.unsqueeze(1)
    quat_inv = _quat_conjugate_wxyz(link1_quat).unsqueeze(1).expand(-1, points_frame.shape[1], -1)
    return _quat_apply_wxyz(quat_inv, points_rel)


def sample_random_window_hole_metadata(
    link1_pose_world: torch.Tensor,
    board_bbox_link1: torch.Tensor,
    rng: torch.Generator | None = None,
    window_prob: float = 0.5,
    width_range: tuple[float, float] = (0.2, 0.55),
    height_range: tuple[float, float] = (0.35, 1.10),
    center_height_range: tuple[float, float] = (0.95, 1.45),
    side_margin_range: tuple[float, float] = (0.10, 0.25),
) -> dict[str, torch.Tensor]:
    pose_batched, pose_was_unbatched = _as_batched_pose(link1_pose_world)
    bbox_min, bbox_max, bbox_was_unbatched = _as_batched_bbox(board_bbox_link1)
    if pose_batched.shape[0] != bbox_min.shape[0]:
        raise ValueError(
            "link1_pose_world and board_bbox_link1 must have matching batch size, got "
            f"{pose_batched.shape[0]} and {bbox_min.shape[0]}"
        )

    device = pose_batched.device
    dtype = pose_batched.dtype
    env_count = pose_batched.shape[0]

    if not 0.0 <= float(window_prob) <= 1.0:
        raise ValueError(f"window_prob must be in [0, 1], got {window_prob}")

    width_min, width_max = map(float, width_range)
    height_min, height_max = map(float, height_range)
    center_h_min, center_h_max = map(float, center_height_range)
    side_margin_min, side_margin_max = map(float, side_margin_range)
    if width_max < width_min or height_max < height_min or center_h_max < center_h_min or side_margin_max < side_margin_min:
        raise ValueError("Range arguments must satisfy min <= max.")

    enabled = _rand_bool((env_count,), window_prob, device=device, rng=rng)

    board_width = (bbox_max[:, 0] - bbox_min[:, 0]).clamp_min(1e-6)
    board_height = (bbox_max[:, 1] - bbox_min[:, 1]).clamp_min(1e-6)
    board_center_x = 0.5 * (bbox_min[:, 0] + bbox_max[:, 0])
    board_center_y = 0.5 * (bbox_min[:, 1] + bbox_max[:, 1])

    side_margin = _rand_uniform((env_count,), side_margin_min, side_margin_max, device, dtype, rng).clamp_min(0.0)
    sampled_width = _rand_uniform((env_count,), width_min, width_max, device, dtype, rng).clamp_min(0.0)
    sampled_height = _rand_uniform((env_count,), height_min, height_max, device, dtype, rng).clamp_min(0.0)
    max_width = (board_width - 2.0 * side_margin).clamp_min(0.0)
    max_height = (board_height - 2.0 * side_margin).clamp_min(0.0)
    hole_width = torch.minimum(sampled_width, max_width)
    hole_height = torch.minimum(sampled_height, max_height)

    min_center_x = bbox_min[:, 0] + side_margin + 0.5 * hole_width
    max_center_x = bbox_max[:, 0] - side_margin - 0.5 * hole_width
    sampled_center_x = _rand_uniform((env_count,), 0.0, 1.0, device, dtype, rng)
    hole_center_x = min_center_x + sampled_center_x * (max_center_x - min_center_x).clamp_min(0.0)
    hole_center_x = torch.where(max_center_x >= min_center_x, hole_center_x, board_center_x)

    sampled_center_height = _rand_uniform((env_count,), center_h_min, center_h_max, device, dtype, rng)
    sampled_center_y = bbox_min[:, 1] + sampled_center_height
    min_center_y = bbox_min[:, 1] + side_margin + 0.5 * hole_height
    max_center_y = bbox_max[:, 1] - side_margin - 0.5 * hole_height
    hole_center_y = sampled_center_y.clamp(min=min_center_y, max=max_center_y)
    hole_center_y = torch.where(max_center_y >= min_center_y, hole_center_y, board_center_y)

    hole_center_z = 0.5 * (bbox_min[:, 2] + bbox_max[:, 2])
    hole_min = torch.stack(
        [
            hole_center_x - 0.5 * hole_width,
            hole_center_y - 0.5 * hole_height,
            bbox_min[:, 2],
        ],
        dim=-1,
    )
    hole_max = torch.stack(
        [
            hole_center_x + 0.5 * hole_width,
            hole_center_y + 0.5 * hole_height,
            bbox_max[:, 2],
        ],
        dim=-1,
    )
    hole_bbox = torch.cat([hole_min, hole_max], dim=-1)
    hole_center = torch.stack([hole_center_x, hole_center_y, hole_center_z], dim=-1)

    metadata = {
        "enabled": enabled,
        "hole_bbox_link1": hole_bbox,
        "hole_width": hole_width,
        "hole_height": hole_height,
        "hole_center_link1": hole_center,
        "side_margin": side_margin,
    }
    if pose_was_unbatched and bbox_was_unbatched:
        return {key: value[0] for key, value in metadata.items()}
    return metadata


def apply_window_dropout_to_door_points(
    points_world: torch.Tensor,
    link1_pose_world: torch.Tensor,
    board_bbox_link1: torch.Tensor,
    hole_metadata: dict[str, torch.Tensor],
    surface_eps: float = 0.03,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points_batched, points_was_unbatched = _as_batched_points(points_world)
    pose_batched, _ = _as_batched_pose(link1_pose_world)
    bbox_min, bbox_max, _ = _as_batched_bbox(board_bbox_link1)

    enabled = hole_metadata["enabled"]
    hole_bbox = hole_metadata["hole_bbox_link1"]
    if enabled.ndim == 0:
        enabled = enabled.view(1)
    if hole_bbox.ndim == 1:
        hole_bbox = hole_bbox.view(1, 6)

    if not (points_batched.shape[0] == pose_batched.shape[0] == bbox_min.shape[0] == enabled.shape[0] == hole_bbox.shape[0]):
        raise ValueError("points_world, link1_pose_world, board_bbox_link1, and hole_metadata batch sizes must match.")

    points_link1 = _transform_points_to_link1_local(points_batched, pose_batched)
    hole_min = hole_bbox[:, :3]
    hole_max = hole_bbox[:, 3:]

    in_hole_x = (points_link1[..., 0] >= hole_min[:, None, 0]) & (points_link1[..., 0] <= hole_max[:, None, 0])
    in_hole_y = (points_link1[..., 1] >= hole_min[:, None, 1]) & (points_link1[..., 1] <= hole_max[:, None, 1])
    within_board_thickness = (
        (points_link1[..., 2] >= bbox_min[:, None, 2] - float(surface_eps))
        & (points_link1[..., 2] <= bbox_max[:, None, 2] + float(surface_eps))
    )
    dist_to_front = (points_link1[..., 2] - bbox_min[:, None, 2]).abs()
    dist_to_back = (points_link1[..., 2] - bbox_max[:, None, 2]).abs()
    near_surface = torch.minimum(dist_to_front, dist_to_back) <= float(surface_eps)

    drop_mask = enabled[:, None] & in_hole_x & in_hole_y & within_board_thickness & near_surface
    filtered_points = points_batched.clone()
    filtered_points[drop_mask] = float("nan")

    out_metadata = dict(hole_metadata)
    out_metadata["drop_mask"] = drop_mask
    out_metadata["num_dropped_points"] = drop_mask.sum(dim=-1)
    if points_was_unbatched:
        filtered_points = filtered_points[0]
        out_metadata = {key: value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 else value for key, value in out_metadata.items()}
    return filtered_points, out_metadata


def apply_random_window_dropout_to_door_points(
    points_world: torch.Tensor,
    link1_pose_world: torch.Tensor,
    board_bbox_link1: torch.Tensor,
    rng: torch.Generator | None = None,
    window_prob: float = 0.5,
    width_range: tuple[float, float] = (0.2, 0.55),
    height_range: tuple[float, float] = (0.35, 1.10),
    center_height_range: tuple[float, float] = (0.95, 1.45),
    side_margin_range: tuple[float, float] = (0.10, 0.25),
    surface_eps: float = 0.03,
):
    """
    Drop board-surface points inside a random rectangular door-panel hole.

    Notes:
    - URDF visual-only window meshes affect rendered RGB/depth and mesh-based visual point clouds.
    - They generally do not affect PhysX raycast lidar when lidar is driven from collision geometry.
    - This helper works directly on point clouds, so it can simulate glass/window holes even when the
      physical door panel remains a solid board in the simulator.

    The input points and pose only need to be expressed in the same frame; despite the historical
    `*_world` names, this helper also works on robot-base-frame point clouds if `link1_pose_world`
    is provided in that same base frame.
    """

    metadata = sample_random_window_hole_metadata(
        link1_pose_world=link1_pose_world,
        board_bbox_link1=board_bbox_link1,
        rng=rng,
        window_prob=window_prob,
        width_range=width_range,
        height_range=height_range,
        center_height_range=center_height_range,
        side_margin_range=side_margin_range,
    )
    return apply_window_dropout_to_door_points(
        points_world=points_world,
        link1_pose_world=link1_pose_world,
        board_bbox_link1=board_bbox_link1,
        hole_metadata=metadata,
        surface_eps=surface_eps,
    )
