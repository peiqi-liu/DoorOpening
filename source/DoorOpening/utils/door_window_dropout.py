import math

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


def _randn(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    rng: torch.Generator | None,
) -> torch.Tensor:
    if rng is None:
        return torch.empty(shape, device=device, dtype=dtype).normal_()
    sample = torch.empty(shape, dtype=dtype).normal_(generator=rng)
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


def sample_glass_reflection_points(
    hole_metadata: dict[str, torch.Tensor],
    board_bbox_link1: torch.Tensor,
    num_points: int,
    reflect_prob: float = 1.0,
    blob_size: tuple[float, float, float] = (0.5, 1.2, 0.3),
    size_fraction_range: tuple[float, float] = (1.0, 1.0),
    num_lobes: int = 3,
    behind_range: tuple[float, float] = (0.1, 1.0),
    density_range: tuple[float, float] = (1.0, 1.0),
    front_sign: torch.Tensor | float | None = None,
    rng: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """A robot-sized (or partial), irregular noise BLOB behind a glass window, in the link_1 LOCAL frame.

    A dark glass door mirrors the robot: you see a rough copy of it -- often just PART of it (an arm, a
    shoulder) -- behind the glass. Rather than ray-tracing a real mirror image, this drops a cheap
    robot-SIZED cluster of noise at a random spot behind the window opening, pushed a random depth BEHIND
    the panel's back face, on the side away from the camera (``front_sign``). The cluster is deliberately
    IRREGULAR (not a uniform box): a few overlapping Gaussian lobes at random sub-centres, then spun by a
    random yaw about the vertical, so it reads as a lumpy robot-ish shape rather than a clean rectangle.
    Seen through the see-through hole (with the opaque panel occluding the rest), a depth camera reads
    that lumpy blob receding behind the glass.

    ``blob_size`` = (width_x, height_y, depth_z) FULL extents (m); pick robot-ish dimensions.
    ``num_lobes`` = number of Gaussian sub-clusters (>=1); more = lumpier. ``front_sign`` (per env, +1/-1,
    scalar, or None->+1) says which link_1 thickness (z) direction points at the camera; the blob goes on
    the opposite side.

    Probability / severity model (per env, per rollout):
      * ``reflect_prob`` -- among envs whose hole is ENABLED, the fraction that get a reflection blob. The
        rest keep the bare see-through hole (the "bright glass" / pure-hole case), so pure holes always
        occur when ``reflect_prob < 1``.
      * ``size_fraction_range`` -- each blob dimension is independently scaled by a fraction drawn from
        this range, so the blob ranges from a small/thin sliver (part of the robot, e.g. an arm) up to
        the full ``blob_size``. It is then placed at a random spot that keeps its core inside the window.
      * ``behind_range`` -- per-env depth (m) behind the panel back face where the blob centre sits.
      * ``density_range`` -- per-env fraction of the ``num_points`` budget kept (rest NaN). Lower = fainter
        (brighter lighting), higher = denser (darker). ``(1.0, 1.0)`` keeps the full budget every time.
    Non-reflecting envs are all NaN (ignored by the renderer). Output points are in the SAME frame as
    ``board_bbox_link1``.

    Returns ``{"reflection_points_link1": (B, num_points, 3), "reflection_enabled": (B,)}`` (unbatched to
    ``(num_points, 3)`` / scalar when ``board_bbox_link1`` is unbatched).
    """
    bbox_min, bbox_max, bbox_was_unbatched = _as_batched_bbox(board_bbox_link1)
    device = bbox_min.device
    dtype = bbox_min.dtype
    env_count = bbox_min.shape[0]
    num_points = int(num_points)

    enabled = hole_metadata["enabled"]
    if enabled.ndim == 0:
        enabled = enabled.view(1)
    hole_center = hole_metadata["hole_center_link1"]
    if hole_center.ndim == 1:
        hole_center = hole_center.view(1, 3)
    hole_width = hole_metadata["hole_width"].reshape(-1, 1)
    hole_height = hole_metadata["hole_height"].reshape(-1, 1)
    if not (enabled.shape[0] == hole_center.shape[0] == env_count):
        raise ValueError("hole_metadata and board_bbox_link1 batch sizes must match.")

    behind_min, behind_max = map(float, behind_range)
    if not (0.0 <= behind_min <= behind_max):
        raise ValueError(f"behind_range must satisfy 0 <= min <= max, got {behind_range}")
    density_min, density_max = map(float, density_range)
    if not (0.0 <= density_min <= density_max <= 1.0):
        raise ValueError(f"density_range must satisfy 0 <= min <= max <= 1, got {density_range}")
    frac_min, frac_max = map(float, size_fraction_range)
    if not (0.0 <= frac_min <= frac_max):
        raise ValueError(f"size_fraction_range must satisfy 0 <= min <= max, got {size_fraction_range}")
    blob_w, blob_h, blob_d = map(float, blob_size)
    num_lobes = max(1, int(num_lobes))

    if front_sign is None:
        fs = torch.ones((env_count, 1), device=device, dtype=dtype)
    else:
        fs = torch.as_tensor(front_sign, device=device, dtype=dtype).reshape(-1)
        if fs.numel() == 1:
            fs = fs.expand(env_count)
        fs = torch.where(fs >= 0, torch.ones_like(fs), -torch.ones_like(fs)).view(env_count, 1)

    reflect_enabled = enabled.to(dtype=torch.bool) & _rand_bool((env_count,), reflect_prob, device=device, rng=rng)

    def _finish(points):
        out = {"reflection_points_link1": points, "reflection_enabled": reflect_enabled}
        if bbox_was_unbatched:
            return {"reflection_points_link1": points[0], "reflection_enabled": reflect_enabled[0]}
        return out

    if num_points <= 0:
        return _finish(torch.zeros((env_count, 0, 3), device=device, dtype=dtype))

    # Per-env, per-dimension size: independently scale each blob extent by a random fraction, so the
    # cluster ranges from a small/thin sliver (just an arm) up to the full robot size.
    bw = blob_w * _rand_uniform((env_count, 1), frac_min, frac_max, device, dtype, rng)
    bh = blob_h * _rand_uniform((env_count, 1), frac_min, frac_max, device, dtype, rng)
    bd = blob_d * _rand_uniform((env_count, 1), frac_min, frac_max, device, dtype, rng)
    blob_ext = torch.cat([bw, bh, bd], dim=-1)  # (env, 3)
    # Blob CENTRE. z = a random distance behind the panel back face (the face away from the camera):
    # back_face is bbox_min_z when the camera is on the +z side (fs=+1), bbox_max_z on the -z side
    # (fs=-1), and "behind" is the -fs direction. x,y = window centre plus a random offset that keeps
    # the (possibly small) blob inside the window opening, so a partial reflection can land off-centre.
    behind = _rand_uniform((env_count, 1), behind_min, behind_max, device, dtype, rng)
    back_face_z = torch.where(fs > 0, bbox_min[:, 2:3], bbox_max[:, 2:3])
    avail_x = (0.5 * hole_width - 0.5 * bw).clamp_min(0.0)
    avail_y = (0.5 * hole_height - 0.5 * bh).clamp_min(0.0)
    off_x = _rand_uniform((env_count, 1), -1.0, 1.0, device, dtype, rng) * avail_x
    off_y = _rand_uniform((env_count, 1), -1.0, 1.0, device, dtype, rng) * avail_y
    center = torch.cat(
        [hole_center[:, 0:1] + off_x, hole_center[:, 1:2] + off_y, back_face_z - fs * behind], dim=-1
    )  # (env, 3)
    # IRREGULAR cluster (not a uniform box): num_lobes Gaussian sub-clumps at random sub-centres, each
    # point drawn around one lobe. Overlapping lobes give a lumpy, robot-ish silhouette.
    lobe_centers = _rand_uniform((env_count, num_lobes, 3), -0.30, 0.30, device, dtype, rng) * blob_ext[:, None, :]
    lobe_std = _rand_uniform((env_count, num_lobes, 3), 0.10, 0.28, device, dtype, rng) * blob_ext[:, None, :]
    lobe_idx = (
        _rand_uniform((env_count, num_points), 0.0, float(num_lobes), device, dtype, rng).floor().clamp_(0, num_lobes - 1).long()
    )
    idx3 = lobe_idx.unsqueeze(-1).expand(env_count, num_points, 3)
    ctr = torch.gather(lobe_centers, 1, idx3)
    std = torch.gather(lobe_std, 1, idx3)
    local = ctr + _randn((env_count, num_points, 3), device, dtype, rng) * std
    # Random yaw about the vertical (height, y) axis so the cluster is not axis-aligned to the door.
    theta = _rand_uniform((env_count, 1), -math.pi, math.pi, device, dtype, rng)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]
    local = torch.stack([lx * cos_t + lz * sin_t, ly, -lx * sin_t + lz * cos_t], dim=-1)
    points = center[:, None, :] + local
    # Per-env blob density: keep a random fraction of the budget so reflections range from faint (bright
    # lighting) to dense (dark). Point kept iff its own uniform draw < that env's fill fraction.
    keep = reflect_enabled[:, None].expand(env_count, num_points)
    if not (density_min == density_max == 1.0):
        fill = _rand_uniform((env_count, 1), density_min, density_max, device, dtype, rng)
        keep = keep & (_rand_uniform((env_count, num_points), 0.0, 1.0, device, dtype, rng) < fill)
    points = torch.where(keep[:, :, None], points, torch.full_like(points, float("nan")))
    return _finish(points)


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
