"""Utilities for writing raw point-cloud trajectories for Viser replay."""

from __future__ import annotations

import os
from pathlib import Path

import torch


DEFAULT_STREAM_COLORS = {
    "ground_truth": (120, 120, 120),
    "robot_obs": (79, 195, 247),
    "policy_input": (0, 170, 120),
    "robot": (79, 195, 247),
    "door": (255, 193, 7),
}


def format_iterated_record_path(path_str: str | os.PathLike[str], tag: str | int) -> str:
    """Append a record-specific suffix to a replay filename."""

    path = Path(path_str)
    return str(path.with_name(f"{path.stem}_{tag}{path.suffix}"))


def prepare_pointcloud(
    pointcloud: torch.Tensor | None,
    env_id: int | None = None,
    max_points: int | None = None,
    *,
    drop_zero_rows: bool = False,
) -> torch.Tensor:
    """Extract one cloud, sanitize it, and downsample it on CPU for replay export."""

    if pointcloud is None:
        return torch.zeros((0, 3), dtype=torch.float32)
    if not isinstance(pointcloud, torch.Tensor):
        pointcloud = torch.as_tensor(pointcloud)

    if pointcloud.ndim == 2 and pointcloud.shape[-1] == 3:
        points = pointcloud.detach()
    elif pointcloud.ndim == 3 and pointcloud.shape[-1] == 3:
        if env_id is None:
            raise ValueError("env_id is required for batched pointclouds.")
        points = pointcloud[int(env_id)].detach()
    else:
        raise ValueError(f"Expected pointcloud with shape (N, 3) or (B, N, 3), got {tuple(pointcloud.shape)}.")

    points = points.to(dtype=torch.float32, device="cpu")
    finite_mask = torch.isfinite(points).all(dim=-1)
    points = points[finite_mask]
    if drop_zero_rows and points.numel() > 0:
        nonzero_mask = torch.any(points.abs() > 1e-6, dim=-1)
        points = points[nonzero_mask]
    if max_points is not None and max_points > 0 and points.shape[0] > max_points:
        sample_idx = torch.linspace(0, points.shape[0] - 1, steps=int(max_points), dtype=torch.float32)
        sample_idx = torch.round(sample_idx).to(dtype=torch.long)
        points = points[sample_idx]
    return points


def quat_apply_wxyz(quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate points by a wxyz quaternion."""

    if points.numel() == 0:
        return points
    quat = quat.to(device=points.device, dtype=points.dtype)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q_xyz = quat[..., 1:]
    q_w = quat[..., :1]
    uv = torch.cross(q_xyz, points, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return points + 2.0 * (q_w * uv + uuv)


def prepare_world_points_from_local(
    pointcloud_local: torch.Tensor | None,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    env_id: int,
    max_points: int | None,
    *,
    drop_zero_rows: bool = False,
) -> torch.Tensor:
    """Convert one env's base-frame point cloud into world-frame CPU points."""

    local_points = prepare_pointcloud(
        pointcloud_local,
        env_id=env_id,
        max_points=max_points,
        drop_zero_rows=drop_zero_rows,
    )
    if local_points.numel() == 0:
        return local_points

    quat = base_quat_w[int(env_id)].detach().to(device="cpu", dtype=local_points.dtype).unsqueeze(0)
    quat = quat.expand(local_points.shape[0], -1)
    pos = base_pos_w[int(env_id)].detach().to(device="cpu", dtype=local_points.dtype).unsqueeze(0)
    return quat_apply_wxyz(quat, local_points) + pos
