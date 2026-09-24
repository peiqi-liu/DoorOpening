"""Cached visual-only scene sampler for bounded point-cloud A/B experiments.

This module deliberately samples URDF ``visual`` geometry only.  Collision geometry is
never parsed or merged into the point cloud.  Mesh sampling is done once; runtime assembly
only applies live poses to cached tensors and concatenates explicitly named source partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from DoorOpening.utils.extract_pointcloud_from_articulation import (
    build_first_visual_link_pointcloud_cache,
    compose_cached_link_pointcloud_world,
)


@dataclass
class VisualSceneBatch:
    """Named scene partitions, all batched as ``(B, N, 3)`` tensors."""

    door: torch.Tensor
    robot: torch.Tensor
    walls: torch.Tensor

    @property
    def static_scene(self) -> torch.Tensor:
        if self.walls.shape[1] == 0:
            return self.door
        return torch.cat((self.door, self.walls), dim=1)

    @property
    def full_scene(self) -> torch.Tensor:
        return torch.cat((self.static_scene, self.robot), dim=1)


class CachedVisualSceneSampler:
    """A/B-compatible cached sampler around the existing visual-link sampler.

    ``link_points`` are built once from ``FrankaGripperSampler.points``.  The caller supplies
    live body poses, so no trimesh/URDF work occurs during ``compose``.  ``replacement_link_points``
    can replace one visual source (for example a closed solid panel proxy) without touching the
    collision model or observation dimensions.
    """

    def __init__(self, sampler, link_names=None, replacement_link_points=None):
        self.sampler = sampler
        self.device = torch.device(sampler.device)
        self.link_names = tuple(link_names or (link.name for link in sampler.links))
        self.link_points = build_first_visual_link_pointcloud_cache(
            sampler, link_names=list(self.link_names), device=self.device
        )
        if replacement_link_points:
            for name, points in replacement_link_points.items():
                points = torch.as_tensor(points, device=self.device, dtype=torch.float32)
                if points.ndim != 2 or points.shape[-1] != 3:
                    raise ValueError(f"Replacement source {name!r} must have shape (N, 3).")
                self.link_points[name] = points.contiguous()

    def compose(self, link_pos_w_by_name, link_quat_w_by_name, num_points=None):
        return compose_cached_link_pointcloud_world(
            self.link_points, link_pos_w_by_name, link_quat_w_by_name, num_points=num_points
        )

    def compose_named(self, link_pos_w_by_name, link_quat_w_by_name, num_points=None):
        """Return a dict of source-partition clouds without changing their concatenation semantics."""
        result = {}
        for name in self.link_names:
            if name not in self.link_points:
                continue
            result[name] = compose_cached_link_pointcloud_world(
                {name: self.link_points[name]}, link_pos_w_by_name, link_quat_w_by_name,
                num_points=None,
            )
        return result

    def benchmark(self, link_pos_w_by_name, link_quat_w_by_name, repeats=20, num_points=None):
        """Measure cached GPU assembly only; first call is excluded as warmup."""
        self.compose(link_pos_w_by_name, link_quat_w_by_name, num_points=num_points)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        for _ in range(int(repeats)):
            self.compose(link_pos_w_by_name, link_quat_w_by_name, num_points=num_points)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = perf_counter() - start
        return {"repeats": int(repeats), "total_s": elapsed, "mean_ms": elapsed * 1000.0 / max(1, int(repeats))}
