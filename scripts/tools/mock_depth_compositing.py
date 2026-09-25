"""Shared DEX-style depth composition helpers for offline mock renderers.

These helpers are intentionally confined to scripts/tools.  They close the
background leak around a sampled robot silhouette without changing the training
renderer or dilating the robot depth itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def suppress_scene_behind_robot(scene_depth: torch.Tensor, robot_depth: torch.Tensor, inflate_px: int = 4):
    """Clear scene depth under the robot's dilated camera-space silhouette.

    Point-sampled robot surfaces can leave empty pixels between hand/finger samples.
    If the scene depth remains there, the door shows through the robot.  DEX-style
    composition treats the robot silhouette as an occluder while retaining the raw,
    crisp robot z-buffer for its actual visible surface.

    Returns ``(scene_depth_cleared, silhouette_mask)`` for diagnostics.
    """
    if scene_depth.shape != robot_depth.shape:
        raise ValueError(
            f"scene and robot depth shapes must match, got {tuple(scene_depth.shape)} and "
            f"{tuple(robot_depth.shape)}"
        )
    radius = max(0, int(inflate_px))
    robot_pixels = torch.isfinite(robot_depth)
    silhouette = robot_pixels
    if radius > 0:
        kernel = 2 * radius + 1
        silhouette = F.max_pool2d(
            robot_pixels.to(scene_depth.dtype).unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(1) > 0
    return torch.where(silhouette, torch.full_like(scene_depth, float("inf")), scene_depth), silhouette


def composite_robot_scene_depth(scene_depth: torch.Tensor, robot_depth: torch.Tensor, inflate_px: int = 4):
    """Apply robot-silhouette occlusion, then nearest-depth composite the two layers."""
    scene_depth, silhouette = suppress_scene_behind_robot(scene_depth, robot_depth, inflate_px)
    return torch.minimum(scene_depth, robot_depth), silhouette
