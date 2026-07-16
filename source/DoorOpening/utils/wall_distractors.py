"""Wall-distractor point-cloud sampling, shared between training and offline tooling.

This module is deliberately free of any Isaac Sim / ``carb`` dependency (pure torch), so it can be
imported both by the training pipeline (``DoorOpening.distillation.multi_pcd_dagger``) and by
standalone scripts that render/inspect wall distractors without launching the simulator.

The sampler builds, per env, four vertical "wall columns" beside the door panel (two jamb sides x
front/back of the slab), in the door-panel local frame with axes inferred from the panel bbox
extents (smallest -> thickness, middle -> width, largest -> height).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class WallDistractorParams:
    """Parsed wall-distractor geometry knobs (see the ``wall_distractors`` config block)."""

    enabled: bool
    num_points: int
    resample_each_step: bool
    side_margin_scale_min: float
    side_margin_scale_max: float
    side_margin_abs_min_m: Optional[float]
    side_margin_abs_max_m: Optional[float]
    gap_min_m: float
    gap_max_m: float
    depth_min_m: float
    depth_max_m: float
    center_offset_min_m: float
    center_offset_max_m: float
    height_min_m: Optional[float]
    height_max_m: Optional[float]
    face_jitter_m: float
    point_density_per_m2: Optional[float]

    @classmethod
    def from_cfg(cls, cfg: dict, scene_door_num_points: int) -> "WallDistractorParams":
        cfg = dict(cfg or {})
        enabled = bool(cfg.get("enabled", True))
        num_points = int(
            cfg.get("wall_num_points", cfg.get("num_points", max(256, int(scene_door_num_points) // 3)))
        )
        side_margin_scale_min, side_margin_scale_max = map(
            float, cfg.get("side_margin_scale", [0.35, 0.75])
        )
        side_margin_abs_range = cfg.get("side_margin_m")
        if side_margin_abs_range is None:
            side_margin_abs_min_m = None
            side_margin_abs_max_m = None
        else:
            side_margin_abs_min_m, side_margin_abs_max_m = map(float, side_margin_abs_range)
        gap_min_m, gap_max_m = map(float, cfg.get("edge_gap_m", [0.015, 0.04]))
        # Clamp to non-negative: the gap is the distance from the wall's inner face to the door
        # panel's own side edge, so a negative value would let the wall overlap into the panel.
        gap_min_m = max(0.0, gap_min_m)
        gap_max_m = max(gap_min_m, gap_max_m)
        depth_min_m, depth_max_m = map(float, cfg.get("depth_m", [0.10, 0.26]))
        center_offset_min_m, center_offset_max_m = map(float, cfg.get("center_offset_m", [-0.20, 0.20]))
        height_range = cfg.get("height_range_m")
        if height_range is None:
            height_min_m = None
            height_max_m = None
        else:
            height_min_m, height_max_m = map(float, height_range)
        face_jitter_m = float(cfg.get("face_jitter_m", 0.004))
        wall_point_density = cfg.get("point_density_per_m2")
        point_density_per_m2 = None if wall_point_density is None else float(wall_point_density)
        resample_each_step = bool(cfg.get("resample_each_step", False))
        return cls(
            enabled=enabled,
            num_points=num_points,
            resample_each_step=resample_each_step,
            side_margin_scale_min=side_margin_scale_min,
            side_margin_scale_max=side_margin_scale_max,
            side_margin_abs_min_m=side_margin_abs_min_m,
            side_margin_abs_max_m=side_margin_abs_max_m,
            gap_min_m=gap_min_m,
            gap_max_m=gap_max_m,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            center_offset_min_m=center_offset_min_m,
            center_offset_max_m=center_offset_max_m,
            height_min_m=height_min_m,
            height_max_m=height_max_m,
            face_jitter_m=face_jitter_m,
            point_density_per_m2=point_density_per_m2,
        )


def compute_wall_bbox_ordering(board_bbox: torch.Tensor):
    """Order each env's door-panel bbox axes by extent (smallest->thickness, ..., largest->height).

    Args:
        board_bbox: (E, 2, 3) with [:, 0]=min corner, [:, 1]=max corner in the door-base frame.
    Returns:
        (axis_order, bbox_min_ordered, bbox_max_ordered), each (E, 3). ``axis_order`` maps ordered
        axis -> original axis (use it to scatter sampled points back to the original frame).
    """
    bbox_min = board_bbox[:, 0]
    bbox_max = board_bbox[:, 1]
    bbox_extent = (bbox_max - bbox_min).clamp_min(1e-4)
    axis_order = torch.argsort(bbox_extent, dim=-1)
    bbox_min_ordered = torch.gather(bbox_min, 1, axis_order)
    bbox_max_ordered = torch.gather(bbox_max, 1, axis_order)
    return axis_order, bbox_min_ordered, bbox_max_ordered


def sample_wall_points_local(
    axis_order: torch.Tensor,
    bbox_min_ordered: torch.Tensor,
    bbox_max_ordered: torch.Tensor,
    num_points: int,
    params: WallDistractorParams,
    device,
) -> torch.Tensor:
    """Sample wall-distractor surface points in the door-base local frame.

    Args:
        axis_order / bbox_min_ordered / bbox_max_ordered: per-env (E, 3), from
            :func:`compute_wall_bbox_ordering` (already sliced to the requested envs).
        num_points: buffer width (also the per-env hard cap when density mode is on).
        params: geometry knobs.
    Returns:
        (E, num_points, 3) points in the door-base frame. In density mode, slots beyond each env's
        area-scaled target are NaN.
    """
    num_points = int(num_points)
    env_count = int(axis_order.shape[0])
    if num_points <= 0 or env_count == 0:
        return torch.zeros((env_count, 0, 3), dtype=torch.float32, device=device)

    # From here on, the ordered coordinates are:
    #   axis 0 = thickness (normal to the door slab)
    #   axis 1 = width (left/right of the panel)
    #   axis 2 = height (bottom/top)
    thickness_min, width_min, height_min = bbox_min_ordered.unbind(dim=-1)
    thickness_max, width_max, height_max = bbox_max_ordered.unbind(dim=-1)
    thickness_extent = (thickness_max - thickness_min).clamp_min(1e-4)
    width_extent = (width_max - width_min).clamp_min(1e-4)

    def rand_range(low, high, shape):
        return torch.empty(shape, device=device, dtype=torch.float32).uniform_(float(low), float(high))

    thickness_center = 0.5 * (thickness_min + thickness_max)

    def sample_seam():
        # Where the front and back segments on one side meet. Randomizing this (instead of
        # pinning it to thickness_center) preserves the old single-column "recessed/protruding"
        # variety while still guaranteeing the two segments butt together exactly.
        return thickness_center + rand_range(
            params.center_offset_min_m, params.center_offset_max_m, (env_count,)
        )

    def sample_column_surfaces(seam, is_front):
        column_depth = thickness_extent + rand_range(params.depth_min_m, params.depth_max_m, (env_count,))
        # Front and back share the seam as a hard boundary (front's min == back's max == seam),
        # so the two segments on a side are always exactly attached: no gap, no overlap.
        if is_front:
            return seam, seam + column_depth
        return seam - column_depth, seam

    def sample_column_width():
        if params.side_margin_abs_min_m is not None:
            # In column mode, the side margin range becomes the column width range.
            column_width = rand_range(params.side_margin_abs_min_m, params.side_margin_abs_max_m, (env_count,))
        else:
            column_width = width_extent * rand_range(
                params.side_margin_scale_min, params.side_margin_scale_max, (env_count,)
            )
        return column_width.clamp_min(1e-4)

    def sample_column_bounds(attach_on_right, seam, is_front):
        edge_gap = rand_range(params.gap_min_m, params.gap_max_m, (env_count,))
        column_width = sample_column_width()
        column_min_surface, column_max_surface = sample_column_surfaces(seam, is_front)
        # The inner face starts just outside the panel side edge.
        # Right column: panel right edge + gap, then extends further in +width.
        # Left column:  panel left edge  - gap, then extends further in -width.
        if attach_on_right:
            column_inner_width = width_max + edge_gap
            column_outer_width = column_inner_width + column_width
        else:
            column_inner_width = width_min - edge_gap
            column_outer_width = column_inner_width - column_width
        column_width_lo = torch.minimum(column_inner_width, column_outer_width)
        column_width_hi = torch.maximum(column_inner_width, column_outer_width)
        # Every column starts at the fixed lower bound; its UPPER bound is randomized INDEPENDENTLY
        # per column (sample_column_bounds runs once per column), so the four walls reach different
        # heights. When height_range_m is unset, fall back to the door panel's own bbox height.
        if params.height_min_m is not None:
            column_height_lo = torch.full((env_count,), params.height_min_m, device=device, dtype=torch.float32)
            column_height_hi = rand_range(params.height_min_m, params.height_max_m, (env_count,))
        else:
            column_height_lo = height_min
            column_height_hi = height_max
        return (
            column_min_surface,
            column_max_surface,
            column_width_lo,
            column_width_hi,
            column_height_lo,
            column_height_hi,
        )

    # Four wall segments: {left, right} x {front, back} of the door plane. Front/back share a
    # seam sampled once per side, so each side's pair butts together exactly (see sample_seam).
    left_seam = sample_seam()
    right_seam = sample_seam()
    column_variants = [
        sample_column_bounds(attach_on_right, seam, is_front)
        for attach_on_right, seam in ((False, left_seam), (True, right_seam))
        for is_front in (False, True)
    ]
    num_columns = len(column_variants)
    # Each stack has shape (env_count, num_columns); column order is
    # [left-back, left-front, right-back, right-front].
    column_min_surface_stack = torch.stack([v[0] for v in column_variants], dim=1)
    column_max_surface_stack = torch.stack([v[1] for v in column_variants], dim=1)
    column_width_lo_stack = torch.stack([v[2] for v in column_variants], dim=1)
    column_width_hi_stack = torch.stack([v[3] for v in column_variants], dim=1)
    column_height_lo_stack = torch.stack([v[4] for v in column_variants], dim=1)
    column_height_hi_stack = torch.stack([v[5] for v in column_variants], dim=1)

    wall_points_ordered = torch.empty((env_count, num_points, 3), dtype=torch.float32, device=device)

    # Per-column extents (E, C), reused for area-weighted sampling and the density cap.
    col_thick_ext = (column_max_surface_stack - column_min_surface_stack).abs()
    col_width_ext = (column_width_hi_stack - column_width_lo_stack).abs()
    col_height_ext = (column_height_hi_stack - column_height_lo_stack).abs()
    # Area of each of the 6 sampled faces per column, ordered [thickness_min, thickness_max,
    # width_min, width_max, height_min (bottom cap), height_max (top cap)]: thickness faces are
    # (width x height), width faces are (thickness x height), height caps are (thickness x width).
    # The two height caps CLOSE the box -- without them short walls render open-topped (visible tops
    # missing) since the camera looks slightly down onto them.
    face_area = torch.stack(
        [
            col_width_ext * col_height_ext,
            col_width_ext * col_height_ext,
            col_thick_ext * col_height_ext,
            col_thick_ext * col_height_ext,
            col_thick_ext * col_width_ext,
            col_thick_ext * col_width_ext,
        ],
        dim=-1,
    )  # (E, C, 6)
    num_faces = face_area.shape[-1]

    if params.point_density_per_m2 is not None:
        # Area-WEIGHTED selection of (column, face): each point picks a face with probability
        # proportional to that face's area, so the resulting density is uniform ACROSS every wall
        # face (constant points/m^2), not merely uniform in total. A big face and a small face then
        # get point counts proportional to their sizes. Flat index = column * 4 + face.
        probs = face_area.reshape(env_count, num_columns * num_faces).clamp_min(1e-12)
        probs = probs / probs.sum(dim=1, keepdim=True)
        flat_idx = torch.multinomial(probs, num_points, replacement=True)  # (E, num_points)
        column_idx = flat_idx // num_faces
        face_ids = flat_idx % num_faces
    else:
        # Legacy UNIFORM selection (equal expected points per column and per face, so smaller faces
        # end up denser). Kept for configs that do not set point_density_per_m2.
        face_ids = torch.randint(0, num_faces, (env_count, num_points), device=device)
        column_idx = torch.randint(0, num_columns, (env_count, num_points), device=device)
        if num_points >= num_columns:
            # Keep every column (both jamb sides, both front/back) populated for each env.
            for column_id in range(num_columns):
                column_idx[:, column_id] = column_id

    column_min_surface = torch.gather(column_min_surface_stack, 1, column_idx)
    column_max_surface = torch.gather(column_max_surface_stack, 1, column_idx)
    column_width_lo = torch.gather(column_width_lo_stack, 1, column_idx)
    column_width_hi = torch.gather(column_width_hi_stack, 1, column_idx)
    column_height_lo = torch.gather(column_height_lo_stack, 1, column_idx)
    column_height_hi = torch.gather(column_height_hi_stack, 1, column_idx)

    wall_points_ordered[..., 0] = column_min_surface + torch.rand(
        (env_count, num_points), device=device
    ) * (column_max_surface - column_min_surface).clamp_min(1e-4)
    wall_points_ordered[..., 1] = column_width_lo + torch.rand(
        (env_count, num_points), device=device
    ) * (column_width_hi - column_width_lo).clamp_min(1e-4)
    wall_points_ordered[..., 2] = column_height_lo + torch.rand(
        (env_count, num_points), device=device
    ) * (column_height_hi - column_height_lo).clamp_min(1e-4)

    thickness_min_face = face_ids == 0
    thickness_max_face = face_ids == 1
    width_min_face = face_ids == 2
    width_max_face = face_ids == 3
    height_min_face = face_ids == 4  # bottom cap
    height_max_face = face_ids == 5  # top cap

    wall_points_ordered[..., 0] = torch.where(thickness_min_face, column_min_surface, wall_points_ordered[..., 0])
    wall_points_ordered[..., 0] = torch.where(thickness_max_face, column_max_surface, wall_points_ordered[..., 0])
    wall_points_ordered[..., 1] = torch.where(width_min_face, column_width_lo, wall_points_ordered[..., 1])
    wall_points_ordered[..., 1] = torch.where(width_max_face, column_width_hi, wall_points_ordered[..., 1])
    # Cap faces: snap the height coord to lo/hi so the box is CLOSED (the thickness/width coords stay
    # uniform, so each cap spans the full thickness x width rectangle).
    wall_points_ordered[..., 2] = torch.where(height_min_face, column_height_lo, wall_points_ordered[..., 2])
    wall_points_ordered[..., 2] = torch.where(height_max_face, column_height_hi, wall_points_ordered[..., 2])

    if params.face_jitter_m > 0.0:
        thickness_face_mask = (thickness_min_face | thickness_max_face).to(torch.float32)
        width_face_mask = (width_min_face | width_max_face).to(torch.float32)
        height_face_mask = (height_min_face | height_max_face).to(torch.float32)
        wall_points_ordered[..., 0] += thickness_face_mask * rand_range(
            -params.face_jitter_m, params.face_jitter_m, (env_count, num_points)
        )
        wall_points_ordered[..., 1] += width_face_mask * rand_range(
            -params.face_jitter_m, params.face_jitter_m, (env_count, num_points)
        )
        wall_points_ordered[..., 2] += height_face_mask * rand_range(
            -params.face_jitter_m, params.face_jitter_m, (env_count, num_points)
        )

    # Keep walls from penetrating the door panel: the inner width face sits only `edge_gap` (<=1cm)
    # outside the panel edge, but face_jitter (up to ~10cm) can push inner-face points across that gap
    # into the slab. Clamp every point to its column's inner edge -- right columns (idx >= num/2) stay
    # at/beyond column_width_lo (= panel_right + gap), left columns stay at/within column_width_hi
    # (= panel_left - gap). This preserves the outward jitter but nothing crosses into the panel width
    # span (walls span the panel's thickness range, but only ever OUTSIDE its width, so no 3D overlap).
    is_right_column = column_idx >= (num_columns // 2)
    wall_points_ordered[..., 1] = torch.where(
        is_right_column,
        torch.maximum(wall_points_ordered[..., 1], column_width_lo),
        torch.minimum(wall_points_ordered[..., 1], column_width_hi),
    )

    wall_points_base = torch.zeros_like(wall_points_ordered)
    wall_points_base.scatter_(
        2,
        axis_order.unsqueeze(1).expand(-1, wall_points_ordered.shape[1], -1),
        wall_points_ordered,
    )

    # Density mode: instead of always emitting `num_points` wall points, keep a count that scales
    # with each env's total wall surface area (points/m^2), so point density stays roughly constant
    # as the walls change size (esp. now that wall height is randomized). The buffer width stays
    # `num_points` (a hard cap); slots beyond the per-env target are set to NaN, which every
    # downstream consumer (depth/lidar render, crops, viser) already treats as "no point".
    if params.point_density_per_m2 is not None:
        total_area = face_area.sum(dim=(1, 2))  # (env_count,) total wall surface area
        target_counts = torch.round(total_area * float(params.point_density_per_m2))
        target_counts = target_counts.clamp(min=float(min(num_columns, num_points)), max=float(num_points)).long()
        slot_valid = torch.arange(num_points, device=device).unsqueeze(0) < target_counts.unsqueeze(1)
        wall_points_base = torch.where(
            slot_valid.unsqueeze(-1), wall_points_base, torch.full_like(wall_points_base, float("nan"))
        )
    return wall_points_base
