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
    # Flush mounting wall -- a SEPARATE, existence-gated component (NOT a mode that replaces the side
    # boxes). With probability ``flush_prob`` per env a thin slab, centered on the door panel's own
    # thickness band and butted against the panel side edges (no gap), flanks the panel left+right and
    # runs floor-to-above-door -- the flat drywall surface a door is set into. With probability
    # 1-flush_prob it is absent (the glass / open-frame case). It coexists with the box side walls;
    # ``flush_point_fraction`` of the wall point budget is reserved for it. Slab thickness comes from
    # ``flush_thickness_(min|max)_m`` (None -> exactly the panel thickness); the outward spread from
    # each panel edge from ``flush_extent_(min|max)_m``; its vertical span from ``flush_height_(min|max)_m``
    # (None -> floor to just above the panel top, so it fills the policy z-crop).
    flush_prob: float
    flush_point_fraction: float
    flush_thickness_min_m: Optional[float]
    flush_thickness_max_m: Optional[float]
    flush_extent_min_m: float
    flush_extent_max_m: float
    flush_height_min_m: Optional[float]
    flush_height_max_m: Optional[float]

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
        flush_prob = float(cfg.get("flush_prob", 0.7))
        if not 0.0 <= flush_prob <= 1.0:
            raise ValueError("wall_distractors.flush_prob must be in [0, 1].")
        flush_point_fraction = float(cfg.get("flush_point_fraction", 0.5))
        if not 0.0 <= flush_point_fraction <= 1.0:
            raise ValueError("wall_distractors.flush_point_fraction must be in [0, 1].")
        flush_thickness_range = cfg.get("flush_thickness_m")
        if flush_thickness_range is None:
            flush_thickness_min_m = None
            flush_thickness_max_m = None
        else:
            flush_thickness_min_m, flush_thickness_max_m = map(float, flush_thickness_range)
            flush_thickness_min_m = max(1e-4, flush_thickness_min_m)
            flush_thickness_max_m = max(flush_thickness_min_m, flush_thickness_max_m)
        flush_extent_min_m, flush_extent_max_m = map(float, cfg.get("flush_extent_m", [0.6, 1.6]))
        flush_extent_min_m = max(1e-3, flush_extent_min_m)
        flush_extent_max_m = max(flush_extent_min_m, flush_extent_max_m)
        flush_height_range = cfg.get("flush_height_range_m")
        if flush_height_range is None:
            flush_height_min_m = None
            flush_height_max_m = None
        else:
            flush_height_min_m, flush_height_max_m = map(float, flush_height_range)
            flush_height_max_m = max(flush_height_min_m, flush_height_max_m)
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
            flush_prob=flush_prob,
            flush_point_fraction=flush_point_fraction,
            flush_thickness_min_m=flush_thickness_min_m,
            flush_thickness_max_m=flush_thickness_max_m,
            flush_extent_min_m=flush_extent_min_m,
            flush_extent_max_m=flush_extent_max_m,
            flush_height_min_m=flush_height_min_m,
            flush_height_max_m=flush_height_max_m,
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


def _sample_columns_ordered(
    col_min_surface,
    col_max_surface,
    col_width_lo,
    col_width_hi,
    col_height_lo,
    col_height_hi,
    is_right_col,
    num_points,
    face_jitter_m,
    suppress_thickness_jitter,
    point_density_per_m2,
    rand_range,
    device,
):
    """Sample surface points on a set of axis-aligned boxes ("columns") in the ordered frame.

    All ``col_*`` inputs are (E, C) stacks giving each column's [min,max] along thickness (axis 0),
    width (axis 1) and height (axis 2). ``is_right_col`` is a (C,) bool marking columns that attach on
    the panel's +width side (used by the anti-penetration clamp). Returns (E, num_points, 3) ordered
    points; in density mode, slots beyond each env's area-scaled target are NaN.
    """
    env_count, num_columns = col_min_surface.shape
    num_faces = 6

    col_thick_ext = (col_max_surface - col_min_surface).abs()
    col_width_ext = (col_width_hi - col_width_lo).abs()
    col_height_ext = (col_height_hi - col_height_lo).abs()
    # Area of each of the 6 faces per column, ordered [thickness_min, thickness_max, width_min,
    # width_max, height_min (bottom cap), height_max (top cap)]. The two height caps CLOSE the box so
    # short walls don't render open-topped when the camera looks slightly down onto them.
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

    if point_density_per_m2 is not None:
        # Area-WEIGHTED (column, face) selection -> constant points/m^2 across every face.
        probs = face_area.reshape(env_count, num_columns * num_faces).clamp_min(1e-12)
        probs = probs / probs.sum(dim=1, keepdim=True)
        flat_idx = torch.multinomial(probs, num_points, replacement=True)  # (E, num_points)
        column_idx = flat_idx // num_faces
        face_ids = flat_idx % num_faces
    else:
        # Legacy UNIFORM selection (kept for configs without point_density_per_m2).
        face_ids = torch.randint(0, num_faces, (env_count, num_points), device=device)
        column_idx = torch.randint(0, num_columns, (env_count, num_points), device=device)
        if num_points >= num_columns:
            for column_id in range(num_columns):
                column_idx[:, column_id] = column_id

    cms = torch.gather(col_min_surface, 1, column_idx)
    cMs = torch.gather(col_max_surface, 1, column_idx)
    cwlo = torch.gather(col_width_lo, 1, column_idx)
    cwhi = torch.gather(col_width_hi, 1, column_idx)
    chlo = torch.gather(col_height_lo, 1, column_idx)
    chhi = torch.gather(col_height_hi, 1, column_idx)

    pts = torch.empty((env_count, num_points, 3), dtype=torch.float32, device=device)
    pts[..., 0] = cms + torch.rand((env_count, num_points), device=device) * (cMs - cms).clamp_min(1e-4)
    pts[..., 1] = cwlo + torch.rand((env_count, num_points), device=device) * (cwhi - cwlo).clamp_min(1e-4)
    pts[..., 2] = chlo + torch.rand((env_count, num_points), device=device) * (chhi - chlo).clamp_min(1e-4)

    thickness_min_face = face_ids == 0
    thickness_max_face = face_ids == 1
    width_min_face = face_ids == 2
    width_max_face = face_ids == 3
    height_min_face = face_ids == 4  # bottom cap
    height_max_face = face_ids == 5  # top cap

    pts[..., 0] = torch.where(thickness_min_face, cms, pts[..., 0])
    pts[..., 0] = torch.where(thickness_max_face, cMs, pts[..., 0])
    pts[..., 1] = torch.where(width_min_face, cwlo, pts[..., 1])
    pts[..., 1] = torch.where(width_max_face, cwhi, pts[..., 1])
    pts[..., 2] = torch.where(height_min_face, chlo, pts[..., 2])
    pts[..., 2] = torch.where(height_max_face, chhi, pts[..., 2])

    if face_jitter_m > 0.0:
        thickness_face_mask = (thickness_min_face | thickness_max_face).to(torch.float32)
        width_face_mask = (width_min_face | width_max_face).to(torch.float32)
        height_face_mask = (height_min_face | height_max_face).to(torch.float32)
        # A flush slab is only ~panel-thickness deep, so a large face_jitter on the thickness faces
        # would balloon it back into a thick fuzzy cloud and destroy the flat look -- suppress it there.
        t_scale = 0.0 if suppress_thickness_jitter else 1.0
        pts[..., 0] += thickness_face_mask * t_scale * rand_range(-face_jitter_m, face_jitter_m, (env_count, num_points))
        pts[..., 1] += width_face_mask * rand_range(-face_jitter_m, face_jitter_m, (env_count, num_points))
        pts[..., 2] += height_face_mask * rand_range(-face_jitter_m, face_jitter_m, (env_count, num_points))

    # Anti-penetration clamp: right columns stay at/beyond their inner (width_lo) edge, left columns
    # at/within their inner (width_hi) edge, so jitter never pushes a point into the panel's width span.
    is_right = torch.gather(is_right_col.view(1, -1).expand(env_count, num_columns), 1, column_idx)
    pts[..., 1] = torch.where(is_right, torch.maximum(pts[..., 1], cwlo), torch.minimum(pts[..., 1], cwhi))

    if point_density_per_m2 is not None:
        # Keep a per-env count scaling with total wall area (constant points/m^2); NaN the rest.
        total_area = face_area.sum(dim=(1, 2))
        target_counts = torch.round(total_area * float(point_density_per_m2))
        target_counts = target_counts.clamp(min=float(min(num_columns, num_points)), max=float(num_points)).long()
        slot_valid = torch.arange(num_points, device=device).unsqueeze(0) < target_counts.unsqueeze(1)
        pts = torch.where(slot_valid.unsqueeze(-1), pts, torch.full_like(pts, float("nan")))

    return pts


def sample_wall_points_local(
    axis_order: torch.Tensor,
    bbox_min_ordered: torch.Tensor,
    bbox_max_ordered: torch.Tensor,
    num_points: int,
    params: WallDistractorParams,
    device,
    flush_bbox_min_ordered: torch.Tensor = None,
    flush_bbox_max_ordered: torch.Tensor = None,
) -> torch.Tensor:
    """Sample wall-distractor surface points in the door-base local frame.

    Two independent components share the point budget:
      * **Box side walls** -- thick left/right columns (front+back per side), ALWAYS sampled. Their
        height range (``height_range_m``) is deliberately low so they often fall below the policy's
        z-crop and vanish from the policy input, while still often poking into it.
      * **Flush mounting wall** -- a thin slab coplanar with the panel, flanking it left+right, present
        per env with ``flush_prob`` (else absent = glass/open). Gets ``flush_point_fraction`` of the
        budget; absent envs leave those slots NaN.

    Args:
        axis_order / bbox_min_ordered / bbox_max_ordered: per-env (E, 3), from
            :func:`compute_wall_bbox_ordering` (already sliced to the requested envs). Drives the BOX
            side walls -- pass the FULL door bbox (frame+panel+handle) so boxes sit outside the door.
        num_points: buffer width (also the per-env hard cap when density mode is on).
        params: geometry knobs.
        flush_bbox_(min|max)_ordered: optional per-env (E, 3), ordered by the SAME ``axis_order``. Drives
            the FLUSH slab -- pass the PANEL (link_1) bbox so the slab stays coplanar with the panel face
            (the full door bbox includes the handle's ~10 cm protrusion, which would make the slab thick
            and pull its center off the panel). Defaults to the box bbox when not given.
    Returns:
        (E, num_points, 3) points in the door-base frame; empty/absent slots are NaN.
    """
    num_points = int(num_points)
    env_count = int(axis_order.shape[0])
    if num_points <= 0 or env_count == 0:
        return torch.zeros((env_count, 0, 3), dtype=torch.float32, device=device)

    # Ordered coordinates: axis 0 = thickness (normal to slab), 1 = width (L/R), 2 = height.
    thickness_min, width_min, height_min = bbox_min_ordered.unbind(dim=-1)
    thickness_max, width_max, height_max = bbox_max_ordered.unbind(dim=-1)
    thickness_extent = (thickness_max - thickness_min).clamp_min(1e-4)
    width_extent = (width_max - width_min).clamp_min(1e-4)
    thickness_center = 0.5 * (thickness_min + thickness_max)

    # Flush slab frame: the panel bbox when supplied (coplanar with the panel face), else the box bbox.
    if flush_bbox_min_ordered is not None and flush_bbox_max_ordered is not None:
        f_thick_min, f_width_min, f_height_min = flush_bbox_min_ordered.unbind(dim=-1)
        f_thick_max, f_width_max, f_height_max = flush_bbox_max_ordered.unbind(dim=-1)
    else:
        f_thick_min, f_width_min, f_height_min = thickness_min, width_min, height_min
        f_thick_max, f_width_max, f_height_max = thickness_max, width_max, height_max
    f_thick_extent = (f_thick_max - f_thick_min).clamp_min(1e-4)
    f_thick_center = 0.5 * (f_thick_min + f_thick_max)

    def rand_range(low, high, shape):
        return torch.empty(shape, device=device, dtype=torch.float32).uniform_(float(low), float(high))

    # --- Budget split: flush wall (existence-gated) + box side walls (always on). ---
    flush_active = params.flush_prob > 0.0 and params.flush_point_fraction > 0.0
    flush_np = min(num_points, int(round(num_points * params.flush_point_fraction))) if flush_active else 0
    box_np = num_points - flush_np

    parts = []

    # ---- Box side walls: {left, right} x {front, back}, each side's pair butted at a shared seam. ----
    if box_np > 0:
        def sample_seam():
            return thickness_center + rand_range(params.center_offset_min_m, params.center_offset_max_m, (env_count,))

        def sample_column_width():
            if params.side_margin_abs_min_m is not None:
                cw = rand_range(params.side_margin_abs_min_m, params.side_margin_abs_max_m, (env_count,))
            else:
                cw = width_extent * rand_range(params.side_margin_scale_min, params.side_margin_scale_max, (env_count,))
            return cw.clamp_min(1e-4)

        def sample_side_height():
            # ONE height per side (a physical wall has a single top), shared by that side's front+back
            # columns. Drawing it per side (not per column) is what lets a whole side drop below the
            # policy z-crop and vanish from the input -- so "no wall on this side" happens often, while
            # the two sides stay independent so a wall is also often present.
            if params.height_min_m is not None:
                hlo = torch.full((env_count,), params.height_min_m, device=device, dtype=torch.float32)
                hhi = rand_range(params.height_min_m, params.height_max_m, (env_count,))
            else:
                hlo = height_min
                hhi = height_max
            return hlo, hhi

        def sample_box_column(attach_on_right, seam, is_front, hlo, hhi):
            depth = thickness_extent + rand_range(params.depth_min_m, params.depth_max_m, (env_count,))
            if is_front:
                cmin_s, cmax_s = seam, seam + depth
            else:
                cmin_s, cmax_s = seam - depth, seam
            edge_gap = rand_range(params.gap_min_m, params.gap_max_m, (env_count,))
            column_width = sample_column_width()
            if attach_on_right:
                inner = width_max + edge_gap
                outer = inner + column_width
            else:
                inner = width_min - edge_gap
                outer = inner - column_width
            wlo = torch.minimum(inner, outer)
            whi = torch.maximum(inner, outer)
            return cmin_s, cmax_s, wlo, whi, hlo, hhi

        left_seam = sample_seam()
        right_seam = sample_seam()
        left_hlo, left_hhi = sample_side_height()
        right_hlo, right_hhi = sample_side_height()
        variants = [
            sample_box_column(attach_on_right, seam, is_front, hlo, hhi)
            for attach_on_right, seam, (hlo, hhi) in (
                (False, left_seam, (left_hlo, left_hhi)),
                (True, right_seam, (right_hlo, right_hhi)),
            )
            for is_front in (False, True)
        ]  # order [left-back, left-front, right-back, right-front]
        box_pts = _sample_columns_ordered(
            *[torch.stack([v[i] for v in variants], dim=1) for i in range(6)],
            is_right_col=torch.tensor([False, False, True, True], device=device),
            num_points=box_np,
            face_jitter_m=params.face_jitter_m,
            suppress_thickness_jitter=False,
            point_density_per_m2=params.point_density_per_m2,
            rand_range=rand_range,
            device=device,
        )
        parts.append(box_pts)

    # ---- Flush mounting wall: thin coplanar slab, left+right of the panel, existence-gated. ----
    if flush_np > 0:
        if params.flush_thickness_min_m is not None:
            flush_t = rand_range(params.flush_thickness_min_m, params.flush_thickness_max_m, (env_count,))
        else:
            flush_t = f_thick_extent  # exactly the panel thickness -> both faces coplanar
        half_t = 0.5 * flush_t
        f_tmin = f_thick_center - half_t
        f_tmax = f_thick_center + half_t
        extent = rand_range(params.flush_extent_min_m, params.flush_extent_max_m, (env_count,)).clamp_min(1e-3)
        if params.flush_height_min_m is not None:
            f_hlo = torch.full((env_count,), params.flush_height_min_m, device=device, dtype=torch.float32)
            f_hhi = torch.full((env_count,), params.flush_height_max_m, device=device, dtype=torch.float32)
        else:
            f_hlo = torch.minimum(f_height_min, torch.zeros_like(f_height_min))  # down to the floor
            f_hhi = f_height_max + 0.3  # just above the door top -> fills the policy z-crop
        # Two columns [left-flush, right-flush]; slab butts the panel edge (no gap) and extends outward.
        cmin = torch.stack([f_tmin, f_tmin], dim=1)
        cmax = torch.stack([f_tmax, f_tmax], dim=1)
        wlo = torch.stack([f_width_min - extent, f_width_max], dim=1)
        whi = torch.stack([f_width_min, f_width_max + extent], dim=1)
        hlo = torch.stack([f_hlo, f_hlo], dim=1)
        hhi = torch.stack([f_hhi, f_hhi], dim=1)
        flush_pts = _sample_columns_ordered(
            cmin, cmax, wlo, whi, hlo, hhi,
            is_right_col=torch.tensor([False, True], device=device),
            num_points=flush_np,
            face_jitter_m=params.face_jitter_m,
            suppress_thickness_jitter=True,
            point_density_per_m2=params.point_density_per_m2,
            rand_range=rand_range,
            device=device,
        )
        # Existence gate: absent (glass/open) envs get all-NaN flush points this episode.
        flush_present = torch.rand((env_count,), device=device) < params.flush_prob
        flush_pts = torch.where(
            flush_present.view(env_count, 1, 1), flush_pts, torch.full_like(flush_pts, float("nan"))
        )
        parts.append(flush_pts)

    wall_points_ordered = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    # Scatter ordered (thickness, width, height) axes back to the original door-base axes. NaN slots
    # scatter to NaN, which every downstream consumer treats as "no point".
    wall_points_base = torch.zeros_like(wall_points_ordered)
    wall_points_base.scatter_(
        2,
        axis_order.unsqueeze(1).expand(-1, wall_points_ordered.shape[1], -1),
        wall_points_ordered,
    )
    return wall_points_base
