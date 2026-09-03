"""Turn captured demonstrations into the fitted constants a planner is written from.

Samples are grouped by ``(variant_class, phase_id, continuity_block, channel)`` and reduced one
bucket at a time. Never across variant_class -- ``capture_schema.load_sessions`` blocks that at the
door, because averaging a +0.15 left offset with a -0.15 right one yields 0.15 m of nothing.

HOW EACH PRIMITIVE IS FITTED

  constant_offset          median (and a 20% trimmed mean, reported alongside). Not a plain mean:
                           one mis-drag in five demonstrations moves a mean by a fifth of the
                           error and a median by nothing.
  linear_gain              numpy.polyfit degree 1 -> (c0, gain). Residual is RMS in the channel's
                           own units.
  rotate_with_theta        joint least squares over the palm_x/palm_y PAIR for (x0, y0), run twice
                           -- once per rotation sense -- keeping whichever residual is lower. The
                           sense is therefore measured, not assumed from handle_side (right-pull
                           and left-push are clockwise; left-pull and right-push are not).
  hold_then_release        three free constants and a kink, so not a polyfit: the level comes from
                           the held plateau, then (hold_until, release_by) are chosen by scanning
                           the candidate grid the observed thetas define, minimising SSE.
  fractional_interpolation endpoint plus step count, with the residual measuring how far the
                           samples fall from the straight line between them.

Rotations are aggregated as the roll/pitch/yaw that go into ``get_rotation_quat``, never as
quaternions -- averaging quaternions would leave the Euler triple the planners are tuned in
unrecoverable, and roll is exactly the channel that ramps with theta.

Every result carries its residual and sample count. Anything above threshold is FLAGGED, not
dropped and not silently accepted; the generator prints flagged fits and marks them in the emitted
source so a bad bucket cannot reach a trajectory unnoticed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from DoorOpening.utils.state_machine.capture_schema import (
    CaptureSchemaError,
    CaptureSession,
    ARM_CHANNELS,
    Keyframe,
    POSITION_CHANNELS,
    ROTATION_CHANNELS,
    VariantClass,
)

# Above these, a fit is flagged for human review. Positions in metres: 1 cm is roughly the spread
# between two careful drags of the same waypoint, so more than that means the bucket is mixing
# things that are not the same waypoint. Rotations in radians: ~3 degrees.
# arm_joint is deliberately TIGHTER than rotation: a task-space rotation can be off a few degrees
# and still grasp, but a recorded posture is only worth pinning if the demonstrations agree on it.
DEFAULT_RESIDUAL_THRESHOLDS = {
    "position": 0.010, "rotation": 0.05, "arm_joint": 0.02, "door": 0.05,
}

TRIM_FRACTION = 0.20


def _channel_kind(channel: str) -> str:
    if channel in POSITION_CHANNELS:
        return "position"
    if channel in ROTATION_CHANNELS:
        return "rotation"
    if channel in ARM_CHANNELS:
        return "arm_joint"
    return "door"


@dataclass
class Spread:
    """Raw sample spread. Not used by the generator; kept for domain-randomization ranges later."""

    n: int
    minimum: float
    maximum: float
    stdev: float
    mad: float  # median absolute deviation -- the robust twin of stdev

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "stdev": round(self.stdev, 6),
            "mad": round(self.mad, 6),
        }


@dataclass
class Aggregate:
    """One fitted channel of one phase: the constants, and everything needed to justify them."""

    phase_id: str
    channel: str
    primitive: str
    continuity_block: str | None
    anchor: str
    mode: str
    anchor_eval_theta: float | None
    constants: dict[str, float]
    method: str
    residual: float
    n_samples: int
    spread: Spread | None = None
    flagged: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return _channel_kind(self.channel)

    def provenance_comment(self) -> str:
        """The one-line justification the generator writes above the emitted constant."""
        bits = [f"{self.n_samples} demo{'s' if self.n_samples != 1 else ''}", self.method]
        if self.primitive == "constant_offset" and self.spread is not None:
            bits.append(f"spread {self.spread.minimum:+.4f}..{self.spread.maximum:+.4f}")
        elif abs(self.residual) < 1e-9:
            bits.append("residual exact")
        else:
            bits.append(f"residual {self.residual:.4g}")
        line = "fitted from " + ", ".join(bits)
        if self.flagged:
            line += "  <-- REVIEW: residual above threshold"
        return line


@dataclass
class ContinuityBlock:
    """A group of keyframes sharing one solve_ik null-space anchor."""

    name: str
    constant_name: str
    joint_pos: dict[str, float]
    n_samples: int
    spread: dict[str, Spread]
    flagged: bool = False


@dataclass
class AggregationResult:
    variant_class: VariantClass
    aggregates: list[Aggregate]
    continuity_blocks: list[ContinuityBlock]
    phase_order: list[str]
    keyframe_plans: dict[str, list[Keyframe]]
    session_paths: list[str]
    # Things that could NOT be fitted yet, in plain language. In the interactive loop these are the
    # agenda for the next capture round, not errors -- an unswept phase simply has no theta axis to
    # fit against until the human sweeps it.
    gaps: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[Aggregate]:
        return [agg for agg in self.aggregates if agg.flagged]

    def by_phase(self, phase_id: str) -> list[Aggregate]:
        return [agg for agg in self.aggregates if agg.phase_id == phase_id]

    def get(self, phase_id: str, channel: str) -> Aggregate | None:
        for agg in self.aggregates:
            if agg.phase_id == phase_id and agg.channel == channel:
                return agg
        return None


# ------------------------------------------------------------------ robust reducers

def _spread(values: list[float]) -> Spread:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    return Spread(
        n=len(values),
        minimum=float(array.min()),
        maximum=float(array.max()),
        stdev=float(array.std(ddof=1)) if len(values) > 1 else 0.0,
        mad=float(np.median(np.abs(array - median))),
    )


def _trimmed_mean(values: list[float], fraction: float = TRIM_FRACTION) -> float:
    ordered = sorted(values)
    cut = int(len(ordered) * fraction)
    kept = ordered[cut: len(ordered) - cut] or ordered
    return float(statistics.fmean(kept))


# ------------------------------------------------------------------------- the fits

def _fit_constant(values: list[float]) -> tuple[dict[str, float], str, float, Spread]:
    median = float(np.median(values))
    spread = _spread(values)
    # Residual reported as the median absolute deviation from the chosen value, so a bucket whose
    # samples disagree gets flagged the same way a bad least-squares fit does.
    residual = float(np.median(np.abs(np.asarray(values, dtype=float) - median)))
    constants = {"value": median, "trimmed_mean": _trimmed_mean(values)}
    return constants, "median", residual, spread


def _fit_linear(thetas: list[float], values: list[float]) -> tuple[dict[str, float], str, float]:
    if len(thetas) < 2:
        raise CaptureSchemaError(
            "linear_gain needs at least two theta samples; capture a sweep, not a single frame."
        )
    gain, c0 = np.polyfit(np.asarray(thetas, float), np.asarray(values, float), 1)
    predicted = c0 + gain * np.asarray(thetas, float)
    residual = float(np.sqrt(np.mean((np.asarray(values, float) - predicted) ** 2)))
    return {"c0": float(c0), "gain": float(gain)}, "least-squares (polyfit deg 1)", residual


def _fit_rotate_xy(
    thetas: list[float], dx: list[float], dy: list[float]
) -> tuple[dict[str, float], str, float]:
    """Recover the closed-door (x0, y0) and the rotation sense from swept offset pairs.

    Both senses are linear in (x0, y0), so each is one least-squares solve; the winner is whichever
    explains the samples better. Stacking dx and dy into a single system is what makes the pair a
    joint fit rather than two independent ones that could disagree about the same offset.
    """
    theta = np.asarray(thetas, float)
    c, s = np.cos(theta), np.sin(theta)
    target = np.concatenate([np.asarray(dx, float), np.asarray(dy, float)])

    best = None
    for sense in ("clockwise", "counterclockwise"):
        if sense == "clockwise":
            design = np.vstack([np.column_stack([c, s]), np.column_stack([-s, c])])
        else:
            design = np.vstack([np.column_stack([c, -s]), np.column_stack([s, c])])
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = float(np.sqrt(np.mean((design @ solution - target) ** 2)))
        if best is None or residual < best[2]:
            best = (sense, solution, residual)

    sense, solution, residual = best
    constants = {"x0": float(solution[0]), "y0": float(solution[1]), "sense": sense}
    return constants, f"least-squares, sense={sense} (both senses tried)", residual


def _fit_hold_then_release(
    thetas: list[float], values: list[float]
) -> tuple[dict[str, float], str, float]:
    """Level from the held plateau, then scan the theta grid for the two knee points.

    Three free constants with a kink between them is not a polynomial, so polyfit cannot express
    it. The candidate set is small -- the knees can only sit between observed thetas -- so an
    exhaustive scan over that grid IS the least-squares solution, not an approximation of one.
    """
    theta = np.asarray(thetas, float)
    value = np.asarray(values, float)
    if len(theta) < 2:
        raise CaptureSchemaError(
            "hold_then_release needs a swept capture; one sample cannot show where the ramp starts."
        )

    peak = float(value.max())
    plateau = value >= 0.95 * peak
    level = float(np.median(value[plateau])) if plateau.any() else peak

    # Candidate knees: the observed thetas, their midpoints, and -- crucially -- the two knees
    # EXTRAPOLATED from the descending ramp. Without the extrapolation a sweep that stops before
    # the lever is fully back (which is the normal case: release_by_theta = 1.25 while a capture
    # sweep may end at 1.1) can only ever place release_by at its last sample, understating the
    # ramp and flagging a fit that was actually fine.
    grid = [theta, (theta[:-1] + theta[1:]) / 2.0]
    ramp = ~plateau & (value > 1e-9)
    if ramp.sum() >= 2:
        slope, intercept = np.polyfit(theta[ramp], value[ramp], 1)
        if slope < -1e-9:
            grid.append(np.array([(level - intercept) / slope, -intercept / slope]))
    grid = np.unique(np.concatenate(grid))
    best = None
    for hold_until in grid:
        for release_by in grid[grid > hold_until]:
            span = release_by - hold_until
            predicted = np.where(
                theta <= hold_until,
                level,
                np.where(theta >= release_by, 0.0, level * (release_by - theta) / span),
            )
            sse = float(np.sum((value - predicted) ** 2))
            if best is None or sse < best[0]:
                best = (sse, float(hold_until), float(release_by))

    if best is None:
        raise CaptureSchemaError("hold_then_release: no valid (hold_until < release_by) pair found.")
    sse, hold_until, release_by = best
    residual = float(np.sqrt(sse / len(value)))
    constants = {"level": level, "hold_until_theta": hold_until, "release_by_theta": release_by}
    return constants, "plateau median + grid-scanned least squares", residual


def _fit_fractional(per_session: list[list[float]]) -> tuple[dict[str, float], str, float]:
    """Endpoint and step count from ordered traverse captures, fitted one demonstration at a time.

    The traverse loops run ``frac = step / steps`` for ``step`` in 1..steps, so the FIRST captured
    sample already sits at frac = 1/steps and the last lands exactly on the endpoint. ``start`` is
    whatever the previous phase left the channel at and is never itself a sample -- but it is the
    intercept of the same line, so both fall out of one least-squares solve per demonstration.

    Fitting per demonstration matters: concatenating several ramps into one list produces a
    sawtooth, and a straight-line fit through a sawtooth reports a residual that says the human
    was sloppy when in fact the aggregation was.
    """
    ends, starts, residuals, lengths = [], [], [], []
    for values in per_session:
        array = np.asarray(values, float)
        steps = len(array)
        if steps < 2:
            raise CaptureSchemaError(
                "fractional_interpolation needs at least two samples across the sweep."
            )
        fractions = np.arange(1, steps + 1, dtype=float) / steps
        slope, intercept = np.polyfit(fractions, array, 1)
        starts.append(float(intercept))
        ends.append(float(intercept + slope))
        predicted = intercept + slope * fractions
        residuals.append(float(np.sqrt(np.mean((array - predicted) ** 2))))
        lengths.append(steps)

    if len(set(lengths)) > 1:
        raise CaptureSchemaError(
            f"fractional_interpolation: demonstrations disagree on step count {sorted(set(lengths))}. "
            "A traverse of 8 steps and one of 6 are different plans, not two samples of one."
        )
    constants = {
        "end": float(np.median(ends)),
        "start_estimate": float(np.median(starts)),
        "steps": float(lengths[0]),
    }
    return constants, "per-demo least squares, median endpoint", float(np.mean(residuals))


# ---------------------------------------------------------------------- aggregation

def _bucket_samples(sessions: list[CaptureSession]):
    """(phase_id, continuity_block, channel) -> list of (session_index, keyframe, ChannelSample).

    The session index is carried because some fits are per-demonstration: pooling several ordered
    sweeps into one sequence turns a ramp into a sawtooth.
    """
    buckets: dict[tuple[str, str | None, str], list[tuple[int, Keyframe, object]]] = {}
    for index, session in enumerate(sessions):
        for keyframe in session.keyframes:
            for channel, sample in keyframe.channels.items():
                key = (keyframe.phase_id, keyframe.continuity_block, channel)
                buckets.setdefault(key, []).append((index, keyframe, sample))
    return buckets


def _dominant(values: list) -> object:
    """The most common tag in a bucket. Buckets should be uniform; this survives one stray tag."""
    return statistics.mode(values) if values else None


def aggregate_sessions(
    sessions: list[CaptureSession],
    *,
    residual_thresholds: dict[str, float] | None = None,
    partial: bool = True,
) -> AggregationResult:
    """Reduce captures to fitted constants.

    ``partial`` (the default) is the interactive-loop mode: a channel that cannot be fitted yet is
    recorded in ``gaps`` and skipped, so one under-demonstrated phase does not block the other
    nine. Pass ``partial=False`` for a final build, where a missing fit should be a hard error.
    """
    if not sessions:
        raise CaptureSchemaError("no capture sessions to aggregate.")
    thresholds = {**DEFAULT_RESIDUAL_THRESHOLDS, **(residual_thresholds or {})}
    variant = sessions[0].variant_class
    gaps: list[str] = []

    buckets = _bucket_samples(sessions)
    aggregates: list[Aggregate] = []

    # rotate_with_theta is a PAIR fit: palm_x and palm_y describe one rotating offset, so they are
    # solved together and share their constants. Collect those phases first and skip them below.
    rotate_phases: dict[tuple[str, str | None], dict] = {}
    for (phase_id, block, channel), samples in buckets.items():
        if channel in ("palm_x", "palm_y") and any(
            s.primitive == "rotate_with_theta" for _, _, s in samples
        ):
            rotate_phases.setdefault((phase_id, block), {})[channel] = samples

    handled: set[tuple[str, str | None, str]] = set()
    for (phase_id, block), pair in rotate_phases.items():
        if set(pair) != {"palm_x", "palm_y"}:
            missing = {"palm_x", "palm_y"} - set(pair)
            message = (
                f"phase {phase_id!r}: rotate_with_theta is a joint fit over palm_x and palm_y, but "
                f"{', '.join(sorted(missing))} was not captured with that primitive. Tag both, or "
                f"neither."
            )
            if not partial:
                raise CaptureSchemaError(message)
            gaps.append(message)
            handled.update((phase_id, block, c) for c in pair)
            continue
        by_theta_x = {(i, kf.id): (kf, s) for i, kf, s in pair["palm_x"]}
        by_theta_y = {(i, kf.id): (kf, s) for i, kf, s in pair["palm_y"]}
        shared_ids = sorted(set(by_theta_x) & set(by_theta_y))
        thetas = [by_theta_x[key][0].theta for key in shared_ids]
        if any(t is None for t in thetas) or len(shared_ids) < 2:
            message = (
                f"phase {phase_id!r}: rotate_with_theta needs at least two keyframes that all carry "
                f"a theta; got {len(shared_ids)}. Sweep the panel and capture across it."
            )
            if not partial:
                raise CaptureSchemaError(message)
            gaps.append(message)
            handled.update((phase_id, block, c) for c in pair)
            continue
        dx = [by_theta_x[key][1].value for key in shared_ids]
        dy = [by_theta_y[key][1].value for key in shared_ids]
        constants, method, residual = _fit_rotate_xy(thetas, dx, dy)
        notes = _collect_notes([by_theta_x[key][0] for key in shared_ids])
        for channel in ("palm_x", "palm_y"):
            sample = pair[channel][0][2]
            aggregates.append(
                Aggregate(
                    phase_id=phase_id,
                    channel=channel,
                    primitive="rotate_with_theta",
                    continuity_block=block,
                    anchor=_dominant([s.anchor for _, _, s in pair[channel]]),
                    mode=_dominant([s.mode for _, _, s in pair[channel]]),
                    anchor_eval_theta=sample.anchor_eval_theta,
                    constants=dict(constants),
                    method=method,
                    residual=residual,
                    n_samples=len(shared_ids),
                    flagged=residual > thresholds["position"],
                    notes=notes,
                )
            )
            handled.add((phase_id, block, channel))

    for key, samples in sorted(buckets.items()):
        if key in handled:
            continue
        phase_id, block, channel = key
        primitive = _dominant([s.primitive for _, _, s in samples])
        values = [s.value for _, _, s in samples]
        thetas = [kf.theta for _, kf, _ in samples]
        kind = _channel_kind(channel)
        spread = None

        try:
            if primitive == "constant_offset":
                constants, method, residual, spread = _fit_constant(values)
            elif primitive == "linear_gain":
                _require_thetas(phase_id, channel, thetas)
                constants, method, residual = _fit_linear(thetas, values)
            elif primitive == "hold_then_release":
                _require_thetas(phase_id, channel, thetas)
                constants, method, residual = _fit_hold_then_release(thetas, values)
            elif primitive == "fractional_interpolation":
                per_session: dict[int, list[tuple[int, float]]] = {}
                for index, keyframe, sample in samples:
                    per_session.setdefault(index, []).append((keyframe.id, sample.value))
                ordered = [
                    [value for _, value in sorted(pairs)]
                    for _, pairs in sorted(per_session.items())
                ]
                constants, method, residual = _fit_fractional(ordered)
            elif primitive == "rotate_with_theta":
                raise CaptureSchemaError(
                    f"phase {phase_id!r} channel {channel!r}: rotate_with_theta is only defined for "
                    "the palm_x/palm_y pair."
                )
            else:
                raise CaptureSchemaError(
                    f"phase {phase_id!r} channel {channel!r}: unknown primitive {primitive!r}. The "
                    "library is fixed at five; if a captured phase genuinely fits none of them, "
                    "stop and say so rather than adding a sixth."
                )
        except CaptureSchemaError as exc:
            if not partial:
                raise
            gaps.append(f"phase {phase_id!r} channel {channel!r}: {exc}")
            continue

        aggregates.append(
            Aggregate(
                phase_id=phase_id,
                channel=channel,
                primitive=primitive,
                continuity_block=block,
                anchor=_dominant([s.anchor for _, _, s in samples]),
                mode=_dominant([s.mode for _, _, s in samples]),
                anchor_eval_theta=_dominant([s.anchor_eval_theta for _, _, s in samples]),
                constants=constants,
                method=method,
                residual=residual,
                n_samples=len(samples),
                spread=spread,
                flagged=residual > thresholds[kind],
                notes=_collect_notes([kf for _, kf, _ in samples]),
            )
        )

    return AggregationResult(
        variant_class=variant,
        aggregates=aggregates,
        continuity_blocks=aggregate_continuity_blocks(sessions, thresholds),
        phase_order=_phase_order(sessions),
        keyframe_plans=_keyframe_plans(sessions, gaps, partial=partial),
        session_paths=[s.source_path for s in sessions],
        gaps=gaps,
    )


def _require_thetas(phase_id: str, channel: str, thetas: list) -> None:
    if any(t is None for t in thetas):
        raise CaptureSchemaError(
            f"phase {phase_id!r} channel {channel!r}: this primitive is fitted against theta, but "
            "at least one contributing keyframe has none. Capture it with the theta sweep control."
        )


def _collect_notes(keyframes: list[Keyframe]) -> list[str]:
    """Human notes, verbatim and de-duplicated. Never paraphrased -- they are evidence."""
    seen, out = set(), []
    for keyframe in keyframes:
        note = (keyframe.notes or "").strip()
        if note and note not in seen:
            seen.add(note)
            out.append(note)
    return out


def _phase_order(sessions: list[CaptureSession]) -> list[str]:
    """Phase order as CAPTURED, not as listed in the vocabulary -- the human's order is the plan."""
    order: list[str] = []
    for session in sessions:
        for keyframe in session.keyframes:
            if keyframe.phase_id not in order:
                order.append(keyframe.phase_id)
    return order


def _keyframe_plans(
    sessions: list[CaptureSession], gaps: list[str], *, partial: bool
) -> dict[str, list[Keyframe]]:
    """Per phase, the keyframe skeleton to emit: how many waypoints, and which are key.

    Structure, not a quantity to average -- a phase that is two waypoints is two waypoints. But in
    the interactive loop a human captures a door at a time, so sessions legitimately disagree on
    how much of the plan they cover. Under ``partial`` the FULLEST session wins per phase and the
    disagreement is recorded as a gap for the next turn to resolve; a hard failure here would stop
    the loop dead over a door that simply has not been demonstrated yet.
    """
    per_phase: dict[str, list[list[Keyframe]]] = {}
    for session in sessions:
        grouped: dict[str, list[Keyframe]] = {}
        for keyframe in session.keyframes:
            grouped.setdefault(keyframe.phase_id, []).append(keyframe)
        for phase_id, keyframes in grouped.items():
            per_phase.setdefault(phase_id, []).append(keyframes)

    plans: dict[str, list[Keyframe]] = {}
    for phase_id, variants in per_phase.items():
        counts = {len(v) for v in variants}
        if len(counts) > 1:
            message = (
                f"phase {phase_id!r}: demonstrations disagree on waypoint count {sorted(counts)}; "
                f"using the fullest ({max(counts)})"
            )
            if not partial:
                raise CaptureSchemaError(
                    message.replace("; using the fullest", " -- fix the odd session out; would use")
                )
            gaps.append(message)
        plans[phase_id] = max(variants, key=len)
    return plans


def aggregate_continuity_blocks(
    sessions: list[CaptureSession], thresholds: dict[str, float]
) -> list[ContinuityBlock]:
    """Median arm posture per block, emitted as a named null-space anchor constant.

    WHAT THIS CONSTANT MEANS. ``reference_joint_pos`` is the null-space ANCHOR the redundant arm
    resolves toward (api.solve_ik applies it at gain 0.5), not a seed -- it picks an IK BRANCH. The
    median of the postures a human actually posed is a fair estimate of the branch they wanted, so
    the derivation is sound. It is NOT equivalent to a hand-tuned anchor: LEFT_PULL_IK_ANCHOR_JOINT_POS
    was derived from a clearance measurement (lifting panda_link3 out of the arx camera arm's
    z 0.65..0.75 band), not from averaging demonstrations. Generated blocks are emitted for review.
    """
    grouped: dict[str, list[dict[str, float]]] = {}
    for session in sessions:
        for keyframe in session.keyframes:
            if keyframe.continuity_block and keyframe.arm_joint_snapshot:
                grouped.setdefault(keyframe.continuity_block, []).append(keyframe.arm_joint_snapshot)

    blocks = []
    for name, snapshots in sorted(grouped.items()):
        joint_names = sorted({j for snap in snapshots for j in snap})
        joint_pos, spread, flagged = {}, {}, False
        for joint in joint_names:
            values = [snap[joint] for snap in snapshots if joint in snap]
            joint_pos[joint] = float(np.median(values))
            spread[joint] = _spread(values)
            if spread[joint].mad > thresholds["rotation"]:
                flagged = True
        blocks.append(
            ContinuityBlock(
                name=name,
                constant_name=f"{name.upper()}_IK_ANCHOR_JOINT_POS",
                joint_pos=joint_pos,
                n_samples=len(snapshots),
                spread=spread,
                flagged=flagged,
            )
        )
    return blocks


__all__ = [
    "Aggregate",
    "AggregationResult",
    "ContinuityBlock",
    "DEFAULT_RESIDUAL_THRESHOLDS",
    "Spread",
    "aggregate_continuity_blocks",
    "aggregate_sessions",
]
