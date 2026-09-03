"""Write a planner module from aggregated demonstrations, then check it actually solves.

    python -m DoorOpening.utils.state_machine.synthesize_planner \
        logs/workbench/scratch_door__rnd_02/capture_*.json \
        --out source/DoorOpening/utils/state_machine/offline_pull_door_generated.py

Input is one or more capture sessions for the SAME variant_class (mixing classes is refused in
capture_schema.load_sessions). Output is a NEW file -- offline_pull_door.py and offline_push_door.py
are never read for writing and never touched.

WHAT THE OUTPUT LOOKS LIKE
The same shape as the hand-written planners, because that is the format that has been tuned and
reviewed: one top-level function per variant (not one parametrized function -- the existing
docstrings say the separation is deliberate so tuning stays local), named offset variables in
``{phase}_{channel}_{quantity}`` order declared before use, ``# ----`` banners with a
``Step N: description`` header, and ``_make_pose`` / ``_append_state`` / ``solve_ik`` /
``get_hinge_pos`` / ``get_board_pos`` called exactly as the originals call them.

Every emitted constant carries a provenance comment: how many demonstrations it came from, how it
was reduced, and the residual or sample spread. Human notes from contributing keyframes are
appended verbatim and attributed, never paraphrased -- a note is evidence about a waypoint, and
rewriting it loses the thing that made it worth recording.

VERIFICATION
``--verify`` (default on) imports the generated function, wraps ``api.solve_ik`` with a recorder,
runs it against a real door, and reports per-keyframe ``success`` and ``best_error_norm`` from the
solver's own debug_info -- surfaced here rather than left as a printed warning. It then runs
``collocate_and_playback`` and counts collapsed adjacent waypoints, the signature of a target the
IK best-efforted onto its neighbour. Infeasible keyframes are reported and written into the file's
header as an UNRESOLVED block; the file is never silently handed over as if it were clean.

No collision checking. Out of scope by design -- that is RL's job.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import textwrap

from DoorOpening.utils.state_machine.capture_schema import (
    CaptureSchemaError,
    ARM_CHANNELS,
    CaptureSession,
    Keyframe,
    load_sessions,
)
from DoorOpening.utils.state_machine.offset_aggregation import (
    Aggregate,
    AggregationResult,
    aggregate_sessions,
)

# phase_id -> (step title, variable prefix). The prefixes match the hand-written planners' own
# names (Step 6 is `retreat_*`, the pull sweep is `pull_*`), so a generated file and a tuned one
# read the same and a constant can be diffed across them by name.
PHASE_META = {
    "pregrasp":      ("Pregrasp", "pregrasp"),
    "grasp":         ("Move to grasp", "grasp"),
    "unlatch":       ("Rotate hinge (unlatch)", "unlatch"),
    "pull_sweep":    ("Pull door open", "pull"),
    "release":       ("Move to the blocking base pose while releasing the hinge", "release"),
    "retract":       ("Retract the arm clear of the panel", "retreat"),
    "push_approach": ("Approach the panel for the push", "push_approach"),
    "push_contact":  ("Push contact on the panel", "push_contact"),
    "push_final":    ("Push the panel to full open", "push_open"),
    "traverse":      ("Traverse through the doorway", "traverse"),
}

ANCHOR_CALLS = {
    "hinge": "get_hinge_pos",
    "handle_bar": "get_handle_bar_pos",
    "board_center": "get_board_pos",
    "board_edge": "get_board_edge",
}

AXIS_INDEX = {"base_x": 0, "base_y": 1, "palm_x": 0, "palm_y": 1, "palm_z": 2}


class GeneratorError(Exception):
    pass


# ------------------------------------------------------------------ name helpers

def _var(prefix: str, channel: str, quantity: str) -> str:
    """`{phase}_{channel}_{quantity}`, the naming pattern the existing planners follow."""
    return f"{prefix}_{channel}_{quantity}" if quantity else f"{prefix}_{channel}"


def _fmt(value: float) -> str:
    """Round-trippable literal, with pi spelled as pi where that is plainly what it is.

    Least squares on clean data lands on things like 1.6e-18 rather than 0.0; those are zero with
    float dust on them, and emitting the dust makes a generated constant look tuned when it is not.
    """
    if abs(value) < 1e-12:
        return "0.0"
    for name, magnitude in (("math.pi", math.pi), ("math.pi / 2", math.pi / 2)):
        for sign, text in ((1.0, name), (-1.0, f"-{name}")):
            if abs(value - sign * magnitude) < 1e-6:
                return text
    text = f"{value:.6g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def _fmt_residual(value: float) -> str:
    """Residuals below float noise read as 'exact', because 3.2e-17 is not a measurement."""
    return "exact" if abs(value) < 1e-9 else f"{value:.4g}"


# Emission order within a phase, so a generated block reads base -> palm -> rotation -> door the
# way the hand-written ones do, instead of in whatever order the channels were captured.
CHANNEL_ORDER = (
    "base_x", "base_y", "base_yaw",
    "palm_x", "palm_y", "palm_z",
    "rot_roll", "rot_pitch", "rot_yaw",
    "door_panel", "door_lever",
)

# Phases where the hand-written planners deliberately pass base_pose=None -- the base is meant to
# hold the pose an earlier step parked it at (pull:531 retract, pull:576 push). Everywhere else the
# carried base_target_pose is passed, because solve_ik optimizes the base joints too and would
# otherwise let them drift.
BASE_HELD_PHASES = frozenset({"retract", "push_approach", "push_contact", "push_final"})


# --------------------------------------------------------------------- the emitter

class _Emitter:
    def __init__(self, result: AggregationResult):
        self.result = result
        self.lines: list[str] = []
        self.indent = "    "

    def line(self, text: str = "") -> None:
        self.lines.append(f"{self.indent}{text}" if text else "")

    def banner(self, step_no: int, title: str) -> None:
        self.line()
        self.line("# -------------------------")
        self.line(f"# Step {step_no}: {title}")
        self.line("# -------------------------")

    def provenance(self, agg: Aggregate) -> None:
        self.line(f"# {agg.provenance_comment()}")

    def phase_notes(self, aggs: list[Aggregate]) -> None:
        """Human notes for the phase, once, verbatim and attributed.

        Per-constant would repeat one note across every channel of the waypoint it was written
        about, which buries the block it is meant to explain.
        """
        seen, notes = set(), []
        for agg in aggs:
            for note in agg.notes:
                if note not in seen:
                    seen.add(note)
                    notes.append(note)
        for note in notes:
            for index, wrapped in enumerate(textwrap.wrap(note, width=92)):
                self.line(f"# {'human note: ' if index == 0 else '  '}{wrapped}")

    def emit_anchors(self, aggs: dict[str, Aggregate], *, in_loop: bool) -> dict[tuple, str]:
        """Evaluate each distinct anchor ONCE per phase and reuse it, as the originals do.

        offline_pull_door.py resolves `handle_pos` once and clones it for both the base and the
        palm target; calling get_hinge_pos twice would resample the door point cloud for a value
        that cannot have changed.
        """
        wanted: dict[tuple, str] = {}
        for channel in CHANNEL_ORDER:
            agg = aggs.get(channel)
            if agg is None or channel not in AXIS_INDEX or agg.anchor not in ANCHOR_CALLS:
                continue
            key = (agg.anchor, agg.anchor_eval_theta)
            if key in wanted:
                continue
            base = "handle_pos" if agg.anchor in ("hinge", "handle_bar") else "board_pos"
            name = base if base not in wanted.values() else f"{base}_{len(wanted)}"
            wanted[key] = name
            if agg.anchor_eval_theta is not None:
                self.line()
                self.line(
                    f"# Anchor evaluated at a VIRTUAL door angle "
                    f"({_fmt(agg.anchor_eval_theta)}), not the commanded one."
                )
                q_expr = (
                    f"torch.tensor([{_fmt(agg.anchor_eval_theta)}, 0.0], device=device).unsqueeze(0)"
                )
            else:
                self.line()
                q_expr = "q_door.unsqueeze(0)"
            self.line(f"{name} = {ANCHOR_CALLS[agg.anchor]}(")
            self.line("    door_urdf_path,")
            self.line("    door_initial_pose,")
            self.line(f"    {q_expr},")
            self.line(").to(device)")
        return wanted

    def const(self, name: str, value: float, agg: Aggregate | None = None) -> None:
        if agg is not None:
            self.provenance(agg)
        self.line(f"{name} = {_fmt(value)}")

    # ------------------------------------------------------------- constant blocks

    def emit_constants(self, phase_id: str, prefix: str, aggs: list[Aggregate]) -> dict[str, str]:
        """Declare every named constant this phase needs, before any of it is used."""
        names: dict[str, str] = {}
        ordered = sorted(aggs, key=lambda a: CHANNEL_ORDER.index(a.channel))
        for agg in ordered:
            channel = agg.channel
            if agg.primitive == "constant_offset":
                quantity = "offset" if channel in AXIS_INDEX else ""
                name = _var(prefix, channel, quantity)
                self.const(name, agg.constants["value"], agg)
                names[channel] = name
            elif agg.primitive == "linear_gain":
                base = _var(prefix, channel, "offset" if channel in AXIS_INDEX else "base")
                gain = _var(prefix, channel, "gain" if channel in AXIS_INDEX else "per_theta")
                self.provenance(agg)
                self.line(f"{base} = {_fmt(agg.constants['c0'])}")
                self.line(f"{gain} = {_fmt(agg.constants['gain'])}")
                names[channel] = f"{base}|{gain}"
            elif agg.primitive == "rotate_with_theta":
                name = _var(prefix, channel, "offset_closed")
                self.const(name, agg.constants["x0" if channel == "palm_x" else "y0"], agg)
                names[channel] = name
            elif agg.primitive == "hold_then_release":
                level = f"{prefix}_hinge_angle"
                hold = f"{prefix}_hinge_hold_until_theta"
                release = f"{prefix}_hinge_release_by_theta"
                self.provenance(agg)
                self.line(f"{level} = {_fmt(agg.constants['level'])}")
                self.line(f"{hold} = {_fmt(agg.constants['hold_until_theta'])}")
                self.line(f"{release} = {_fmt(agg.constants['release_by_theta'])}")
                names[channel] = f"{level}|{hold}|{release}"
            elif agg.primitive == "fractional_interpolation":
                name = _var(prefix, channel, "end")
                self.const(name, agg.constants["end"], agg)
                names[channel] = name
        return names

    # -------------------------------------------------------------- target blocks

    def _anchor_expr(self, agg: Aggregate, anchors: dict[tuple, str]) -> str:
        anchor = agg.anchor
        if anchor in ANCHOR_CALLS:
            return f"{anchors[(anchor, agg.anchor_eval_theta)]}.clone()"
        if anchor == "prev_palm":
            return "palm_target_pose[:, :3].clone()"
        if anchor == "prev_base":
            return "base_target_pos.clone()"
        if anchor == "door_origin":
            return "door_initial_pose[:, :3].to(device).clone()"
        return "torch.zeros(1, 3, device=device)"

    def emit_position_target(
        self,
        target: str,                   # "base" or "palm"
        aggs: dict[str, Aggregate],
        names: dict[str, str],
        anchors: dict[tuple, str],
        *,
        theta_expr: str | None = None,
    ) -> bool:
        channels = [c for c in aggs if c.startswith(target) and c in AXIS_INDEX]
        if not channels:
            return False

        anchor_agg = aggs[channels[0]]
        pos_var = f"{target}_target_pos"
        self.line()
        self.line(f"{pos_var} = {self._anchor_expr(anchor_agg, anchors)}")

        for channel in sorted(channels, key=lambda c: AXIS_INDEX[c]):
            agg = aggs[channel]
            idx = AXIS_INDEX[channel]
            op = "=" if agg.mode == "world_absolute" else "+="
            if agg.primitive == "constant_offset" or agg.primitive == "fractional_interpolation":
                self.line(f"{pos_var}[:, {idx}] {op} {names[channel]}")
            elif agg.primitive == "linear_gain":
                base, gain = names[channel].split("|")
                self.line(f"{pos_var}[:, {idx}] {op} linear_gain({theta_expr}, {base}, {gain})")
            elif agg.primitive == "rotate_with_theta":
                pass  # emitted as a pair below
        return True

    def emit_rotate_pair(self, aggs: dict[str, Aggregate], names: dict[str, str], theta_expr: str):
        if "palm_x" not in aggs or aggs["palm_x"].primitive != "rotate_with_theta":
            return
        sense = aggs["palm_x"].constants["sense"]
        self.line()
        self.line("palm_dx, palm_dy = rotate_xy(")
        self.line(f"    {names['palm_x']},")
        self.line(f"    {names['palm_y']},")
        self.line("    theta,")
        self.line(f'    "{sense}",')
        self.line(")")

    def emit_rotation(self, aggs: dict[str, Aggregate], names: dict[str, str], prefix: str,
                      theta_expr: str | None, var: str) -> bool:
        rot = {c: aggs[c] for c in ("rot_roll", "rot_pitch", "rot_yaw") if c in aggs}
        if not rot:
            return False
        parts = []
        for channel in ("rot_roll", "rot_pitch", "rot_yaw"):
            if channel not in rot:
                parts.append("0.0")
                continue
            agg = rot[channel]
            if agg.primitive == "linear_gain":
                base, gain = names[channel].split("|")
                parts.append(f"linear_gain({theta_expr}, {base}, {gain})")
            else:
                parts.append(names[channel])
        self.line()
        self.line(f"{var} = get_rotation_quat(")
        for part in parts:
            self.line(f"    {part},")
        self.line("    device,")
        self.line(")")
        return True

    def emit_solve_and_append(self, keyframe: Keyframe, *, base: bool, palm: bool,
                              block: str | None, base_available: bool = False) -> None:
        # A phase that sets no base channel still passes the carried base_target_pose, because
        # solve_ik optimizes the base joints too and would otherwise let the base drift off the
        # pose the approach parked it at. Only the phases that deliberately hold it pass None.
        if base:
            base_arg = "base_target_pose"
        elif base_available and keyframe.phase_id not in BASE_HELD_PHASES:
            base_arg = "base_target_pose"
        else:
            base_arg = "None"
        self.line()
        self.line("q_robot[:10] = solve_ik(")
        self.line("    robot_urdf_path,")
        self.line("    q_robot[:10],")
        self.line(f"    palm_pose={'palm_target_pose' if palm else 'None'},")
        self.line(f"    base_pose={base_arg},")
        self.line("    robot_initial_pose=robot_initial_pose,")
        if block:
            self.line(f"    reference_joint_pos={block},")
        if keyframe.num_attempts != 8:
            self.line(
                f"    num_attempts={keyframe.num_attempts},  # continuity-critical: one seed, so a "
                "hard frame stays"
            )
            self.line("    # near its predecessor instead of jumping to a distant IK branch.")
        self.line(")[0]")
        if keyframe.gripper_width is not None:
            self.line(f"_set_gripper(q_robot, {_fmt(keyframe.gripper_width)})")
        self.line()
        self.line("_append_state(")
        self.line("    robot_traj,")
        self.line("    door_traj,")
        self.line("    key_idx_in_key_indices,")
        self.line("    q_robot,")
        self.line("    q_door,")
        self.line(f"    mark_keyframe={keyframe.mark_keyframe},")
        self.line(")")


# ------------------------------------------------------------------ phase emitters

def _emit_static_phase(emitter: _Emitter, phase_id: str, aggs: list[Aggregate],
                       plan: list[Keyframe], block: str | None, state: dict) -> None:
    prefix = PHASE_META[phase_id][1]
    emitter.phase_notes(aggs)
    names = emitter.emit_constants(phase_id, prefix, aggs)
    by_channel = {agg.channel: agg for agg in aggs}
    anchors = emitter.emit_anchors(by_channel, in_loop=False)

    for keyframe in plan:
        if "door_panel" in by_channel or "door_lever" in by_channel:
            panel = names.get("door_panel", "q_door[0].item()")
            lever = names.get("door_lever", "0.0")
            emitter.line()
            emitter.line(f"q_door = torch.tensor([{panel}, {lever}], device=device)")

        base = emitter.emit_position_target("base", by_channel, names, anchors)
        if base:
            yaw = names.get("base_yaw")
            if yaw is None:
                rot = "base_target_rot"
            else:
                state["base_yaw_expr"] = f"robot_initial_yaw.item() + {yaw}"
                rot = f"get_rotation_quat(0.0, 0.0, {state['base_yaw_expr']}, device)"
            emitter.line(f"base_target_pose = _make_pose(base_target_pos, {rot})")
            state["base_available"] = True

        palm = emitter.emit_position_target("palm", by_channel, names, anchors)
        if palm:
            has_rot = emitter.emit_rotation(by_channel, names, prefix, None, "palm_target_rot")
            rot = "palm_target_rot" if has_rot else "default_palm_rot"
            emitter.line(f"palm_target_pose = _make_pose(palm_target_pos, {rot})")
            state["palm_available"] = True

        emitter.emit_solve_and_append(
            keyframe, base=base, palm=state["palm_available"] and (palm or state["palm_available"]),
            block=block, base_available=state["base_available"],
        )


def _emit_swept_phase(emitter: _Emitter, phase_id: str, aggs: list[Aggregate],
                      plan: list[Keyframe], block: str | None, state: dict) -> None:
    prefix = PHASE_META[phase_id][1]
    emitter.phase_notes(aggs)
    thetas = sorted(kf.theta for kf in plan if kf.theta is not None)
    step = min((b - a) for a, b in zip(thetas, thetas[1:])) if len(thetas) > 1 else 0.1

    emitter.line(f"# theta range and step taken from the captured sweep ({len(thetas)} samples).")
    emitter.line(f"{prefix}_theta_start = {_fmt(thetas[0])}")
    emitter.line(f"{prefix}_theta_stop = {_fmt(thetas[-1])}")
    emitter.line(f"{prefix}_theta_step = {_fmt(step)}")
    emitter.line()

    names = emitter.emit_constants(phase_id, prefix, aggs)
    by_channel = {agg.channel: agg for agg in aggs}

    emitter.line()
    emitter.line("theta_values = torch.arange(")
    emitter.line(f"    {prefix}_theta_start,")
    emitter.line(f"    {prefix}_theta_stop + 1e-6,")
    emitter.line(f"    {prefix}_theta_step,")
    emitter.line("    device=device,")
    emitter.line(")")
    emitter.line()
    emitter.line("for theta in theta_values:")
    emitter.indent = "        "

    lever = names.get("door_lever")
    if lever and "|" in lever:
        level, hold, release = lever.split("|")
        emitter.line("q_door = torch.tensor(")
        emitter.line("    [")
        emitter.line("        theta.item(),")
        emitter.line("        hold_then_release(")
        emitter.line("            theta.item(),")
        emitter.line(f"            {level},")
        emitter.line(f"            {hold},")
        emitter.line(f"            {release},")
        emitter.line("        ),")
        emitter.line("    ],")
        emitter.line("    device=device,")
        emitter.line(")")
    else:
        emitter.line("q_door = torch.tensor([theta.item(), 0.0], device=device)")

    anchors = emitter.emit_anchors(by_channel, in_loop=True)
    base = emitter.emit_position_target(
        "base", by_channel, names, anchors, theta_expr="theta.item()"
    )
    if base:
        emitter.line("base_target_pose = _make_pose(base_target_pos, base_target_rot)")
        state["base_available"] = True

    emitter.emit_rotate_pair(by_channel, names, "theta.item()")
    palm = emitter.emit_position_target(
        "palm", by_channel, names, anchors, theta_expr="theta.item()"
    )
    if palm:
        if "palm_x" in by_channel and by_channel["palm_x"].primitive == "rotate_with_theta":
            emitter.line("palm_target_pos[:, 0] += palm_dx")
            emitter.line("palm_target_pos[:, 1] += palm_dy")
        has_rot = emitter.emit_rotation(by_channel, names, prefix, "theta.item()", "palm_target_rot")
        rot = "palm_target_rot" if has_rot else "default_palm_rot"
        emitter.line(f"palm_target_pose = _make_pose(palm_target_pos, {rot})")
        state["palm_available"] = True

    loop_keyframe = Keyframe(id=-1, phase_id=phase_id, mark_keyframe=False, num_attempts=1,
                             gripper_width=plan[0].gripper_width)
    emitter.emit_solve_and_append(
        loop_keyframe, base=base, palm=palm or state["palm_available"], block=block,
        base_available=state["base_available"],
    )
    emitter.indent = "    "
    emitter.line()
    emitter.line("# The sweep is ONE spline segment: only its final frame is a keyframe, so")
    emitter.line("# collocate_and_playback interpolates through the arc instead of across it.")
    emitter.line("key_idx_in_key_indices.append(len(robot_traj) - 1)")


def _emit_traverse_phase(emitter: _Emitter, phase_id: str, aggs: list[Aggregate],
                         plan: list[Keyframe], block: str | None, state: dict) -> None:
    prefix = PHASE_META[phase_id][1]
    emitter.phase_notes(aggs)
    names = emitter.emit_constants(phase_id, prefix, aggs)
    by_channel = {agg.channel: agg for agg in aggs}
    steps = int(next(
        (agg.constants["steps"] for agg in aggs if agg.primitive == "fractional_interpolation"),
        len(plan),
    ))
    emitter.line(f"{prefix}_steps = {steps}")
    emitter.line()
    emitter.line("start_base_x = base_target_pos[:, 0].clone()")
    emitter.line("start_base_y = base_target_pos[:, 1].clone()")
    if "base_yaw" in names:
        emitter.line(f"start_base_yaw = {state['base_yaw_expr']}")
    emitter.line()
    emitter.line(f"for traverse_step in range(1, {prefix}_steps + 1):")
    emitter.indent = "        "
    emitter.line(f"frac = traverse_step / {prefix}_steps")
    emitter.line("base_target_pos = base_target_pos.clone()")
    if "base_x" in names:
        emitter.line(f"base_target_pos[:, 0] = lerp(start_base_x, {names['base_x']}, frac)")
    if "base_y" in names:
        emitter.line(f"base_target_pos[:, 1] = lerp(start_base_y, {names['base_y']}, frac)")
    if "base_yaw" in names:
        emitter.line(f"step_yaw = lerp(start_base_yaw, {names['base_yaw']}, frac)")
        emitter.line("base_target_pose = _make_pose(")
        emitter.line("    base_target_pos, get_rotation_quat(0.0, 0.0, step_yaw, device)")
        emitter.line(")")
    else:
        emitter.line("base_target_pose = _make_pose(base_target_pos, base_target_rot)")

    loop_keyframe = Keyframe(id=-1, phase_id=phase_id, mark_keyframe=False, num_attempts=1,
                             gripper_width=plan[0].gripper_width)
    emitter.emit_solve_and_append(
        loop_keyframe, base=True, palm=state["palm_available"], block=block, base_available=True,
    )
    emitter.indent = "    "
    emitter.line()
    emitter.line("# The traverse is one spline segment; only its final frame is a keyframe.")
    emitter.line("key_idx_in_key_indices.append(len(robot_traj) - 1)")


# ------------------------------------------------------------------- the whole file

HEADER = '''"""GENERATED by synthesize_planner.py -- review before trusting, do not hand-edit in place.

Aggregated from {n_sessions} capture session(s) for variant_class {variant}:
{sessions}

Same shape as offline_pull_door.py / offline_push_door.py: every waypoint is an anchor position
plus tuned offsets, run through solve_ik and appended with _append_state. The anchor functions in
api.py absorb per-door geometry, which is why these offsets generalize across an asset set.

Every constant below carries the number of demonstrations it was fitted from, how it was reduced,
and its residual or sample spread. Human notes are reproduced verbatim.
{unresolved}"""

import math

import torch
{imports}'''

# Only emitted when the generated body actually references them -- an unused import in a file that
# is going to be reviewed line by line is noise that makes the reader doubt the rest.
OPTIONAL_IMPORTS = [
    ("isaaclab.utils.math", ["euler_xyz_from_quat"]),
    ("DoorOpening.constants.robot_constants",
     ["FRANKA_DEFAULT_JOINT_POS", "FRANKA_JOINT_NAMES", "GRIPPER_OPEN_WIDTH"]),
    ("DoorOpening.utils.state_machine.api",
     ["get_board_edge", "get_board_pos", "get_handle_bar_pos", "get_hinge_pos", "solve_ik"]),
    ("DoorOpening.utils.state_machine.primitives",
     ["_append_state", "_init_planner_state", "_make_pose", "_set_gripper", "get_rotation_quat",
      "hold_then_release", "lerp", "linear_gain", "rotate_xy"]),
]


def _import_block(body: str) -> str:
    """Import exactly the names the generated body uses."""
    blocks = []
    for module, names in OPTIONAL_IMPORTS:
        used = [name for name in names if re.search(rf"\b{re.escape(name)}\b", body)]
        if not used:
            continue
        if len(used) == 1:
            blocks.append(f"from {module} import {used[0]}")
        else:
            joined = "".join(f"    {name},\n" for name in used)
            blocks.append(f"from {module} import (\n{joined})")
    return "\n".join(blocks) + "\n"


def _reject_unemittable(result: AggregationResult) -> None:
    """Fail on captured channels the emitter has no code path for.

    Arm-joint channels are capturable today so the demonstrations can be COLLECTED, but nothing
    below knows how to turn a posture into a waypoint yet. Dropping them silently would produce a
    planner that looks complete and quietly ignores every posture the human pinned.
    """
    pinned = sorted({agg.channel for agg in result.aggregates if agg.channel in ARM_CHANNELS})
    if pinned:
        phases = sorted({agg.phase_id for agg in result.aggregates
                         if agg.channel in ARM_CHANNELS})
        raise GeneratorError(
            f"captured arm-joint channels ({', '.join(pinned)}) in phase(s) "
            f"{', '.join(phases)}, but the generator cannot emit a joint-space waypoint yet. "
            "Capture them for the record, or drop them from the spec to generate."
        )


def generate_source(result: AggregationResult, *, unresolved: list[str] | None = None) -> str:
    _reject_unemittable(result)
    variant = result.variant_class
    sessions = "\n".join(f"  {path}" for path in result.session_paths)
    unresolved_block = ""
    if unresolved:
        listing = "\n".join(f"  - {item}" for item in unresolved)
        unresolved_block = (
            "\nUNRESOLVED -- these keyframes did not solve cleanly and are NOT accepted:\n"
            f"{listing}\n"
        )

    out = []

    block_name = None
    for block in result.continuity_blocks:
        block_name = block.constant_name
        out.append("")
        out.append(
            f"# Null-space posture anchor for this planner's IK (see api.solve_ik). Median over\n"
            f"# {block.n_samples} demonstrated arm postures in continuity block {block.name!r}.\n"
            f"#\n"
            f"# reference_joint_pos is the null-space ANCHOR the redundant arm resolves toward, at\n"
            f"# gain 0.5 -- it picks an IK BRANCH, and the branch a human posed by hand is exactly\n"
            f"# what this estimates. It is NOT the same kind of value as a hand-derived anchor such\n"
            f"# as LEFT_PULL_IK_ANCHOR_JOINT_POS, which came from a link-clearance measurement\n"
            f"# rather than from demonstrations. Review this against a rendered trajectory."
            + ("\n# REVIEW: demonstrated postures disagree by more than the review threshold."
               if block.flagged else "")
        )
        out.append(f"{block.constant_name} = {{")
        out.append("    **FRANKA_DEFAULT_JOINT_POS,")
        for joint, value in block.joint_pos.items():
            out.append(f"    {joint!r}: {_fmt(value)},")
        out.append("}")

    out.append("")
    out.append("")
    out.append(f"def {variant.function_name()}(")
    out.append("    robot_urdf_path,")
    out.append("    door_urdf_path,")
    out.append("    robot_initial_pose,   # (1, 7) world")
    out.append("    door_initial_pose,    # (1, 7) world")
    out.append("    robot_initial_q,      # (ndof,)")
    out.append("    door_initial_q,       # (2,) [board, hinge]")
    out.append('    device="cpu",')
    out.append("):")
    out.append('    """')
    out.append(
        f"    Offline planner for a {variant.handle_side}-side handle, "
        f"{variant.opening_direction}-type door."
    )
    out.append("")
    out.append(
        f"    This is intentionally separate from the other variants' functions so all"
    )
    out.append(f"    {variant.handle_side}-door tuning stays local and obvious.")
    out.append('    """')

    emitter = _Emitter(result)
    emitter.line()
    emitter.line("q_robot, q_door, robot_traj, door_traj, key_idx_in_key_indices = _init_planner_state(")
    emitter.line("    robot_initial_q, door_initial_q")
    emitter.line(")")
    emitter.line()
    emitter.line("base_target_rot = robot_initial_pose[:, 3:].to(device).clone()")
    emitter.line("_, _, robot_initial_yaw = euler_xyz_from_quat(base_target_rot)")
    emitter.line("default_palm_rot = get_rotation_quat(math.pi, math.pi, math.pi, device)")
    emitter.line()
    emitter.line("_append_state(")
    emitter.line("    robot_traj,")
    emitter.line("    door_traj,")
    emitter.line("    key_idx_in_key_indices,")
    emitter.line("    q_robot,")
    emitter.line("    q_door,")
    emitter.line("    mark_keyframe=True,")
    emitter.line(")")

    # Carried across phases: whether a base/palm target exists yet to be passed to solve_ik.
    state = {
        "base_available": False,
        "palm_available": False,
        # The yaw the base is currently commanded at. A traverse interpolates FROM it, so it has
        # to be carried rather than re-derived: by then a release phase may have tilted the base.
        "base_yaw_expr": "robot_initial_yaw.item()",
    }
    for step_no, phase_id in enumerate(result.phase_order, start=1):
        aggs = result.by_phase(phase_id)
        if not aggs:
            continue
        plan = result.keyframe_plans.get(phase_id, [])
        block = plan[0].continuity_block if plan else None
        block_const = block_name if block else None
        title = PHASE_META[phase_id][0]
        emitter.banner(step_no, title)

        primitives = {agg.primitive for agg in aggs}
        if "fractional_interpolation" in primitives:
            _emit_traverse_phase(emitter, phase_id, aggs, plan, block_const, state)
        elif primitives & {"rotate_with_theta", "linear_gain", "hold_then_release"}:
            _emit_swept_phase(emitter, phase_id, aggs, plan, block_const, state)
        else:
            _emit_static_phase(emitter, phase_id, aggs, plan, block_const, state)

    emitter.line()
    emitter.line("return robot_traj, door_traj, key_idx_in_key_indices")

    out.extend(emitter.lines)
    body = "\n".join(out) + "\n"
    header = HEADER.format(
        n_sessions=len(result.session_paths),
        variant=variant.key(),
        sessions=sessions,
        unresolved=unresolved_block,
        imports=_import_block(body),
    )
    return header + body


# ------------------------------------------------------------------- verification

def verify_generated_planner(module_path: str, function_name: str, session: CaptureSession,
                             *, device: str = "cpu") -> dict:
    """Run the generated planner and report what the IK actually managed at every keyframe.

    ``solve_ik`` already computes ``success`` and ``debug_info['best_error_norm']`` and merely
    PRINTS a warning when it fails (api.py:110). Wrapping the module-level function is what lets
    that reach a caller without threading a debug flag through every generated call site and
    cluttering source that is meant to read like the hand-written planners.
    """
    import importlib.util

    import torch

    from DoorOpening.utils.state_machine import api

    records: list[dict] = []
    original = api.solve_ik

    def recording_solve_ik(*args, **kwargs):
        kwargs["return_debug"] = True
        q, success, debug_info = original(*args, **kwargs)
        records.append({
            "index": len(records),
            "success": bool(success),
            "best_error_norm": (
                None if debug_info.get("best_error_norm") is None
                else float(debug_info["best_error_norm"])
            ),
        })
        return q

    spec = importlib.util.spec_from_file_location("_generated_planner", module_path)
    module = importlib.util.module_from_spec(spec)
    api.solve_ik = recording_solve_ik
    module.solve_ik = recording_solve_ik
    try:
        spec.loader.exec_module(module)
        module.solve_ik = recording_solve_ik
        entry = getattr(module, function_name)
        robot_traj, door_traj, key_idx = entry(
            session.robot_urdf_path,
            session.door_urdf_path,
            torch.tensor([session.robot_initial_pose_world], device=device),
            torch.tensor([session.door_initial_pose_world], device=device),
            torch.zeros(17, device=device),
            torch.tensor([0.0, 0.0], device=device),
            device=device,
        )
    finally:
        api.solve_ik = original

    from DoorOpening.utils.state_machine.compute_waypoint import collocate_and_playback

    stacked = torch.stack([q.detach().cpu() for q in robot_traj])
    collapsed = int((torch.linalg.norm(stacked[1:] - stacked[:-1], dim=-1) < 1e-9).sum())
    collocate_and_playback(robot_traj, door_traj, key_idx, length=600)

    failures = [record for record in records if not record["success"]]
    return {
        "waypoints": len(robot_traj),
        "keyframes": len(key_idx),
        "ik_calls": len(records),
        "ik_failures": failures,
        "collapsed_adjacent_waypoints": collapsed,
        "worst_error_norm": max(
            (r["best_error_norm"] for r in records if r["best_error_norm"] is not None),
            default=None,
        ),
        "ok": not failures and collapsed == 0,
    }


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize an offline door planner from captured demonstrations.",
    )
    parser.add_argument("sessions", nargs="+", help="capture session JSON files (globs allowed)")
    parser.add_argument("--out", required=True, help="output .py path (must not be an existing planner)")
    parser.add_argument("--no-verify", action="store_true", help="skip the IK feasibility check")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.sessions:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    protected = {"offline_pull_door.py", "offline_push_door.py"}
    if os.path.basename(args.out) in protected:
        parser.error(
            f"refusing to write {args.out}: the hand-tuned planners are never overwritten. "
            "Generate to a new file (e.g. offline_pull_door_generated.py) and merge by hand."
        )
    if os.path.exists(args.out) and not args.force:
        parser.error(f"{args.out} exists; pass --force to overwrite.")

    try:
        sessions = load_sessions(paths)
        result = aggregate_sessions(sessions)
    except CaptureSchemaError as exc:
        print(f"[synthesize] {exc}", file=sys.stderr)
        return 2

    try:
        source = generate_source(result)
    except GeneratorError as exc:
        print(f"[synthesize] {exc}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(source)
    print(f"[synthesize] wrote {args.out} ({len(source.splitlines())} lines)")

    flagged = result.flagged
    if flagged:
        print(f"[synthesize] {len(flagged)} fit(s) FLAGGED for review:")
        for agg in flagged:
            print(
                f"  {agg.phase_id}.{agg.channel:10s} {agg.primitive:24s} "
                f"residual {agg.residual:.5f} over {agg.n_samples} demos"
            )

    if args.no_verify:
        print("[synthesize] verification skipped (--no-verify); do not accept the file untested.")
        return 0

    try:
        report = verify_generated_planner(
            args.out, result.variant_class.function_name(), sessions[0]
        )
    except Exception as exc:  # the sim env may be absent; say so rather than claiming success
        print(f"[synthesize] verification could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print(
        f"[synthesize] verify: {report['waypoints']} waypoints, {report['keyframes']} keyframes, "
        f"{report['ik_calls']} IK calls, worst best_error_norm={report['worst_error_norm']}"
    )
    if report["ik_failures"] or report["collapsed_adjacent_waypoints"]:
        unresolved = [
            f"IK call {r['index']} did not converge (best_error_norm={r['best_error_norm']})"
            for r in report["ik_failures"]
        ]
        if report["collapsed_adjacent_waypoints"]:
            unresolved.append(
                f"{report['collapsed_adjacent_waypoints']} adjacent waypoint pair(s) collapsed to "
                "the same config -- an unreachable target best-efforted onto its neighbour"
            )
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(generate_source(result, unresolved=unresolved))
        print("[synthesize] INFEASIBLE KEYFRAMES -- not accepting this file as clean:")
        for item in unresolved:
            print(f"  - {item}")
        print(f"[synthesize] the unresolved list is recorded in {args.out}'s header.")
        return 1

    print("[synthesize] all keyframes solved; file is ready for review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
