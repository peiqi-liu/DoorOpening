"""Build the paste-ready turns for co-authoring a planner with an AI in a chat.

THE LOOP THIS SERVES

    1. capture a few keyframes in the workbench, on one door
    2. paste a turn into the chat; the AI writes or revises the planner
    3. run check_planner.py on the result -- "did the traj actually compute"
    4. paste the check output plus your own feedback back as the next turn
    5. capture a bit more, on the next door, and go round again

That loop, not a single batch prompt, is why this module exists. Two consequences shape it:

SEND ONCE WHAT ONLY NEEDS SAYING ONCE. The conventions, the primitive library and the code
skeleton run to ~250 lines and are already in the chat after turn 1, so a follow-up turn drops
them, and with ``--since`` it also drops the measurements for phases that did not change. What a
follow-up turn does still carry is the current planner source in full -- the AI is revising it, so
it has to see it -- which means turn 2 is not necessarily shorter than turn 1. The point is not
brevity for its own sake: it is that the feedback, the trajectory check and the new keyframes sit
at the TOP, above the source, instead of being buried under a re-run of the briefing.

PARTIAL CAPTURES ARE NORMAL. You are demonstrating one door at a time, so most turns have phases
with no samples yet and swept phases with too few. Those are reported as an agenda ("capture more
of this") rather than as errors, and the aggregator runs in partial mode to match.

    # turn 1 -- full briefing plus whatever is captured so far
    python -m DoorOpening.utils.state_machine.planner_prompt logs/workbench/*/capture_*.json \
        --out turn.md

    # turn N -- delta only
    python -m DoorOpening.utils.state_machine.planner_prompt logs/workbench/*/capture_*.json \
        --planner logs/workbench/door_02/draft_planner.py \
        --check check_report.txt \
        --feedback "pull sweep drives the wrist into the panel after theta 0.9" \
        --out turn.md
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from DoorOpening.utils.state_machine.capture_schema import (
    CaptureSchemaError,
    PHASE_IDS,
    VariantClass,
    load_sessions,
)

CONVENTIONS = '''\
CONVENTIONS -- these are the ones that are silently wrong if you guess.

FRAME. Every palm_pose is a `panda_hand` pose, because that is what solve_ik drives (api.py builds
PinocchioIKSolver with ee_link_name="panda_hand"). panda_hand is the WRIST MOUNT, not the contact
point: glorbot.urdf hangs `palm_center`, the grasp centre between the fingers, at (0, 0, 0.1034)
off it. So a target places the wrist and the fingers close ~10 cm beyond it. Offsets are tuned
against that convention; they are not contact points.

QUATERNIONS. api.solve_ik hands world_to_base_frame's output (IsaacLab wxyz) to
PinocchioIKSolver.compute_ik, which does R.from_quat(...) -- scipy xyzw. The orientation a planner
actually commands is therefore its get_rotation_quat tuple read one slot over. Every existing
offset is tuned against that behaviour. Reproduce it; do not "fix" it.

JOINT LAYOUT. q_robot is FULL_JOINT_NAMES order: base(3) + panda(7) + gripper(1) + x5 camera(6)
= 17. solve_ik takes and returns q_robot[:10]. The gripper is ONE driven joint at
FULL_JOINT_NAMES.index(DRIVEN_FINGER_JOINT_NAME) -- use _set_gripper, never a q_robot[10:26] slice
(that is LEAP-era and would overwrite all six camera joints).

ANCHORS, not coordinates. Waypoints are written as an anchor position plus a tuned offset:
  get_hinge_pos(door_urdf, door_initial_pose, q_door)       # link_2, plate + lever together
  get_handle_bar_pos(...)                                    # the LEVER BAR only -- use this to
                                                             # touch the handle; get_hinge_pos
                                                             # lands on the escutcheon plate
  get_board_pos(...)                                         # panel centroid
  get_board_edge(...)                                        # panel free edge, inset
These absorb per-door geometry, which is why one planner file generalizes across an asset set.

`+=` vs `=` IS LOAD-BEARING. Adjacent lines of the same target mix the two:
    base_target_pos[:, 0] += pull_base_x_offset                  # relative to the hinge anchor
    base_target_pos[:, 1]  = theta.item() * pull_base_y_gain     # an ABSOLUTE world coordinate
Treating an absolute channel as relative puts the base off by the anchor's own coordinate (~0.4 m
on these assets). Decide per channel and say which you meant.

CONTINUITY. Inside any for-loop over theta, pass num_attempts=1 so a hard frame returns a
best-effort NEAR the previous pose instead of jumping to a distant IK branch. Pass
reference_joint_pos when the default posture puts a link somewhere the task cannot tolerate -- it
is the null-space ANCHOR (applied at gain 0.5), not a seed, so it selects the IK branch.

KEYFRAMES. collocate_and_playback splines between KEYFRAMES, so mark_keyframe changes the
trajectory. A sweep is normally ONE segment: every frame mark_keyframe=False, then a single
`key_idx_in_key_indices.append(len(robot_traj) - 1)` after the loop.

STYLE. One top-level function per variant, never one parametrized function -- the existing
docstrings say the separation is deliberate so tuning stays local. Named offsets declared before
use in `{phase}_{channel}_{quantity}` order (pregrasp_base_x_offset, grasp_palm_y_offset,
unlatch_rot_yaw, pull_hinge_hold_until_theta, release_door_open_angle). Step blocks separated by
`# ----` banners with a `Step N: description` header. Comments explain WHY a constant has its
value, not what it is.
'''

PRIMITIVES = '''\
THE PRIMITIVE LIBRARY -- five, fixed. If a phase fits none of them, STOP and say so rather than
inventing a sixth. Import from DoorOpening.utils.state_machine.primitives.

1. constant_offset          pose = anchor.clone() then per-axis += (or = for absolute channels),
                            with a fixed euler rotation. Covers pregrasp, grasp, unlatch, release,
                            retract, and the push contacts.

2. rotate_with_theta        rotate_xy(x0, y0, theta, "clockwise" | "counterclockwise") -- the
                            closed-door offset carried around the hinge as the panel sweeps. The
                            sense is NOT a function of handle_side: right-pull and left-push are
                            clockwise, left-pull and right-push are not.

3. hold_then_release        hold_then_release(theta, level, hold_until_theta, release_by_theta) --
                            the door's own lever angle during a pull. Held against its mechanical
                            stop while the handle is gripped, then ramped back to 0. Holding is
                            what gives the pull a rigid reaction point; releasing on the first
                            pull frame lets the lever re-rotate under the grasp instead of moving
                            the panel.

4. linear_gain              linear_gain(theta, c0, gain) = c0 + gain * theta. Used for the base's
                            lateral drift through a sweep AND for a rotation channel that ramps
                            (pull_rot_roll_base + pull_rot_roll_per_theta * theta) -- same
                            arithmetic, so it is one primitive, not two.

5. fractional_interpolation lerp(start, end, frac) with frac = step / steps for step in 1..steps.
                            The traverse loops. Pin the palm pose through it so the arm does not
                            jerk away from the panel it is holding.
'''

TEMPLATE = '''\
FILL THIS IN. Replace every <TODO> and delete phases the variant does not use. Keep the banners,
the naming, and the helper calls exactly as they are.

```python
import math

import torch
from isaaclab.utils.math import euler_xyz_from_quat

from DoorOpening.constants.robot_constants import FRANKA_DEFAULT_JOINT_POS, GRIPPER_OPEN_WIDTH
from DoorOpening.utils.state_machine.api import (
    get_board_edge, get_board_pos, get_handle_bar_pos, get_hinge_pos, solve_ik,
)
from DoorOpening.utils.state_machine.primitives import (
    _append_state, _init_planner_state, _make_pose, _set_gripper,
    get_rotation_quat, hold_then_release, lerp, linear_gain, rotate_xy,
)

# Null-space posture anchor, only if the default puts a link somewhere the task cannot tolerate.
# Say WHY in the comment -- which link, which clearance, measured how.
# {VARIANT_UPPER}_IK_ANCHOR_JOINT_POS = {{**FRANKA_DEFAULT_JOINT_POS, "panda_joint2": <TODO>}}


def {FUNCTION_NAME}(
    robot_urdf_path,
    door_urdf_path,
    robot_initial_pose,   # (1, 7) world
    door_initial_pose,    # (1, 7) world
    robot_initial_q,      # (ndof,)
    door_initial_q,       # (2,) [board, hinge]
    device="cpu",
):
    """
    Offline planner for a {HANDLE_SIDE}-side handle, {DIRECTION}-type door.

    This is intentionally separate from the other variants' functions so all
    {HANDLE_SIDE}-door tuning stays local and obvious.
    """
    q_robot, q_door, robot_traj, door_traj, key_idx_in_key_indices = _init_planner_state(
        robot_initial_q, door_initial_q
    )

    base_target_rot = robot_initial_pose[:, 3:].to(device).clone()
    _, _, robot_initial_yaw = euler_xyz_from_quat(base_target_rot)
    default_palm_rot = get_rotation_quat(<TODO roll>, <TODO pitch>, <TODO yaw>, device)

    _append_state(robot_traj, door_traj, key_idx_in_key_indices, q_robot, q_door,
                  mark_keyframe=True)

    # -------------------------
    # Step 1: Pregrasp
    # -------------------------
    # <TODO: why these values -- what was wrong before, what this fixes>
    pregrasp_base_x_offset = <TODO>
    pregrasp_base_y_offset = <TODO>
    pregrasp_palm_x_offset = <TODO>
    pregrasp_palm_y_offset = <TODO>
    pregrasp_palm_z_offset = <TODO>

    handle_pos = get_hinge_pos(door_urdf_path, door_initial_pose, q_door.unsqueeze(0)).to(device)

    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += pregrasp_base_x_offset
    base_target_pos[:, 1] += pregrasp_base_y_offset
    base_target_pose = _make_pose(base_target_pos, base_target_rot)

    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += pregrasp_palm_x_offset
    palm_target_pos[:, 1] += pregrasp_palm_y_offset
    palm_target_pos[:, 2] += pregrasp_palm_z_offset
    palm_target_pose = _make_pose(palm_target_pos, default_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path, q_robot[:10],
        palm_pose=palm_target_pose, base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _append_state(robot_traj, door_traj, key_idx_in_key_indices, q_robot, q_door,
                  mark_keyframe=True)

    # -------------------------
    # Step 2: Move to grasp
    # -------------------------
    # Base pose is REUSED from Step 1 -- do not re-derive it.
    grasp_palm_x_offset = <TODO>
    grasp_palm_y_offset = <TODO>
    grasp_palm_z_offset = <TODO>
    # ... same shape as Step 1, then _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

    # -------------------------
    # Step 3: Rotate hinge (unlatch)
    # -------------------------
    # Target the lever's HARD STOP, and stay above the highest randomized unlatch threshold.
    unlatch_hinge_angle = <TODO>
    unlatch_palm_y_delta = <TODO>      # delta off the GRASP palm pose, not off the anchor
    unlatch_palm_z_delta = <TODO>
    unlatch_rot_roll = <TODO>
    unlatch_rot_pitch = <TODO>
    unlatch_rot_yaw = <TODO>
    q_door = torch.tensor([0.0, unlatch_hinge_angle], device=device)

    # -------------------------
    # Step 4: Pull door open        (primitives 2, 3 and 4 all live here)
    # -------------------------
    pull_theta_start = <TODO>
    pull_theta_stop = <TODO>
    pull_theta_step = <TODO>

    pull_hinge_hold_until_theta = <TODO>   # hold the lever down until the bolt is clear
    pull_hinge_release_by_theta = pull_theta_stop

    pull_base_x_offset = <TODO>
    pull_base_y_offset = <TODO>            # NOTE: base_y is assigned, not added -- world absolute
    pull_base_y_gain = <TODO>

    pull_palm_x_offset_closed = <TODO>     # the offset AT theta = 0; rotate_xy carries it round
    pull_palm_y_offset_closed = <TODO>
    pull_palm_z_offset = <TODO>

    pull_rot_roll_base = <TODO>
    pull_rot_roll_per_theta = <TODO>
    pull_rot_pitch = <TODO>
    pull_rot_yaw = <TODO>

    for theta in torch.arange(pull_theta_start, pull_theta_stop + 1e-6, pull_theta_step,
                              device=device):
        q_door = torch.tensor(
            [theta.item(),
             hold_then_release(theta.item(), unlatch_hinge_angle,
                               pull_hinge_hold_until_theta, pull_hinge_release_by_theta)],
            device=device,
        )
        handle_pos = get_hinge_pos(door_urdf_path, door_initial_pose,
                                   q_door.unsqueeze(0)).to(device)

        base_target_pos = handle_pos.clone()
        base_target_pos[:, 0] += pull_base_x_offset
        base_target_pos[:, 1] = linear_gain(theta.item(), pull_base_y_offset, pull_base_y_gain)
        base_target_pose = _make_pose(base_target_pos, base_target_rot)

        palm_dx, palm_dy = rotate_xy(pull_palm_x_offset_closed, pull_palm_y_offset_closed,
                                     theta, "<TODO clockwise|counterclockwise>")
        palm_target_pos = handle_pos.clone()
        palm_target_pos[:, 0] += palm_dx
        palm_target_pos[:, 1] += palm_dy
        palm_target_pos[:, 2] += pull_palm_z_offset
        palm_target_pose = _make_pose(
            palm_target_pos,
            get_rotation_quat(
                linear_gain(theta.item(), pull_rot_roll_base, pull_rot_roll_per_theta),
                pull_rot_pitch, pull_rot_yaw, device,
            ),
        )

        q_robot[:10] = solve_ik(
            robot_urdf_path, q_robot[:10],
            palm_pose=palm_target_pose, base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # loop body: single seed for continuity
        )[0]
        _append_state(robot_traj, door_traj, key_idx_in_key_indices, q_robot, q_door,
                      mark_keyframe=False)

    key_idx_in_key_indices.append(len(robot_traj) - 1)   # the sweep is ONE spline segment

    # -------------------------
    # Step 5: Release / Step 6: Retract / Step 7: Push / Step 8: Traverse
    # -------------------------
    # Same shape. Release and retract are constant_offset off the PREVIOUS palm/base pose.
    # Traverse is primitive 5:
    #     for traverse_step in range(1, traverse_steps + 1):
    #         frac = traverse_step / traverse_steps
    #         base_target_pos[:, 0] = lerp(start_base_x, traverse_base_x_end, frac)
    #         ...   # palm_pose stays PINNED so the arm keeps holding the panel

    return robot_traj, door_traj, key_idx_in_key_indices
```
'''


def _measured_block(sessions, only_phases: set[str] | None = None) -> str:
    """The numbers actually measured, so the code is filled from data rather than from taste.

    ``only_phases`` trims a follow-up turn to the phases that actually gained keyframes. The rest
    were already sent in an earlier turn and are unchanged; repeating them every round pushes the
    part that DID change off the top of the message.
    """
    from DoorOpening.utils.state_machine.offset_aggregation import aggregate_sessions

    result = aggregate_sessions(sessions)
    scope = (
        "MEASURED ON THIS DOOR -- fitted from "
        f"{len(sessions)} capture session(s), variant {result.variant_class.key()}."
        if only_phases is None else
        f"RE-MEASURED (only the phases that gained keyframes; the rest are unchanged from earlier "
        f"turns) -- {len(sessions)} capture session(s), variant {result.variant_class.key()}."
    )
    lines = [scope, "Use these values; the residual/spread tells you how much to trust each one.", ""]
    for phase_id in result.phase_order:
        if only_phases is not None and phase_id not in only_phases:
            continue
        aggs = result.by_phase(phase_id)
        if not aggs:
            continue
        lines.append(f"  {phase_id}:")
        for agg in aggs:
            constants = ", ".join(
                f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in agg.constants.items()
            )
            flag = "   <-- REVIEW, residual above threshold" if agg.flagged else ""
            lines.append(
                f"    {agg.channel:10s} {agg.primitive:24s} {constants}"
                f"  [{agg.n_samples} demos, {agg.method}]{flag}"
            )
        notes = {note for agg in aggs for note in agg.notes}
        for note in sorted(notes):
            lines.append(f"    human note: {note}")
        lines.append("")

    if result.gaps:
        lines.append("  NOT FITTED YET -- the agenda for the next capture round:")
        for gap in result.gaps:
            lines.append(f"    - {gap}")
        lines.append("")

    for block in result.continuity_blocks if only_phases is None else []:
        lines.append(
            f"  continuity block {block.name!r} -> {block.constant_name}, median over "
            f"{block.n_samples} demonstrated postures:"
        )
        for joint, value in block.joint_pos.items():
            lines.append(f"    {joint}: {value:.5g}")
        lines.append("")
    return "\n".join(lines)


def build_skeleton_prompt(
    variant: VariantClass,
    sessions=None,
) -> str:
    parts = [
        "Write an offline door-opening planner for this repo. Match the existing hand-written "
        "planners exactly -- offline_pull_door.py and offline_push_door.py are the reference and "
        "must NOT be modified; the answer goes in a new file.",
        "",
        CONVENTIONS,
        "",
        PRIMITIVES,
        "",
    ]
    if sessions:
        parts += [_measured_block(sessions), ""]
    else:
        parts += [
            "No capture session was supplied, so no measured values are included. Leave every "
            "<TODO> as a named constant with a comment saying what it should be measured against "
            "-- do not invent numbers.",
            "",
        ]
    parts.append(
        TEMPLATE.format(
            FUNCTION_NAME=variant.function_name(),
            VARIANT_UPPER=f"{variant.handle_side}_{variant.opening_direction}".upper(),
            HANDLE_SIDE=variant.handle_side,
            DIRECTION=variant.opening_direction,
        )
    )
    parts += [
        "",
        "BEFORE YOU ANSWER: say which primitive you chose for each swept channel and why. If a "
        "phase fits none of the five, stop and ask rather than adding a sixth.",
    ]
    return "\n".join(parts)


def _keyframe_digest(sessions, since: int | None) -> str:
    """What has been captured, newest first, so a turn shows the delta rather than the whole pile."""
    rows, total = [], 0
    for session in sessions:
        for keyframe in session.keyframes:
            total += 1
            if since is not None and keyframe.id < since:
                continue
            label = f"  {keyframe.phase_id}"
            if keyframe.theta is not None:
                label += f" theta={keyframe.theta:.2f}"
            if not keyframe.mark_keyframe:
                label += " (non-key)"
            if keyframe.notes:
                label += f"\n      note: {keyframe.notes}"
            rows.append(label)
    header = (
        f"CAPTURED SINCE LAST TURN ({len(rows)} of {total} keyframes)"
        if since is not None else f"CAPTURED SO FAR ({total} keyframes)"
    )
    return header + "\n" + ("\n".join(rows) if rows else "  (nothing new)")


def build_turn_prompt(
    variant: VariantClass,
    sessions,
    *,
    planner_source: str = "",
    planner_path: str = "",
    check_report: str = "",
    feedback: str = "",
    since_keyframe: int | None = None,
) -> str:
    """One conversational turn.

    With no ``planner_source`` this is turn 1 and carries the full briefing. With one, it carries
    only what changed -- the chat already holds the conventions.
    """
    first_turn = not planner_source.strip()
    parts: list[str] = []

    if first_turn:
        parts += [
            "Write an offline door-opening planner for this repo. We will iterate: I capture "
            "keyframes on a door at a time, you write the code, I run it and report back what the "
            "trajectory did, and we go round again. Expect to revise, not to get it right once.",
            "",
            "offline_pull_door.py and offline_push_door.py are the reference for style and must "
            "NOT be modified -- your answer goes in a new file.",
            "",
            CONVENTIONS,
            "",
            PRIMITIVES,
            "",
        ]
    else:
        parts += [
            "Next turn of the door-planner loop. Conventions and the primitive library are "
            "unchanged from earlier in this conversation -- do not restate them.",
            "",
        ]

    if feedback.strip():
        parts += ["WHAT I SAW / WHAT I WANT CHANGED", feedback.strip(), ""]

    if check_report.strip():
        parts += [
            "I ran the current planner and computed the trajectory. Result:",
            "",
            check_report.strip(),
            "",
        ]

    if sessions:
        changed = None
        if not first_turn and since_keyframe is not None:
            changed = {
                keyframe.phase_id
                for session in sessions
                for keyframe in session.keyframes
                if keyframe.id >= since_keyframe
            }
        parts += [
            _keyframe_digest(sessions, since_keyframe),
            "",
            _measured_block(sessions, changed),
            "",
        ]

    if planner_source.strip():
        parts += [
            f"CURRENT PLANNER ({planner_path or 'draft'}) -- revise this, do not start over:",
            "",
            "```python",
            planner_source.rstrip(),
            "```",
            "",
            "Give me the full revised file back, and say in one or two lines what you changed and "
            "why. If a change is a guess rather than something the numbers support, say so.",
        ]
    else:
        parts += [
            TEMPLATE.format(
                FUNCTION_NAME=variant.function_name(),
                VARIANT_UPPER=f"{variant.handle_side}_{variant.opening_direction}".upper(),
                HANDLE_SIDE=variant.handle_side,
                DIRECTION=variant.opening_direction,
            ),
            "",
            "Fill in what the measured values support. Leave anything not yet demonstrated as a "
            "named constant with a <TODO> and a comment saying what it should be measured "
            "against -- do not invent numbers. Say which primitive you chose for each swept "
            "channel and why.",
        ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print one turn of the planner co-authoring loop, ready to paste into a chat.",
    )
    parser.add_argument("sessions", nargs="*", help="capture session JSON files (globs ok)")
    parser.add_argument("--handle-side", choices=("left", "right"), default="right")
    parser.add_argument("--direction", choices=("pull", "push"), default="pull")
    parser.add_argument(
        "--planner", help="current draft planner .py; supplying it makes this a follow-up turn "
        "(delta only) instead of the full first-turn briefing",
    )
    parser.add_argument("--check", help="output of check_planner.py to include verbatim")
    parser.add_argument("--feedback", default="", help="what you saw and want changed")
    parser.add_argument(
        "--since", type=int, default=None,
        help="only list keyframes with id >= this, so a turn shows just what you added",
    )
    parser.add_argument("--out", help="write here instead of stdout")
    args = parser.parse_args(argv)

    sessions = None
    variant = VariantClass(args.handle_side, args.direction)
    if args.sessions:
        paths: list[str] = []
        for pattern in args.sessions:
            paths.extend(sorted(glob.glob(pattern)) or [pattern])
        try:
            sessions = load_sessions(paths)
        except CaptureSchemaError as exc:
            print(f"[prompt] {exc}", file=sys.stderr)
            return 2
        variant = sessions[0].variant_class

    planner_source = ""
    if args.planner:
        if not os.path.exists(args.planner):
            print(f"[prompt] no such planner: {args.planner}", file=sys.stderr)
            return 2
        with open(args.planner, encoding="utf-8") as handle:
            planner_source = handle.read()

    check_report = ""
    if args.check:
        with open(args.check, encoding="utf-8") as handle:
            check_report = handle.read()

    text = build_turn_prompt(
        variant,
        sessions,
        planner_source=planner_source,
        planner_path=args.planner or "",
        check_report=check_report,
        feedback=args.feedback,
        since_keyframe=args.since,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        kind = "follow-up turn" if planner_source else "first turn"
        print(
            f"[prompt] wrote {args.out} -- {kind}, {len(text.splitlines())} lines. "
            "Paste it into the chat."
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
