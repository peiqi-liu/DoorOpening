#!/usr/bin/env python3
"""Open the planner workbench on one door.

A Viser session where you and an LLM jointly write a planner module shaped like
offline_pull_door.py: pose the door and robot, drag the panda_hand / base gizmos, read the
anchor-relative offsets those imply, and hand them to the LLM to turn into code. The draft runs
against the live door and plays back in the same scene.

    python scripts/capture_demo.py --door-urdf-path source/DoorOpening/assets/door/PartNetv6/<v>/mobility.urdf

WHICH ANCHOR AN OFFSET IS ABOUT is the one thing a capture cannot be corrected for afterwards --
"palm_y = -0.08" means nothing until you know it was measured from the handle bar rather than the
hinge plate, and the aggregator cannot infer it. Each phase ships the anchors the hand-written
planners use, and you change them in the workbench: the Capture folder has a "palm anchor (end
effector)" and a "base anchor" dropdown, set independently. A dropdown holds its value until you
change it, so picking one applies to every keyframe you capture afterwards.

Use --show-anchors to print the per-phase defaults and exit without opening a window.

The LLM side is a file bridge, not an API call -- see DoorOpening/utils/state_machine/llm_bridge.py.
Hitting "Send to LLM" writes logs/workbench/<variant>/bridge/request_NNNN.json; whoever is playing
the LLM (a Claude Code session, or a real client later) answers with write_response(). Nothing is
sent anywhere on its own.
"""

import argparse
import sys

from DoorOpening.utils.state_machine import capture_schema
from DoorOpening.utils.state_machine.planner_workbench import DEFAULT_ROBOT_URDF, run_workbench


def parse_phase_list(value, flag):
    """'all' or a comma-separated phase list, validated against the phase vocabulary."""
    if not value:
        return set()
    if value.strip() == "all":
        return set(capture_schema.PHASE_IDS)
    phases = {p.strip() for p in value.split(",") if p.strip()}
    unknown = phases - set(capture_schema.PHASE_IDS)
    if unknown:
        raise capture_schema.CaptureSchemaError(
            f"{flag}: unknown phase(s) {', '.join(sorted(unknown))}; "
            f"expected 'all' or a comma-separated subset of {', '.join(capture_schema.PHASE_IDS)}"
        )
    return phases


def show_anchors(joint_phases=None):
    """Print every phase's default channel plan -- what a capture records before you touch a dropdown."""
    joint_phases = joint_phases or {"arm": set(), "door": set()}
    for phase_id in capture_schema.PHASE_IDS:
        base = capture_schema.default_channel_spec(phase_id)
        spec = capture_schema.default_channel_spec(phase_id)
        capture_schema.with_joint_channels(
            spec,
            arm=phase_id in joint_phases["arm"],
            door=phase_id in joint_phases["door"],
        )
        print(f"\n{phase_id}")
        for name, entry in spec.items():
            moved = base.get(name, {}).get("anchor") != entry["anchor"]
            note = f"   <- was {base[name]['anchor']}" if moved and name in base else ""
            print(
                f"  {'*' if moved else ' '} {name:<11} {entry['primitive']:<26} "
                f"{entry['anchor']:<12} {'ABS' if entry['mode'] == 'world_absolute' else 'rel'}{note}"
            )


def main():
    parser = argparse.ArgumentParser(description="Interactive door-planner workbench.")
    parser.add_argument(
        "--robot-urdf-path",
        default=DEFAULT_ROBOT_URDF,
        help="Path to the robot URDF used for IK and playback.",
    )
    parser.add_argument(
        "--door-urdf-path",
        required=True,
        help="Door mobility.urdf to author against.",
    )
    parser.add_argument(
        "--handle-side",
        default="auto",
        choices=["auto", "right", "left"],
        help="Handle side, or read it from variant_meta.json when available.",
    )
    parser.add_argument(
        "--opening-direction",
        default="auto",
        choices=["auto", "pull", "push"],
        help="Opening direction, or read it from variant_meta.json when available.",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Where bridge requests, the draft planner, and exports live. Default logs/workbench/<variant>.",
    )
    parser.add_argument(
        "--planner-path",
        default=None,
        help="Draft planner module. Default <session-dir>/draft_planner.py, seeded if absent.",
    )
    parser.add_argument(
        "--show-anchors",
        action="store_true",
        help="Print the per-phase default anchor table and exit.",
    )
    parser.add_argument(
        "--record-arm-joints",
        default=None,
        metavar="PHASE[,PHASE...]|all",
        help="Also record the seven panda joint angles in these phases, pinning the waypoint to "
             "the demonstrated posture (anchor 'robot_joints').",
    )
    parser.add_argument(
        "--record-door-joints",
        default=None,
        metavar="PHASE[,PHASE...]|all",
        help="Also record the door's panel and lever angles in these phases (anchor "
             "'door_joints'). Phases that already command a door joint have it by default.",
    )
    parser.add_argument("--port", type=int, default=None, help="Optional Viser server port.")
    args = parser.parse_args()

    try:
        joint_phases = {
            "arm": parse_phase_list(args.record_arm_joints, "--record-arm-joints"),
            "door": parse_phase_list(args.record_door_joints, "--record-door-joints"),
        }
    except capture_schema.CaptureSchemaError as exc:
        # Fail before the sim spins up: a mistyped phase found 40 s into a load is a mistyped
        # phase found after you have stopped reading the terminal.
        parser.error(str(exc))

    if args.show_anchors:
        show_anchors(joint_phases)
        return 0

    run_workbench(
        args.robot_urdf_path,
        args.door_urdf_path,
        handle_side=args.handle_side,
        opening_direction=args.opening_direction,
        session_dir=args.session_dir,
        planner_path=args.planner_path,
        joint_phases=joint_phases,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
