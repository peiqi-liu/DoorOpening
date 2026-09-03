"""Run a planner against a real door and report what the IK actually managed, ready to paste back.

This is the "I checked the code by computing the traj" step of the co-authoring loop. It is
deliberately separate from synthesize_planner: the planner under test may be generated, may be
hand-edited afterwards, or may have been written from scratch in a chat, and all three need the
same check.

    python -m DoorOpening.utils.state_machine.check_planner \
        logs/workbench/scratch_door__rnd_02/draft_planner.py \
        --door source/DoorOpening/assets/door/v5_1/scratch_door__rnd_02/door.urdf

``solve_ik`` already computes ``success`` and ``debug_info['best_error_norm']`` and merely PRINTS a
warning when a solve fails (api.py:110). Wrapping the module-level function is what lets that
reach a report without threading a debug flag through every call site in source that is meant to
read like the hand-written planners.

WHAT THE NUMBERS MEAN
  success=False          the solver did not converge. A LARGE best_error_norm means the target is
                         out of reach (base parked too far, or a bad target); a SMALL one means it
                         stalled near the goal and the pose is probably usable.
  collapsed pairs        two adjacent waypoints that came out at the same configuration. That is
                         the signature of an unreachable target the IK best-efforted onto its
                         neighbour, and collocate_and_playback silently drops it into the spline,
                         so it is counted here rather than left to vanish.

No collision checking -- out of scope by design, that is RL's job.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

DEFAULT_ROBOT_URDF = "source/DoorOpening/assets/glorbot/glorbot.urdf"


def run_and_diagnose(
    module_path: str,
    function_name: str | None,
    door_urdf_path: str,
    robot_urdf_path: str = DEFAULT_ROBOT_URDF,
    *,
    device: str = "cpu",
    playback_length: int = 600,
) -> dict:
    """Import the planner, run it with a recording solve_ik, and collect per-call diagnostics."""
    import torch

    from DoorOpening.constants.env_constants import (
        DOOR_INITIAL_POS,
        DOOR_INITIAL_ROT,
        ROBOT_INITIAL_POS,
        ROBOT_INITIAL_ROT,
    )
    from DoorOpening.utils.state_machine import api
    from DoorOpening.utils.state_machine.compute_waypoint import (
        _apply_initial_state_metadata,
        collocate_and_playback,
        get_robot_constants,
        resolve_planner_options,
    )

    robot_initial_pose = torch.tensor([[*ROBOT_INITIAL_POS, *ROBOT_INITIAL_ROT]], device=device)
    door_initial_pose = torch.tensor([[*DOOR_INITIAL_POS, *DOOR_INITIAL_ROT]], device=device)
    _, robot_initial_q = get_robot_constants()
    door_initial_q = torch.tensor([0.0, 0.0], device=device)
    (
        robot_initial_pose,
        door_initial_pose,
        robot_initial_q,
        door_initial_q,
        _,
    ) = _apply_initial_state_metadata(
        door_urdf_path, robot_initial_pose, door_initial_pose, robot_initial_q, door_initial_q
    )
    handle_side, opening_direction = resolve_planner_options(door_urdf_path, "auto", "auto")

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

    spec = importlib.util.spec_from_file_location("_planner_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    api.solve_ik = recording_solve_ik
    try:
        spec.loader.exec_module(module)
        # The planner imported its own reference to solve_ik at import time, so rebind that too --
        # patching only api.solve_ik would record nothing.
        module.solve_ik = recording_solve_ik
        if function_name is None:
            function_name = _guess_entry(module, handle_side, opening_direction)
        entry = getattr(module, function_name)
        robot_traj, door_traj, key_idx = entry(
            robot_urdf_path,
            door_urdf_path,
            robot_initial_pose,
            door_initial_pose,
            robot_initial_q,
            door_initial_q,
            device=device,
        )
    finally:
        api.solve_ik = original

    stacked = torch.stack([q.detach().cpu() for q in robot_traj])
    deltas = torch.linalg.norm(stacked[1:] - stacked[:-1], dim=-1)
    collapsed = [int(i) for i, d in enumerate(deltas) if float(d) < 1e-9]

    playback_error = None
    try:
        collocate_and_playback(robot_traj, door_traj, key_idx, length=playback_length)
    except Exception as exc:
        playback_error = f"{type(exc).__name__}: {exc}"

    failures = [r for r in records if not r["success"]]
    return {
        "module_path": module_path,
        "function_name": function_name,
        "door_urdf_path": door_urdf_path,
        "handle_side": handle_side,
        "opening_direction": opening_direction,
        "waypoints": len(robot_traj),
        "keyframes": len(key_idx),
        "key_indices": [int(i) for i in key_idx],
        "ik_calls": len(records),
        "ik_failures": failures,
        "collapsed_adjacent_waypoints": collapsed,
        "worst_error_norm": max(
            (r["best_error_norm"] for r in records if r["best_error_norm"] is not None),
            default=None,
        ),
        "playback_error": playback_error,
        "ok": not failures and not collapsed and playback_error is None,
    }


def _guess_entry(module, handle_side: str, opening_direction: str) -> str:
    """Find the planner entry point without being told, so the CLI usually needs no --function."""
    candidates = [
        f"state_machine_offline_{handle_side}_{opening_direction}_door",
        f"state_machine_offline_{opening_direction}_{handle_side}_door",
        "state_machine_offline_door",
    ]
    for name in candidates:
        if hasattr(module, name):
            return name
    found = [n for n in dir(module) if n.startswith("state_machine_offline")]
    if len(found) == 1:
        return found[0]
    raise AttributeError(
        f"cannot pick an entry point; tried {candidates} and found {found or 'none'}. "
        "Pass --function explicitly."
    )


def format_report(report: dict) -> str:
    """A compact block meant to be pasted straight back into the chat."""
    lines = [
        "TRAJECTORY CHECK",
        f"  planner   {report['module_path']}::{report['function_name']}",
        f"  door      {report['door_urdf_path']}  ({report['handle_side']}/"
        f"{report['opening_direction']})",
        f"  result    {report['waypoints']} waypoints, {report['keyframes']} keyframes, "
        f"{report['ik_calls']} IK calls",
        f"  worst best_error_norm  {report['worst_error_norm']}",
    ]
    if report["ik_failures"]:
        lines.append(f"  IK FAILURES ({len(report['ik_failures'])}):")
        for record in report["ik_failures"]:
            lines.append(
                f"    call {record['index']:3d}  best_error_norm={record['best_error_norm']}"
            )
    else:
        lines.append("  IK failures            none")
    if report["collapsed_adjacent_waypoints"]:
        lines.append(
            f"  COLLAPSED PAIRS        at waypoints "
            f"{report['collapsed_adjacent_waypoints']} -- unreachable targets best-efforted onto "
            "their neighbour"
        )
    else:
        lines.append("  collapsed pairs        none")
    if report["playback_error"]:
        lines.append(f"  PLAYBACK FAILED        {report['playback_error']}")
    lines.append(f"  verdict   {'CLEAN' if report['ok'] else 'NEEDS WORK'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a planner and report its IK feasibility.")
    parser.add_argument("planner", help="path to the planner .py file")
    parser.add_argument("--door", required=True, help="door URDF to run against")
    parser.add_argument("--robot", default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--function", default=None, help="entry point (guessed if omitted)")
    parser.add_argument("--out", help="also write the report here, for pasting")
    args = parser.parse_args(argv)

    if not os.path.exists(args.planner):
        print(f"[check] no such planner: {args.planner}", file=sys.stderr)
        return 2
    try:
        report = run_and_diagnose(args.planner, args.function, args.door, args.robot)
    except Exception as exc:
        print(f"[check] planner raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    text = format_report(report)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"\n[check] wrote {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
