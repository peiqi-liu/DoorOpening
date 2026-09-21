#!/usr/bin/env python3
"""Ask the human a question inside the running planner workbench, and wait for the answer.

This is the agent's half of the interactive authoring loop. The workbench polls the same bridge
directory, applies the setup the question is about, renders the question, and writes the answer
back. The human never writes code and never has to reproduce a configuration from prose.

    # ask for a measurement, having already decided WHAT is measured and FROM WHAT
    python scripts/agent_ask.py ask --session-dir logs/workbench/<variant> \
        --kind capture --phase grasp --palm-anchor handle_bar --gripper 0.056 \
        --question "Pose the hand so the fingers close on the lever bar. 3 captures please."

    # play the current draft and ask what is wrong with it
    python scripts/agent_ask.py ask --session-dir logs/workbench/<variant> \
        --kind review --play-first \
        --question "Does the unlatch wrist roll look right, or is the hand fighting the lever?"

Both forms block until answered unless --no-wait is passed. The answer prints as JSON.
"""

import argparse
import json
import sys
import time

from DoorOpening.utils.state_machine import llm_bridge


def cmd_ask(args) -> int:
    setup = {}
    for key, value in (
        ("phase_id", args.phase),
        ("palm_anchor", args.palm_anchor),
        ("base_anchor", args.base_anchor),
        ("gripper", args.gripper),
        ("theta", args.theta),
    ):
        if value is not None:
            setup[key] = value

    turn, path = llm_bridge.write_ask(
        args.session_dir,
        args.question,
        kind=args.kind,
        options=args.option or None,
        setup=setup,
        play_first=args.play_first,
    )
    print(f"[ask] turn {turn} -> {path}", file=sys.stderr)
    if args.no_wait:
        print(json.dumps({"turn": turn, "path": path}))
        return 0
    return _wait(args.session_dir, turn, args.timeout)


def cmd_wait(args) -> int:
    return _wait(args.session_dir, args.turn, args.timeout)


def _wait(session_dir: str, turn: int, timeout: float) -> int:
    """Block until the human answers. Long timeouts are correct here: a human posing a waypoint
    carefully takes minutes, and a short poll that gives up would just be asked again."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        answer = llm_bridge.read_answer(session_dir, turn)
        if answer is not None:
            print(json.dumps(answer, indent=2))
            return 0
        time.sleep(2.0)
    print(
        f"[ask] turn {turn} still unanswered after {timeout:.0f}s. The workbench may not be "
        f"running, or the human is still posing. Re-run: agent_ask.py wait --turn {turn}",
        file=sys.stderr,
    )
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="post a question and wait for the answer")
    ask.add_argument("--session-dir", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--kind", default="capture", choices=list(llm_bridge.ASK_KINDS))
    ask.add_argument("--option", action="append", help="repeatable, for --kind choose")
    ask.add_argument("--phase", default=None, help="phase_id to switch the workbench to")
    ask.add_argument("--palm-anchor", default=None)
    ask.add_argument("--base-anchor", default=None)
    ask.add_argument("--gripper", type=float, default=None)
    ask.add_argument("--theta", type=float, default=None)
    ask.add_argument("--play-first", action="store_true",
                     help="run and play the draft planner before showing the question")
    ask.add_argument("--no-wait", action="store_true")
    ask.add_argument("--timeout", type=float, default=10800.0)
    ask.set_defaults(func=cmd_ask)

    wait = sub.add_parser("wait", help="wait for an answer to an already-posted turn")
    wait.add_argument("--session-dir", required=True)
    wait.add_argument("--turn", type=int, required=True)
    wait.add_argument("--timeout", type=float, default=10800.0)
    wait.set_defaults(func=cmd_wait)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
