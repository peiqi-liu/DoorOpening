"""File-based request/response bridge between the planner workbench and an LLM.

The workbench never calls an API. It writes a request JSON into ``<session>/bridge/`` and polls
for a matching response JSON. Anything that can read and write those two files can play the LLM
role -- a human, a Claude Code session, or a real API client wired in later. The protocol is the
same in every case, so swapping the backend never touches the workbench.

Layout::

    <session_dir>/bridge/
        request_0001.json     written by the workbench when the human hits Send
        response_0001.json    written by whoever is playing the LLM
        transcript.jsonl      append-only log of every exchange, in order

A request carries the FULL scene state, not just the prompt, because the whole point is that the
responder writes planner code against it: which door, where every anchor sits at the current door
angle, where the human has dragged the gizmos, what offset that implies against each anchor, and
what the IK made of it. See ``build_request`` for the field list.

A response may carry ``planner_source``; when it does the workbench writes it to the draft planner
file and hot-reloads. A response with only ``reply`` is a conversational turn and changes nothing.
"""

from __future__ import annotations

import json
import os
import time

REQUEST_PREFIX = "request_"
RESPONSE_PREFIX = "response_"
TRANSCRIPT_NAME = "transcript.jsonl"


def bridge_dir(session_dir: str) -> str:
    path = os.path.join(session_dir, "bridge")
    os.makedirs(path, exist_ok=True)
    return path


def _path(session_dir: str, prefix: str, turn: int) -> str:
    return os.path.join(bridge_dir(session_dir), f"{prefix}{turn:04d}.json")


def request_path(session_dir: str, turn: int) -> str:
    return _path(session_dir, REQUEST_PREFIX, turn)


def response_path(session_dir: str, turn: int) -> str:
    return _path(session_dir, RESPONSE_PREFIX, turn)


def next_turn(session_dir: str) -> int:
    """One past the highest request already on disk, so a resumed session keeps counting up."""
    existing = [
        int(name[len(REQUEST_PREFIX):-len(".json")])
        for name in os.listdir(bridge_dir(session_dir))
        if name.startswith(REQUEST_PREFIX) and name.endswith(".json")
    ]
    return max(existing, default=0) + 1


def build_request(
    prompt: str,
    *,
    turn: int,
    planner_path: str,
    planner_source: str,
    scene_state: dict,
    last_run: dict | None = None,
) -> dict:
    """Assemble the payload the responder needs to write planner code with no other context.

    ``scene_state`` is whatever ``PlannerWorkbench.scene_state()`` produced: door/robot URDF paths,
    both joint vectors with their joint names, every anchor position at the current door angle, the
    dragged gizmo poses, the gizmo-minus-anchor offsets (these are the numbers that become planner
    constants), and the key-body FK poses.

    ``last_run`` is the result of the most recent planner execution -- per-keyframe IK success and
    error norms, or the traceback if it raised.
    """
    return {
        "turn": turn,
        "timestamp": time.time(),
        "prompt": prompt,
        "planner_path": planner_path,
        "planner_source": planner_source,
        "scene_state": scene_state,
        "last_run": last_run,
        "response_contract": {
            "reply": "str -- shown in the workbench transcript",
            "planner_source": "str | null -- full replacement source for planner_path; null to change nothing",
        },
    }


def write_request(session_dir: str, request: dict) -> str:
    path = request_path(session_dir, request["turn"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2)
    return path


def read_response(session_dir: str, turn: int) -> dict | None:
    """Return the response for ``turn``, or None if the responder hasn't answered yet.

    A partially-written file reads as absent rather than as corrupt: the responder may be mid-write
    when the workbench polls, and a JSONDecodeError there would kill the poll loop for a file that
    is about to become valid.
    """
    path = response_path(session_dir, turn)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def write_response(session_dir: str, turn: int, reply: str, planner_source: str | None = None) -> str:
    """Answer a request. This is the entry point for whoever is playing the LLM."""
    path = response_path(session_dir, turn)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"turn": turn, "timestamp": time.time(), "reply": reply, "planner_source": planner_source},
            handle,
            indent=2,
        )
    return path


def append_transcript(session_dir: str, role: str, text: str) -> None:
    with open(os.path.join(bridge_dir(session_dir), TRANSCRIPT_NAME), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": time.time(), "role": role, "text": text}) + "\n")


def pending_requests(session_dir: str) -> list[int]:
    """Turns that have a request but no response -- what the responder still owes an answer to."""
    base = bridge_dir(session_dir)
    requests, responses = set(), set()
    for name in os.listdir(base):
        if not name.endswith(".json"):
            continue
        if name.startswith(REQUEST_PREFIX):
            requests.add(int(name[len(REQUEST_PREFIX):-len(".json")]))
        elif name.startswith(RESPONSE_PREFIX):
            responses.add(int(name[len(RESPONSE_PREFIX):-len(".json")]))
    return sorted(requests - responses)


# --------------------------------------------------------------- agent -> human

# The direction above is human-initiated: the human hits Send and the responder answers. Authoring
# a planner needs the OPPOSITE direction too -- the agent is the one who knows that it cannot pick
# an unlatch wrist roll from first principles, and it has to be able to stop and ask. So an ASK is
# a question the agent puts INTO the workbench, and an ANSWER is what the human gives back by
# posing the scene, picking an option, or watching a playback. The human never writes code, and the
# agent never invents a number: this file is where those two rules meet.

ASK_PREFIX = "ask_"
ANSWER_PREFIX = "answer_"

# What the human is being asked to DO. The workbench renders different controls for each.
ASK_KINDS = (
    "capture",   # pose the scene and press Add keyframe -- answer carries the capture file
    "choose",    # pick one of `options`
    "review",    # watch the draft play back, then approve or describe what is wrong
    "confirm",   # yes / no
)


def ask_path(session_dir: str, turn: int) -> str:
    return _path(session_dir, ASK_PREFIX, turn)


def answer_path(session_dir: str, turn: int) -> str:
    return _path(session_dir, ANSWER_PREFIX, turn)


def next_ask_turn(session_dir: str) -> int:
    existing = [
        int(name[len(ASK_PREFIX):-len(".json")])
        for name in os.listdir(bridge_dir(session_dir))
        if name.startswith(ASK_PREFIX) and name.endswith(".json")
    ]
    return max(existing, default=0) + 1


def write_ask(
    session_dir: str,
    question: str,
    *,
    kind: str = "capture",
    options: list[str] | None = None,
    setup: dict | None = None,
    play_first: bool = False,
    turn: int | None = None,
) -> tuple[int, str]:
    """Post a question to the human in the workbench. Returns (turn, path).

    ``setup`` is the workbench state the question is about -- ``phase_id``, ``palm_anchor``,
    ``base_anchor``, ``gripper``, ``theta``. The workbench APPLIES it before showing the question,
    so the human is never asked to reproduce a configuration from prose. That is the whole point:
    the agent decides what is being measured and from what, the human only supplies the pose.

    ``play_first`` runs and plays back the current draft planner before the question appears, for
    the "watch this and tell me what is wrong" turn.
    """
    if kind not in ASK_KINDS:
        raise ValueError(f"unknown ask kind {kind!r}; expected one of {', '.join(ASK_KINDS)}")
    turn = next_ask_turn(session_dir) if turn is None else turn
    payload = {
        "turn": turn,
        "timestamp": time.time(),
        "question": question,
        "kind": kind,
        "options": options or [],
        "setup": setup or {},
        "play_first": bool(play_first),
    }
    path = ask_path(session_dir, turn)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    append_transcript(session_dir, "agent_ask", question)
    return turn, path


def read_ask(session_dir: str, turn: int) -> dict | None:
    path = ask_path(session_dir, turn)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def write_answer(
    session_dir: str,
    turn: int,
    *,
    choice: str = "",
    note: str = "",
    capture_path: str = "",
    keyframe_ids: list[int] | None = None,
) -> str:
    """The workbench's side: what the human did about ask ``turn``."""
    path = answer_path(session_dir, turn)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "turn": turn,
                "timestamp": time.time(),
                "choice": choice,
                "note": note,
                "capture_path": capture_path,
                "keyframe_ids": keyframe_ids or [],
            },
            handle,
            indent=2,
        )
    append_transcript(session_dir, "human_answer", choice or note or capture_path)
    return path


def read_answer(session_dir: str, turn: int) -> dict | None:
    """The agent's side: poll until the human has answered. None means not yet."""
    path = answer_path(session_dir, turn)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def pending_asks(session_dir: str) -> list[int]:
    """Asks the human has not answered yet -- what the workbench should be showing."""
    base = bridge_dir(session_dir)
    asks, answers = set(), set()
    for name in os.listdir(base):
        if not name.endswith(".json"):
            continue
        if name.startswith(ASK_PREFIX):
            asks.add(int(name[len(ASK_PREFIX):-len(".json")]))
        elif name.startswith(ANSWER_PREFIX):
            answers.add(int(name[len(ANSWER_PREFIX):-len(".json")]))
    return sorted(asks - answers)
