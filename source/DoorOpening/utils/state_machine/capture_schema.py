"""Versioned on-disk schema for interactive waypoint capture sessions.

A capture session is what a human produces in the planner workbench: a door, a variant class, and
an ordered list of keyframes, each tagged with the phase it belongs to and, per channel, which
anchor the offset is measured from and which primitive is expected to explain it. The aggregator
turns a pile of these into fitted constants; the generator turns those into planner source.

STDLIB ONLY, ON PURPOSE. No torch, no isaaclab, no api imports. Loading and aggregating sessions
has to work outside the simulator environment, and a schema module that drags in the sim is a
schema module nobody can lint a session file with.

VERSIONING
``schema_version`` is written into every file.

  v0  the original ``PlannerWorkbench._export()`` payload: ONE ``scene_state`` snapshot, no
      ``schema_version`` key at all, no keyframe list, no phase tags, no channel modes.
  v2  the first keyframe format. (v1 is skipped so the version number can never be confused with
      the one-snapshot-per-file era, which had no number.) Door channels were tagged with the
      "world" anchor, and there were no arm-joint channels.
  v3  this format. Adds the arm_j1..arm_j7 channels and the robot_joints / door_joints /
      robot_initial anchors, and REQUIRES the two joint families to name their own anchor -- a v2
      file's `door_lever @ world` is invalid here. A v2 file therefore cannot be read: its door
      channels claim an anchor this build reserves for genuinely absolute world coordinates, and
      silently rewriting the tag would be exactly the kind of invented provenance this module
      refuses to produce. Re-capture in the workbench.

v0 files are REJECTED, loudly, by name. They are not upgraded. An untagged snapshot has no phase,
no per-channel anchor and no relative/absolute mode -- exactly the three things the aggregator
needs and cannot infer -- so a shim would have to invent them, and a generated planner built on
invented tags is worse than no planner. Re-capture in the workbench instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any

SCHEMA_VERSION = 3
_REJECTED_VERSIONS = {0, 1, 2}


class CaptureSchemaError(Exception):
    """Raised for anything the aggregator must not silently work around."""


# --------------------------------------------------------------------- vocabulary

# Ordered: this is also the order steps are emitted in a generated planner.
PHASE_IDS = (
    "pregrasp",
    "grasp",
    "unlatch",
    "pull_sweep",
    "release",
    "retract",
    "push_approach",
    "push_contact",
    "push_final",
    "traverse",
)

# Phases whose keyframes carry a theta and are fitted against it rather than pooled as constants.
SWEPT_PHASES = frozenset({"pull_sweep", "traverse"})

# Where an offset is measured from. The first four are api.py anchor functions and take
# (door_urdf_path, door_initial_pose, joint_angles); the rest are resolved by the planner's own
# running state, which is how the existing code already works:
#   prev_palm    -- pull:299 (unlatch deltas) and pull:520 (retract) offset the LAST palm pose
#   prev_base    -- pull:455/481 (release) offset the base pose carried out of the pull loop
#   door_origin  -- push:673 anchors the forward sweep on door_initial_pose[:, :3]
#   world        -- the offset IS the world coordinate (pull:695 traverse_far_x = -1.0)
#   robot_joints -- not a position at all. The channel IS a commanded arm joint angle, so there is
#                   nothing to subtract; only ARM_CHANNELS may use it.
#   door_joints  -- likewise for the door's own two joints. Kept SEPARATE from robot_joints because
#                   they are different articulations with different owners: the planner commands
#                   the arm through solve_ik and the door through q_door, and pooling the two under
#                   one "joint" anchor would let a fit average an arm angle against a panel angle.
ANCHOR_NAMES = (
    "handle_bar",
    "hinge",
    "board_center",
    "board_edge",
    "prev_palm",
    "prev_base",
    #   robot_initial -- the robot's OWN starting pose. base_yaw is measured from it and emitted as
    #                    robot_initial_yaw.item() + delta (pull:174, 795, 973, 1061), so tagging it
    #                    "world" claimed an absolute heading the planner never commands.
    "robot_initial",
    "door_origin",
    "robot_joints",
    "door_joints",
    "world",
)

# Whether a channel adds to its anchor or replaces it. Mixed on adjacent lines of the same target
# in the existing planners (pull:382 is relative, pull:383 absolute), so it is per-channel.
CHANNEL_MODES = ("anchor_relative", "world_absolute")

POSITION_CHANNELS = ("base_x", "base_y", "palm_x", "palm_y", "palm_z")
ROTATION_CHANNELS = ("rot_roll", "rot_pitch", "rot_yaw", "base_yaw")
# The seven panda joints, in q_robot[3:10] order -- arm_j1 is FULL_JOINT_NAMES[3]. Spelled
# positionally rather than by URDF name so this module keeps its no-robot-imports promise.
ARM_CHANNELS = tuple(f"arm_j{i}" for i in range(1, 8))
DOOR_CHANNELS = ("door_panel", "door_lever")
CHANNEL_NAMES = POSITION_CHANNELS + ROTATION_CHANNELS + ARM_CHANNELS + DOOR_CHANNELS

# The two things a waypoint positions, kept addressable on their own. The planner sets them from
# DIFFERENT anchors on the same line (pull:216 puts the base on the door landmark while pull:236
# puts the palm on it with its own offsets), and pull:382-383 even mixes modes across one target's
# own channels -- so "which anchor" is a per-target question, never a per-phase one.
CHANNEL_GROUPS = {
    "base": ("base_x", "base_y"),
    "palm": ("palm_x", "palm_y", "palm_z"),
}

# Anchors that only make sense for one target. A palm cannot be measured from the base's carried
# pose and vice versa; the planner never does it, and letting it through would produce an offset
# whose meaning depends on which target happened to be written first.
_ANCHOR_SCOPE = {
    "prev_palm": ("palm_x", "palm_y", "palm_z"),
    "prev_base": ("base_x", "base_y"),
    "robot_initial": ("base_yaw",),
}

# Which anchor each joint family is REQUIRED to use. A joint angle measured from a door landmark is
# meaningless, and the two families must not share an anchor, so this is enforced, not defaulted.
_REQUIRED_ANCHORS = {
    **{c: "robot_joints" for c in ARM_CHANNELS},
    **{c: "door_joints" for c in DOOR_CHANNELS},
}

PRIMITIVE_TYPES = (
    "constant_offset",
    "rotate_with_theta",
    "hold_then_release",
    "linear_gain",
    "fractional_interpolation",
)

HANDLE_SIDES = ("left", "right")
OPENING_DIRECTIONS = ("pull", "push")


# ------------------------------------------------------------------ phase defaults

def _spec(anchor: str, mode: str, primitive: str) -> dict[str, str]:
    return {"anchor": anchor, "mode": mode, "primitive": primitive}


_CONST_REL = "constant_offset"

# What each phase normally SETS, read straight off the hand-written planners. A channel absent from
# a phase's spec is one that phase leaves alone -- the planners carry base_target_pose forward
# unchanged through grasp and unlatch, and this reproduces that rather than re-declaring it.
# These are GUI defaults only. The human can override any cell per keyframe.
PHASE_CHANNEL_DEFAULTS: dict[str, dict[str, dict[str, str]]] = {
    "pregrasp": {
        "base_x": _spec("hinge", "anchor_relative", _CONST_REL),
        "base_y": _spec("hinge", "anchor_relative", _CONST_REL),
        "base_yaw": _spec("robot_initial", "anchor_relative", _CONST_REL),
        "palm_x": _spec("hinge", "anchor_relative", _CONST_REL),
        "palm_y": _spec("hinge", "anchor_relative", _CONST_REL),
        "palm_z": _spec("hinge", "anchor_relative", _CONST_REL),
        "rot_roll": _spec("world", "world_absolute", _CONST_REL),
        "rot_pitch": _spec("world", "world_absolute", _CONST_REL),
        "rot_yaw": _spec("world", "world_absolute", _CONST_REL),
    },
    "grasp": {
        "palm_x": _spec("hinge", "anchor_relative", _CONST_REL),
        "palm_y": _spec("hinge", "anchor_relative", _CONST_REL),
        "palm_z": _spec("hinge", "anchor_relative", _CONST_REL),
    },
    # Step 3 nudges the GRASP pose rather than re-deriving it from the handle, and commands the
    # lever to its hard stop. Both deltas are off prev_palm for that reason.
    "unlatch": {
        "palm_y": _spec("prev_palm", "anchor_relative", _CONST_REL),
        "palm_z": _spec("prev_palm", "anchor_relative", _CONST_REL),
        "rot_roll": _spec("world", "world_absolute", _CONST_REL),
        "rot_pitch": _spec("world", "world_absolute", _CONST_REL),
        "rot_yaw": _spec("world", "world_absolute", _CONST_REL),
        "door_lever": _spec("door_joints", "world_absolute", _CONST_REL),
    },
    "pull_sweep": {
        "base_x": _spec("hinge", "anchor_relative", _CONST_REL),
        "base_y": _spec("world", "world_absolute", "linear_gain"),
        "palm_x": _spec("hinge", "anchor_relative", "rotate_with_theta"),
        "palm_y": _spec("hinge", "anchor_relative", "rotate_with_theta"),
        "palm_z": _spec("hinge", "anchor_relative", _CONST_REL),
        "rot_roll": _spec("world", "world_absolute", "linear_gain"),
        "rot_pitch": _spec("world", "world_absolute", _CONST_REL),
        "rot_yaw": _spec("world", "world_absolute", _CONST_REL),
        "door_lever": _spec("door_joints", "world_absolute", "hold_then_release"),
    },
    "release": {
        "base_x": _spec("prev_base", "anchor_relative", _CONST_REL),
        "base_y": _spec("world", "world_absolute", _CONST_REL),
        "base_yaw": _spec("robot_initial", "anchor_relative", _CONST_REL),
        "palm_x": _spec("prev_palm", "anchor_relative", _CONST_REL),
        "palm_y": _spec("prev_palm", "anchor_relative", _CONST_REL),
        "door_panel": _spec("door_joints", "world_absolute", _CONST_REL),
    },
    # pull:524 sets retract z ABSOLUTELY (retreat_palm_pos[:, 2] = 1.2) while x and y stay deltas
    # off the previous palm -- keeping the hand high is what makes the swing pass over the arx arm.
    "retract": {
        "palm_x": _spec("prev_palm", "anchor_relative", _CONST_REL),
        "palm_y": _spec("prev_palm", "anchor_relative", _CONST_REL),
        "palm_z": _spec("world", "world_absolute", _CONST_REL),
    },
    "push_approach": {
        "palm_x": _spec("board_center", "anchor_relative", _CONST_REL),
        "palm_y": _spec("board_center", "anchor_relative", _CONST_REL),
        "palm_z": _spec("board_center", "anchor_relative", _CONST_REL),
        "rot_roll": _spec("world", "world_absolute", _CONST_REL),
        "rot_pitch": _spec("world", "world_absolute", _CONST_REL),
        "rot_yaw": _spec("world", "world_absolute", _CONST_REL),
    },
    "push_contact": {
        "palm_x": _spec("board_center", "anchor_relative", _CONST_REL),
        "palm_y": _spec("board_center", "anchor_relative", _CONST_REL),
        "palm_z": _spec("board_center", "anchor_relative", _CONST_REL),
    },
    "push_final": {
        "palm_x": _spec("board_center", "anchor_relative", _CONST_REL),
        "palm_y": _spec("board_center", "anchor_relative", _CONST_REL),
        "palm_z": _spec("board_center", "anchor_relative", _CONST_REL),
        "door_panel": _spec("door_joints", "world_absolute", _CONST_REL),
    },
    "traverse": {
        "base_x": _spec("world", "world_absolute", "fractional_interpolation"),
        "base_y": _spec("world", "world_absolute", "fractional_interpolation"),
        "base_yaw": _spec("robot_initial", "anchor_relative", "fractional_interpolation"),
    },
}


def default_channel_spec(phase_id: str) -> dict[str, dict[str, str]]:
    """GUI pre-fill for a phase: which channels it sets, and how each is meant to be read."""
    if phase_id not in PHASE_CHANNEL_DEFAULTS:
        raise CaptureSchemaError(
            f"unknown phase_id {phase_id!r}; expected one of {', '.join(PHASE_IDS)}"
        )
    return {name: dict(spec) for name, spec in PHASE_CHANNEL_DEFAULTS[phase_id].items()}


# ------------------------------------------------------------- anchor overrides

# Which mode an anchor implies. Naming an anchor without flipping the mode is a SILENT NO-OP:
# ``_channel_value`` returns the raw world coordinate whenever mode is world_absolute, whatever the
# anchor says. So an override sets both, and "world" is the only way to ask for an absolute channel.
_ANCHOR_IMPLIED_MODE = {
    "handle_bar": "anchor_relative",
    "hinge": "anchor_relative",
    "board_center": "anchor_relative",
    "board_edge": "anchor_relative",
    "prev_palm": "anchor_relative",
    "prev_base": "anchor_relative",
    "door_origin": "anchor_relative",
    "robot_initial": "anchor_relative",
    # A joint angle is the value COMMANDED, not a displacement from a landmark, so both joint
    # anchors are absolute. "relative to the previous posture" is a different feature; if it is
    # ever wanted it needs its own anchor (prev_arm), not a mode flip on this one.
    "robot_joints": "world_absolute",
    "door_joints": "world_absolute",
    "world": "world_absolute",
}

def implied_mode(anchor: str) -> str:
    """The mode an anchor forces. See ``_ANCHOR_IMPLIED_MODE`` for why it is not a free choice."""
    if anchor not in _ANCHOR_IMPLIED_MODE:
        raise CaptureSchemaError(f"unknown anchor {anchor!r}")
    return _ANCHOR_IMPLIED_MODE[anchor]


def with_joint_channels(spec: dict, *, arm: bool = False, door: bool = False) -> dict:
    """Add the joint-angle channels to a phase spec, in place.

    Opt-in per phase: recording a posture turns a waypoint from "wherever IK lands" into "this
    exact configuration", which is what you want for a retract or a home pose and emphatically not
    what you want for a grasp that has to generalize across doors.
    """
    if arm:
        for channel in ARM_CHANNELS:
            spec.setdefault(
                channel,
                {"anchor": "robot_joints", "mode": "world_absolute",
                 "primitive": "constant_offset"},
            )
    if door:
        for channel in DOOR_CHANNELS:
            spec.setdefault(
                channel,
                {"anchor": "door_joints", "mode": "world_absolute",
                 "primitive": "constant_offset"},
            )
    return spec


# --------------------------------------------------------------------- dataclasses

@dataclass
class ChannelSample:
    """One channel's value at one keyframe, plus how to read it.

    ``value`` is metres for position channels, radians for rotation and door channels. For a
    rotation it is the roll/pitch/yaw fed to ``get_rotation_quat`` -- never a quaternion component,
    because that is the parametrization every existing offset was tuned in.
    """

    value: float
    anchor: str = "world"
    mode: str = "world_absolute"
    primitive: str = "constant_offset"
    # The door angle the ANCHOR is evaluated at, when it differs from the keyframe's own theta.
    # pull:559 evaluates get_board_pos at contact_virtual_door_angle = 1.1 while q_door is 1.35;
    # one number cannot carry both, so this stays separate and defaults to "same as the keyframe".
    anchor_eval_theta: float | None = None

    def validate(self, where: str, channel: str | None = None) -> None:
        required = _REQUIRED_ANCHORS.get(channel or "")
        if required and self.anchor != required:
            raise CaptureSchemaError(
                f"{where}: channel {channel!r} must use the {required!r} anchor, got "
                f"{self.anchor!r}. Arm angles and door angles are separate articulations; "
                "neither is measured from a door landmark."
            )
        scope = _ANCHOR_SCOPE.get(self.anchor)
        if scope and channel and channel not in scope:
            raise CaptureSchemaError(
                f"{where}: anchor {self.anchor!r} applies to {', '.join(scope)}, not "
                f"{channel!r}. The base and the end effector carry their own anchors."
            )
        if not required and self.anchor in ("robot_joints", "door_joints"):
            raise CaptureSchemaError(
                f"{where}: anchor {self.anchor!r} is only valid on "
                f"{'arm' if self.anchor == 'robot_joints' else 'door'} joint channels"
            )
        if self.anchor not in ANCHOR_NAMES:
            raise CaptureSchemaError(f"{where}: unknown anchor {self.anchor!r}")
        if self.mode not in CHANNEL_MODES:
            raise CaptureSchemaError(f"{where}: unknown mode {self.mode!r}")
        if self.primitive not in PRIMITIVE_TYPES:
            raise CaptureSchemaError(f"{where}: unknown primitive {self.primitive!r}")


@dataclass
class Keyframe:
    id: int
    phase_id: str
    # Groups keyframes that share one solve_ik reference_joint_pos. Note the granularity the
    # existing code actually uses: state_machine_offline_left_pull_door passes
    # LEFT_PULL_IK_ANCHOR_JOINT_POS to ALL of its solve_ik calls (Steps 1-8), and the right-pull
    # planner passes none at all. So one block per variant is the normal case, not one per step.
    continuity_block: str | None = None
    # False marks a real intermediate waypoint, not an oversight: pull:586's push approach is
    # non-key while its near-identical twin at pull:610 is key. collocate_and_playback splines
    # between KEYFRAMES, so this changes the trajectory, and it has to survive capture.
    mark_keyframe: bool = True
    theta: float | None = None
    q_door: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # q_robot[3:10] -- the seven panda joints, named. Records which IK BRANCH the human posed the
    # arm in, which is exactly what a null-space reference_joint_pos selects.
    arm_joint_snapshot: dict[str, float] = field(default_factory=dict)
    gripper_width: float | None = None
    # 1 inside continuity-critical loops so a hard frame stays near its predecessor instead of
    # jumping IK branches; 8 elsewhere. Matches the num_attempts argument in the planners.
    num_attempts: int = 8
    channels: dict[str, ChannelSample] = field(default_factory=dict)
    notes: str = ""

    def validate(self, where: str) -> None:
        if self.phase_id not in PHASE_IDS:
            raise CaptureSchemaError(f"{where}: unknown phase_id {self.phase_id!r}")
        for name, sample in self.channels.items():
            if name not in CHANNEL_NAMES:
                raise CaptureSchemaError(f"{where}: unknown channel {name!r}")
            sample.validate(f"{where} channel {name!r}", name)
        if self.phase_id in SWEPT_PHASES and self.theta is None:
            raise CaptureSchemaError(
                f"{where}: phase {self.phase_id!r} is swept and needs a theta, but none was "
                "recorded. Re-capture it with the theta sweep control."
            )


@dataclass(frozen=True)
class VariantClass:
    handle_side: str
    opening_direction: str
    door_geometry_bucket: str = "any"

    def key(self) -> str:
        return f"{self.handle_side}_{self.opening_direction}_{self.door_geometry_bucket}"

    def function_name(self) -> str:
        """The planner entry point this variant generates, matching the existing naming."""
        return (
            f"state_machine_offline_{self.handle_side}_{self.opening_direction}_door"
        )

    def validate(self, where: str) -> None:
        if self.handle_side not in HANDLE_SIDES:
            raise CaptureSchemaError(f"{where}: handle_side must be left/right, got {self.handle_side!r}")
        if self.opening_direction not in OPENING_DIRECTIONS:
            raise CaptureSchemaError(
                f"{where}: opening_direction must be pull/push, got {self.opening_direction!r}"
            )


@dataclass
class CaptureSession:
    variant_class: VariantClass
    door_urdf_path: str = ""
    robot_urdf_path: str = ""
    robot_initial_pose_world: list[float] = field(default_factory=list)
    door_initial_pose_world: list[float] = field(default_factory=list)
    door_z_lift_m: float = 0.0
    keyframes: list[Keyframe] = field(default_factory=list)
    exported_at: float = 0.0
    source_path: str = ""
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        where = self.source_path or "<session>"
        self.variant_class.validate(where)
        for keyframe in self.keyframes:
            keyframe.validate(f"{where} keyframe {keyframe.id}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


# ------------------------------------------------------------------- (de)serialize

def session_from_dict(payload: dict[str, Any], *, source_path: str = "") -> CaptureSession:
    where = source_path or "<session>"
    version = payload.get("schema_version")

    if version is None:
        raise CaptureSchemaError(
            f"{where}: no 'schema_version' key -- this is a v0 workbench export (one scene_state "
            f"snapshot, no keyframe list). v0 sessions cannot be aggregated: they carry no "
            f"phase_id, no per-channel anchor and no anchor_relative/world_absolute mode, and "
            f"guessing those would silently fabricate the offsets the generator emits. Re-capture "
            f"this door in the workbench (schema v{SCHEMA_VERSION})."
        )
    if version in _REJECTED_VERSIONS:
        raise CaptureSchemaError(
            f"{where}: schema_version {version} is no longer supported; re-capture at "
            f"v{SCHEMA_VERSION}."
            + (" v2 tagged door channels with the 'world' anchor and had no arm-joint channels; "
               "v3 requires door_joints / robot_joints." if version == 2 else "")
        )
    if version != SCHEMA_VERSION:
        raise CaptureSchemaError(
            f"{where}: schema_version {version} but this build reads v{SCHEMA_VERSION}."
        )
    if "keyframes" not in payload:
        raise CaptureSchemaError(f"{where}: v{SCHEMA_VERSION} session has no 'keyframes' list.")

    raw_variant = payload.get("variant_class") or {}
    variant = VariantClass(
        handle_side=raw_variant.get("handle_side", ""),
        opening_direction=raw_variant.get("opening_direction", ""),
        door_geometry_bucket=raw_variant.get("door_geometry_bucket", "any"),
    )

    keyframes = []
    for raw in payload["keyframes"]:
        channels = {
            name: ChannelSample(
                value=float(spec["value"]),
                anchor=spec.get("anchor", "world"),
                mode=spec.get("mode", "world_absolute"),
                primitive=spec.get("primitive", "constant_offset"),
                anchor_eval_theta=(
                    None if spec.get("anchor_eval_theta") is None
                    else float(spec["anchor_eval_theta"])
                ),
            )
            for name, spec in (raw.get("channels") or {}).items()
        }
        keyframes.append(
            Keyframe(
                id=int(raw["id"]),
                phase_id=raw["phase_id"],
                continuity_block=raw.get("continuity_block"),
                mark_keyframe=bool(raw.get("mark_keyframe", True)),
                theta=None if raw.get("theta") is None else float(raw["theta"]),
                q_door=[float(v) for v in raw.get("q_door", [0.0, 0.0])],
                arm_joint_snapshot={k: float(v) for k, v in (raw.get("arm_joint_snapshot") or {}).items()},
                gripper_width=(
                    None if raw.get("gripper_width") is None else float(raw["gripper_width"])
                ),
                num_attempts=int(raw.get("num_attempts", 8)),
                channels=channels,
                notes=raw.get("notes", "") or "",
            )
        )

    session = CaptureSession(
        variant_class=variant,
        door_urdf_path=payload.get("door_urdf_path", ""),
        robot_urdf_path=payload.get("robot_urdf_path", ""),
        robot_initial_pose_world=list(payload.get("robot_initial_pose_world") or []),
        door_initial_pose_world=list(payload.get("door_initial_pose_world") or []),
        door_z_lift_m=float(payload.get("door_z_lift_m", 0.0)),
        keyframes=keyframes,
        exported_at=float(payload.get("exported_at", 0.0)),
        source_path=source_path,
    )
    session.validate()
    return session


def load_session(path: str) -> CaptureSession:
    with open(path, encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise CaptureSchemaError(f"{path}: not valid JSON ({exc}).") from exc
    return session_from_dict(payload, source_path=path)


def load_sessions(paths: list[str]) -> list[CaptureSession]:
    """Load several sessions and refuse to mix variant classes.

    Left and right offsets are mirrored, not interchangeable -- pregrasp_palm_y_offset is +0.15 on
    one side and -0.15 on the other in the existing code, and a mean across them is 0.0, i.e. an
    offset that grasps nothing. Pooling is blocked here rather than checked later.
    """
    sessions = [load_session(path) for path in paths]
    if not sessions:
        raise CaptureSchemaError("no capture sessions given.")
    keys = {session.variant_class.key() for session in sessions}
    if len(keys) > 1:
        listing = "\n".join(
            f"  {session.variant_class.key():28s} {session.source_path}" for session in sessions
        )
        raise CaptureSchemaError(
            "capture sessions span more than one variant_class, which must never be pooled "
            f"(left/right offsets are mirrored; push/pull are unrelated):\n{listing}\n"
            "Run the generator once per variant_class."
        )
    return sessions


def save_session(session: CaptureSession, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(session.to_dict(), handle, indent=2)
    return path


# ---------------------------------------------------------------- geometry bucket

def derive_geometry_bucket(door_urdf_path: str, *, split: bool = False) -> str:
    """Bucket label for a door, read from its ``variant_meta.json``.

    Defaults to ``"any"`` -- ONE bucket for every door. The anchor functions in api.py already
    absorb per-door geometry (that is the whole reason one planner file generalizes across an asset
    set), so splitting buckets mostly just starves each fit of samples. Turn ``split`` on only when
    a fit residual says a single bucket genuinely cannot cover the spread.
    """
    if not split:
        return "any"
    meta_path = os.path.join(os.path.dirname(door_urdf_path), "variant_meta.json")
    if not os.path.exists(meta_path):
        return "any"
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    handle_type = (meta.get("handle") or {}).get("type", "unknown")
    width = float((meta.get("panel") or {}).get("width_m", 0.0))
    height = float((meta.get("actual_properties") or {}).get("handle_height_m", 0.0))
    # Coarse on purpose: 10 cm bins. Finer bins split demonstrations without splitting behaviour.
    return f"{handle_type}_w{round(width, 1):g}_h{round(height, 1):g}"


__all__ = [
    "ANCHOR_NAMES",
    "CHANNEL_MODES",
    "CHANNEL_NAMES",
    "CaptureSchemaError",
    "CaptureSession",
    "ChannelSample",
    "DOOR_CHANNELS",
    "HANDLE_SIDES",
    "Keyframe",
    "OPENING_DIRECTIONS",
    "PHASE_CHANNEL_DEFAULTS",
    "PHASE_IDS",
    "POSITION_CHANNELS",
    "PRIMITIVE_TYPES",
    "ROTATION_CHANNELS",
    "SCHEMA_VERSION",
    "SWEPT_PHASES",
    "VariantClass",
    "default_channel_spec",
    "derive_geometry_bucket",
    "load_session",
    "load_sessions",
    "save_session",
    "session_from_dict",
]
