"""The five waypoint primitives every offline door planner is built from.

Each one already existed, written inline (and under a different variable name) in
``offline_pull_door.py`` / ``offline_push_door.py``. Nothing here is new math; this module is the
single place they now live so the synthesized planners and the hand-written ones agree by
construction instead of by eye.

Where each primitive came from
------------------------------

1. ``offset_from_anchor``  -- ``anchor.clone()`` then ``[:, i] += k`` per axis.
   pull:217-225 (pregrasp base+palm), 256-260 (grasp palm), 455-456/481-482 (release base),
   520-525 (retract, off the previous palm pose), 566-571/590-595/620-627 (push contact);
   push Steps 1-3 in both variants.

2. ``rotate_xy``           -- ``_rotate_xy_clockwise`` / ``_rotate_xy_counterclockwise``,
   defined VERBATIM TWICE (pull:122,128 and push:40,46). pull:386, pull:975, push:291, push:621.

3. ``hold_then_release``   -- ``_pull_hinge_angle`` (pull:90-120). Pull planners only.

4. ``linear_gain``         -- one expression, four spellings:
   pull:383  ``base_target_pos[:, 1] = theta * pull_base_y_gain``           (c0 implicitly 0)
   pull:969  ``base_target_pos[:, 1] = pull_base_y_offset + theta * gain``
   pull:395  ``pull_rot_roll_base + pull_rot_roll_per_theta * theta``       (a ROTATION channel)
   push:301  ``hold_palm_rot_roll_base + 0.9 * theta``                      (gain unnamed inline)

5. ``lerp``                -- pull:667-682 (base x, y and yaw as scalars) and
   push:678-690 (base position as a vector). ``frac = step / steps`` in both.

WHY THE ROLL RAMP IS #4 AND NOT PART OF #2
The XY offset rotation and the roll ramp are independent, and the ramp is arithmetically the same
``c0 + gain * theta`` as the base-y gain -- just on a rotation channel. Folding it into #2 would
make #2 contain #4 and force a 2-D fit; kept apart, every channel is a single 1-D fit.

ANCHOR-RELATIVE vs WORLD-ABSOLUTE
``offset_from_anchor`` takes an ``absolute`` axis set because the existing planners mix the two on
adjacent lines of the SAME target -- pull:382-383 does ``[:, 0] += offset`` (relative to the hinge)
and ``[:, 1] = value`` (a world coordinate) one line apart. Treating a world-absolute channel as
anchor-relative puts the base off by the anchor's own coordinate, ~0.4 m on these assets.

REUSED, NOT RE-IMPLEMENTED
``_make_pose``, ``_append_state``, ``_init_planner_state``, ``get_rotation_quat``, ``_set_gripper``
and ``_pull_hinge_angle`` are imported from ``offline_pull_door`` and re-exported here, so a
generated planner reuses the exact code the hand-written ones run. That points the shared module at
a concrete planner, which is backwards -- but the alternative is either a third copy of each helper
or editing two tuned files, and neither is worth it. Moving the definitions here and leaving
imports behind in the planners is a safe follow-up whenever you want it.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence

import torch

from DoorOpening.utils.state_machine.offline_pull_door import (  # noqa: F401  (re-export)
    GRIPPER_Q_IDX,
    _append_state,
    _init_planner_state,
    _make_pose,
    _pull_hinge_angle,
    _set_gripper,
    get_rotation_quat,
)

RotationSense = Literal["clockwise", "counterclockwise"]

# The channels a captured keyframe can carry a value for. Positions are metres in the world frame;
# rotations are the roll/pitch/yaw fed to get_rotation_quat, NOT raw quaternion components -- every
# existing planner parametrizes orientation that way and every offset is tuned against it.
POSITION_CHANNELS = ("base_x", "base_y", "palm_x", "palm_y", "palm_z")
ROTATION_CHANNELS = ("rot_roll", "rot_pitch", "rot_yaw", "base_yaw")
DOOR_CHANNELS = ("door_panel", "door_lever")
CHANNEL_NAMES = POSITION_CHANNELS + ROTATION_CHANNELS + DOOR_CHANNELS

PRIMITIVE_TYPES = (
    "constant_offset",
    "rotate_with_theta",
    "hold_then_release",
    "linear_gain",
    "fractional_interpolation",
)


# --------------------------------------------------------------- 1. constant_offset

def offset_from_anchor(
    anchor: torch.Tensor,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    *,
    absolute: Iterable[str] = (),
) -> torch.Tensor:
    """``anchor`` (1, 3) displaced by (dx, dy, dz), per-axis relative or absolute.

    Axes named in ``absolute`` (any of ``"x"``, ``"y"``, ``"z"``) are ASSIGNED the given value as a
    world coordinate instead of added to the anchor -- reproducing pull:383's
    ``base_target_pos[:, 1] = ...`` beside pull:382's ``base_target_pos[:, 0] += ...``.
    """
    absolute = set(absolute)
    out = anchor.clone()
    for idx, (axis, value) in enumerate(zip("xyz", (dx, dy, dz))):
        if axis in absolute:
            out[:, idx] = value
        else:
            out[:, idx] += value
    return out


# ------------------------------------------------------------- 2. rotate_with_theta

def rotate_xy(
    x_offset: float,
    y_offset: float,
    theta: torch.Tensor | float,
    sense: RotationSense,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The closed-door XY offset carried around the hinge as the panel sweeps to ``theta``.

    ``sense`` picks which of the two existing helpers this is. It is NOT a function of handle_side:
    right-pull and left-push are clockwise, left-pull and right-push counterclockwise. The
    aggregator fits both and keeps whichever has the lower residual rather than guessing.
    """
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(float(theta))
    c, s = torch.cos(theta), torch.sin(theta)
    if sense == "clockwise":
        return x_offset * c + y_offset * s, -x_offset * s + y_offset * c
    if sense == "counterclockwise":
        return x_offset * c - y_offset * s, x_offset * s + y_offset * c
    raise ValueError(f"unknown rotation sense {sense!r}; expected clockwise/counterclockwise")


def _rotate_xy_clockwise(x_offset, y_offset, theta):
    """Kept at its original name so this module is a drop-in for the planners' local copies."""
    return rotate_xy(x_offset, y_offset, theta, "clockwise")


def _rotate_xy_counterclockwise(x_offset, y_offset, theta):
    return rotate_xy(x_offset, y_offset, theta, "counterclockwise")


# ------------------------------------------------------------- 3. hold_then_release

def hold_then_release(
    theta: float,
    level: float,
    hold_until_theta: float,
    release_by_theta: float,
) -> float:
    """Hold ``level`` until ``hold_until_theta``, then ramp linearly to 0 by ``release_by_theta``.

    Identical to ``_pull_hinge_angle``, which is re-exported above under its own name; this is the
    channel-agnostic spelling the generator emits. See that function's docstring for why the lever
    is held against its stop through the early pull instead of released on the first frame.
    """
    return _pull_hinge_angle(theta, level, hold_until_theta, release_by_theta)


# ------------------------------------------------------------------ 4. linear_gain

def linear_gain(theta: float, c0: float, gain: float) -> float:
    """``c0 + gain * theta``. Covers both the base-y drift and the palm roll ramp."""
    return c0 + gain * float(theta)


# ------------------------------------------------- 5. fractional_interpolation

def lerp(start, end, frac: float):
    """``start + frac * (end - start)`` for scalars, tuples or (1, N) tensors alike.

    ``frac = step / steps`` with ``step`` running 1..steps, so the final iteration lands exactly on
    ``end`` -- the convention both traverse loops already use.
    """
    if isinstance(start, torch.Tensor) or isinstance(end, torch.Tensor):
        return start + frac * (end - start)
    if isinstance(start, (tuple, list)):
        return type(start)(s + frac * (e - s) for s, e in zip(start, end))
    return start + frac * (end - start)


def lerp_steps(steps: int) -> Sequence[float]:
    """The ``frac`` sequence for a traverse loop of ``steps`` iterations."""
    return [step / steps for step in range(1, steps + 1)]


__all__ = [
    "CHANNEL_NAMES",
    "DOOR_CHANNELS",
    "GRIPPER_Q_IDX",
    "POSITION_CHANNELS",
    "PRIMITIVE_TYPES",
    "ROTATION_CHANNELS",
    "RotationSense",
    "_append_state",
    "_init_planner_state",
    "_make_pose",
    "_pull_hinge_angle",
    "_rotate_xy_clockwise",
    "_rotate_xy_counterclockwise",
    "_set_gripper",
    "get_rotation_quat",
    "hold_then_release",
    "lerp",
    "lerp_steps",
    "linear_gain",
    "offset_from_anchor",
    "rotate_xy",
]
