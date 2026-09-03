"""Offline pull-door planners for the Franka 2-finger gripper.

Gripper conventions used throughout this file (see glorbot.urdf):

* ``panda_hand`` +z is the APPROACH axis (out through the jaws); the grasp center
  ``palm_center`` sits ``GRIPPER_TCP_OFFSET`` along it.
* ``panda_hand`` +y is the JAW TRAVEL axis. The outer faces of the two fingers are the flat
  SIDES of the gripper, with outward normals along hand +y / -y.
* ``solve_ik`` drives ``panda_hand``, not the TCP, so every target here is authored as a TCP
  pose and converted with ``_hand_pose_from_tcp``.

Two wrist attitudes are used:

* FRONT GRASP -- rpy(pi/2, 0, yaw). Approach points horizontally at the door, jaws close
  VERTICALLY so they clamp across a horizontal lever bar (one finger above it, one below).
  The robot reaches in from the FRONT; it does not come down on the handle from above.
* SIDE BLOCK -- rpy(pi, 0, yaw). Fingers point DOWN and a finger's flat outer face is
  presented to the panel, so the door is blocked/pushed with the SIDE of the gripper over a
  long flat contact instead of being poked with the fingertips.
"""

import math
from typing import Literal

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_mul

from DoorOpening.constants.robot_constants import (
    DRIVEN_FINGER_JOINT_NAME,
    FRANKA_DEFAULT_JOINT_POS,
    FRANKA_JOINT_NAMES,
    FULL_JOINT_NAMES,
    GRIPPER_CLOSED_WIDTH,
    GRIPPER_OPEN_WIDTH,
)
from DoorOpening.utils.state_machine.api import get_board_pos, get_handle_bar_pos, get_hinge_pos, solve_ik

HandleSide = Literal["right", "left"]

# The hand is ONE number in q_robot now, not 16. Resolved from FULL_JOINT_NAMES so it follows
# the constants instead of being a hardcoded slice bound.
GRIPPER_Q_IDX = FULL_JOINT_NAMES.index(DRIVEN_FINGER_JOINT_NAME)

# palm_center's offset along panda_hand +z (matches the URDF's palm_center_joint).
GRIPPER_TCP_OFFSET = 0.1034

# Commanded half-opening. GRASP is fully closed: the handle bar stops the jaws early and the
# drive squeezes at its effort limit, which is how the ~50 N grasp force in glorbot_cfg was
# derived. OPEN is the full 40 mm so the jaws clear the bar on approach and release.
GRASP_WIDTH = GRIPPER_CLOSED_WIDTH
OPEN_WIDTH = GRIPPER_OPEN_WIDTH

# Half the outer width of a closed gripper: the distance from the TCP out to a finger's flat
# side face. Used to stand the TCP off a panel so the SIDE touches it, not the fingertips.
GRIPPER_SIDE_HALF_WIDTH = 0.027


def get_rotation_quat(roll, pitch, yaw, device):
    return quat_from_euler_xyz(
        roll=torch.tensor([[roll]], device=device),
        pitch=torch.tensor([[pitch]], device=device),
        yaw=torch.tensor([[yaw]], device=device),
    ).squeeze(0)


def _front_grasp_rot(yaw: float, device):
    """Wrist attitude for reaching in at the handle from the FRONT.

    roll=pi/2 puts the jaw-travel axis vertical (fingers straddle a horizontal lever bar);
    ``yaw`` aims the approach axis. At yaw=-pi/2 the approach points along world -x, i.e.
    straight at a door whose face the robot is standing in front of.
    """
    return get_rotation_quat(FRONT_GRASP_ROLL, 0.0, yaw, device)


def _side_block_rot(yaw: float, device, roll: float = math.pi):
    """Wrist attitude for blocking/pushing the panel with a finger's flat SIDE face.

    roll=pi points the fingers DOWN and leaves the jaw axis horizontal, so the outer face of a
    finger (normal = hand +y) faces sideways. ``yaw`` swings that face onto the panel: the
    world-frame face normal is (sin yaw, -cos yaw, 0), so yaw=pi presses toward +y and yaw=0
    presses toward -y.

    ``roll=0`` points the fingers UP instead. That presents the SAME flat side face to the panel
    and blocks it just as well, but reaches the pose through a completely different arm posture --
    which is the point: there is no single correct blocking attitude, and the fingers-down one is
    not always the one the arm can hold without contorting. Candidates are generated over both
    rolls (and both jaw flips) and picked on joint quality; see the block-attitude search.
    """
    return get_rotation_quat(roll, 0.0, yaw, device)


FRONT_APPROACH_YAW = -math.pi / 2
# Roll of the front-grasp attitude. The claim that +pi/2 and -pi/2 give "the same grasp... up
# to sign" is false for a PARALLEL jaw: they put the APPROACH axis on opposite sides, not just a
# different IK branch. At -pi/2 (the value this used to be) approach = world +x -- AWAY from a
# door standing at -x from ROBOT_INITIAL_POS -- which forces panda_hand (offset
# GRIPPER_TCP_OFFSET BEHIND the TCP along approach) onto the panel's far side of the TCP the
# planner is aiming at the handle for. Confirmed with scripts/debug/check_pull_collisions.py:
# panda_link6 (forearm) clips the panel on 89/150 sampled frames at -pi/2 versus 44/150 at
# +pi/2, over nearly the whole grasp+pull (board 0.00-1.50 rad), matching the module docstring
# above ("roll=pi/2 ... approach points horizontally at the door") that this constant contradicted.
# TILTED approach: come in at APPROACH_TILT above horizontal -- "half top half front".
#
# This has TWO halves and they must move together, which an earlier attempt got wrong:
#   * the wrist ROLL (here, via FRONT_GRASP_ROLL) aims the approach AXIS, and
#   * the standoff DIRECTION (see approach_dir in Step 1) is where the hand actually starts from.
# Setting only the roll rotates the hand about a path that is still a level, pure +x reach, which
# is a different grasp entirely and measured as a regression. Both are now derived from this one
# angle, so the hand genuinely comes in from above-and-in-front.
#
# A purely horizontal reach (tilt 0) makes the WRIST carry the whole ~72 deg door swing, which is
# what pinned panda_joint6/joint7 on their limits. A purely vertical one leans on the forearm for
# the entire motion. 60 deg splits the load.
APPROACH_TILT = math.pi / 3
FRONT_GRASP_ROLL = math.pi / 2 + APPROACH_TILT


def _jaw_flip(quat: torch.Tensor, device) -> torch.Tensor:
    """The SAME grasp with the jaws swapped: 180 deg about the gripper's own approach axis."""
    return quat_mul(quat, get_rotation_quat(0.0, 0.0, math.pi, device))


def _closest_jaw_equivalent(desired: torch.Tensor, previous: torch.Tensor | None, device) -> torch.Tensor:
    """Pick whichever of the two equivalent jaw attitudes is nearer the previous one.

    A parallel gripper is symmetric under 180 deg about its approach axis -- both attitudes grasp
    identically -- so the IK is free to return either, and left to itself it flips between them
    frame to frame. On screen that is the gripper spinning on the handle for no reason. Choosing
    the nearer one each frame pins the whole trajectory to a single branch.
    """
    if previous is None:
        return desired
    flipped = _jaw_flip(desired, device)
    d0 = float(torch.abs((desired * previous).sum()))
    d1 = float(torch.abs((flipped * previous).sum()))
    return desired if d0 >= d1 else flipped


# Index of panda_joint6 within q_robot[:10] == BASE_JOINT_NAMES(3) + FRANKA_JOINT_NAMES(7), the
# same joint order PinocchioIKSolver is constructed with everywhere in this file.
_PANDA_JOINT6_IDX = 3 + FRANKA_JOINT_NAMES.index("panda_joint6")
# glorbot.urdf's safety_controller soft range for panda_joint6 (the URDF's own solve_ik-enforced
# HARD <limit> is -0.0873..3.8223 -- wider, and NOT what a real controller running under the
# safety_controller would tolerate for long).
_PANDA_JOINT6_SOFT_RANGE = (-0.0175, 3.7525)
_PANDA_JOINT7_IDX = 3 + FRANKA_JOINT_NAMES.index("panda_joint7")
# panda_joint7's safety_controller soft range (hard <limit> is +-2.9671).
#
# Guarding this joint is what stops the wrist making a pointless ~360 deg revolution in the block
# phase. joint7 is symmetric about zero and the block attitude sits near its limit, so IK parks the
# wrist AT +2.9671 for the whole block approach; the moment a later waypoint needs slightly more,
# the only way round is to unwind to the OPPOSITE limit, and the collocation spline renders that as
# a full spin (measured: +2.967 -> -2.649, a 322 deg sweep, between the last block_approach and
# block_contact). Keeping joint7 off its limit in the first place removes the escape entirely.
_PANDA_JOINT7_SOFT_RANGE = (-2.8973, 2.8973)
# Tighter WORKING range for the grasp and pull, reserving headroom for what comes after them.
#
# Guarding those phases to the full soft range is not enough: it only asks "is joint7 legal now?",
# and the answer stayed yes right up to |j7| = 2.83 -- legal, but with nothing left. The pull then
# turns the wrist by up to pull_wrist_swing_cap about the door normal and the retract has to turn
# it again to the block attitude, so arriving at the release with no headroom left the retract able
# to escape only by unwinding ~300 deg to the opposite limit. Reserving ~0.6 rad here (the swing
# cap) keeps the whole grasp->pull->retract chain inside one continuous branch.
_PANDA_JOINT7_WORKING_RANGE = (-2.30, 2.30)
_GRASP_PULL_GUARDED_JOINTS = (
    (_PANDA_JOINT6_IDX, *_PANDA_JOINT6_SOFT_RANGE),
    (_PANDA_JOINT7_IDX, *_PANDA_JOINT7_WORKING_RANGE),
)


def _solve_ik_avoiding_wrist_limits(
    robot_urdf_path,
    q_robot: torch.Tensor,
    palm_pose,
    base_pose,
    robot_initial_pose,
    device,
    num_attempts: int = 8,
    retries: int = 30,
    first_num_attempts: int | None = None,
    guarded_joints: tuple = (
        (_PANDA_JOINT6_IDX, *_PANDA_JOINT6_SOFT_RANGE),
        (_PANDA_JOINT7_IDX, *_PANDA_JOINT7_SOFT_RANGE),
    ),
) -> torch.Tensor:
    """solve_ik, re-rolled up to `retries` times, keeping whichever result leaves the guarded
    wrist joints furthest inside their safety-controller soft ranges.

    Two joints are guarded, for two different failure modes:

    * panda_joint6 -- lands anywhere from 2.8 rad (safe) to 3.82 rad (its hard limit) for an
      IDENTICAL target depending purely on which redundant elbow branch the seed falls into.
    * panda_joint7 -- the block attitude sits near its limit, so IK parks it AT +-2.9671 and the
      only later escape is a full unwind to the opposite limit, which the collocation spline
      renders as a pointless ~360 deg wrist revolution.

    IMPORTANT, and the reason this perturbs the SEED itself rather than just raising
    `num_attempts`: pin.py's compute_ik tries seeds = [q_init, *random restarts] in order and
    "returns the FIRST converged solution immediately" -- so once the continuity seed q_init
    itself converges (the common case for a frame-to-frame continuity call), calling solve_ik
    again with a bigger num_attempts is a no-op: it deterministically reproduces the SAME
    q_init-converged result every time, because the random restarts are never even reached. A
    first version of this function did exactly that and measurably did not help the pull loop.
    So: retry 0 uses q_robot as-is (`first_num_attempts`, if given, controls its num_attempts --
    e.g. the pull loop passes 1, matching its original single-seed-continuity behaviour exactly
    in the common/already-safe case). Only if that is unsafe do later retries perturb q_robot's
    ARM joints with fresh random noise before calling solve_ik again, so a genuinely different
    seed -- not just a bigger num_attempts on the same seed -- gets tried.

    Traced with dump_pull_geometry-style instrumentation: for an IDENTICAL target, panda_joint6
    (URDF hard limit 3.8223 rad, safety_controller soft limit 3.7525) landed anywhere from 2.8 rad
    (safe, mid-range) to 3.82 rad (pinned at the hard limit) purely depending on which redundant
    elbow/null-space branch the seed put the DLS solve into.

    `retries=30` was chosen empirically, not guessed: this is also the answer to "is a FRONT
    approach even solvable for a ~72 deg door swing" (see the debug session's change log) --
    reducing the WRIST's tracking demand instead (pull_wrist_swing_cap 0.6->0.35, restoring
    pull_base_yaw_gain to share the swing, lifting the TCP during the pull) measurably did NOT
    help or made it worse, across repeated 10-trial panda_joint6 sweeps. What DOES help is simply
    searching harder for the safe branch: 10 retries -> 6/10 clean trials, 1-2/10 with a HARD
    (3.8223) violation; 20 retries -> 8/10 clean, 0/10 hard; 30 retries -> 9/10 clean, 0/10 hard,
    at ~5s/door extra planning time (acceptable for an offline planner). A safe branch reliably
    EXISTS for the front approach; it just isn't always the first one solve_ik's own random
    restarts happen to find.
    """
    arm_slice = slice(3, 10)  # FRANKA_JOINT_NAMES within q_robot[:10] == BASE(3) + FRANKA(7)
    best_q = None
    best_margin = -float("inf")
    best_travel = float("inf")
    # Among solutions that are SAFE, take the one closest to the pose we came from. The random
    # seed perturbation below explores different elbow branches on purpose, but a distant branch
    # that happens to be safe is still a large unexplained wrist/arm swing between two adjacent
    # keyframes -- which is exactly what collocation renders as a spin. Safety first, then
    # continuity; without this tie-break the retries themselves were a source of the swings.
    for attempt_idx in range(retries):
        if attempt_idx == 0:
            q_seed = q_robot[:10]
            attempts = first_num_attempts if first_num_attempts is not None else num_attempts
        else:
            q_seed = q_robot[:10].clone()
            q_seed[arm_slice] = q_seed[arm_slice] + torch.randn(7, device=q_seed.device) * 0.5
            attempts = num_attempts
        q_try = solve_ik(
            robot_urdf_path,
            q_seed,
            palm_pose=palm_pose,
            base_pose=base_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=attempts,
        )[0]
        # Worst margin over every guarded joint: a solution is only as good as its tightest one.
        margin = min(
            min(float(q_try[idx]) - j_lo, j_hi - float(q_try[idx]))
            for idx, j_lo, j_hi in guarded_joints
        )
        travel = float(torch.abs(q_try[arm_slice] - q_robot[arm_slice]).max())
        if attempt_idx == 0 and margin >= 0:
            # Attempt 0 IS the continuity seed. If it is already safe it is also, by construction,
            # the most continuous answer available -- take it and skip the search entirely. This
            # is the common case, and short-circuiting keeps planning at its original speed.
            return q_try
        if best_margin < 0:
            # Nothing safe found yet: margin is all that matters.
            better = margin > best_margin
        else:
            # A safe solution is already in hand: only take another safe one, and only if it
            # moves the arm LESS than the incumbent.
            better = margin >= 0 and travel < best_travel
        if better:
            best_margin, best_travel, best_q = margin, travel, q_try
        # Stop once we hold a safe solution that is also a small move; keep looking if the only
        # safe one so far is a big branch jump, since a nearer safe branch may still turn up.
        if best_margin >= 0 and best_travel <= 0.8:
            break
    return best_q


def _turn_about_door_normal(base_quat: torch.Tensor, angle: float, device) -> torch.Tensor:
    """Spin an attitude about WORLD x -- the shut door's normal, and the lever's pivot axis.

    Applied as a world-frame PRE-multiply. It cannot be done by bumping the roll term of
    ``rpy(roll, 0, yaw)``: yaw is applied last there, so a larger roll pitches the approach axis
    down out of horizontal (at 1.05 rad it tips 60 deg and the gripper stops facing the door)
    instead of rotating the gripper about the axis it is gripping.
    """
    return quat_mul(get_rotation_quat(angle, 0.0, 0.0, device), base_quat)


def _unit(vec, device, dtype):
    return torch.tensor([vec], device=device, dtype=dtype)


def _hand_pose_from_tcp(tcp_pos: torch.Tensor, quat: torch.Tensor, device) -> torch.Tensor:
    """Convert a desired TCP (grasp-center) pose into the panda_hand pose solve_ik expects."""
    approach = quat_apply(quat, _unit([0.0, 0.0, 1.0], device, tcp_pos.dtype))
    return _make_pose(tcp_pos - GRIPPER_TCP_OFFSET * approach, quat)


def _side_face_normal(quat: torch.Tensor, sign: float, device, dtype) -> torch.Tensor:
    """World direction of the gripper face that meets the panel.

    ``sign`` picks WHICH finger's outer face does the blocking: +1 is the hand's +y face
    (panda_leftfinger), -1 the -y face (panda_rightfinger). Because the blocking attitude rolls
    the hand by pi to point the fingers down, the URDF's leftfinger is the one that ends up on
    the robot's visual RIGHT -- so flipping ``sign`` swaps which side of the gripper touches the
    panel WITHOUT changing the direction it presses (``block_face_yaw`` is adjusted to match).
    """
    return sign * quat_apply(quat, _unit([0.0, 1.0, 0.0], device, dtype))


def _make_pose(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([position, quat], dim=-1)


def _set_gripper(q_robot: torch.Tensor, width: float) -> None:
    q_robot[GRIPPER_Q_IDX] = width


def _append_state(
    robot_traj: list[torch.Tensor],
    door_traj: list[torch.Tensor],
    key_indices: list[int],
    q_robot: torch.Tensor,
    q_door: torch.Tensor,
    *,
    mark_keyframe: bool,
) -> None:
    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())
    if mark_keyframe:
        key_indices.append(len(robot_traj) - 1)


def _rotate_xy_clockwise(x_offset: float, y_offset: float, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return x_offset * c + y_offset * s, -x_offset * s + y_offset * c


def _rotate_xy_counterclockwise(x_offset: float, y_offset: float, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return x_offset * c - y_offset * s, x_offset * s + y_offset * c


def _rotate_xy_about_z(x_offset: float, y_offset: float, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed rotation about +z. Positive is counterclockwise seen from above."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    return x_offset * c - y_offset * s, x_offset * s + y_offset * c


def _init_planner_state(robot_initial_q, door_initial_q):
    robot_traj: list[torch.Tensor] = []
    door_traj: list[torch.Tensor] = []
    key_idx_in_key_indices: list[int] = []

    q_robot = robot_initial_q.clone()
    q_door = door_initial_q.clone()

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=False,
    )

    return q_robot, q_door, robot_traj, door_traj, key_idx_in_key_indices


def state_machine_offline_right_pull_door(
    robot_urdf_path,
    door_urdf_path,
    robot_initial_pose,   # (1, 7) world
    door_initial_pose,    # (1, 7) world
    robot_initial_q,      # (ndof,)
    door_initial_q,       # (2,) [board, hinge]
    device="cpu",
):
    """
    Offline planner for a right-side handle, pull-type door.

    This is intentionally separate from the left-door function so all
    right-door tuning stays local and obvious.
    """

    q_robot, q_door, robot_traj, door_traj, key_idx_in_key_indices = _init_planner_state(
        robot_initial_q, door_initial_q
    )

    base_target_rot = robot_initial_pose[:, 3:].to(device).clone()
    grasp_rot = _front_grasp_rot(FRONT_APPROACH_YAW, device)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    franka_default_q = torch.tensor(
        [FRANKA_DEFAULT_JOINT_POS[name] for name in FRANKA_JOINT_NAMES],
        device=device,
    )

    # -------------------------
    # Step 1: Pregrasp
    # -------------------------
    handle_pos = get_hinge_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    # Base stands further back than the LEAP build's 0.67. Pulling from further out keeps the
    # arm nearer full extension along -x, so the pull force runs down the arm instead of being
    # carried by the wrist, and leaves room for the base to retreat as the door comes at it.
    pregrasp_base_x_offset = 0.85
    pregrasp_base_y_offset = -0.35
    # FRONT approach: the TCP lines up with the handle in y/z and stands off along +x only, so
    # the gripper comes straight in at the bar. (The LEAP build hovered 25 cm ABOVE the handle
    # and came down on it; a parallel jaw has to arrive along the bar's normal instead.)
    pregrasp_tcp_x_standoff = 0.25
    pregrasp_tcp_y_offset = 0.0
    pregrasp_tcp_z_offset = 0.0

    base_target_pos = handle_assembly_pos.clone()
    base_target_pos[:, 0] += pregrasp_base_x_offset
    base_target_pos[:, 1] += pregrasp_base_y_offset
    base_target_pose = _make_pose(base_target_pos, base_target_rot)

    tcp_target_pos = handle_pos.clone()
    tcp_target_pos[:, 0] += pregrasp_tcp_x_standoff
    tcp_target_pos[:, 1] += pregrasp_tcp_y_offset
    tcp_target_pos[:, 2] += pregrasp_tcp_z_offset
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, grasp_rot, device)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 2: Move to grasp
    # -------------------------
    # DEPTH knob: how far past the handle centroid the grasp center is driven, along -x toward
    # the door. Negative reaches DEEPER. Used by the GRASP and, unchanged, by the whole pull
    # sweep -- so this one number decides whether the bar stays seated between the jaws while
    # the door is being dragged open, which is exactly where a shallow grip lets go.
    # Limit: the fingertips sit ~9 mm beyond the TCP, so this cannot exceed the bar's standoff
    # from the panel or the tips bottom out on the door before the jaws close.
    grasp_tcp_x_offset = -0.035
    # Slide along the lever toward its free end (-y on a right-handle door) so the jaws clamp a
    # solid section of bar rather than the root where it meets the rose.
    grasp_tcp_y_offset = -0.03
    grasp_tcp_z_offset = 0.0

    # Walk in along the approach in small steps rather than jumping the whole standoff at once.
    # A single ~28 cm move is enough for the solver to come back on a different branch, which is
    # what makes the arm look like it re-solved from scratch instead of adjusting from where it
    # was. These are non-keyframes: they only shape the path in, they are not poses to hold.
    approach_steps = 4
    for approach_step in range(1, approach_steps + 1):
        frac = approach_step / (approach_steps + 1)
        step_tcp = handle_pos.clone()
        step_tcp[:, 0] += pregrasp_tcp_x_standoff + (grasp_tcp_x_offset - pregrasp_tcp_x_standoff) * frac
        step_tcp[:, 1] += pregrasp_tcp_y_offset + (grasp_tcp_y_offset - pregrasp_tcp_y_offset) * frac
        step_tcp[:, 2] += pregrasp_tcp_z_offset + (grasp_tcp_z_offset - pregrasp_tcp_z_offset) * frac
        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=_hand_pose_from_tcp(step_tcp, grasp_rot, device),
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # seeded from the previous frame, so the arm adjusts rather than re-solves
        )[0]
        _set_gripper(q_robot, OPEN_WIDTH)
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=False,
        )

    tcp_target_pos = handle_pos.clone()
    tcp_target_pos[:, 0] += grasp_tcp_x_offset
    tcp_target_pos[:, 1] += grasp_tcp_y_offset
    tcp_target_pos[:, 2] += grasp_tcp_z_offset
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, grasp_rot, device)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    # Arrive at the bar with the jaws STILL OPEN. Closing is a separate waypoint below at the
    # SAME arm pose, so the spline finishes the reach before the fingers start to move --
    # otherwise the gripper closes across the whole approach and swipes the handle aside on its
    # way in.
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # Now that the bar is between the jaws, close on it. Base and arm are untouched, so this
    # keyframe moves the gripper number and nothing else.
    _set_gripper(q_robot, GRASP_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 3: Rotate hinge (unlatch)
    # -------------------------
    unlatch_hinge_angle = 1.05
    # Turning a lever presses its free end DOWN, so the grasped point swings DOWNWARD about the
    # rose and the wrist rolls with it about the approach axis (world x, the door normal).
    # The TCP arc is commanded EXPLICITLY here instead of being read back from door FK:
    # get_hinge_pos returns the handle centroid, which barely drops as the lever turns, so
    # following it slid the gripper sideways rather than pressing the lever down.
    unlatch_tcp_z_delta = -0.06
    # Toward the rose as the lever swings down (its y-reach shortens by cos(angle)). A
    # RIGHT-handle lever's free end points -y, so that is +y here.
    unlatch_tcp_y_delta = 0.02
    # Sign set from OBSERVED behaviour, not derived: the free-end-swings-down argument (which
    # gives +1 here) turns the gripper the wrong way in sim, so the lever's actual travel in the
    # door URDF is the opposite of that. The left door carries the mirrored value.
    unlatch_roll_per_angle = -1.0

    q_door = torch.tensor([0.0, unlatch_hinge_angle], device=device)

    unlatch_rot = _turn_about_door_normal(
        grasp_rot,
        unlatch_roll_per_angle * unlatch_hinge_angle,
        device,
    )

    tcp_target_pos = handle_pos.clone()
    tcp_target_pos[:, 0] += grasp_tcp_x_offset
    tcp_target_pos[:, 1] += grasp_tcp_y_offset + unlatch_tcp_y_delta
    tcp_target_pos[:, 2] += grasp_tcp_z_offset + unlatch_tcp_z_delta
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, unlatch_rot, device)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, GRASP_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 4: Pull door open
    # -------------------------
    # Starts at theta=0, so the FIRST frame reproduces the unlatch keyframe exactly and the door
    # walks open from there. Previously this began at 0.3 with the lever snapped straight and the
    # base teleported to its pull pose, so solve_ik was handed a target a long way from its seed
    # and returned a completely different arm branch -- that is the thrashing.
    pull_theta_start = 0.0
    # How far the gripper drags the door before letting go. Shortening this is tempting -- the
    # gripper stays much better aligned to the handle (14 deg vs 29 deg at 0.60) -- but MEASURED
    # over 5 doors it is far worse overall, because the blocking phase then has to reach around a
    # panel that has not swung clear:
    #     stop 1.25 rad (72 deg): panel collisions  23/200 frames
    #     stop 0.60 rad (34 deg): panel collisions 131/200 frames
    # The long pull is what buys the arm room to get around the leaf. Do not shorten it without
    # rebuilding the blocking approach for a less-open door.
    pull_theta_stop = 1.25
    # Halved: more waypoints means each solve is a smaller step from the previous seed, which
    # keeps the arm on one IK branch instead of hopping between them.
    pull_theta_step = 0.025

    # panda_link0 is mounted at +0.178 on the chassis, i.e. on the side AWAY from the door, and
    # the front approach puts panda_hand 0.1034 nearer the robot than the grasp point. So the
    # x-reach the arm must cover is (standoff + 0.178 - 0.1034) -- the front approach needs LESS
    # reach than the old top-down grasp, not more, and standing further back buys elbow room
    # rather than costing it.
    pull_base_x_offset = 0.55
    # LATERAL stance, held relative to the SWINGING handle -- the x offset already works this
    # way. This used to be an absolute world y near 0 while the handle swept from -0.33 to +0.26,
    # so the base never followed sideways and the arm was left reaching ~0.5 m across the body by
    # the end of the pull. That, not the standoff, was what put the gripper off the handle.
    pull_base_y_offset = -0.35
    # Door travel over which the lever unwinds and the base eases from its grasp pose into its
    # pull pose. Everything that used to jump at the phase boundary is blended across this.
    pull_ease_theta = 0.35
    prev_grip_rot = grasp_rot   # pins the jaw branch so the gripper cannot flip mid-pull

    # SINGLE knob for how the door swings, used for BOTH the offset rotation and the wrist yaw.
    # The handle is rigid with the panel, so a gripper holding it must rotate by exactly the
    # panel's angle -- the grip offset and the wrist have to turn together or the jaws twist off
    # the bar. Keeping one sign makes disagreeing impossible. Negative = clockwise seen from
    # above, which is how a right-hand door swings. Flip if the gripper unwinds off the handle.
    pull_swing_sign = -1.0
    # Let the BASE absorb part of the swing so the wrist does not have to take all ~72 deg of it
    # alone; a wrist near its limit is where the IK starts returning contorted solutions. Set to
    # 0.0 to restore a base held at fixed yaw.
    # Base does NOT turn during the pull. Measured, turning it buys nothing (panel collisions 23
    # either way, jamb 10 vs 14 over 5 doors) while costing a visible base rotation, so it is off.
    pull_base_yaw_gain = 0.0
    # ...and BACKS OFF as it turns: yawing toward the door without retreating swings the chassis
    # corner into the panel's path. Metres of extra standoff per radian of door opening.
    pull_base_backoff_gain = 0.25
    # WHICH WAY the base turns during the pull. Must match the blocking tilt in step 5. This was
    # driven off pull_swing_sign, which is the DOOR's rotation and happens to be OPPOSITE, so the
    # base yawed one way through the whole pull and then spun back the other way to block -- a
    # 2.25 rad reversal that reads as the base turning the wrong way.
    blocking_turn_sign = 1.0
    # How much of the door's swing the WRIST tracks. Kinematically this "should" be 1.0 -- the
    # gripper is clamped to the bar, so it ought to rotate exactly with the panel. Measured, the
    # arm cannot do it: scripts/debug/dump_pull_geometry.py shows mean TCP-to-handle error of
    # 0.099 m at gain 1.0 versus 0.045 m at 0.2-0.3, and jaw-off-vertical 38 deg versus 14 deg.
    # Past ~0.4 the IK gives up on the orientation and throws position away with it. The gap is
    # taken up by the bar rotating within the jaws, which a parallel gripper on a round lever
    # tolerates. Raise it only if the arm gains reach or the pull range shrinks.
    pull_wrist_swing_cap = 0.6
    pull_keyframe_every = 2

    _, _, pull_base_yaw0 = euler_xyz_from_quat(base_target_rot)
    # Where the grasp left the base -- the pull eases out of this rather than jumping.
    grasp_base_x = base_target_pos[:, 0].clone()
    grasp_base_y = base_target_pos[:, 1].clone()

    theta_values = torch.arange(
        pull_theta_start,
        pull_theta_stop + 1e-6,
        pull_theta_step,
        device=device,
    )
    for pull_step, theta in enumerate(theta_values):
        th = theta.item()
        ease = min(1.0, th / pull_ease_theta) if pull_ease_theta > 0 else 1.0
        # The lever is still held down as the pull begins and unwinds as the door takes over,
        # instead of snapping straight in one frame while the gripper is still gripping it.
        lever_frac = 1.0 - ease
        lever_angle = unlatch_hinge_angle * lever_frac
        q_door = torch.tensor([th, lever_angle], device=device)

        handle_pos = get_hinge_pos(
            door_urdf_path,
            door_initial_pose,
            q_door.unsqueeze(0),
        ).to(device)

        # Stance is held in WORLD axes, deliberately NOT rotated with the swing: a pull door
        # comes AT the robot, so the base has to back away along -x and hold its side, not orbit
        # the handle. Rotating the stance was measured worse (0.073 m vs 0.046 m mean) and would
        # walk the base through the doorway/wall at large angles.
        swing = pull_swing_sign * theta
        base_target_pos = handle_pos.clone()
        base_target_pos[:, 0] = grasp_base_x + (
            handle_pos[:, 0] + pull_base_x_offset + pull_base_backoff_gain * th - grasp_base_x
        ) * ease
        base_target_pos[:, 1] = grasp_base_y + (
            handle_pos[:, 1] + pull_base_y_offset - grasp_base_y
        ) * ease
        base_target_pose = _make_pose(
            base_target_pos,
            get_rotation_quat(
                0.0,
                0.0,
                pull_base_yaw0.item() + blocking_turn_sign * pull_base_yaw_gain * th,
                device,
            ),
        )

        # EXACTLY the grasp offset, carried rigidly with the panel: same x depth, same y along
        # the bar, same z, so the gripper holds the identical point it closed on in step 2. The
        # unlatch deltas fade out on the same schedule as the lever, so theta=0 reproduces the
        # unlatch pose and theta>=pull_ease_theta is the pure grasp pose.
        tcp_dx, tcp_dy = _rotate_xy_about_z(
            grasp_tcp_x_offset,
            grasp_tcp_y_offset + unlatch_tcp_y_delta * lever_frac,
            swing,
        )

        tcp_target_pos = handle_pos.clone()
        tcp_target_pos[:, 0] += tcp_dx
        tcp_target_pos[:, 1] += tcp_dy
        tcp_target_pos[:, 2] += grasp_tcp_z_offset + unlatch_tcp_z_delta * lever_frac

        # Wrist follows the door 1:1 up to a CAP, then saturates -- rather than a fractional gain
        # that is wrong everywhere. Early on, while the grasp is being established and the jaws
        # must stay square to the bar, the gripper tracks the panel exactly; once the cap is hit
        # the bar rotates within the jaws, which a parallel gripper on a round lever tolerates.
        wrist_follow = pull_swing_sign * min(th, pull_wrist_swing_cap)
        # Wrist: swing yaw for the door, plus the unwinding lever roll about the door normal.
        pull_rot = _turn_about_door_normal(
            _front_grasp_rot(FRONT_APPROACH_YAW + wrist_follow, device),
            unlatch_roll_per_angle * lever_angle,
            device,
        )
        pull_rot = _closest_jaw_equivalent(pull_rot, prev_grip_rot, device)
        prev_grip_rot = pull_rot
        palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, pull_rot, device)

        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # loop body: single seed for continuity (no random-restart branch jumps)
        )[0]
        _set_gripper(q_robot, GRASP_WIDTH)

        # Pin the sweep down with keyframes every pull_keyframe_every frames instead of leaving
        # the whole pull as ONE spline segment. collocate_and_playback splines between KEYFRAMES
        # with bc_type="clamped"; over a 51-knot segment it bulges badly at the far end.
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=(pull_step > 0 and pull_step % pull_keyframe_every == 0),
        )

    if (len(theta_values) - 1) % pull_keyframe_every != 0:
        key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 5: Move to the blocking base pose while releasing the hinge
    # -------------------------
    _, _, robot_initial_yaw = euler_xyz_from_quat(base_target_rot)

    release_base_x_delta_1 = -0.12
    # Blocking pose Y: -y approaches the open leaf (left), +y backs away (right).
    release_base_y = 0.10
    release_tcp_x_delta = 0.25
    release_tcp_y_delta = -0.1
    release_base_x_delta_2 = -0.10
    release_door_open_angle = pull_theta_stop + 0.10

    # Turn the base toward the opened panel on +y before the blocking phase.
    tilt_base_yaw = blocking_turn_sign * 1.0
    tilted_base_rot = get_rotation_quat(
        0.0,
        0.0,
        robot_initial_yaw.item() + tilt_base_yaw,
        device,
    )
    # Panel ends up on the robot's +y side, so the contact face must point +y. yaw=0 with
    # sign=-1 presses toward +y using the hand's -y finger face -- the gripper's LEFT side as
    # seen on the robot, since the blocking attitude rolls the hand pi to point the fingers down.
    # (yaw=pi with sign=+1 presses the same way but leads with the other face, which reads as
    # blocking with the gripper's right side.)
    block_face_yaw = 0.0
    block_face_sign = -1.0
    # Both jaw-equivalents press the panel in the SAME direction with opposite finger faces, so
    # take whichever is nearer the attitude the wrist already holds. Without this the wrist spins
    # up to 180 deg on its way into the block for no reason. block_face_sign flips with it so the
    # contact face stays on the side that meets the panel.
    _blk = _side_block_rot(block_face_yaw, device)
    block_rot = _closest_jaw_equivalent(_blk, prev_grip_rot, device)
    if not bool(torch.allclose(block_rot, _blk)):
        block_face_sign = -block_face_sign
    # Where the panel actually IS when the arm goes to block it. Must follow the door angle at
    # release -- pinned at a constant it silently assumed a long pull, so shortening the pull left
    # the arm reaching for a panel that had not swung that far and driving into it instead.
    contact_virtual_door_angle = release_door_open_angle
    push_door_open_angle = 1.5

    # LET GO FIRST, in place, before the base moves anywhere. Previously the base tilted a full
    # radian and translated with the jaws still shut on the bar; the playback spline then dragged
    # the gripper 0.2-0.56 m off the handle across ~37 frames while nominally still gripping it.
    # Releasing first turns that whole segment into a free-space move.
    _set_gripper(q_robot, OPEN_WIDTH)
    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    base_target_pos[:, 0] += release_base_x_delta_1
    base_target_pos[:, 1] = release_base_y
    base_target_pose = _make_pose(base_target_pos, tilted_base_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=False,
    )

    # Back the gripper off the handle and OPEN it -- the handle is released here.
    release_tcp_pos = handle_pos.clone()
    release_tcp_pos[:, 0] += release_tcp_x_delta
    release_tcp_pos[:, 1] += release_tcp_y_delta
    palm_target_pose = _hand_pose_from_tcp(release_tcp_pos, grasp_rot, device)

    base_target_pos[:, 0] += release_base_x_delta_2
    base_target_pos[:, 1] = release_base_y
    base_target_pose = _make_pose(base_target_pos, tilted_base_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    _set_gripper(q_robot, OPEN_WIDTH)
    q_door = torch.tensor([release_door_open_angle, 0.0], device=device)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 6: Retract the arm, already turned to the side-block attitude
    # -------------------------
    # Retract offset applied DIRECTLY in world frame: +x pulls the hand BACKWARD off the door,
    # +y nudges it to the RIGHT to clear the panel. Lifting the hand high matters -- the arm
    # swings a wide arc from here around to the panel, and doing that low sweeps it through the
    # arx camera arm on the base. The wrist is already in the SIDE-BLOCK attitude so the swing
    # into contact does not also have to flip the wrist.
    retreat_back_x = 0.30
    # Retract HIGH. 1.20 was barely 20 cm over the handle, so the swing round to the panel still
    # grazed the x5 camera arm. Absolute world height, well above both the handle and the x5.
    retreat_lift_z = 1.55

    retreat_tcp_pos = palm_target_pose[:, :3].clone()
    retreat_tcp_pos[:, 0] += retreat_back_x
    retreat_tcp_pos[:, 1] += retreat_side_y
    retreat_tcp_pos[:, 2] = retreat_lift_z

    # TWO stages, and the wrist does NOT turn during the first one. Translating and flipping the
    # wrist in the same move is what made the retract twist unnaturally; here the hand pulls back
    # and up holding the attitude it grasped with, and only re-orients to the side-block attitude
    # once it is clear. The LIFT is what keeps it off the x5 camera arm, which is mounted at
    # (-0.03, -0.08, 0.65) on the chassis -- behind and beside the franka, squarely in the path of
    # a low retract.
    for stage_rot in (grasp_rot, block_rot):
        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=_hand_pose_from_tcp(retreat_tcp_pos, stage_rot, device),
            base_pose=None,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # seeded from the previous pose so the arm adjusts, not re-solves
        )[0]
        _set_gripper(q_robot, OPEN_WIDTH)
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=True,
        )

    # -------------------------
    # Step 7: Block/push the panel with the SIDE of the gripper, base held still
    # -------------------------
    # Contact anchored at the panel CENTER (get_board_pos) then shifted toward the FREE/OUTER
    # edge for a longer lever arm (lighter push force). The TCP is then stood off along the
    # contact face normal by half the gripper's outer width, so what actually lands on the panel
    # is a finger's flat SIDE, not the fingertips.
    block_contact_x_offset = 0.13
    block_contact_y_offset = 0.25
    block_contact_z_offset = 0.22
    block_face_standoff = GRIPPER_SIDE_HALF_WIDTH
    # Extra clearance for the non-key approach frame, so the gripper arrives beside the panel
    # and then presses in rather than driving through it.
    # Descending clearances along the face normal. The first is far enough out that the swing
    # from the retract pose stays clear of the leaf entirely.
    block_approach_clearances = (0.45, 0.30, 0.18, 0.10)

    face_normal = _side_face_normal(block_rot, block_face_sign, device, handle_pos.dtype)

    contact_board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        torch.tensor([contact_virtual_door_angle, 0.0], device=device).unsqueeze(0),
    ).to(device)

    contact_anchor = contact_board_pos.clone()
    contact_anchor[:, 0] += block_contact_x_offset
    contact_anchor[:, 1] += block_contact_y_offset
    contact_anchor[:, 2] += block_contact_z_offset

    # --- Staged approach: come in ALONG the face normal, from well clear of the panel ---
    # A single approach frame 0.10 m off the face left the spline free to cut the corner between
    # the retract pose and the panel, driving the gripper straight THROUGH the door on the way in.
    # Walking down the normal in stages keeps every intermediate frame on the robot's side of the
    # panel, so the arm reaches around the leaf instead of penetrating it.
    for clearance in block_approach_clearances:
        tcp_target_pos = contact_anchor - (block_face_standoff + clearance) * face_normal
        palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, block_rot, device)

        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=None,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # seeded from the previous stage, so the arm walks in smoothly
        )[0]
        _set_gripper(q_robot, OPEN_WIDTH)

        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            # keyframes: the spline must pass through each stage, not shortcut across them
            mark_keyframe=True,
        )

    # --- Side-face contact (keyframe) ---
    tcp_target_pos = contact_anchor - block_face_standoff * face_normal
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, block_rot, device)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # --- Drive the panel to full open, side face still on it ---
    q_door = torch.tensor([push_door_open_angle, 0.0], device=device)
    board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    open_anchor = board_pos.clone()
    open_anchor[:, 0] += block_contact_x_offset
    open_anchor[:, 2] += block_contact_z_offset
    tcp_target_pos = open_anchor - block_face_standoff * face_normal
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, block_rot, device)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 8: Restore the base to normal yaw, traverse with a suitable arm pose,
    # then finish the traverse with default arm joints
    # -------------------------
    # Traverse forward THROUGH the doorway with a smooth, interpolated base sweep while the arm
    # keeps HOLDING the door panel (TCP pinned), so the arm does not jerk away from the panel.
    # The tilted blocking yaw is restored to normal over the same sweep.
    traverse_mid_x = 0.0
    traverse_mid_y = 0.10
    # Drive well past the doorway so the closing door panel can't smash the base from behind.
    traverse_far_x = -1.0
    traverse_steps = 8

    start_base_x = base_target_pos[:, 0].clone()
    start_base_y = base_target_pos[:, 1].clone()
    blocking_yaw = robot_initial_yaw.item() + tilt_base_yaw
    for traverse_step in range(1, traverse_steps + 1):
        frac = traverse_step / traverse_steps
        base_target_pos = base_target_pos.clone()
        base_target_pos[:, 0] = start_base_x + frac * (traverse_mid_x - start_base_x)
        base_target_pos[:, 1] = start_base_y + frac * (traverse_mid_y - start_base_y)
        step_yaw = blocking_yaw + frac * (robot_initial_yaw.item() - blocking_yaw)
        base_target_pose = _make_pose(
            base_target_pos, get_rotation_quat(0.0, 0.0, step_yaw, device)
        )
        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # loop body: single seed for continuity (no random-restart branch jumps)
        )[0]
        _set_gripper(q_robot, OPEN_WIDTH)
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=(traverse_step == traverse_steps),
        )

    base_target_pos[:, 0] = traverse_far_x
    base_target_pos[:, 1] = 0.0
    base_target_pose = _make_pose(base_target_pos, base_target_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=None,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[3:10] = franka_default_q
    _set_gripper(q_robot, OPEN_WIDTH)
    # Hold the door where the robot actually left it. This used to reset to [0, 0], teleporting
    # the panel from fully open to shut in a single frame while the robot stands a metre away --
    # visible as the door slamming at the end of playback, and meaningless as a tracking target
    # for a reference trajectory (nothing the robot does explains the panel moving).
    q_door = q_door.clone()

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    return robot_traj, door_traj, key_idx_in_key_indices


def state_machine_offline_left_pull_door(
    robot_urdf_path,
    door_urdf_path,
    robot_initial_pose,   # (1, 7) world
    door_initial_pose,    # (1, 7) world
    robot_initial_q,      # (ndof,)
    door_initial_q,       # (2,) [board, hinge]
    device="cpu",
):
    """
    Offline planner for a left-side handle, pull-type door.

    This is intentionally separate from the right-door function so all left-door
    tuning stays local and obvious.
    """
    q_robot, q_door, robot_traj, door_traj, key_idx_in_key_indices = _init_planner_state(
        robot_initial_q, door_initial_q
    )

    franka_default_q = torch.tensor(
        [FRANKA_DEFAULT_JOINT_POS[name] for name in FRANKA_JOINT_NAMES],
        device=device,
    )

    base_target_rot = robot_initial_pose[:, 3:].to(device).clone()

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 1: Pregrasp
    # -------------------------
    # BASE placement uses the handle-assembly centroid (cm precision is irrelevant for parking the
    # chassis), but every PALM target below aims at the lever bar itself -- see get_handle_bar_pos
    # for why the centroid is the wrong point to reach for.
    handle_assembly_pos = get_hinge_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)
    handle_pos = get_handle_bar_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    # Deeper standoff than the LEAP build's 0.67, matching the right-door planner: pulling from
    # further out runs the load down the extended arm instead of through the wrist.
    pregrasp_base_x_offset = 0.85
    pregrasp_base_y_offset = 0.25
    # Stand off along the TILTED approach direction, not along +x. approach_dir points from the
    # handle back toward where the hand starts: +x is "in front of the door" and +z is "above it",
    # so at APPROACH_TILT=60 deg the hand begins high and in front and descends into the bar along
    # its own approach axis. FRONT_GRASP_ROLL aims the wrist down the same line (verified: at
    # tilt 60 the hand's approach axis comes out as (-0.50, 0, -0.87) = -approach_dir).
    pregrasp_tcp_standoff = 0.25
    approach_dir = torch.tensor(
        [[math.cos(APPROACH_TILT), 0.0, math.sin(APPROACH_TILT)]], device=device
    )
    pregrasp_tcp_x_standoff = pregrasp_tcp_standoff * float(approach_dir[0, 0])
    pregrasp_tcp_y_offset = 0.0
    pregrasp_tcp_z_offset = pregrasp_tcp_standoff * float(approach_dir[0, 2])

    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += pregrasp_base_x_offset
    base_target_pos[:, 1] += pregrasp_base_y_offset
    # Left-door camera FOV: tilt the base a little toward the handle for pregrasp -> unlatch, so
    # the ARX/x5 camera arm keeps the handle in good view. base_target_rot (untilted) is restored
    # from Step 4 onward, so only pregrasp/grasp/unlatch are tilted.
    pregrasp_base_tilt_yaw = 0.3
    _, _, _base_yaw = euler_xyz_from_quat(base_target_rot)
    pregrasp_tilt_base_rot = get_rotation_quat(0.0, 0.0, _base_yaw.item() + pregrasp_base_tilt_yaw, device)
    base_target_pose = _make_pose(base_target_pos, pregrasp_tilt_base_rot)

    # Offsets shared with Step 2 (grasp), Step 3 (unlatch) and Step 4 (pull), defined here --
    # rather than at their own steps, as before -- because the grasp_rot SEARCH just below needs
    # to score candidates at those same three downstream anchors before any of them run. Step 2/3/4
    # below reference these by name instead of redefining them.
    grasp_tcp_x_offset = 0.0
    grasp_tcp_y_offset = 0.0
    grasp_tcp_z_offset = 0.0
    unlatch_hinge_angle = 1.05
    unlatch_tcp_z_delta = -0.06
    unlatch_tcp_y_delta = -0.02
    unlatch_roll_per_angle = 1.0
    pull_theta_stop = 1.25
    pull_base_x_offset = 0.45
    pull_base_y_offset = 0.25
    pull_base_backoff_gain = 0.25
    pull_wrist_swing_cap = 0.6
    pull_swing_sign = 1.0

    # Search grasp_rot the same way block_rot is searched later: the two candidates are the SAME
    # physical grasp (180 deg jaw flip about the gripper's own approach axis), but resolve to
    # different IK branches with different joint6/joint7 margins. The un-flipped candidate saturates
    # panda_joint6 near its hard limit through most of the pull -- which is why this used to always
    # jaw-flip -- but that criterion only ever looked at the PULL. _closest_jaw_equivalent pins the
    # whole pull AND the retract that follows release to whichever branch gets seeded here, so a
    # choice that is safe for the pull can still leave joint7 pinned near ITS limit for the retract,
    # which is exactly the contorted retract pose seen in rendered frames. Score both candidates at
    # three points spanning the whole chain this seeds -- grasp, unlatch, and the END of the pull
    # (board=pull_theta_stop, wrist tracking saturated at pull_wrist_swing_cap, which is also where
    # the retract's joint-space interpolation actually starts from) -- on the WORST joint6/joint7
    # margin against the same ranges _solve_ik_avoiding_wrist_limits guards, and keep whichever
    # candidate's worst point is least bad.
    _pull_end_handle_pos = get_handle_bar_pos(
        door_urdf_path,
        door_initial_pose,
        torch.tensor([[pull_theta_stop, 0.0]], device=device),
    ).to(device)
    _pull_end_base_pos = _pull_end_handle_pos.clone()
    _pull_end_base_pos[:, 0] += pull_base_x_offset + pull_base_backoff_gain * pull_theta_stop
    _pull_end_base_pos[:, 1] += pull_base_y_offset
    _pull_end_base_pose = _make_pose(
        _pull_end_base_pos, get_rotation_quat(0.0, 0.0, _base_yaw.item(), device)
    )
    _wrist_follow_end = pull_swing_sign * min(pull_theta_stop, pull_wrist_swing_cap)

    _grasp_rot_candidates = (
        _front_grasp_rot(FRONT_APPROACH_YAW, device),
        _jaw_flip(_front_grasp_rot(FRONT_APPROACH_YAW, device), device),
    )
    _best_grasp = None
    for _cand_rot in _grasp_rot_candidates:
        _cand_unlatch_rot = _turn_about_door_normal(
            _cand_rot, unlatch_roll_per_angle * unlatch_hinge_angle, device
        )
        _cand_pull_end_rot = _closest_jaw_equivalent(
            _front_grasp_rot(FRONT_APPROACH_YAW + _wrist_follow_end, device), _cand_rot, device
        )
        _anchors = (
            (handle_pos, base_target_pose, _cand_rot),
            (handle_pos + torch.tensor([[0.0, unlatch_tcp_y_delta, unlatch_tcp_z_delta]], device=device),
             base_target_pose, _cand_unlatch_rot),
            (_pull_end_handle_pos, _pull_end_base_pose, _cand_pull_end_rot),
        )
        _margin = float("inf")
        for _anchor_tcp, _anchor_base, _anchor_rot in _anchors:
            _cand_q = solve_ik(
                robot_urdf_path,
                q_robot[:10],
                palm_pose=_hand_pose_from_tcp(_anchor_tcp, _anchor_rot, device),
                base_pose=_anchor_base,
                robot_initial_pose=robot_initial_pose,
                num_attempts=4,
            )[0]
            _margin = min(_margin, min(
                min(float(_cand_q[_idx]) - _lo, _hi - float(_cand_q[_idx]))
                for _idx, _lo, _hi in (
                    (_PANDA_JOINT6_IDX, *_PANDA_JOINT6_SOFT_RANGE),
                    (_PANDA_JOINT7_IDX, *_PANDA_JOINT7_WORKING_RANGE),
                )
            ))
        if _best_grasp is None or _margin > _best_grasp[0]:
            _best_grasp = (_margin, _cand_rot)
    grasp_rot = _best_grasp[1]

    tcp_target_pos = handle_pos.clone()
    tcp_target_pos[:, 0] += pregrasp_tcp_x_standoff
    tcp_target_pos[:, 1] += pregrasp_tcp_y_offset
    tcp_target_pos[:, 2] += pregrasp_tcp_z_offset
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, grasp_rot, device)

    # This is where the arm's redundant elbow/null-space branch for the rest of the trajectory
    # effectively gets decided (see _solve_ik_avoiding_wrist_limits's docstring) -- pick the
    # re-roll that keeps panda_joint6 safest rather than whichever one solve_ik's internal random
    # restarts happen to return first.
    q_robot[:10] = _solve_ik_avoiding_wrist_limits(
        robot_urdf_path,
        q_robot,
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        device=device,
        guarded_joints=_GRASP_PULL_GUARDED_JOINTS,
    )
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 2: Move to grasp
    # -------------------------
    # Aim the TCP AT THE BAR, with no nudges.
    #
    # These offsets claimed to seat the bar between the jaws. Measured, they did the opposite: at
    # every gripping frame the lever was 4.6-8.9 cm from the TCP and the jaws closed on the
    # escutcheon plate instead -- confirmed both in mesh renders (the bar hangs visibly free beside
    # the closed gripper) and numerically (a bar-vs-jaw span test scores ZERO frames with the bar
    # between the fingers, and TCP-to-plate is consistently SHORTER than TCP-to-bar).
    #
    # The reason this was never caught is that every distance check measured TCP to the link_2
    # CENTROID, and link_2 is escutcheon + lever + hook with the plate dominating -- so sitting on
    # the plate scores as "on the handle". get_hinge_pos already lands within ~2 cm of the bar; it
    # was these offsets, ~4.6 cm combined, that walked the gripper off it.
    # (grasp_tcp_x/y/z_offset are defined above, before Step 1's grasp_rot search -- that search
    # needs them to build this step's own anchor target, so they moved up rather than being
    # duplicated here.)

    # Walk in along the approach in small steps rather than jumping the whole standoff at once.
    # A single ~28 cm move is enough for the solver to come back on a different branch, which is
    # what makes the arm look like it re-solved from scratch instead of adjusting from where it
    # was. These are non-keyframes: they only shape the path in, they are not poses to hold.
    approach_steps = 4
    for approach_step in range(1, approach_steps + 1):
        frac = approach_step / (approach_steps + 1)
        step_tcp = handle_pos.clone()
        step_tcp[:, 0] += pregrasp_tcp_x_standoff + (grasp_tcp_x_offset - pregrasp_tcp_x_standoff) * frac
        step_tcp[:, 1] += pregrasp_tcp_y_offset + (grasp_tcp_y_offset - pregrasp_tcp_y_offset) * frac
        step_tcp[:, 2] += pregrasp_tcp_z_offset + (grasp_tcp_z_offset - pregrasp_tcp_z_offset) * frac
        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=_hand_pose_from_tcp(step_tcp, grasp_rot, device),
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # seeded from the previous frame, so the arm adjusts rather than re-solves
        )[0]
        _set_gripper(q_robot, OPEN_WIDTH)
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=False,
        )

    tcp_target_pos = handle_pos.clone()
    tcp_target_pos[:, 0] += grasp_tcp_x_offset
    tcp_target_pos[:, 1] += grasp_tcp_y_offset
    tcp_target_pos[:, 2] += grasp_tcp_z_offset
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, grasp_rot, device)

    # Same joint6-safety re-roll as pregrasp -- this call also uses solve_ik's default
    # (randomized) num_attempts, so it can independently drift onto a bad elbow branch even
    # though pregrasp already landed on a safe one.
    q_robot[:10] = _solve_ik_avoiding_wrist_limits(
        robot_urdf_path,
        q_robot,
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        device=device,
        guarded_joints=_GRASP_PULL_GUARDED_JOINTS,
    )
    # Arrive at the bar with the jaws STILL OPEN. Closing is a separate waypoint below at the
    # SAME arm pose, so the spline finishes the reach before the fingers start to move --
    # otherwise the gripper closes across the whole approach and swipes the handle aside on its
    # way in.
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # Do NOT close the jaws on the bar -- this is a HOOK grasp: the jaws stay OPEN and pull by
    # catching the underside of the lever, the way a hand hooks a finger under a door handle
    # instead of squeezing it shut. Closing to GRASP_WIDTH (0.0) would clamp fully shut on a
    # ~2.1 cm round bar, which is not the grasp this is meant to be. Kept as its own keyframe
    # (arm/base untouched) so the spline still has a waypoint here; only the gripper number no
    # longer changes.
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 3: Rotate hinge (unlatch)
    # -------------------------
    # unlatch_hinge_angle / unlatch_tcp_z_delta / unlatch_tcp_y_delta / unlatch_roll_per_angle are
    # defined above, before Step 1's grasp_rot search (same reason as Step 2's grasp offsets).

    q_door = torch.tensor([0.0, unlatch_hinge_angle], device=device)

    unlatch_rot = _turn_about_door_normal(
        grasp_rot,
        unlatch_roll_per_angle * unlatch_hinge_angle,
        device,
    )

    tcp_target_pos = handle_pos.clone()
    tcp_target_pos[:, 0] += grasp_tcp_x_offset
    tcp_target_pos[:, 1] += grasp_tcp_y_offset + unlatch_tcp_y_delta
    tcp_target_pos[:, 2] += grasp_tcp_z_offset + unlatch_tcp_z_delta
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, unlatch_rot, device)

    # Same joint6-safety re-roll -- unlatch is the last of the three default-num_attempts calls
    # before the pull loop's num_attempts=1 continuity takes over, so this is the last chance to
    # correct onto a safe branch before it gets locked in for the rest of the trajectory.
    q_robot[:10] = _solve_ik_avoiding_wrist_limits(
        robot_urdf_path,
        q_robot,
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        device=device,
        guarded_joints=_GRASP_PULL_GUARDED_JOINTS,
    )
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 4: Pull door open
    # -------------------------
    # Starts at theta=0 so the first frame reproduces the unlatch keyframe (see the right-door
    # planner for why the old 0.3 start made the arm thrash), with a finer step for more
    # waypoints and therefore smaller moves between IK seeds.
    pull_theta_start = 0.0
    # How far the gripper drags the door before letting go. Shortening this is tempting -- the
    # gripper stays much better aligned to the handle (14 deg vs 29 deg at 0.60) -- but MEASURED
    # over 5 doors it is far worse overall, because the blocking phase then has to reach around a
    # panel that has not swung clear:
    #     stop 1.25 rad (72 deg): panel collisions  23/200 frames
    #     stop 0.60 rad (34 deg): panel collisions 131/200 frames
    # The long pull is what buys the arm room to get around the leaf. Do not shorten it without
    # rebuilding the blocking approach for a less-open door.
    # (pull_theta_stop, pull_base_x_offset, pull_base_y_offset, pull_base_backoff_gain,
    # pull_wrist_swing_cap and pull_swing_sign are all defined above, before Step 1's grasp_rot
    # search -- that search needs them to build the end-of-pull anchor. Comments on each stayed
    # here, next to where they are actually used.)
    pull_theta_step = 0.025

    # See the right-door planner: standing further back buys elbow room here, it does not cost
    # reach (the arm is mounted on the far side of the chassis from the door).
    # Lateral stance held relative to the handle; see the right-door planner.
    # Door travel over which the lever unwinds and the base eases into its pull pose.
    pull_ease_theta = 0.35
    prev_grip_rot = grasp_rot   # pins the jaw branch so the gripper cannot flip mid-pull

    # SINGLE knob for the door swing, driving both the offset rotation and the wrist yaw.
    # Positive = counterclockwise seen from above, how a left-hand door swings.
    # Base absorbs part of the swing so the wrist is not driven to its limit. 0.0 restores a
    # base held at fixed yaw. TESTING 1.0 (matching the right-door planner) -- with the base
    # fixed, 100% of the door's yaw tracking falls on the wrist alone, which is exactly what
    # keeps saturating panda_joint6. See the debug session's change log for the measurement.
    pull_base_yaw_gain = 0.0
    # ...and BACKS OFF as it turns. Yawing toward the door without retreating swings the
    # chassis corner into the panel's path. Metres of extra standoff per radian.
    # WHICH WAY the base turns. This must match the blocking tilt in step 5, otherwise the base
    # yaws one way through the whole pull and then spins back the other way to block -- a 2.25 rad
    # reversal that reads as the base rotating the wrong way. Driven off pull_swing_sign before,
    # Mirror of the right door: must match THIS planner's blocking tilt (-1.0).
    # which is the DOOR's rotation and happens to be opposite.
    blocking_turn_sign = -1.0
    # How much of the door's swing the WRIST tracks. Kinematically this "should" be 1.0 -- the
    # gripper is clamped to the bar, so it ought to rotate exactly with the panel. Measured, the
    # arm cannot do it: scripts/debug/dump_pull_geometry.py shows mean TCP-to-handle error of
    # 0.099 m at gain 1.0 versus 0.045 m at 0.2-0.3, and jaw-off-vertical 38 deg versus 14 deg.
    # Past ~0.4 the IK gives up on the orientation and throws position away with it. The gap is
    # taken up by the bar rotating within the jaws, which a parallel gripper on a round lever
    # tolerates. Raise it only if the arm gains reach or the pull range shrinks.
    # TESTING: keep the FRONT approach (not switching to top/top-front), but lift the TCP target
    # slightly as the wrist tracks the swing, so the elbow/shoulder take up some of the reach
    # geometry instead of leaving it all to the wrist. Ramps on the SAME schedule as wrist_follow
    # (0 at th=0, capped at pull_wrist_swing_cap) so it only grows while the wrist is actively
    # working, and holds once the wrist itself saturates. Metres of lift per radian of tracked
    # swing.
    #
    # TESTED 0.15: WORSE, not better -- 10-trial panda_joint6 sweep went from 6/10 clean trials
    # (1/10 with a hard-limit violation) at gain=0.0 to 4/10 clean (4/10 with a hard-limit
    # violation) at 0.15. Also tested pull_base_yaw_gain=1.0 (restoring base rotation to share
    # the door's yaw with the wrist): also worse, 0/5 clean trials became... worse than the
    # gain=0.0 baseline's 0/5-violation run. Both point the same way: the strain is not fixable
    # by moving the SAME front-approach geometry around -- see the debug session's change log for
    # the "is front-approach solvable" writeup. Left at 0.0 (disabled).
    pull_lift_gain = 0.0
    pull_keyframe_every = 2

    _, _, pull_base_yaw0 = euler_xyz_from_quat(base_target_rot)
    grasp_base_x = base_target_pos[:, 0].clone()
    grasp_base_y = base_target_pos[:, 1].clone()

    theta_values = torch.arange(
        pull_theta_start,
        pull_theta_stop + 1e-6,
        pull_theta_step,
        device=device,
    )
    for pull_step, theta in enumerate(theta_values):
        th = theta.item()
        ease = min(1.0, th / pull_ease_theta) if pull_ease_theta > 0 else 1.0
        lever_frac = 1.0 - ease
        lever_angle = unlatch_hinge_angle * lever_frac
        q_door = torch.tensor([th, lever_angle], device=device)

        handle_pos = get_hinge_pos(
            door_urdf_path,
            door_initial_pose,
            q_door.unsqueeze(0),
        ).to(device)

        # Stance is held in WORLD axes, deliberately NOT rotated with the swing: a pull door
        # comes AT the robot, so the base has to back away along -x and hold its side, not orbit
        # the handle. Rotating the stance was measured worse (0.073 m vs 0.046 m mean) and would
        # walk the base through the doorway/wall at large angles.
        swing = pull_swing_sign * theta
        base_target_pos = handle_pos.clone()
        base_target_pos[:, 0] = grasp_base_x + (
            handle_pos[:, 0] + pull_base_x_offset + pull_base_backoff_gain * th - grasp_base_x
        ) * ease
        base_target_pos[:, 1] = grasp_base_y + (
            handle_pos[:, 1] + pull_base_y_offset - grasp_base_y
        ) * ease
        base_target_pose = _make_pose(
            base_target_pos,
            get_rotation_quat(
                0.0,
                0.0,
                pull_base_yaw0.item() + blocking_turn_sign * pull_base_yaw_gain * th,
                device,
            ),
        )

        # EXACTLY the grasp offset, carried rigidly with the panel, with the unlatch deltas
        # fading out on the lever's schedule (see the right-door planner).
        tcp_dx, tcp_dy = _rotate_xy_about_z(
            grasp_tcp_x_offset,
            grasp_tcp_y_offset + unlatch_tcp_y_delta * lever_frac,
            swing,
        )

        tcp_target_pos = handle_pos.clone()
        tcp_target_pos[:, 0] += tcp_dx
        tcp_target_pos[:, 1] += tcp_dy
        tcp_target_pos[:, 2] += (
            grasp_tcp_z_offset + unlatch_tcp_z_delta * lever_frac
            + pull_lift_gain * min(th, pull_wrist_swing_cap)
        )

        # Wrist follows the door 1:1 up to a CAP, then saturates -- rather than a fractional gain
        # that is wrong everywhere. Early on, while the grasp is being established and the jaws
        # must stay square to the bar, the gripper tracks the panel exactly; once the cap is hit
        # the bar rotates within the jaws, which a parallel gripper on a round lever tolerates.
        wrist_follow = pull_swing_sign * min(th, pull_wrist_swing_cap)
        pull_rot = _turn_about_door_normal(
            _front_grasp_rot(FRONT_APPROACH_YAW + wrist_follow, device),
            unlatch_roll_per_angle * lever_angle,
            device,
        )
        pull_rot = _closest_jaw_equivalent(pull_rot, prev_grip_rot, device)
        prev_grip_rot = pull_rot
        palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, pull_rot, device)

        # Plain single-seed continuity solve first (unchanged from before -- no random-restart
        # branch jumps in the common case); only if THAT drifts panda_joint6 unsafe does this
        # spend the extra re-rolls to correct it. Traced frames where the continuity path alone
        # pushes joint6 over its soft limit mid-pull even from an already-safe unlatch start (the
        # pregrasp/grasp/unlatch fix alone does not cover this -- see the debug session's change
        # log), so the same safety net is needed here too.
        q_robot[:10] = _solve_ik_avoiding_wrist_limits(
            robot_urdf_path,
            q_robot,
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            device=device,
            first_num_attempts=1,
            guarded_joints=_GRASP_PULL_GUARDED_JOINTS,
        )
        _set_gripper(q_robot, OPEN_WIDTH)

        # Pin the sweep down with keyframes every pull_keyframe_every frames instead of leaving
        # the whole pull as ONE spline segment. collocate_and_playback splines between KEYFRAMES
        # with bc_type="clamped"; over a 51-knot segment it bulges badly at the far end.
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=(pull_step > 0 and pull_step % pull_keyframe_every == 0),
        )

    if (len(theta_values) - 1) % pull_keyframe_every != 0:
        key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 5: Move to the blocking base pose while releasing the hinge
    # -------------------------
    _, _, robot_initial_yaw = euler_xyz_from_quat(base_target_rot)

    release_base_x_delta_1 = -0.12
    # Blocking pose Y: +y approaches the open leaf, -y backs away. Kept a margin off the leaf.
    #
    # Measured this door's geometry directly (fit a circle through get_board_edge at
    # board=0/1.25/1.35/1.50): hinge at world (-0.03, 0.44), leaf radius 0.78 m. At board=1.25 the
    # leaf occupies y in [0.18, 0.44]; by board=1.35 (where the block/push phase runs) that band is
    # [0.25, 0.44]. -0.20 sits ~0.4 m clear of that band on the latch-jamb side, comfortably inside
    # the frame opening (jambs at roughly y=-0.34 and y=+0.44 when closed) without walking the
    # chassis into either post.
    release_base_y = -0.20
    release_tcp_x_delta = 0.3
    release_tcp_y_delta = 0.0
    # Total x-travel of this step (delta_1 + delta_2) lands the chassis at x~1.2, still ~0.8 m short
    # of the door frame (frame plane at x~0, per the same geometry probe) -- but this step's base
    # move happens with the PALM STILL PINNED to the (stationary) handle at x~0.73, and TRIED
    # deepening this delta to reach x~0.35-0.4 directly here: solve_ik's returned error against the
    # pinned target grows monotonically over the 4 sub-steps as the base pulls away from the handle
    # under yaw, from ~0.2 at step 1 to ~0.67 by step 4 -- the arm is not tracking the handle by the
    # end, it is failing quietly and returning its best-effort contorted guess, reproducing exactly
    # the "dragged the gripper 0.2-0.56 m off the bar" failure this step's staging was built to
    # avoid. Reverted to the original, validated delta. The base's remaining walk to the doorway
    # threshold happens in a NEW step below, AFTER Step 6's retract -- once the arm is parked in
    # JOINT space at franka_default_q and the jaws are open, moving the base under it is no longer
    # an IK problem at all (nothing is being tracked), so it can travel as far as needed for free.
    release_base_x_delta_2 = -0.18
    release_door_open_angle = pull_theta_stop + 0.10

    # Negative relative yaw turns the base toward the opened panel on -y.
    tilt_base_yaw = blocking_turn_sign * 1.0
    tilted_base_rot = get_rotation_quat(
        0.0,
        0.0,
        robot_initial_yaw.item() + tilt_base_yaw,
        device,
    )
    # Panel ends up on the robot's -y side, so the contact face must point -y. Mirror of the
    # right door: same sign=-1 (block with the gripper's LEFT side), yaw rotated by pi to aim
    # that face at the opposite panel. Only the yaw differs between the two handle sides.
    block_face_yaw = math.pi
    block_face_sign = -1.0
    # Halved from -0.30 (measured: this door has a self-closing spring -- release_open to
    # block_contact was consuming ~200 of 400 total playback frames, more than half the whole
    # trajectory, almost entirely as ARM travel with the base already parked and idle. Shortening
    # the retract's side excursion is the most direct lever on that without touching the base
    # (which is already positioned by the end of Step 5) or retreat_lift_z (kept at 1.55 -- the
    # x5-camera-arm clearance height, not touched, see the comment on retreat_lift_z below).
    retreat_side_y = -0.15     # mirror: clear the panel on the -y side
    # Both jaw-equivalents press the panel in the SAME direction with opposite finger faces, so
    # take whichever is nearer the attitude the wrist already holds. Without this the wrist spins
    # up to 180 deg on its way into the block for no reason. block_face_sign flips with it so the
    # contact face stays on the side that meets the panel.
    _blk = _side_block_rot(block_face_yaw, device)
    block_rot = _closest_jaw_equivalent(_blk, prev_grip_rot, device)
    if not bool(torch.allclose(block_rot, _blk)):
        block_face_sign = -block_face_sign
    retreat_local_x = -0.10
    retreat_local_y = -0.42
    # TRIED matching the right-door planner's retreat height (raising this to an absolute
    # z=1.2, "lifting the hand high... so the arc doesn't sweep through the arx camera arm")
    # instead of this small RELATIVE bump above palm_target_pose. Measured with
    # scripts/debug/check_pull_collisions.py over 4 doors: panel-hit frames went from 131 to 153
    # (worse, not better) -- lifting this high apparently swings the forearm INTO the fully-open
    # panel instead of over it, unlike the right door's geometry. Reverted; the remaining
    # panda_link6-vs-panel contact through board 1.0-1.5 rad (worst ~-0.05 to -0.10 m) during
    # retreat/block-approach is UNRESOLVED -- see the debug session's change log.
    retreat_z_lift = 0.04
    block_contact_x_offset = 0.10
    block_contact_y_offset = -0.2
    block_contact_z_offset = 0.1
    # Follows the door angle at release; see the right-door planner.
    contact_virtual_door_angle = release_door_open_angle
    push_door_open_angle = 1.5

    # BLOCK WITH THE BASE FIRST, THEN LET GO.
    #
    # This used to release in place and only then drive the base over. That leaves the door -- which
    # has a self-closing spring -- unheld for the whole base move, free to swing back into the robot
    # before anything is in its way. Blocking first means the chassis is already across the leaf's
    # path at the moment the jaws open.
    #
    # The original reason for releasing first still has to be respected: moving the base with the
    # jaws shut once dragged the gripper 0.2-0.56 m off the bar, because ONE waypoint spanned the
    # whole tilt+translate and the spline interpolated base and arm independently between its ends.
    # The fix is not to give up on holding the handle, it is to stop asking the spline to guess --
    # walk the base over in several keyframed steps with the palm pinned to the (stationary) handle,
    # so the arm visibly gives way to the base instead of the TCP cutting a chord off the bar.
    base_block_steps = 4
    block_base_x = base_target_pos[:, 0] + release_base_x_delta_1 + release_base_x_delta_2
    start_base_bx = base_target_pos[:, 0].clone()
    start_base_by = base_target_pos[:, 1].clone()
    for block_step in range(1, base_block_steps + 1):
        frac = block_step / base_block_steps
        base_target_pos = base_target_pos.clone()
        base_target_pos[:, 0] = start_base_bx + frac * (block_base_x - start_base_bx)
        base_target_pos[:, 1] = start_base_by + frac * (release_base_y - start_base_by)
        base_target_pose = _make_pose(
            base_target_pos,
            get_rotation_quat(0.0, 0.0, robot_initial_yaw.item() + frac * tilt_base_yaw, device),
        )
        q_robot[:10] = _solve_ik_avoiding_wrist_limits(
            robot_urdf_path,
            q_robot,
            palm_pose=palm_target_pose,      # pinned: the handle does not move while it is held
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            device=device,
            first_num_attempts=1,            # continuity: the arm adjusts, it does not re-solve
            guarded_joints=_GRASP_PULL_GUARDED_JOINTS,
        )
        _set_gripper(q_robot, OPEN_WIDTH)  # HOOK grasp (never closes) + STILL tracking the bar through the whole base move
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=True,
        )

    # Base is across the leaf now -- safe to let go, in place.
    _set_gripper(q_robot, OPEN_WIDTH)
    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )


    # Back the JAWS off the bar, in free space. The base is ALREADY at its blocking pose from the
    # loop above (both release_base_x_delta_1 and _2 are applied there), so it does not move again
    # here -- only the arm withdraws. Holding the attitude the wrist currently has, rather than
    # snapping back to grasp_rot, for the same reason the retract does.
    release_tcp_pos = handle_pos.clone()
    release_tcp_pos[:, 0] += release_tcp_x_delta
    release_tcp_pos[:, 1] += release_tcp_y_delta
    palm_target_pose = _hand_pose_from_tcp(release_tcp_pos, prev_grip_rot, device)

    base_target_pose = _make_pose(base_target_pos, tilted_base_rot)

    q_robot[:10] = _solve_ik_avoiding_wrist_limits(
        robot_urdf_path,
        q_robot,
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        device=device,
        first_num_attempts=1,
        guarded_joints=_GRASP_PULL_GUARDED_JOINTS,
    )

    _set_gripper(q_robot, OPEN_WIDTH)
    q_door = torch.tensor([release_door_open_angle, 0.0], device=device)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 6: Retract the arm, already turned to the side-block attitude
    # -------------------------
    # The retract is a JOINT-SPACE move to a known-good arm configuration, not a Cartesian target.
    #
    # Authoring it as a TCP pose was the mistake. Every version of that -- back 0.15 + lift to an
    # absolute 1.55 (which is +0.47..+0.60 m UP against 0.15 m back: a climb, not a withdrawal,
    # leaving the forearm swept across the leaf and the arm near its reach limit), back 0.40 with a
    # relative lift (panel hits 45 -> 70), back 0.40 with the old lift (65) -- just moves the
    # contortion around, because a pose target lets IK pick any configuration that reaches it,
    # including a terrible one. And on a PULL door the leaf swings toward the robot, so there is no
    # fixed world direction that reliably means "away from the panel" anyway.
    #
    # The arm does not need to be anywhere in particular here. It needs to be OUT OF THE WAY in a
    # posture it can hold. So command the configuration directly: the Franka home pose
    # [0, -0.785, 0, -2.356, 0, 1.571, 0] is compact, folded back over the chassis, and sits
    # mid-range on every joint (joint6 at 1.571 of 0..3.82, joint7 exactly centred) -- by
    # construction it cannot be the contorted pose, and it is the same anchor the null-space IK
    # bias already pulls toward.
    retreat_arm_q = franka_default_q
    retreat_steps = 2

    # Block-phase geometry, computed here (before the retract) because the BRIDGE waypoint below
    # needs the first block-approach target as an endpoint. None of this depends on the retract.
    block_face_standoff = GRIPPER_SIDE_HALF_WIDTH
    # Descending clearances along the face normal. The first is far enough out that the swing
    # from the retract pose stays clear of the leaf entirely.
    block_approach_clearances = (0.45, 0.30, 0.18, 0.10)

    face_normal = _side_face_normal(block_rot, block_face_sign, device, handle_pos.dtype)

    contact_board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        torch.tensor([contact_virtual_door_angle, 0.0], device=device).unsqueeze(0),
    ).to(device)

    contact_anchor = contact_board_pos.clone()
    contact_anchor[:, 0] += block_contact_x_offset
    contact_anchor[:, 1] += block_contact_y_offset
    contact_anchor[:, 2] += block_contact_z_offset

    # Pick the blocking attitude by ARM QUALITY, over every attitude that does the same job.
    #
    # There is no single correct way to block the leaf. The flat side face can be presented with
    # the fingers pointing DOWN (roll=pi) or UP (roll=0), and either jaw can be the one that meets
    # the panel (the jaw flip, with block_face_sign flipping to match). All four press the panel in
    # the same direction with the same flat face; they differ only in how the arm has to be folded
    # to get there. Committing to one a priori -- as _closest_jaw_equivalent did, choosing purely
    # on nearness to the attitude the wrist already held -- is what parked panda_joint7 ON its
    # +-2.9671 limit for the whole block phase, from which the only later escape was unwinding to
    # the opposite limit (measured: +2.967 -> -2.649, rendered as a pointless ~320 deg revolution).
    #
    # Seed/null-space search cannot rescue that: joint7 is fixed by the target orientation, not
    # redundant, so _solve_ik_avoiding_wrist_limits just re-rolls without moving it. Changing the
    # ATTITUDE is the only lever, so try them all and keep the one whose wrist sits furthest inside
    # its limits, requiring only that the contact face still points the way we need to press.
    _required_press = _side_face_normal(block_rot, block_face_sign, device, handle_pos.dtype)
    _best_blk = None
    for _cand_roll in (math.pi, 0.0):
        _base_rot = _side_block_rot(block_face_yaw, device, roll=_cand_roll)
        for _cand_rot in (_base_rot, _jaw_flip(_base_rot, device)):
            for _cand_sign in (1.0, -1.0):
                _cand_normal = _side_face_normal(_cand_rot, _cand_sign, device, handle_pos.dtype)
                # Must still press the panel the same way; a face pointing elsewhere blocks nothing.
                if float((_cand_normal * _required_press).sum()) < 0.9:
                    continue
                # Score the candidate at BOTH ends of the block phase -- the far approach it enters
                # on and the contact pose it has to finish in -- and keep the worse of the two. The
                # attitude is committed once and then held all the way through, so an attitude that
                # is comfortable at the first waypoint and pinned at contact is not usable; scoring
                # only the entry waypoint picked exactly such a branch and put joint7 back on its
                # hard limit for one door in three.
                _margin = float("inf")
                for _score_clearance in (block_approach_clearances[0], 0.0):
                    _cand_tcp = (contact_anchor
                                 - (block_face_standoff + _score_clearance) * _cand_normal)
                    _cand_q = solve_ik(
                        robot_urdf_path,
                        q_robot[:10],
                        palm_pose=_hand_pose_from_tcp(_cand_tcp, _cand_rot, device),
                        base_pose=None,
                        robot_initial_pose=robot_initial_pose,
                        num_attempts=4,
                    )[0]
                    # Worst wrist margin, the same measure the IK guard uses, so the choice made
                    # here and the search done later agree on what a good arm posture is.
                    _margin = min(_margin, min(
                        min(float(_cand_q[_idx]) - _lo, _hi - float(_cand_q[_idx]))
                        for _idx, _lo, _hi in (
                            (_PANDA_JOINT6_IDX, *_PANDA_JOINT6_SOFT_RANGE),
                            (_PANDA_JOINT7_IDX, *_PANDA_JOINT7_SOFT_RANGE),
                        )
                    ))
                if _best_blk is None or _margin > _best_blk[0]:
                    _best_blk = (_margin, _cand_rot, _cand_sign)
    if _best_blk is not None:
        _, block_rot, block_face_sign = _best_blk
    face_normal = _side_face_normal(block_rot, block_face_sign, device, handle_pos.dtype)


    # Interpolated in JOINT space, so there is no orientation to sequence and no IK branch to pick:
    # the wrist cannot snap or spin here because nothing is asking it to reach a pose. This also
    # retires the old two-stage translate-then-reorient dance and its 224 deg joint7 swing, and the
    # x5 camera arm at (-0.03, -0.08, 0.65) is cleared by construction -- the home pose folds the
    # arm up over the chassis rather than sweeping it low across the base.
    retreat_from_q = q_robot[3:10].clone()
    for retreat_step in range(1, retreat_steps + 1):
        frac = retreat_step / retreat_steps
        q_robot[3:10] = retreat_from_q + frac * (retreat_arm_q - retreat_from_q)
        _set_gripper(q_robot, OPEN_WIDTH)
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=True,
        )

    # -------------------------
    # Step 6.5: Walk the base the REST of the way to the doorway threshold
    # -------------------------
    # The arm is now parked in JOINT space at retreat_arm_q with the jaws open, so nothing
    # downstream of the base is being tracked any more -- moving the base here is a pure base_pose
    # write (palm_pose=None leaves q[3:] untouched, see solve_ik in api.py), not an IK problem, unlike
    # Step 5's pinned-grip move. This is what actually gets the chassis blocking the doorway instead
    # of parking ~0.8 m short of it and leaving Step 7's arm to reach all the way out to the panel
    # alone (see the note on release_base_x_delta_2 above for why that walk does not happen in Step
    # 5 itself). Stops short of the frame plane (measured x~0 for this door) rather than crossing it
    # -- the actual crossing to the far side is Step 8's traverse, after the panel has been pushed
    # out of the way.
    # MEASURED (scripts printed panda_hand world position via FK at each candidate): the folded
    # retreat_arm_q pose holds the hand at a FIXED +0.41 m lateral (y) offset from the base,
    # independent of base x -- so with the base's y untouched (still release_base_y=-0.20) the
    # hand sits at world y~0.21 for the whole walk, only ~0.04-0.13 m under the leaf's y-band
    # (measured [0.25, 0.44] at board=1.35) for whatever range of x the walk sweeps through. That
    # is already true at kf36 (the retract's own end pose, BEFORE this step moves the base at all)
    # -- it is the retract pose itself that runs the hand close past the leaf, matching the
    # panda_link6-vs-panel note above; walking the base in x just sweeps that same tight margin
    # across a wider stretch. Deepening y here (not just x) is the direct fix: -0.45 puts the
    # folded hand at world y~-0.04, clear on the WRONG side of the leaf's band instead of grazing
    # its near edge.
    block_threshold_x = 0.45
    block_threshold_y = -0.45
    block_threshold_steps = 3
    _start_block_bx = base_target_pos[:, 0].clone()
    _start_block_by = base_target_pos[:, 1].clone()
    for _threshold_step in range(1, block_threshold_steps + 1):
        _frac = _threshold_step / block_threshold_steps
        base_target_pos = base_target_pos.clone()
        base_target_pos[:, 0] = _start_block_bx + _frac * (block_threshold_x - _start_block_bx)
        base_target_pos[:, 1] = _start_block_by + _frac * (block_threshold_y - _start_block_by)
        base_target_pose = _make_pose(base_target_pos, tilted_base_rot)
        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=None,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
        )[0]
        _set_gripper(q_robot, OPEN_WIDTH)
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=True,
        )

    # NO bridge waypoint any more. It existed only to halve a ~1.08 m single-step jump
    # between the retract's parked TCP and the first block-approach target -- a gap created
    # by lifting the retract to z=1.55. The retract is now a joint-space move to the folded
    # home pose, so there is no far-flung parking spot to climb back down from, and the
    # staged block_approach_clearances below already walk the hand in.
    first_block_tcp = contact_anchor - (block_face_standoff + block_approach_clearances[0]) * face_normal

    # -------------------------
    # Step 7: Block/push the panel with the SIDE of the gripper, base held still
    # -------------------------
    # --- Staged approach: come in ALONG the face normal, from well clear of the panel ---
    # A single approach frame 0.10 m off the face left the spline free to cut the corner between
    # the retract pose and the panel, driving the gripper straight THROUGH the door on the way in.
    # Walking down the normal in stages keeps every intermediate frame on the robot's side of the
    # panel, so the arm reaches around the leaf instead of penetrating it.
    for clearance in block_approach_clearances:
        tcp_target_pos = contact_anchor - (block_face_standoff + clearance) * face_normal
        palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, block_rot, device)

        q_robot[:10] = _solve_ik_avoiding_wrist_limits(
            robot_urdf_path,
            q_robot,
            palm_target_pose,
            None,
            robot_initial_pose,
            device,
            first_num_attempts=1,  # seeded from the previous stage: walk in smoothly
        )
        _set_gripper(q_robot, OPEN_WIDTH)

        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            # keyframes: the spline must pass through each stage, not shortcut across them
            mark_keyframe=True,
        )

    # --- Side-face contact (keyframe) ---
    tcp_target_pos = contact_anchor - block_face_standoff * face_normal
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, block_rot, device)

    q_robot[:10] = _solve_ik_avoiding_wrist_limits(
        robot_urdf_path,
        q_robot,
        palm_target_pose,
        None,
        robot_initial_pose,
        device,
    )
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # --- Drive the panel to full open, side face still on it ---
    q_door = torch.tensor([push_door_open_angle, 0.0], device=device)
    board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    open_anchor = board_pos.clone()
    open_anchor[:, 0] += block_contact_x_offset
    open_anchor[:, 2] += block_contact_z_offset
    tcp_target_pos = open_anchor - block_face_standoff * face_normal
    palm_target_pose = _hand_pose_from_tcp(tcp_target_pos, block_rot, device)

    q_robot[:10] = _solve_ik_avoiding_wrist_limits(
        robot_urdf_path,
        q_robot,
        palm_target_pose,
        None,
        robot_initial_pose,
        device,
    )
    _set_gripper(q_robot, OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 8: Restore the base to normal yaw, traverse with a suitable arm pose,
    # then finish the traverse with default arm joints
    # -------------------------
    traverse_mid_x = 0.0
    traverse_mid_y = -0.05
    # Drive well past the doorway so the closing door panel can't smash the base from behind.
    traverse_far_x = -1.0
    traverse_steps = 8

    start_base_x = base_target_pos[:, 0].clone()
    start_base_y = base_target_pos[:, 1].clone()
    blocking_yaw = robot_initial_yaw.item() + tilt_base_yaw
    for traverse_step in range(1, traverse_steps + 1):
        frac = traverse_step / traverse_steps
        base_target_pos = base_target_pos.clone()
        base_target_pos[:, 0] = start_base_x + frac * (traverse_mid_x - start_base_x)
        base_target_pos[:, 1] = start_base_y + frac * (traverse_mid_y - start_base_y)
        step_yaw = blocking_yaw + frac * (robot_initial_yaw.item() - blocking_yaw)
        base_target_pose = _make_pose(
            base_target_pos, get_rotation_quat(0.0, 0.0, step_yaw, device)
        )
        # NOT routed through _solve_ik_avoiding_wrist_limits, deliberately. Tried it: this loop
        # does touch panda_joint7's limit at the very last frames (frame 83 of 85, board 1.50,
        # jaws already open, door already held), but guarding it made things WORSE -- when the
        # continuity seed is marginally unsafe the guard re-rolls with random seeds, lands on a
        # distant branch, and brings back the ~320 deg wrist spin in a third of runs, with
        # planning time up to 45 s/door. A limit graze in the last two frames of the traverse is
        # a far cheaper defect than a full revolution mid-motion, so this stays a plain
        # single-seed continuity solve.
        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # loop body: single seed for continuity (no random-restart jumps)
        )[0]
        _set_gripper(q_robot, OPEN_WIDTH)
        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=(traverse_step == traverse_steps),
        )

    base_target_pos[:, 0] = traverse_far_x
    base_target_pos[:, 1] = 0.0
    # Finish the traverse at normal (zero) yaw -- no end tilt.
    base_target_pose = _make_pose(base_target_pos, base_target_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=None,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[3:10] = franka_default_q
    _set_gripper(q_robot, OPEN_WIDTH)
    # Hold the door where the robot actually left it. This used to reset to [0, 0], teleporting
    # the panel from fully open to shut in a single frame while the robot stands a metre away --
    # visible as the door slamming at the end of playback, and meaningless as a tracking target
    # for a reference trajectory (nothing the robot does explains the panel moving).
    q_door = q_door.clone()

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    return robot_traj, door_traj, key_idx_in_key_indices


def state_machine_offline_pull_door(
    robot_urdf_path,
    door_urdf_path,
    robot_initial_pose,   # (1, 7) world
    door_initial_pose,    # (1, 7) world
    robot_initial_q,      # (ndof,)
    door_initial_q,       # (2,) [board, hinge]
    *,
    handle_side: HandleSide = "right",
    device="cpu",
):
    if handle_side == "right":
        return state_machine_offline_right_pull_door(
            robot_urdf_path,
            door_urdf_path,
            robot_initial_pose,
            door_initial_pose,
            robot_initial_q,
            door_initial_q,
            device=device,
        )
    if handle_side == "left":
        return state_machine_offline_left_pull_door(
            robot_urdf_path,
            door_urdf_path,
            robot_initial_pose,
            door_initial_pose,
            robot_initial_q,
            door_initial_q,
            device=device,
        )
    raise ValueError(f"Unsupported handle_side '{handle_side}'. Expected 'right' or 'left'.")
