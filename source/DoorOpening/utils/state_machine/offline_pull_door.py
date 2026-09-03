import math
from typing import Literal

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from DoorOpening.constants.robot_constants import FRANKA_DEFAULT_JOINT_POS, FRANKA_JOINT_NAMES
from DoorOpening.utils.state_machine.api import get_board_pos, get_hinge_pos, solve_ik
from DoorOpening.constants.robot_constants import (
    DRIVEN_FINGER_JOINT_NAME,
    FULL_JOINT_NAMES,
    GRIPPER_OPEN_WIDTH,
)

HandleSide = Literal["right", "left"]

# Null-space posture anchor for the LEFT pull planner's IK (see api.solve_ik).
#
# The stock FRANKA_DEFAULT_JOINT_POS anchors panda_joint2 at -0.25*pi, which resolves the arm's
# redundancy into a SHOULDER-DOWN branch: panda_link2's far end (the panda_link3 origin) sinks to
# z = 0.70 m, straight into the arx camera arm's band (z 0.65..0.75, |xy| <= 0.282 off the base
# axis). Measured over the first 8 PartNetv5_plusplus doors, the closest franka<->arx approach on
# the reference trajectory was 1 mm -- i.e. the reference itself interpenetrates, on essentially
# every door, and no reward penalty can fix a demo that is already inside the camera arm.
#
# Anchoring the shoulder at +0.15*pi instead lifts that branch. Only panda_joint2 moves; every
# other joint keeps the stock anchor, and the spawn pose (DEFAULT_JOINT_POS) is a separate constant
# and is untouched. This is ONE of three changes that had to land together -- see the retract and
# pull_base_y_offset notes in the left planner; the anchor alone tops out at -0.028 m (still
# interpenetrating), because it cannot move a waypoint that is itself sitting on the base.
LEFT_PULL_IK_ANCHOR_JOINT_POS = {
    **FRANKA_DEFAULT_JOINT_POS,
    "panda_joint2": 0.15 * math.pi,
}


# The Franka gripper is ONE commanded DOF, at this index of FULL_JOINT_NAMES -- NOT the 16-joint
# LEAP block the old code wrote. q_robot is laid out base(3) + panda(7) + finger(1) + x5 camera(6)
# = 17, so the old `q_robot[10:26] = open_hand(...)` both mismatched shape (7 slots vs 2 values)
# and, had it fitted, would have overwritten all six x5 CAMERA joints along with the gripper.
GRIPPER_Q_IDX = FULL_JOINT_NAMES.index(DRIVEN_FINGER_JOINT_NAME)


def _set_gripper(q_robot: torch.Tensor, width: float) -> None:
    """Command the single driven finger joint. The follower finger mimics it in the model."""
    q_robot[GRIPPER_Q_IDX] = width


def get_rotation_quat(roll, pitch, yaw, device):
    return quat_from_euler_xyz(
        roll=torch.tensor([[roll]], device=device),
        pitch=torch.tensor([[pitch]], device=device),
        yaw=torch.tensor([[yaw]], device=device),
    ).squeeze(0)


def _make_pose(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([position, quat], dim=-1)

# FRAME: every palm_pose below is a panda_hand pose, because that is what solve_ik drives
# (api.py builds PinocchioIKSolver with ee_link_name="panda_hand").
#
# panda_hand is the WRIST MOUNT, not the contact point -- glorbot.urdf hangs palm_center, the grasp
# centre between the fingers, off it at xyz (0, 0, 0.1034), i.e. 103.4 mm further along the
# approach axis. So a target here places the wrist, and the fingers close ~10 cm beyond it. The
# offsets in each step are tuned against THAT convention; they are not contact points, and the
# LEAP-era palm_lower values they grew from meant something different again.


def _identity_quat(device) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)


def _append_state(
    robot_traj: list[torch.Tensor],
    door_traj: list[torch.Tensor],
    key_indices: list[int],
    q_robot: torch.Tensor,
    q_door: torch.Tensor,
    *,
    mark_keyframe: bool,
) -> None:
    # print(f"Appending state: {q_robot}, {q_door}")
    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())
    if mark_keyframe:
        key_indices.append(len(robot_traj) - 1)


def _pull_hinge_angle(
    theta: float,
    unlatch_angle: float,
    hold_until_theta: float,
    release_by_theta: float,
) -> float:
    """Lever (joint_2) angle to command at panel angle ``theta`` during the pull sweep.

    The lever is HELD down at ``unlatch_angle`` (i.e. against its mechanical stop) until the panel
    has opened to ``hold_until_theta``, then ramps linearly back to 0 by ``release_by_theta``.

    This replaces dropping the lever to 0 on the first pull frame, which was wrong in three ways:

    1. Mechanically, a depressed lever resting on its hard stop is what gives the pull a RIGID
       reaction point. Released to 0, the lever is free to swing back down through its whole travel,
       so a gripper pulling on the handle first re-rotates the lever (up to the stop again) before
       any of that force reaches the panel. The handle turns under the grasp instead of moving the
       door -- which matters far more for a 2-finger pinch than it did for a wrapped hand.
    2. It is what a person does: you keep the lever pressed while you are still holding the handle
       and let it spring back only as you release, which is now Step 5.
    3. It keeps ``edit_door_articulation``'s relock test (panel near closed AND lever below the
       unlatch threshold) decisively false through the early pull, so the latch cannot re-catch if
       the panel drifts back toward closed.
    """
    if theta <= hold_until_theta:
        return unlatch_angle
    if theta >= release_by_theta:
        return 0.0
    span = release_by_theta - hold_until_theta
    return unlatch_angle * (release_by_theta - theta) / span


def _rotate_xy_clockwise(x_offset: float, y_offset: float, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return x_offset * c + y_offset * s, -x_offset * s + y_offset * c


def _rotate_xy_counterclockwise(x_offset: float, y_offset: float, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(theta)
    s = torch.sin(theta)
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
    default_palm_rot = get_rotation_quat(math.pi, math.pi, math.pi, device)

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

    # Unified with the push-right planner so pregrasp/grasp base + palm offsets match
    # (keeps the base off the side wall, like the push planner).
    # Base pulled back for more grasp standoff (0.67); the robot grasps from a bit more standoff (palm offsets
    # unchanged -> the arm just reaches slightly further forward).
    # Base stands 5 cm further BACK from the door (0.67 -> 0.72; +x is away from the panel, the
    # robot faces -x). Shared by pregrasp, grasp AND unlatch -- Steps 2 and 3 reuse this same
    # base_target_pose -- so this one value backs the whole approach off. Palm offsets are
    # unchanged and measured from the handle, so the arm simply reaches 5 cm further forward.
    pregrasp_base_x_offset = 0.72
    pregrasp_base_y_offset = -0.35
    # Moved back (0.40 -> 0.45): larger palm<->door x gap to compensate for removing the
    # finger<->panel contact penalty (the demo grasps from further out so fingers don't press panel).
    pregrasp_palm_x_offset = 0.4
    pregrasp_palm_y_offset = -0.15
    pregrasp_palm_z_offset = 0.25

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
        mark_keyframe=True,
    )

    # -------------------------
    # Step 2: Move to grasp
    # -------------------------
    # Unified with the push-right planner so grasp palm<->handle offsets match.
    # Right-handle door: nudge grasp EE ~2.5 cm LEFT (-y, toward door center) and ~1.5 cm FORWARD
    # (-x, toward the handle/door). Robot faces -x, so right=+y / left=-y / forward=-x.
    # Palm<->door x gap kept at 0.035: enough clearance that the grasp doesn't drive fingers into
    # the panel, without reaching as deep as the 0.025 tuned for the thicker HEAD lever bars.
    grasp_palm_x_offset = 0.035
    grasp_palm_y_offset = -0.085
    grasp_palm_z_offset = 0.10
    grasp_open_ratio = 0.70

    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += grasp_palm_x_offset
    palm_target_pos[:, 1] += grasp_palm_y_offset
    palm_target_pos[:, 2] += grasp_palm_z_offset
    palm_target_pose = _make_pose(palm_target_pos, default_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

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
    # Target the lever's HARD STOP (HANDLE_OPEN_LIMIT_RAD = 0.95 rad in the door generator), not
    # past it: the reference presses the handle firmly against its mechanical stop, which is both
    # what a person does and what gives the pull a rigid reaction point. Must stay above the
    # highest randomized unlatch threshold (0.85 rad) so every door actually unlatches.
    unlatch_hinge_angle = 0.95
    unlatch_palm_y_delta = 0.0
    unlatch_palm_z_delta = -0.08
    unlatch_rot_roll = math.pi
    unlatch_rot_pitch = math.pi
    unlatch_rot_yaw = math.pi + 0.25

    q_door = torch.tensor([0.0, unlatch_hinge_angle], device=device)

    palm_target_pose = palm_target_pose.clone()
    palm_target_pose[:, 1] += unlatch_palm_y_delta
    palm_target_pose[:, 2] += unlatch_palm_z_delta
    palm_target_pose[:, 3:] = get_rotation_quat(
        unlatch_rot_roll,
        unlatch_rot_pitch,
        unlatch_rot_yaw,
        device,
    )

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
        mark_keyframe=True,
    )

    # -------------------------
    # Step 4: Pull door open
    # -------------------------
    pull_theta_start = 0.3
    pull_theta_stop = 1.25
    pull_theta_step = 0.10

    # Keep the lever pressed against its stop for most of the pull, then let it spring back over the
    # last stretch so it is fully restored by the time Step 5 releases the handle. See
    # _pull_hinge_angle for why holding beats releasing on the first frame.
    # Start restoring once the panel is ~29 deg open (0.5 rad). By then the latch bolt is long
    # clear of the strike, so holding the lever down further only fights the return spring.
    pull_hinge_hold_until_theta = 0.5
    pull_hinge_release_by_theta = pull_theta_stop

    # Base held 5 cm further back through the pull sweep too (0.55 -> 0.60), matching the
    # backed-off approach above so the arm does not have to re-close the gap mid-pull.
    pull_base_x_offset = 0.60
    pull_base_y_gain = -0.25 / 1.45

    pull_palm_x_offset_closed = 0.05
    pull_palm_y_offset_closed = -0.08
    pull_palm_z_offset = 0.08

    pull_rot_roll_base = math.pi
    pull_rot_roll_per_theta = -1.0
    pull_rot_pitch = math.pi
    pull_rot_yaw = math.pi

    theta_values = torch.arange(
        pull_theta_start,
        pull_theta_stop + 1e-6,
        pull_theta_step,
        device=device,
    )

    for theta in theta_values:
        q_door = torch.tensor(
            [
                theta.item(),
                _pull_hinge_angle(
                    theta.item(),
                    unlatch_hinge_angle,
                    pull_hinge_hold_until_theta,
                    pull_hinge_release_by_theta,
                ),
            ],
            device=device,
        )

        handle_pos = get_hinge_pos(
            door_urdf_path,
            door_initial_pose,
            q_door.unsqueeze(0),
        ).to(device)

        base_target_pos = handle_pos.clone()
        base_target_pos[:, 0] += pull_base_x_offset
        base_target_pos[:, 1] = theta.item() * pull_base_y_gain
        base_target_pose = _make_pose(base_target_pos, base_target_rot)

        palm_dx, palm_dy = _rotate_xy_clockwise(
            pull_palm_x_offset_closed,
            pull_palm_y_offset_closed,
            theta,
        )

        palm_target_pos = handle_pos.clone()
        palm_target_pos[:, 0] += palm_dx
        palm_target_pos[:, 1] += palm_dy
        palm_target_pos[:, 2] += pull_palm_z_offset

        palm_target_rot = get_rotation_quat(
            pull_rot_roll_base + pull_rot_roll_per_theta * theta.item(),
            pull_rot_pitch,
            pull_rot_yaw,
            device,
        )
        palm_target_pose = _make_pose(palm_target_pos, palm_target_rot)

        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            num_attempts=1,  # loop body: single seed for continuity (no random-restart branch jumps)
        )[0]

        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=False,
        )

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 5: Move to the blocking base pose while releasing the hinge
    # -------------------------
    _, _, robot_initial_yaw = euler_xyz_from_quat(base_target_rot)

    release_base_x_delta_1 = -0.12
    # Blocking pose Y: -y approaches the open leaf (left), +y backs away (right). Moved a bit LEFT
    # (0.20 -> 0.10), toward the leaf, for better blocking -- the arm now retracts to the right, so
    # the base no longer needs to stay clear of it on that side.
    release_base_y = 0.10
    release_palm_x_delta = 0.25
    release_palm_y_delta = -0.1
    # Backed the blocking base off the door (-0.18 -> -0.10, i.e. +0.08 behind in x) so the arm has
    # reach headroom to retract further. This is the door-blocking pose held through the push, so
    # keep the back-off small to preserve blocking margin. (+x back is fine; +y right was not.)
    release_base_x_delta_2 = -0.10
    release_door_open_angle = 1.35

    # Turn the base toward the opened panel on +y before the push phase.
    tilt_base_yaw = 1.0
    tilted_base_rot = get_rotation_quat(
        0.0,
        0.0,
        robot_initial_yaw.item() + tilt_base_yaw,
        device,
    )
    push_palm_rot = get_rotation_quat(0.0, 0.0, -math.pi / 2, device)
    contact_virtual_door_angle = 1.1
    push_door_open_angle = 1.5

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

    palm_target_pose = palm_target_pose.clone()
    palm_target_pose[:, 0] += release_palm_x_delta
    palm_target_pose[:, 1] += release_palm_y_delta
    palm_target_pose[:, 3:] = default_palm_rot

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

    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)
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
    # Step 6: Retract the arm to the left side of the tilted base
    # -------------------------
    # Retract offset applied DIRECTLY in world frame -- no base-yaw rotation (that rotation was
    # mixing the axes: "dx too small, dy too large"). Now these map straight to world directions:
    # +x pulls the hand BACKWARD off the door; +y nudges it to the RIGHT to clear the panel.
    retreat_local_x = 0.3
    retreat_local_y = 0.3
    # LIFT the retract target up: the arm swings a wide arc from here around to the panel-hold
    # pose, and doing that low sweeps it through the arx camera arm on the base. Keeping the hand
    # high makes the swing pass OVER the arx instead of colliding with it.
    # retreat_z_lift = 0.05
    # No base move at the retract stage: the base stays at the door-blocking pose set in step 5.
    # Append the retract offset to the LAST palm location (step 5's palm pose), NOT the base pose,
    # so dx/dy/dz are all deltas from where the hand currently is.
    retreat_palm_pos = palm_target_pose[:, :3].clone()
    retreat_palm_pos[:, 0] += retreat_local_x
    retreat_palm_pos[:, 1] += retreat_local_y
    # retreat_palm_pos[:, 2] += retreat_z_lift
    retreat_palm_pos[:, 2] = 1.2
    retreat_palm_pose = _make_pose(retreat_palm_pos, default_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=retreat_palm_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
        num_attempts=1,  # retract must continue smoothly from the blocking pose; the random-
        # restart fallback would otherwise flip the arm ~180 deg to a different IK branch.
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 7: Push the panel open with the arm while keeping the base still
    # -------------------------
    # Push contact anchored at the panel CENTER (get_board_pos) then shifted toward the FREE/OUTER
    # edge for a longer lever arm (lighter push force). At the contact door angle the panel has
    # swung toward the robot, so its outer edge is at +x from the center -- a POSITIVE x offset
    # moves ALONG the panel toward that edge. (The old -0.1 pushed toward the hinge/inner edge.)
    push_contact_x_offset = 0.13
    push_contact_y_offset = 0.25
    # Contact height on the panel: raised back up so the hand holds/pushes the panel HIGHER (the
    # low contact made the arm swing down toward the arx on the way in).
    push_contact_z_offset = 0.22
    contact_board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        torch.tensor([contact_virtual_door_angle, 0.0], device=device).unsqueeze(0),
    ).to(device)

    # Base stays at the (shifted) door-blocking pose from step 6 -- base_pose=None keeps it put.
    # --- Non-key approach: move outward (extra +y) before pushing in ---
    palm_target_pos = contact_board_pos.clone()
    palm_target_pos[:, 0] += push_contact_x_offset
    palm_target_pos[:, 1] += push_contact_y_offset
    palm_target_pos[:, 2] += push_contact_z_offset
    palm_target_pose = _make_pose(palm_target_pos, push_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=False,
    )

    # --- Push contact (keyframe) ---
    palm_target_pos = contact_board_pos.clone()
    palm_target_pos[:, 0] += push_contact_x_offset
    palm_target_pos[:, 1] += push_contact_y_offset
    palm_target_pos[:, 2] += push_contact_z_offset
    palm_target_pose = _make_pose(palm_target_pos, push_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    q_door = torch.tensor([push_door_open_angle, 0.0], device=device)
    board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    palm_target_pos = board_pos.clone()
    # Apply the forward (x) offset plus a SMALL lateral (y) offset so the palm stays just off
    # the panel face at full-open instead of penetrating it (a large y would float it off).
    push_open_y_offset = 0.1
    palm_target_pos[:, 0] += push_contact_x_offset
    palm_target_pos[:, 1] += push_open_y_offset
    palm_target_pos[:, 2] += push_contact_z_offset
    palm_target_pose = _make_pose(palm_target_pos, push_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

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
    # keeps HOLDING the door panel (palm pinned), so the arm does not jerk away from the panel.
    # The tilted blocking yaw is restored to normal over the same sweep.
    traverse_mid_x = 0.0
    # Keep the base a bit closer to the +y door panel during the traverse (was 0.0) so the arm
    # can still reach the panel it is holding instead of drifting out of reach.
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
        _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)
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
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)
    q_door = torch.tensor([0.0, 0.0], device=device)

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
    default_palm_rot = get_rotation_quat(math.pi / 2, 0, -math.pi / 2 - math.pi / 4, device)

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
    handle_pos = get_hinge_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    # Unified with the push-left planner so pregrasp/grasp base + palm offsets match
    # (keeps the base off the side wall, like the push planner).
    # Base standoff kept CONSTANT with the right-door planner (0.67) so left/right grasp the same
    # distance out; palm offsets unchanged -> the arm just reaches slightly further forward.
    # Base stands 5 cm further BACK from the door (0.67 -> 0.72; +x is away from the panel, the
    # robot faces -x). Shared by pregrasp, grasp AND unlatch -- Steps 2 and 3 reuse this same
    # base_target_pose -- so this one value backs the whole approach off. Palm offsets are
    # unchanged and measured from the handle, so the arm simply reaches 5 cm further forward.
    pregrasp_base_x_offset = 0.72
    # Pulled 5cm back off the +y side so the left door pregrasp doesn't reach so far right.
    pregrasp_base_y_offset = 0.25
    # Moved back (0.35 -> 0.40): larger palm<->door x gap to compensate for removing the
    # finger<->panel penalty.
    pregrasp_palm_x_offset = 0.25
    pregrasp_palm_y_offset = 0.15
    pregrasp_palm_z_offset = 0.25

    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += pregrasp_base_x_offset
    base_target_pos[:, 1] += pregrasp_base_y_offset
    # Left-door camera FOV: tilt the base a little toward the handle for pregrasp -> unlatch, so the
    # ARX/x5 camera arm keeps the handle in good view. A positive yaw delta turns the base toward the
    # handle here; flip the sign if the camera looks the wrong way. base_target_rot (untilted) is
    # restored from Step 4 onward, so only pregrasp/grasp/unlatch (which reuse this base_target_pose)
    # are tilted.
    pregrasp_base_tilt_yaw = 0.3
    _, _, _base_yaw = euler_xyz_from_quat(base_target_rot)
    pregrasp_tilt_base_rot = get_rotation_quat(0.0, 0.0, _base_yaw.item() + pregrasp_base_tilt_yaw, device)
    base_target_pose = _make_pose(base_target_pos, pregrasp_tilt_base_rot)

    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += pregrasp_palm_x_offset
    palm_target_pos[:, 1] += pregrasp_palm_y_offset
    palm_target_pos[:, 2] += pregrasp_palm_z_offset
    palm_target_pose = _make_pose(palm_target_pos, default_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]
    

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
    # Left-handle door: nudge grasp EE ~2.5 cm RIGHT (+y, toward door center) and ~1.5 cm FORWARD
    # (-x, toward the handle/door). Robot faces -x, so right=+y / left=-y / forward=-x.
    # Palm<->door x gap kept at 0.035, matching the right-door planner so left/right grasp the
    # same distance out from the panel.
    grasp_palm_x_offset = 0.06
    grasp_palm_y_offset = 0.015
    grasp_palm_z_offset = 0.04
    grasp_open_ratio = 0.7

    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += grasp_palm_x_offset
    palm_target_pos[:, 1] += grasp_palm_y_offset
    palm_target_pos[:, 2] += grasp_palm_z_offset
    palm_target_pose = _make_pose(palm_target_pos, default_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

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
    # Target the lever's HARD STOP (HANDLE_OPEN_LIMIT_RAD = 0.95 rad in the door generator), not
    # past it: the reference presses the handle firmly against its mechanical stop, which is both
    # what a person does and what gives the pull a rigid reaction point. Must stay above the
    # highest randomized unlatch threshold (0.85 rad) so every door actually unlatches.
    unlatch_hinge_angle = 0.95
    unlatch_palm_y_delta = 0.015
    unlatch_palm_z_delta = -0.10
    unlatch_rot_roll = math.pi / 2
    unlatch_rot_pitch = 0.85
    unlatch_rot_yaw = -math.pi / 2 - math.pi / 3

    q_door = torch.tensor([0.0, unlatch_hinge_angle], device=device)

    palm_target_pose = palm_target_pose.clone()
    palm_target_pose[:, 1] += unlatch_palm_y_delta
    palm_target_pose[:, 2] += unlatch_palm_z_delta
    palm_target_pose[:, 3:] = get_rotation_quat(
        unlatch_rot_roll,
        unlatch_rot_pitch,
        unlatch_rot_yaw,
        device,
    )

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]

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
    pull_theta_start = 0.30
    pull_theta_stop = 1.25
    pull_theta_step = 0.10

    # Base held 5 cm further back through the pull sweep too (0.6 -> 0.65), matching the
    # backed-off approach above so the arm does not have to re-close the gap mid-pull.
    pull_base_x_offset = 0.65
    # Constant lateral shift of the base held through the pull, in WORLD y (+y = the robot's right,
    # per this file's robot-faces--x convention). The base y was previously a pure function of theta
    # with no standing offset, so the chassis -- and the arx bolted to it -- tracked straight up the
    # line the arm was working along. Measured: this is what clears the END of the pull sweep, where
    # panda_link3 was hitting arx link4.
    pull_base_y_offset = 0.08
    pull_base_y_gain = -0.1 / 1.45

    # Keep the lever pressed against its stop for most of the pull, then let it spring back over the
    # last stretch so it is fully restored by the time Step 5 releases the handle. Same values as the
    # right-door planner; see _pull_hinge_angle for why holding beats releasing on the first frame.
    # Start restoring once the panel is ~29 deg open (0.5 rad). By then the latch bolt is long
    # clear of the strike, so holding the lever down further only fights the return spring.
    pull_hinge_hold_until_theta = 0.5
    pull_hinge_release_by_theta = pull_theta_stop

    pull_palm_x_offset_closed = 0.055
    pull_palm_y_offset_closed = 0.03
    pull_palm_z_offset = 0.05

    pull_rot_roll_base = math.pi / 2
    pull_rot_roll_per_theta = 0.9
    pull_rot_pitch = 0
    pull_rot_yaw = - 3 * math.pi / 4

    theta_values = torch.arange(
        pull_theta_start,
        pull_theta_stop + 1e-6,
        pull_theta_step,
        device=device,
    )
    
    # Retract active perception arms to safe range

    for theta in theta_values:
        q_door = torch.tensor(
            [
                theta.item(),
                _pull_hinge_angle(
                    theta.item(),
                    unlatch_hinge_angle,
                    pull_hinge_hold_until_theta,
                    pull_hinge_release_by_theta,
                ),
            ],
            device=device,
        )

        handle_pos = get_hinge_pos(
            door_urdf_path,
            door_initial_pose,
            q_door.unsqueeze(0),
        ).to(device)

        base_target_pos = handle_pos.clone()
        base_target_pos[:, 0] += pull_base_x_offset
        base_target_pos[:, 1] = pull_base_y_offset + theta.item() * pull_base_y_gain
        pull_open_base_tilt_yaw = 0.2
        _, _, _base_yaw = euler_xyz_from_quat(base_target_rot)
        pull_open_tilt_base_rot = get_rotation_quat(0.0, 0.0, _base_yaw.item() + pull_open_base_tilt_yaw, device)
        base_target_pose = _make_pose(base_target_pos, pull_open_tilt_base_rot)

        palm_dx, palm_dy = _rotate_xy_counterclockwise(
            pull_palm_x_offset_closed,
            pull_palm_y_offset_closed,
            theta,
        )

        palm_target_pos = handle_pos.clone()
        palm_target_pos[:, 0] += palm_dx
        palm_target_pos[:, 1] += palm_dy
        palm_target_pos[:, 2] += pull_palm_z_offset

        palm_target_rot = get_rotation_quat(
            pull_rot_roll_base + pull_rot_roll_per_theta * theta.item(),
            pull_rot_pitch,
            pull_rot_yaw,
            device,
        )
        palm_target_pose = _make_pose(palm_target_pos, palm_target_rot)

        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
            num_attempts=1,  # loop body: single seed for continuity (no random-restart branch jumps)
        )[0]

        _append_state(
            robot_traj,
            door_traj,
            key_idx_in_key_indices,
            q_robot,
            q_door,
            mark_keyframe=False,
        )

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 5: Move to the blocking base pose while releasing the hinge
    # -------------------------
    _, _, robot_initial_yaw = euler_xyz_from_quat(base_target_rot)

    release_base_x_delta_1 = -0.12
    # Blocking pose Y: +y approaches the open leaf, -y backs away. Kept a margin off the leaf.
    release_base_y = -0.20
    release_palm_x_delta = 0.25
    release_palm_y_delta = 0.0
    release_base_x_delta_2 = -0.18
    release_door_open_angle = 1.35

    # Positive relative yaw turns the base toward the opened panel on -y.
    tilt_base_yaw = -0.8
    tilted_base_rot = get_rotation_quat(
        0.0,
        0.0,
        robot_initial_yaw.item() + tilt_base_yaw,
        device,
    )
    push_palm_rot = get_rotation_quat(0, 0, math.pi, device)
    # Retract target: anchored on the PALM in world axes with an ABSOLUTE height, not on the base
    # with a base-yaw-rotated offset and a handle-relative lift. The old form aimed the hand 0.3 m
    # from the base axis, which IS the arx camera arm's volume (its links sit within |xy| = 0.282 at
    # z = 0.65..0.75), and the relative lift meant the height that separated them varied with handle
    # height. 1.25 m is at the practical top of the arm's reach from the blocking base pose -- pushing
    # it to 1.35 measured WORSE, because the target goes unreachable and the IK returns a stretched
    # best-effort pose.
    retreat_world_x = 0.30
    retreat_world_y = -0.30
    retreat_palm_z = 1.25
    # Larger x offset magnitude (was -0.3) so the arm reaches a bit further forward into the panel.
    # Contact point is now the panel CENTER (get_board_pos); only a small x offset so we push
    # near the center, not toward the hinge axis (short lever -> strong forces on the franka).
    # Push contact shifted toward the panel FREE/OUTER edge for a longer lever arm (+x moves along
    # the panel toward that edge, since it swings toward the robot as it opens). Pulled back a bit
    # toward the CENTER (0.18 -> 0.10) so the contact isn't so far out on the panel.
    push_contact_x_offset = 0.10
    push_contact_y_offset = -0.2
    push_contact_z_offset = 0.1
    contact_virtual_door_angle = 1.0
    push_door_open_angle = 1.5

    base_target_pos[:, 0] += release_base_x_delta_1
    base_target_pos[:, 1] = release_base_y
    base_target_pose = _make_pose(base_target_pos, tilted_base_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=False,
    )

    palm_target_pose = palm_target_pose.clone()
    palm_target_pose[:, 0] += release_palm_x_delta
    palm_target_pose[:, 1] += release_palm_y_delta
    palm_target_pose[:, 3:] = default_palm_rot

    base_target_pos[:, 0] += release_base_x_delta_2
    base_target_pos[:, 1] = release_base_y
    base_target_pose = _make_pose(base_target_pos, tilted_base_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]

    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)
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
    # Step 6: Retract the arm to the right side of the tilted base
    # -------------------------
    retreat_palm_pos = palm_target_pose[:, :3].clone()
    retreat_palm_pos[:, 0] += retreat_world_x
    retreat_palm_pos[:, 1] += retreat_world_y
    retreat_palm_pos[:, 2] = retreat_palm_z
    retreat_palm_pose = _make_pose(retreat_palm_pos, default_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=retreat_palm_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
        num_attempts=1,  # retract must continue smoothly from the blocking pose; the random-
        # restart fallback would otherwise flip the arm ~180 deg to a different IK branch.
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)
    # retreat_arm_q = q_robot[3:10].clone()

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 7: Push the panel open with the arm while keeping the base still
    # -------------------------
    contact_board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        torch.tensor([contact_virtual_door_angle, 0.0], device=device).unsqueeze(0),
    ).to(device)

    palm_target_pos = contact_board_pos.clone()
    palm_target_pos[:, 0] += push_contact_x_offset
    palm_target_pos[:, 1] += push_contact_y_offset
    palm_target_pos[:, 2] += push_contact_z_offset
    palm_target_pose = _make_pose(palm_target_pos, push_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    q_door = torch.tensor([push_door_open_angle, 0.0], device=device)
    board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    palm_target_pos = board_pos.clone()
    # Apply the forward (x) offset so the hand walks along the panel off the bare free edge, PLUS a
    # SMALL lateral (y) offset so the palm stays just OFF the panel face instead of penetrating it.
    # At full-open the panel normal is ~y and the hand is on the -y side, so a small -y keeps it
    # off the surface (mirror of the right door's +y push_open_y_offset).
    push_open_y_offset = -0.08
    palm_target_pos[:, 0] += push_contact_x_offset
    palm_target_pos[:, 1] += push_open_y_offset
    palm_target_pos[:, 2] += push_contact_z_offset
    palm_target_pose = _make_pose(palm_target_pos, push_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)

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
    # keeps HOLDING the door panel (palm pinned), so the arm does not jerk away from the panel.
    # The tilted blocking yaw is restored to normal over the same sweep.
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
        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
            reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
            num_attempts=1,  # loop body: single seed for continuity (no random-restart branch jumps)
        )[0]
        _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)
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
        reference_joint_pos=LEFT_PULL_IK_ANCHOR_JOINT_POS,
    )[0]
    q_robot[3:10] = franka_default_q
    _set_gripper(q_robot, GRIPPER_OPEN_WIDTH)
    q_door = torch.tensor([0.0, 0.0], device=device)

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
