import math
from typing import Literal

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from DoorOpening.constants.robot_constants import FRANKA_DEFAULT_JOINT_POS, FRANKA_JOINT_NAMES, CAMERA_JOINT_DEFAULT_VALUES, CAMERA_JOINT_VALUES_WHEN_SEARCHING_HINGE, CAMERA_JOINT_VALUES_WHEN_OBSERVING_LEFT
from DoorOpening.utils.state_machine.api import get_board_pos, get_hinge_pos, open_hand, solve_ik

HandleSide = Literal["right", "left"]


def get_rotation_quat(roll, pitch, yaw, device):
    return quat_from_euler_xyz(
        roll=torch.tensor([[roll]], device=device),
        pitch=torch.tensor([[pitch]], device=device),
        yaw=torch.tensor([[yaw]], device=device),
    ).squeeze(0)


def _make_pose(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([position, quat], dim=-1)


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
    # Step 0: Observe
    # -------------------------

    # q_robot[3: 10] = franka_default_q
    # q_robot[:3] = torch.tensor([0.15, 0.2, 0])

    # q_robot[-6:] = torch.tensor(list(CAMERA_JOINT_VALUES_WHEN_SEARCHING_HINGE.values()))

    # _append_state(
    #     robot_traj,
    #     door_traj,
    #     key_idx_in_key_indices,
    #     q_robot,
    #     q_door,
    #     mark_keyframe=True,
    # )

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
    pregrasp_base_x_offset = 0.6
    pregrasp_base_y_offset = -0.30
    # Moved back (0.40 -> 0.45): larger palm<->door x gap to compensate for removing the
    # finger<->panel contact penalty (the demo grasps from further out so fingers don't press panel).
    pregrasp_palm_x_offset = 0.45
    pregrasp_palm_y_offset = -0.20
    pregrasp_palm_z_offset = 0.2

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
    
    q_robot[-6:] = torch.tensor(list(CAMERA_JOINT_DEFAULT_VALUES.values()))

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
    # Moved back (0.035 -> 0.05): larger palm<->door x gap compensates for removing the
    # finger<->panel penalty, so the grasp doesn't drive fingers into the panel.
    grasp_palm_x_offset = 0.05
    grasp_palm_y_offset = -0.085
    grasp_palm_z_offset = 0.08
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
    q_robot[10:26] = open_hand(grasp_open_ratio).to(q_robot.device)

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
    unlatch_hinge_angle = 1.0
    unlatch_palm_y_delta = 0.0
    unlatch_palm_z_delta = -0.08
    unlatch_rot_roll = math.pi
    unlatch_rot_pitch = math.pi
    unlatch_rot_yaw = math.pi + 0.20

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
    pull_theta_start = 0.25
    pull_theta_stop = 1.35
    pull_theta_step = 0.10

    pull_base_x_offset = 0.55
    pull_base_y_gain = -0.25 / 1.45

    pull_palm_x_offset_closed = 0.05
    pull_palm_y_offset_closed = -0.10
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
        q_door = torch.tensor([theta.item(), 0.0], device=device)

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
    safe_open_hand_q = open_hand(1.0).to(q_robot.device)
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

    q_robot[10:26] = safe_open_hand_q
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
    retreat_local_x = 0.22
    retreat_local_y = 0.35
    # LIFT the retract target up: the arm swings a wide arc from here around to the panel-hold
    # pose, and doing that low sweeps it through the arx camera arm on the base. Keeping the hand
    # high makes the swing pass OVER the arx instead of colliding with it.
    retreat_z_lift = 0.28
    # No base move at the retract stage: the base stays at the door-blocking pose set in step 5.
    # Append the retract offset to the LAST palm location (step 5's palm pose), NOT the base pose,
    # so dx/dy/dz are all deltas from where the hand currently is.
    retreat_palm_pos = palm_target_pose[:, :3].clone()
    retreat_palm_pos[:, 0] += retreat_local_x
    retreat_palm_pos[:, 1] += retreat_local_y
    retreat_palm_pos[:, 2] += retreat_z_lift
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
    q_robot[10:26] = safe_open_hand_q

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
    # Extra outward +y for a NON-KEY approach waypoint: the arm first swings OUT to the side of the
    # panel, then moves in to contact -- so it does not penetrate straight through. Reduced from
    # 0.20 -> 0.10 because the approach was landing too far to the right.
    push_approach_out_y = 0.10
    contact_board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        torch.tensor([contact_virtual_door_angle, 0.0], device=device).unsqueeze(0),
    ).to(device)

    # Base stays at the (shifted) door-blocking pose from step 6 -- base_pose=None keeps it put.
    # --- Non-key approach: move outward (extra +y) before pushing in ---
    palm_target_pos = contact_board_pos.clone()
    palm_target_pos[:, 0] += push_contact_x_offset
    palm_target_pos[:, 1] += push_contact_y_offset + push_approach_out_y
    palm_target_pos[:, 2] += push_contact_z_offset
    palm_target_pose = _make_pose(palm_target_pos, push_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[10:26] = safe_open_hand_q

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
    q_robot[10:26] = safe_open_hand_q

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
    q_robot[10:26] = safe_open_hand_q

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
        q_robot[10:26] = safe_open_hand_q
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
    q_robot[10:26] = safe_open_hand_q
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
    default_palm_rot = get_rotation_quat(math.pi, math.pi, math.pi, device)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )
    
    # -------------------------
    # Step 0: Observe
    # -------------------------

    # q_robot[3: 10] = franka_default_q
    # q_robot[:3] = torch.tensor([0.15, 0.2, 0])

    # q_robot[-6:] = torch.tensor(list(CAMERA_JOINT_VALUES_WHEN_SEARCHING_HINGE.values()))

    # _append_state(
    #     robot_traj,
    #     door_traj,
    #     key_idx_in_key_indices,
    #     q_robot,
    #     q_door,
    #     mark_keyframe=True,
    # )

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
    pregrasp_base_x_offset = 0.55
    # Pulled 5cm back off the +y side so the left door pregrasp doesn't reach so far right.
    pregrasp_base_y_offset = 0.25
    # Moved back (0.35 -> 0.40): larger palm<->door x gap to compensate for removing the
    # finger<->panel penalty.
    pregrasp_palm_x_offset = 0.40
    pregrasp_palm_y_offset = 0.15
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
    
    q_robot[-6:] = torch.tensor(list(CAMERA_JOINT_VALUES_WHEN_OBSERVING_LEFT.values()))

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
    # Moved back (0.025 -> 0.05): larger palm<->door x gap compensates for removing the
    # finger<->panel penalty.
    grasp_palm_x_offset = 0.05
    grasp_palm_y_offset = 0.015
    grasp_palm_z_offset = 0.10
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
    )[0]
    q_robot[10:26] = open_hand(grasp_open_ratio).to(q_robot.device)

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
    unlatch_hinge_angle = 1.0
    unlatch_palm_y_delta = 0.0
    unlatch_palm_z_delta = -0.08
    unlatch_rot_roll = math.pi
    unlatch_rot_pitch = math.pi
    unlatch_rot_yaw = math.pi - 0.25

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
    pull_theta_start = 0.25
    pull_theta_stop = 1.25
    pull_theta_step = 0.10

    pull_base_x_offset = 0.45
    pull_base_y_gain = -0.1 / 1.45

    pull_palm_x_offset_closed = 0.05
    pull_palm_y_offset_closed = 0.03
    pull_palm_z_offset = 0.08

    pull_rot_roll_base = math.pi
    pull_rot_roll_per_theta = 0.9
    pull_rot_pitch = math.pi
    pull_rot_yaw = math.pi - 0.15

    theta_values = torch.arange(
        pull_theta_start,
        pull_theta_stop + 1e-6,
        pull_theta_step,
        device=device,
    )
    
    # Retract active perception arms to safe range
    q_robot[-6:] = torch.tensor(list(CAMERA_JOINT_DEFAULT_VALUES.values()))

    for theta in theta_values:
        q_door = torch.tensor([theta.item(), 0.0], device=device)

        handle_pos = get_hinge_pos(
            door_urdf_path,
            door_initial_pose,
            q_door.unsqueeze(0),
        ).to(device)

        base_target_pos = handle_pos.clone()
        base_target_pos[:, 0] += pull_base_x_offset
        base_target_pos[:, 1] = theta.item() * pull_base_y_gain
        base_target_pose = _make_pose(base_target_pos, base_target_rot)

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
    release_palm_x_delta = 0.3
    release_palm_y_delta = 0.0
    release_base_x_delta_2 = -0.18
    release_door_open_angle = 1.35

    # Positive relative yaw turns the base toward the opened panel on -y.
    tilt_base_yaw = -1.0
    tilted_base_rot = get_rotation_quat(
        0.0,
        0.0,
        robot_initial_yaw.item() + tilt_base_yaw,
        device,
    )
    safe_open_hand_q = open_hand(1.0).to(q_robot.device)
    push_palm_rot = get_rotation_quat(0.0, 0.0, math.pi / 2, device)
    retreat_local_x = 0.10
    retreat_local_y = -0.42
    retreat_z_lift = 0.04
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

    q_robot[10:26] = safe_open_hand_q
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
    tilted_base_yaw_world = robot_initial_yaw + tilt_base_yaw
    retreat_dx, retreat_dy = _rotate_xy_counterclockwise(
        retreat_local_x,
        retreat_local_y,
        tilted_base_yaw_world,
    )

    retreat_palm_pos = base_target_pos.clone()
    retreat_palm_pos[:, 0] += retreat_dx
    retreat_palm_pos[:, 1] += retreat_dy
    retreat_palm_pos[:, 2] = palm_target_pose[:, 2].clone() + retreat_z_lift
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
    q_robot[10:26] = safe_open_hand_q
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
    )[0]
    q_robot[10:26] = safe_open_hand_q

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
    )[0]
    q_robot[10:26] = safe_open_hand_q

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
            num_attempts=1,  # loop body: single seed for continuity (no random-restart branch jumps)
        )[0]
        q_robot[10:26] = safe_open_hand_q
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
    q_robot[10:26] = safe_open_hand_q
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
