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
    pregrasp_palm_x_offset = 0.40
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
    grasp_palm_x_offset = 0.05
    grasp_palm_y_offset = -0.07
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
    # Blocking pose Y: -y approaches the open leaf, +y backs away. Slightly closer to the leaf.
    release_base_y = 0.20
    release_palm_x_delta = 0.25
    release_palm_y_delta = -0.1
    release_base_x_delta_2 = -0.18
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
    tilted_base_yaw_world = robot_initial_yaw + tilt_base_yaw
    # Pull the retract target IN to a reachable tuck (the old 0.60/0.50 offset put it ~1.0 m
    # from the franka base, at/past its ~0.93 m reach, so the arm pinned at full extension and
    # stabbed the panel instead of retracting). x stays large enough to avoid the shoulder-
    # mount back-fold; +y keeps it on the opposite side from the left door.
    retreat_local_x = 0.50
    retreat_local_y = 0.45
    # Lower the retract target so it drops into comfortable reach instead of sitting above the
    # shoulder at the extension limit -- this is what keeps the larger x/y offset reachable
    # (verified: (0.55,0.50) at z~0.9-1.0 still solves with the retract orientation).
    retreat_z_lift = -0.10
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
    # Contact point is now the panel CENTER (get_board_pos), which keeps a stable moment arm
    # even on narrow doors. Only a small x offset so we push near the center, not toward the
    # hinge axis (short lever -> strong forces on the franka).
    push_contact_x_offset = -0.1
    push_contact_y_offset = 0.25
    # Lower contact point on the panel (was 0.25) so the hand pushes lower on the door.
    push_contact_z_offset = 0.10
    # Extra outward +y for a NON-KEY approach waypoint: the arm first swings OUT to the side of
    # the panel, then moves in to contact -- so it does not penetrate straight through the panel.
    push_approach_out_y = 0.20
    contact_board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        torch.tensor([contact_virtual_door_angle, 0.0], device=device).unsqueeze(0),
    ).to(device)

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
    traverse_mid_x = -0.05
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
    pregrasp_palm_x_offset = 0.35
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
    grasp_palm_x_offset = 0.04
    grasp_palm_y_offset = 0.0
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
    push_contact_x_offset = -0.1
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
    # Apply the forward (x) offset so the hand walks along the panel off the bare free edge,
    # but NOT the lateral (y) offset: at full-open the panel face normal is ~y, so y is a
    # perpendicular gap that would hold the hand off the panel surface.
    palm_target_pos[:, 0] += push_contact_x_offset
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
    traverse_mid_x = -0.05
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
