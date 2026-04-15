import math
from typing import Literal

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from DoorOpening.constants.robot_constants import FRANKA_DEFAULT_JOINT_POS, FRANKA_JOINT_NAMES
from DoorOpening.utils.state_machine.api import get_board_edge, get_board_pos, get_hinge_pos, open_hand, solve_ik

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

    The tuning values are kept inline inside each step so they can be edited
    the same way as the archived ``state_machine_offline`` function.
    """

    q_robot, q_door, robot_traj, door_traj, key_idx_in_key_indices = _init_planner_state(
        robot_initial_q, door_initial_q
    )

    base_target_rot = robot_initial_pose[:, 3:].to(device).clone()
    default_palm_rot = get_rotation_quat(math.pi, math.pi, math.pi, device)

    # -------------------------
    # Step 1: Pregrasp
    # -------------------------
    handle_pos = get_hinge_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    pregrasp_base_x_offset = 0.55
    pregrasp_base_y_offset = -0.30
    pregrasp_palm_x_offset = 0.25
    pregrasp_palm_y_offset = -0.10
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
    grasp_palm_x_offset = 0.05
    grasp_palm_y_offset = -0.10
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
    unlatch_palm_y_delta = -0.02
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
    pull_base_y_gain = 0.25 / 1.45

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
    release_base_y = 0.15
    release_palm_x_delta = 0.60
    release_palm_y_delta = -0.20
    release_base_x_delta_2 = -0.28
    release_door_open_angle = 1.35

    # Negative relative yaw turns the base toward the opened panel on +y.
    tilt_base_yaw = 0.35
    tilted_base_rot = get_rotation_quat(
        0.0,
        0.0,
        robot_initial_yaw.item() + tilt_base_yaw,
        device,
    )
    block_palm_rot = get_rotation_quat(math.pi, math.pi, math.pi - math.pi / 2, device)
    block_finger_joint_q = torch.tensor([0.0, math.pi / 2, 0.0, 0.0], device=device)

    traverse_mid_x = 0.0
    traverse_mid_y = -0.05
    traverse_far_x = -1.0

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

    q_robot[10:26] = open_hand(1.0).to(q_robot.device)
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
    # Step 6: Keep the base still and move the arm onto the door panel center
    # -------------------------
    palm_target_pos = palm_target_pose[:, :3].clone()
    palm_target_pos[:, 0] -= 0.60
    palm_target_pos[:, 1] += 0.40
    palm_target_pose = _make_pose(palm_target_pos, block_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[22:26] = block_finger_joint_q
    q_door = torch.tensor([1.2, 0.0], device=device)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    q_door = torch.tensor([1.4, 0.0], device=device)
    board_pos = get_board_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    palm_target_pos = board_pos.clone()
    palm_target_pos[:, 1] += 0.10
    palm_target_pose = _make_pose(palm_target_pos, block_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=None,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[22:26] = block_finger_joint_q

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 7: Traverse through the door while returning the base yaw to zero
    # -------------------------
    prev_q = q_robot[3:10].clone()

    base_target_pos[:, 0] = traverse_mid_x
    base_target_pos[:, 1] = traverse_mid_y
    base_target_pose = _make_pose(base_target_pos, base_target_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=None,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[3:10] = prev_q
    q_robot[22:26] = block_finger_joint_q
    q_door = torch.tensor([1.5, 0.0], device=device)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # Release after the base has half traversed the doorway.
    q_robot[3:10] = robot_initial_q[3:10]
    q_robot[10:26] = open_hand(1.0).to(q_robot.device)
    q_door = torch.tensor([1.2, 0.0], device=device)

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
    )

    # -------------------------
    # Step 8: Finish traversing the doorway
    # -------------------------
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
    q_robot[3:10] = robot_initial_q[3:10]
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
    # Step 1: Pregrasp
    # -------------------------
    handle_pos = get_hinge_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    pregrasp_base_x_offset = 0.55
    pregrasp_base_y_offset = 0.3
    pregrasp_palm_x_offset = 0.25
    pregrasp_palm_y_offset = 0.03
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
    grasp_palm_x_offset = 0.04
    grasp_palm_y_offset = 0.02
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
    unlatch_palm_y_delta = 0.02
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
    release_base_y = -0.25
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
    push_contact_x_offset = -0.3
    push_contact_y_offset = -0.2
    push_contact_z_offset = 0.1
    contact_virtual_door_angle = 1.0
    push_door_open_angle = 1.5

    franka_default_q = torch.tensor(
        [FRANKA_DEFAULT_JOINT_POS[name] for name in FRANKA_JOINT_NAMES],
        device=device,
        dtype=q_robot.dtype,
    )
    traverse_mid_x = 0.45
    traverse_mid_y = -0.05
    traverse_far_x = -0.5

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
    contact_board_pos = get_board_edge(
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
    board_pos = get_board_edge(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0),
    ).to(device)

    palm_target_pos = board_pos.clone()
    # palm_target_pos[:, 0] += push_contact_x_offset
    # palm_target_pos[:, 1] += push_contact_y_offset
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
    base_target_pos[:, 0] = traverse_mid_x
    base_target_pos[:, 1] = traverse_mid_y
    base_target_pose = _make_pose(base_target_pos, base_target_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    # q_robot[3:10] = retreat_arm_q
    q_robot[10:26] = safe_open_hand_q

    _append_state(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        q_robot,
        q_door,
        mark_keyframe=True,
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
