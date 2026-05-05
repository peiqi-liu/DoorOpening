import argparse
import math
from DoorOpening.utils.state_machine.api import compute_base_joint, solve_ik, get_hinge_pos, open_hand
import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_from_matrix, combine_frame_transforms, quat_mul, quat_inv
from DoorOpening.constants.robot_constants import FULL_JOINT_NAMES, CAMERA_JOINT_DEFAULT_VALUES, DEFAULT_JOINT_POS, OPEN_FINGER_JOINT_VALUES, ROBOT_KEY_BODY_NAMES, DM_JOINT_NAMES
from DoorOpening.constants.door_constants import DOOR_BODY_NAMES, DOOR_JOINT_NAMES
import numpy as np
import time
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT, DOOR_INITIAL_POS, DOOR_INITIAL_ROT
import pickle as pkl
import os
from scipy.interpolate import CubicSpline
import random
import viser
from viser.extras import ViserUrdf

from yourdfpy import URDF
from DoorOpening.utils.state_machine.pin import PinocchioIKSolver
from DoorOpening.utils.state_machine.offline_pull_door import state_machine_offline_pull_door
import glob


def get_robot_constants():
    all_joint_names = FULL_JOINT_NAMES + list(CAMERA_JOINT_DEFAULT_VALUES.keys())
    default_joint_pos_dict = {}
    joint_vector = torch.zeros(len(all_joint_names))
    for i, name in enumerate(all_joint_names):
        if name in DEFAULT_JOINT_POS:
            default_joint_pos_dict[name] = DEFAULT_JOINT_POS[name]
            joint_vector[i] = DEFAULT_JOINT_POS[name]
        elif name in OPEN_FINGER_JOINT_VALUES:
            default_joint_pos_dict[name] = OPEN_FINGER_JOINT_VALUES[name]
            joint_vector[i] = OPEN_FINGER_JOINT_VALUES[name]
        else:
            default_joint_pos_dict[name] = 0.0
            joint_vector[i] = 0.0
    return default_joint_pos_dict, joint_vector
    

def get_rotation_quat(roll, pitch, yaw, device):
    # yaw z
    # roll x
    # pitch y
    return quat_from_euler_xyz(roll = torch.tensor([[roll]]).to(device), pitch = torch.tensor([[pitch]]).to(device), yaw = torch.tensor([[yaw]]).to(device)).squeeze(0)

import torch


def sample_robot_initial_base_joints_on_door_ring(
    robot_initial_pose: torch.Tensor,
    door_initial_pose: torch.Tensor,
    *,
    radius: float = 1.2,
    angle_range_deg: float = 25.0,
):
    """Sample base xy joints for a start pose on a ring around the door while
    keeping the base rotation joint unchanged."""

    angle_limit = math.radians(angle_range_deg)
    approach_angle = random.uniform(-angle_limit, angle_limit)

    sampled_x = door_initial_pose[:, 0] + radius * math.cos(approach_angle)
    sampled_y = door_initial_pose[:, 1] + radius * math.sin(approach_angle)

    sampled_pose = robot_initial_pose.clone()
    sampled_pose[:, 0] = sampled_x
    sampled_pose[:, 1] = sampled_y

    sampled_base_joint = compute_base_joint(
        robot_initial_pose[:, :3],
        robot_initial_pose[:, 3:],
        sampled_pose[:, :3],
    ).squeeze(0)

    return sampled_base_joint, sampled_pose, approach_angle

def state_machine_offline(
    robot_urdf_path,
    door_urdf_path,
    robot_initial_pose,   # (1, 7) world
    door_initial_pose,    # (1, 7) world
    robot_initial_q,      # (ndof,)
    door_initial_q,       # (2,) [board, hinge]
    device="cpu",
):
    """
    Keypoints:
    # 0. Start
    1. Pregrasp
    2. Actual Grasp
    3. Rotate hinge (unlatch)
    4. Pull door open
    5. Base forward
    6. Move base completely through the door
    7. Move base completely through the door

    So contact with hinge should be between 2 and 4, contact with board should be between 5 and 6.

    Returns:
        robot_traj: list[Tensor(ndof)]
        door_traj:  list[Tensor(2)]
    """

    robot_traj = []
    door_traj = []
    key_idx_in_key_indices = []

    q_robot = robot_initial_q.clone()
    q_door = door_initial_q.clone()

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    # -------------------------
    # Step 1: Pregrasp
    # -------------------------
    handle_pos = get_hinge_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0)
    ).to(device)

    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += 0.55
    base_target_pos[:, 1] -= 0.3
    base_target_rot = torch.tensor([[0, 0, 0, 1]], device=device)
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.25
    palm_target_pos[:, 1] -= 0.1
    palm_target_pos[:, 2] +=0.25
    palm_target_rot = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi, device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 2: Move to grasp
    # -------------------------
    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.05
    palm_target_pos[:, 1] -= 0.1
    palm_target_pos[:, 2] += 0.08
    palm_target_rot = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi, device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    q_robot[10:10+16] = open_hand(0.7)

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 3: Rotate hinge (unlatch)
    # -------------------------
    q_door = torch.tensor([0.0, 1.0], device=device)
    palm_target_pose[:, 2] -= 0.08
    palm_target_pose[:, 1] -= 0.02
    # palm_target_pose[:, 3:] = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi + 1.0, device)
    palm_target_pose[:, 3:] = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi + 0.2, device)
    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 4: Pull door open
    # -------------------------
    for theta in torch.arange(0.25, 1.36, 0.1):
        q_door = torch.tensor([theta, 0], device=device)

        new_handle_pos = get_hinge_pos(
            door_urdf_path,
            door_initial_pose,
            q_door.unsqueeze(0)
        ).to(device)

        base_target_pos = new_handle_pos.clone()
        base_target_pos[:, 0] += 0.55
        base_target_pos[:, 1] = theta / 1.45 * 0.25
        base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

        palm_target_rot = get_rotation_quat(-theta.item() + torch.pi, 0.0 + torch.pi, torch.pi, device)

        new_handle_pos[:, 0] += (0.05 * torch.cos(theta) - 0.1 * torch.sin(theta))
        new_handle_pos[:, 1] -= (0.1 * torch.cos(theta) + 0.05 * torch.sin(theta))
        new_handle_pos[:, 2] += 0.08
        palm_target_pose = torch.cat([new_handle_pos, palm_target_rot], dim=-1)

        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
        )[0]

        robot_traj.append(q_robot.clone())
        door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 5 : Base forward
    # -------------------------

    base_target_pos[:, 0] -= 0.12
    base_target_pos[:, 1] = 0.15
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    palm_target_pose[:, 0] += 0.45
    palm_target_pose[:, 1] += 0.2
    palm_target_pose[:, 3:] = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi, device)

    base_target_pos[:, 0] -= 0.18
    base_target_pos[:, 1] = 0.15
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    q_robot[10:10+16] = open_hand(1.0)
    q_door = torch.tensor([1.35, 0.0], device=device)

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 7: Move base completely through the door
    # -------------------------
    delta_palm_pos = palm_target_pos - base_target_pos
    base_target_pos[:, 0] = 0.0
    base_target_pos[:, 1] = -0.05
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    # palm_pose = base_target_pos.clone()
    # palm_pose[:, :3] += delta_palm_pos[:, :3]
    # palm_pose = torch.cat([palm_pose, base_target_rot], dim=-1)
    prev_q = q_robot[3:10].clone()

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        # palm_pose=palm_pose,
        palm_pose=None,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[3:10] = prev_q
    q_robot[10:10+16] = open_hand(1.0)
    q_door = torch.tensor([1.5, 0.0], device="cpu")

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    base_target_pos[:, 0] = -1.0
    base_target_pos[:, 1] = 0.0
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    # palm_pose = base_target_pos.clone()
    # palm_pose[:, 2] = 0.75
    # palm_pose[:, 0] -= 0.5
    # palm_pose = torch.cat([palm_pose, base_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=None,
        # palm_pose=palm_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    q_robot[3:10] = robot_initial_q[3:10]
    q_door = torch.tensor([0.0, 0.0], device="cpu")

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    return robot_traj, door_traj, key_idx_in_key_indices

def collocate_and_playback(robot_traj, door_traj, key_idx_in_key_indices, length=1000):
    """
    Interpolate trajectory between keyframes using cubic splines
    with segment-wise time allocation proportional to geometric length.
    """

    # ---- Convert to numpy ----
    robot_traj = torch.stack(robot_traj).detach().cpu().numpy()
    door_traj = torch.stack(door_traj).detach().cpu().numpy()
    traj = np.concatenate([robot_traj, door_traj], axis=-1)

    N = len(key_idx_in_key_indices)

    # ---- Compute geometric length of each keyframe segment ----
    seg_lengths = []
    for i in range(N - 1):
        start = key_idx_in_key_indices[i]
        end = key_idx_in_key_indices[i + 1]

        seg = traj[start:end + 1]
        if len(seg) < 2:
            seg_lengths.append(1e-6)
            continue

        dists = np.linalg.norm(seg[1:] - seg[:-1], axis=1)
        seg_lengths.append(max(dists.sum(), 1e-6))

    seg_lengths = np.array(seg_lengths)
    seg_ratios = seg_lengths / seg_lengths.sum()

    # ---- Interpolation ----
    traj_out = []
    traj_d_out = []
    key_indices = [0]
    samples_used = 0

    for i in range(N - 1):

        start = key_idx_in_key_indices[i]
        end = key_idx_in_key_indices[i + 1]
        ps = traj[start:end + 1]

        # allocate samples proportionally
        seg_len = int(np.round(seg_ratios[i] * length))

        # ensure final segment fills remainder
        if i == N - 2:
            seg_len = length - samples_used

        seg_len = max(seg_len, 1)
        samples_used += seg_len
        key_indices.append(key_indices[-1] + seg_len)

        # ---- Chord-length parameterization ----
        if len(ps) == 1:
            # Degenerate case: repeat point
            seg_traj = np.repeat(ps, seg_len, axis=0)
            seg_traj_d = np.zeros_like(seg_traj)
        else:
            dists = np.linalg.norm(ps[1:] - ps[:-1], axis=1)
            t_local = np.concatenate([[0.0], np.cumsum(dists)])
            t_local = t_local / max(t_local[-1], 1e-6)

            cs = CubicSpline(t_local, ps, axis=0, bc_type="clamped")

            t_samples = np.linspace(
                0.0, 1.0,
                seg_len,
                endpoint=(i == N - 2)
            )

            seg_traj = cs(t_samples)
            seg_traj_d = cs(t_samples, 1)

        traj_out.append(seg_traj)
        traj_d_out.append(seg_traj_d)

    # ---- Concatenate segments ----
    traj_interp = np.concatenate(traj_out, axis=0)
    traj_d_interp = np.concatenate(traj_d_out, axis=0)

    traj_interp = torch.tensor(traj_interp, dtype=torch.float32)
    traj_d_interp = torch.tensor(traj_d_interp, dtype=torch.float32)

    # ---- Split robot / door ----
    robot_traj = traj_interp[:, :-2]
    door_traj = traj_interp[:, -2:]
    robot_traj_d = traj_d_interp[:, :-2]
    door_traj_d = traj_d_interp[:, -2:]

    # clamp door values
    door_traj[:, 0] = door_traj[:, 0].clamp(min=0.0, max=1.5)
    door_traj[:, 1] = door_traj[:, 1].clamp(min=0.0, max=1.0)

    return robot_traj, door_traj, robot_traj_d, door_traj_d, key_indices


def play_trajectories_in_viser(
    robot_urdf,
    door_urdf,
    robot_traj: list[torch.Tensor] | torch.Tensor,   # (T, Ndof)
    door_traj: list[torch.Tensor] | torch.Tensor,    # (T, 2)
    robot_world_pos: torch.Tensor,                  # (T, 3)
    door_world_pos: torch.Tensor,                   # (T, 3)
    robot_world_quat: torch.Tensor,                  # (T, 4)
    door_world_quat: torch.Tensor,                   # (T, 4)
    hz: float = 60.0,
):
    """
    robot_traj: (T, Ndof) torch.Tensor or list of tensors
    door_traj:  (T, 2)    torch.Tensor or list of tensors
    sample_pc_fn(q_door_np) -> (N, 3) numpy array in world frame (optional)
    """

    # Normalize input format
    if isinstance(robot_traj, list):
        robot_traj = torch.stack(robot_traj, dim=0)
    if isinstance(door_traj, list):
        door_traj = torch.stack(door_traj, dim=0)

    T = min(len(robot_traj), len(door_traj))

    # -------------------------
    # Start viser
    # -------------------------
    server = viser.ViserServer()

    server.scene.add_frame(
        "/robot_root",
        position=robot_world_pos,
        wxyz=robot_world_quat,
    )

    server.scene.add_frame(
        "/door_root",
        position=door_world_pos,
        wxyz=door_world_quat,
    )

    # server.scene.add_frame(
    #     "/palm_center",
    #     wxyz=(1.0, 0.0, 0.0, 0.0),
    #     position=robot_world_pos,
    # )

    viser_robot = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot_root", load_meshes=True)
    viser_door  = ViserUrdf(server, urdf_or_path=door_urdf, root_node_name="/door_root", load_meshes=True)

    # -------------------------
    # GUI controls
    # -------------------------
    playing = True
    speed = 1.0

    with server.gui.add_folder("Playback"):
        play_btn = server.gui.add_button("Play")
        pause_btn = server.gui.add_button("Pause")
        reset_btn = server.gui.add_button("Reset")
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    @play_btn.on_click
    def _(_):
        nonlocal playing
        playing = True

    @pause_btn.on_click
    def _(_):
        nonlocal playing
        playing = False

    @reset_btn.on_click
    def _(_):
        nonlocal t_idx
        t_idx = 0

    # -------------------------
    # Playback loop
    # -------------------------
    t_idx = 0

    print("Viser running. Open the URL in your browser.")

    # timestamp = time.time()

    # while time.time() - timestamp < 60:
    while True:
        if playing:
            q_robot = robot_traj[t_idx].detach().cpu().numpy()
            q_door  = door_traj[t_idx].detach().cpu().numpy()


            # hinge_pos = get_hinge_pos(
            #     door_urdf_path,
            #     door_initial_pose,
            #     torch.tensor(q_door)
            # ).squeeze().detach().cpu().numpy()
            # fk_pos, fk_quat = ik_solver.compute_fk(q_robot[:10])
            # fk_pos, fk_quat = base_to_world_frame(torch.tensor(robot_world_pos).to(torch.float32), torch.tensor(robot_world_quat).to(torch.float32), torch.tensor(fk_pos).to(torch.float32), torch.tensor(fk_quat).to(torch.float32))
            # server.scene.add_frame(
            #     "/hinge",
            #     wxyz=(1.0, 0.0, 0.0, 0.0),
            #     position=hinge_pos,
            # )
            # server.scene.add_frame(
            #     "/fk",
            #     wxyz=fk_quat[[3, 0, 1, 2]],
            #     position=fk_pos,
            # )

            viser_robot.update_cfg(q_robot)
            viser_door.update_cfg(q_door)

            # t_idx = (t_idx + 1) % T
            t_idx += 1
            if t_idx == T:
                break

        time.sleep(1.0 / (hz * speed_slider.value))
    server.stop()

def compute_link_twist(pos_traj: torch.Tensor,
                       quat_traj: torch.Tensor) -> torch.Tensor:
    """
    Compute per-timestep link twist from position and quaternion trajectory in world frame.

    Args:
        pos_traj:  (T, B, 3) positions
        quat_traj: (T, B, 4) quaternions in (w, x, y, z)

    Returns:
        twist: (T, B, 6) tensor
               [vx, vy, vz, wx, wy, wz] per timestep
    """

    assert pos_traj.shape[:2] == quat_traj.shape[:2]
    assert quat_traj.shape[-1] == 4

    # ---------- Linear velocity (per step) ----------
    linear_vel = pos_traj[1:] - pos_traj[:-1]  # (T-1, B, 3)

    # ---------- Angular velocity ----------
    q_t = quat_traj[:-1]      # (T-1, B, 4)
    q_next = quat_traj[1:]    # (T-1, B, 4)

    # Fix quaternion sign discontinuity
    sign = torch.sign((q_t * q_next).sum(dim=-1, keepdim=True))
    q_next = q_next * sign

    # Relative rotation
    q_rel = quat_mul(quat_inv(q_t), q_next)

    # Small-angle approximation:
    # rotation vector ≈ 2 * imaginary part
    angular_vel = 2.0 * q_rel[..., 1:]  # (T-1, B, 3)

    # ---------- Combine ----------
    twist = torch.cat([linear_vel, angular_vel], dim=-1)  # (T-1, B, 6)

    # Pad last step to keep same length T
    twist = torch.cat([twist, twist[-1:].clone()], dim=0)

    return twist


def play_and_save_traj(
    robot_urdf_path,
    door_urdf_path,
    handle_side="right",
    randomize_start_base=True,
    start_base_radius=1.0,
    start_base_angle_range_deg=30.0,
):
    dir_path = os.path.dirname(door_urdf_path)
    robot_initial_pose = torch.tensor([[ROBOT_INITIAL_POS[0], ROBOT_INITIAL_POS[1], ROBOT_INITIAL_POS[2], ROBOT_INITIAL_ROT[0], ROBOT_INITIAL_ROT[1], ROBOT_INITIAL_ROT[2], ROBOT_INITIAL_ROT[3]]], device="cpu")
    door_initial_pose = torch.tensor([[DOOR_INITIAL_POS[0], DOOR_INITIAL_POS[1], DOOR_INITIAL_POS[2], DOOR_INITIAL_ROT[0], DOOR_INITIAL_ROT[1], DOOR_INITIAL_ROT[2], DOOR_INITIAL_ROT[3]]], device="cpu")
    robot_constants, robot_initial_q = get_robot_constants()
    if randomize_start_base:
        sampled_base_joint, sampled_world_pose, sampled_angle = sample_robot_initial_base_joints_on_door_ring(
            robot_initial_pose,
            door_initial_pose,
            radius=start_base_radius,
            angle_range_deg=start_base_angle_range_deg,
        )
        robot_initial_q[:3] = sampled_base_joint
        print(
            "Randomized start base joints:",
            robot_initial_q[:3],
            "world pose:",
            sampled_world_pose,
            f"(radius={start_base_radius:.2f} m, angle={math.degrees(sampled_angle):.1f} deg)",
        )
    door_initial_q = torch.tensor([0.0, 0.0], device="cpu")
    start_time = time.time()
    robot_traj, door_traj, key_idx_in_key_indices = state_machine_offline_pull_door(
        robot_urdf_path,
        door_urdf_path,
        robot_initial_pose,
        door_initial_pose,
        robot_initial_q,
        door_initial_q,
        handle_side=handle_side,
        device="cpu",
    )
    print(f"Time taken: {time.time() - start_time} seconds")
    torch.set_printoptions(precision=4, sci_mode=False)
    # new_robot_traj = []
    # new_door_traj = []
    # for i, (robot_point, door_point) in enumerate(zip(robot_traj, door_traj)):
    #     if i in key_idx_in_key_indices:
    #         for _ in range(100):
    #             new_robot_traj.append(robot_point)
    #             new_door_traj.append(door_point)
    # robot_traj = new_robot_traj
    # door_traj = new_door_traj
    robot_traj, door_traj, robot_traj_d, door_traj_d, key_indices = collocate_and_playback(robot_traj, door_traj, key_idx_in_key_indices, length=1000)
    print(robot_traj.shape)
    print(door_traj.shape)
    print(robot_traj_d.shape)
    print(door_traj_d.shape)

    robot_urdf = URDF.load(robot_urdf_path)
    door_urdf  = URDF.load(door_urdf_path)

    robot_world_pos = robot_initial_pose[:, :3].squeeze(0).numpy()
    door_world_pos = door_initial_pose[:, :3].squeeze(0).numpy()
    robot_world_quat = robot_initial_pose[:, 3:].squeeze(0).numpy()
    door_world_quat = door_initial_pose[:, 3:].squeeze(0).numpy()

    play_trajectories_in_viser(
        robot_urdf=robot_urdf,
        door_urdf=door_urdf,
        robot_traj=robot_traj,
        door_traj=door_traj,
        robot_world_pos=robot_world_pos,
        door_world_pos=door_world_pos,
        robot_world_quat=robot_world_quat,
        door_world_quat=door_world_quat,
        hz=60,
    )

    robot_ik_solver = PinocchioIKSolver(
        urdf_path=robot_urdf_path, 
        ee_link_name="palm_center", 
        controlled_joints=DM_JOINT_NAMES
    ) 

    robot_key_bodies = ROBOT_KEY_BODY_NAMES
    robot_body_pos_traj = []
    robot_body_quat_traj = []

    for robot_point in robot_traj:
        body_poses = []
        body_quats = []
        for node_a in robot_key_bodies:
            transform = robot_ik_solver.get_frame_pose(config = robot_point[:10], node_b = "base_link", node_a = node_a)
            translation, rotation = torch.tensor(transform.translation).unsqueeze(0).float(), torch.tensor(transform.rotation).unsqueeze(0).float()
            quat = quat_from_matrix(rotation)
            body_world_pos, body_world_quat = combine_frame_transforms(t01 = torch.tensor(robot_world_pos).unsqueeze(0).float(), q01 = torch.tensor(robot_world_quat).unsqueeze(0).float(), t12 = translation, q12 = quat)
            body_poses.append(body_world_pos.squeeze())
            body_quats.append(body_world_quat.squeeze())
        robot_body_pos_traj.append(torch.stack(body_poses, dim=0))
        robot_body_quat_traj.append(torch.stack(body_quats, dim=0))

    robot_body_pos_traj = torch.stack(robot_body_pos_traj, dim=0)
    robot_body_quat_traj = torch.stack(robot_body_quat_traj, dim=0)

    # translation = get_hinge_pos(door_urdf_path, door_initial_pose, door_traj)
    # body_world_pos, _ = combine_frame_transforms(t01 = torch.tensor(door_world_pos).unsqueeze(0).float(), q01 = torch.tensor(door_world_quat).unsqueeze(0).float(), t12 = translation, q12 = None)
    # door_body_pos_traj = body_world_pos.squeeze()
    door_body_pos_traj = get_hinge_pos(door_urdf_path, door_initial_pose, door_traj)

    # door_body_pos_traj = torch.stack(door_body_pos_traj, dim=0)

    print("robot_body_pos_traj.shape: ", robot_body_pos_traj.shape)
    print("robot_body_quat_traj.shape: ", robot_body_quat_traj.shape)
    print("door_body_pos_traj.shape: ", door_body_pos_traj.shape)

    robot_body_pos_twist = compute_link_twist(robot_body_pos_traj, robot_body_quat_traj)

    # print(robot_key_bodies)
    # for i in torch.arange(0, robot_body_pos_traj.shape[0], 100):
    #     print(i, robot_body_pos_traj[i], robot_body_quat_traj[i])

    mask = torch.zeros(len(robot_traj), dtype=torch.int8)
    # Contact with hinge should happen between keyframe 2 and 4
    mask[key_indices[1]:key_indices[3]] = 1

    data = {
        "handle_side": handle_side,
        "door_traj": door_traj, 
        "robot_body_pos_traj": robot_body_pos_traj,
        "robot_body_quat_traj": robot_body_quat_traj,
        "robot_joint_pos_traj": robot_traj,
        "robot_joint_vel_traj": robot_traj_d,
        # "key_indices": torch.tensor(key_indices, dtype=torch.int32)[key_idx_in_key_indices]
        "hinge_contact_mask": mask,
        "key_indices": key_indices,
        "robot_body_pos_twist": robot_body_pos_twist,
        "door_body_pos_traj": door_body_pos_traj
    }
    print(key_indices)
    # print(torch.tensor(key_indices, dtype=torch.int32)[key_idx_in_key_indices])
    traj_file = "traj.pkl"
    traj_path = os.path.join(dir_path, traj_file)
    with open(traj_path, "wb") as f:
        pkl.dump(data, f)
        print("Trajectory saved to " + traj_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save door waypoint trajectories.")
    parser.add_argument(
        "--robot-urdf-path",
        default="source/DoorOpening/assets/glorbot/glorbot.urdf",
        help="Path to the robot URDF used for offline IK and playback.",
    )
    parser.add_argument(
        "--asset-base-folder",
        default="source/DoorOpening/assets/door/PartNetv6",
        help="Folder to scan recursively for door mobility.urdf files.",
    )
    parser.add_argument(
        "--door-urdf-path",
        default=None,
        help="Optional single door URDF path. If set, this overrides --asset-base-folder.",
    )
    parser.add_argument(
        "--handle-side",
        default="right",
        choices=["right", "left"],
        help="Select the pull-door planner variant. 'right' keeps the legacy path; 'left' uses the mirrored planner.",
    )
    args = parser.parse_args()

    robot_urdf_path = args.robot_urdf_path
    if args.door_urdf_path is not None:
        asset_paths = [args.door_urdf_path]
    else:
        asset_base_folder = args.asset_base_folder
        asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.urdf"), recursive=True), reverse=False)

    for i, door_urdf_path in enumerate(asset_paths):
        play_and_save_traj(robot_urdf_path, door_urdf_path, handle_side=args.handle_side)
        print("Finished processing ", door_urdf_path, ", index: ", i)
    
