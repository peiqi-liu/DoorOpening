import argparse
import hashlib
import json
import math
from DoorOpening.utils.state_machine.api import compute_base_joint, solve_ik, get_hinge_pos, open_hand
import torch
from pathlib import Path
from isaaclab.utils.math import quat_from_euler_xyz, quat_from_matrix, combine_frame_transforms, quat_mul, quat_inv
from DoorOpening.constants.robot_constants import FULL_JOINT_NAMES, DEFAULT_JOINT_POS, OPEN_FINGER_JOINT_VALUES, ROBOT_KEY_BODY_NAMES, DM_JOINT_NAMES, FRANKA_JOINT_NAMES
import numpy as np
import time
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT, DOOR_INITIAL_POS, DOOR_INITIAL_ROT
import pickle as pkl
import os
from scipy.interpolate import CubicSpline
import random

from DoorOpening.utils.state_machine.pin import PinocchioIKSolver
from DoorOpening.utils.state_machine.offline_pull_door import state_machine_offline_pull_door
from DoorOpening.utils.state_machine.offline_push_door import state_machine_offline_push_door
import glob


DEFAULT_TRAJ_DT = 1.0 / 40.0


def _load_variant_metadata(door_urdf_path):
    meta_path = os.path.join(os.path.dirname(door_urdf_path), "variant_meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, encoding="utf-8") as meta_file:
        return json.load(meta_file)


def _load_initial_state_metadata(door_urdf_path):
    metadata = _load_variant_metadata(door_urdf_path)
    if not isinstance(metadata, dict):
        return None
    initial_state = metadata.get("initial_state")
    if not isinstance(initial_state, dict):
        return None
    return initial_state


def _parse_pose_tensor(pose_meta, default_tensor):
    if not isinstance(pose_meta, dict):
        return default_tensor
    pos = pose_meta.get("pos")
    rot = pose_meta.get("rot")
    if not (
        isinstance(pos, (list, tuple))
        and len(pos) == 3
        and isinstance(rot, (list, tuple))
        and len(rot) == 4
    ):
        return default_tensor
    values = [float(v) for v in list(pos) + list(rot)]
    return torch.tensor([values], dtype=default_tensor.dtype, device=default_tensor.device)


def _parse_joint_tensor(values, default_tensor, expected_len):
    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        return default_tensor
    return torch.tensor([float(v) for v in values], dtype=default_tensor.dtype, device=default_tensor.device)


def _apply_named_joint_positions(target_tensor, joint_names, joint_values):
    if not isinstance(joint_names, (list, tuple)) or not isinstance(joint_values, (list, tuple)):
        return target_tensor, False
    if len(joint_names) != len(joint_values):
        return target_tensor, False

    joint_index_by_name = {name: idx for idx, name in enumerate(FULL_JOINT_NAMES)}
    target_tensor = target_tensor.clone()
    applied_any = False

    for joint_name, joint_value in zip(joint_names, joint_values):
        joint_idx = joint_index_by_name.get(str(joint_name))
        if joint_idx is None:
            continue
        target_tensor[joint_idx] = float(joint_value)
        applied_any = True

    return target_tensor, applied_any


def _apply_initial_state_metadata(door_urdf_path, robot_initial_pose, door_initial_pose, robot_initial_q, door_initial_q):
    initial_state = _load_initial_state_metadata(door_urdf_path)
    if initial_state is None:
        return robot_initial_pose, door_initial_pose, robot_initial_q, door_initial_q, False

    robot_initial_pose = _parse_pose_tensor(initial_state.get("robot_world_pose"), robot_initial_pose)
    door_initial_pose = _parse_pose_tensor(initial_state.get("door_world_pose"), door_initial_pose)

    robot_joint_pos = initial_state.get("robot_joint_pos")
    if isinstance(robot_joint_pos, (list, tuple)) and len(robot_joint_pos) == len(robot_initial_q):
        robot_initial_q = _parse_joint_tensor(robot_joint_pos, robot_initial_q, len(robot_initial_q))
    else:
        robot_joint_names = initial_state.get("robot_joint_names")
        robot_initial_q, applied_robot_joint_pos = _apply_named_joint_positions(
            robot_initial_q,
            robot_joint_names,
            robot_joint_pos,
        )
        if not applied_robot_joint_pos and isinstance(robot_joint_pos, (list, tuple)) and len(robot_joint_pos) == len(DM_JOINT_NAMES):
            robot_initial_q, _ = _apply_named_joint_positions(
                robot_initial_q,
                DM_JOINT_NAMES,
                robot_joint_pos,
            )

        robot_base_joint_pos = initial_state.get("robot_base_joint_pos")
        if isinstance(robot_base_joint_pos, (list, tuple)) and len(robot_base_joint_pos) >= 3:
            robot_initial_q = robot_initial_q.clone()
            robot_initial_q[:3] = _parse_joint_tensor(robot_base_joint_pos[:3], robot_initial_q[:3], 3)

    robot_franka_joint_pos = initial_state.get("robot_franka_joint_pos")
    robot_franka_joint_names = initial_state.get("robot_franka_joint_names")
    robot_initial_q, applied_franka_joint_pos = _apply_named_joint_positions(
        robot_initial_q,
        robot_franka_joint_names,
        robot_franka_joint_pos,
    )
    if not applied_franka_joint_pos and isinstance(robot_franka_joint_pos, (list, tuple)) and len(robot_franka_joint_pos) == len(FRANKA_JOINT_NAMES):
        robot_initial_q, _ = _apply_named_joint_positions(
            robot_initial_q,
            FRANKA_JOINT_NAMES,
            robot_franka_joint_pos,
        )

    door_joint_pos = initial_state.get("door_joint_pos")
    if isinstance(door_joint_pos, (list, tuple)) and len(door_joint_pos) == len(door_initial_q):
        door_initial_q = _parse_joint_tensor(door_joint_pos, door_initial_q, len(door_initial_q))

    return robot_initial_pose, door_initial_pose, robot_initial_q, door_initial_q, True


def _metadata_handle_side_to_planner_side(handle_side):
    if handle_side == "max":
        return "right"
    if handle_side == "min":
        return "left"
    return None


def resolve_planner_options(door_urdf_path, handle_side, opening_direction):
    metadata = _load_variant_metadata(door_urdf_path)

    resolved_opening_direction = opening_direction
    if resolved_opening_direction == "auto":
        resolved_opening_direction = "pull"
        if metadata is not None:
            resolved_opening_direction = metadata.get("target_properties", {}).get(
                "opening_direction",
                resolved_opening_direction,
            )

    resolved_handle_side = handle_side
    if resolved_handle_side == "auto":
        resolved_handle_side = "right"
        if metadata is not None:
            metadata_handle_side = metadata.get("actual_properties", {}).get("handle_side")
            resolved_handle_side = (
                _metadata_handle_side_to_planner_side(metadata_handle_side)
                or resolved_handle_side
            )

    if resolved_opening_direction not in {"pull", "push"}:
        raise ValueError(
            f"Unsupported opening_direction '{resolved_opening_direction}'. Expected 'pull' or 'push'."
        )
    if resolved_handle_side not in {"right", "left"}:
        raise ValueError(f"Unsupported handle_side '{resolved_handle_side}'. Expected 'right' or 'left'.")

    return resolved_handle_side, resolved_opening_direction


def resolve_start_base_pair_key(door_urdf_path):
    metadata = _load_variant_metadata(door_urdf_path)
    if metadata is None:
        return os.path.basename(os.path.dirname(door_urdf_path))

    copied_from_pull_asset = metadata.get("copied_from_pull_asset")
    if copied_from_pull_asset:
        return os.path.basename(os.path.normpath(str(copied_from_pull_asset)))

    direction_pair_key = metadata.get("direction_pair_key")
    if direction_pair_key:
        return str(direction_pair_key)

    variant_name = metadata.get("variant_name")
    if variant_name:
        return str(variant_name)

    return os.path.basename(os.path.dirname(door_urdf_path))


def _stable_unit_interval(key):
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)


def paired_start_angle(pair_key, seed, angle_range_deg):
    angle_limit = math.radians(angle_range_deg)
    unit = _stable_unit_interval(f"dooropening-start-base-v1:{int(seed)}:{pair_key}")
    return -angle_limit + 2.0 * angle_limit * unit


def get_robot_constants():
    # FULL_JOINT_NAMES already contains base, arm, finger, and camera joints.
    all_joint_names = FULL_JOINT_NAMES
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
    rng=None,
    approach_angle: float | None = None,
):
    """Sample base xy joints for a start pose on a ring around the door while
    keeping the base rotation joint unchanged."""

    angle_limit = math.radians(angle_range_deg)
    if approach_angle is None:
        rng = random if rng is None else rng
        approach_angle = rng.uniform(-angle_limit, angle_limit)
    else:
        approach_angle = max(-angle_limit, min(angle_limit, float(approach_angle)))

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
    def weighted_dist(x1, x2, base_weight=2.5):
        diff = x1 - x2
        diff = diff.copy()

        # first 3 joints / base dofs are weighted more heavily
        diff[..., :3] *= base_weight

        return np.linalg.norm(diff, axis=-1)
    
    seg_lengths = []
    for i in range(N - 1):
        start = key_idx_in_key_indices[i]
        end = key_idx_in_key_indices[i + 1]

        seg = traj[start:end + 1]
        if len(seg) < 2:
            seg_lengths.append(1e-6)
            continue

        # dists = np.linalg.norm(seg[1:] - seg[:-1], axis=1)
        dists = weighted_dist(seg[1:], seg[:-1])
        seg_lengths.append(max(dists.sum(), 1e-6))

    seg_lengths = np.array(seg_lengths)
    seg_ratios = seg_lengths / seg_lengths.sum()

    # ---- Interpolation ----
    traj_out = []
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
        # Clamp to a VALID index. These are cumulative sample counts, so the last one totals
        # `length` -- one past the end of a 0..length-1 array. Anything indexing the returned
        # trajectory by its own last keyframe (renderers, keyframe-triggered logic) was reading
        # out of range or silently dropping that frame.
        key_indices.append(min(key_indices[-1] + seg_len, length - 1))

        # ---- Chord-length parameterization ----
        # Drop consecutive duplicate waypoints first: the offline IK can return the same config
        # for two adjacent waypoints (e.g. an unreachable/saturated target that best-errors onto
        # its neighbour's pose), which makes the chord-length parameter non-strictly-increasing
        # and crashes CubicSpline ("x must be strictly increasing").
        if len(ps) > 1:
            seg_dists = np.linalg.norm(ps[1:] - ps[:-1], axis=1)
            keep = np.concatenate([[True], seg_dists > 1e-9])
            ps = ps[keep]

        if len(ps) < 2:
            # Degenerate case: single (or all-duplicate) point -> repeat it.
            seg_traj = np.repeat(ps[:1], seg_len, axis=0)
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

        traj_out.append(seg_traj)

    # ---- Concatenate segments ----
    traj_interp = np.concatenate(traj_out, axis=0)

    traj_interp = torch.tensor(traj_interp, dtype=torch.float32)

    # ---- Split robot / door ----
    robot_traj = traj_interp[:, :-2]
    door_traj = traj_interp[:, -2:]

    # clamp door values
    door_traj[:, 0] = door_traj[:, 0].clamp(min=0.0, max=1.5)
    door_traj[:, 1] = door_traj[:, 1].clamp(min=0.0, max=1.0)
    # MotionLib recomputes reference velocities from positions and sim_dt on load.
    robot_traj_d = torch.zeros_like(robot_traj)
    door_traj_d = torch.zeros_like(door_traj)

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
    playback_speed: float = 10.0,
    port: int | None = None,
    capture_dir: str | None = None,
    capture_every: int = 8,
    capture_size: tuple[int, int] = (720, 1100),
    capture_timeout: float = 420.0,
):
    """
    robot_traj: (T, Ndof) torch.Tensor or list of tensors
    door_traj:  (T, 2)    torch.Tensor or list of tensors
    sample_pc_fn(q_door_np) -> (N, 3) numpy array in world frame (optional)
    """
    import viser
    from viser.extras import ViserUrdf

    # Normalize input format
    if isinstance(robot_traj, list):
        robot_traj = torch.stack(robot_traj, dim=0)
    if isinstance(door_traj, list):
        door_traj = torch.stack(door_traj, dim=0)

    T = min(len(robot_traj), len(door_traj))

    # -------------------------
    # Start viser
    # -------------------------
    server_kwargs = {}
    if port is not None:
        server_kwargs["port"] = port
    server = viser.ViserServer(**server_kwargs)

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

    # --- offscreen capture -------------------------------------------------------------------
    # client.get_render() renders THE REAL SCENE (actual URDF meshes, not a stick figure) and
    # returns it as an array, so playback can be dumped to PNGs and inspected without watching
    # the browser. It needs a connected client to render from -- any browser pointed at the
    # server's URL will do, including a headless one.
    # Fixed capture viewpoint: an iso view parked on the robot's side of the door, far enough back
    # to hold BOTH the door and the whole base sweep (base starts ~1 m out at +x and traverses to
    # traverse_far_x = -1 m) for every frame of the trajectory.
    capture_camera_pos = (3.6, -3.0, 2.4)
    capture_camera_look_at = (0.2, 0.0, 1.0)

    def _pin_camera_and_render(client, size):
        """Pin the capture viewpoint, then render -- both inside the worker thread.

        The camera MUST be re-pinned every frame: get_render draws from the client's current
        camera, and viser re-fits that camera as the scene bounds change (the door swinging open
        moves them a lot), so a once-at-connect camera silently drifts and the captured frames
        come out at different zooms and angles -- useless for comparing one keyframe to the next.

        It must also happen HERE rather than on the playback thread. These setters round-trip to
        the browser, and a throttled/backgrounded tab may never answer; on the playback thread
        that hangs the whole run with no timeout and no error (a hang is not an exception, so the
        caller's try/except cannot see it). Inside the worker it is covered by fut.result(timeout).
        """
        client.camera.position = capture_camera_pos
        client.camera.look_at = capture_camera_look_at
        return client.get_render(height=size[0], width=size[1], transport_format="png")

    # Capture is enabled by the FLAG, not by whether a browser happened to be attached in time.
    # The wait below is only there to give the browser a head start and let meshes load; the
    # playback loop already re-checks server.get_clients() every frame and skips cleanly when
    # there is none. Latching this off after a timeout meant a browser that connected a moment
    # late produced a full run with zero frames and no way to recover short of re-running.
    capture_enabled = capture_dir is not None
    if capture_dir is not None:
        import imageio.v3 as iio
        from concurrent.futures import ThreadPoolExecutor
        _cap_pool = ThreadPoolExecutor(max_workers=2)
        Path(capture_dir).mkdir(parents=True, exist_ok=True)
        print(f"[capture] waiting up to {capture_timeout:.0f}s for a browser to connect ...", flush=True)
        deadline = time.time() + capture_timeout
        connected = False
        while time.time() < deadline:
            if server.get_clients():
                connected = True
                break
            time.sleep(0.5)
        if not connected:
            print("[capture] no browser yet; capturing anyway, frames land once one connects.", flush=True)
        else:
            print(f"[capture] client connected, writing frames to {capture_dir}", flush=True)
            # Frame BOTH the door and the robot. The client's default camera looks at the origin
            # from close in, which puts the door in shot but leaves the robot (base starts ~1 m
            # out at +x, and traverses to -1 m by the end) off the edge -- captured frames were
            # useless for judging hand-vs-panel geometry. Park an iso view on the robot's side of
            # the door, far enough back to hold the whole base sweep.
            for _cam_client in server.get_clients().values():
                _cam_client.camera.position = capture_camera_pos
                _cam_client.camera.look_at = capture_camera_look_at
            time.sleep(3.0)  # let meshes finish loading before the first frame

    # -------------------------
    # GUI controls
    # -------------------------
    playing = True

    with server.gui.add_folder("Playback"):
        play_btn = server.gui.add_button("Play")
        pause_btn = server.gui.add_button("Pause")
        reset_btn = server.gui.add_button("Reset")
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.5,
            max=playback_speed,
            step=0.1,
            initial_value=playback_speed,
        )

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

            if capture_enabled and t_idx % capture_every == 0:
                # Re-pick a LIVE client every time. Holding one handle blocks forever the moment
                # that browser goes away (get_render waits on a reply that never comes), which
                # silently stalls the whole playback.
                live = list(server.get_clients().values())
                if not live:
                    print(f"[capture] frame {t_idx}: no client, skipping", flush=True)
                else:
                    # Hard timeout. A browser tab that loses focus gets throttled by the OS,
                    # its render loop stops, and get_render then blocks forever -- which silently
                    # freezes playback. Skipping a slow frame is always better than stalling.
                    try:
                        fut = _cap_pool.submit(
                            _pin_camera_and_render,
                            live[-1],
                            capture_size,
                        )
                        iio.imwrite(f"{capture_dir}/frame_{t_idx:05d}.png", fut.result(timeout=15.0))
                    except Exception as exc:
                        print(f"[capture] frame {t_idx} skipped ({type(exc).__name__})", flush=True)

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
    handle_side="auto",
    opening_direction="auto",
    randomize_start_base=True,
    start_base_sampling="paired",
    start_base_radius=1.0,
    start_base_angle_range_deg=30.0,
    start_base_pair_seed=0,
    playback_speed=4.0,
    capture_dir=None,
    capture_every=8,
    traj_dt=DEFAULT_TRAJ_DT,
    port=None,
    visualize=False,
):
    dir_path = os.path.dirname(door_urdf_path)
    robot_initial_pose = torch.tensor([[ROBOT_INITIAL_POS[0], ROBOT_INITIAL_POS[1], ROBOT_INITIAL_POS[2], ROBOT_INITIAL_ROT[0], ROBOT_INITIAL_ROT[1], ROBOT_INITIAL_ROT[2], ROBOT_INITIAL_ROT[3]]], device="cpu")
    door_initial_pose = torch.tensor([[DOOR_INITIAL_POS[0], DOOR_INITIAL_POS[1], DOOR_INITIAL_POS[2], DOOR_INITIAL_ROT[0], DOOR_INITIAL_ROT[1], DOOR_INITIAL_ROT[2], DOOR_INITIAL_ROT[3]]], device="cpu")
    robot_constants, robot_initial_q = get_robot_constants()
    door_initial_q = torch.tensor([0.0, 0.0], device="cpu")
    robot_initial_pose, door_initial_pose, robot_initial_q, door_initial_q, loaded_initial_state = _apply_initial_state_metadata(
        door_urdf_path,
        robot_initial_pose,
        door_initial_pose,
        robot_initial_q,
        door_initial_q,
    )
    if loaded_initial_state:
        print(f"Using initial_state from variant_meta.json: {door_urdf_path}")
    else:
        if start_base_sampling == "none":
            randomize_start_base = False
        elif start_base_sampling not in {"paired", "random"}:
            raise ValueError(
                f"Unsupported start_base_sampling '{start_base_sampling}'. Expected paired, random, or none."
            )

        if randomize_start_base:
            approach_angle = None
            if start_base_sampling == "paired":
                pair_key = resolve_start_base_pair_key(door_urdf_path)
                approach_angle = paired_start_angle(
                    pair_key,
                    start_base_pair_seed,
                    start_base_angle_range_deg,
                )

            sampled_base_joint, sampled_world_pose, sampled_angle = sample_robot_initial_base_joints_on_door_ring(
                robot_initial_pose,
                door_initial_pose,
                radius=start_base_radius,
                angle_range_deg=start_base_angle_range_deg,
                approach_angle=approach_angle,
            )
            robot_initial_q[:3] = sampled_base_joint
            sampling_label = "Paired" if start_base_sampling == "paired" else "Randomized"
            print(
                f"{sampling_label} start base joints:",
                robot_initial_q[:3],
                "world pose:",
                sampled_world_pose,
                f"(radius={start_base_radius:.2f} m, angle={math.degrees(sampled_angle):.1f} deg)",
            )

    planner_handle_side, planner_opening_direction = resolve_planner_options(
        door_urdf_path,
        handle_side,
        opening_direction,
    )
    print(
        f"Using {planner_opening_direction} planner "
        f"for {planner_handle_side}-side handle: {door_urdf_path}"
    )
    start_time = time.time()
    if planner_opening_direction == "pull":
        robot_traj, door_traj, key_idx_in_key_indices = state_machine_offline_pull_door(
            robot_urdf_path,
            door_urdf_path,
            robot_initial_pose,
            door_initial_pose,
            robot_initial_q,
            door_initial_q,
            handle_side=planner_handle_side,
            device="cpu",
        )
    else:
        robot_traj, door_traj, key_idx_in_key_indices = state_machine_offline_push_door(
            robot_urdf_path,
            door_urdf_path,
            robot_initial_pose,
            door_initial_pose,
            robot_initial_q,
            door_initial_q,
            handle_side=planner_handle_side,
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
    # Pull motions get a longer horizon (1440 frames = 36 s at the 1/40 s traj dt) than push (1200);
    # pull has more phases (retract + push-panel + traverse). The multi-motion lib pads the shorter
    # (push) motions up to the longest on load, so mixing lengths is safe.
    traj_length = 1440 if planner_opening_direction == "pull" else 1200
    robot_traj, door_traj, robot_traj_d, door_traj_d, key_indices = collocate_and_playback(
        robot_traj,
        door_traj,
        key_idx_in_key_indices,
        length=traj_length,
    )
    print(robot_traj.shape)
    print(door_traj.shape)
    print(robot_traj_d.shape)
    print(door_traj_d.shape)

    robot_world_pos = robot_initial_pose[:, :3].squeeze(0).numpy()
    door_world_pos = door_initial_pose[:, :3].squeeze(0).numpy()
    robot_world_quat = robot_initial_pose[:, 3:].squeeze(0).numpy()
    door_world_quat = door_initial_pose[:, 3:].squeeze(0).numpy()

    if visualize:
        from yourdfpy import URDF

        robot_urdf = URDF.load(robot_urdf_path)
        door_urdf  = URDF.load(door_urdf_path)

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
            playback_speed=playback_speed,
            capture_dir=capture_dir,
            capture_every=capture_every,
            port=port,
        )

    robot_ik_solver = PinocchioIKSolver(
        urdf_path=robot_urdf_path, 
        ee_link_name="palm_center", 
        controlled_joints=DM_JOINT_NAMES
    ) 

    robot_key_bodies = ROBOT_KEY_BODY_NAMES
    T = len(robot_traj)
    K = len(robot_key_bodies)

    # Gather key-body poses (in base_link frame) with ONE forwardKinematics per frame instead of
    # one per (frame, body): the base<-body transforms come back as raw numpy, and the base->world
    # composition (quat_from_matrix + combine_frame_transforms) is done once as a single batched
    # torch op over all T*K poses rather than per element.
    body_trans = np.empty((T, K, 3), dtype=np.float32)   # base_link-frame translation
    body_rot = np.empty((T, K, 3, 3), dtype=np.float32)  # base_link-frame rotation matrix
    for t, robot_point in enumerate(robot_traj):
        trans, rot = robot_ik_solver.get_frames_pose_batch(
            config=robot_point[:10].detach().cpu().numpy(),
            node_a_list=robot_key_bodies,
            node_b="base_link",
        )
        body_trans[t] = trans
        body_rot[t] = rot

    body_trans = torch.from_numpy(body_trans).reshape(-1, 3)          # (T*K, 3)
    body_quat = quat_from_matrix(torch.from_numpy(body_rot).reshape(-1, 3, 3))  # (T*K, 4)

    robot_world_pos_t = torch.tensor(robot_world_pos, dtype=torch.float32).expand(T * K, 3)
    robot_world_quat_t = torch.tensor(robot_world_quat, dtype=torch.float32).expand(T * K, 4)

    body_world_pos, body_world_quat = combine_frame_transforms(
        t01=robot_world_pos_t,
        q01=robot_world_quat_t,
        t12=body_trans,
        q12=body_quat,
    )
    robot_body_pos_traj = body_world_pos.reshape(T, K, 3)
    robot_body_quat_traj = body_world_quat.reshape(T, K, 4)

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

    # ---- Contact masks (time-gated reward shaping) ----
    # Keyframe layout (both planners): 0=start, 1=pregrasp, 2=grasp, 3=rotate-hinge,
    # 4=open/pull-end, 5=base-forward (push) / blocking-pose (pull). Pull additionally
    # has 6=retract-arm, 7-8=push-panel-open, 9-10=traverse.
    #
    # hinge_contact_mask: ENCOURAGE palm<->handle contact while the hand actually works
    # the handle, i.e. keyframes 2..5 for both directions:
    #   pull: rotate handle (3) -> pull open (4) -> move base to blocking pose (5)
    #   push: rotate handle (3) -> open door (4) -> base forward (4.5)
    hinge_contact_mask = torch.zeros(len(robot_traj), dtype=torch.int8)
    if len(key_indices) > 5:
        hinge_contact_mask[key_indices[2]:key_indices[5]] = 1

    # panel_contact_mask: gates the unified finger<->door (panel Door/link_1 + handle Door/link_2)
    # FORCE-PROTECTION penalty.
    #   push: ON during the open-door + base-forward phase (keyframes 3..5), where the partly-open
    #         hand presses the panel open -- there we still want the penalty to keep the push gentle.
    #   pull: OFF for the whole trajectory (mask stays all-zero). Penalizing finger<->door contact on
    #         pull made the robot "scared" to push the door panel with the arm, so pull carries no
    #         finger<->door penalty at all.
    panel_contact_mask = torch.zeros(len(robot_traj), dtype=torch.int8)
    if planner_opening_direction == "push":
        if len(key_indices) > 5:
            panel_contact_mask[key_indices[3]:key_indices[5]] = 1
    # else: pull -> no finger<->door penalty (see above); mask stays all-zero.

    # grasp_stage_mask: the PREGRASP -> GRASP window (keyframes 1..3: pregrasp, grasp, up to the
    # rotate keyframe). Used ONLY to gate the finger<->panel normal-force LOGGING (no reward) so we
    # can monitor how hard the fingers press the panel while approaching + closing on the handle.
    grasp_stage_mask = torch.zeros(len(robot_traj), dtype=torch.int8)
    if len(key_indices) > 3:
        grasp_stage_mask[key_indices[1]:key_indices[3]] = 1

    data = {
        "handle_side": planner_handle_side,
        "opening_direction": planner_opening_direction,
        "sim_dt": float(traj_dt),
        "door_traj": door_traj, 
        "robot_body_pos_traj": robot_body_pos_traj,
        "robot_body_quat_traj": robot_body_quat_traj,
        "robot_joint_pos_traj": robot_traj,
        "robot_joint_vel_traj": robot_traj_d,
        # "key_indices": torch.tensor(key_indices, dtype=torch.int32)[key_idx_in_key_indices]
        "hinge_contact_mask": hinge_contact_mask,
        "panel_contact_mask": panel_contact_mask,
        "grasp_stage_mask": grasp_stage_mask,
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
        "--capture-dir",
        type=str,
        default=None,
        help="With --visualize: render the REAL viser scene to PNGs here (needs a browser, "
             "headless is fine, pointed at the printed URL).",
    )
    parser.add_argument("--capture-every", type=int, default=8, help="capture 1 in N frames")
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
        default="auto",
        choices=["auto", "right", "left"],
        help="Select handle-side planner tuning, or read it from variant_meta.json when available.",
    )
    parser.add_argument(
        "--opening-direction",
        default="auto",
        choices=["auto", "pull", "push"],
        help="Select the door planner, or read pull/push from variant_meta.json when available.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=10.0,
        help="Initial viser playback speed multiplier when --visualize is set.",
    )
    parser.add_argument(
        "--traj-dt",
        type=float,
        default=DEFAULT_TRAJ_DT,
        help="Time step represented by adjacent trajectory samples.",
    )
    parser.add_argument(
        "--start-base-sampling",
        default="paired",
        choices=["paired", "random", "none"],
        help=(
            "How to choose the initial mobile-base waypoint. 'paired' hashes a "
            "pull/push-shared asset identity; 'random' preserves the old fresh "
            "RNG behavior; 'none' uses the default robot base."
        ),
    )
    parser.add_argument(
        "--start-base-radius",
        type=float,
        default=1.0,
        help="Radius in meters for the initial mobile-base waypoint around the door.",
    )
    parser.add_argument(
        "--start-base-angle-range-deg",
        type=float,
        default=30.0,
        help="Symmetric angle range in degrees for initial mobile-base waypoint sampling.",
    )
    parser.add_argument(
        "--start-base-pair-seed",
        type=int,
        default=0,
        help="Seed mixed into the deterministic paired start-base hash.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional Viser server port when --visualize is set.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Play each trajectory in Viser before saving. Disabled by default for batch processing.",
    )
    args = parser.parse_args()

    robot_urdf_path = args.robot_urdf_path
    if args.door_urdf_path is not None:
        asset_paths = [args.door_urdf_path]
    else:
        asset_base_folder = args.asset_base_folder
        asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.urdf"), recursive=True), reverse=False)

    for i, door_urdf_path in enumerate(asset_paths):
        play_and_save_traj(
            robot_urdf_path,
            door_urdf_path,
            handle_side=args.handle_side,
            opening_direction=args.opening_direction,
            start_base_sampling=args.start_base_sampling,
            start_base_radius=args.start_base_radius,
            start_base_angle_range_deg=args.start_base_angle_range_deg,
            start_base_pair_seed=args.start_base_pair_seed,
            playback_speed=args.playback_speed,
            capture_dir=args.capture_dir,
            capture_every=args.capture_every,
            traj_dt=args.traj_dt,
            port=args.port,
            visualize=args.visualize,
        )
        print("Finished processing ", door_urdf_path, ", index: ", i)
    
