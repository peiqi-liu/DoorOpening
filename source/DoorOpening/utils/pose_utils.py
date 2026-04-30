import torch
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.math import (
    quat_mul,
    quat_conjugate,
    quat_apply_inverse,
    quat_apply,
)
from typing import Optional

def world_to_base_frame(base_pos, base_quat, palm_pos_w, palm_quat_w):
    """
    Convert palm pose from world frame to base frame using IsaacLab math utils.

    All quaternions are [w, x, y, z].
    Supports batched tensors.
    """

    # Position: p_b = R_wb^T * (p_w - t_wb)
    palm_pos_b = quat_apply_inverse(base_quat, palm_pos_w - base_pos)

    # Orientation: q_b = q_wb^{-1} * q_wp
    palm_quat_b = quat_mul(quat_conjugate(base_quat), palm_quat_w)

    return palm_pos_b, palm_quat_b

def base_to_world_frame(base_pos, base_quat, palm_pos_b, palm_quat_b):
    """
    Convert palm pose from base frame to world frame using IsaacLab math utils.

    All quaternions are [w, x, y, z].
    Supports batched tensors.
    """

    # Position: p_w = R_wb * p_b + t_wb
    palm_pos_w = quat_apply(base_quat, palm_pos_b) + base_pos

    # Orientation: q_w = q_wb * q_b
    palm_quat_w = quat_mul(base_quat, palm_quat_b)

    return palm_pos_w, palm_quat_w

@torch.jit.script
def world_to_local(
    points: torch.Tensor,   # (E, N, 3)
    pos: Optional[torch.Tensor],      # (E, 3)
    quat: torch.Tensor      # (E, 4)
) -> torch.Tensor:

    E, N, _ = points.shape

    # 1. subtract translation
    if pos is not None:
        rel = points - pos.unsqueeze(1)            # (E, N, 3)
    else:
        rel = points

    # 2. invert quaternion
    quat_inv = quat_conjugate(quat)            # (E, 4)

    # 3. expand to match N
    quat_inv = quat_inv.unsqueeze(1).expand(-1, N, -1)  # (E, N, 4)

    # 4. rotate
    return quat_apply(quat_inv, rel)

def unbase_goal(abs_pos, orig_pos, orig_quat, velocity=False):
    """
    Converts an absolute world-frame pose (x, y, theta) to
    a robot-relative pose (x, y, theta).
    
    Args:
        abs_pos:  (x, y, theta) in world frame
        orig_pos: robot base (x, y, z)
        orig_quat: robot base (w, x, y, z)
        velocity: whether this is for velocities (no translation offset)

    Returns:
        torch.Tensor: (x_rel, y_rel, theta_rel)
    """
    abs_x, abs_y, abs_theta = abs_pos.unbind(dim=-1)
    orig_x, orig_y, orig_z = orig_pos.unbind(dim=-1)

    # Get original yaw
    _, _, orig_yaw = euler_xyz_from_quat(orig_quat)
    
    # Compute relative position
    if not velocity:
        # Subtract robot's position to get vector from robot to goal
        rel_x = abs_x - orig_x
        rel_y = abs_y - orig_y
    else:
        # For velocities, don't subtract position (just rotate the velocity vector)
        rel_x = abs_x
        rel_y = abs_y
    
    # Rotate by negative of robot's yaw to get robot-relative coordinates
    cos_yaw = torch.cos(-orig_yaw)
    sin_yaw = torch.sin(-orig_yaw)
    
    x_rel = rel_x * cos_yaw - rel_y * sin_yaw
    y_rel = rel_x * sin_yaw + rel_y * cos_yaw
    
    # For orientation, subtract robot's yaw to get relative angle
    theta_rel = abs_theta - orig_yaw
    
    # Normalize angle to [-pi, pi]
    theta_rel = torch.atan2(torch.sin(theta_rel), torch.cos(theta_rel))
    
    return torch.stack([x_rel, y_rel, theta_rel], dim=-1)

def base_goal(rel_pos, orig_pos, orig_quat, velocity=False):
    """
    Converts a robot-relative pose (x, y, theta) to
    an absolute world-frame pose (x, y, theta).
    
    This is the inverse of the unbase_goal function.
    
    Args:
        rel_pos:  (x_rel, y_rel, theta_rel) in robot frame
        orig_pos: robot base (x, y, z) in world frame
        orig_quat: robot base (w, x, y, z) in world frame
        velocity: whether this is for velocities (no translation offset)

    Returns:
        torch.Tensor: (x_abs, y_abs, theta_abs) in world frame
    """
    rel_x, rel_y, rel_theta = rel_pos.unbind(dim=-1)
    orig_x, orig_y, orig_z = orig_pos.unbind(dim=-1)

    # Get original yaw from quaternion
    _, _, orig_yaw = euler_xyz_from_quat(orig_quat)
    
    # Rotate relative coordinates by robot's yaw to get world-frame direction
    cos_yaw = torch.cos(orig_yaw)
    sin_yaw = torch.sin(orig_yaw)
    
    # Apply rotation: [cosθ -sinθ; sinθ cosθ] * [rel_x; rel_y]
    x_rot = rel_x * cos_yaw - rel_y * sin_yaw
    y_rot = rel_x * sin_yaw + rel_y * cos_yaw
    
    # For velocities, no translation needed
    if velocity:
        x_abs = x_rot
        y_abs = y_rot
    else:
        # Add robot's position to get absolute coordinates
        x_abs = x_rot + orig_x
        y_abs = y_rot + orig_y
    
    # For orientation, add robot's yaw to get absolute orientation
    theta_abs = rel_theta + orig_yaw
    
    # Normalize angle to [-pi, pi]
    theta_abs = torch.atan2(torch.sin(theta_abs), torch.cos(theta_abs))
    
    return torch.stack([x_abs, y_abs, theta_abs], dim=-1)

def get_base_pos_and_quat(articulation, base_name="base_link"):
    base_idx, _ = articulation.find_bodies(base_name)
    base_id = base_idx[0]
    base_pos = articulation.data.body_pos_w[:, base_id]
    base_quat = articulation.data.body_quat_w[:, base_id]
    return base_pos, base_quat

def unbase_goal_tool(articulation, abs_pos, abs_vel):
    if not isinstance(abs_pos, torch.Tensor):
        abs_pos = torch.from_numpy(abs_pos)
    if not isinstance(abs_vel, torch.Tensor):
        abs_vel = torch.from_numpy(abs_vel)
    base_pos, base_quat = get_base_pos_and_quat(articulation)
    base_pos = unbase_goal(abs_pos, base_pos, base_quat, velocity = False)
    base_vel = unbase_goal(abs_vel, torch.zeros_like(base_pos), base_quat, velocity = True)
    return base_pos, base_vel

def wrap_to_pi(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def quat_mul_batch(q, r):
    """
    Multiply two quaternions: q * r
    Both in wxyz format.
    q: (B, M, 4) or (B, 1, 4)
    r: (B, M, 4)
    Returns:
        (B, M, 4)
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack((w, x, y, z), dim=-1)


def quat_inv_batch(q):
    """
    Inverse of quaternion in wxyz format.
    For unit quaternions, inverse is conjugate.
    q: (B, 4)
    Returns:
        (B, 4)
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def quat_apply_batch(q, v):
    """
    Rotate vector(s) v by quaternion(s) q.
    q: (B, M, 4) or (B, 1, 4)  (wxyz)
    v: (B, M, 3)
    Returns:
        rotated vectors: (B, M, 3)
    """
    # Convert vector to quaternion with w=0
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)  # (B, M, 4)

    # q * v * q_inv
    q_inv = quat_inv_batch(q)
    rotated = quat_mul_batch(quat_mul_batch(q, qv), q_inv)

    return rotated[..., 1:]  # return vector part


def normalize_to_center_frame(p_center, q_center, poses, quats):
    """
    Works for:
        p_center: (..., M, 3)
        q_center: (..., M, 4)
        poses:    (..., M, 3) OR (..., K, M, 3)
        quats:    (..., M, 4) OR (..., K, M, 4)
    """

    assert poses.shape[:-1] == quats.shape[:-1]
    assert quats.shape[-1] == 4

    # If poses has extra twist dimension
    if poses.ndim == p_center.ndim + 1:
        p_center = p_center.unsqueeze(-3)
        q_center = q_center.unsqueeze(-3)

    # Invert center rotation
    q_center_inv = quat_inv_batch(q_center)

    # Relative rotations (broadcast automatically)
    quats_rel = quat_mul_batch(q_center_inv, quats)

    # Relative positions
    delta_p = poses - p_center
    poses_rel = quat_apply_batch(q_center_inv, delta_p)

    return poses_rel, quats_rel


if __name__ == "__main__":

    poses = torch.tensor([
        [[1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0]],

        [[0.0, 1.0, 0.0],
         [0.0, 2.0, 0.0],
         [0.0, 3.0, 0.0]]
    ], dtype=torch.float32)

    quats = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],

        [[1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    ], dtype=torch.float32)

    center_idx = 1

    # Keep body dimension!
    p_center = poses[:, center_idx:center_idx+1, :]   # (B,1,3)
    q_center = quats[:, center_idx:center_idx+1, :]   # (B,1,4)

    poses_rel, quats_rel = normalize_to_center_frame(
        p_center,
        q_center,
        poses,
        quats
    )

    print("Relative positions:\n", poses_rel)
    print("Relative quaternions:\n", quats_rel)