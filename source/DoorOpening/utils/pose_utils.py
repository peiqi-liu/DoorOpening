import torch
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.math import (
    quat_mul,
    quat_conjugate,
    quat_apply_inverse,
)

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

def world_to_local(points, pos, quat):
    """
    Convert world-frame (N,3) points into local frame defined by pos+quat.
    quat: (w,x,y,z) rotation of frame in world
    """

    if quat.ndim != 1:
        quat = quat.squeeze()
    if pos.ndim != 1:
        pos = pos.squeeze()

    # Convert quaternion to rotation matrix
    w, x, y, z = quat
    R = torch.tensor([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),       2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], dtype=points.dtype, device=points.device)

    # Translate then rotate by inverse (R^T)
    return (points - pos) @ R

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


def compute_base_joint(articulation, abs_pos):
    """
    Convert desired world-frame base pose to robot-frame (x, y, theta).

    abs_pos:
      - torch or np array
      - shape (3,) or (7,)
    """

    # --- ensure torch ---
    if not isinstance(abs_pos, torch.Tensor):
        abs_pos = torch.tensor(abs_pos, dtype=torch.float32)

    # --- current robot base pose (world) ---
    base_pos_w, base_quat_w = get_base_pos_and_quat(articulation)
    base_pos_w = base_pos_w.cpu().clone()
    base_quat_w = base_quat_w.cpu().clone()
    abs_pos = abs_pos.cpu().clone()
    # base_pos_w: (3,)
    # base_quat_w: (4,) (x,y,z,w)

    # --- desired world position ---
    target_pos_w = abs_pos[:, :3]

    # --- displacement in world frame (planar) ---
    dp_w = target_pos_w[:, :2] - base_pos_w[:, :2]

    # --- robot yaw from quaternion ---
    _, _, base_yaw = euler_xyz_from_quat(base_quat_w)

    # --- rotate into robot frame ---
    c = torch.cos(base_yaw)
    s = torch.sin(base_yaw)

    x_r =  c * dp_w[:, 0] + s * dp_w[:, 1]
    y_r = -s * dp_w[:, 0] + c * dp_w[:, 1]

    # --- theta action ---
    if abs_pos.numel() == 7:
        target_quat_w = abs_pos[:, 3:7]
        _, _, target_yaw = euler_xyz_from_quat(target_quat_w)
        theta_r = wrap_to_pi(target_yaw - base_yaw)
    else:
        theta_r = torch.zeros(abs_pos.shape[0], device=abs_pos.device)

    return torch.cat([x_r, y_r, theta_r], dim=-1)
