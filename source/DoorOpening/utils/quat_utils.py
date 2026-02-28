import torch
from isaaclab.utils.math import euler_xyz_from_quat

@torch.jit.script
def quat_pos(x):
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q

@torch.jit.script
def quat_to_axis_angle(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    eps = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3
    
    # need to make sure w is not negative to calculate geodesic distance
    q = quat_pos(q)
    length = torch.norm(q[..., qx:qw], dim=-1, p=2)
    
    angle = 2.0 * torch.atan2(length, q[..., qw])
    axis = q[..., qx:qw] / length.unsqueeze(-1)

    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1
    mask = length > eps

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return axis, angle

@torch.jit.script
def quat_mul(a, b):
    # type: (Tensor, Tensor) -> Tensor
    assert a.shape == b.shape

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1)
    return quat


@torch.jit.script
def quat_conjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

@torch.jit.script
def quat_diff(q0, q1):
    dq = quat_mul(q1, quat_conjugate(q0))
    return dq

@torch.jit.script
def quat_diff_angle(q0, q1):
    dq = quat_diff(q0, q1)
    _, angle = quat_to_axis_angle(dq)
    return angle

def hinge_angle_diff(theta_a, theta_b):
    diff = theta_b - theta_a
    err = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi
    return torch.abs(err)

def quat_to_euler(quat_twist):
    """
    Convert quaternion to euler
    Input: quaternion: (..., 4)
    Output: euler: (..., 3)
    """
    quat = quat_twist  # (..., 4)

    # Flatten batch dims
    orig_shape = quat.shape[:-1]
    quat_flat = quat.reshape(-1, 4)

    # Convert
    roll, pitch, yaw = euler_xyz_from_quat(quat_flat)

    # Stack to (..., 3)
    euler = torch.stack((roll, pitch, yaw), dim=-1)

    # Restore original batch shape
    euler = euler.view(*orig_shape, 3)

    return euler
