import torch

def tendon_to_joint_angle_utils(
    robot,
    tendon,
    total_finger_joints = 16,
):
    finger_joint_names = [f"finger_joint_{i}" for i in range(total_finger_joints)]
    dof_idx, finger_joint_names = robot.find_joints(finger_joint_names)
    dof_names_to_id = {}
    for name, idx in zip(finger_joint_names, dof_idx):
        dof_names_to_id[name] = idx
    return leap_tendon_to_joints(
        tendon,
        dof_names_to_id,
        robot.data.default_joint_pos.clone(),
        device = robot.data.joint_pos.device,
    )

def leap_tendon_to_joints(
    tendon: torch.Tensor,
    dof_names_to_id,
    cur_joint_pos,
    coeffs=(1.0, 1.0, 1.0),
    device=None,
    tendon_names={
        0: ["finger_joint_0", "finger_joint_1", "finger_joint_2", "finger_joint_3"], 
        1: ["finger_joint_4", "finger_joint_5", "finger_joint_6", "finger_joint_7"], 
        2: ["finger_joint_8", "finger_joint_9", "finger_joint_10", "finger_joint_11"], 
        3: ["finger_joint_13", "finger_joint_12", "finger_joint_14", "finger_joint_15"]
    }
):
    """
    Convert 4 tendon values → 16 LEAP joint positions.

    Args:
        tendon: Tensor [B, 4] or [4]
        coeffs: (MCP, PIP, DIP) coupling coefficients
    """
    if tendon.ndim == 1:
        tendon = tendon.unsqueeze(0)

    B = tendon.shape[0]
    device = device or tendon.device

    coeffs = torch.tensor(coeffs, device=device)
    tendon = tendon.to(device)

    for f, joints in tendon_names.items():
        # abduction_joint = joints[0]
        # dof_id = dof_names_to_id[abduction_joint]
        for j in range(1, len(joints)):
            if joints[j] == "finger_joint_12":
                continue
            dof_id = dof_names_to_id[joints[j]]
            cur_joint_pos[:, dof_id] = tendon[:, f] * coeffs[j-1]

    return cur_joint_pos

def joint_angle_to_tendon_utils(
    robot,
    total_finger_joints = 16,
):
    cur_joint_pos = robot.data.joint_pos.clone()
    finger_joint_names = [f"finger_joint_{i}" for i in range(total_finger_joints)]
    dof_idx, finger_joint_names = robot.find_joints(finger_joint_names)
    finger_dof = cur_joint_pos[:, dof_idx]
    dof_names_to_id = {name: idx for idx, name in enumerate(finger_joint_names)}
    return leap_joints_to_tendon(finger_dof, dof_names_to_id)

def leap_joints_to_tendon(
    cur_joint_pos: torch.Tensor,
    dof_names_to_id,
    coeffs=(1.0, 1.0, 1.0),
    device=None,
    tendon_names={
        0: ["finger_joint_0", "finger_joint_1", "finger_joint_2", "finger_joint_3"], 
        1: ["finger_joint_4", "finger_joint_5", "finger_joint_6", "finger_joint_7"], 
        2: ["finger_joint_8", "finger_joint_9", "finger_joint_10", "finger_joint_11"], 
        3: ["finger_joint_13", "finger_joint_12", "finger_joint_14", "finger_joint_15"]
    }
):
    """
    Convert LEAP joint positions → 4 tendon values.

    Args:
        cur_joint_pos: Tensor [B, num_dofs] or [num_dofs]
        dof_names_to_id: dict joint_name -> index
        coeffs: (MCP, PIP, DIP)

    Returns:
        tendon: Tensor [B, 4]
    """
    if cur_joint_pos.ndim == 1:
        cur_joint_pos = cur_joint_pos.unsqueeze(0)

    device = device or cur_joint_pos.device
    B = cur_joint_pos.shape[0]

    coeffs = torch.tensor(coeffs, device=device)

    tendon = torch.zeros(B, 4, device=device)

    for f, joints in tendon_names.items():
        # joints[0] is abduction → ignored
        q = []
        for j in range(1, len(joints)):
            if joints[j] == "finger_joint_12":
                continue
            dof_id = dof_names_to_id[joints[j]]
            q.append(cur_joint_pos[:, dof_id])

        q = torch.stack(q, dim=1)  # [B, 3]

        denom = torch.sum(coeffs ** 2) if f != 3 else torch.sum(coeffs[1:] ** 2) 
        tendon[:, f] = ((q * coeffs).sum(dim=1)) / denom if f != 3 else ((q * coeffs[1:]).sum(dim=1) / denom)

    return tendon

if __name__ == "__main__":
    t = torch.rand(8, 4)
    dof_names_to_id = {f"finger_joint_{i}": i for i in range(16)}
    q = leap_tendon_to_joints(t, dof_names_to_id, torch.zeros(8, 16))
    q[:, [0, 4, 8, 12]] = 5.0
    t_hat = leap_joints_to_tendon(q, dof_names_to_id)

    print(torch.max(torch.abs(t - t_hat)))