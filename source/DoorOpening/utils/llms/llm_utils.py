from this import d
from DoorOpening.utils.extract_pointcloud_from_articulation import sample_pointcloud as sample_pointcloud_from_asset_path
from DoorOpening.assets.glorbot.glorbot_cfg import FRANKA_JOINT_NAMES, BASE_JOINT_NAMES
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat, quat_mul
import torch

def get_board_frame_joint_angle(door):
    board_frame_joint_idx, _ = door.find_joints("joint_1")
    board_frame_joint_idx = board_frame_joint_idx[0]
    return door.data.joint_pos[:, board_frame_joint_idx].cpu().clone()

def get_frame_hinge_joint_angle(door):
    frame_hinge_joint_idx, _ = door.find_joints("joint_2")
    frame_hinge_joint_idx = frame_hinge_joint_idx[0]
    return door.data.joint_pos[:, frame_hinge_joint_idx].cpu().clone()

def get_board_pos(door):
    board_body_idx, _ = door.find_bodies("link_1")
    board_body_idx = board_body_idx[0]
    return door.data.body_pos_w[:, board_body_idx].cpu().clone()

def get_hinge_pos(door):
    hinge_body_idx, _ = door.find_bodies("link_2")
    hinge_body_idx = hinge_body_idx[0]
    return door.data.body_pos_w[:, hinge_body_idx].cpu().clone()

def sample_pointcloud(door, joint_angles):
    door_pointcloud = sample_pointcloud_from_asset_path(door.cfg.spawn.asset_path, joint_angles, device=door.data.joint_pos.device)
    door_pointcloud = quat_apply(door.data.body_quat_w[:, 0], door_pointcloud) + door.data.body_pos_w[:, 0]
    return door_pointcloud

def solve_ik(
    robot,
    elbow_pose=None,
    palm_pose=None,
    base_pose=None,
    damping=0.5,
    step_scale = 0.1
):
    """
    IK solver for tidybot2 + franka arm.
    Accepts elbow/palm/base target poses (3 or 7 dims).
    Uses damped least squares and writes result to simulator.
    """
    assert elbow_pose is not None or palm_pose is not None or base_pose is not None, \
        "At least one of elbow_pose, palm_pose, or base_pose must be provided"

    device = "cpu"

    # --- body names ---
    elbow_body_name = "panda_link4"
    palm_body_name  = "palm_center"
    base_body_name  = "tidybot2_base_link"

    elbow_body_idx, _ = robot.find_bodies(elbow_body_name)
    palm_body_idx, _  = robot.find_bodies(palm_body_name)
    base_body_idx, _  = robot.find_bodies(base_body_name)
    if isinstance(elbow_body_idx, list):
        elbow_body_idx = elbow_body_idx[0]
    if isinstance(palm_body_idx, list):
        palm_body_idx = palm_body_idx[0]
    if isinstance(base_body_idx, list):
        base_body_idx = base_body_idx[0]

    # --- joints ---
    actuated_joint_names = FRANKA_JOINT_NAMES + BASE_JOINT_NAMES
    actuated_joint_ids, _ = robot.find_joints(actuated_joint_names)

    # current joint positions
    q = robot.data.joint_pos[:, actuated_joint_ids].cpu().clone()

    def get_quat_err(target_quat, cur_quat):
        # q_err = q_target * q_cur^{-1}
        conj = torch.cat([-cur_quat[:, :3], cur_quat[:, 3:]], dim=1)
        q_err = quat_mul(target_quat, conj)

        sign = torch.sign(q_err[:, 3:4])
        q_err = q_err * sign

        angle = 2.0 * torch.atan2(
            torch.norm(q_err[:, :3], dim=1),
            q_err[:, 3].clamp(-1.0, 1.0)
        )
        axis = q_err[:, :3] / (torch.norm(q_err[:, :3], dim=1, keepdim=True) + 1e-8)
        return axis * angle.unsqueeze(1)

    # recompute Jacobians every iter
    # robot_jacobians = robot.root_physx_view.get_jacobians()[:, [palm_body_idx, base_body_idx, elbow_body_idx], :, actuated_joint_ids]
    robot_jacobians = robot.root_physx_view.get_jacobians().cpu().index_select(1, torch.tensor([palm_body_idx, base_body_idx, elbow_body_idx], device=device)).index_select(3, torch.tensor(actuated_joint_ids, device=device))

    # current body positions / orientations
    cur_elbow_pos = robot.data.body_pos_w[:, elbow_body_idx].cpu().clone()
    cur_palm_pos  = robot.data.body_pos_w[:, palm_body_idx].cpu().clone()
    cur_base_pos  = robot.data.body_pos_w[:, base_body_idx].cpu().clone()

    cur_elbow_quat = robot.data.body_quat_w[:, elbow_body_idx].cpu().clone()
    cur_palm_quat  = robot.data.body_quat_w[:, palm_body_idx].cpu().clone()
    cur_base_quat  = robot.data.body_quat_w[:, base_body_idx].cpu().clone()

    J_list = []
    err_list = []

    # elbow target
    if elbow_pose is not None:
        if len(elbow_pose.shape) > 2:
            elbow_pose = elbow_pose[..., 0, :]
        if len(elbow_pose.shape) < 2:
            elbow_pose = elbow_pose.unsqueeze(0)
        if elbow_pose.shape[-1] == 3:
            tgt_pos = elbow_pose
            tgt_quat = None
        else:
            tgt_pos = elbow_pose[..., :3]
            tgt_quat = elbow_pose[..., 3:]

        pos_err = (tgt_pos - cur_elbow_pos)
        Jpos = robot_jacobians[:, 2, :3, :]
        J_list.append(Jpos)
        err_list.append(pos_err)

        if tgt_quat is not None:
            rot_err = get_quat_err(tgt_quat, cur_elbow_quat)
            Jrot = robot_jacobians[:, 2, 3:6, :]
            J_list.append(Jrot)
            err_list.append(rot_err)

    # palm target
    if palm_pose is not None:
        if len(palm_pose.shape) > 2:
            palm_pose = palm_pose[..., 0, :]
        if len(palm_pose.shape) < 2:
            palm_pose = palm_pose.unsqueeze(0)
        if palm_pose.shape[-1] == 3:
            tgt_pos = palm_pose
            tgt_quat = None
        else:
            tgt_pos = palm_pose[..., :3]
            tgt_quat = palm_pose[..., 3:]

        pos_err = (tgt_pos - cur_palm_pos)
        Jpos = robot_jacobians[:, 0, :3, :]
        J_list.append(Jpos)
        err_list.append(pos_err)

        if tgt_quat is not None:
            rot_err = get_quat_err(tgt_quat, cur_palm_quat)
            Jrot = robot_jacobians[:, 0, 3:6, :]
            J_list.append(Jrot)
            err_list.append(rot_err)

    # base target
    if base_pose is not None:
        if len(base_pose.shape) > 2:
            base_pose = base_pose[..., 0, :]
        if len(base_pose.shape) < 2:
            base_pose = base_pose.unsqueeze(0)
        if base_pose.shape[-1] == 3:
            tgt_pos = base_pose
            tgt_quat = None
        else:
            tgt_pos = base_pose[..., :3]
            tgt_quat = base_pose[..., 3:]

        pos_err = (tgt_pos - cur_base_pos)
        Jpos = robot_jacobians[:, 1, :3, :]
        J_list.append(Jpos)
        err_list.append(pos_err)

        if tgt_quat is not None:
            rot_err = get_quat_err(tgt_quat, cur_base_quat)
            Jrot = robot_jacobians[:, 1, 3:6, :]
            J_list.append(Jrot)
            err_list.append(rot_err)

    # stack into single task
    J = torch.cat(J_list, dim=1)      # [env, m, n]
    err = torch.cat(err_list, dim=1)  # [env, m]

    # damped least squares
    JT = J.transpose(1, 2)
    m = J.shape[1]
    I = torch.eye(m, device=device).unsqueeze(0)
    A = J @ JT + (damping ** 2) * I
    delta = JT @ torch.linalg.solve(A, err.unsqueeze(-1))
    delta = delta.squeeze(-1)
    # update joint positions
    delta = torch.clamp(delta, -step_scale, step_scale)
    q = q + step_scale * delta

    return q

def open_hand(robot):
    open_joint_values = {
        "finger_joint_0": 0.0,
        "finger_joint_1": 0.0,
        "finger_joint_2": 0.0,
        "finger_joint_3": 0.0,
        "finger_joint_4": 0.0,
        "finger_joint_5": 0.0,
        "finger_joint_6": 0.0,
        "finger_joint_7": 0.0,
        "finger_joint_8": 0.0,
        "finger_joint_9": 0.0,
        "finger_joint_10": 0.0,
        "finger_joint_11": 0.0,
        "finger_joint_12": torch.pi / 2,
        "finger_joint_13": 0.0,
        "finger_joint_14": 0.0,
        "finger_joint_15": 0.0,
    }
    robot_finger_joint_ids, joint_names = robot.find_joints(open_joint_values.keys())
    joint_values = torch.zeros_like(robot.data.joint_pos[..., robot_finger_joint_ids])
    for id in range(len(robot_finger_joint_ids)):
        joint_values[..., id] = open_joint_values[joint_names[id]]
    robot.write_joint_position_to_sim(joint_values.to(robot.data.joint_pos.device), joint_ids=robot_finger_joint_ids)

def close_hand(robot):
    close_joint_values = {
        "finger_joint_0": 0.0,
        "finger_joint_1": torch.pi / 2,
        "finger_joint_2": 1.8,
        "finger_joint_3": 1.0,
        "finger_joint_4": 0.0,
        "finger_joint_5": torch.pi / 2,
        "finger_joint_6": 1.8,
        "finger_joint_7": 1.0,
        "finger_joint_8": 0.0,
        "finger_joint_9": torch.pi / 2,
        "finger_joint_10": 1.8,
        "finger_joint_11": 1.0,
        "finger_joint_12": torch.pi / 2,
        "finger_joint_13": 0.0,
        "finger_joint_14": 0.5,
        "finger_joint_15": 1.0,
    }
    robot_finger_joint_ids, joint_names = robot.find_joints(close_joint_values.keys())
    joint_values = torch.zeros_like(robot.data.joint_pos[..., robot_finger_joint_ids])
    for id in range(len(robot_finger_joint_ids)):
        joint_values[..., id] = close_joint_values[joint_names[id]]
    robot.write_joint_position_to_sim(joint_values.to(robot.data.joint_pos.device), joint_ids=robot_finger_joint_ids)

def write_joint_angle_to_robot(robot, target_joint_angle):
    robot_joint_ids, joint_names = robot.find_joints(FRANKA_JOINT_NAMES + BASE_JOINT_NAMES)
    robot.write_joint_position_to_sim(target_joint_angle.to(robot.data.joint_pos.device), joint_ids=robot_joint_ids)

def record_joint_angles(robot, door, buffer):
    buffer.append(torch.cat((robot.data.joint_pos.clone(), door.data.joint_pos.clone()), dim=-1))

def get_robot_link_pose(robot, link_name):
    link_names_correspondance = {
        "elbow": "panda_link4",
        "palm": "palm_center",
        "base": "tidybot2_base_link",
    }
    link_name = link_names_correspondance[link_name]
    link_idx, _ = robot.find_bodies(link_name)
    link_idx = link_idx[0]
    return robot.data.body_pos_w[:, link_idx].cpu().clone(), robot.data.body_quat_w[:, link_idx].cpu().clone()

def write_joint_angle_to_door(door, target_board_joint_angle, target_hinge_joint_angle):
    door_joint_ids, joint_names = door.find_joints(["joint_1", "joint_2"])
    if isinstance(target_board_joint_angle, float):
        target_board_joint_angle = [target_board_joint_angle]
    if isinstance(target_hinge_joint_angle, float):
        target_hinge_joint_angle = [target_hinge_joint_angle]
    if not isinstance(target_board_joint_angle, torch.Tensor):
        target_board_joint_angle = torch.tensor(target_board_joint_angle)
    if not isinstance(target_hinge_joint_angle, torch.Tensor):
        target_hinge_joint_angle = torch.tensor(target_hinge_joint_angle)
    if len(target_board_joint_angle.shape) < 2:
        target_board_joint_angle = target_board_joint_angle.unsqueeze(0)
    if len(target_hinge_joint_angle.shape) < 2:
        target_hinge_joint_angle = target_hinge_joint_angle.unsqueeze(0)
    if len(target_board_joint_angle.shape) > 2:
        target_board_joint_angle = target_board_joint_angle.squeeze()
    if len(target_hinge_joint_angle.shape) > 2:
        target_hinge_joint_angle = target_hinge_joint_angle.squeeze()
    target_joint_angle = torch.cat((target_board_joint_angle, target_hinge_joint_angle), dim=-1)
    door.write_joint_position_to_sim(target_joint_angle.to(door.data.joint_pos.device), joint_ids=door_joint_ids)

def step_sim(scene, sim):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


import ast

def wrap_into_policy(code: str) -> str:
    lines = code.splitlines()

    # Drop empty leading/trailing lines
    lines = [l.rstrip() for l in lines if l.strip() != ""]

    # Strip *all* leading indentation
    stripped = [l.lstrip() for l in lines]

    # Re-indent uniformly
    body = "\n".join("    " + l for l in stripped)

    return (
        "def policy_step(robot, door, scene, sim):\n"
        + body
        + "\n"
    )

def load_policy(code: str):
    wrapped = wrap_into_policy(code)

    tree = ast.parse(wrapped)  # ✅ now this cannot fail from indentation

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(funcs) == 1

    env = {}
    exec(wrapped, env)
    return env["policy_step"]
