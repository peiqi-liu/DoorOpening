from DoorOpening.utils.extract_pointcloud_from_articulation import sample_pointcloud as sample_pointcloud_from_asset_path
from DoorOpening.assets.glorbot.glorbot_cfg import FRANKA_JOINT_NAMES, BASE_JOINT_NAMES
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat, quat_mul
import torch
from DoorOpening.utils.pose_utils import compute_base_joint, wrap_to_pi

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
    # joint_angles = door.data.joint_pos.clone()
    # pointcloud = sample_pointcloud_from_link_name(door.cfg.spawn.asset_path, joint_angles, "link_1", device = "cpu")
    # door_pointcloud = quat_apply(door.data.body_quat_w[:, 0].cpu(), pointcloud) + door.data.body_pos_w[:, 0].cpu()
    # door_pointcloud = door_pointcloud.squeeze()
    # board_pos = door_pointcloud.median(dim=0).values
    # if board_pos.ndim == 1:
    #     board_pos = board_pos.unsqueeze(0)
    # return board_pos.cpu()

def get_hinge_pos(door):
    hinge_body_idx, _ = door.find_bodies("link_2")
    hinge_body_idx = hinge_body_idx[0]
    return door.data.body_pos_w[:, hinge_body_idx].cpu().clone()
    # joint_angles = door.data.joint_pos.clone()
    # pointcloud = sample_pointcloud_from_link_name(door.cfg.spawn.asset_path, joint_angles, "link_2", device = "cpu")
    # door_pointcloud = quat_apply(door.data.body_quat_w[:, 0].cpu(), pointcloud) + door.data.body_pos_w[:, 0].cpu()
    # door_pointcloud = door_pointcloud.squeeze()
    # hinge_pos = door_pointcloud.median(dim=0).values
    # if hinge_pos.ndim == 1:
    #     hinge_pos = hinge_pos.unsqueeze(0)
    # print("hinge pos: ", hinge_pos)
    # return hinge_pos.cpu()

def sample_pointcloud(door, joint_angles):
    door_pointcloud = sample_pointcloud_from_asset_path(door.cfg.spawn.asset_path, joint_angles.cpu(), device="cpu")
    door_pointcloud = quat_apply(door.data.body_quat_w[:, 0].cpu(), door_pointcloud) + door.data.body_pos_w[:, 0].cpu()
    return door_pointcloud

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

def solve_ik(robot, palm_pose=None, base_pose=None):
    ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 1.0}
        )
    ik_controller = DifferentialIKController(
        ik_cfg, num_envs=1, device=robot.data.joint_pos.device
    )
    joint_names = ["base_x_joint", "base_y_joint", "base_rotation_joint", "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
    joint_ids, joint_names = robot.find_joints(joint_names)
    base_pose_idx, _ = robot.find_bodies("tidybot2_base_link")
    base_pose_idx = base_pose_idx[0]
    key_pose_idx, _ = robot.find_bodies("palm_center")
    key_pose_idx = key_pose_idx[0]
    ee_pos = robot.data.body_pos_w[:, key_pose_idx]
    ee_quat = robot.data.body_quat_w[:, key_pose_idx]
    joint_pos_des = robot.data.joint_pos[:, joint_ids].clone()
    if base_pose is not None:
        base_joint_pos = compute_base_joint(robot, base_pose[:, :3]).to(robot.data.joint_pos.device)
    if base_pose is not None and torch.linalg.norm(joint_pos_des[:, :3] - base_joint_pos) >= 0.05:
    # if base_pose is not None:
        joint_pos_des[:, :3] = base_joint_pos
        joint_pos_des[:, 3:] = wrap_to_pi(joint_pos_des[:, 3:])
        return joint_pos_des
    elif palm_pose is not None:
        ik_controller.reset()
        ik_controller.set_command(command=palm_pose, ee_pos=ee_pos, ee_quat=ee_quat)
        hand_jac = robot.root_physx_view.get_jacobians()[:, key_pose_idx, :, joint_ids[3:]]
        joint_pos_des[:, 3:] = ik_controller.compute(ee_pos, ee_quat, hand_jac, joint_pos_des[:, 3:])
        joint_pos_des[:, 3:] = wrap_to_pi(joint_pos_des[:, 3:])
        return joint_pos_des
    return joint_pos_des

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
    # buffer.append(torch.cat((robot.data.joint_pos.clone(), door.data.joint_pos.clone()), dim=-1))
    body_idx, _ = robot.find_bodies(["palm_center", "tidybot2_base_link"])
    robot_pos = robot.data.body_pos_w[:, body_idx].cpu().clone().reshape(robot.data.body_pos_w.shape[0], -1)
    robot_quat = robot.data.body_quat_w[:, body_idx].cpu().clone().reshape(robot.data.body_quat_w.shape[0], -1)
    buffer.append(torch.cat((robot_pos, robot_quat, door.data.joint_pos.clone().cpu()), dim=-1))

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
    door.set_joint_position_target(target_joint_angle.to(door.data.joint_pos.device), joint_ids=door_joint_ids)

def step_sim(scene, sim):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
