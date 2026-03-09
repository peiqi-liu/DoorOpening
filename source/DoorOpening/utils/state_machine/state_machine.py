from DoorOpening.utils.llms.llm_utils import solve_ik, write_joint_angle_to_robot, write_joint_angle_to_door, record_joint_angles, get_hinge_pos, get_robot_link_pose, open_hand, close_hand, step_sim, get_board_frame_joint_angle
from DoorOpening.utils.state_machine.api import solve_ik_iter as solve_ik
from DoorOpening.utils.llms.llm_utils import open_hand_by
import torch
from DoorOpening.assets.glorbot.glorbot_cfg import FRANKA_JOINT_NAMES, BASE_JOINT_NAMES
from DoorOpening.utils.pose_utils import quat_mul, quat_conjugate
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

def get_rotation_quat(roll, pitch, yaw, device):
    # yaw z
    # roll x
    # pitch y
    return quat_from_euler_xyz(roll = torch.tensor([[roll]]).to(device), pitch = torch.tensor([[pitch]]).to(device), yaw = torch.tensor([[yaw]]).to(device)).squeeze(0)

def state_machine(robot, door, scene, sim, buffer):
    joint_ids, _ = robot.find_joints(FRANKA_JOINT_NAMES + BASE_JOINT_NAMES)
    # record_joint_angles(robot, door, buffer)
    print("Step 1: Move to pregrasp pose")
    num_steps = 500
    handle_pos = get_hinge_pos(door)
    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += 0.8
    base_target_pos[:, 1] -= 0.3
    base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)
    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.2
    palm_target_pos[:, 2] += 0.3
    palm_target_rot = get_rotation_quat(roll = 0.0, pitch = 0.0, yaw = torch.pi, device = handle_pos.device)
    print("palm_target_rot: ", palm_target_rot)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)
    open_hand(robot)
    step_sim(scene, sim)
    q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
    i = 0
    # while torch.max(torch.abs(robot.data.joint_pos[0, joint_ids] - q)) > 0.045:
    while True:
        cur_palm_pos, cur_palm_quat = get_robot_link_pose(robot, "palm")
        if torch.linalg.norm(cur_palm_pos - palm_target_pos) < 0.03 and quat_mul(cur_palm_quat, quat_conjugate(palm_target_rot))[0,0] < 0.03:
            break
        cur_joint_pos = robot.data.joint_pos.clone()[:, joint_ids]
        step = (q - cur_joint_pos).clamp(-0.2, 0.2)
        robot.set_joint_position_target(cur_joint_pos + step, joint_ids=joint_ids)
        step_sim(scene, sim)
        i += 1
        if i > num_steps:
            print("Failed to reach target pose")
            break
    
    # record_joint_angles(robot, door, buffer)

    print("Step 2: Grasp the handle")
    num_steps = 150
    handle_pos = get_hinge_pos(door)
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pose = torch.cat([curr_base_pos, curr_base_rot], dim=-1)
    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.01
    # palm_target_pos[:, 2] -= 0.01
    palm_target_pos[:, 1] += 0.01
    palm_target_rot = get_rotation_quat(roll = 0.0, pitch = 0.0, yaw = torch.pi, device = handle_pos.device)
    # palm_target_rot = get_rotation_quat(roll = 0.0, pitch = 0.0, yaw = torch.pi + 1.0, device = handle_pos.device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)
    q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
    for _ in range(4):
        open_hand_by(robot, -0.1)
        step_sim(scene, sim)
    for _ in range(num_steps):
        cur_palm_pos, cur_palm_quat = get_robot_link_pose(robot, "palm")
        if torch.linalg.norm(cur_palm_pos - palm_target_pos) < 0.03 and quat_mul(cur_palm_quat, quat_conjugate(palm_target_rot))[0,0] < 0.03:
            print("Successfully reached target pose")
            break
        cur_joint_pos = robot.data.joint_pos.clone()[:, joint_ids]
        step = (q - cur_joint_pos).clamp(-0.1, 0.1)
        robot.set_joint_position_target(cur_joint_pos + step, joint_ids=joint_ids)
        step_sim(scene, sim)
    for _ in range(6):
        open_hand_by(robot, -0.1)
        step_sim(scene, sim)

    # record_joint_angles(robot, door, buffer)

    print("Step 3: Rotate hinge to unlatch")
    target_hinge_angle = torch.tensor([[0.8]]).to(handle_pos.device)
    target_board_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    target_joint_angles = torch.cat([target_board_angle, target_hinge_angle], dim=-1)
    # # target_hinge_pos = get_hinge_pos(door, target_joint_angles)
    # # cur_palm_pos, cur_palm_quat = get_robot_link_pose(robot, "palm")
    # # cur_palm_pos[:, 2] -= 0.1
    # # cur_palm_pos[:, 1] += 0.1
    # # palm_target_pose_down = torch.cat([cur_palm_pos, get_rotation_quat(roll = 0.0, pitch = 0.0, yaw = torch.pi + 1.0, device = handle_pos.device)], dim=-1)
    # # q = solve_ik(robot, palm_pose=palm_target_pose_down, base_pose=base_target_pose)
    # q = robot.data.joint_pos.clone()[:, joint_ids]
    # # panda_joint_1
    # q[:, 3] -= 0.02
    # # panda_joint_2
    # # q[:, 4] += 0.05
    # # panda_joint_5
    # # q[:, 7] -= 0.01
    # # panda_joint_4
    # q[:, 6] -= 0.01
    # for _ in range(num_steps):
    #     robot.set_joint_position_target(q, joint_ids=joint_ids)
    #     step_sim(scene, sim)
    # open_hand_by(robot, -0.1)
    # step_sim(scene, sim)
    door.write_joint_position_to_sim(target_joint_angles)
    step_sim(scene, sim)

    print("Step 4: Pull door open by 1.4 radians")
    for i in torch.arange(0, 1.6, 0.3):
        target_board_angle = torch.tensor([[i]]).to(handle_pos.device)
        # target_hinge_angle = torch.tensor([[max(1 - i, 0.0)]]).to(handle_pos.device)
        target_hinge_angle = torch.tensor([[0.0]]).to(handle_pos.device)
        # write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
        # step_sim(scene, sim)
        target_joint_angles = torch.cat([target_board_angle, target_hinge_angle], dim=-1)
        new_handle_pos = get_hinge_pos(door, target_joint_angles)
        base_target_pos = new_handle_pos.clone()
        base_target_pose[:, 0] = new_handle_pos[:, 0] + 0.8
        base_target_pose[:, 1] = i / 1.45 * 0.2
        # palm_target_rot = get_rotation_quat(roll = -i, pitch = 0.0, yaw = max(torch.pi + 1.0 - i, torch.pi), device = handle_pos.device)
        palm_target_rot = get_rotation_quat(roll = -min(i, 1.0), pitch = 0.0, yaw = torch.pi, device = handle_pos.device)
        new_handle_pos[:, 0] += 0.1 * torch.cos(i)
        new_handle_pos[:, 1] -= 0.1 * torch.sin(i)
        new_handle_pos[:, 2] += 0.1
        palm_target_pose = torch.cat([new_handle_pos, palm_target_rot], dim=-1)
        num_steps = 200
        i = 0
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        while True:
            cur_palm_pos, cur_palm_quat = get_robot_link_pose(robot, "palm")
            door_joint_pos = door.data.joint_pos.clone().detach().cpu()
            door_joint_pos_err = target_joint_angles - door_joint_pos
            if torch.linalg.norm(cur_palm_pos - new_handle_pos) < 0.03 and quat_mul(cur_palm_quat, quat_conjugate(palm_target_rot))[0,0] < 0.03 and torch.linalg.norm(door_joint_pos_err) < 0.1:
                break
            if i > num_steps:
                print("Failed to reach target pose")
                break
            robot.set_joint_position_target(q, joint_ids=joint_ids)
            door.set_joint_position_target(target_joint_angles)
            step_sim(scene, sim)
            i += 1
    for _ in range(2):
        door.write_joint_position_to_sim(target_joint_angles)
        step_sim(scene, sim)

    print("Step 5 & 6: Move the arm backward and the base move forward")
    num_steps = 40
    handle_pos = get_hinge_pos(door)
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pos = curr_base_pos.clone()
    palm_target_pose[:, 0] -= 0.3
    palm_target_pose[:, 3:] = get_rotation_quat(roll = 0.0, pitch = 0.0, yaw = torch.pi, device = handle_pos.device)
    for i in range(8):
        base_target_pos[:, 0] -= 0.1
        base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        for _ in range(num_steps):
            robot.set_joint_position_target(q, joint_ids=joint_ids)
            door.set_joint_position_target(target_joint_angles)
            step_sim(scene, sim)
    # base_target_pos[:, 0] -= 0.8
    # base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
    # curr_palm_pos, curr_palm_quat = get_robot_link_pose(robot, "palm")
    # palm_target_pose = torch.cat([curr_palm_pos, curr_palm_quat], dim=-1)
    # q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
    # i = 0
    # num_steps = 500
    # while True:
    #     if i > num_steps:
    #         print("Failed to reach target pose")
    #         break
    #     cur_base_pos, cur_base_rot = get_robot_link_pose(robot, "base")
    #     if torch.linalg.norm(cur_base_pos - base_target_pos) < 0.03 and quat_mul(cur_base_rot, quat_conjugate(base_target_rot))[0,0] < 0.03:
    #         break
    #     cur_joint_pos = robot.data.joint_pos.clone()[:, joint_ids]
    #     step = (q - cur_joint_pos).clamp(-0.5, 0.5)
    #     robot.set_joint_position_target(cur_joint_pos + step, joint_ids=joint_ids)
    #     door.set_joint_position_target(target_joint_angles)
    #     step_sim(scene, sim)
    #     i += 1

    # print("Step 7: Move base completely through the door")
    # num_steps = 100
    # curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    # base_target_pos = curr_base_pos.clone()
    # base_target_pos[:, 0] = -1.2
    # base_target_pos[:, 1] = 0
    # base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
    # # palm_target_pose = base_target_pos.clone()
    # # retract_palm_pos[:, 2] = 0.75
    # # retract_palm_pos[:, 0] += 0.5
    # # retract_palm_pos[:, 1] -= 0.3
    # # palm_target_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    # # palm_target_pose = torch.cat([palm_target_pose, palm_target_rot], dim=-1)
    # for _ in range(num_steps):
    #     # q = solve_ik(robot, base_pose=base_target_pose, palm_pose=palm_target_pose)
    #     q = solve_ik(robot, base_pose=base_target_pose)
    #     write_joint_angle_to_robot(robot, q)
    #     step_sim(scene, sim)

    # target_board_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    # target_hinge_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    # write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
    # step_sim(scene, sim)

    # record_joint_angles(robot, door, buffer)