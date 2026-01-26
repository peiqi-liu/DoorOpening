from DoorOpening.utils.llms.llm_utils import solve_ik, write_joint_angle_to_robot, write_joint_angle_to_door, record_joint_angles, get_hinge_pos, get_robot_link_pose, open_hand, close_hand, step_sim
import torch

def state_machine(robot, door, scene, sim, buffer):
    num_steps = 50
    handle_pos = get_hinge_pos(door)
    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += 0.8
    base_target_pos[:, 2] = 0.0
    base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)
    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.2
    palm_target_pos[:, 2] += 0.15
    palm_target_rot = torch.tensor([[0.7071, 0.0, -0.7071, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)
    open_hand(robot)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)

    num_steps = 40
    handle_pos = get_hinge_pos(door)
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pose = torch.cat([curr_base_pos, curr_base_rot], dim=-1)
    palm_target_pos = handle_pos.clone()
    palm_target_rot = torch.tensor([[0.7071, 0.0, -0.7071, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)
    close_hand(robot)

    record_joint_angles(robot, door, buffer)

    for _ in range(10):
        step_sim(scene, sim)
    target_hinge_angle = torch.tensor([[-1.0]]).to(handle_pos.device)
    target_board_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
    step_sim(scene, sim)
    new_handle_pos = get_hinge_pos(door)
    palm_target_pose_down = torch.cat([new_handle_pos, palm_target_rot], dim=-1)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose_down, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)

    num_steps = 60
    target_board_angle = torch.tensor([[1.4]]).to(handle_pos.device)
    target_hinge_angle = torch.tensor([[-1.0]]).to(handle_pos.device)
    write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
    step_sim(scene, sim)
    new_handle_pos = get_hinge_pos(door)
    base_target_pos = new_handle_pos.clone()
    base_target_pos[:, 0] += 0.5
    base_target_pos[:, 1] += 0.5
    base_target_pos[:, 2] = 0.0
    base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)
    palm_target_rot = torch.tensor([[0.7071, 0.0, -0.7071, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    palm_target_pose = torch.cat([new_handle_pos, palm_target_rot], dim=-1)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)

    num_steps = 40
    handle_pos = get_hinge_pos(door)
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pos = curr_base_pos.clone()
    base_target_pos[:, 0] += 0.4
    base_target_pos[:, 1] += 0.2
    base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
    palm_target_rot = torch.tensor([[0.7071, 0.0, -0.7071, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    palm_target_pose = torch.cat([handle_pos, palm_target_rot], dim=-1)
    target_board_angle = torch.tensor([[1.4]]).to(handle_pos.device)
    target_hinge_angle = torch.tensor([[-1.0]]).to(handle_pos.device)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)

    open_hand(robot)
    for _ in range(10):
        step_sim(scene, sim)
    retract_palm_pos = base_target_pos.clone()
    retract_palm_pos[:, 2] += 0.6
    retract_palm_pose = torch.cat([retract_palm_pos, palm_target_rot], dim=-1)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=retract_palm_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)

    num_steps = 100
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pos = curr_base_pos.clone()
    base_target_pos[:, 0] = -1.5
    base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
    for _ in range(num_steps):
        q = solve_ik(robot, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)