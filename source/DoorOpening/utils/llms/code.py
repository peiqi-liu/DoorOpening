from DoorOpening.utils.llms.llm_utils import solve_ik, write_joint_angle_to_robot, write_joint_angle_to_door, record_joint_angles, get_hinge_pos, get_robot_link_pose, open_hand, close_hand, step_sim, get_board_frame_joint_angle
import torch

def state_machine(robot, door, scene, sim, buffer):
    record_joint_angles(robot, door, buffer)
    print("Step 1: Move to pregrasp pose")
    num_steps = 50
    handle_pos = get_hinge_pos(door)
    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += 0.7
    base_target_pos[:, 1] -= 0.3
    base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)
    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.2
    palm_target_pos[:, 2] += 0.1
    palm_target_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)
    open_hand(robot)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)

    print("Step 2: Grasp the handle")
    num_steps = 40
    close_hand(robot)
    handle_pos = get_hinge_pos(door)
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pose = torch.cat([curr_base_pos, curr_base_rot], dim=-1)
    palm_target_pos = handle_pos.clone()
    palm_target_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)

    print("Step 3: Rotate hinge to unlatch")
    target_hinge_angle = torch.tensor([[1.0]]).to(handle_pos.device)
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

    print("Step 4: Pull door open by 1.4 radians")
    num_steps = 60
    # target_board_angle = torch.tensor([[1.4]]).to(handle_pos.device)
    # target_hinge_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    # write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
    # step_sim(scene, sim)
    # new_handle_pos = get_hinge_pos(door)
    # base_target_pos = new_handle_pos.clone()
    # base_target_pos[:, 0] += 0.5
    # base_target_pos[:, 1] = 0.0
    # base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    # base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)
    # palm_target_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    # palm_target_pose = torch.cat([new_handle_pos, palm_target_rot], dim=-1)
    # for _ in range(num_steps):
    #     q = solve_ik(robot, base_pose=base_target_pose, palm_pose=palm_target_pose)
    #     write_joint_angle_to_robot(robot, q)
    #     step_sim(scene, sim)

    for i in torch.arange(0, 1.41, 0.2):
        # base_target_pos, base_target_rot = get_robot_link_pose(robot, "base")
        # base_target_pos[:, 0] += 0.7
        # base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)
        # palm_target_pos, palm_target_rot = get_robot_link_pose(robot, "palm")
        # palm_target_pos[:, 0] += 0.7
        # palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)
        # for _ in range(num_steps):
        #     q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        #     write_joint_angle_to_robot(robot, q)
        #     step_sim(scene, sim)

        target_board_angle = torch.tensor([[i]]).to(handle_pos.device)
        target_hinge_angle = torch.tensor([[0.0]]).to(handle_pos.device)
        write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
        step_sim(scene, sim)
        new_handle_pos = get_hinge_pos(door)
        base_target_pos = new_handle_pos.clone()
        base_target_pos[:, 0] += 0.7
        # base_target_pos[:, 1] += 0.5
        base_target_pos[:, 1] = 0.2
        base_target_pos[:, 2] = 0.0
        base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
        base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)
        palm_target_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
        # new_handle_pos[:, 0] -= 0.1
        # new_handle_pos[:, 1] -= 0.1
        palm_target_pose = torch.cat([new_handle_pos, palm_target_rot], dim=-1)
        for _ in range(num_steps):
            q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
            write_joint_angle_to_robot(robot, q)
            step_sim(scene, sim)
        write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
        step_sim(scene, sim)
        record_joint_angles(robot, door, buffer)

    # record_joint_angles(robot, door, buffer)

    print("Step 5 & 6: Move the arm backward and the base move forward")
    num_steps = 40
    handle_pos = get_hinge_pos(door)
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pos = curr_base_pos.clone()
    base_target_pos[:, 0] -= 0.6
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
        step_sim(scene, sim)
    record_joint_angles(robot, door, buffer)

    base_target_pos[:, 0] -= 0.4
    # retract_palm_pos = handle_pos.clone()
    # retract_palm_pos[:, 0] += 0.2
    # retract_palm_pos[:, 1] += 0.2
    # retract_palm_pos[:, 2] = 0.75
    retract_palm_pos = base_target_pos.clone()
    retract_palm_pos[:, 2] = 0.75
    retract_palm_pos[:, 0] += 0.5
    retract_palm_pos[:, 1] -= 0.3
    retract_palm_pose = torch.cat([retract_palm_pos, palm_target_rot], dim=-1)
    base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
    target_board_angle = torch.tensor([[1.4]]).to(handle_pos.device)
    target_hinge_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    for _ in range(num_steps):
        q = solve_ik(robot, palm_pose=retract_palm_pose, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
        step_sim(scene, sim)
    record_joint_angles(robot, door, buffer)

    # print("Step 5: Hold the door with the arm, the base move forward")
    # num_steps = 40
    # handle_pos = get_hinge_pos(door)
    # curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    # base_target_pos = curr_base_pos.clone()
    # base_target_pos[:, 0] -= 0.8
    # # base_target_pos[:, 1] -= 0.2
    # base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
    # palm_target_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    # palm_target_pose = torch.cat([handle_pos, palm_target_rot], dim=-1)
    # target_board_angle = torch.tensor([[1.4]]).to(handle_pos.device)
    # target_hinge_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    # for _ in range(num_steps):
    #     q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
    #     write_joint_angle_to_robot(robot, q)
    #     write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
    #     step_sim(scene, sim)

    # record_joint_angles(robot, door, buffer)

    # print("Step 6: Release your hand, and retract the arm WITHOUT colliding with the door")
    # open_hand(robot)
    # for _ in range(10):
    #     step_sim(scene, sim)
    # retract_palm_pos = handle_pos.clone()
    # retract_palm_pos[:, 0] += 0.1
    # retract_palm_pos[:, 1] += 0.3
    # retract_palm_pos[:, 2] = 0.75
    # retract_palm_pose = torch.cat([retract_palm_pos, palm_target_rot], dim=-1)
    # for _ in range(num_steps):
    #     q = solve_ik(robot, palm_pose=retract_palm_pose, base_pose=base_target_pose)
    #     write_joint_angle_to_robot(robot, q)
    #     step_sim(scene, sim)
    # write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
    # step_sim(scene, sim)

    # record_joint_angles(robot, door, buffer)

    print("Step 7: Move base completely through the door")
    num_steps = 100
    curr_base_pos, curr_base_rot = get_robot_link_pose(robot, "base")
    base_target_pos = curr_base_pos.clone()
    base_target_pos[:, 0] = -1.5
    base_target_pos[:, 1] = 0
    base_target_pose = torch.cat([base_target_pos, curr_base_rot], dim=-1)
    # palm_target_pose = base_target_pos.clone()
    # retract_palm_pos[:, 2] = 0.75
    # retract_palm_pos[:, 0] += 0.5
    # retract_palm_pos[:, 1] -= 0.3
    # palm_target_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
    # palm_target_pose = torch.cat([palm_target_pose, palm_target_rot], dim=-1)
    for _ in range(num_steps):
        # q = solve_ik(robot, base_pose=base_target_pose, palm_pose=palm_target_pose)
        q = solve_ik(robot, base_pose=base_target_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)

    target_board_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    target_hinge_angle = torch.tensor([[0.0]]).to(handle_pos.device)
    write_joint_angle_to_door(door, target_board_angle, target_hinge_angle)
    step_sim(scene, sim)

    record_joint_angles(robot, door, buffer)