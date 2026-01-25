# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `DoorOpening` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import DoorOpening.tasks  # noqa: F401


def main():
    """Print all environments registered in `DoorOpening` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if "Dooropening" in task_spec.id:
            # add details to table
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()


#         codes = """
# # Define simulation parameters
# num_steps = 100
# # Retrieve door handle position (this is the target for grasping)
# handle_pos = get_hinge_pos(door) # Shape (N, 3)

# # Define Target Base Pose
# # Move base to be ~65cm away from the handle in X direction (robot is at +X, door at 0)
# # We align the base Y with the handle Y to be centered on it
# base_target_pos = handle_pos.clone()
# base_target_pos[:, 0] += 0.65 
# base_target_pos[:, 2] = 0.0 # Base is on the floor
# # Base Orientation: Face -X direction (towards the door)
# # Identity is +X. Rotation 180 deg (pi) around Z axis.
# # Quaternion (w, x, y, z) for 180 Z rot: [0, 0, 0, 1]
# base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
# base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

# # Define Target Palm Pose (Pre-grasp)
# # Position: 10cm in front of handle (X), aligned in Y, 10cm above in Z to prepare for push down
# palm_target_pos = handle_pos.clone()
# palm_target_pos[:, 0] += 0.10
# palm_target_pos[:, 2] += 0.10
# # Orientation: Point fingers towards -X. 
# # Rotate -90 deg around Y axis: [0.707, 0, -0.707, 0]
# palm_target_rot = torch.tensor([[0.7071, 0.0, -0.7071, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
# palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)

# # Define Target Elbow Pose (Constraint for IK)
# # Heuristic: Elbow up and to the side to avoid body collision
# elbow_target_pos = base_target_pos.clone()
# elbow_target_pos[:, 2] += 1.0 # 1m height
# elbow_target_pos[:, 0] -= 0.1 # Slightly forward

# # Prepare Hand
# open_hand(robot)

# # Execute Motion
# for _ in range(num_steps):
#     # Solve IK for the defined targets
#     # We provide full 7D poses for base and palm to constrain orientation
#     q = solve_ik(robot, elbow_pose=elbow_target_pos, palm_pose=palm_target_pose, base_pose=base_target_pose)
    
#     # Write to robot and step simulation
#     write_joint_angle_to_robot(robot, q)
#     step_sim(scene, sim)
        
#         """
#         import textwrap
#         codes = textwrap.dedent(codes)
#         exec(codes)

#         codes = """
# num_steps_approach = 30
# num_steps_push = 40
# door_y_center = 0.0

# # Get environment state
# handle_pos = get_hinge_pos(door) # (N, 3)
# current_base_pose = get_robot_link_pose(robot, "base")[0] # Get current base pos (N, 3) to lock it
# current_base_rot = get_robot_link_pose(robot, "base")[1] # (N, 4)
# # Re-construct base pose tensor
# base_target_pose = torch.cat([current_base_pose, current_base_rot], dim=-1)

# # Determine lever direction offset
# # Handle is on "right part". If y > 0, it's on left (w.r.t robot facing -X)? 
# # Let's rely on coordinate sign. If handle.y is positive, lever likely points -y (inward). 
# # If handle.y is negative, lever likely points +y (inward).
# # We add a small offset towards y=0 to hit the lever arm.
# lever_offset_y = -0.06 * torch.sign(handle_pos[:, 1:2]) 
# # Note: If handle is at 0, this does nothing, but handle is on the side.

# # Define Orientation (Fingers pointing -X, same as before)
# palm_target_rot = torch.tensor([[0.7071, 0.0, -0.7071, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)


# # Waypoint 2: Contact Ready (Above and close in X)
# # Target: Handle Z + 0.10, Handle X + 0.02 (Very close to door plane to ensure overlap)
# wp2_pos = handle_pos.clone()
# wp2_pos[:, 0]
# wp2_pos[:, 1] += lever_offset_y.squeeze(-1)
# wp2_pos[:, 2]
# wp2_pose = torch.cat([wp2_pos, palm_target_rot], dim=-1)

# # Waypoint 3: Pushed Down
# # Target: Handle Z - 0.20 (Push well below to ensure threshold is met), Keep X close
# wp3_pos = handle_pos.clone()
# wp3_pos[:, 0] += 0.02
# wp3_pos[:, 1] += lever_offset_y.squeeze(-1)
# wp3_pos[:, 2] -= 0.20
# wp3_pose = torch.cat([wp3_pos, palm_target_rot], dim=-1)

# # Elbow constraint (keep high)
# elbow_target_pos = current_base_pose.clone()
# elbow_target_pos[:, 2] += 1.0
# elbow_target_pos[:, 0] -= 0.1

# # 1. Close hand to form a "fist" for pushing
# close_hand(robot)
# step_sim(scene, sim)

# # 3. Move to WP2 (Align)
# for _ in range(20):
#     q = solve_ik(robot, elbow_pose=elbow_target_pos, palm_pose=wp2_pose, base_pose=base_target_pose)
#     write_joint_angle_to_robot(robot, q)
#     step_sim(scene, sim)

# # 4. Move to WP3 (Push)
# for _ in range(num_steps_push):
#     q = solve_ik(robot, elbow_pose=elbow_target_pos, palm_pose=wp3_pose, base_pose=base_target_pose)
#     write_joint_angle_to_robot(robot, q)
#     step_sim(scene, sim)
#         """
#         codes = textwrap.dedent(codes)
#         # print("codes:", codes)
#         exec(codes)
#         # policy_step = load_policy(codes)
#         # policy_step()

#         codes = """
# import math
# # Configuration
# num_steps = 40
# target_angle = 1.3 # Open to ~75 degrees

# # 1. Get Initial Environment State
# # Pivot: The axis of rotation for the door
# pivot_pos = get_board_pos(door) 

# # Handle: The current position of the handle (which the robot is holding)
# handle_pos = get_hinge_pos(door)

# # Current Robot Poses (Reference for relative movement)
# base_pos, base_quat = get_robot_link_pose(robot, "base")
# palm_pos, palm_quat = get_robot_link_pose(robot, "palm")

# # 2. Calculate Geometry
# # We operate in the XY plane for the trajectory
# # Vector from Pivot to Handle
# radius_vec = handle_pos - pivot_pos
# radius_vec[:, 2] = 0 # Project to XY

# # Vector from Pivot to Robot Base
# base_vec = base_pos - pivot_pos
# base_vec[:, 2] = 0

# # Offset of Palm relative to Handle (to maintain the grasp offset, e.g., pushed down state)
# palm_handle_offset = palm_pos - handle_pos

# # 3. Execute Trajectory
# for i in range(num_steps):
#     # Interpolate angle
#     theta = (i + 1) * (target_angle / num_steps)
    
#     # Calculate Rotation (Rotation around Z-axis by theta)
#     # Since we are opening the door, we assume the positive angle direction 
#     # corresponds to the door opening "out" (towards +X where the robot is).
#     # Precompute sin/cos
#     c = math.cos(theta)
#     s = math.sin(theta)
    
#     # --- Update Door State ---
#     # We force the door to the new angle. We keep the handle pressed down (angle ~1.0 or current).
#     # Assuming handle needs to stay down to keep latch open, or just for consistency.
#     # We'll use a fixed value for the handle lever (e.g. 0.5 rad) or retrieved value.
#     # Let's use 1.0 rad to be safe (pushed down).
#     door_targets = torch.zeros((pivot_pos.shape[0], 2), device=pivot_pos.device)
#     door_targets[:, 0] = theta
#     door_targets[:, 1] = 1.0 
#     write_joint_angle_to_door(door, door_targets[:, 0], door_targets[:, 1])
    
#     # --- Calculate Target Robot State ---
#     # We rotate the robot's target positions around the Door Pivot by `theta`
    
#     # 1. New Handle Position (Rotation of radius_vec + Pivot)
#     # x' = x*c - y*s, y' = x*s + y*c
#     rx = radius_vec[:, 0]
#     ry = radius_vec[:, 1]
#     new_rx = rx * c - ry * s
#     new_ry = rx * s + ry * c
    
#     target_handle_pos = pivot_pos.clone()
#     target_handle_pos[:, 0] += new_rx
#     target_handle_pos[:, 1] += new_ry
#     # Keep original handle Z height (geometry of door)
#     target_handle_pos[:, 2] = handle_pos[:, 2] 
    
#     # 2. New Palm Position
#     # Follow the handle, maintaining the grasp offset (rotated?)
#     # For simplicity, we apply the offset to the new handle pos, 
#     # but strictly the offset vector should also rotate if the hand rotates.
#     # Let's rotate the palm_handle_offset too.
#     px = palm_handle_offset[:, 0]
#     py = palm_handle_offset[:, 1]
#     new_px = px * c - py * s
#     new_py = px * s + py * c
    
#     target_palm_pos = target_handle_pos.clone()
#     target_palm_pos[:, 0] += new_px
#     target_palm_pos[:, 1] += new_py
#     target_palm_pos[:, 2] += palm_handle_offset[:, 2] # Z offset stays same
    
#     # 3. New Base Position
#     # Rotate the base around the pivot to maintain workspace ("Dance with door")
#     bx = base_vec[:, 0]
#     by = base_vec[:, 1]
#     new_bx = bx * c - by * s
#     new_by = bx * s + by * c
    
#     target_base_pos = pivot_pos.clone()
#     target_base_pos[:, 0] += new_bx
#     target_base_pos[:, 1] += new_by
#     target_base_pos[:, 2] = base_pos[:, 2] # Floor height
    
#     # 4. New Orientations (Rotate Base and Palm quaternions by theta around Z)
#     # Quaternion for Z-rotation of angle theta: [cos(t/2), 0, 0, sin(t/2)]
#     cr = math.cos(theta / 2.0)
#     sr = math.sin(theta / 2.0)
    
#     # Update Base Orientation
#     # q_new = q_rot * q_orig
#     w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
#     nb_w = w*cr - z*sr
#     nb_x = x*cr - y*sr
#     nb_y = y*cr + x*sr
#     nb_z = z*cr + w*sr
#     target_base_rot = torch.stack([nb_w, nb_x, nb_y, nb_z], dim=1)
    
#     # Update Palm Orientation
#     w, x, y, z = palm_quat[:, 0], palm_quat[:, 1], palm_quat[:, 2], palm_quat[:, 3]
#     np_w = w*cr - z*sr
#     np_x = x*cr - y*sr
#     np_y = y*cr + x*sr
#     np_z = z*cr + w*sr
#     target_palm_rot = torch.stack([np_w, np_x, np_y, np_z], dim=1)
    
#     # Concatenate Poses
#     target_base_pose = torch.cat([target_base_pos, target_base_rot], dim=1)
#     target_palm_pose = torch.cat([target_palm_pos, target_palm_rot], dim=1)
    
#     # Elbow Hint: Keep it relative to base (or just above base)
#     target_elbow_pos = target_base_pos.clone()
#     target_elbow_pos[:, 2] += 1.0
    
#     # Solve IK and Step
#     q = solve_ik(robot, target_elbow_pos, target_palm_pose, target_base_pose)
#     write_joint_angle_to_robot(robot, q)
#     step_sim(scene, sim)
# """
#         codes = textwrap.dedent(codes)
#         exec(codes)

#         codes = """
# # Configuration
# num_steps = 40
# door_open_angle = 1.3
# handle_pressed_angle = 1.0

# # 1. Retrieve Current State
# # We need to lock the hand in space to hold the door
# current_palm_pos, current_palm_rot = get_robot_link_pose(robot, "palm")
# # Construct the fixed palm target pose
# target_palm_pose = torch.cat([current_palm_pos, current_palm_rot], dim=-1)

# # Get Pivot position (Door Frame) to determine where to move the base
# pivot_pos = get_board_pos(door) # (N, 3)

# # 2. Compute Target Base Pose
# # Strategy: Move base into the doorway gap.
# # X: Move to +0.4m (Robot is at +X, Door at 0). This puts base 40cm from the frame plane.
# # Y: Move to halfway between pivot (0) and current palm Y. This minimizes arm stretch.
# # Z: Keep at 0.
# target_base_pos = pivot_pos.clone()
# target_base_pos[:, 0] += 0.40 
# target_base_pos[:, 1] = current_palm_pos[:, 1] * 0.5 
# target_base_pos[:, 2] = 0.0

# # Orientation: Face -X (Traversal direction)
# # Quaternion for 180 deg Z rotation: (0, 0, 0, 1)
# target_base_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(pivot_pos.shape[0], 1).to(pivot_pos.device)
# target_base_pose = torch.cat([target_base_pos, target_base_rot], dim=-1)

# # Get current base for interpolation
# current_base_pos, current_base_rot = get_robot_link_pose(robot, "base")

# # 3. Execute Motion
# for i in range(num_steps):
#     t = (i + 1) / num_steps
    
#     # Interpolate Base Position (Linear)
#     interp_base_pos = current_base_pos * (1 - t) + target_base_pos * t
    
#     # Interpolate Base Rotation (Slerp-like or just Linear for small changes, but let's use Target directly if close, or just set target)
#     # Since we are moving and rotating, let's just feed the target pose to IK and let it converge over the path?
#     # No, to get a smooth trajectory for the arm, we should interpolate inputs to solve_ik.
#     # Simplified: Linear interp of quaternions (normalized later by solver or roughly valid)
#     interp_base_rot = current_base_rot * (1 - t) + target_base_rot * t
#     interp_base_rot = interp_base_rot / torch.norm(interp_base_rot, dim=-1, keepdim=True)
    
#     interp_base_pose = torch.cat([interp_base_pos, interp_base_rot], dim=-1)
    
#     # Keep Elbow up
#     target_elbow_pos = interp_base_pos.clone()
#     target_elbow_pos[:, 2] += 1.0
    
#     # Solve IK: Palm is FIXED, Base is MOVING
#     q = solve_ik(robot, elbow_pose=target_elbow_pos, palm_pose=target_palm_pose, base_pose=interp_base_pose)
    
#     # Write to Robot
#     write_joint_angle_to_robot(robot, q)
    
#     # Write to Door (Keep it open)
#     write_joint_angle_to_door(door, torch.tensor([door_open_angle]), torch.tensor([handle_pressed_angle]))
    
#     # Step Sim
#     step_sim(scene, sim)
#         """
#         codes = textwrap.dedent(codes)
#         exec(codes)
#         break

#         scene.write_data_to_sim()
#         # perform step
#         sim.step()
#         # update sim-time
#         sim_time += sim_dt
#         count += 1
#         # update buffers
#         scene.update(sim_dt)

#         # Extract camera data
#     single_cam_data = convert_dict_to_backend(
#         {k: v[camera_index] for k, v in scene_camera.data.output.items()}, backend="numpy"
#     )

#     # Extract the other information
#     single_cam_info = camera.data.info[camera_index]

#     # Pack data back into replicator format to save them using its writer
#     rep_output = {"annotators": {}}
#     for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
#         print(key, data.shape, info)
#         if info is not None:
#             rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
#         else:
#             rep_output["annotators"][key] = {"render_product": {"data": data}}
#     # Save images
#     # # Note: We need to provide On-time data for Replicator to save the images.
#     rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
#     rep_writer.write(rep_output)