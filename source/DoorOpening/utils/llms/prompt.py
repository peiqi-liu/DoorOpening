PULL_LEVER_PROMPT = """
Imagine you are a robot with a tidybot base, franka arm, and a leap hand.
You are about to open a door and traverse through it.
The door has a lever door hinge. You need to push the door hinge down in order to unlock the latch.
The door should be pulled to open.
Only after you rotate the door hinge s.t. the door hinge angle value's absolute is above an (unknown) threshold, the door board can be opened. This is the latching mechanisms model
The door hinge is on the right part of the board.
The door board-frame joint might be with some stiffness, that means AFTER OPENING IT, UNLESS YOU HOLD IT OPEN EITHER WITH THE ARM OR THE BASE, THE SPRING OF THE DOOR WILL CLOSE IT AUTOMATICALLY.
For simplicity the door will be sitting at (0, 0, 0), the board will be perpendiculer to x axis and the door hinge will be on the +x axis side, so will the robot be. 
While we know the robot is on the +x side, we don't know the exact position of the robot.

The robot's base is roughly 50 cm * 50 cm * 50 cm cube

The robot has the access to this information and tools 
(The input and returned output of each function, if being an array, will be expressed in torch tensor format, so don't use numpy array or list)
(All of the quaternions in these function calls will be in (w, x, y, z) convention, the same as that used in IsaacLab)
(If you see scene in a function's args, it means this function needs to write to the simulation)
(All joint angles or joint poses written to the function call or returned from function call would be in the shape of (N, -1), or (N, num_bodies, -1) instead of (-1) or (num_bodies, -1))
: 
1. get_board_frame_joint_angle(door): It takes the door articulation object as input, returns the angle value of the joint between the door frame and the door board. (The value wll range from [0, 1.57], by default it is 0, which means the door is closed, 1.57 means the door has been opened 90 degrees)
2. get_frame_hinge_joint_angle(door): It takes the door articulation object as input, returns the angle value of the hinge joint. (The value will range from [-1.57, 1.57], by default it is 0, when you rotate the hinge counterclockwise, this value will increase)
3. get_board_pos(door): It takes the door arituculation object as input, returns the door board's xyz pose in world frame (the uppermost point of board_frame_joint's rotation axis)
4. get_hinge_pos(door): It takes the door articulation object as input, returns the door hinge's xyz pose in world frame (but please note that this is the mass center of the hinge, but instead somewhere slightly above the actual hinge rotational axis)
5. solve_ik(robot, elbow_pose, palm_pose, base_pose): It is an ik solver taking the desired robot's elbow, palm, and base translation (and optionally rotation, [dx, dy, dz, (droll, dpitch, dyaw)]) as input and returns the target joint angles
    Some suggestions on using this function call:
    - While you only need to specify one pose's translation, you can optionally provide as much information as possible for the optiaml performance, especially you hope to avoid body collisions, reduce controlling null space as much as possible
        If you provide some inappropriate information, like the robot penetrates the door, or the elbow pose is not feasible or will influence reducing palm pose error, this trajectory will be very bad.
        You need to trade off between the risk of introducing erronous link poses and the risk of inaccurate trajectory due to the lack of essential link poses.
    - It is not an accurate algo, you should probably use it like:
        for _ in range(max_iters):
            q = solve_ik
            write_joint_angle_to_robot
            step_sim
6. close_hand(robot): It takes the robot articulation object as input, closes the robot hand to grasp. (This will only help you call robot.write_joint_state_to_sim, you still need to call step_sim later to actually make the changes)
7. open_hand(robot): It takes the robot articulation object as input, opens the robot hand to release. (This will only help you call robot.write_joint_state_to_sim, you still need to call step_sim later to actually make the changes)
8. write_joint_angle_to_robot(robot, target_joint_angle): It takes the robot articulation object as input, and it will write target_joint_angle (size: (10, ), the base (x, y, rotation) and the 7 panda joint angle of franka) to the simulator (This will only help you call robot.write_joint_state_to_sim, you still need to call step_sim later to actually make the changes)
9. record_joint_angles(robot, buffer): It will write the current robots' joint angles to the buffer
10. get_robot_link_pose(robot, link_name): It takes the robot and the link_name (palm, elbow, base) as input and returns the (x, y, z), (quaternions) as the output
11. write_joint_angle_to_door(door, target_board_joint_angle, target_hinge_joint_angle): It takes the door articulation object as input, and it will write target_door_joint_angle (size(2, ), the first joint is between the frame and the board and the second joint is for the hinge); Since we are only hardcoding the status to the simulation, we should also include all 
12. step_sim(scene, sim): After you write joint angle to the door or the robot, you should run this to step the simulation.

I hope you come up with a trajectory using tools above to help the robot open the door and traverse through the door in following matters:
1. Move to a good pregrasp pose (You should compute a base pose and a palm pose to do ik)
2. Push down the door lever to unlock the door (You should write the door hinge joint angle to the simulation and make sure your robot's hand should face downward or face towards the rotated hinge)
3. Pull the door open by 75 - 90 degrees (during this step, the robot's hand should still grasp the door hinge. That means you should first write the door board frame joint angle to the simulation, and then use solve_ik to force the robot's hand to be at the door hinge's position,
 and base moving backward, carefully determine whether you need to give the elbow pose to avoid collision with the door board)
4. Hold the door with the arm, the base move forward (you should ensure the robot's palm is still grasping the door hinge, in other words, do ik with the robot's current palm pose (or current hand pose) and the base pose moving forward)
5. When the base can stop the door from closing, realse your hand, and retract the arm WITHOUT colliding with the door (You should open hand, and do ik with the robot's retracted palm pose and the base pose staying still to block the door from closing)
6. Move forward and traverse through the frame (You should do ik with only the base pose moving forward, don't forget to set all door joint angles to zero in order to close the door manually)

This trajectory is not required to be super physically realistic, but it just needs to roughly makes sense. 
To do this, you need to come up with python codes to compute and determine the important waypoints in between. We can later interpolate these waypoints into a full trajectory.
The more waypoints you have, the better trajectories you are going to have, but if one waypoint is physically unrealistic, like the robot penetrates the door in this waypoint, this trajectory will be very bad.

Here is the pipeline, in each step, I will send you an image showing the status of the robot and the door, you will check that image, and your past commands, and a description of the current env status, to code how I can achieve the next waypoint.
Please keep in mind that we are just having a rough idea on how things would move, so even though you change the robots' states, the door might not be actually influenced. To change the doors' states, you need to use write_joint_angle_to_door

Example:
Input: <History Commands> <Images>.
Output: 
# Define simulation parameters
num_steps = 40
# Retrieve door handle position (this is the target for grasping)
handle_pos = get_hinge_pos(door) # Shape (N, 3)

# Define Target Base Pose
# Move base to be ~65cm away from the handle in X direction (robot is at +X, door at 0)
# We align the base Y with the handle Y to be centered on it
base_target_pos = handle_pos.clone()
base_target_pos[:, 0] += 0.65 
base_target_pos[:, 2] = 0.0 # Base is on the floor
# Base Orientation: Face -X direction (towards the door)
# Identity is +X. Rotation 180 deg (pi) around Z axis.
# Quaternion (w, x, y, z) for 180 Z rot: [0, 0, 0, 1]
base_target_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

# Define Target Palm Pose (Pre-grasp)
# Position: 10cm in front of handle (X), aligned in Y, 10cm above in Z to prepare for push down
palm_target_pos = handle_pos.clone()
palm_target_pos[:, 0] += 0.10
palm_target_pos[:, 2] += 0.10
# Orientation: Point fingers towards -X. 
# Rotate -90 deg around Y axis: [0.707, 0, -0.707, 0]
palm_target_rot = torch.tensor([[0.7071, 0.0, -0.7071, 0.0]]).repeat(handle_pos.shape[0], 1).to(handle_pos.device)
palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)

# Prepare Hand
open_hand(robot)

# Execute Motion
for _ in range(num_steps):
    # Solve IK for the defined targets
    # We provide full 7D poses for base and palm to constrain orientation
    q = solve_ik(robot, palm_pose=palm_target_pose, base_pose=base_target_pose)
    
    # Write to robot and step simulation
    write_joint_angle_to_robot(robot, q)
    step_sim(scene, sim)

PLEASE DONT INCLUDE ANY UNNECESSARY INDENT INSIDE YOUR CODES! AND PLEASE ONLY INCLUDE THE CODE in your output, NO OTHER TEXT such as "reasoning:", "output:", "answer:", etc.

You should process one step at a time, even one step at 2 times if you need to gather more waypoints for this step (but no more than 2 times).
If you feel the last step is not finished nicely, you should process one more time (with optimized codes), but no more than 2 times.
If the all steps are finished, instead of generating any codes, please send one single message "FINISHED". (no codes, no other text, no indent, no comma)
"""