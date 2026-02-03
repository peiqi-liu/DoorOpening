# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

from operator import truediv
import os
import argparse

from isaaclab.app import AppLauncher
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="LLM agent for door opening.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--door_number", type=int, default=0, help="Door number.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

import omni.replicator.core as rep
from isaaclab.utils import convert_dict_to_backend

from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG, DEFAULT_JOINT_POS   
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG, DOOR_CONFIGS

from isaaclab.utils.math import quat_from_euler_xyz
import numpy as np
import pickle as pkl

euler_angles = torch.tensor([-np.pi / 4, 0.0, 0])  # (roll, pitch, yaw) in radians
quat = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2]) 

scene_euler_angles = torch.tensor([0.0, np.pi / 8, -np.pi - np.pi / 4])  # (roll, pitch, yaw) in radians
scene_quat = quat_from_euler_xyz(scene_euler_angles[0], scene_euler_angles[1], scene_euler_angles[2]) 
print("scene_quat:", scene_quat)

from DoorOpening.utils.llms.llm_utils import *
from DoorOpening.utils.llms.llm_api import GeminiAgent
from DoorOpening.utils.llms.prompt import PULL_LEVER_PROMPT
from isaaclab.sim.utils import stage as stage_utils

# llm_agent = GeminiAgent(prompt=PULL_LEVER_PROMPT, model="gemini-3-pro-preview")

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = GLORBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=DEFAULT_JOINT_POS,
            pos=(1.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0)
        ),
    )

    door: ArticulationCfg = DOOR_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Door"
    )

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x5_camera_link/cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=quat, convention="world"),
    )

    scene_camera = CameraCfg(
        prim_path="/World/SceneCamera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            # pos=[4.0, -4.0, 3.5],
            # rot=scene_quat,
            pos = [4.0, -4.0, 3.5],
            rot = scene_quat,
            convention="world",
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    camera = scene["camera"]
    scene_camera = scene["scene_camera"]

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    camera_index = 0

    # Simulate physics
    while simulation_app.is_running():
        root_state = scene["robot"].data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        scene["robot"].write_root_pose_to_sim(root_state[:, :7])
        scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
        # set joint positions with some noise
        joint_pos, joint_vel = (
            scene["robot"].data.default_joint_pos.clone(),
            scene["robot"].data.default_joint_vel.clone(),
        )
        scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
        # clear internal buffers
        scene.reset()
        print("[INFO]: Resetting robot state...")
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update buffers
        scene.update(sim_dt)
        print("updating buffers")
        code_list = []
        # api_call_and_code_execution(scene, sim, code_list)
        # final_code = "\n".join(code_list)
        # with open("code.py", "w") as f:
        #     f.write(final_code)
        door = scene["door"]
        robot = scene["robot"]
        buffer = []
        # from DoorOpening.utils.llms.code import codes
        # exec(codes)
        from DoorOpening.utils.llms.code import state_machine
        state_machine(robot, door, scene, sim, buffer)
        print("buffer: ", buffer)
        collocate_and_playback(scene, sim, robot, door, buffer)
        break

def collocate_and_playback(scene, sim, robot, door, key_joint_angles, length=1000):
    from scipy.interpolate import CubicSpline
    qs = torch.stack([kp[0] for kp in key_joint_angles]).detach().cpu().numpy()

    # automatic timing
    dq = np.linalg.norm(qs[1:] - qs[:-1], axis=1)
    dt = np.maximum(dq / 1.0, 0.1)
    t_key = np.concatenate([[0], np.cumsum(dt)])
    t_key /= t_key[-1]

    cs = CubicSpline(t_key, qs, axis=0, bc_type="clamped")

    t = np.linspace(0, 1, length)
    traj = torch.tensor(cs(t))
    traj_d = torch.tensor(cs(t, 1))

    key_indices = np.searchsorted(t, t_key)
    key_indices = np.clip(key_indices, 0, len(t) - 1)

    door_traj = traj[:, -2:]
    traj_quat = traj[:, 2 * 3:-2]
    traj = traj[:, :2 * 3]

    playback_and_save_traj(scene, sim, robot, door, door_traj, traj, traj_quat, traj_d, key_indices)


def playback_and_save_traj(scene, sim, robot, door, door_traj, traj, traj_quat, traj_d, key_indices):
    robot_body_pos_traj = []
    robot_body_quat_traj = []
    door_pos_traj = []
    robot_joint_angle_traj = []
    robot_base_vel_traj = []
    robot_palm_vel_traj = []

    robot.write_joint_position_to_sim(robot.data.default_joint_pos)
    step_sim(scene, sim)
    for door_point, robot_points, robot_quat, vel in zip(door_traj, traj, traj_quat, traj_d):
        vel = vel[:2*3]
        vel = vel.reshape(2, -1)
        base_vel, palm_vel = vel[0], vel[1]

        robot_points = robot_points.reshape(2, -1)
        base_pose, palm_pose = robot_points[0], robot_points[1]
        
        robot_quat = robot_quat.reshape(2, -1)
        base_quat, palm_quat = robot_quat[0], robot_quat[1]
        
        base_pose = torch.cat((base_pose, base_quat), dim=-1).unsqueeze(0)
        palm_pose = torch.cat((palm_pose, palm_quat), dim=-1).unsqueeze(0)
        
        door.write_joint_position_to_sim(door_point)
        joint_pos_des = solve_ik(robot, base_pose=base_pose, palm_pose=palm_pose)
        write_joint_angle_to_robot(robot, joint_pos_des)

        step_sim(scene, sim)
        robot_body_pos_traj.append(robot.data.body_pos_w.squeeze().cpu().clone())
        robot_body_quat_traj.append(robot.data.body_quat_w.squeeze().cpu().clone())
        door_pos_traj.append(door.data.body_pos_w.squeeze().cpu().clone())
        robot_joint_angle_traj.append(robot.data.joint_pos.squeeze().cpu().clone())

        robot_base_vel_traj.append(base_vel.squeeze().cpu().clone())
        robot_palm_vel_traj.append(palm_vel.squeeze().cpu().clone())
    
    robot_body_pos_traj = torch.stack(robot_body_pos_traj, dim = 0)
    robot_body_quat_traj = torch.stack(robot_body_quat_traj, dim = 0)
    door_pos_traj = torch.stack(door_pos_traj, dim = 0)
    robot_joint_angle_traj = torch.stack(robot_joint_angle_traj, dim = 0)
    robot_base_vel_traj = torch.stack(robot_base_vel_traj, dim = 0)
    robot_palm_vel_traj = torch.stack(robot_palm_vel_traj, dim = 0)
    # print("robot_base_vel_traj: ", robot_base_vel_traj.shape)
    # print("robot_palm_vel_traj: ", robot_palm_vel_traj.shape)

    data = {
        "door_traj": door_traj, 
        "robot_body_pos_traj": robot_body_pos_traj,
        "robot_body_quat_traj": robot_body_quat_traj,
        "door_pos_traj": door_pos_traj,
        "robot_joint_pos_traj": robot_joint_angle_traj,
        "robot_base_vel_traj": robot_base_vel_traj,
        "robot_palm_vel_traj": robot_palm_vel_traj,
        "key_indices": torch.from_numpy(key_indices)
    }

    # answer = input("Do you want to save the trajectory? (y/n)")
    answer = "y"
    if answer.lower() == "y":
        dir_path = os.path.dirname(door.cfg.spawn.asset_path)
        with open(os.path.join(dir_path, "traj.pkl"), "wb") as f:
            pkl.dump(data, f)
            print(f"Trajectory saved to {os.path.join(dir_path, 'traj.pkl')}")
    else:
        print("Trajectory not saved")


def api_call_and_code_execution(scene, sim, code_list):
    door = scene["door"]
    robot = scene["robot"]
    scene_camera = scene["scene_camera"]
    camera_index = 0
    count = 0
    while count < 9:
        image = get_image_from_scene_camera(scene_camera, camera_index)
        current_state = get_current_state(door, robot)
        response = llm_agent.query(image, current_state)
        print("response:", response)
        if "finished" in response.lower():
            break
        code_list.append(response)
        exec(response)
        count += 1

def get_image_from_scene_camera(scene_camera, camera_index: int):
    return scene_camera.data.output["rgb"][camera_index]

def get_current_state(door, robot):
    current_state = ""
    current_state += "The door board frame joint angle is " + str(get_board_frame_joint_angle(door)) + "\n"
    current_state += "The hinge joint angle is " + str(get_frame_hinge_joint_angle(door)) + "\n"
    current_state += "The door board position is " + str(get_board_pos(door)) + "\n"
    current_state += "The hinge position is " + str(get_hinge_pos(door)) + "\n"
    current_state += "The robot base position is " + str(get_robot_link_pose(robot, "base")) + "\n"
    current_state += "The robot palm position is " + str(get_robot_link_pose(robot, "palm")) + "\n"
    current_state += "The robot elbow position is " + str(get_robot_link_pose(robot, "elbow")) + "\n"
    return current_state

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[4.0, -4.0, 3.5], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    door_cfg = DOOR_CONFIGS[args_cli.door_number].replace(prim_path="{ENV_REGEX_NS}/Door")
    scene_cfg.door = door_cfg
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # print("closing simulation app")
    # simulation_app.close()
    # print("simulation app closed")
    os._exit(0)
