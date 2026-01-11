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

import pickle as pkl
import argparse

from isaaclab.app import AppLauncher
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
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
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg

from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG, DEFAULT_JOINT_POS
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG

from DoorOpening.motion.motion_generator import MotionGenerator

from isaaclab.utils.math import quat_from_euler_xyz

from DoorOpening.assets.glorbot.glorbot_cfg import DM_JOINT_NAMES

euler_angles = torch.tensor([0.0, 0.0, np.pi])  # (roll, pitch, yaw) in radians
quat = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2])

torch.set_printoptions(precision=3, sci_mode=False)

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
        prim_path="{ENV_REGEX_NS}/Door",
        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(-1.0, 0.0, 0.75),
        #     rot=[1.0, 0.0, 0.0, 0.0]
        # )
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    motion_generator = MotionGenerator(scene, device=args_cli.device)
    # Simulate physics

    # Initialize the trajectory buffers
    joint_ids, _ = scene["robot"].find_joints(DM_JOINT_NAMES)
    robot_trajs = [scene["robot"].data.joint_pos.squeeze().cpu()]
    door_trajs = [scene["door"].data.joint_pos.squeeze().cpu()]

    key_body_ids, _ = scene["robot"].find_bodies(["base_x_link", "palm_center"])

    while simulation_app.is_running():
        # Reset
        if count % 700 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            print("root state: ", root_state)
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            root_state_door = scene["door"].data.default_root_state.clone()
            root_state_door[:, :3] += scene.env_origins
            print("root state door: ", root_state_door)
            scene["door"].write_root_pose_to_sim(root_state_door[:, :7])
            scene["door"].write_root_velocity_to_sim(root_state_door[:, 7:])
            door_pos = torch.zeros_like(scene["door"].data.soft_joint_pos_limits[..., 0])
            scene["door"].write_joint_position_to_sim(door_pos)

            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands


        # Move to the door

        if count < 150:
            joint_pos = motion_generator.compute_approach_target()
            # print("actions: ", actions)
            scene["robot"].set_joint_position_target(joint_pos)
            ik_count = 0
            move_away_count = 0

            # Write data to buffers
            robot_trajs.append(scene["robot"].data.joint_pos.squeeze().cpu().clone())
            door_trajs.append(scene["door"].data.joint_pos.squeeze().cpu().clone())

        # Open the door

        elif ik_count < 25:
            print("ik_count: ", ik_count)
            ik_joint_pos = motion_generator.door_opening_motion()
            if ik_joint_pos is not None:
                scene["robot"].write_joint_position_to_sim(ik_joint_pos)
                
                # Write data to buffers
                record_pos = scene["robot"].data.joint_pos.squeeze().cpu().clone()
                record_door_pos = scene["door"].data.joint_pos.squeeze().cpu().clone()
                for i in range(1, 25 + 1):
                    new_waypoint = robot_trajs[len(robot_trajs)-1] + (record_pos - robot_trajs[len(robot_trajs)-1]) / 25
                    robot_trajs.append(new_waypoint.cpu().clone())
                    # new_door_waypoint = door_trajs[len(door_trajs)-1] + (record_door_pos - door_trajs[len(door_trajs)-1]) / 15
                    # door_trajs.append(new_door_waypoint.cpu().clone())
                
                ik_count += 1

        # Move away from the door

        # elif move_away_count < 5:
        #     ik_joint_pos = motion_generator.move_away_from_door()
        #     scene["robot"].write_joint_position_to_sim(ik_joint_pos)

        #     # Write data to buffers
        #     record_pos = scene["robot"].data.joint_pos.squeeze().cpu()
        #     record_door_pos = scene["door"].data.joint_pos.squeeze().cpu()
        #     for i in range(1, 30 + 1):
        #         new_waypoint = robot_trajs[len(robot_trajs)-1] + (record_pos - robot_trajs[len(robot_trajs)-1]) / 30
        #         robot_trajs.append(new_waypoint.cpu())
        #         # new_door_waypoint = door_trajs[len(door_trajs)-1] + (record_door_pos - door_trajs[len(door_trajs)-1]) / 30
        #         # door_trajs.append(new_door_waypoint.cpu())

        #     move_away_count += 1
        
        else:
            while len(robot_trajs) > len(door_trajs):
                door_trajs.append(scene["door"].data.joint_pos.squeeze().cpu().clone())
            robot_trajs = torch.stack(robot_trajs, dim = 0)
            door_trajs = torch.stack(door_trajs, dim = 0)
            robot_body_pos_traj = []
            robot_body_quat_traj = []

            for robot_traj, door_traj in zip(robot_trajs, door_trajs):
                robot_full_joint_pos = robot_traj
                scene["robot"].write_joint_position_to_sim(robot_full_joint_pos)
                scene["door"].write_joint_position_to_sim(door_traj)
                scene.write_data_to_sim()
                sim.step()
                sim_time += sim_dt
                scene.update(sim_dt)

                robot_body_pos_traj.append(scene["robot"].data.body_pos_w.cpu())
                robot_body_quat_traj.append(scene["robot"].data.body_quat_w.cpu())

            robot_body_pos_traj = torch.stack(robot_body_pos_traj, dim = 0)
            robot_body_quat_traj = torch.stack(robot_body_quat_traj, dim = 0)
            
            motions = {
                "robot_joint_pos_traj": robot_trajs,
                "door_joint_pos_traj": door_trajs,
                "robot_body_pos_traj": robot_body_pos_traj,
                "robot_body_quat_traj": robot_body_quat_traj,
            }
            with open("traj.pkl", "wb") as f:
                pkl.dump(motions, f)
            print("Saved trajectory to traj.pkl")
            break

        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.0, -2.5, 3.2], target=[0.0, 0.0, 0.7])
    # Design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
