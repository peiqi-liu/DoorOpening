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
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg

from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG, DEFAULT_JOINT_POS
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG

from DoorOpening.motion.motion_generator import MotionGenerator

from isaaclab.utils.math import quat_from_euler_xyz

euler_angles = torch.tensor([0.0, 0.0, np.pi * 4 / 5])  # (roll, pitch, yaw) in radians
quat = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2]) 

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
            pos=(1.0, 0, 0.0),
            # rot=[0.707, 0, 0, 0.707]
            rot = quat
        ),
    )

    door: ArticulationCfg = DOOR_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Door",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.0, 0.0, 0.75),
            rot=[1.0, 0.0, 0.0, 0.0]
        )
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    motion_generator = MotionGenerator(scene, device=args_cli.device)
    motion_generator.reset()
    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 1500 == 0:
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
            # door_pos = scene["door"].data.soft_joint_pos_limits[..., 0]
            door_pos = torch.zeros_like(scene["door"].data.soft_joint_pos_limits[..., 0])
            scene["door"].write_joint_position_to_sim(door_pos)

            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands

        if count < 500:
            actions = motion_generator.compute_approach_target()
            joint_pos = scene["robot"].data.default_joint_pos.clone()
            # print("joint_pos: ", joint_pos[..., :3])
            joint_pos[..., :3] = actions
            # print("actions: ", actions)
            scene["robot"].set_joint_position_target(joint_pos)

        else:
            ik_joint_pos = motion_generator.compute_arm_target()
            if count % 50 == 0:
                motion_generator.compute_arm_target()
                FRANKA_JOINT_NAMES = [
                    'panda_joint1',
                    'panda_joint2',
                    'panda_joint3',
                    'panda_joint4',
                    'panda_joint5',
                    'panda_joint6',
                    'panda_joint7',
                ]
                hand_idx = scene["robot"].find_joints(FRANKA_JOINT_NAMES)[0]
                print("ik_joint_pos: ", ik_joint_pos[:, hand_idx])
            scene["robot"].set_joint_position_target(ik_joint_pos)

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
    sim.set_camera_view(eye=[0.0, 0.0, 2.0], target=[0.0, 0.0, 0.7])
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
