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

from DoorOpening.assets.glorbot.glorbot_cfg import FULL_JOINT_NAMES

from DoorOpening.motion.slider_controller import OmniJointController

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz

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
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics

    controller = OmniJointController(scene, FULL_JOINT_NAMES)

    cfg = FRAME_MARKER_CFG.replace(prim_path="/World/GoalFrame")
    cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    goal_marker = VisualizationMarkers(cfg)

    # initialize marker at current EE pose
    roll, pitch, yaw = controller.euler_angles.unbind(dim=-1)
    goal_quat = quat_from_euler_xyz(roll, pitch, yaw)

    goal_marker.visualize(
        translations=controller.xyz,
        orientations=goal_quat,
    )

    # give controller access to marker
    controller.goal_marker = goal_marker
    

    while simulation_app.is_running():
        # Reset
        if count == 0:
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            print("root state: ", root_state)
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
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

        slider_pos = controller.q_slider.clone()
        joint_pos = scene["robot"].data.default_joint_pos.clone()
        joint_pos[..., :] = slider_pos

        door_pos = controller.door_q_slider.clone()
        door_joint_pos = scene["door"].data.default_joint_pos.clone()
        door_joint_pos[..., :] = door_pos
        # print("door_joint_pos: ", door_joint_pos)
        # scene["robot"].write_joint_position_to_sim(joint_pos)
        scene["door"].write_joint_position_to_sim(door_joint_pos)
        scene["robot"].set_joint_position_target(joint_pos)
        # scene["door"].set_joint_position_target(door_joint_pos)
        # -- write data to sim
        if controller.playback:
            q = controller.traj[controller.play_idx]
            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_pos[..., :] = q
            scene["robot"].write_joint_position_to_sim(joint_pos)

            door_q = controller.door_traj[controller.play_idx]
            door_joint_pos = scene["door"].data.default_joint_pos.clone()
            door_joint_pos[..., :] = door_q
            scene["door"].write_joint_position_to_sim(door_joint_pos)

            # controller.door_joint_pos_traj.append(scene["door"].data.joint_pos.squeeze().cpu().clone())
            controller.robot_body_pos_traj.append(scene["robot"].data.body_pos_w.squeeze().cpu().clone())
            controller.robot_body_quat_traj.append(scene["robot"].data.body_quat_w.squeeze().cpu().clone())

            controller.play_idx += 1
            if controller.play_idx >= len(controller.traj):
                controller.playback = False
                print("[PLAYBACK] Finished")
        
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
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device)
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
