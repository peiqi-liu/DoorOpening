# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

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

from DoorOpening.assets.door.door_cfg import ALL_DOOR_CONFIGS


torch.set_printoptions(precision=4, sci_mode=False)

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    door : ArticulationCfg = ALL_DOOR_CONFIGS.replace(
        prim_path="{ENV_REGEX_NS}/Door",
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # door_generator = BASIC_DOOR_CFG

    # doors = []

    # for env_id in range(scene.num_envs):
    #     env_ns = scene.env_ns[env_id]
    #     prim_path = f"{env_ns}/Door"

    #     door = door_generator.spawn(
    #         prim_path=prim_path,
    #         translation=(0.0, 0.0, 0.0),
    #     )

    #     doors.append(door)

    # scene.add_articulation("door", doors)

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            print("Resetting door state...")
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            root_state = scene["door"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["door"].write_root_pose_to_sim(root_state[:, :7])
            scene["door"].write_root_velocity_to_sim(root_state[:, 7:])
            door_pos = scene["door"].data.joint_pos_limits[..., 0]
            scene["door"].write_joint_position_to_sim(door_pos)
            # clear internal buffers
            scene.reset()
            # print("joint_pos: ", scene["door"].data.joint_pos)

        # door_target_pos = (scene["door"].data.joint_pos_limits[..., 1] - scene["door"].data.soft_joint_pos_limits[..., 0]) * ((count % 500) / 500) + scene["door"].data.soft_joint_pos_limits[..., 0]
        # scene["door"].write_joint_position_to_sim(door_target_pos)
        # if count % 100 == 0:
        #     print("joint_pos: ", scene["door"].data.joint_pos)
        #     print("door pos: ", scene["door"].data.body_pos_w)
        #     print("effort_limit_sim: ", scene["door"].data.effort_limit_sim)
        #     print("velocity_limit_sim: ", scene["door"].data.velocity_limit_sim)
        #     print("position_limit_sim: ", scene["door"].data.position_limit_sim)
        #     print("velocity_limit_sim: ", scene["door"].data.velocity_limit_sim)
        #     print("velocity_limit_sim: ", scene["door"].data.velocity_limit_sim)
        #     print("joint_pos_target: ", door_target_pos)
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
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 3.0], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
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
