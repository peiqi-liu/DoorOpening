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

from DoorOpening.assets.door.door_cfg import ALL_DOOR_CONFIGS


torch.set_printoptions(precision=4, sci_mode=False)

def _body_names(door):
    names = getattr(door, "body_names", None)
    if names is None:
        names = getattr(door.data, "body_names", None)
    return list(names) if names is not None else []


def _joint_names(door):
    names = getattr(door, "joint_names", None)
    if names is None:
        names = getattr(door.data, "joint_names", None)
    return list(names) if names is not None else []


def _joint_index(door, joint_name, fallback_idx):
    names = _joint_names(door)
    if joint_name in names:
        return names.index(joint_name)
    num_joints = door.data.joint_pos.shape[1]
    return min(fallback_idx, max(0, num_joints - 1))


def _body_indices_for_stats(door):
    names = _body_names(door)
    if names:
        wanted_names = ("link_1", "link_2")
        indices = [(names.index(name), name) for name in wanted_names if name in names]
        if indices:
            return indices

    num_bodies = door.data.body_pos_w.shape[1]
    fallback_indices = [idx for idx in (2, 3) if idx < num_bodies]
    return [(idx, f"body_{idx}") for idx in fallback_indices]


def _format_vec(values):
    return "[" + ", ".join(f"{value:.4f}" for value in values.tolist()) + "]"


def print_door_body_position_stats(scene: InteractiveScene, count: int):
    door = scene["door"]
    body_pos_w = door.data.body_pos_w
    body_pos_env = body_pos_w - scene.env_origins[:, None, :]

    print(f"[DOOR POS STATS] step={count}")
    for body_idx, body_name in _body_indices_for_stats(door):
        pos_w = body_pos_w[:, body_idx, :]
        pos_env = body_pos_env[:, body_idx, :]
        print(
            f"  {body_name}[{body_idx}] world_z "
            f"max={pos_w[:, 2].max().item():.4f} "
            f"min={pos_w[:, 2].min().item():.4f} "
            f"std={pos_w[:, 2].std(unbiased=False).item():.4f} "
            f"var={pos_w[:, 2].var(unbiased=False).item():.6f}"
        )
        print(
            f"  {body_name}[{body_idx}] env_pos "
            f"mean={_format_vec(pos_env.mean(dim=0))} "
            f"std={_format_vec(pos_env.std(dim=0, unbiased=False))} "
            f"min={_format_vec(pos_env.min(dim=0).values)} "
            f"max={_format_vec(pos_env.max(dim=0).values)}"
        )


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
    door = scene["door"]
    joint_1_idx = _joint_index(door, "joint_1", fallback_idx=0)
    joint_2_idx = _joint_index(door, "joint_2", fallback_idx=1)

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
            door_pos = door.data.joint_pos_limits[..., 0].clone()
            door.write_joint_position_to_sim(door_pos)
            # clear internal buffers
            scene.reset()
            # print("joint_pos: ", scene["door"].data.joint_pos)

        joint_lower = door.data.soft_joint_pos_limits[..., 0]
        joint_upper = door.data.joint_pos_limits[..., 1]
        door_target_pos = joint_lower.clone()
        door_target_pos[:, joint_1_idx] = (
            (joint_upper[:, joint_1_idx] - joint_lower[:, joint_1_idx]) * ((count % 500) / 500)
            + joint_lower[:, joint_1_idx]
        )
        door_target_pos[:, joint_2_idx] = (
            (joint_upper[:, joint_2_idx] - joint_lower[:, joint_2_idx]) * ((count % 500) / 500)
            + joint_lower[:, joint_2_idx]
        )
        # door.write_joint_position_to_sim(door_target_pos)
        if count % 100 == 0:
            # print("joint_pos: ", scene["door"].data.joint_pos)
            print("door pos: ", scene["door"].data.body_pos_w[..., 2, 2].max(), scene["door"].data.body_pos_w[..., 2, 2].min())
            print_door_body_position_stats(scene, count)
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
