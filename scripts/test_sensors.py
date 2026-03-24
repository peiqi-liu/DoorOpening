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

import os
import argparse
import time

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
parser.add_argument(
    "--benchmark_pointcloud",
    action="store_true",
    default=False,
    help="Benchmark depth-to-pointcloud against FrankaLeapSampler on matching simulation frames.",
)
parser.add_argument(
    "--benchmark_warmup_frames",
    type=int,
    default=5,
    help="Number of camera frames to warm up before timing pointcloud methods.",
)
parser.add_argument(
    "--benchmark_frames",
    type=int,
    default=50,
    help="Number of camera frames to time when benchmarking pointcloud methods.",
)
parser.add_argument(
    "--benchmark_num_points",
    type=int,
    default=1000,
    help="Number of points to keep in each benchmarked pointcloud.",
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
from isaaclab.actuators import ImplicitActuatorCfg

import omni.replicator.core as rep
from isaaclab.utils import convert_dict_to_backend

from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG, door_asset_path

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT

from DoorOpening.utils.camera_utils import depth_to_pointcloud
from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaLeapSampler


euler_angles = torch.tensor([-np.pi / 4, 0.0, 0])  # (roll, pitch, yaw) in radians
quat = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2])


def _sync_timing_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_timing_stats(values_ms: list[float]) -> str:
    values = np.asarray(values_ms, dtype=np.float64)
    return (
        f"mean={values.mean():.3f} ms | median={np.median(values):.3f} ms | "
        f"p90={np.percentile(values, 90):.3f} ms | min={values.min():.3f} ms | max={values.max():.3f} ms"
    )


def _print_benchmark_summary(benchmark_state: dict, num_envs: int):
    depth_mean = float(np.mean(benchmark_state["depth_ms"]))
    sampler_mean = float(np.mean(benchmark_state["sampler_ms"]))
    sampler_total_mean = float(np.mean(benchmark_state["sampler_total_ms"]))

    raw_faster_name, raw_slower_name = (
        ("depth_to_pointcloud", "FrankaLeapSampler.sample")
        if depth_mean <= sampler_mean
        else ("FrankaLeapSampler.sample", "depth_to_pointcloud")
    )
    raw_speedup = max(depth_mean, sampler_mean) / max(min(depth_mean, sampler_mean), 1e-9)

    total_faster_name, total_slower_name = (
        ("depth_to_pointcloud", "FrankaLeapSampler.sample + world transform")
        if depth_mean <= sampler_total_mean
        else ("FrankaLeapSampler.sample + world transform", "depth_to_pointcloud")
    )
    total_speedup = max(depth_mean, sampler_total_mean) / max(min(depth_mean, sampler_total_mean), 1e-9)

    print(
        f"[INFO]: Pointcloud benchmark summary ({benchmark_state['timed_frames']} camera frames, "
        f"{num_envs} envs, {args_cli.benchmark_num_points} points)"
    )
    print(f"[INFO]: depth_to_pointcloud output shape: {benchmark_state['depth_shape']}")
    print(f"[INFO]: FrankaLeapSampler output shape: {benchmark_state['sampler_shape']}")
    print(
        "[INFO]: depth_to_pointcloud timing is projection + crop only; "
        "camera rendering cost is not included in this number."
    )
    print(f"[INFO]: depth_to_pointcloud: {_format_timing_stats(benchmark_state['depth_ms'])}")
    print(f"[INFO]: FrankaLeapSampler.sample: {_format_timing_stats(benchmark_state['sampler_ms'])}")
    print(
        "[INFO]: FrankaLeapSampler.sample + world transform: "
        f"{_format_timing_stats(benchmark_state['sampler_total_ms'])}"
    )
    print(
        f"[INFO]: Raw generation winner: {raw_faster_name} "
        f"({raw_speedup:.2f}x faster than {raw_slower_name} on mean time)."
    )
    print(
        f"[INFO]: Including sampler world transform, winner: {total_faster_name} "
        f"({total_speedup:.2f}x faster than {total_slower_name} on mean time)."
    )

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
        # data_types=["rgb", "distance_to_image_plane"],
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8.0, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=quat, convention="world"),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    camera = scene["camera"]

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    camera_index = 0

    benchmark_state = None
    if args_cli.benchmark_pointcloud:
        timing_device = torch.device(args_cli.device)
        door_base_body_idx = int(scene["door"].find_bodies("base")[0][0])
        benchmark_state = {
            "device": timing_device,
            "sampler": FrankaLeapSampler(
                door_asset_path,
                device=args_cli.device,
                num_points=args_cli.benchmark_num_points,
            ),
            "door_base_body_idx": door_base_body_idx,
            "last_camera_frame": -1,
            "warmup_frames_left": args_cli.benchmark_warmup_frames,
            "timed_frames": 0,
            "depth_ms": [],
            "sampler_ms": [],
            "sampler_total_ms": [],
            "depth_shape": None,
            "sampler_shape": None,
        }
        print(
            f"[INFO]: Benchmarking pointcloud methods for {args_cli.benchmark_frames} camera frames "
            f"after {args_cli.benchmark_warmup_frames} warmup frames."
        )

    targets = scene["robot"].data.default_joint_pos.clone()
    # targets[..., :2] += 0.5

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands

        joint_names_list = scene["robot"].joint_names
        # print(scene["robot"].data.soft_joint_pos_limits)
        # print(scene["robot"].data.joint_names)
        # print(len(scene["robot"].data.joint_names))
        # print(scene["robot"].data.body_pos_w)
        # print(len(scene["robot"].data.body_pos_w[0]))

        # print(scene["robot"].data.body_names)
        # print(len(scene["robot"].data.body_names))

        # print("body_pos_w: ", scene["door"].data.body_pos_w)
        # print("body_names: ", scene["door"].data.body_names)
        # print("joint_names: ", scene["door"].data.joint_names)
        # print("joint_pos: ", scene["door"].data.joint_pos)
        # print("lower limit: ", scene["door"].data.soft_joint_pos_limits[..., 0])
        # print("upper limit: ", scene["door"].data.soft_joint_pos_limits[..., 1])
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        if benchmark_state is not None:
            current_camera_frame = int(camera.frame[0].item())
            if current_camera_frame != benchmark_state["last_camera_frame"]:
                benchmark_state["last_camera_frame"] = current_camera_frame
                depth = camera.data.output["distance_to_image_plane"]
                door_joint_pos = scene["door"].data.joint_pos

                if benchmark_state["warmup_frames_left"] > 0:
                    depth_to_pointcloud(
                        depth,
                        debug=False,
                        num_local_points=args_cli.benchmark_num_points,
                    )
                    benchmark_state["sampler"].sample(door_joint_pos)
                    _sync_timing_device(benchmark_state["device"])
                    benchmark_state["warmup_frames_left"] -= 1
                else:
                    _sync_timing_device(benchmark_state["device"])
                    start_time = time.perf_counter()
                    depth_pcd = depth_to_pointcloud(
                        depth,
                        debug=False,
                        num_local_points=args_cli.benchmark_num_points,
                    )
                    _sync_timing_device(benchmark_state["device"])
                    depth_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    benchmark_state["depth_ms"].append(depth_elapsed_ms)

                    _sync_timing_device(benchmark_state["device"])
                    start_time = time.perf_counter()
                    sampled_local_pcd = benchmark_state["sampler"].sample(door_joint_pos)
                    _sync_timing_device(benchmark_state["device"])
                    sampler_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    benchmark_state["sampler_ms"].append(sampler_elapsed_ms)

                    door_base_pos_w = scene["door"].data.body_pos_w[:, benchmark_state["door_base_body_idx"]]
                    door_base_quat_w = scene["door"].data.body_quat_w[:, benchmark_state["door_base_body_idx"]]
                    _sync_timing_device(benchmark_state["device"])
                    start_time = time.perf_counter()
                    quat = door_base_quat_w.unsqueeze(1).expand(-1, sampled_local_pcd.shape[1], -1)
                    quat_apply(quat, sampled_local_pcd) + door_base_pos_w.unsqueeze(1)
                    _sync_timing_device(benchmark_state["device"])
                    world_transform_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    benchmark_state["sampler_total_ms"].append(sampler_elapsed_ms + world_transform_elapsed_ms)

                    if benchmark_state["depth_shape"] is None:
                        benchmark_state["depth_shape"] = tuple(depth_pcd.shape)
                    if benchmark_state["sampler_shape"] is None:
                        benchmark_state["sampler_shape"] = tuple(sampled_local_pcd.shape)

                    benchmark_state["timed_frames"] += 1
                    if benchmark_state["timed_frames"] >= args_cli.benchmark_frames:
                        _print_benchmark_summary(benchmark_state, scene.num_envs)
                        break

        # Extract camera data
        if args_cli.save and count % 100 == 0:
            # Save images from camera at camera_index
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )

            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                print(depth_to_pointcloud(data).shape)
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

        # print information from the sensors
        # print("-------------------------------")
        # print(scene["camera"])
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)


def main():
    """Main function."""

    if (args_cli.save or args_cli.benchmark_pointcloud) and not getattr(args_cli, "enable_cameras", False):
        raise ValueError("--enable_cameras is required when saving or benchmarking camera pointclouds.")

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[4.0, -4.0, 3.5], target=[0.0, 0.0, 0.0])
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
