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
    help="Benchmark depth-to-pointcloud against FrankaGripperSampler on matching simulation frames.",
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
parser.add_argument(
    "--save_lidar_points",
    action="store_true",
    default=False,
    help="Save rendered lidar points to .ply using Open3D.",
)
parser.add_argument(
    "--log_viser",
    action="store_true",
    default=False,
    help="Stream rendered lidar points to a viser pointcloud viewer.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Preload viser before AppLauncher mutates import paths via Isaac Sim/Kit.
# This avoids importing an incompatible `websockets` module from Isaac Sim internals.
VISER_MODULE = None
VISER_IMPORT_ERROR = None
if args_cli.log_viser:
    try:
        import websockets.asyncio.server  # noqa: F401
        import viser as _viser_module

        VISER_MODULE = _viser_module
    except Exception as exc:
        VISER_IMPORT_ERROR = exc

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

from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG, glorbot_urdf_path
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG, door_asset_path

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT

from DoorOpening.utils.camera_utils import depth_to_pointcloud, simulate_lidar_render_from_pose
from DoorOpening.constants.robot_constants import FINGER_JOINT_NAMES
from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler


euler_angles = torch.tensor([-np.pi / 4, 0.0, 0])  # (roll, pitch, yaw) in radians
quat = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2])

LIDAR_DOOR_SCENE_NUM_POINTS = 60000
LIDAR_ROBOT_SCENE_NUM_POINTS = 20000
LIDAR_NUM_POINTS = 10000
LIDAR_NUM_AZIMUTH = 512
LIDAR_NUM_POLAR = 128
LIDAR_NEAR_M = 0.1
LIDAR_FAR_M = 30.0
LIDAR_SUPPRESS_BINS = 2
LIDAR_JITTER_STD_M = 0.001
LIDAR_USE_COMPILE = True
LIDAR_ENV_ID = 0


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
        ("depth_to_pointcloud", "FrankaGripperSampler.sample")
        if depth_mean <= sampler_mean
        else ("FrankaGripperSampler.sample", "depth_to_pointcloud")
    )
    raw_speedup = max(depth_mean, sampler_mean) / max(min(depth_mean, sampler_mean), 1e-9)

    total_faster_name, total_slower_name = (
        ("depth_to_pointcloud", "FrankaGripperSampler.sample + world transform")
        if depth_mean <= sampler_total_mean
        else ("FrankaGripperSampler.sample + world transform", "depth_to_pointcloud")
    )
    total_speedup = max(depth_mean, sampler_total_mean) / max(min(depth_mean, sampler_total_mean), 1e-9)

    print(
        f"[INFO]: Pointcloud benchmark summary ({benchmark_state['timed_frames']} camera frames, "
        f"{num_envs} envs, {args_cli.benchmark_num_points} points)"
    )
    print(f"[INFO]: depth_to_pointcloud output shape: {benchmark_state['depth_shape']}")
    print(f"[INFO]: FrankaGripperSampler output shape: {benchmark_state['sampler_shape']}")
    print(
        "[INFO]: depth_to_pointcloud timing is projection + crop only; "
        "camera rendering cost is not included in this number."
    )
    print(f"[INFO]: depth_to_pointcloud: {_format_timing_stats(benchmark_state['depth_ms'])}")
    print(f"[INFO]: FrankaGripperSampler.sample: {_format_timing_stats(benchmark_state['sampler_ms'])}")
    print(
        "[INFO]: FrankaGripperSampler.sample + world transform: "
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


def _save_open3d_pointcloud(pointcloud: torch.Tensor, output_path: str):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for --save_lidar_points") from exc

    np_points = pointcloud.detach().cpu().numpy().astype(np.float64)
    finite_mask = np.isfinite(np_points).all(axis=-1)
    np_points = np_points[finite_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(np_points))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO]: Saved lidar pointcloud to {output_path} ({np_points.shape[0]} points)")


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
        update_latest_camera_pose=True,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        # data_types=["distance_to_image_plane"],
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
            "sampler": FrankaGripperSampler(
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

    enable_lidar_render = args_cli.save_lidar_points or args_cli.log_viser
    lidar_state = None
    if enable_lidar_render:
        door_base_body_idx = int(scene["door"].find_bodies("base")[0][0])
        lidar_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "lidar")
        os.makedirs(lidar_output_dir, exist_ok=True)

        viser_server = None
        viser_handle = None
        if args_cli.log_viser:
            if VISER_MODULE is None:
                raise RuntimeError(
                    "Failed to import viser for --log_viser. "
                    "This is usually caused by an incompatible `websockets` module being picked up after Isaac Sim "
                    "updates sys.path. Please ensure your active env has a recent `websockets` package and retry."
                ) from VISER_IMPORT_ERROR
            viser_server = VISER_MODULE.ViserServer()
            viser_handle = viser_server.scene.add_point_cloud(
                name="/rendered_lidar_points",
                points=np.zeros((0, 3), dtype=np.float16),
                colors=(0, 255, 0),
                point_size=0.01 / 3.0,
                precision="float16",
            )
            print("[INFO]: Viser logging enabled for rendered_lidar_points.")

        lidar_state = {
            "sampler": FrankaGripperSampler(
                door_asset_path,
                device=args_cli.device,
                num_points=LIDAR_DOOR_SCENE_NUM_POINTS,
            ),
            "robot_sampler": FrankaGripperSampler(
                glorbot_urdf_path,
                device=args_cli.device,
                num_points=LIDAR_ROBOT_SCENE_NUM_POINTS,
            ),
            "door_base_body_idx": door_base_body_idx,
            "lidar_body_idx": int(scene["robot"].find_bodies("lidar")[0][0]),
            "last_camera_frame": -1,
            "rendered_frames": 0,
            "saved_frames": 0,
            "saved_once": False,
            "output_dir": lidar_output_dir,
            "viser_server": viser_server,
            "viser_handle": viser_handle,
        }

        robot_sampler_joint_names = list(lidar_state["robot_sampler"].robot.actuated_joint_names)
        robot_joint_ids, robot_joint_names = scene["robot"].find_joints(robot_sampler_joint_names)
        lidar_state["robot_sampler_joint_ids"] = torch.tensor(robot_joint_ids, device=args_cli.device, dtype=torch.long)
        robot_joint_name_to_idx = {name: idx for idx, name in enumerate(robot_joint_names)}
        lidar_state["robot_sampler_joint_reorder"] = [robot_joint_name_to_idx[name] for name in robot_sampler_joint_names]
        lidar_state["robot_root_body_idx"] = 0
        print(
            f"[INFO]: Rendering lidar points (num_points={LIDAR_NUM_POINTS}, "
            f"azimuth={LIDAR_NUM_AZIMUTH}, polar={LIDAR_NUM_POLAR}, compile={LIDAR_USE_COMPILE})."
        )

    targets = scene["robot"].data.default_joint_pos.clone()

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
                        if not enable_lidar_render:
                            break
                        benchmark_state = None

        if lidar_state is not None:
            current_camera_frame = int(camera.frame[0].item())
            # if current_camera_frame != lidar_state["last_camera_frame"]:
            if True:
                lidar_state["last_camera_frame"] = current_camera_frame
                door_joint_pos = scene["door"].data.joint_pos
                sampled_local_pcd = lidar_state["sampler"].sample(door_joint_pos)

                door_base_pos_w = scene["door"].data.body_pos_w[:, lidar_state["door_base_body_idx"]]
                door_base_quat_w = scene["door"].data.body_quat_w[:, lidar_state["door_base_body_idx"]]
                quat = door_base_quat_w.unsqueeze(1).expand(-1, sampled_local_pcd.shape[1], -1)
                door_pcd_world = quat_apply(quat, sampled_local_pcd) + door_base_pos_w.unsqueeze(1)

                robot_joint_pos = scene["robot"].data.joint_pos[:, lidar_state["robot_sampler_joint_ids"]]
                robot_joint_pos = robot_joint_pos[:, lidar_state["robot_sampler_joint_reorder"]]
                robot_local_pcd = lidar_state["robot_sampler"].sample(robot_joint_pos)
                robot_root_pos_w = scene["robot"].data.body_pos_w[:, lidar_state["robot_root_body_idx"]]
                robot_root_quat_w = scene["robot"].data.body_quat_w[:, lidar_state["robot_root_body_idx"]]
                robot_quat = robot_root_quat_w.unsqueeze(1).expand(-1, robot_local_pcd.shape[1], -1)
                robot_pcd_world = quat_apply(robot_quat, robot_local_pcd) + robot_root_pos_w.unsqueeze(1)

                source_pcd_world = torch.cat([door_pcd_world, robot_pcd_world], dim=1)

                lidar_pos_w = scene["robot"].data.body_pos_w[:, lidar_state["lidar_body_idx"]]
                lidar_quat_w = scene["robot"].data.body_quat_w[:, lidar_state["lidar_body_idx"]]  # wxyz
                lidar_quat_xyzw = lidar_quat_w[:, [1, 2, 3, 0]]
                lidar_pose7 = torch.cat([lidar_pos_w, lidar_quat_xyzw], dim=-1)
                print("lidar_pose7: ", lidar_pose7)

                import time
                _sync_timing_device(torch.device(args_cli.device))
                start_time = time.perf_counter()
                lidar_pcd, lidar_logs = simulate_lidar_render_from_pose(
                    pcd=source_pcd_world,
                    lidar_pose=lidar_pose7,
                    num_points=LIDAR_NUM_POINTS,
                    num_azimuth=LIDAR_NUM_AZIMUTH,
                    num_polar=LIDAR_NUM_POLAR,
                    near_m=LIDAR_NEAR_M,
                    far_m=LIDAR_FAR_M,
                    suppress_bins=LIDAR_SUPPRESS_BINS,
                    jitter_std_m=LIDAR_JITTER_STD_M,
                    use_compile=LIDAR_USE_COMPILE,
                )
                _sync_timing_device(torch.device(args_cli.device))
                end_time = time.perf_counter()
                print(f"Lidar render time: {end_time - start_time} seconds")

                lidar_state["rendered_frames"] += 1
                env_id = max(0, min(LIDAR_ENV_ID, scene.num_envs - 1))
                valid_env_points = torch.isfinite(lidar_pcd[env_id]).all(dim=-1).sum().item()
                print(
                    f"[INFO]: Lidar frame={current_camera_frame} env={env_id} "
                    f"valid_points={int(valid_env_points)}/{LIDAR_NUM_POINTS} "
                    f"avg_valid_bins={lidar_logs['sim_lidar_render/avg_num_valid_points']:.1f}"
                )

                if lidar_state["viser_handle"] is not None:
                    np_points = lidar_pcd[env_id].detach().cpu().numpy().astype(np.float32)
                    finite_mask = np.isfinite(np_points).all(axis=-1)
                    lidar_state["viser_handle"].points = np_points[finite_mask].astype(np.float16, copy=False)

                if args_cli.save_lidar_points and not lidar_state["saved_once"]:
                    out_name = (
                        f"lidar_env{env_id:02d}_camframe{current_camera_frame:06d}.ply"
                    )
                    output_path = os.path.join(lidar_state["output_dir"], out_name)
                    _save_open3d_pointcloud(lidar_pcd[env_id], output_path)
                    lidar_state["saved_frames"] += 1
                    lidar_state["saved_once"] = True
                    if not args_cli.log_viser and benchmark_state is None:
                        print("[INFO]: Saved lidar pointcloud. Exiting.")
                        break

        # Extract camera data
        if args_cli.save and count % 100 == 0:
            targets[..., 0] += 0.1
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
                # print(depth_to_pointcloud(data).shape)
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
    print("[FINGER ORDER]", list(scene["robot"].find_joints(FINGER_JOINT_NAMES, preserve_order=True)[1]))
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
