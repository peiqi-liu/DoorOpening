# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyframe/teleop harness for glorbot.

Slider panel, IK-to-pose button, record key poses, spline them into a trajectory, play it back
and save it. The panel's hand section is a single gripper-width slider + Open/Close buttons,
because the Franka hand is one actuator.

.. code-block:: bash

    # interactive (slider panel)
    ./isaaclab.sh -p scripts/hard_code_traj.py --enable_cameras

    # headless-ish smoke test: no panel interaction needed, the gripper opens/closes on a loop
    ./isaaclab.sh -p scripts/hard_code_traj.py --enable_cameras --gripper_cycle
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyframe/teleop harness for glorbot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--debug", action="store_true", default=False, help="Debug output.")
parser.add_argument("--force", action="store_true", default=False, help="Force reset the scene.")
parser.add_argument(
    "--gripper_cycle",
    action="store_true",
    default=False,
    help="Continuously open/close the gripper to verify the finger DOFs actuate.",
)
parser.add_argument(
    "--gripper_cycle_steps",
    type=int,
    default=120,
    help="Sim steps spent per open/close half-cycle when --gripper_cycle is set.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG, disable_collision_scope_instancing
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG, edit_door_articulation

from DoorOpening.constants.robot_constants import (
    DEFAULT_JOINT_POS,
    FINGER_JOINT_NAMES,
    FULL_JOINT_NAMES,
    GRIPPER_CLOSED_WIDTH,
    GRIPPER_OPEN_WIDTH,
)
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT

from DoorOpening.motion.slider_controller import OmniJointController

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab.sensors import CameraCfg, ContactSensorCfg

torch.set_printoptions(precision=3, sci_mode=False)

CAMERA_EULER_ANGLES = torch.tensor([-np.pi / 4, 0.0, 0.0])
CAMERA_QUAT = quat_from_euler_xyz(*CAMERA_EULER_ANGLES)

# Bodies printed at startup so the gripper's placement can be eyeballed.
REPORTED_BODY_NAMES = [
    "tidybot2_base_link",
    "panda_link6",
    "panda_hand",
    "palm_center",
    "left_fingertip",
    "right_fingertip",
]


@configclass
class GripperSceneCfg(InteractiveSceneCfg):
    """Scene with the gripper robot, a door, an onboard camera and door contact sensors."""

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
            pos=ROBOT_INITIAL_POS,
            rot=ROBOT_INITIAL_ROT,
        ),
    )

    door: ArticulationCfg = DOOR_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Door",
    )

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x5_camera_link/cam",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8.0, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=CAMERA_QUAT, convention="world"),
    )

    contact_forces_door1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Door/link_1",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot"],
    )

    contact_forces_door2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Door/link_2",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot"],
    )

    # Finger-side contact: tells you whether the pads are actually loaded when a grasp closes on
    # the handle, rather than only what the door feels in total.
    contact_forces_left_finger = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/link_1", "{ENV_REGEX_NS}/Door/link_2"],
    )

    contact_forces_right_finger = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/link_1", "{ENV_REGEX_NS}/Door/link_2"],
    )


class IsaacCameraViewer:
    """Render RGB camera frames inside an Isaac Kit UI window."""

    def __init__(self, title: str, width: int = 640, height: int = 480):
        import omni.ui as ui

        self._ui = ui
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            title,
            width=width + 24,
            height=height + 48,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )

        blank_frame = np.zeros((height, width, 4), dtype=np.uint8)
        blank_frame[..., 3] = 255

        with self._window.frame:
            with self._ui.VStack(spacing=4):
                self._ui.Label("RGB stream", height=20)
                with self._ui.Frame(width=width, height=height):
                    self._ui.ImageWithProvider(self._provider)

        self.update_image(blank_frame)

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected image shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4), got {image.shape}")

        if image.shape[2] == 3:
            alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=-1)

        return np.ascontiguousarray(image)

    def update_image(self, image: np.ndarray):
        rgba = self._to_rgba(np.asarray(image))
        height, width = rgba.shape[:2]
        self._provider.set_bytes_data(rgba.reshape(-1).data, [width, height])

    def update_from_camera(self, camera, camera_index: int = 0):
        rgb_frame = camera.data.output.get("rgb")
        if rgb_frame is None or not self._window.visible:
            return

        frame = rgb_frame[camera_index].detach().cpu().numpy()
        self.update_image(frame)

    def close(self):
        if self._window is not None:
            self._window.visible = False


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    camera = scene["camera"]
    camera_viewer = IsaacCameraViewer("Glorbot Gripper RGB Camera")

    # reset the scene entities
    # root state
    # we offset the root state by the origin since the states are written in simulation world frame
    root_state = scene["robot"].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
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
    door_pos = torch.zeros_like(scene["door"].data.soft_joint_pos_limits[..., 0])
    scene["door"].write_joint_position_to_sim(door_pos)

    scene.reset()
    print("[INFO]: Resetting robot state...")

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

    body_idx, body_name = scene["robot"].find_bodies(REPORTED_BODY_NAMES)
    print("body idx: ", body_idx)
    print("body name: ", body_name)
    print("robot body pos: ", scene["robot"].data.body_pos_w[:, body_idx])
    print("robot body quat: ", scene["robot"].data.body_quat_w[:, body_idx])

    # Both DOFs, only for reading state back: the gripper is commanded through the single driven
    # joint (panda_finger_joint1); panda_finger_joint2 follows it via the URDF mimic coupling.
    gripper_joint_ids, gripper_joint_names = scene["robot"].find_joints(
        FINGER_JOINT_NAMES, preserve_order=True
    )
    print("gripper joints (driven, follower): ", gripper_joint_names, gripper_joint_ids)

    count = 0
    try:
        while simulation_app.is_running():
            # --gripper_cycle overrides the panel's gripper width with a square wave so the fingers
            # can be verified without touching the UI.
            if args_cli.gripper_cycle:
                half_cycle = max(1, args_cli.gripper_cycle_steps)
                closing = (count // half_cycle) % 2 == 1
                target_width = GRIPPER_CLOSED_WIDTH if closing else GRIPPER_OPEN_WIDTH
                # Only on transitions: set_gripper_width also clears any recorded trajectory.
                if abs(controller.gripper_width - target_width) > 1e-6:
                    controller.set_gripper_width(target_width)
                    print(f"[GRIPPER] step {count}: commanding width {target_width:.3f}")

            slider_pos = controller.q_slider.clone()
            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_pos[..., :] = slider_pos

            door_pos = controller.door_q_slider.clone()
            door_joint_pos = scene["door"].data.default_joint_pos.clone()
            door_joint_pos[..., :] = door_pos

            edit_door_articulation(scene["door"], door_closed_range=0.02, hinge_range=0.4)
            if not args_cli.force:
                scene["robot"].set_joint_position_target(joint_pos)
                scene["door"].set_joint_position_target(door_joint_pos)
            else:
                scene["robot"].write_joint_position_to_sim(joint_pos)
                scene["door"].write_joint_position_to_sim(door_joint_pos)

            if count % 10 == 0 and args_cli.debug:
                # Follower should track the driven joint; a growing gap between the two columns
                # means the mimic coupling is not in effect (see glorbot_gripper_cfg.py).
                measured = scene["robot"].data.joint_pos[:, gripper_joint_ids]
                print("gripper [driven, follower] target / measured: ", joint_pos[:, gripper_joint_ids], measured)
                # net_forces_w is total contact load on each door link; the finger sensors below are
                # filtered, so they report handle/board contact only.
                print("Received contact force of link 1: ", scene["contact_forces_door1"].data.net_forces_w)
                print("Received contact force of link 2: ", scene["contact_forces_door2"].data.net_forces_w)
                print("left finger vs door: ", scene["contact_forces_left_finger"].data.force_matrix_w)
                print("right finger vs door: ", scene["contact_forces_right_finger"].data.force_matrix_w)
                print("-------------------------------")

            # -- playback of a recorded trajectory
            if controller.playback:
                q = controller.traj[controller.play_idx]
                joint_pos = scene["robot"].data.default_joint_pos.clone()
                joint_pos[..., :] = q

                qd = controller.qd[controller.play_idx]
                joint_vel = scene["robot"].data.default_joint_vel.clone()
                joint_vel[..., :] = qd
                scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

                door_q = controller.door_traj[controller.play_idx]
                door_joint_pos = scene["door"].data.default_joint_pos.clone()
                door_joint_pos[..., :] = door_q
                scene["door"].write_joint_position_to_sim(door_joint_pos)

                # Write these poses data to the controller so it can be saved to pickle later
                controller.robot_body_pos_traj.append(scene["robot"].data.body_pos_w.squeeze().cpu().clone())
                controller.robot_body_quat_traj.append(scene["robot"].data.body_quat_w.squeeze().cpu().clone())
                controller.door_pos_traj.append(scene["door"].data.body_pos_w.squeeze().cpu().clone())

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

            camera_viewer.update_from_camera(camera)
    finally:
        camera_viewer.close()


def main():
    """Main function."""
    if not getattr(args_cli, "enable_cameras", False):
        raise ValueError("--enable_cameras is required to stream the robot RGB camera in the Isaac UI.")

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.0, -2.5, 3.2], target=[0.0, 0.0, 0.7])
    # Design scene
    scene_cfg = GripperSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # De-instance the robot's `collisions` scopes so the collider debug viz (green overlay)
    # renders them; see disable_collision_scope_instancing's docstring.
    disable_collision_scope_instancing()  # default expr "/World/envs/env_.*/Robot"
    disable_collision_scope_instancing("/World/envs/env_.*/Door")
    # Play the simulator
    sim.reset()
    material_properties = scene["robot"].root_physx_view.get_material_properties()
    print("material properties: ", material_properties.shape)
    material_properties[..., 0] = 2.0
    material_properties[..., 1] = 1.5
    env_ids = torch.arange(scene.num_envs, device="cpu")
    scene["robot"].root_physx_view.set_material_properties(material_properties, env_ids)
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
