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
parser.add_argument("--debug", action="store_true", default=False, help="Debug output.")
parser.add_argument("--force", action="store_true", default=False, help="Force reset the scene.")
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
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG, edit_door_articulation

from DoorOpening.motion.slider_controller import OmniJointController

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab.sensors import ContactSensorCfg
from DoorOpening.assets.glorbot.glorbot_cfg import FRANKA_JOINT_NAMES, BASE_JOINT_NAMES

from DoorOpening.utils.llm_utils import solve_ik, open_hand, close_hand, write_joint_angle_to_robot, record_joint_angles, get_robot_link_pose, write_joint_angle_to_door, step_sim
from DoorOpening.utils.llm_utils import get_board_frame_joint_angle, get_frame_hinge_joint_angle, get_board_pos, get_hinge_pos

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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics


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
    
    cfg = FRAME_MARKER_CFG.replace(prim_path="/World/GoalFrame")
    cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    goal_marker = VisualizationMarkers(cfg)

    goal_marker.visualize(
        translations=torch.tensor([[0.0, 0.0, 0.0]]),
        orientations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )

    # elbow_body_idx, _ = scene["robot"].find_bodies("panda_link4")
    # palm_body_idx, _ = scene["robot"].find_bodies("palm_center")
    # base_body_idx, _ = scene["robot"].find_bodies("tidybot2_base_link")
    # elbow_body_idx = elbow_body_idx[0]
    # palm_body_idx = palm_body_idx[0]
    # base_body_idx = base_body_idx[0]

    # def run_door_open_and_traverse(scene, sim, robot, door):
    #     key_joint_angles = []
    #     device = "cuda"

    #     assert robot.data.joint_pos.shape[0] == 1, "This script assumes env=1 (see write_joint_angle_to_door implementation)."
    #     assert door.data.joint_pos.shape[0] == 1, "This script assumes env=1 (see write_joint_angle_to_door implementation)."

    #     # -------------------------
    #     # Small utilities
    #     # -------------------------
    #     def to_t(xyz):
    #         return torch.tensor(xyz, device=device, dtype=torch.float32).unsqueeze(0)

    #     def goto_ik(palm_pos=None, elbow_pos=None, base_pos=None, iters=20, damping=0.15, record=False):
    #         """Iteratively run the provided solve_ik/write/step pattern."""
    #         for _ in range(iters):
    #             q = solve_ik(
    #                 robot,
    #                 elbow_pose=elbow_pos,
    #                 palm_pose=palm_pos,
    #                 base_pose=base_pos,
    #                 damping=damping,
    #             )
    #             write_joint_angle_to_robot(robot, q)
    #             step_sim(scene, sim)
    #         if record:
    #             record_joint_angles(robot, key_joint_angles)

    #     def set_door(board_angle, hinge_angle, settle_steps=2):
    #         """Hard-set door DOFs and step a bit."""
    #         write_joint_angle_to_door(
    #             door,
    #             target_board_joint_angle=torch.tensor([board_angle], device=device),
    #             target_hinge_joint_angle=torch.tensor([hinge_angle], device=device),
    #         )
    #         for _ in range(settle_steps):
    #             step_sim(scene, sim)

    #     # -------------------------
    #     # 0) Initialize / open hand
    #     # -------------------------
    #     robot = scene["robot"]
    #     door = scene["door"]
    #     open_hand(robot)
    #     step_sim(scene, sim)
    #     record_joint_angles(robot, key_joint_angles)

    #     # Door starts closed/locked
    #     set_door(board_angle=0.0, hinge_angle=0.0, settle_steps=2)

    #     # Query hinge/board geometry from sim
    #     hinge_pos = get_hinge_pos(door)   # [1,3]
    #     board_pos = get_board_pos(door)   # [1,3]

    #     # Robot is on +y side (same side as hinge). Approach from +y towards the door.
    #     # Pregrasp base: stay ~0.75m away in +y and slightly to +x (hinge is on right side).
    #     print("hinge_pos: ", hinge_pos.shape)
    #     pre_base_pos = hinge_pos.clone()
    #     pre_base_pos[:, 1] += 0.75
    #     pre_base_pos[:, 0] += 0.10
    #     pre_base_pos[:, 2] = 0.0

    #     # Arm approach offsets (towards robot is +y)
    #     approach_offset = to_t([0.00, 0.12, 0.00])     # palm a bit in front of hinge (robot side)
    #     lift_offset     = to_t([0.00, 0.00, 0.05])     # slight z lift to avoid scraping

    #     # A mild elbow bias "up and toward robot" to avoid door plane
    #     def elbow_bias_from_base(base_xyz):
    #         return base_xyz + to_t([0.00, -0.05, 0.45])

    #     # -------------------------
    #     # 1) Move to a good pregrasp pose
    #     # -------------------------
    #     # Move base first
    #     goto_ik(base_pos=pre_base_pos, iters=40, damping=0.25, record=True)

    #     # Then bring palm near the hinge
    #     hinge_pos = get_hinge_pos(door)
    #     palm_pregrasp = hinge_pos + approach_offset + lift_offset
    #     base_xyz, _ = get_robot_link_pose(robot, "base")
    #     elbow_pregrasp = elbow_bias_from_base(base_xyz)
    #     print("elbow_pregrasp: ", elbow_pregrasp.shape)

    #     goto_ik(palm_pos=palm_pregrasp, elbow_pos=elbow_pregrasp, iters=40, damping=0.15, record=True)

    #     # Close hand to "grasp" the round hinge/knob
    #     close_hand(robot)
    #     step_sim(scene, sim)
    #     record_joint_angles(robot, key_joint_angles)

    #     # -------------------------
    #     # 2) Rotate door hinge until latch releases (unknown threshold)
    #     #    Strategy:
    #     #      - try CCW first (+), then CW (-)
    #     #      - after each hinge increment, "probe" by commanding a tiny board open angle
    #     #      - if board angle actually increases (measured), latch is released.
    #     # -------------------------
    #     latch_released = False
    #     hinge_unlocked = 0.0

    #     probe_board = 0.06      # small attempt to open
    #     hinge_step  = 0.12
    #     max_hinge   = 1.40

    #     for direction in [+1.0, -1.0]:
    #         # reset door
    #         set_door(board_angle=0.0, hinge_angle=0.0, settle_steps=2)
    #         goto_ik(palm_pos=get_hinge_pos(door) + approach_offset + lift_offset,
    #                 elbow_pos=elbow_bias_from_base(get_robot_link_pose(robot, "base")[0]),
    #                 iters=10, damping=0.15)

    #         for k in range(int(max_hinge / hinge_step)):
    #             hinge_cmd = direction * (k + 1) * hinge_step

    #             # "Turn" the hinge
    #             set_door(board_angle=0.0, hinge_angle=hinge_cmd, settle_steps=2)

    #             # Keep palm on hinge while we rotate it (re-acquire hinge pose since it can move slightly)
    #             hinge_pos = get_hinge_pos(door)
    #             goto_ik(
    #                 palm_pos=hinge_pos + approach_offset + lift_offset,
    #                 elbow_pos=elbow_bias_from_base(get_robot_link_pose(robot, "base")[0]),
    #                 iters=15,
    #                 damping=0.15,
    #             )

    #             # Probe the latch by trying to open board a tiny amount
    #             set_door(board_angle=probe_board, hinge_angle=hinge_cmd, settle_steps=2)
    #             cur_board = float(get_board_frame_joint_angle(door).item())

    #             if cur_board > 0.02:
    #                 latch_released = True
    #                 hinge_unlocked = hinge_cmd
    #                 break
    #             else:
    #                 # return board closed and continue turning
    #                 set_door(board_angle=0.0, hinge_angle=hinge_cmd, settle_steps=1)

    #         if latch_released:
    #             break

    #     # Record after latch release attempt
    #     record_joint_angles(robot, key_joint_angles)

    #     # If latch never released, we still proceed by forcing it (trajectory demo).
    #     if not latch_released:
    #         hinge_unlocked = 0.9  # arbitrary "unlocked" angle
    #         set_door(board_angle=0.0, hinge_angle=hinge_unlocked, settle_steps=2)

    #     # -------------------------
    #     # 3) Pull the door open by 60-90 degrees (≈ 1.05 to 1.57 rad)
    #     #    We open in increments; keep palm tracking hinge position (as if pulling).
    #     # -------------------------
    #     target_open = 1.25  # ~72 degrees
    #     n_open = 10

    #     for i in range(n_open + 1):
    #         board_cmd = float(target_open * (i / n_open))
    #         set_door(board_angle=board_cmd, hinge_angle=hinge_unlocked, settle_steps=2)

    #         hinge_pos = get_hinge_pos(door)
    #         palm_hold = hinge_pos + approach_offset + lift_offset

    #         # Slightly "pull" by keeping palm on robot side of hinge
    #         base_xyz, _ = get_robot_link_pose(robot, "base")
    #         elbow_hold = elbow_bias_from_base(base_xyz)

    #         goto_ik(palm_pos=palm_hold, elbow_pos=elbow_hold, iters=18, damping=0.12)

    #     record_joint_angles(robot, key_joint_angles)

    #     # -------------------------
    #     # 4) Hold the door with the arm, move base forward
    #     #    We keep door commanded open and keep palm on hinge while base advances.
    #     # -------------------------
    #     # Move base through the doorway along -y (toward the other side).
    #     # Use a few waypoints so the arm can keep up.
    #     base_xyz, _ = get_robot_link_pose(robot, "base")
    #     base_start = base_xyz.clone()

    #     base_waypoints = [
    #         base_start + to_t([0.00, -0.20, 0.00]),
    #         base_start + to_t([0.00, -0.40, 0.00]),
    #         base_start + to_t([0.00, -0.60, 0.00]),
    #     ]

    #     for bp in base_waypoints:
    #         # Keep door open (spring would otherwise close it)
    #         set_door(board_angle=target_open, hinge_angle=hinge_unlocked, settle_steps=1)

    #         hinge_pos = get_hinge_pos(door)
    #         palm_hold = hinge_pos + approach_offset + lift_offset

    #         goto_ik(
    #             palm_pos=palm_hold,
    #             elbow_pos=elbow_bias_from_base(bp),
    #             base_pos=bp,
    #             iters=35,
    #             damping=0.18,
    #         )

    #         record_joint_angles(robot, key_joint_angles)

    #     # -------------------------
    #     # 5) When the base can stop the door from closing, release hand
    #     #    (We approximate: after moving forward enough, base is now "in the way".)
    #     # -------------------------
    #     open_hand(robot)
    #     step_sim(scene, sim)
    #     record_joint_angles(robot, key_joint_angles)

    #     # -------------------------
    #     # 6) Retract the arm WITHOUT colliding with the door
    #     #    Pull the hand up and to robot-left (-x) to clear the open door.
    #     # -------------------------
    #     base_xyz, _ = get_robot_link_pose(robot, "base")

    #     palm_retract_1 = base_xyz + to_t([-0.25, 0.05, 0.55])
    #     elbow_retract_1 = base_xyz + to_t([-0.10, 0.00, 0.70])
    #     goto_ik(palm_pos=palm_retract_1, elbow_pos=elbow_retract_1, iters=40, damping=0.18, record=True)

    #     palm_retract_2 = base_xyz + to_t([-0.20, 0.15, 0.65])
    #     elbow_retract_2 = base_xyz + to_t([-0.05, 0.10, 0.80])
    #     goto_ik(palm_pos=palm_retract_2, elbow_pos=elbow_retract_2, iters=35, damping=0.18, record=True)

    #     # -------------------------
    #     # 7) Move forward and traverse through the frame
    #     # -------------------------
    #     base_xyz, _ = get_robot_link_pose(robot, "base")
    #     traverse_waypoints = [
    #         base_xyz + to_t([0.00, -0.35, 0.00]),
    #         base_xyz + to_t([0.00, -0.70, 0.00]),
    #     ]
    #     for bp in traverse_waypoints:
    #         goto_ik(base_pos=bp, iters=45, damping=0.25, record=True)

    #     return key_joint_angles

    robot = scene["robot"]
    door = scene["door"]
    # key_joint_angles = run_door_open_and_traverse(scene, sim, robot, door)
    # print("key_joint_angles: ", key_joint_angles)
    # run_gemini_response(scene, sim, robot, door)
    key_joint_angles = run_gpt_response(scene, sim, robot, door)
    collocate_and_playback(scene, sim, robot, door, key_joint_angles)

def run_gpt_response(scene, sim, robot, door):
    key_joint_angles = []          # buffer for key frames

    # --- 1. Pregrasp ---
    open_hand(robot)
    step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    board_pos = get_board_pos(door)
    hinge_pos = get_hinge_pos(door)

    handle_offset   = torch.tensor([-0.45, 0.0, 0.0])  # 45 cm from hinge along −x
    approach_offset = torch.tensor([0.0, -0.10, 0.0])  # 10 cm in front of door
    pregrasp_pos    = board_pos + handle_offset + approach_offset

    base_pos, base_quat = get_robot_link_pose(robot, "base")
    target_base_pose = torch.cat((base_pos + torch.tensor([0.0, 0.3, 0.0]), base_quat), dim=-1)

    for _ in range(10):
        q = solve_ik(robot, palm_pose=pregrasp_pos, base_pose=target_base_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    # --- 2. Grasp the handle ---
    grasp_pos = board_pos + handle_offset
    for _ in range(10):
        q = solve_ik(robot, palm_pose=grasp_pos, base_pose=target_base_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)
    close_hand(robot)
    step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    # --- 3. Rotate hinge to unlatch ---
    hinge_angle = get_frame_hinge_joint_angle(door).squeeze()
    target_hinge_angle = hinge_angle + 0.6
    for a in torch.linspace(hinge_angle, target_hinge_angle, 5):
        write_joint_angle_to_door(door, get_board_frame_joint_angle(door).squeeze().item(), a.item())
        step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    # --- 4. Pull door open by ~60° ---
    target_open_angle = 1.0
    for a in torch.linspace(0., target_open_angle, 6):
        write_joint_angle_to_door(door, a.item(), get_frame_hinge_joint_angle(door).squeeze().item())
        step_sim(scene, sim)
        palm_pos, _ = get_robot_link_pose(robot, "palm")
        new_palm = palm_pos + torch.tensor([0.0, -0.05, 0.0])
        q = solve_ik(robot, palm_pose=new_palm, base_pose=target_base_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)
        record_joint_angles(robot, door, key_joint_angles)

    # --- 5. Hold, move base into opening ---
    door_hold_base = torch.cat((base_pos + torch.tensor([0.0, 0.4, 0.0]), base_quat), dim=-1)
    for _ in range(15):
        q = solve_ik(robot, palm_pose=new_palm, base_pose=door_hold_base)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    # --- 6. Release, then retract arm ---
    open_hand(robot)
    step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    retract_pose = new_palm + torch.tensor([0.0, -0.2, 0.2])
    for _ in range(10):
        q = solve_ik(robot, palm_pose=retract_pose, base_pose=door_hold_base)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    # --- 7. Move base completely through door ---
    print("door_hold_base: ", door_hold_base.shape)
    through_pose = torch.cat((door_hold_base[0, :3] + torch.tensor([0.0, 0.5, 0.0]), door_hold_base[0, 3:]), dim=-1)
    for _ in range(20):
        q = solve_ik(robot, palm_pose=retract_pose, base_pose=through_pose)
        write_joint_angle_to_robot(robot, q)
        step_sim(scene, sim)
    record_joint_angles(robot, door, key_joint_angles)

    print("Collected", len(key_joint_angles), "key waypoints")
    return key_joint_angles

def collocate_and_playback(scene, sim, robot, door, key_joint_angles):
    from scipy.interpolate import CubicSpline
    qs = torch.stack([kp[0] for kp in key_joint_angles]).detach().cpu().numpy()

    # automatic timing
    dq = np.linalg.norm(qs[1:] - qs[:-1], axis=1)
    dt = np.maximum(dq / 1.0, 0.1)
    t_key = np.concatenate([[0], np.cumsum(dt)])
    t_key /= t_key[-1]

    cs = CubicSpline(t_key, qs, axis=0, bc_type="clamped")

    t = np.linspace(0, 1, 1000)
    traj = torch.tensor(cs(t))
    qd = torch.tensor(cs(t, 1))
    qdd = torch.tensor(cs(t, 2))

    key_indices = np.searchsorted(t, t_key)
    key_indices = np.clip(key_indices, 0, len(t) - 1)

    door_traj = traj[:, -2:]
    traj = traj[:, :-2]

    print("starting playback...")

    for door_point, robot_point in zip(door_traj, traj):
        door.write_joint_position_to_sim(door_point)
        robot.write_joint_position_to_sim(robot_point)
        step_sim(scene, sim)

def run_gemini_response(scene, sim, robot, door):
    # --- Configuration & Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key_joint_angles = []

    # Helper function to run IK loop, step sim, and record
    def run_ik_and_record(robot, scene, sim, target_palm_pos=None, target_palm_rot=None, target_base_pos=None, target_base_rot=None, steps=10):
        """
        Solves IK iteratively and steps the simulation.
        target_palm_rot: Expected as (roll, pitch, yaw) or None
        target_base_rot: Expected as (roll, pitch, yaw) or None
        """
        palm_pose_arg = None
        if target_palm_pos is not None:
            # Combine pos and rot for the argument tuple if rot is provided
            if target_palm_rot is not None:
                palm_pose_arg = (target_palm_pos, target_palm_rot)
            else:
                palm_pose_arg = target_palm_pos
                
        base_pose_arg = None
        if target_base_pos is not None:
            if target_base_rot is not None:
                base_pose_arg = (target_base_pos, target_base_rot)
            else:
                base_pose_arg = target_base_pos

        for _ in range(steps):
            q = solve_ik(
                robot, 
                elbow_pose=None, 
                palm_pose=palm_pose_arg, 
                base_pose=base_pose_arg
            )
            write_joint_angle_to_robot(robot, q)
            step_sim(scene, sim)
            
        # Record the final stable pose of this waypoint
        record_joint_angles(robot, key_joint_angles)

    # --- Initialization ---
    # Ensure the door is closed initially
    write_joint_angle_to_door(door, torch.tensor([0.0], device=device), torch.tensor([0.0], device=device))
    step_sim(scene, sim)
    open_hand(robot)
    step_sim(scene, sim)

    # Get initial geometry
    # We assume robot is at +X side facing -X. Door is at (0,0,0).
    # We need to find the handle.
    initial_hinge_pos = get_hinge_pos(door) # This is the handle position
    initial_board_pos = get_board_pos(door)

    # --- Phase 1: Move to Pre-grasp Pose ---
    # Target: ~15cm "behind" the handle (in +x direction since robot is at +x) to align the gripper
    # Orientation: Palm facing -X (towards door). 
    # Assuming standard Franka hand frame, we might need specific RPY. 
    # Let's assume (0, 0, 0) orients the hand along default axes and IK solves the rest, 
    # but specifically we want the palm pointing -x.
    pre_grasp_offset = torch.tensor([0.15, 0.0, 0.0], device=device)
    pre_grasp_pos = initial_hinge_pos + pre_grasp_offset

    # Current Base Pose (starting position)
    # Assuming robot starts roughly at (0.8, 0, 0) facing -x
    current_base_pos = torch.tensor([0.8, 0.0, 0.0], device=device)
    current_base_rot = torch.tensor([0.0, 0.0, 3.14159], device=device) # Face -X

    # Move arm to pre-grasp
    run_ik_and_record(
        robot, scene, sim, 
        target_palm_pos=pre_grasp_pos,
        target_palm_rot=torch.tensor([0.0, 1.57, 0.0], device=device), # Pitch 90 deg to point forward? Adjusting based on standard conventions
        target_base_pos=current_base_pos,
        target_base_rot=current_base_rot
    )

    # --- Phase 2: Grasping ---
    # Move palm to the handle position
    run_ik_and_record(
        robot, scene, sim,
        target_palm_pos=initial_hinge_pos,
        target_palm_rot=torch.tensor([0.0, 1.57, 0.0], device=device),
        target_base_pos=current_base_pos,
        target_base_rot=current_base_rot
    )

    # Close hand
    close_hand(robot)
    step_sim(scene, sim)
    record_joint_angles(robot, key_joint_angles) # Record grasped state

    # --- Phase 3: Rotate Hinge (Unlock) ---
    # The door requires the knob to be rotated.
    # We physically rotate the robot hand, but we also "cheat" by forcing the door state
    # to ensure the IK target matches the door's mechanics perfectly.
    target_knob_angle = torch.tensor([-1.57], device=device) # Rotate -90 degrees
    write_joint_angle_to_door(door, torch.tensor([0.0], device=device), target_knob_angle)
    step_sim(scene, sim)

    # Now solve IK for the robot to match this new rotation
    # The handle position shouldn't change much (it spins in place), but orientation does.
    # We rotate the palm roll by -1.57
    run_ik_and_record(
        robot, scene, sim,
        target_palm_pos=initial_hinge_pos,
        target_palm_rot=torch.tensor([-1.57, 1.57, 0.0], device=device), # Add roll
        target_base_pos=current_base_pos,
        target_base_rot=current_base_rot
    )

    # --- Phase 4: Pull Door Open ---
    # We need to pull the door from 0 to ~1.4 radians (~80 degrees).
    # As we pull, the robot base must move backwards (+X) and sideways (Y) to avoid the swing.
    # We determine the "Sideways" direction based on where the handle is relative to the center.
    # If handle.y > 0, door swings "left" (relative to robot facing door), so we move right (-y).
    sideways_direction = -1.0 if initial_hinge_pos[0, 1] > 0 else 1.0

    num_open_steps = 15
    max_door_angle = 1.4

    for i in range(num_open_steps):
        alpha = (i + 1) * (max_door_angle / num_open_steps)
        
        # 1. Force simulation to this door angle so we can get the exact handle position
        write_joint_angle_to_door(door, torch.tensor([alpha], device=device), target_knob_angle)
        step_sim(scene, sim)
        
        # 2. Get the new handle position (IK target)
        current_handle_pos = get_hinge_pos(door)
        
        # 3. Compute desired base position
        # Strategy: Maintain relative distance to handle, but back up and step aside.
        # Back up: Increase X. Step aside: Move Y.
        # Heuristic: Base X = Handle X + 0.6m. Base Y = Handle Y + (0.2m * sideways_dir)
        target_base_x = current_handle_pos[0, 0] + 0.65
        target_base_y = current_handle_pos[0, 1] + (0.2 * sideways_direction)
        
        # Construct base pose
        step_base_pos = torch.tensor([target_base_x, target_base_y, 0.0], device=device)
        
        # 4. Solve IK to hold the handle while moving base
        # We maintain the "unlatched" rotation of the hand
        run_ik_and_record(
            robot, scene, sim,
            target_palm_pos=current_handle_pos,
            target_palm_rot=torch.tensor([-1.57, 1.57, 0.0], device=device), 
            target_base_pos=step_base_pos,
            target_base_rot=current_base_rot
        )
        
        # Update current base pos tracker
        current_base_pos = step_base_pos

    # --- Phase 5: Block Door with Base ---
    # The door is now open ~80 degrees. If we release, it closes.
    # We need to move the base *into* the path of the closing door (the "gap")
    # while still holding the handle, then release.
    # The door board is roughly perpendicular to its current radius.
    # We move the base towards the door hinge frame to wedge it.

    # Target blocking position: Roughly where we are, but maybe slightly forward/lateral 
    # to ensure the chassis hits the door if it swings back.
    # We move 20cm towards the door frame (decrease X) and slightly into the door plane.
    block_base_pos = current_base_pos.clone()
    block_base_pos[0] -= 0.2 # Move forward
    block_base_pos[1] -= (0.1 * sideways_direction) # Move towards the gap center

    run_ik_and_record(
        robot, scene, sim,
        target_palm_pos=get_hinge_pos(door), # Keep holding handle
        target_palm_rot=torch.tensor([-1.57, 1.57, 0.0], device=device),
        target_base_pos=block_base_pos,
        target_base_rot=current_base_rot
    )

    # --- Phase 6: Release and Retract ---
    open_hand(robot)
    step_sim(scene, sim)
    record_joint_angles(robot, key_joint_angles)

    # Retract arm to a safe travel pose (tucked in)
    # Defined relative to base
    tucked_palm_offset = torch.tensor([0.3, 0.0, 0.4], device=device) # Front-center, slightly up
    tucked_palm_global = block_base_pos + tucked_palm_offset

    run_ik_and_record(
        robot, scene, sim,
        target_palm_pos=tucked_palm_global,
        target_palm_rot=torch.tensor([0.0, 0.0, 0.0], device=device), # Neutral orientation
        target_base_pos=block_base_pos,
        target_base_rot=current_base_rot
    )

    # --- Phase 7: Traverse Through ---
    # Move the base forward through the door (Negative X direction)
    # We go from current X to X = -1.0
    traverse_target_pos = torch.tensor([-1.0, 0.0, 0.0], device=device)

    # We interpolate this movement so the robot doesn't teleport in IK calculation
    # Using just 3 waypoints for the traversal
    for t in range(3):
        interp_pos = block_base_pos + (traverse_target_pos - block_base_pos) * ((t + 1) / 3.0)
        
        # Keep arm tucked relative to base during movement
        current_tuck_pos = interp_pos + tucked_palm_offset
        
        run_ik_and_record(
            robot, scene, sim,
            target_palm_pos=current_tuck_pos,
            target_base_pos=interp_pos,
            target_base_rot=current_base_rot # Keep facing -X? Or rotate to face movement? Omni base can strafe.
        )

    print("Trajectory generation complete.")


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
