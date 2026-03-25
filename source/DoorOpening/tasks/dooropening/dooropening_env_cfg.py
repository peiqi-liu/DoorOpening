# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from DoorOpening.assets.door.door_cfg import DOOR_CONFIG, ALL_DOOR_CONFIGS
from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT
from DoorOpening.constants.door_constants import DOOR_BODY_NAMES, DOOR_JOINT_NAMES
from DoorOpening.constants.robot_constants import (
    CAMERA_JOINT_DEFAULT_VALUES,
    CLOSE_FINGER_JOINT_VALUES,
    OPEN_FINGER_JOINT_VALUES,
    ROBOT_KEY_BODY_NAMES,
    ROBOT_RESET_KEY_BODY_NAMES,
    ROBOT_PALM_LINK_NAME,
    ROBOT_BASE_BODY_LINK_NAME,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.common import ViewerCfg
import torch
import numpy as np
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.math import quat_from_euler_xyz

import isaaclab.sim as sim_utils

euler_angles = torch.tensor([-np.pi / 4, 0.0, 0])  # (roll, pitch, yaw) in radians
POINTCLOUD_CAMERA_QUAT = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2])
POINTCLOUD_CAMERA_QUAT = tuple(POINTCLOUD_CAMERA_QUAT.tolist())

@configclass
class EventCfg:
    """Configuration for reset-time physics randomization."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    door_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    door_latch_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_1"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    door_hinge_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_2"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    door_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_(1|2)"),
            "friction_distribution_params": (0.0, 0.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

@configclass
class DooropeningEnvCfg(DirectRLEnvCfg):
    sim_dt = 1/60.
    decimation = 1
    episode_length_s = 10.
    num_sim_steps_to_render=2
    # - spaces definition
    state_space = 0
    num_states = 0
    asymmetric_obs = True

    viewer: ViewerCfg = ViewerCfg(eye=(1.5, -2.0, 1.0), lookat=(0.4, 0.0, 0.7), origin_type="env")

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=num_sim_steps_to_render,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15
        ),
    )

    # Useful constants

    base_link_name = "base_link"

    base_joints = [
        'base_rotation_joint',
        'base_x_joint',
        'base_y_joint',
    ]

    arm_joints = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
    ]

    # finger_joints = [
    #     'finger_joint_0',
    #     'finger_joint_1',
    #     'finger_joint_2',
    #     'finger_joint_3',
    #     'finger_joint_4',
    #     'finger_joint_5',
    #     'finger_joint_6',
    #     'finger_joint_7',
    #     'finger_joint_8',
    #     'finger_joint_9',
    #     'finger_joint_10',
    #     'finger_joint_11',
    #     'finger_joint_12',
    #     'finger_joint_13',
    #     'finger_joint_14',
    #     'finger_joint_15',
    # ]

    finger_joints = [
        'finger_joint_1',
        'finger_joint_2',
        'finger_joint_3',
        'finger_joint_5',
        'finger_joint_6',
        'finger_joint_7',
        'finger_joint_9',
        'finger_joint_10',
        'finger_joint_11',
    ]

    abduction_joints = [
        # actual abduction joints
        'finger_joint_0',
        'finger_joint_12',
        'finger_joint_4',
        'finger_joint_8',
        # additional joints we want to fix at default position
        'finger_joint_3',
        'finger_joint_7',
        'finger_joint_11',
        'finger_joint_13',
        'finger_joint_14',
        'finger_joint_15',
    ]

    contact_forces_door1 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_1",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot", "/World/envs/env_.*/Door/link_2"],
    )

    contact_forces_door2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_2",
        update_period=0.02,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/palm_center", 
            "/World/envs/env_.*/Robot/palm_lower", 
            "/World/envs/env_.*/Robot/mcp_joint_1", 
            "/World/envs/env_.*/Robot/pip_1", 
            "/World/envs/env_.*/Robot/dip_1", 
            "/World/envs/env_.*/Robot/realtip_1",
            "/World/envs/env_.*/Robot/fingertip_1",
            "/World/envs/env_.*/Robot/mcp_joint_2", 
            "/World/envs/env_.*/Robot/pip_2", 
            "/World/envs/env_.*/Robot/dip_2", 
            "/World/envs/env_.*/Robot/realtip_2", 
            "/World/envs/env_.*/Robot/fingertip_2",
            "/World/envs/env_.*/Robot/mcp_joint_3", 
            "/World/envs/env_.*/Robot/pip_3", 
            "/World/envs/env_.*/Robot/dip_3", 
            "/World/envs/env_.*/Robot/realtip_3", 
            "/World/envs/env_.*/Robot/fingertip_3",
        ],
    )

    contact_forces_robot_palm_center = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/palm_center",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Door/link_2"],
    )

    # contact_sensor_names = ["contact_forces_door1", "contact_forces_door2", "contact_forces_robot_palm_center"]
    contact_sensor_names = ["contact_forces_door2"]

    enable_pointcloud_camera = False
    pointcloud_camera_height = 480
    pointcloud_camera_width = 640
    pointcloud_camera_update_period = 0.1
    pointcloud_camera_data_types = ["distance_to_image_plane"]
    pointcloud_camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/x5_camera_link/cam",
        update_period=pointcloud_camera_update_period,
        update_latest_camera_pose=True,
        height=pointcloud_camera_height,
        width=pointcloud_camera_width,
        data_types=pointcloud_camera_data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=POINTCLOUD_CAMERA_QUAT, convention="world"),
    )

    close_finger_joints = CLOSE_FINGER_JOINT_VALUES

    open_finger_joints = OPEN_FINGER_JOINT_VALUES

    door_body_names = DOOR_BODY_NAMES

    door_base_frame_name = "base"

    door_joint_names = DOOR_JOINT_NAMES

    robot_key_bodies = ROBOT_KEY_BODY_NAMES
    robot_reset_key_bodies = ROBOT_RESET_KEY_BODY_NAMES

    robot_palm_link_name = ROBOT_PALM_LINK_NAME
    robot_base_body_link_name = ROBOT_BASE_BODY_LINK_NAME

    # robot(s)
    robot_cfg: ArticulationCfg = GLORBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=CAMERA_JOINT_DEFAULT_VALUES,
            pos=ROBOT_INITIAL_POS,
            rot=ROBOT_INITIAL_ROT
        ),
    )

    twist_indices = [1, 5, 20]

    # door(s)
    door_cfg: ArticulationCfg = ALL_DOOR_CONFIGS.replace(prim_path="/World/envs/env_.*/Door")

    actuated_joints_num = len(arm_joints) + len(base_joints) + len(finger_joints)
    action_space = actuated_joints_num * 1
    # action_space = len(arm_joints) + len(base_joints) + 4
    # observation_space = actuated_joints_num * 2 + len(door_body_names) * 3 + len(robot_key_bodies) * 3 * 2 + len(door_joint_names) + len(door_joint_names) + len(contact_sensor_names) * 3
    observation_space = \
        actuated_joints_num * 2 +\
        len(door_body_names) * 3 +\
        len(arm_joints) + len(base_joints) +\
        (len(robot_key_bodies) - 1) * (3 + 6) + 6 +\
        len(robot_key_bodies) * 3 +\
        len(door_joint_names) * 2 +\
        actuated_joints_num
    state_space = observation_space
    num_observations = observation_space
    num_states = state_space
    #  5 * 3 +\
    # len(twist_indices) * (len(robot_key_bodies) * 3 + len(robot_key_bodies) * 6 + 3 + len(door_joint_names) + len(arm_joints) + len(base_joints)) +\
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=False)

    base_action_scale = 1.0
    arm_action_scale = 0.6
    finger_action_scale = 0.5

    # Deep Mimic Reward Parameters
    robot_body_quat_w = 1.0
    robot_key_body_pos_w = 2.0
    robot_base_joint_pos_w = 3.0
    robot_arm_joint_pos_w = 3.0
    robot_finger_joint_pos_w = 1.0
    robot_base_joint_vel_w = 1.0
    robot_arm_joint_vel_w = 2.0
    robot_finger_joint_vel_w = 0.5
    door_joint_pos_w = 4.0
    hinge_contact_reward_w = 1.0
    robot_body_lin_vel_w = 1.0
    robot_body_ang_vel_w = 0.5
    joint_limit_penalty_w = 1.0
    joint_limit_penalty_margin_ratio = 0.1

    robot_body_quat_scale = 1.0
    robot_key_body_pos_scale = 3.0
    robot_base_joint_pos_scale = 0.5
    robot_arm_joint_pos_scale = 0.2
    robot_finger_joint_pos_scale = 1.0
    robot_base_joint_vel_scale = 0.5
    robot_arm_joint_vel_scale = 0.5
    robot_finger_joint_vel_scale = 0.5
    door_joint_pos_scale = 5.0
    robot_body_lin_vel_scale = 10.0
    robot_body_ang_vel_scale = 10.0

    reset_key_body_pos_delta_min = 0.5
    reset_key_body_quat_delta_min = 1.5
    reset_key_body_pos_delta_max = 0.9
    reset_key_body_quat_delta_max = 3.0
    reset_door_joint_pos_delta_min = 0.5
    reset_door_joint_pos_delta_max = 0.8
    # We are slowly increasing our tolerance on base position drift and slowly only resettting the env from the first key frame
    # This variable is used to indicate when we stop increasing the tolerance and reset the env from the first key frame for the greatest probability
    reset_progress_total = 7e5

    alive_base = 10.0
    alive_bonus = 20.0
    termination_penalty = -100.0

    velocity = 1.0

    enable_adr = False
    num_adr_increments = 20
    starting_adr_increments = num_adr_increments

    events: EventCfg = EventCfg()

    adr_cfg_dict = {
        "num_increments": num_adr_increments,
        "robot_physics_material": {
            "static_friction_range": (0.6, 1.25),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.0),
        },
        "door_physics_material": {
            "static_friction_range": (0.6, 1.25),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.0),
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
        },
        "robot_joint_friction": {
            "friction_distribution_params": (0.0, 0.02),
        },
        "door_latch_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
        },
        "door_hinge_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
        },
        "door_joint_friction": {
            "friction_distribution_params": (0.0, 0.02),
        },
    }

    adr_custom_cfg_dict = {
        "robot_spawn": {
            "base_xy_joint_pos_noise": (0.0, 0.01),
            "base_rot_joint_pos_noise": (0.0, 0.03),
            "arm_joint_pos_noise": (0.0, 0.03),
            "finger_joint_pos_noise": (0.0, 0.05),
        },
        "robot_state_noise": {
            "base_xy_joint_pos_noise": (0.0, 0.003),
            "base_xy_joint_pos_bias": (0.0, 0.002),
            "base_rot_joint_pos_noise": (0.0, 0.01),
            "base_rot_joint_pos_bias": (0.0, 0.006),
            "arm_joint_pos_noise": (0.0, 0.01),
            "arm_joint_pos_bias": (0.0, 0.006),
            "finger_joint_pos_noise": (0.0, 0.02),
            "finger_joint_pos_bias": (0.0, 0.01),
            "base_xy_joint_vel_noise": (0.0, 0.03),
            "base_xy_joint_vel_bias": (0.0, 0.015),
            "base_rot_joint_vel_noise": (0.0, 0.08),
            "base_rot_joint_vel_bias": (0.0, 0.04),
            "arm_joint_vel_noise": (0.0, 0.1),
            "arm_joint_vel_bias": (0.0, 0.05),
            "finger_joint_vel_noise": (0.0, 0.15),
            "finger_joint_vel_bias": (0.0, 0.08),
            "key_body_pos_noise": (0.0, 0.01),
            "key_body_pos_bias": (0.0, 0.005),
            "key_body_rot_noise": (0.0, 0.02),
            "key_body_rot_bias": (0.0, 0.01),
            "base_lin_vel_noise": (0.0, 0.05),
            "base_lin_vel_bias": (0.0, 0.025),
            "base_ang_vel_noise": (0.0, 0.08),
            "base_ang_vel_bias": (0.0, 0.04),
            "door_to_base_pos_noise": (0.0, 0.01),
            "door_to_base_pos_bias": (0.0, 0.005),
            "key_pos_err_noise": (0.0, 0.01),
            "key_pos_err_bias": (0.0, 0.005),
        },
        "pd_targets": {
            "base_xy_target_noise": (0.0, 0.0015),
            "base_rot_target_noise": (0.0, 0.005),
            "arm_target_noise": (0.0, 0.003),
            "finger_target_noise": (0.0, 0.005),
            "target_lag_alpha": (0.0, 0.15),
        },
    }

    # Change this to where you store your motions
    motion_file = "trajectory.pkl"
