# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from DoorOpening.constants.robot_constants import CAMERA_JOINT_DEFAULT_VALUES, CLOSE_FINGER_JOINT_VALUES, OPEN_FINGER_JOINT_VALUES, ROBOT_KEY_BODY_NAMES, ROBOT_RESET_KEY_BODY_NAMES    
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG, ALL_DOOR_CONFIGS
from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.common import ViewerCfg
import torch
from isaaclab.sensors import ContactSensorCfg

@configclass
class DooropeningEnvCfg(DirectRLEnvCfg):
    sim_dt = 1/60.
    decimation = 1
    episode_length_s = 10.
    num_sim_steps_to_render=2
    # - spaces definition
    state_space = 0

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

    finger_joints = [
        'finger_joint_0',
        'finger_joint_1',
        'finger_joint_2',
        'finger_joint_3',
        'finger_joint_4',
        'finger_joint_5',
        'finger_joint_6',
        'finger_joint_7',
        'finger_joint_8',
        'finger_joint_9',
        'finger_joint_10',
        'finger_joint_11',
        'finger_joint_12',
        'finger_joint_13',
        'finger_joint_14',
        'finger_joint_15',
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
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot"],
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

    close_finger_joints = CLOSE_FINGER_JOINT_VALUES

    open_finger_joints = OPEN_FINGER_JOINT_VALUES

    door_body_names = ["link_1", "link_2"]

    door_joint_names = ["joint_1", "joint_2"]

    robot_key_bodies = ROBOT_KEY_BODY_NAMES
    robot_reset_key_bodies = ROBOT_RESET_KEY_BODY_NAMES

    robot_palm_link_name = "palm_center"
    robot_base_body_link_name = "tidybot2_base_link"

    # robot(s)
    robot_cfg: ArticulationCfg = GLORBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=CAMERA_JOINT_DEFAULT_VALUES,
            pos=ROBOT_INITIAL_POS,
            rot=ROBOT_INITIAL_ROT
        ),
    )

    twist_indices = [1, 5, 20, 100]

    # door(s)
    door_cfg: ArticulationCfg = ALL_DOOR_CONFIGS.replace(prim_path="/World/envs/env_.*/Door")

    actuated_joints_num = len(arm_joints) + len(base_joints) + len(finger_joints)
    action_space = actuated_joints_num * 1
    # action_space = len(arm_joints) + len(base_joints) + 4
    # observation_space = actuated_joints_num * 2 + len(door_body_names) * 3 + len(robot_key_bodies) * 3 * 2 + len(door_joint_names) + len(door_joint_names) + len(contact_sensor_names) * 3
    observation_space = \
        actuated_joints_num * 2 +\
        len(door_body_names) * 3 +\
        len(robot_key_bodies) * 3 * 2 +\
        len(door_joint_names) * 2 +\
        len(twist_indices) * (len(robot_key_bodies) * 3 + len(robot_key_bodies) * 4 + len(door_joint_names)) +\
        actuated_joints_num
    #  5 * 3 +\
    
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

    reset_base_pos_delta_min = 0.35
    reset_key_body_pos_delta_min = 0.6
    reset_key_body_quat_delta_min = 0.8
    reset_base_pos_delta_max = 1.0
    reset_key_body_pos_delta_max = 1.5
    reset_key_body_quat_delta_max = 3.0
    reset_door_joint_pos_delta_min = 0.5
    reset_door_joint_pos_delta_max = 0.8
    # We are slowly increasing our tolerance on base position drift and slowly only resettting the env from the first key frame
    # This variable is used to indicate when we stop increasing the tolerance and reset the env from the first key frame for the greatest probability
    reset_progress_total = 1e6

    velocity = 1.0

    # Change this to where you store your motions
    motion_file = "trajectory.pkl"