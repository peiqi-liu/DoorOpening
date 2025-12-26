# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG, CAMERA_JOINT_DEFAULT_VALUES
from DoorOpening.assets.door.door_cfg import DOOR_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.common import ViewerCfg

@configclass
class DooropeningEnvCfg(DirectRLEnvCfg):
    sim_dt = 1/60.
    decimation = 2
    episode_length_s = 3.
    fabric_decimation = 2 # number of fabric steps per physics step
    num_sim_steps_to_render=2
    # - spaces definition
    state_space = 0

    viewer: ViewerCfg = ViewerCfg(eye=(0.5, 2.0, 0.5), lookat=(0.5, 0.0, 0.6), origin_type="env")

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

    door_body_names = ["link_1", "link_2"]
    door_handle_body_name = "link_1"

    # robot_key_bodies = ["base_x_link", "panda_link1",  "panda_link2",  "panda_link3",  "panda_link4",  "panda_link5",  "panda_link6",  "panda_link7",  "palm_center"]
    # robot_key_bodies = ["base_x_link",  "panda_link4", "palm_center", "fingertip_3"]
    robot_key_bodies = ["base_x_link",  "palm_center"]

    # robot(s)
    robot_cfg: ArticulationCfg = GLORBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=CAMERA_JOINT_DEFAULT_VALUES,
            pos=(1.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0)
        ),
    )

    # door(s)
    door_cfg: ArticulationCfg = DOOR_CONFIG.replace(prim_path="/World/envs/env_.*/Door")

    actuated_joints_num = len(arm_joints) + len(base_joints) + len(finger_joints)
    action_space = actuated_joints_num * 1
    observation_space = actuated_joints_num * 2 + len(door_body_names) * 3 + len(robot_key_bodies) * 3

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=6.0, replicate_physics=True)

    action_scale = 10.0

    # Deep Mimic Reward Parameters
    robot_body_quat_w = 1.0
    robot_key_body_pos_w = 3.0
    door_joint_pos_w = 3.0
    robot_base_joint_pos_w = 1.0
    robot_arm_joint_pos_w = 1.5
    robot_finger_joint_pos_w = 1.0

    robot_body_quat_scale = 1.0
    robot_key_body_pos_scale = 3.0
    robot_base_joint_pos_scale = 0.5
    robot_arm_joint_pos_scale = 0.5
    robot_finger_joint_pos_scale = 0.5
    door_joint_pos_scale = 5.0

    reset_base_pos_delta = 0.1
    reset_key_body_pos_delta = 0.2
    reset_door_pos_delta = 0.25

    velocity = 0.6

    # Change this to where you store your motions
    motion_file = "trajectory.pkl"