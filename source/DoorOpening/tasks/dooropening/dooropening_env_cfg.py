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
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

@configclass
class DooropeningEnvCfg(DirectRLEnvCfg):
    sim_dt = 1/120.
    decimation = 2 # 60 Hz
    episode_length_s = 10. #10.0
    fabric_decimation = 2 # number of fabric steps per physics step
    num_sim_steps_to_render=2
    # - spaces definition
    action_space = 26
    observation_space = 26 + 3
    state_space = 0

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

    actuated_joints = base_joints + arm_joints + finger_joints

    hand_body_name = "palm_lower"

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

    door_handle_body_name = "link_1"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    handle_pos_error_scale = 1.0