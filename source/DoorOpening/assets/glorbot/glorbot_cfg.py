# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Defines the Glorbot robot configuration for simulation with Isaac Sim.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

module_path = os.path.dirname(__file__)
root_path = os.path.dirname(module_path)
glorbot_urdf_path = os.path.join(root_path, "glorbot/glorbot.urdf")

import numpy as np

# default camera pose for the camera to look front
CAMERA_JOINT_DEFAULT_VALUES = {
    "x5_joint1": 0.0, 
    "x5_joint2": 0.785, 
    "x5_joint3": 0.785, 
    "x5_joint4": 0.0, 
    "x5_joint5": 0.0, 
    "x5_joint6": 0.0,
}

FRANKA_DEFAULT_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.25 * np.pi,
    "panda_joint3": 0.0,
    "panda_joint4": -0.75 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 0.5 * np.pi,
    "panda_joint7": 0.0,
}

DEFAULT_JOINT_POS = {
    "x5_joint1": 0.0, 
    "x5_joint2": 0.785, 
    "x5_joint3": 0.785, 
    "x5_joint4": 0.0, 
    "x5_joint5": 0.0, 
    "x5_joint6": 0.0,
    "panda_joint1": 0.0,
    "panda_joint2": -0.25 * np.pi,
    "panda_joint3": 0.0,
    "panda_joint4": -0.75 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 0.5 * np.pi,
    "panda_joint7": 0.0,
}

FRANKA_JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]

BASE_JOINT_NAMES = [
    'base_rotation_joint',
    'base_x_joint',
    'base_y_joint',
]

DM_JOINT_NAMES = BASE_JOINT_NAMES + FRANKA_JOINT_NAMES

GLORBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        asset_path=glorbot_urdf_path,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=DEFAULT_JOINT_POS,
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0)
    ),
    actuators={
        # "body": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     stiffness=4000.0,
        #     damping=2000.0,
        # ),
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_.*"],
            effort_limit_sim=10000.0,
            stiffness=10000,
            damping=200,
        ),
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=5200.0,
            velocity_limit_sim=2.175,
            stiffness=1100.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=720.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),
        "x5_arm": ImplicitActuatorCfg(
            joint_names_expr=["x5_joint[1-6]"],
            effort_limit_sim=1440.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint_.*"],
            effort_limit_sim=5,
            stiffness=5,
            damping=0.5,
        ),
    }
)