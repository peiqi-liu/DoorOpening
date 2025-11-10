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

# default camera pose for the camera to look front
CAMERA_JOINT_DEFAULT_VALUES = {
    "x5_joint1": 0.0, 
    "x5_joint2": 0.785, 
    "x5_joint3": 0.785, 
    "x5_joint4": 0.0, 
    "x5_joint5": 0.0, 
    "x5_joint6": 0.0,
}

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
        joint_pos=CAMERA_JOINT_DEFAULT_VALUES,
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0)
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=1000.0,
            damping=200.0,
        ),
    },
)