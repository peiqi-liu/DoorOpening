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
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

module_path = os.path.dirname(__file__)
root_path = os.path.dirname(module_path)
glorbot_urdf_path = os.path.join(root_path, "glorbot/glorbot.urdf")
glorbot_usd_path = os.path.join(root_path, "glorbot/glorbot.usd")
print("glorbot_usd_path: ", glorbot_usd_path)

import numpy as np

# default camera pose for the camera to look front
from DoorOpening.constants.robot_constants import DEFAULT_JOINT_POS, OPEN_FINGER_JOINT_VALUES, CLOSE_FINGER_JOINT_VALUES, FULL_JOINT_NAMES, BASE_JOINT_NAMES, FRANKA_JOINT_NAMES
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT

from datetime import datetime
import random


def _make_usd_dir() -> str:
    cache_root = os.path.expanduser("IsaacLab_tmp")
    os.makedirs(cache_root, exist_ok=True)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(cache_root, f"usd_{time_tag}_{random.randrange(10000)}")


GLORBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        asset_path=glorbot_urdf_path,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        # scale = (0.8, 0.8, 0.8),
        activate_contact_sensors=True,
        usd_dir=_make_usd_dir(),
    ),
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path=glorbot_usd_path,
    #     scale = (0.8, 0.8, 0.8),
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=DEFAULT_JOINT_POS,
        pos=ROBOT_INITIAL_POS,
        rot=ROBOT_INITIAL_ROT
    ),
    actuators={
        # "body": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     stiffness=4000.0,
        #     damping=2000.0,
        # ),
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_.*"],
            effort_limit_sim=1000.0,
            stiffness=1000,
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
            effort_limit_sim=50,
            stiffness=800,
            damping=80,
        ),
    }
)
