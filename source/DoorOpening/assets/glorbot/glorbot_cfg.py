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

from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from DoorOpening.assets.cache_utils import resolve_converter_cache_dir, should_force_usd_conversion

module_path = Path(__file__).resolve().parent
root_path = module_path.parent
glorbot_urdf_path = str(root_path / "glorbot" / "glorbot.urdf")
glorbot_usd_dir = resolve_converter_cache_dir(glorbot_urdf_path, asset_root=root_path)
glorbot_usd_file_name = "glorbot.usd"
glorbot_usd_path = str(Path(glorbot_usd_dir) / glorbot_usd_file_name)

import numpy as np

# default camera pose for the camera to look front
from DoorOpening.constants.robot_constants import DEFAULT_JOINT_POS, OPEN_FINGER_JOINT_VALUES, CLOSE_FINGER_JOINT_VALUES, FULL_JOINT_NAMES, BASE_JOINT_NAMES, FRANKA_JOINT_NAMES
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT

ROBOT_SOLVER_POSITION_ITERS = 8
ROBOT_SOLVER_VELOCITY_ITERS = 2
ROBOT_CONTACT_OFFSET = 0.01
ROBOT_REST_OFFSET = 0.001
ROBOT_MAX_DEPENETRATION_VELOCITY = 500.0


GLORBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        asset_path=glorbot_urdf_path,
        usd_dir=glorbot_usd_dir,
        usd_file_name=glorbot_usd_file_name,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=ROBOT_MAX_DEPENETRATION_VELOCITY,
            solver_position_iteration_count=ROBOT_SOLVER_POSITION_ITERS,
            solver_velocity_iteration_count=ROBOT_SOLVER_VELOCITY_ITERS,
        ),
        force_usd_conversion=should_force_usd_conversion(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=False,
            enabled_self_collisions=True,
            solver_position_iteration_count=ROBOT_SOLVER_POSITION_ITERS,
            solver_velocity_iteration_count=ROBOT_SOLVER_VELOCITY_ITERS,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=ROBOT_CONTACT_OFFSET,
            rest_offset=ROBOT_REST_OFFSET,
        ),
        # scale = (0.8, 0.8, 0.8),
        activate_contact_sensors=True,
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
            stiffness=10000,
            damping=1000,
        ),
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=50.0,
            velocity_limit_sim=2.175,
            stiffness=600.0,
            damping=100.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=190.0,
            velocity_limit_sim=2.61,
            stiffness=600.0,
            damping=100.0,
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
            effort_limit_sim=1.0,
            stiffness=60,
            damping=1,
        ),
    }
)
