# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Defines the door configuration for simulation with Isaac Sim.
"""

import os
import glob

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

def create_door_cfg(asset_path: str) -> ArticulationCfg:
    """Helper to create an ArticulationCfg from a URDF path."""
    return ArticulationCfg(
        # spawn=sim_utils.UrdfFileCfg(
        #     fix_base=True,
        #     merge_fixed_joints=False,
        #     make_instanceable=False,
        #     asset_path=asset_path,
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False,
        #         solver_position_iteration_count=4,
        #         solver_velocity_iteration_count=0,
        #     ),
        #     joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        #         gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        #     ),
        #     # Note: joint_drive is usually not needed for URDF; PD gains can be in actuators
        #     scale = (1.0, 1.2, 0.95),
        #     activate_contact_sensors=True,
        # ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset_path,
            scale = (1.0, 1.2, 0.95),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(0, 0, 0, 1)
            # rot=(0, -0.7071, 0, 0.7071)
        ),
        actuators={
            "joint_1": ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                stiffness=100,
                damping=10,
            ),
            "joint_2": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                stiffness=10,
                damping=0.1,
            ),
        },
    )


root_path = os.path.dirname(os.path.dirname(__file__))
asset_base_folder = os.path.join(root_path, "door/PartNetv2")
asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.usd"), recursive=True))

# An example of door urdf
door_asset_path = asset_paths[0]

print("door_asset_path: ", door_asset_path)

DOOR_CONFIG = create_door_cfg(door_asset_path)

def setup_doors():
    """Load all door cfg"""
    door_configs = []
    for asset_path in asset_paths:
        door_configs.append(create_door_cfg(asset_path))
    return door_configs

ALL_DOOR_CONFIGS = setup_doors()