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
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg, Articulation

def create_urdf_door_cfg(asset_path: str, training_mode: bool = False):
    return sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=False,
        asset_path=asset_path,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        scale = (1.0, 1.2, 0.95),
        activate_contact_sensors=True,
        collider_type = "convex_hull" if training_mode else "convex_decomposition",
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    )

def create_door_cfg(asset_path: str, training_mode: bool = False) -> ArticulationCfg:
    """Helper to create an ArticulationCfg from a URDF path."""
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            asset_path=asset_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=10,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            # Note: joint_drive is usually not needed for URDF; PD gains can be in actuators
            scale = (1.0, 1.3, 1.0),
            activate_contact_sensors=True,
            collider_type = "convex_hull" if training_mode else "convex_decomposition",
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.03, rest_offset=0.0),
        ),
        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=asset_path,
        #     scale = (1.0, 1.2, 0.95),
        #     activate_contact_sensors=True,
        # ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.02),
            rot=(0, 0, 0, 1)
            # rot=(0, -0.7071, 0, 0.7071)
        ),
        actuators={
            "joint_1": ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                stiffness=5,
                damping=1,
            ),
            "joint_2": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                stiffness=2.5,
                damping=1,
            ),
        },
    )


root_path = os.path.dirname(os.path.dirname(__file__))
asset_base_folder = os.path.join(root_path, "door/PartNetv4")
asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.urdf"), recursive=True))

motion_traj_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/traj.pkl"), recursive=True))

# for i in [7, 10, 13, 20, 25, 27, 28, 29]:
#     print(asset_paths[i - 1])
# An example of door urdf
door_asset_path = asset_paths[0]
print("door_asset_path: ", door_asset_path)

DOOR_CONFIG = create_door_cfg(door_asset_path, training_mode=False)

DOOR_CONFIGS = []
for asset_path in asset_paths:
    DOOR_CONFIGS.append(create_door_cfg(asset_path, training_mode=False))

def setup_doors(training_mode: bool = False):
    """Load all door cfg"""
    door_urdf_configs = []
    for asset_path in asset_paths:
        door_urdf_configs.append(create_urdf_door_cfg(asset_path, training_mode=training_mode))
    return ArticulationCfg(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=door_urdf_configs,
            random_choice=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.91),
            rot=(0, 0, 0, 1)
            # rot=(0, -0.7071, 0, 0.7071)
        ),
        actuators={
            "joint_1": ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                stiffness=5,
                damping=1,
            ),
            "joint_2": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                stiffness=2.5,
                damping=1,
            ),
        },
    )

ALL_DOOR_CONFIGS = setup_doors()


def edit_door_articulation(
    door: Articulation, 
    door_closed_range = 0.01,     # radians
    hinge_range = 0.4,
    # Optional: disable the latching behavior by setting the hinge range to a negative value
    # hinge_range = -0.1,
):
    joint_idx, joint_names = door.find_joints(["joint_1", "joint_2"])
    j1 = joint_idx[joint_names.index("joint_1")]
    j2 = joint_idx[joint_names.index("joint_2")]

    # joint positions
    q = door.data.joint_pos

    # locked mask: (num_envs,)
    locked = (q[:, j1].abs() < door_closed_range) & (q[:, j2].abs() < hinge_range)

    joint_stiffness = door.data.default_joint_stiffness.clone()
    joint_damping = door.data.default_joint_damping.clone()
    joint_stiffness[locked, j1] = 1e6
    joint_damping[locked, j1] = 1e5
    door.write_joint_stiffness_to_sim(joint_stiffness)
    door.write_joint_damping_to_sim(joint_damping)