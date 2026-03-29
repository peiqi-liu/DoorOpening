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
from DoorOpening.constants.env_constants import DOOR_INITIAL_POS, DOOR_INITIAL_ROT
import json
from DoorOpening.utils.urdf_utils import compute_exact_door_keypoints


def load_meta_data(board_meta_data_paths: str, handle_meta_data_paths: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the meta data from the json files.
    """
    handle_bboxes = []
    board_bboxes = []

    for handle_path, board_path in zip(handle_meta_data_paths, board_meta_data_paths):

        # ----- Handle -----
        with open(handle_path, "r") as f:
            handle_data = json.load(f)

        handle_min = handle_data["handle_min"]
        handle_max = handle_data["handle_max"]

        # xyzxyz format
        handle_bbox = torch.tensor(handle_min + handle_max,
                                dtype=torch.float32,
                                device=device)

        handle_bboxes.append(handle_bbox)

        # ----- Board -----
        with open(board_path, "r") as f:
            board_data = json.load(f)

        board_min = board_data["min"]
        board_max = board_data["max"]

        board_bbox = torch.tensor(board_min + board_max,
                                dtype=torch.float32,
                                device=device)

        board_bboxes.append(board_bbox)
    
    return handle_bboxes, board_bboxes

def create_initial_state():
    return ArticulationCfg.InitialStateCfg(
        pos=DOOR_INITIAL_POS,
        rot=DOOR_INITIAL_ROT
    )

def create_actuators():
    return {
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
    }

def create_urdf_door_cfg(
    asset_path: str,
    training_mode: bool = False,
    activate_contact_sensors: bool = True,
):
    return sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=True,
            make_instanceable=False,
            asset_path=asset_path,
            # Keep Isaac Lab's default absolute temp USD path here.
            # The relative repo-local cache path was generating broken mobility sublayer references.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=5,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            # Note: joint_drive is usually not needed for URDF; PD gains can be in actuators
            # scale = (1.0, 1.2, 1.1),
            activate_contact_sensors=activate_contact_sensors,
            collider_type = "convex_hull" if training_mode else "convex_decomposition",
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.03, rest_offset=0.0),
    )

def create_door_cfg(
    asset_path: str,
    training_mode: bool = False,
    activate_contact_sensors: bool = True,
) -> ArticulationCfg:
    """Helper to create an ArticulationCfg from a URDF path."""
    return ArticulationCfg(
        spawn=create_urdf_door_cfg(
            asset_path,
            training_mode=training_mode,
            activate_contact_sensors=activate_contact_sensors,
        ),
        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=asset_path,
        #     scale = (1.0, 1.2, 0.95),
        #     activate_contact_sensors=True,
        # ),
        init_state=create_initial_state(),
        actuators=create_actuators(),
    )


root_path = os.path.dirname(os.path.dirname(__file__))
asset_base_folder = os.path.join(root_path, "door/PartNetv4")
asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.urdf"), recursive=True))
board_offsets = []
handle_offsets = []

for asset_path in asset_paths:
    keypoints = compute_exact_door_keypoints(asset_path)
    board_offsets.append(keypoints["link_1"])
    handle_offsets.append(keypoints["link_2"])

board_offsets = torch.tensor(board_offsets)
handle_offsets = torch.tensor(handle_offsets)

motion_traj_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/traj.pkl"), recursive=True))

door_asset_path = asset_paths[0]
board_offset = board_offsets[0]
handle_offset = handle_offsets[0]
print("door_asset_path: ", door_asset_path)

DOOR_CONFIG = create_door_cfg(door_asset_path, training_mode=False)

DOOR_CONFIGS = []
for asset_path in asset_paths:
    DOOR_CONFIGS.append(create_door_cfg(asset_path, training_mode=False))

def setup_doors(training_mode: bool = False):
    """Load all door cfg"""
    door_urdf_configs = []
    for asset_path in asset_paths:
        door_urdf_configs.append(
            create_urdf_door_cfg(
                asset_path,
                training_mode=training_mode,
                activate_contact_sensors=False,
            )
        )
    return ArticulationCfg(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=door_urdf_configs,
            random_choice=False,
            activate_contact_sensors=False,
        ),
        init_state=create_initial_state(),
        actuators=create_actuators(),
    )

ALL_DOOR_CONFIGS = setup_doors()


def edit_door_articulation(
    door: Articulation, 
    door_closed_range = 0.01,     # radians
    hinge_range = 0.05,
    locked_stiffness = 1e3,
    locked_damping = 1e2,
    # Optional: disable the latching behavior by setting the hinge range to a negative value
    # hinge_range = -0.1,
):
    joint_idx, joint_names = door.find_joints(["joint_1", "joint_2"])
    j1 = joint_idx[joint_names.index("joint_1")]
    j2 = joint_idx[joint_names.index("joint_2")]

    # joint positions
    q = door.data.joint_pos

    # Only relock when both joints are still very close to the closed pose.
    locked = (q[:, j1].abs() < door_closed_range) & (q[:, j2].abs() < hinge_range)

    default_joint_stiffness = door.data.default_joint_stiffness
    default_joint_damping = door.data.default_joint_damping

    # Start from the live joint gains so reset-time randomization survives the lock logic.
    joint_stiffness = door.data.joint_stiffness.clone()
    joint_damping = door.data.joint_damping.clone()
    stiffness_scale = torch.ones_like(joint_stiffness[:, j1])
    damping_scale = torch.ones_like(joint_damping[:, j1])
    valid_stiffness = default_joint_stiffness[:, j1].abs() > 1e-6
    valid_damping = default_joint_damping[:, j1].abs() > 1e-6
    stiffness_scale[valid_stiffness] = joint_stiffness[valid_stiffness, j1] / default_joint_stiffness[valid_stiffness, j1]
    damping_scale[valid_damping] = joint_damping[valid_damping, j1] / default_joint_damping[valid_damping, j1]
    joint_stiffness[locked, j1] = locked_stiffness * stiffness_scale[locked]
    joint_damping[locked, j1] = locked_damping * damping_scale[locked]
    door.write_joint_stiffness_to_sim(joint_stiffness)
    door.write_joint_damping_to_sim(joint_damping)
