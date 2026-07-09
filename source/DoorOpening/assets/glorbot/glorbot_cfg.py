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
ROBOT_CONTACT_OFFSET = 0.005
ROBOT_REST_OFFSET = 0.001
ROBOT_MAX_DEPENETRATION_VELOCITY = 500.0


def disable_collision_scope_instancing(robot_prim_path_expr: str = "/World/envs/env_.*/Robot") -> int:
    """De-instance each link's ``collisions`` scope so the collider debug viz renders.

    The URDF->USD converter authors every link's ``collisions`` (and ``visuals``) scope as a
    USD instance referencing a shared prototype, and Isaac Sim's collider debug visualization
    (the green overlay) does not draw colliders that live inside instanced prototypes. Setting
    ``make_instanceable=False`` in the converter cfg does NOT change this (verified). This flips
    only the ``collisions`` scopes to uninstanceable (``visuals`` stay instanced), so physics and
    memory are unaffected. Call once after the robot is spawned. Returns the count de-instanced.
    """
    from pxr import Usd
    import isaaclab.sim as sim_utils

    scopes = [
        scope
        for robot_prim in sim_utils.find_matching_prims(robot_prim_path_expr)
        for scope in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())
        if scope.GetName() == "collisions" and scope.IsInstance()
    ]
    for scope in scopes:
        scope.SetInstanceable(False)
    return len(scopes)


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
        # Decompose collision meshes into convex pieces (default is a single convex_hull, which
        # would turn the chassis + tall lidar-mast mesh into one solid wedge). Needed so the
        # tidybot2_base_link mast collision approximates the real thin stick, not a big blob.
        # collider_type="convex_decomposition",
        collider_type="convex_hull",
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
            damping=3000,
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
            stiffness=600,
            damping=40,
            # Joint friction for the LEAP fingers (was unset -> 0). Matches the real geared-Dynamixel
            # finger friction and damps overshoot alongside the armature below.
            friction=0.01,
            # The LEAP finger links are ultralight (izz ~1e-5 kg m^2). With a 1.0 Nm effort limit that
            # is ~1e5 rad/s^2 of angular acceleration when the PD saturates, so the fingers overshoot
            # in a single step and jitter. Armature adds effective rotor inertia the implicit PD sees,
            # capping the per-step acceleration. NOTE: this is only the nominal / ADR-increment-0 value;
            # at training time the `robot_finger_armature` EventTerm (see multi_dooropening_env_cfg.py)
            # randomizes it per-episode -- lower that ADR range too if you want 0.002 to hold in training.
            armature=0.002,
        ),
    }
)
