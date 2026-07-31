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

# LEAP hand finger PD gains. The three abduction ("MCP Side") DOFs -- finger_joint_0/4/8 -- are run at
# ABDUCTION_GAIN_SCALE of the flexion gains to mirror the real hand, whose leap_hand_node.py drives
# motor IDs 0/4/8 at 0.75x the P/D gain of every other joint. That softer PD is what makes those
# joints droop ~0.3 rad under grasp load on hardware; matching the asymmetry here reproduces the
# steady-state tracking error in sim instead of tracking the target cleanly. Set to 1.0 for the old
# uniform-stiffness behavior.
FINGER_STIFFNESS = 600
FINGER_DAMPING = 40
ABDUCTION_GAIN_SCALE = 0.75


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
        collider_type="convex_decomposition",
        # collider_type="convex_hull",
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
        # Per-joint Franka PD gains matched to the REAL-WORLD deploy controller (kp/kd used on hardware),
        # set per joint because the real values vary within each group (kd differs across joint1-4; both
        # kp and kd differ across joint5-7). stiffness/damping accept a {joint_name: value} dict so each
        # joint gets its exact gain. NOTE: these are the ADR-increment-0 nominal gains -- at eval the
        # `robot_joint_stiffness_and_damping` EventTerm still scales them by the ADR band (kp x[0.8,1.2],
        # kd x[0.7,1.3]); pin the ADR increment / bypass _force_max_adr_for_eval to hold these exactly.
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=50.0,
            velocity_limit_sim=2.175,
            stiffness=1600.0,
            damping={
                "panda_joint1": 24.0,
                "panda_joint2": 20.0,
                "panda_joint3": 24.0,
                "panda_joint4": 24.0,
            },
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=190.0,
            velocity_limit_sim=2.61,
            stiffness={
                "panda_joint5": 410.0,
                "panda_joint6": 410.0,
                "panda_joint7": 130.0,
            },
            damping={
                "panda_joint5": 6.0,
                "panda_joint6": 6.0,
                "panda_joint7": 4.0,
            },
        ),
        "x5_arm": ImplicitActuatorCfg(
            joint_names_expr=["x5_joint[1-6]"],
            effort_limit_sim=1440.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),
        # Index + middle + ring FLEXION joints. The abduction ("MCP Side") DOFs finger_joint_0/4/8 are
        # split into the "finger_abduction" group below so they can run at the softer real-hand gains.
        # Thumb (finger_joint_12..15) is a separate group so it can carry a higher effort limit.
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint_(1|2|3|5|6|7|9|1[01])"],
            effort_limit_sim=1.0,
            stiffness=FINGER_STIFFNESS,
            damping=FINGER_DAMPING,
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
        # Abduction ("MCP Side") DOFs finger_joint_0/4/8 (real motor IDs 0/4/8). Same params as the
        # flexion "finger" group EXCEPT the P/D gains are scaled by ABDUCTION_GAIN_SCALE to mirror the
        # real hand's softer side-to-side PD (kp/kd 0.75x), reproducing its grasp-load droop in sim.
        "finger_abduction": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint_(0|4|8)"],
            effort_limit_sim=1.0,
            stiffness=FINGER_STIFFNESS * ABDUCTION_GAIN_SCALE,
            damping=FINGER_DAMPING * ABDUCTION_GAIN_SCALE,
            friction=0.01,
            armature=0.002,
        ),
        # Thumb (finger_joint_12..15). Split out from the other fingers with a HIGHER effort limit:
        # the thumb is NOT policy-controlled (commented out of finger_joints), so it is only held by
        # this PD at a fixed reset target. At 1.0 Nm it got shoved off that target by self-collision
        # from the closing fingers + inertial loads during arm motion and never recovered ("fell"). A
        # larger effort limit lets it hold. Same gains/friction/armature as the fingers otherwise; the
        # armature still caps per-step acceleration so the higher cap stays stable. Tune if needed.
        "thumb": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint_1[2-5]"],
            effort_limit_sim=5.0,
            stiffness=600,
            damping=40,
            friction=0.01,
            armature=0.002,
        ),
    }
)
