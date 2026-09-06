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

Glorbot is the tidybot2 chassis carrying a Franka Panda arm with Franka's stock 2-finger gripper,
plus an ARX X5 camera arm. 18 actuated DOFs, of which 17 are commanded (the gripper's second
finger follows the first through a mimic coupling).
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
from DoorOpening.constants.robot_constants import (
    DEFAULT_JOINT_POS,
    DRIVEN_FINGER_JOINT_NAME,
    MIMIC_FINGER_JOINT_NAME,
    OPEN_FINGER_JOINT_VALUES,
    CLOSE_FINGER_JOINT_VALUES,
    FULL_JOINT_NAMES,
    BASE_JOINT_NAMES,
    FRANKA_JOINT_NAMES,
)
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


# How the two fingers are held together as ONE DOF.
#   True  -> the URDF <mimic> tag is honored and PhysX enforces joint2 = joint1 with a
#            PhysxMimicJointAPI constraint (the follower's PD is then switched off below).
#   False -> the mimic tag is ignored, joint2 is an ordinary joint, and the coupling is done in
#            software by every controller writing the same target to both DOFs.
# Both give a single commanded DOF. Flip to False if the fingers behave asymmetrically (which
# means the constraint was not created) or to rule the constraint out while debugging.
HONOR_URDF_MIMIC = True

# ---------------------------------------------------------------------------------------------
# Gripper drive parameters, taken from the Franka Hand Product Manual (R50010, rev 1.2, section
# 5.1 "Mechanical Data") rather than from Isaac Lab's stock Franka asset. The manual's numbers:
#     Grasping (continuous) force adjustable  30-70   N
#     Travel Range                            80      mm  (i.e. 0.04 per finger -- matches URDF)
#     Travel Speed (per finger)               50      mm/s
#     Weight                                  730     g
#
# effort_limit = the top of the hardware's continuous band. Isaac Lab's stock 200 N is ~3x what
# the real hand can produce; on a 15 g finger it is also ~13,000 m/s^2, enough to cross the whole
# 40 mm travel in 2.5 ms (a 60 Hz step is 16.7 ms), so the drive slams rather than moves.
GRIPPER_EFFORT_LIMIT = 70.0

# Max finger speed, straight from the manual. NOTE this is 4x SLOWER than the 0.2 m/s that Isaac
# Lab's Franka and this URDF's <limit> author: the real hand needs 0.8 s to close from fully open,
# not 0.2 s. Worth keeping honest -- a policy trained against a 4x-fast gripper learns grasp
# timing that does not exist on hardware.
GRIPPER_VELOCITY_LIMIT = 0.05

# The manual specifies force, not stiffness, because the real hand is force-controlled
# (libfranka's grasp() takes a target force). Under Isaac Lab's implicit POSITION drive the grasp
# force is instead an emergent quantity: F = stiffness * (commanded - actual), clipped at
# effort_limit. So stiffness is chosen to land the resulting force inside the hardware's 30-70 N
# band for a realistic command. Commanding "fully closed" (0.0) on a ~20 mm door handle leaves
# 10 mm of unreachable travel per finger, which gives:
#     k = 2e3 (Isaac Lab stock) ->  20 N   below the hardware's 30 N minimum
#     k = 5e3                   ->  50 N   mid-band  <-- chosen
#     k = 1e4                   -> 100 N   clipped to 70 N, i.e. always saturated
GRIPPER_STIFFNESS = 5e3

# Damping is picked for zeta >= 1 in BOTH coupling modes, since the effective mass seen by the
# driven DOF doubles when the mimic constraint drags the second finger along:
#     m_eff = 0.065 kg (one finger + armature) -> d_crit = 36 -> zeta 1.66
#     m_eff = 0.130 kg (both, via the mimic)   -> d_crit = 51 -> zeta 1.18
# No ringing either way. Isaac Lab's stock 1e2 also satisfies this but is ~2x more sluggish.
GRIPPER_DAMPING = 60.0

# Added mass (kg) on each finger DOF. Without it the 15 g fingers plus a high effort limit
# overshoot within a single step and jitter. The real fingers are driven through a screw drive
# whose reflected inertia dwarfs the 15 g finger, so this is physical, not a fudge -- but it is
# the first knob to turn if the fingers feel sluggish.
GRIPPER_ARMATURE = 0.05


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
        # Honor the <mimic> tag on panda_finger_joint2 so the gripper is ONE commanded DOF, like
        # the real single-actuator Franka hand.
        # WARNING: this field's name is the opposite of what it does. UrdfConverter passes it
        # verbatim to the importer's `set_parse_mimic()` (urdf_converter.py: "import_config.
        # set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)"), and in the importer
        # parse_mimic=True means "emit a PhysxMimicJointAPI constraint" while False means "treat
        # the mimic joint as an ordinary independent joint" -- its GUI checkbox for this is
        # literally labelled "Ignore Mimic" with `default_val=not config.parse_mimic`. So True
        # here KEEPS the coupling; the default (False) would silently give two free fingers.
        convert_mimic_joints_to_normal_joints=HONOR_URDF_MIMIC,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
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
        # A single convex hull would swallow the lidar mast into a
        # solid wedge. It also matters for the gripper itself -- hand.obj is a concave C shape, and
        # a single hull would fill the slot the fingers slide in.
        collider_type="convex_decomposition",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=DEFAULT_JOINT_POS,
        pos=ROBOT_INITIAL_POS,
        rot=ROBOT_INITIAL_ROT,
    ),
    actuators={
        # Split translation/rotation into separate groups -- their gains are not the same physical
        # quantity (prismatic N/m vs revolute N-m/rad), and both land overdamped where the dominant
        # pole is -k/d, so a shared group gave translation (56.5 kg) and yaw (~3 kg-m^2) an identical
        # 0.3 s lag. Rotation damping is 10x lower so yaw responds ~10x faster, as on the real base.
        # Base gains taken from the gripper branch, which runs the same q_base + dt*scale*action
        # command form. The target is re-anchored to the measured pose every step, so the PD error is
        # pinned at dt*scale*action and the realized speed is (kp*dt/kd) * commanded.
        "base_translation": ImplicitActuatorCfg(
            joint_names_expr=["base_x_joint", "base_y_joint"],
            effort_limit_sim=1000.0,
            stiffness=80000,  # was 10000
            damping=4000,  # was 3000
        ),
        "base_rotation": ImplicitActuatorCfg(
            joint_names_expr=["base_rotation_joint"],
            effort_limit_sim=1000.0,
            stiffness=11000,  # was 10000
            damping=400,  # was 300
        ),
        # Per-joint Franka PD gains matched to the REAL-WORLD deploy controller (kp/kd used on
        # hardware), set per joint because the real values vary within each group. effort_limit_sim
        # is deliberately unset: when None on an IMPLICIT actuator, IsaacLab keeps the URDF-authored
        # limit (87 N-m on joint1-4, 12 N-m on joint5-7) instead of a hand-picked cap.
        # kp and kd both scaled 0.8x, so kp/kd (the pole) is unchanged and only the force
        # authority drops -- same response shape, softer arm.
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            velocity_limit_sim=2.175,
            stiffness=1280.0,  # was 1600.0
            damping={
                "panda_joint1": 19.2,  # was 24.0
                "panda_joint2": 16.0,  # was 20.0
                "panda_joint3": 19.2,  # was 24.0
                "panda_joint4": 19.2,  # was 24.0
            },
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            velocity_limit_sim=2.61,
            stiffness={
                "panda_joint5": 328.0,  # was 410.0
                "panda_joint6": 328.0,  # was 410.0
                "panda_joint7": 104.0,  # was 130.0
            },
            damping={
                "panda_joint5": 4.8,  # was 6.0
                "panda_joint6": 4.8,  # was 6.0
                "panda_joint7": 3.2,  # was 4.0
            },
        ),
        "x5_arm": ImplicitActuatorCfg(
            joint_names_expr=["x5_joint[1-6]"],
            effort_limit_sim=1440.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),
        # The commanded finger DOF.
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=[DRIVEN_FINGER_JOINT_NAME],
            effort_limit_sim=GRIPPER_EFFORT_LIMIT,
            velocity_limit_sim=GRIPPER_VELOCITY_LIMIT,
            stiffness=GRIPPER_STIFFNESS,
            damping=GRIPPER_DAMPING,
            armature=GRIPPER_ARMATURE,
        ),
        # The follower. Its gains depend on HOW the coupling is enforced, which is why they are
        # derived from HONOR_URDF_MIMIC rather than hardcoded:
        #   mimic constraint  -> zero gains. A PhysX mimic joint already forces joint2 = joint1;
        #       leaving a 2000 N/m PD on joint2 as well is redundant actuation, and the drive and
        #       the constraint then fight each other every step, which chatters.
        #   software coupling -> same gains as the master, since then joint2 is an ordinary free
        #       joint and the controller's duplicated target is the only thing holding it.
        # If the fingers ever go limp/asymmetric, the constraint was not created: flip
        # HONOR_URDF_MIMIC to False and the software path takes over.
        "panda_hand_follower": ImplicitActuatorCfg(
            joint_names_expr=[MIMIC_FINGER_JOINT_NAME],
            effort_limit_sim=GRIPPER_EFFORT_LIMIT,
            velocity_limit_sim=GRIPPER_VELOCITY_LIMIT,
            stiffness=0.0 if HONOR_URDF_MIMIC else GRIPPER_STIFFNESS,
            damping=0.0 if HONOR_URDF_MIMIC else GRIPPER_DAMPING,
            armature=GRIPPER_ARMATURE,
        ),
    },
)
