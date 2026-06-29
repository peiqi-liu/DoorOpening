# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from DoorOpening.assets.door.multi_door_cfg import ALL_DOOR_CONFIGS
from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT
from DoorOpening.constants.door_constants import DOOR_BODY_NAMES, DOOR_JOINT_NAMES
from DoorOpening.constants.robot_constants import (
    CAMERA_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    ROBOT_KEY_BODY_NAMES,
    ROBOT_RESET_KEY_BODY_NAMES,
    ROBOT_PALM_LINK_NAME,
    ROBOT_BASE_BODY_LINK_NAME,
)
from DoorOpening.tasks.dooropening.contact_force_utils import (
    DOOR_BODY_CONTACT_FILTER_PRIM_PATHS,
    FLANGE_CONTACT_PRIM_PATH,
    FLANGE_SELF_COLLISION_FILTER_PRIM_PATHS,
    HANDLE_CONTACT_FILTER_PRIM_PATHS,
    SELF_COLLISION_CONTACT_PRIM_PATH,
    SELF_COLLISION_FILTER_PRIM_PATHS,
    X5_BODY_NAMES,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.mdp.events import (
    randomize_actuator_gains,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.common import ViewerCfg
import torch
import numpy as np
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.math import quat_from_euler_xyz

import isaaclab.sim as sim_utils

euler_angles = torch.tensor([-np.pi / 4, 0.0, 0])  # (roll, pitch, yaw) in radians
POINTCLOUD_CAMERA_QUAT = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2])
POINTCLOUD_CAMERA_QUAT = tuple(POINTCLOUD_CAMERA_QUAT.tolist())


def randomize_joint_effort_limits(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    effort_limit_distribution_params: tuple[float, float],
    operation: str = "abs",
    distribution: str = "uniform",
):
    """Randomize joint effort limits directly in PhysX for the selected articulation joints."""

    if operation != "abs":
        raise ValueError(f"randomize_joint_effort_limits only supports operation='abs', got {operation!r}.")
    if distribution != "uniform":
        raise ValueError(
            f"randomize_joint_effort_limits only supports distribution='uniform', got {distribution!r}."
        )

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if len(env_ids) == 0:
        return

    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    if isinstance(joint_ids, slice):
        current_limits = asset.data.joint_effort_limits[env_ids, joint_ids]
        num_joints = current_limits.shape[1]
    else:
        joint_ids = torch.as_tensor(joint_ids, device=asset.device, dtype=torch.long)
        num_joints = int(joint_ids.numel())

    low, high = float(effort_limit_distribution_params[0]), float(effort_limit_distribution_params[1])
    if high < low:
        raise ValueError(
            "randomize_joint_effort_limits requires effort_limit_distribution_params[0] <= "
            f"effort_limit_distribution_params[1], got {(low, high)}."
        )
    sampled_limits = low + (high - low) * torch.rand((len(env_ids), num_joints), device=asset.device)
    asset.write_joint_effort_limit_to_sim(sampled_limits, joint_ids=joint_ids, env_ids=env_ids)

@configclass
class EventCfg:
    """Configuration for reset-time physics randomization."""

    robot_physics_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.9, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    door_physics_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.9, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    door_board_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names="link_1"),
            # Control the door board mass directly in kilograms.
            "mass_distribution_params": (80.0, 80.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
        },
    )

    door_board_joint_stiffness_and_damping = EventTerm(
        func=randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_1"),
            "stiffness_distribution_params": (38.0, 38.0),
            "damping_distribution_params": (5.0, 5.0),
            # Use absolute values so the curriculum is expressed in physical gains, not multipliers of the
            # board actuator defaults (whose damping is 0.2).
            "operation": "abs",
        },
    )

    door_hinge_joint_stiffness_and_damping = EventTerm(
        func=randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_2"),
            "stiffness_distribution_params": (35.0, 35.0),
            "damping_distribution_params": (0.6, 0.6),
            "operation": "abs",
        },
    )

    door_hinge_joint_effort_limit = EventTerm(
        func=randomize_joint_effort_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_2"),
            "effort_limit_distribution_params": (1.0, 1.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

@configclass
class DooropeningEnvCfg(DirectRLEnvCfg):
    sim_dt = 1/60
    decimation = 2
    episode_length_s = 25.
    num_sim_steps_to_render=2
    # - spaces definition
    state_space = 0
    num_states = 0
    # Actor gets noisy deployment-style observations while the critic keeps the clean privileged state.
    asymmetric_obs = True

    viewer: ViewerCfg = ViewerCfg(eye=(1.5, -2.0, 1.0), lookat=(0.4, 0.0, 0.7), origin_type="env")
    door_handle_effort_limit_range_nm = (1.0, 4.0)
    door_handle_effort_limit_sim = door_handle_effort_limit_range_nm[0]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=num_sim_steps_to_render,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            solve_articulation_contact_last=True,
            min_position_iteration_count=4,
            max_position_iteration_count=64,
            min_velocity_iteration_count=2,
            max_velocity_iteration_count=16,
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15
        ),
    )

    # Useful constants

    base_link_name = "base_link"

    base_joints = [
        'base_rotation_joint',
        'base_x_joint',
        'base_y_joint',
    ]

    arm_joints = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
    ]

    finger_joints = [
        'finger_joint_0',
        'finger_joint_1',
        'finger_joint_2',
        'finger_joint_3',
        'finger_joint_4',
        'finger_joint_5',
        'finger_joint_6',
        'finger_joint_7',
        'finger_joint_8',
        'finger_joint_9',
        'finger_joint_10',
        'finger_joint_11',
        # 'finger_joint_12',
        # 'finger_joint_13',
        # 'finger_joint_14',
        # 'finger_joint_15',
    ]

    # finger_joints = [
    #     'finger_joint_1',
    #     'finger_joint_2',
    #     'finger_joint_3',
    #     'finger_joint_5',
    #     'finger_joint_6',
    #     'finger_joint_7',
    #     'finger_joint_9',
    #     'finger_joint_10',
    #     'finger_joint_11',
    # ]

    arx_joints = CAMERA_JOINT_NAMES[:4]

    contact_forces_door2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_2",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(HANDLE_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_base = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[0]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link1 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[1]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link2 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[2]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link3 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[3]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link4 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[4]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link5 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[5]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_camera = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[6]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    # Self-collision sensor: the fine-grained per-link self-collision set (franka arm +
    # LEAP hand + x5 arm) filtered against itself. force_matrix_w[n, b] aggregates the
    # self-contact on link b from every other (non-adjacent) link, which we
    # threshold-count into the self-collision penalty r_contact.
    contact_forces_self_collision = ContactSensorCfg(
        prim_path=SELF_COLLISION_CONTACT_PRIM_PATH,
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(SELF_COLLISION_FILTER_PRIM_PATHS),
    )
    # Flange self-collision sensor: catches a finger folding back onto the Franka flange
    # (panda_link7) while structurally excluding the permanent palm_lower<->flange contact
    # (the flange is filtered to the distal finger links only, never palm_lower).
    contact_forces_flange_self_collision = ContactSensorCfg(
        prim_path=FLANGE_CONTACT_PRIM_PATH,
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(FLANGE_SELF_COLLISION_FILTER_PRIM_PATHS),
    )
    handle_contact_force_threshold = 1.0
    x5_body_contact_force_threshold = 1.5
    # Self-collision penalty (r_contact): per step we count the self-collision links whose
    # net self-contact force exceeds this threshold, then scale by self_collision_penalty_w.
    self_collision_force_threshold = 1.0

    # Pointcloud render mode:
    # - "none": no on-robot pointcloud camera sensor (default).
    # - "depth": enable the x5 depth camera sensor and use its depth map.
    # - "lidar": no pointcloud camera sensor; render from the lidar body pose.
    pointcloud_render_mode = "none"
    pointcloud_lidar_body_name = "lidar"
    enable_pointcloud_camera = False
    pointcloud_camera_height = 480
    pointcloud_camera_width = 640
    pointcloud_camera_update_period = 0.1  # Depth camera refreshes at 10 Hz; replay frames still follow env dt.
    pointcloud_camera_data_types = ["distance_to_image_plane"]
    pointcloud_camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/x5_camera_link/cam",
        update_period=pointcloud_camera_update_period,
        update_latest_camera_pose=True,
        height=pointcloud_camera_height,
        width=pointcloud_camera_width,
        data_types=pointcloud_camera_data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=POINTCLOUD_CAMERA_QUAT, convention="world"),
    )

    # Raw `.pt` point-cloud replay dumps for teacher RL training on headless/HPC nodes.
    # This uses geometry samplers, not cameras or renderer video: one robot cloud and one door cloud per saved frame.
    viser_pointcloud = {
        "enabled": False,
        "path": "teacher_viser_replay.pt",
        "env_id": 80,
        "capture_interval": 3,
        "save_interval": 10_000,
        "max_points": 18_000,
        "robot_num_points": 15_000,
        "door_num_points": 3_000,
        "max_frames": 1000,
    }

    door_body_names = DOOR_BODY_NAMES

    door_base_frame_name = "base"

    door_joint_names = DOOR_JOINT_NAMES

    robot_key_bodies = ROBOT_KEY_BODY_NAMES
    robot_reset_key_bodies = ROBOT_RESET_KEY_BODY_NAMES

    robot_palm_link_name = ROBOT_PALM_LINK_NAME
    robot_base_body_link_name = ROBOT_BASE_BODY_LINK_NAME

    # robot(s)
    robot_cfg: ArticulationCfg = GLORBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # Keep both the Panda arm and the x5 camera arm in an explicit pose at spawn.
            joint_pos=DEFAULT_JOINT_POS,
            pos=ROBOT_INITIAL_POS,
            rot=ROBOT_INITIAL_ROT
        ),
    )

    twist_indices = [1, 5, 20]

    # door(s)
    door_cfg: ArticulationCfg = ALL_DOOR_CONFIGS.replace(prim_path="/World/envs/env_.*/Door")

    actuated_joints_num = len(arm_joints) + len(base_joints) + len(finger_joints) + len(arx_joints)
    # action_space = actuated_joints_num * 1
    action_space = len(arm_joints) + len(base_joints) + len(finger_joints)
    # Per twist index we concatenate:
    # - future robot key-body positions: `len(robot_key_bodies) * 3`
    # - future robot key-body 6D rotations: `len(robot_key_bodies) * 6`
    # - future door body position in the robot base frame: `3`
    # - future door joint positions: `len(door_joint_names)`
    # - future Panda arm joint deltas: `len(arm_joints)`
    # - future ARX joint deltas: `len(arx_joints)`
    # - future base joint deltas: `len(base_joints)`
    twist_observation_space = len(twist_indices) * (
        len(robot_key_bodies) * 3 +
        len(robot_key_bodies) * 6 +
        3 +
        len(door_joint_names) +
        len(arm_joints) +
        len(arx_joints) +
        len(base_joints)
    )

    # Observation layout:
    # - proprioception: current actuated joint positions + joint velocities + PD targets
    #   => `actuated_joints_num * 3`
    #   Adding the 4 ARX joints increases this block by `4 * 3 = 12` dims.
    # - current base and arm joint diffs to the reference motion
    #   => currently disabled in _build_observations()
    # - key-body position tracking error in the base frame
    #   => `len(robot_key_bodies) * 3`
    # - non-base key-body poses in the base frame:
    #   local position (3) + 6D rotation (6) for each key body except the base body itself
    #   => `(len(robot_key_bodies) - 1) * (3 + 6)`
    # - base linear/angular velocity in the base frame
    #   => `6`
    # - door body positions in the base frame
    #   => `len(door_body_names) * 3`
    # - current and reference door joint positions
    #   => `len(door_joint_names) * 2`
    # - reference ARX/x5 joint positions
    #   => `len(arx_joints)`
    # - future reference motion summary at twist indices
    #   => currently disabled in _build_observations(), so not counted in observation_space
    proprioception_observation_space = actuated_joints_num * 3
    joint_reference_error_observation_space = len(finger_joints)
    key_body_error_observation_space = len(robot_key_bodies) * 3
    robot_pose_observation_space = (len(robot_key_bodies) - 1) * (3 + 6)
    base_velocity_observation_space = 6
    door_body_observation_space = len(door_body_names) * 3
    door_joint_observation_space = len(door_joint_names) * 2
    arx_joint_reference_observation_space = len(arx_joints)

    observation_space = (
        proprioception_observation_space
        # + joint_reference_error_observation_space
        + key_body_error_observation_space
        + robot_pose_observation_space
        + base_velocity_observation_space
        + door_body_observation_space
        + door_joint_observation_space
        + arx_joint_reference_observation_space
    )
    state_space = observation_space
    num_observations = observation_space
    num_states = state_space
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=False)

    base_action_scale = 0.6
    arm_action_scale = 0.6
    finger_action_scale = 1.5
    arx_action_scale = 0.6

    # Deep Mimic Reward Parameters
    robot_body_quat_w = 1.0
    robot_key_body_pos_w = 2.0
    robot_base_joint_pos_w = 3.0
    robot_arm_joint_pos_w = 3.0
    robot_finger_joint_pos_w = 1.0
    robot_arx_joint_pos_w = 5.0
    robot_arx_tuck_joint_pos_w = 2.0
    robot_base_joint_vel_w = 1.0
    robot_arm_joint_vel_w = 2.0
    robot_finger_joint_vel_w = 0.5
    door_joint_pos_w = 4.0
    hinge_contact_reward_w = 1.0
    robot_body_lin_vel_w = 1.0
    robot_body_ang_vel_w = 0.5
    joint_limit_penalty_w = 40.0
    joint_limit_penalty_margin_ratio = 0.05
    # lambda_c in r = r_target - lambda_l*r_limit - lambda_c*r_contact
    self_collision_penalty_w = 1.0

    robot_body_quat_scale = 1.0
    robot_key_body_pos_scale = 3.0
    robot_base_joint_pos_scale = 0.5
    robot_arm_joint_pos_scale = 0.2
    robot_finger_joint_pos_scale = 1.0
    robot_arx_tuck_joint_pos_scale = 0.2
    robot_base_joint_vel_scale = 0.5
    robot_arm_joint_vel_scale = 0.5
    robot_finger_joint_vel_scale = 0.5
    door_joint_pos_scale = 5.0
    robot_body_lin_vel_scale = 10.0
    robot_body_ang_vel_scale = 10.0

    reset_key_body_pos_delta_min = 0.5
    reset_key_body_quat_delta_min = 1.5
    reset_key_body_pos_delta_max = 0.9
    reset_key_body_quat_delta_max = 3.0
    reset_door_joint_pos_delta_min = 0.5
    reset_door_joint_pos_delta_max = 0.8
    # We are slowly increasing our tolerance on base position drift and slowly only resettting the env from the first key frame
    # This variable is used to indicate when we stop increasing the tolerance and reset the env from the first key frame for the greatest probability
    reset_progress_total = 4e5
    use_motion_ref = True
    # ADR should ramp faster than the reference-motion reset curriculum so physics randomization is not lagging behind.
    adr_reset_progress_total = 1.5e5

    alive_base = 10.0
    alive_bonus = 20.0
    termination_penalty = -100.0

    # Keep DR opt-in so the default task is the clean baseline.
    enable_adr = True
    num_adr_increments = 20
    starting_adr_increments = 0
    dr_metrics_interval = 100
    log_verbose_dr_metrics = True

    events: EventCfg = EventCfg()

    # These are the ADR endpoints for simulator parameters handled by EventTerms at reset.
    # Door-board mass and door gains are specified in physical units, while robot gains use multipliers.
    # The door board starts at stiffness=100 and damping=10, and the hinge starts at stiffness=1 and damping=1.
    adr_cfg_dict = {
        "num_increments": num_adr_increments,
        "door_board_mass": {
            "mass_distribution_params": (60.0, 150.0),
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
        },
        "door_board_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (1.0, 75.0),
            "damping_distribution_params": (1.0, 10.0),
        },
        "door_hinge_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (10.0, 60.0),
            "damping_distribution_params": (0.03, 1.0),
        },
        "door_hinge_joint_effort_limit": {
            "effort_limit_distribution_params": door_handle_effort_limit_range_nm,
        },
    }

    # These terms are sampled inside the env because they perturb reset state, observations, and controller targets.
    adr_custom_cfg_dict = {
        "robot_spawn": {
            "base_xy_joint_pos_noise": (0.0, 0.1),
            "base_rot_joint_pos_noise": (0.0, 0.05),
            "arm_joint_pos_noise": (0.0, 0.15),
            "finger_joint_pos_noise": (0.0, 0.1),
        },
        "robot_state_noise": {
            "base_xy_joint_pos_noise": (0.0, 0.003),
            "base_xy_joint_pos_bias": (0.0, 0.002),
            "base_rot_joint_pos_noise": (0.0, 0.01),
            "base_rot_joint_pos_bias": (0.0, 0.006),
            "arm_joint_pos_noise": (0.0, 0.01),
            "arm_joint_pos_bias": (0.0, 0.006),
            "finger_joint_pos_noise": (0.0, 0.02),
            "finger_joint_pos_bias": (0.0, 0.01),
            "key_body_pos_noise": (0.0, 0.01),
            "key_body_pos_bias": (0.0, 0.005),
            "key_body_quat_noise": (0.0, 0.01),
            "key_body_quat_bias": (0.0, 0.005),
            "base_xy_joint_vel_noise": (0.0, 0.03),
            "base_xy_joint_vel_bias": (0.0, 0.015),
            "base_rot_joint_vel_noise": (0.0, 0.08),
            "base_rot_joint_vel_bias": (0.0, 0.04),
            "arm_joint_vel_noise": (0.0, 0.1),
            "arm_joint_vel_bias": (0.0, 0.05),
            "finger_joint_vel_noise": (0.0, 0.15),
            "finger_joint_vel_bias": (0.0, 0.08),
            "body_lin_vel_noise": (0.0, 0.05),
            "body_lin_vel_bias": (0.0, 0.03),
            "body_ang_vel_noise": (0.0, 0.1),
            "body_ang_vel_bias": (0.0, 0.05),
        },
        "door_state_noise": {
            "door_pos_noise": (0.0, 0.01),
            "door_pos_bias": (0.0, 0.005),
            "door_joint_pos_noise": (0.0, 0.01),
            "door_joint_pos_bias": (0.0, 0.005),
        },
        "pd_targets": {
            "base_xy_target_noise": (0.0, 0.0015),
            "base_rot_target_noise": (0.0, 0.005),
            "arm_target_noise": (0.0, 0.003),
            "finger_target_noise": (0.0, 0.005),
        },
        # Action latency (in env/control steps; env dt = sim_dt * decimation = 1/30 s).
        # The PD target applied on a given step is the one the policy produced `latency` steps
        # earlier. ADR ramps the upper bound from 1 step up to 4 steps (~4/30 s ≈ 133 ms); the
        # lower bound stays at 1, so latency is sampled uniformly in [1, current_max] per env at
        # each reset. Index [0] is the starting/minimum latency, index [1] the max at full ADR.
        "action_latency": {
            "latency_steps": (1, 3),
        },
    }

    # Change this to where you store your motions
    motion_file = "trajectory.pkl"
