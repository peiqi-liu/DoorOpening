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
    BASE_DOOR_CONTACT_BODY_NAMES,
    DOOR_BODY_CONTACT_FILTER_PRIM_PATHS,
    FRANKA_BOX_DOOR_CONTACT_PRIM_PATH,
    HANDLE_CONTACT_FILTER_PRIM_PATHS,
    PALM_ONLY_HANDLE_CONTACT_FILTER_PRIM_PATHS,
    PANEL_CONTACT_FILTER_PRIM_PATHS,
    SELF_COLLISION_FRANKA_FILTER_PRIM_PATHS,
    SELF_COLLISION_FRANKA_PRIM_PATH,
    SELF_COLLISION_HAND_FILTER_PRIM_PATHS,
    SELF_COLLISION_HAND_PRIM_PATH,
    X5_BODY_NAMES,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.mdp.events import (
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.envs.common import ViewerCfg
import torch
import numpy as np
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.math import quat_from_euler_xyz

import isaaclab.sim as sim_utils

euler_angles = torch.tensor([-np.pi / 4, 0.0, 0])  # (roll, pitch, yaw) in radians
POINTCLOUD_CAMERA_QUAT = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2])
POINTCLOUD_CAMERA_QUAT = tuple(POINTCLOUD_CAMERA_QUAT.tolist())


def _robot_body_contact_sensor(body_name: str, filter_prim_paths) -> ContactSensorCfg:
    """A single-body filtered contact sensor on a robot body.

    IsaacLab reports filtered contacts (``force_matrix_w``) correctly only when the sensor
    ``prim_path`` matches a SINGLE body per env (see ContactSensorCfg docs). So contact
    groups that used to be one multi-body sensor (base<->door, self-collision) are declared
    one sensor per body here and aggregated in the env. Field names must stay
    ``contact_forces_<group>_<body>`` -- the env builds the sensor keys from the same body-name
    lists, so a missing field fails loudly at scene setup.
    """
    return ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{body_name}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(filter_prim_paths),
    )


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


class randomize_body_material_subset(ManagerTermBase):
    """Randomize the physics material of a SUBSET of an articulation's bodies.

    The stock ``randomize_rigid_body_material`` cannot randomize a single body of these generated
    door assets: its per-body shape-count parse fails ("Expected total shapes: 8, but got: 7") on the
    convex-decomposition door colliders, because summing the per-link shape counts does not equal
    ``root_physx_view.max_shapes``. This term sidesteps that assertion by writing the flat material
    buffer directly over just the target body's shape slice. It reuses the SHARED ``root_physx_view``
    (no per-env view creation), so it does not reintroduce the host-RAM OOM the old custom per-body
    panel term caused at num_envs=4096.

    It must run AFTER a door-wide material term so the non-target shapes keep the door-wide sample and
    only the target body (the handle) is overwritten with the slipperier range.
    """

    def __init__(self, cfg: EventTerm, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]

        view = self.asset.root_physx_view
        link_paths = view.link_paths[0]
        self._total_shapes = view.max_shapes
        # Per-link shape counts (same source IsaacLab uses) but WITHOUT the strict sum==total assert
        # that crashes on this asset. For a trailing body we extend its slice to the buffer end so a
        # shape the per-link parse misses is still covered by the handle material.
        counts = [
            self.asset._physics_sim_view.create_rigid_body_view(link_path).max_shapes
            for link_path in link_paths
        ]
        body_ids = self.asset_cfg.body_ids
        if isinstance(body_ids, slice):
            body_ids = list(range(len(link_paths)))
        elif not isinstance(body_ids, (list, tuple)):
            body_ids = [int(b) for b in torch.as_tensor(body_ids).flatten().tolist()]

        self._shape_slices: list[tuple[int, int]] = []
        for body_id in body_ids:
            start = sum(counts[:body_id])
            end = self._total_shapes if body_id == len(link_paths) - 1 else start + counts[body_id]
            start = max(0, min(start, self._total_shapes))
            end = max(start, min(end, self._total_shapes))
            self._shape_slices.append((start, end))

        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))
        ranges = torch.tensor([static_friction_range, dynamic_friction_range, restitution_range], device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # Read the CURRENT buffer (already carries the door-wide sample) and overwrite only the
        # target body's shape slice, then write the whole buffer back.
        materials = self.asset.root_physx_view.get_material_properties()
        for start, end in self._shape_slices:
            width = end - start
            if width <= 0:
                continue
            bucket_ids = torch.randint(0, num_buckets, (len(env_ids), width), device="cpu")
            materials[env_ids, start:end] = self.material_buckets[bucket_ids]
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


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

    # Door-wide friction/restitution for the frame + PANEL (link_1). The panel keeps a broad range
    # so it still trains across slip<->jam. The handle (link_2) is deliberately re-materialized to a
    # much slipperier range by door_handle_physics_material BELOW (it runs after this term, so it
    # overrides link_2's material). Both use the stock num_buckets material mechanism (bounded by
    # num_buckets, not num_envs), so neither reintroduces the host-RAM OOM the old custom per-body
    # panel term caused at num_envs=4096.
    door_physics_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door"),
            "static_friction_range": (0.7, 2.5),
            "dynamic_friction_range": (0.7, 2.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    # Handle-only (link_2) friction. A real door handle is slippery metal, NOT like the panel: it
    # gets a much smaller friction range so the fingers cannot simply stick to it. Scoped to link_2
    # and defined AFTER door_physics_material so it overwrites the handle's material (event terms run
    # in definition order). Uses randomize_body_material_subset (not the stock term) because the
    # stock per-body shape-count parse crashes on this convex-decomposition door. Tune the range if
    # grasping the handle becomes too hard.
    door_handle_physics_material = EventTerm(
        func=randomize_body_material_subset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names="link_2"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
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

    # Per-episode randomization of the LEAP finger joint armature (reflected rotor inertia the
    # implicit PD sees). Lowered to a small armature (nominal 0.001) now that the finger actuator
    # also carries joint friction (0.01, see glorbot_cfg): friction damps the overshoot/jitter that
    # previously required a larger armature. Absolute values (not a scale). The ADR curriculum in
    # `adr_cfg_dict` widens (0.001, 0.001) -> (0.001, 0.005). To also randomize the arm armature
    # (currently fixed at 0), add an analogous term scoped to joint_names=["panda_joint.*",
    # "x5_joint.*"] with an endpoint like (0.0, 0.08).
    robot_finger_armature = EventTerm(
        func=randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint_.*"]),
            "armature_distribution_params": (0.002, 0.002),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    door_board_joint_stiffness_and_damping = EventTerm(
        func=randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_1"),
            # Start ~midpoint of the ADR endpoints (stiffness 1..125 -> 63, damping 1..16.7 -> ~8.8).
            "stiffness_distribution_params": (63.0, 63.0),
            "damping_distribution_params": (8.8, 8.8),
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
    sim_dt = 1/120
    decimation = 4
    episode_length_s = 36.
    num_sim_steps_to_render=4
    # - spaces definition
    state_space = 0
    num_states = 0
    # Actor gets noisy deployment-style observations while the critic keeps the clean privileged state.
    asymmetric_obs = True

    viewer: ViewerCfg = ViewerCfg(eye=(1.5, -2.0, 1.0), lookat=(0.4, 0.0, 0.7), origin_type="env")
    # Upper bound bumped (4.0 -> 6.0): the door mechanism should resist more than before (the panel
    # is stronger than expected).
    door_handle_effort_limit_range_nm = (1.0, 6.0)
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
    # PUSH-door palm-only handle contact sensor: only palm_center/palm_lower vs the handle (link_2).
    # Drives the push palm-handle reward (fingers excluded).
    contact_forces_door2_palm = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_2",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(PALM_ONLY_HANDLE_CONTACT_FILTER_PRIM_PATHS),
    )
    # Finger<->panel contact sensor: LEAP hand bodies against the door panel (Door/link_1).
    # Gated by panel_contact_mask, a force above threshold here is penalized (fingers should
    # grip the handle, not the panel) except where pushing the panel is the task.
    contact_forces_door_panel = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_1",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(PANEL_CONTACT_FILTER_PRIM_PATHS),
    )
    # Base<->door contact: every vertical base face + chassis against all door bodies. Any
    # contact here is penalized. Split one-sensor-per-body (see _robot_body_contact_sensor):
    # a multi-body sensor cannot report filtered contacts. The env aggregates these back into
    # a per-body [N, 5] tensor via BASE_DOOR_CONTACT_BODY_NAMES.
    contact_forces_base_door_left_panel = _robot_body_contact_sensor("left_panel", DOOR_BODY_CONTACT_FILTER_PRIM_PATHS)
    contact_forces_base_door_right_panel = _robot_body_contact_sensor("right_panel", DOOR_BODY_CONTACT_FILTER_PRIM_PATHS)
    contact_forces_base_door_tidybot2_base_link = _robot_body_contact_sensor("tidybot2_base_link", DOOR_BODY_CONTACT_FILTER_PRIM_PATHS)
    # Franka control box <-> door: folded into the harsh x5<->door penalty + termination.
    contact_forces_door_franka_box = ContactSensorCfg(
        prim_path=FRANKA_BOX_DOOR_CONTACT_PRIM_PATH,
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
    # Self-collision penalty (r_contact): THREE multi-body group sensors (franka arm / x5 arm /
    # base) instead of one filtered sensor per body. 15 single-body filtered sensors created 15
    # PhysX contact views (+ Warp kernels + buffers x num_envs) and OOM'd host RAM at 4096 envs.
    # Each group's force_matrix_w is [N, group_bodies, num_filters, 3]; the env reduces it to a
    # per-body force and counts bodies over threshold. Each group filters against the OTHER groups
    # + the fixed door frame (see contact_force_utils).
    # Only the moving franka arm needs self-collision sensing: the x5 arm and mobile base are fixed
    # relative to each other (can't self-collide), and their door-frame contacts are already caught
    # by the x5_door / base_door sensors. So a single franka sensor (filtered vs x5 + base + frame).
    contact_forces_self_collision_franka = ContactSensorCfg(
        prim_path=SELF_COLLISION_FRANKA_PRIM_PATH,
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(SELF_COLLISION_FRANKA_FILTER_PRIM_PATHS),
    )
    # Finger<->flange self-collision: LEAP digit links (fingers + thumb) filtered ONLY against the
    # franka panda_link7 flange. Counted with the same self_collision_penalty_w. Intra-hand
    # contacts are not in the filter set, so finger<->finger / finger<->thumb are NOT penalized.
    contact_forces_self_collision_hand = ContactSensorCfg(
        prim_path=SELF_COLLISION_HAND_PRIM_PATH,
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(SELF_COLLISION_HAND_FILTER_PRIM_PATHS),
    )
    handle_contact_force_threshold = 1.0
    # Finger<->door contact PROTECTION: finger<->panel (link_1) and finger<->handle (link_2) are
    # processed TOGETHER. When the fingers press EITHER door body harder than this (N), a strong
    # penalty (finger_door_contact_penalty_w) is applied -- but ONLY while the panel-contact mask is
    # on (push: open-door + base-forward; pull: retract-arm + push-panel + hold-traverse). Below the
    # threshold, or with the mask off, is free. Tune against stats/finger_door_contact_force_norm_max.
    finger_door_contact_force_threshold = 10.0
    # Contact between a non-front base face and any door body above this (N) is penalized.
    base_door_contact_force_threshold = 5.0
    x5_body_contact_force_threshold = 1.5
    # (Removed franka_box_contact_force_threshold: the franka control box no longer terminates the
    # episode. It is handled by the graded franka_box_contact_penalty_* params above instead.)
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
    # Reference joint-angle error (sim reading - reference) for base + arm(franka) + finger.
    joint_reference_error_observation_space = len(base_joints) + len(arm_joints) + len(finger_joints)
    key_body_error_observation_space = len(robot_key_bodies) * 3
    robot_pose_observation_space = (len(robot_key_bodies) - 1) * (3 + 6)
    base_velocity_observation_space = 6
    door_body_observation_space = len(door_body_names) * 3
    door_joint_observation_space = len(door_joint_names) * 2
    arx_joint_reference_observation_space = len(arx_joints)

    observation_space = (
        proprioception_observation_space
        # + joint_reference_error_observation_space  # DISABLED: joint-angle-diff obs commented out
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
    robot_finger_joint_pos_w = 1.5
    robot_arx_joint_pos_w = 5.0
    robot_arx_tuck_joint_pos_w = 2.0
    robot_base_joint_vel_w = 1.0
    robot_arm_joint_vel_w = 2.0
    robot_finger_joint_vel_w = 0.5
    door_joint_pos_w = 4.0
    # Palm-handle contact reward (push AND pull): +palm_handle_reward_w per step whenever palm_center/
    # palm_lower press the handle (> handle_contact_force_threshold) during the grasp->open window
    # (hinge_contact_mask, keyframes 2..5). Binary. Unified across modes -- finger<->hinge contact is
    # no longer rewarded; only the palm is.
    palm_handle_reward_w = 5.0
    robot_body_lin_vel_w = 1.0
    robot_body_ang_vel_w = 0.5
    joint_limit_penalty_w = 40.0
    joint_limit_penalty_margin_ratio = 0.05
    # lambda_c in r = r_target - lambda_l*r_limit - lambda_c*r_contact
    self_collision_penalty_w = 5.0
    # Penalty weight for the unified finger<->door protection penalty: applied while the panel-contact
    # mask is on and the finger<->panel OR finger<->handle force exceeds finger_door_contact_force_threshold.
    # Deliberately STRONG (this replaces the old weak graded panel/handle penalties). Tune it.
    finger_door_contact_penalty_w = 10.0
    # Penalty weight (per non-front base face in contact with the door). High on purpose: a
    # base panel hitting the door in the real world means a securely-mounted robot is injured.
    base_door_contact_penalty_w = 10.0
    # Penalty weight for the x5/arx camera arm contacting any door body (frame/panel/handle).
    # Harsh: the slender camera arm hitting the door is a serious real-world failure.
    x5_door_contact_penalty_w = 20.0
    # Franka control-box <-> door contact is NO LONGER episode-ending (the box is sturdy). Instead
    # a graded penalty ramps linearly from 0 penalty at min_force to the full weight at max_force
    # (clamped): <25 N is free, 25-75 N ramps up, >=75 N is the max penalty.
    # NOTE: franka_box_contact_penalty_w (the magnitude at >=75 N) is a starting guess -- tune it.
    franka_box_contact_penalty_w = 10.0
    franka_box_contact_penalty_min_force = 25.0
    franka_box_contact_penalty_max_force = 75.0

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
    # Task-based success: the robot must end up traversed to the FAR side of the door (measured as
    # distance the base has moved past the door plane along the approach axis, +x -> door -> far side).
    # Push refs traverse well past the door, so they need a real clearance; pull refs only end just
    # past (or at) the doorway, so they use a lenient threshold. Push vs pull is inferred from how far
    # the reference motion's final base moves past the door (> success_far_push_ref_dist => push).
    success_far_push_dist = 0.5
    success_far_pull_dist = 0.0
    success_far_push_ref_dist = 0.75
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
    # The door board starts at stiffness=63 and damping=8.8 (~midpoint of its ADR endpoints).
    adr_cfg_dict = {
        "num_increments": num_adr_increments,
        "door_board_mass": {
            "mass_distribution_params": (60.0, 150.0),
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
        },
        "robot_finger_armature": {
            # Widen from the nominal 0.001 toward a small physical band for the geared LEAP fingers.
            # Kept low -- joint friction (0.01) now handles jitter damping, so armature can stay near
            # the real reflected inertia; ceiling well under the ~0.03 stability limit.
            "armature_distribution_params": (0.001, 0.005),
        },
        "door_board_joint_stiffness_and_damping": {
            # Stiffer panel endpoint (was 75). Damping scaled proportionally (10 * 125/75 ~= 16.7)
            # to keep the same damping/stiffness ratio (~7.5) the curriculum was tuned around.
            "stiffness_distribution_params": (1.0, 125.0),
            "damping_distribution_params": (1.0, 16.7),
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
