# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from DoorOpening.assets.door.multi_door_cfg import ALL_DOOR_CONFIGS
from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG, GRIPPER_VELOCITY_LIMIT
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT
from DoorOpening.constants.door_constants import DOOR_BODY_NAMES, DOOR_JOINT_NAMES
from DoorOpening.constants.robot_constants import (
    CAMERA_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    DRIVEN_FINGER_JOINT_NAME,
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

# Camera mount orientation on x5_camera_link as (roll, pitch, yaw).
#   roll  = -45deg  -> compensates for the 45deg-tilted RealSense bracket on the ARX x5 wrist
#                      (rotation in the yz plane, about the link x-axis).
#   pitch = small   -> tilts the optical axis DOWN so the camera looks down a bit (rotation in the xz
#                      plane, about the link y-axis), keeping the handle inside the vertical FoV during
#                      approach. If a visual check shows it tilting UP instead of down, flip this sign.
CAMERA_MOUNT_ROLL_RAD = -np.pi / 4
# CAMERA_MOUNT_PITCH_RAD = float(np.deg2rad(10.0))  # look-down pitch, disabled for now
CAMERA_MOUNT_PITCH_RAD = float(np.deg2rad(0.0))
euler_angles = torch.tensor([CAMERA_MOUNT_ROLL_RAD, CAMERA_MOUNT_PITCH_RAD, 0.0])  # (roll, pitch, yaw) in radians
POINTCLOUD_CAMERA_QUAT = quat_from_euler_xyz(euler_angles[0], euler_angles[1], euler_angles[2])
POINTCLOUD_CAMERA_QUAT = tuple(POINTCLOUD_CAMERA_QUAT.tolist())


# NOTE on `history_length=0` (used by EVERY contact sensor in this file): it is a throughput knob,
# not a semantics change. SensorBase.update() takes an UNCONDITIONAL `_update_outdated_buffers()`
# branch whenever `history_length > 0`, which defeats IsaacLab's lazy sensor evaluation -- and that
# refresh opens with `_is_outdated.nonzero()`, a data-dependent shape, i.e. a blocking CUDA sync.
# DirectRLEnv.step() calls scene.update() INSIDE the decimation loop, so at history_length=1 every
# sensor did a full PhysX contact-matrix readback + a sync on all `decimation` substeps while only
# the last one is ever read. At 0, `force_matrix_w` / `net_forces_w` are still populated exactly the
# same way (ContactSensor just aliases the `*_history` buffers as an unsqueezed view) and the fetch
# happens lazily, once, when the reward code first touches `.data`. Raise this back above 0 only if
# something starts reading `net_forces_w_history` / `force_matrix_w_history` -- nothing does today.
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
        history_length=0,
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
            # Widened (was (0.2, 1.0)) after real-world observation that a metal lever can be very
            # slippery: floor dropped to 0.05 so the policy trains on fingers that barely grip, ceiling
            # raised to 1.2 so grippy handles are still covered. Real handle grip should be a subset.
            # Widened (was (0.2, 1.0)) after real-world observation that a metal lever can be very
            # slippery: floor 0.05 (fingers barely grip), ceiling 1.2 (grippy handles covered).
            # Widened (was (0.2, 1.0)): floor 0.05 (fingers barely grip), ceiling 1.2 (grippy handles).
            "static_friction_range": (0.05, 1.2),
            "dynamic_friction_range": (0.05, 1.2),
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

    # NOTE: there is deliberately NO gripper-armature randomization term. The Franka hand is a
    # single-actuator screw drive whose reflected inertia is a fixed property of the mechanism, and
    # the LEAP-era term that used to live here randomized it in ROTATIONAL units (kg-m^2) on what is
    # now a PRISMATIC DOF (where armature is added MASS in kg). The nominal 0.05 kg is set once, in
    # glorbot_cfg's GRIPPER_ARMATURE.

    # ADR-ramped panel-swing spring. Design intent: the panel is deliberately allowed to be very stiff,
    # and the door's perceived HEAVINESS is governed by the effort-limit cap
    # (door_panel_effort_limit_range_nm), not by stiffness. Restoring torque k*theta saturates at that
    # cap, so a stiff panel means a sharp breakaway that then holds a constant resistance in Nm, while
    # the low-stiffness end still gives a soft, freer-swinging door.
    #
    # Both gains are sampled LOG-uniformly in absolute physical units: the stiffness range spans 120x
    # (5..600 Nm/rad at full ADR), and uniform sampling would put almost every door at the stiff end
    # and almost never visit the soft regime.
    door_board_joint_stiffness_and_damping = EventTerm(
        func=randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_1"),
            # Start band 5..100 Nm/rad; ADR endpoint widens to 5..600. The FLOOR is now the same at
            # both ends, so genuinely free-swinging doors are in the mix from step 0 and ADR only
            # widens the stiff side. Lowered again for the 2-finger gripper (was 15..150).
            "stiffness_distribution_params": (5.0, 100.0),
            # Damping floor lowered 7 -> 3 alongside the stiffness floor. zeta = c/(2*sqrt(k*I)) with
            # I ~= 24 kg*m^2, so at k = 5 the old floor of 7 was already zeta ~= 0.32 and the top of
            # the range (30) was zeta ~= 1.4, i.e. the softest doors came out overdamped and sluggish
            # rather than freely swinging. 3 keeps the soft end light.
            "damping_distribution_params": (3.0, 30.0),
            # Absolute physical gains, not multipliers of the board actuator defaults.
            "operation": "abs",
            "distribution": "log_uniform",
        },
    )

    # Dry/Coulomb friction on the panel swing (joint_1) -- the "breakaway" resistance the door otherwise
    # lacks (default joint friction is 0), added after real doors were seen to be hard to crack open then
    # free once moving. This is a load-dependent COEFFICIENT, not a torque in Nm, so its effective drag
    # scales with the hinge's constraint load (an ~80 kg panel makes even small coefficients bite). Base
    # starts at 0 (no friction early); the ADR endpoint widens to (0..0.7). 0.5 reproduced the real
    # behavior, so 0.7 over-covers it -- if late-curriculum doors weld shut, lower the adr_cfg_dict
    # endpoint toward 0.5.
    door_board_joint_friction = EventTerm(
        func=randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names="joint_1"),
            # Coulomb breakaway friction: starts at 0, ADR endpoint widens to (0..0.7). Coefficient, not Nm.
            "friction_distribution_params": (0.0, 0.0),
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
    door_handle_effort_limit_range_nm = (1.0, 5.0)
    door_handle_effort_limit_sim = door_handle_effort_limit_range_nm[0]

    # Panel-swing (joint_1) effort-limit CAP applied while unlatched (edit_door_articulation switches it
    # to 1e6 while latched so the latch still holds). It caps the high-stiffness restoring torque so the
    # door plateaus at a constant "heaviness" in Nm instead of growing unbounded with opening angle. The
    # cap is sampled per-env at reset (uniformly, see _sample_door_panel_effort_limits) and ADR-ramped
    # from a narrow start band out to the full outer range, so heavy/light doors spread out only as the
    # curriculum advances.
    #
    # This is the PRIMARY door-difficulty knob: panel stiffness is sampled wide (see
    # door_board_joint_stiffness_and_damping) precisely so that the restoring torque saturates here,
    # making the door feel like a constant-torque load of this many Nm.
    #
    # Lowered again for the 2-finger gripper (start 10..25 -> 5..15, ADR endpoint 5..60 -> 3..40).
    # These Nm convert almost directly into the force the grasp has to transmit: with the handle
    # ~0.8 m from the hinge, required pull = cap / 0.8, so the start band drops from 12..31 N to
    # 6..19 N and the ADR ceiling from 75 N to 50 N. A pinch grasp can only pass 2*mu*F_grip
    # (~50 N of clamp on a 20 mm bar), so this moves the slip threshold from mu >= 0.31 down to
    # mu >= 0.19 -- i.e. from ~23% of the handle-friction draws slipping to ~12%.
    door_panel_effort_limit_start_range_nm = (5.0, 15.0)
    door_panel_effort_limit_range_nm = (3.0, 60.0)

    # Handle (joint_2) unlatch angle threshold (radians): the door stays latched until the handle is
    # rotated past this. Per-env, ADR-ramped from the fixed 0.8 start out to (0.65, 0.95) so the policy
    # must learn to fully turn handles that unlatch late. Read every step by edit_door_articulation.
    # Handle (joint_2) unlatch angle threshold (radians): per-env, ADR-ramped from the fixed 0.8 start
    # out to (0.75, 0.9) -- tightened from (0.65, 0.95). Read every step by edit_door_articulation.
    door_latch_threshold_start_range_rad = (0.8, 0.8)
    door_latch_threshold_range_rad = (0.75, 0.85)

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

    # The gripper is ONE commanded DOF, so the action space carries a single finger entry. The
    # follower (panda_finger_joint2) tracks it through the mimic coupling and must never get a
    # target of its own. NOTE this makes the action space 1-wide here where the LEAP hand was 12,
    # so policies/checkpoints trained on that hand are not loadable against this robot.
    finger_joints = [
        DRIVEN_FINGER_JOINT_NAME,
    ]


    arx_joints = CAMERA_JOINT_NAMES[:4]

    contact_forces_door2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_2",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(HANDLE_CONTACT_FILTER_PRIM_PATHS),
    )
    # PUSH-door hand-only handle contact sensor: only panda_hand vs the handle (link_2).
    # Drives the push palm-handle reward (fingers excluded).
    contact_forces_door2_palm = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_2",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(PALM_ONLY_HANDLE_CONTACT_FILTER_PRIM_PATHS),
    )
    # Finger<->panel contact sensor: the gripper fingers against the door panel (Door/link_1).
    # DIAGNOSTIC ONLY -- it feeds stats/finger_panel_contact_force_norm_* and the per-env
    # `finger_panel_contact_force_norm` the play/eval scripts print. It drives no reward term
    # (the old finger<->door contact penalty was removed along with the LEAP hand).
    contact_forces_door_panel = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Door/link_1",
        update_period=0.0,
        history_length=0,
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
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link2 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[2]}",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link3 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[3]}",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link4 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[4]}",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_link5 = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[5]}",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(DOOR_BODY_CONTACT_FILTER_PRIM_PATHS),
    )
    contact_forces_door_x5_camera = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{X5_BODY_NAMES[6]}",
        update_period=0.0,
        history_length=0,
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
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(SELF_COLLISION_FRANKA_FILTER_PRIM_PATHS),
    )
    # Finger<->flange self-collision: the two gripper fingers filtered ONLY against the
    # franka panda_link7 flange. Counted with the same self_collision_penalty_w. Intra-hand
    # contacts are not in the filter set, so finger<->finger contact is NOT penalized.
    contact_forces_self_collision_hand = ContactSensorCfg(
        prim_path=SELF_COLLISION_HAND_PRIM_PATH,
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(SELF_COLLISION_HAND_FILTER_PRIM_PATHS),
    )
    handle_contact_force_threshold = 1.0
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
    key_body_error_observation_space = len(robot_key_bodies) * 3
    robot_pose_observation_space = (len(robot_key_bodies) - 1) * (3 + 6)
    base_velocity_observation_space = 6
    door_body_observation_space = len(door_body_names) * 3
    door_joint_observation_space = len(door_joint_names) * 2
    arx_joint_reference_observation_space = len(arx_joints)

    observation_space = (
        proprioception_observation_space
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


    base_action_scale = 1.0
    arm_action_scale = 0.6
    # Actions are a delta on the measured joint angle (target = q + dt * scale * action, dt = 1/30 s),
    # so a scale is a commanded RATE. The gripper DOF is PRISMATIC, so unlike every other scale here this
    # one is a linear speed in METRES PER SECOND, not rad/s. Same convention as IsaacLab's own
    # franka_cabinet direct env, whose finger target rate is dof_speed_scale * action_scale =
    # 0.1 * 7.5 = 0.75 m/s (franka_cabinet_env.py:199/285).
    #
    # It is expressed as a multiple of GRIPPER_VELOCITY_LIMIT (0.05 m/s = the Franka Hand manual's
    # "Travel Speed (per finger) 50 mm/s", also the URDF <limit velocity>) because THAT is the real
    # ceiling: IsaacLab writes an implicit actuator's velocity_limit_sim into PhysX as the DOF max
    # velocity (articulation.py: _process_actuators_cfg -> write_joint_velocity_limit_to_sim ->
    # set_dof_max_velocities), so the joint physically cannot travel faster no matter what is
    # commanded. Raising the multiplier does not speed the finger up; it only lets the target lead
    # the state.
    #
    # 2x is deliberate headroom rather than an exact match:
    #  - the drive lags the target, so a 1.0x command settles slightly BELOW the velocity cap;
    #  - once the fingers are blocked by the handle, grasp force is stiffness * (target - actual),
    #    and since the target is re-referenced to q every step that error is exactly dt * scale --
    #    so this multiplier sets the clamp force outright (2x -> 5e3 * 0.1/30 = 16.7 N);
    #  - it leaves room if GRIPPER_VELOCITY_LIMIT is ever raised toward the 0.2 m/s the official
    #    franka_description URDF and IsaacLab's stock Franka asset use.
    # The cost is that |action| > 0.5 all produces the same (velocity-capped) motion, so push this
    # much higher only if you want a bang-bang gripper. For a faster gripper, raise
    # GRIPPER_VELOCITY_LIMIT in glorbot_cfg -- this scale follows it automatically.
    gripper_action_speed_headroom = 2.0
    finger_action_scale = gripper_action_speed_headroom * GRIPPER_VELOCITY_LIMIT
    arx_action_scale = 0.6

    # Deep Mimic Reward Parameters
    robot_body_quat_w = 1.0
    robot_key_body_pos_w = 2.0
    robot_base_joint_pos_w = 3.0
    robot_arm_joint_pos_w = 3.0
    # Gripper-opening tracking (the single driven finger DOF). There is deliberately no matching
    # VELOCITY term: the gripper is a 1-DOF open/close command whose speed is already capped at the
    # hardware limit, so tracking the reference's finger velocity added a term that was numerically
    # constant and shaped nothing.
    robot_gripper_joint_pos_w = 1.5
    robot_arx_joint_pos_w = 5.0
    robot_arx_tuck_joint_pos_w = 2.0
    robot_base_joint_vel_w = 1.0
    robot_arm_joint_vel_w = 2.0
    door_joint_pos_w = 4.0
    # PULL-only binary bonus for the GRIPPER (panda_hand + the two fingers, see
    # HANDLE_CONTACT_FILTER_PRIM_PATHS) touching the handle during the grasp/pull window.
    hinge_gripper_contact_reward_w = 1.5
    # PUSH-only palm-handle contact reward: +palm_handle_reward_w per step whenever palm_center/
    # panda_hand presses the handle (> handle_contact_force_threshold) during the grasp->push-open window
    # (hinge_contact_mask, keyframes 2..5). Binary. For PULL this is inactive (is_push gate = 0), and
    # for PUSH the finger-inclusive hinge_gripper_contact reward is disabled -- fingers on the handle
    # are no longer rewarded on push.
    palm_handle_reward_w = 5.0
    robot_body_lin_vel_w = 1.0
    robot_body_ang_vel_w = 0.5
    # Joint-limit penalty, applied to the FRANKA ARM JOINTS ONLY (see joint_limit_penalty_joints).
    joint_limit_penalty_w = 40.0
    joint_limit_penalty_margin_ratio = 0.05
    # Which DOFs the joint-limit penalty is scored over. Everything else is excluded on purpose:
    #  - base x/y/rotation are virtual chassis DOFs whose limits are the arena, not the hardware;
    #  - the arx/x5 joints are pinned to a fixed tuck pose and never driven by the policy;
    #  - the gripper's limits ARE its operating points. Fully open (0.04) and fully closed (0.0) are
    #    exactly the URDF limits, so including it charged a constant penalty for simply holding the
    #    reference gripper pose and punished every real grasp squeeze.
    joint_limit_penalty_joints = list(arm_joints)
    # lambda_c in r = r_target - lambda_l*r_limit - lambda_c*r_contact.
    # Raised 5 -> 25, i.e. above even x5_door_contact_penalty_w, because franka<->arx interpenetration
    # was visibly happening and being tolerated. This is charged PER COLLIDING LINK per step, so a
    # 2-link franka/camera-arm contact now costs 50/step against an alive reward of 10 (30 during
    # grasp/pull) -- self-collision is meant to be strictly worse than making no progress.
    #
    # PAIRED WITH the sensing fix in contact_force_utils (panda_link7/panda_hand/fingers vs the x5
    # arm were previously in NO sensor, so any weight here was zero for exactly the collision this
    # is aimed at) and with the reference-motion changes in offline_pull_door.py. Order matters: a
    # harsh weight on a reference trajectory that itself passes through collision is unlearnable --
    # the policy gets punished for tracking the demo. Verify stats/self_collision_body_count_mean is
    # ~0 on a reference replay before trusting a training run at this weight.
    self_collision_penalty_w = 25.0
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
    # Gripper tracking sharpness for exp(-scale * err^2). The error is a PRISMATIC opening in
    # METRES, so the LEAP-era 1.0 (tuned for radians over a ~1.5 rad finger sweep) made the term
    # numerically constant: the worst possible error is the 0.04 m stroke, i.e. err^2 = 1.6e-3, and
    # exp(-1.6e-3) = 0.998 for every state. Scale is therefore derived from the stroke instead of
    # hand-picked: 4 / stroke^2 puts a half-stroke error (0.02 m) at exp(-1) = 0.37.
    gripper_stroke_m = 0.04
    robot_gripper_joint_pos_scale = 4.0 / (gripper_stroke_m ** 2)
    robot_arx_tuck_joint_pos_scale = 0.2
    robot_base_joint_vel_scale = 0.5
    robot_arm_joint_vel_scale = 0.5
    door_joint_pos_scale = 5.0
    robot_body_lin_vel_scale = 10.0
    robot_body_ang_vel_scale = 10.0

    # Curriculum drift-termination thresholds (larger = looser = less likely to reset on drift). The
    # `_min` end is the EARLY-exploration setting (curriculum progress 0); the POSITION mins were
    # relaxed (key-body pos 0.5->0.7, door 0.5->0.7) so the policy is not killed so aggressively while
    # exploring, but the quat min was left at 1.8. The `_max` end (progress 1) is what distillation
    # pins to directly.
    reset_key_body_pos_delta_min = 0.7
    reset_key_body_quat_delta_min = 1.8
    reset_key_body_pos_delta_max = 0.9
    reset_key_body_quat_delta_max = 3.0
    reset_door_joint_pos_delta_min = 0.7
    reset_door_joint_pos_delta_max = 0.8
    # Task-based success: the robot must end up traversed to the FAR side of the door (measured as
    # distance the base has moved past the door plane along the approach axis, +x -> door -> far side).
    # Push refs traverse well past the door, so they need a real clearance; pull refs only end just
    # past (or at) the doorway, so they use a lenient threshold. Push vs pull is inferred from how far
    # the reference motion's final base moves past the door (> success_far_push_ref_dist => push).
    success_far_push_dist = 0.5
    success_far_pull_dist = 0.0
    success_far_push_ref_dist = 0.75

    reset_progress_total = 2.5e5
    use_motion_ref = True
    adr_reset_progress_total = 1e5

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
        "door_board_joint_stiffness_and_damping": {
            # FINAL (full-ADR) stiffness band, log-uniform over 5..400 Nm/rad. Ceiling cut 600 -> 400
            # alongside the effort-cap reduction: the cap is what the door's steady heaviness actually
            # is, and stiffness only decides how ABRUPTLY the restoring torque reaches it. The panel
            # saturates the cap at theta = cap/k, so with the ADR cap now 40 Nm, k = 600 saturated
            # after 0.067 rad (3.8 deg) -- an almost instantaneous wall -- while k = 400 gives 0.1 rad
            # (5.7 deg) and k = 5 stays soft over the whole swing. Keeps a sharp breakaway in the
            # distribution without the very hardest hit, which is what tears a pinch grasp loose.
            "stiffness_distribution_params": (5.0, 600.0),
            # Nm*s/rad. zeta = c/(2*sqrt(k*I)), I ~= 24 kg*m^2. Floor lowered 4 -> 2 to match the start
            # band's 3 (otherwise ADR would RAISE the damping floor as it progressed) and to keep the
            # k = 5 doors genuinely free-swinging rather than overdamped.
            "damping_distribution_params": (2.0, 60.0),
        },
        "door_board_joint_friction": {
            # Coulomb breakaway friction on the panel swing. Ramps from the (0, 0) EventTerm base out to
            # (0, 0.7); 0.5 reproduced real, so 0.7 over-covers it. Coefficient (load-dependent), not Nm.
            # Coulomb breakaway friction endpoint; 0.5 reproduced real, 0.7 over-covers.
            "friction_distribution_params": (0.0, 0.7),
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
    #
    # NOTE: the gripper DOF is deliberately absent from EVERY group here. All four LEAP-era finger
    # entries were in radians against a ~1.5 rad finger sweep, which on the 0.04 m prismatic gripper
    # meant spawn noise of 2.5x the whole stroke, observation noise of half the stroke, and velocity
    # noise 3x the joint's own speed limit -- i.e. randomization far larger than the signal. The real
    # hand is a single actuator with an accurate width encoder, so it now carries no reset,
    # observation, or PD-target noise at all.
    adr_custom_cfg_dict = {
        "robot_spawn": {
            "base_xy_joint_pos_noise": (0.0, 0.1),
            "base_rot_joint_pos_noise": (0.0, 0.05),
            "arm_joint_pos_noise": (0.0, 0.15),
        },
        # MATCHED to the sibling config: a flat additive noise with the SAME magnitude on every
        # observation entry regardless of its units, because that is literally what their
        # noise_lambda does (one torch.randn_like(obs_buf) * 0.002 over the whole concatenated,
        # unnormalized vector -- radians on joints, metres on positions, unitless on 6D rotations).
        #
        # UNITS OF THE NUMBERS BELOW ARE NOT THE NOISE MAGNITUDE. This env samples
        #     width ~ U(0, value)  per episode,   noise ~ U(-width, +width)  per step
        # so the effective standard deviations are value/3 (white) and value/6 (bias), NOT value.
        # Their targets are Gaussian std 0.002 white and 0.001 correlated, so:
        #     0.006 / 3 = 0.002   and   0.006 / 6 = 0.001   -> 0.006 everywhere.
        # Note the original 0.006 bias here was ALREADY an exact match; only the white term moved.
        #
        # Two remaining differences, deliberate: this stays UNIFORM (hard-bounded, no tails) where
        # theirs is Gaussian, and the width resamples per episode where theirs redraws the
        # correlated term every 720 steps.
        "robot_state_noise": {
            # BASE POSITION IS NOT AN ENCODER. base_x/base_y are virtual DOFs standing in for where
            # the chassis is in the world, which on hardware comes from wheel odometry or lidar SLAM.
            # That signal is SMOOTH short-term and DRIFTS long-term, so its error is mostly bias, not
            # white noise -- the flat 0.006 gave it 2 mm of per-step jitter, which odometry does not
            # have. White cut to 0.0015 (std 0.5 mm); the bias term is left at full size.
            #
            # This matters here specifically because base kp is 80000: force noise = kp * position
            # noise, so 2 mm of white noise became 160 N of random force on a 56.5 kg chassis.
            "base_xy_joint_pos_noise": (0.006, 0.006),
            "base_xy_joint_pos_bias": (0.006, 0.006),
            # Same reasoning; base yaw comes from odometry/IMU fusion, not a joint encoder.
            # Reduced 3x alongside base_xy. Yaw also comes from odometry/IMU fusion rather than a
            # joint encoder, and its kp is 11000, so it amplifies command-path noise too -- just less
            # than translation's 80000. The BIAS stays at full: a per-episode offset with randomized
            # magnitude and sign is useful randomization, and it is the white term that drives the
            # per-step random walk.
            "base_rot_joint_pos_noise": (0.006, 0.006),
            "base_rot_joint_pos_bias": (0.006, 0.006),
            # Reduced 3x alongside the base, same reason: this vector also feeds the COMMAND path
            # (_measured_joint_pos), so its white term drives an arm random walk as well as an
            # observation error. 0.006 -> std 2.0 mrad, which is ~20x a real Franka joint encoder
            # (the arm's +/-0.1 mm pose repeatability back-solves to ~0.1-0.2 mrad). 0.002 -> std
            # 0.67 mrad is still generous for that hardware. The BIAS stays at full.
            "arm_joint_pos_noise": (0.006, 0.006),
            "arm_joint_pos_bias": (0.006, 0.006),
            "key_body_pos_noise": (0.006, 0.006),
            "key_body_pos_bias": (0.006, 0.006),
            "key_body_quat_noise": (0.006, 0.006),
            "key_body_quat_bias": (0.006, 0.006),
            "base_xy_joint_vel_noise": (0.006, 0.006),
            "base_xy_joint_vel_bias": (0.006, 0.006),
            "base_rot_joint_vel_noise": (0.006, 0.006),
            "base_rot_joint_vel_bias": (0.006, 0.006),
            "arm_joint_vel_noise": (0.006, 0.006),
            "arm_joint_vel_bias": (0.006, 0.006),
            "body_lin_vel_noise": (0.006, 0.006),
            "body_lin_vel_bias": (0.006, 0.006),
            "body_ang_vel_noise": (0.006, 0.006),
            "body_ang_vel_bias": (0.006, 0.006),
            # ---- COMMAND PATH ONLY (see env._measured_joint_pos) ----------------------------
            # The joint reading the DELTA IS ADDED TO, deliberately far cleaner than what the policy
            # observes. These are different physical quantities, not one signal at two settings:
            #
            #  * the observation noise above bundles calibration, latency and estimation error --
            #    everything that makes the policy's picture of the world wrong. The policy should be
            #    robust to all of it.
            #  * the command path sees the RAW ENCODER, which on a Franka is ~1e-4 rad (its +/-0.1 mm
            #    pose repeatability back-solves to ~0.1-0.2 mrad). Nothing about calibration error
            #    enters here, because the controller adds its delta to whatever the encoder says.
            #
            # And the BIAS is 0 on purpose, not as a shortcut: on hardware a constant offset appears
            # in BOTH the commanded target and the drive's own error term and cancels exactly. Only
            # sim keeps it, because PhysX's PD reads ground truth. Leaving it in models a
            # disturbance the real robot cannot have -- it is the term that produced steady drift.
            #
            # This matters because target = q_measured + delta re-references to the robot every step,
            # so any white noise here is not bounded jitter but a RANDOM WALK, and the base amplifies
            # it by kp = 80000.
            # ARM: bias is SMALL but nonzero. The theory says it should cancel -- the Franka is a
            # single-encoder loop, you read robot_state.q to build the target and the 1 kHz internal
            # controller closes on robot_state.q, so a constant offset appears in both terms and
            # subtracts out. But that rests on an assumption about what FCI does internally that has
            # not been verified against the hardware, so this is a hedge rather than a model.
            #
            # Kept well under the base's: std 3.3e-4 rad is ~3x a real Franka encoder, and costs
            # a = b/(dt*scale) = 1.7% of the action range to cancel (the arm costs MORE per unit
            # bias than the base, 50x vs 30x, because arm_action_scale is 0.6 not 1.0).
            # Raise it if FCI turns out not to cancel; drop to 0.0 if you confirm that it does.
            "command_arm_joint_pos_noise": (0.0005, 0.0005),
            "command_arm_joint_pos_bias": (0.002, 0.002),
            # BASE: bias is small but NONZERO, because the base is not one sensor. If the policy's
            # chassis pose comes from SLAM while the base controller closes on wheel odometry, those
            # are two different estimates and the offset between them does NOT cancel.
            # Raised to the same magnitude as the OBSERVATION bias (0.006 -> std 1 mm), because a
            # constant offset is a fundamentally different animal from white noise:
            #   * white noise re-references every step, so it is an unrejectable RANDOM WALK;
            #   * a bias is a constant DRIFT the policy cancels with a constant action offset of
            #     a = b / (dt * scale). Note kp/kd cancels out of that -- the cost does not depend on
            #     the gains -- and it works out to just 3% of the [-1, 1] action range here. The base
            #     position error is directly observable, so even a memoryless MLP can hold it.
            # So this costs authority, not stability. Beyond ~0.012 (6%) it starts to eat into the
            # action range meaningfully; 0.030 would take 15% and is not worth it.
            # Set to 0.0 if your deploy stack reads base pose from the same source it controls on.
            "command_base_joint_pos_noise": (0.0005, 0.0005),
            "command_base_joint_pos_bias": (0.006, 0.006),
        },
        "door_state_noise": {
            "door_pos_noise": (0.006, 0.006),
            "door_pos_bias": (0.006, 0.006),
            "door_joint_pos_noise": (0.006, 0.006),
            "door_joint_pos_bias": (0.006, 0.006),
        },
        # Noise on the ACTION, replacing the old per-DOF PD-target noise.
        #
        # Placement matters more than magnitude here. Perturbing the PD target (or the measured
        # angle the target is built from) injects an error that sim's PD sees but hardware's does
        # not -- on the real robot the controller and the PD read the SAME encoder, so that error
        # cancels. Perturbing the action instead models command/actuation error, which is a real
        # channel and survives the cancellation. This is what the sibling config does.
        #
        # Applied to the RAW policy output before clamping and scaling, so the units are fractions
        # of the [-1, 1] action range: 0.05 white per step + 0.015 constant per episode, matching
        # the sibling's action noise.
        "action_noise": {
            # Same width-then-uniform pipeline as above, so effective std is value/3 and value/6.
            # Their action noise is Gaussian std 0.05 white + 0.015 correlated on the normalized
            # [-1, 1] action, so: 0.15/3 = 0.05 and 0.09/6 = 0.015.
            "action_noise": (0.15, 0.15),
            "action_bias": (0.09, 0.09),
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
