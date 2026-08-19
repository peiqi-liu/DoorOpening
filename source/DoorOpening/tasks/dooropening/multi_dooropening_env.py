import inspect
import math
import os
import torch
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from DoorOpening.utils.quat_utils import quat_diff_angle, hinge_angle_diff
from DoorOpening.assets.door.multi_door_cfg import (
    DOOR_FAMILY_NAMES,
    asset_paths as door_asset_paths,
    asset_family_ids,
    board_offsets,
    closed_handle_offsets_base,
    configure_multi_door_assets_for_rank,
    edit_door_articulation,
    get_multi_door_asset_start_index,
    get_multi_door_env_asset_indices,
    handle_offsets,
    motion_traj_paths,
)
from DoorOpening.tasks.dooropening.dooropening_adr import DoorOpeningADR
from DoorOpening.tasks.dooropening.multi_dooropening_env_cfg import DooropeningEnvCfg
from DoorOpening.assets.glorbot.glorbot_cfg import glorbot_urdf_path, disable_collision_scope_instancing
from isaaclab.sensors import Camera, ContactSensor
from DoorOpening.constants.robot_constants import CAMERA_JOINT_DEFAULT_VALUES, CAMERA_JOINT_NAMES, FULL_JOINT_NAMES, ROBOT_KEY_BODY_NAMES
from DoorOpening.constants.env_constants import DOOR_INITIAL_POS, ROBOT_INITIAL_POS
from DoorOpening.tasks.dooropening.contact_force_utils import (
    BASE_DOOR_CONTACT_BODY_NAMES,
    DOOR_FRAME_FILTER_INDEX,
    SELF_COLLISION_X5_BODIES,
    get_filtered_contact_force_w,
    get_self_contact_body_force_norm,
)
from DoorOpening.utils.pose_utils import world_to_local
from isaaclab.utils.math import quat_conjugate, quat_apply, quat_mul
from DoorOpening.utils.quat_utils import quat_to_6d
from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler
from DoorOpening.utils.viser_pt import format_iterated_record_path, prepare_pointcloud
from typing import Tuple


# Contact-sensor keys. base_door is still one single-body sensor per base face (each filtered
# against the door). self_collision is now a SINGLE multi-body sensor on the franka arm: the x5
# camera arm and the mobile base are fixed relative to each other so they cannot self-collide, so
# only the moving franka arm needs self-collision checks (filtered against x5 + base + door frame).
# x5/base <-> door-frame contacts are already covered by the dedicated x5_door / base_door sensors.
# Consumers threshold-count over the per-body force_matrix, so order is not load-bearing. Keys must
# match the cfg field names; a mismatch raises AttributeError at setup.
BASE_DOOR_SENSOR_NAMES = tuple(f"contact_forces_base_door_{b}" for b in BASE_DOOR_CONTACT_BODY_NAMES)
SELF_COLLISION_SENSOR_NAMES = (
    "contact_forces_self_collision_franka",
    # Finger<->flange: the gripper fingers vs panda_link7 (counted at self_collision_penalty_w).
    "contact_forces_self_collision_hand",
)


class DooropeningEnv(DirectRLEnv):
    cfg: DooropeningEnvCfg

    def __init__(self, cfg: DooropeningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._initialize_runtime_event_terms()
        self.early_stopping = True

        self.num_base_joints = len(self.cfg.base_joints)
        self.num_arm_joints = len(self.cfg.arm_joints)
        self.num_finger_joints = len(self.cfg.finger_joints)
        self.num_arx_joints = len(self.cfg.arx_joints)

        actuated_joints = self.cfg.base_joints + self.cfg.arm_joints + self.cfg.finger_joints + self.cfg.arx_joints
        self._robot_dof_idx, joint_names = self.robot.find_joints(actuated_joints, preserve_order=True)
        self._robot_dof_idx = torch.tensor(self._robot_dof_idx, device=self.device)
        self.ref_robot_dof_idx = torch.tensor([FULL_JOINT_NAMES.index(name) for name in joint_names], device=self.device)

        self._robot_key_body_idx, robot_key_body_names = self.robot.find_bodies(self.cfg.robot_key_bodies)
        self._robot_reset_key_body_idx, robot_reset_key_body_names = self.robot.find_bodies(self.cfg.robot_reset_key_bodies)
        # Kept for offline drift diagnosis (test_teacher_diagnose.py): the names behind the per-body
        # position/orientation errors that drive the tracking-drift termination in _get_dones.
        self._robot_reset_key_body_names = list(robot_reset_key_body_names)
        self._robot_key_body_names = list(robot_key_body_names)
        self._robot_base_id_in_key_body_idx = robot_key_body_names.index(self.cfg.robot_base_body_link_name)
        self._robot_palm_id_in_key_body_idx = robot_key_body_names.index(self.cfg.robot_palm_link_name)

        self.ref_key_body_idx = torch.tensor(
            [ROBOT_KEY_BODY_NAMES.index(name) for name in robot_key_body_names],
            device=self.device,
            dtype=torch.long,
        )
        self.ref_reset_key_body_idx = torch.tensor(
            [ROBOT_KEY_BODY_NAMES.index(name) for name in robot_reset_key_body_names],
            device=self.device,
            dtype=torch.long,
        )

        self._robot_base_dof_idx, base_joint_names = self.robot.find_joints(self.cfg.base_joints, preserve_order=True)
        self._robot_arm_dof_idx, arm_joint_names = self.robot.find_joints(self.cfg.arm_joints, preserve_order=True)
        self._robot_finger_dof_idx, finger_joint_names = self.robot.find_joints(self.cfg.finger_joints, preserve_order=True)
        self._robot_arx_dof_idx, arx_joint_names = self.robot.find_joints(self.cfg.arx_joints, preserve_order=True)
        self._robot_base_dof_idx = torch.tensor(self._robot_base_dof_idx, device=self.device)
        self._robot_arm_dof_idx = torch.tensor(self._robot_arm_dof_idx, device=self.device)
        self._robot_finger_dof_idx = torch.tensor(self._robot_finger_dof_idx, device=self.device)
        self._robot_arx_dof_idx = torch.tensor(self._robot_arx_dof_idx, device=self.device)
        base_joint_name_to_local_idx = {name: idx for idx, name in enumerate(base_joint_names)}
        self._robot_base_rot_dof_idx = torch.tensor(
            [self._robot_base_dof_idx[base_joint_name_to_local_idx["base_rotation_joint"]]], device=self.device
        )
        self._robot_base_xy_dof_idx = torch.tensor(
            [
                self._robot_base_dof_idx[base_joint_name_to_local_idx["base_x_joint"]],
                self._robot_base_dof_idx[base_joint_name_to_local_idx["base_y_joint"]],
            ],
            device=self.device,
        )
        self._robot_base_rot_local_idx = torch.tensor(
            [base_joint_name_to_local_idx["base_rotation_joint"]], device=self.device
        )
        self._robot_base_xy_local_idx = torch.tensor(
            [base_joint_name_to_local_idx["base_x_joint"], base_joint_name_to_local_idx["base_y_joint"]],
            device=self.device,
        )

        self.ref_base_joint_idx = [FULL_JOINT_NAMES.index(name) for name in base_joint_names]
        self.ref_arm_joint_idx = [FULL_JOINT_NAMES.index(name) for name in arm_joint_names]
        self.ref_finger_joint_idx = [FULL_JOINT_NAMES.index(name) for name in finger_joint_names]
        self.ref_arx_joint_idx = [FULL_JOINT_NAMES.index(name) for name in arx_joint_names]

        # Camera (x5) joints that are NOT in the policy-tracked arx set (arx_joints covers only the
        # first few x5 joints). These are never actuated or observed, so the articulation only
        # PD-holds them compliantly -> they sag/oscillate under base motion and contact, swinging
        # the wrist-mounted camera. We hold them rigidly at their default pose every step (see
        # _enforce_fixed_camera_joint_state). They stay fully OUT of the action/observation
        # accounting, so action_space, observation_space and teacher/student dims are unchanged.
        extra_camera_joint_names = [name for name in CAMERA_JOINT_NAMES if name not in self.cfg.arx_joints]
        self.num_extra_camera_joints = len(extra_camera_joint_names)
        if self.num_extra_camera_joints > 0:
            extra_camera_idx, extra_camera_resolved_names = self.robot.find_joints(extra_camera_joint_names)
            self._robot_extra_camera_dof_idx = torch.tensor(extra_camera_idx, device=self.device)
            self._robot_extra_camera_default_pos = torch.tensor(
                [float(CAMERA_JOINT_DEFAULT_VALUES[name]) for name in extra_camera_resolved_names],
                device=self.device,
            )
        else:
            self._robot_extra_camera_dof_idx = torch.empty(0, dtype=torch.long, device=self.device)
            self._robot_extra_camera_default_pos = torch.empty(0, device=self.device)

        self._robot_base_link_idx, self.robot_base_link_name = self.robot.find_bodies(self.cfg.base_link_name)
        self._door_body_idx, self.door_body_names = self.door.find_bodies(self.cfg.door_body_names)
        self._door_base_link_idx, self.door_base_link_name = self.door.find_bodies(self.cfg.door_base_frame_name)
        self._door_base_link_idx = self._door_base_link_idx[0]
        self._door_joint_idx, self.door_joint_names = self.door.find_joints(self.cfg.door_joint_names)
        door_joint_name_to_idx = {name: idx for idx, name in enumerate(self.door_joint_names)}
        self._door_board_joint_idx = int(self._door_joint_idx[door_joint_name_to_idx["joint_1"]])
        self._door_hinge_joint_idx = int(self._door_joint_idx[door_joint_name_to_idx["joint_2"]])

        self._robot_base_body_link_idx, self.robot_base_body_link_name = self.robot.find_bodies(self.cfg.robot_base_body_link_name)
        self._robot_base_body_link_idx = self._robot_base_body_link_idx[0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # This is the actual environment/control step and the rate replays should follow.
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 1].to(device=self.device)

        # We are going to update this variables to control the robot
        self.robot_dof_targets = torch.zeros((self.num_envs, len(self._robot_dof_idx)), device=self.device)
        self.applied_robot_dof_targets = torch.zeros_like(self.robot_dof_targets)
        self._target_base_rot_slice = slice(0, 1)
        self._target_base_xy_slice = slice(1, self.num_base_joints)
        self._target_arm_slice = slice(self.num_base_joints, self.num_base_joints + self.num_arm_joints)
        self._target_finger_slice = slice(self._target_arm_slice.stop, self._target_arm_slice.stop + self.num_finger_joints)
        self._target_arx_slice = slice(self._target_finger_slice.stop, self.robot_dof_targets.shape[1])
        self.num_policy_actions = int(self.cfg.action_space)
        self.num_robot_actions = int(self.robot_dof_targets.shape[1])
        expected_policy_actions = self.num_base_joints + self.num_arm_joints + self.num_finger_joints
        if self.num_policy_actions != expected_policy_actions:
            raise ValueError(
                "Unexpected policy action dim for multi-door env. "
                f"Expected base+arm+fingers = {expected_policy_actions}, got {self.num_policy_actions}."
            )
        self.fixed_arx_pose = bool(getattr(self.cfg, "fixed_arx_pose", True))
        self._policy_base_rot_slice = slice(0, 1)
        self._policy_base_xy_slice = slice(1, self.num_base_joints)
        self._policy_arm_slice = slice(self.num_base_joints, self.num_base_joints + self.num_arm_joints)
        self._policy_finger_slice = slice(
            self._policy_arm_slice.stop, self._policy_arm_slice.stop + self.num_finger_joints
        )

        # Loading all reward parameters
        self.robot_key_body_pos_scale = self.cfg.robot_key_body_pos_scale
        self.robot_body_quat_scale = self.cfg.robot_body_quat_scale
        self.door_joint_pos_scale = self.cfg.door_joint_pos_scale
        self.robot_base_joint_pos_scale = self.cfg.robot_base_joint_pos_scale
        self.robot_arm_joint_pos_scale = self.cfg.robot_arm_joint_pos_scale
        self.robot_finger_joint_pos_scale = self.cfg.robot_finger_joint_pos_scale
        self.robot_base_joint_vel_scale = self.cfg.robot_base_joint_vel_scale
        self.robot_arm_joint_vel_scale = self.cfg.robot_arm_joint_vel_scale
        self.robot_finger_joint_vel_scale = self.cfg.robot_finger_joint_vel_scale
        self.robot_body_lin_vel_scale = self.cfg.robot_body_lin_vel_scale
        self.robot_body_ang_vel_scale = self.cfg.robot_body_ang_vel_scale

        self.robot_key_body_pos_w = self.cfg.robot_key_body_pos_w
        self.robot_body_quat_w = self.cfg.robot_body_quat_w
        self.door_joint_pos_w = self.cfg.door_joint_pos_w
        self.robot_base_joint_pos_w = self.cfg.robot_base_joint_pos_w
        self.robot_arm_joint_pos_w = self.cfg.robot_arm_joint_pos_w
        self.robot_finger_joint_pos_w = self.cfg.robot_finger_joint_pos_w
        self.robot_arx_joint_pos_w = self.cfg.robot_arx_joint_pos_w
        self.robot_arx_tuck_joint_pos_w = self.cfg.robot_arx_tuck_joint_pos_w
        self.robot_base_joint_vel_w = self.cfg.robot_base_joint_vel_w
        self.robot_arm_joint_vel_w = self.cfg.robot_arm_joint_vel_w
        self.robot_finger_joint_vel_w = self.cfg.robot_finger_joint_vel_w
        self.hinge_contact_reward_w = self.cfg.hinge_contact_reward_w
        self.palm_handle_reward_w = self.cfg.palm_handle_reward_w
        self.base_door_contact_penalty_w = self.cfg.base_door_contact_penalty_w
        self.x5_door_contact_penalty_w = self.cfg.x5_door_contact_penalty_w
        self.robot_body_lin_vel_w = self.cfg.robot_body_lin_vel_w
        self.robot_body_ang_vel_w = self.cfg.robot_body_ang_vel_w
        self.joint_limit_penalty_w = self.cfg.joint_limit_penalty_w
        self.joint_limit_penalty_margin_ratio = self.cfg.joint_limit_penalty_margin_ratio
        self.self_collision_penalty_w = self.cfg.self_collision_penalty_w
        self.finger_door_contact_penalty_w = self.cfg.finger_door_contact_penalty_w
        self.franka_box_contact_penalty_w = self.cfg.franka_box_contact_penalty_w

        self.reset_key_body_pos_delta_min = self.cfg.reset_key_body_pos_delta_min
        self.reset_key_body_quat_delta_min = self.cfg.reset_key_body_quat_delta_min
        self.reset_key_body_pos_delta_max = self.cfg.reset_key_body_pos_delta_max
        self.reset_key_body_quat_delta_max = self.cfg.reset_key_body_quat_delta_max
        self.reset_door_joint_pos_delta_min = self.cfg.reset_door_joint_pos_delta_min
        self.reset_door_joint_pos_delta_max = self.cfg.reset_door_joint_pos_delta_max

        # self.last_actions = torch.zeros(
        #     (self.num_envs, len(self._robot_dof_idx)),
        #     device=self.device
        # )

        self.twist_indices = self.cfg.twist_indices
        self.robot_arx_tuck_joint_pos_scale = self.cfg.robot_arx_tuck_joint_pos_scale
        self._robot_arx_tuck_joint_pos_target = torch.tensor(
            [float(CAMERA_JOINT_DEFAULT_VALUES[joint_name]) for joint_name in self.cfg.arx_joints],
            device=self.device,
            dtype=self.robot.data.joint_pos.dtype,
        )

        self.num_door_assets = len(handle_offsets)
        self.env_asset_start_index = get_multi_door_asset_start_index(self.num_envs)
        self.env_asset_indices = get_multi_door_env_asset_indices(self.num_envs, device=self.device)
        self.env_family_ids = asset_family_ids.to(device=self.device, dtype=torch.long)[self.env_asset_indices]
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            sample_indices = self.env_asset_indices[: min(8, self.num_envs)].detach().cpu().tolist()
            sample_names = [Path(door_asset_paths[int(asset_idx)]).parent.name for asset_idx in sample_indices]
            print(
                f"[INFO][rank {rank}] Multi-door asset start index {self.env_asset_start_index}; "
                f"first env assets: {sample_names}"
            )
        self.handle_offsets = handle_offsets.to(self.device)[self.env_asset_indices]
        self.board_offsets = board_offsets.to(self.device)[self.env_asset_indices]
        # Closed-door (joints=0) handle center per env, in the door "base" link frame. Precomputed
        # from the URDF (fix_base => static geometry), so the closed-handle anchor needs no sim
        # capture; see get_closed_handle_position_in_base_frame().
        self.closed_handle_pos_door_base = closed_handle_offsets_base.to(
            device=self.device, dtype=torch.float32
        )[self.env_asset_indices]
        self.use_motion_ref = bool(getattr(self.cfg, "use_motion_ref", True))
        env_to_file_map = self.env_asset_indices.detach().cpu().tolist()
        if self.use_motion_ref:
            from DoorOpening.motion.multi_motion_lib import ReferenceMotionManager

            self.ref_motion_lib = ReferenceMotionManager(
                num_envs=self.num_envs,
                device=self.device,
                reset_from_start=False,
                env_to_file_map=env_to_file_map,
                twist_indices=self.twist_indices,
                step_dt=self.dt,
            )
            default_trial_steps = math.ceil(
                max(float(self.ref_motion_lib.num_frames - 1), 0.0) / max(float(self.ref_motion_lib.frame_step), 1e-6)
            ) + 1
            self.success_frame_idx = float(self.ref_motion_lib.num_frames - 1)
        else:
            self.ref_motion_lib = None
            default_trial_steps = max(1, math.ceil(float(self.cfg.episode_length_s) / max(float(self.dt), 1e-6)))
            self.success_frame_idx = float("inf")
        self.prob_get_first_key_frame = None
        self.max_trial_steps = default_trial_steps * torch.ones_like(self.episode_length_buf, device=self.device)
        self.games_to_track = 100
        self.completed_successes = deque(maxlen=self.games_to_track)
        self.completed_successes_by_family = {
            family_name: deque(maxlen=self.games_to_track) for family_name in DOOR_FAMILY_NAMES
        }
        self.episode_reached_last_frame = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-episode collision LATCHES + rolling completed-episode buffers, so we can report the
        # FRACTION OF ROLLOUTS that experienced any x5-arm / franka-box door collision -- far clearer
        # than the per-step instantaneous fraction.
        self.completed_x5_collisions = deque(maxlen=self.games_to_track)
        self.completed_franka_box_collisions = deque(maxlen=self.games_to_track)
        self.episode_x5_collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_franka_box_collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Success of the LAST completed episode per env (reached the last reference frame), recorded at
        # done and PERSISTED across reset so a distillation/eval loop can read it after env.step() --
        # by then the live episode_reached_last_frame latch has already been cleared in _reset_idx.
        self.last_success = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._door_minus_robot_x = float(DOOR_INITIAL_POS[0] - ROBOT_INITIAL_POS[0])
        # Per-motion reference FINAL base_x JOINT value. Push refs end far past the door (very negative
        # base_x), pull refs end near it -> tells us the door type per env for the far-side threshold.
        self.motion_final_base_rel_x = None
        if self.ref_motion_lib is not None:
            _traj = getattr(self.ref_motion_lib, "robot_joint_pos_traj", None)
            if _traj is not None and _traj.ndim == 3:
                _base_x_traj_idx = FULL_JOINT_NAMES.index("base_x_joint")
                _last_frame = max(int(self.ref_motion_lib.num_frames) - 1, 0)
                self.motion_final_base_rel_x = _traj[:, _last_frame, _base_x_traj_idx].detach().clone().to(self.device)

        torch.set_printoptions(precision=4, sci_mode=False)

        self.reset_progress_total = self.cfg.reset_progress_total
        self.adr_reset_progress_total = self.cfg.adr_reset_progress_total
        self._rlgames_env_frames = 0

        self.alive_base = self.cfg.alive_base
        self.alive_bonus = self.cfg.alive_bonus
        self.termination_penalty = self.cfg.termination_penalty

        # DEXTRAH-style split: EventTerms handle reset-time physics DR, while the env samples
        # reset/observation/controller noise from ADR.
        self.dooropening_adr = DoorOpeningADR(self.event_manager, self.cfg.adr_cfg_dict, self.cfg.adr_custom_cfg_dict)
        self._adr_enabled = bool(self.cfg.enable_adr)
        initial_adr_increments = self.cfg.starting_adr_increments if self._adr_enabled else 0
        self.dooropening_adr.set_num_increments(initial_adr_increments)

        self.robot_spawn_noise_widths = self._make_env_buffer_dict(self.cfg.adr_custom_cfg_dict["robot_spawn"].keys())
        robot_state_cfg = self.cfg.adr_custom_cfg_dict["robot_state_noise"]
        door_state_cfg = self.cfg.adr_custom_cfg_dict["door_state_noise"]
        self.robot_state_noise_widths = self._make_env_buffer_dict(
            [key for key in robot_state_cfg if key.endswith("_noise")]
        )
        self.robot_state_biases = self._make_env_buffer_dict([key for key in robot_state_cfg if key.endswith("_bias")])
        self.door_state_noise_widths = self._make_env_buffer_dict(
            [key for key in door_state_cfg if key.endswith("_noise")]
        )
        self.door_state_biases = self._make_env_buffer_dict([key for key in door_state_cfg if key.endswith("_bias")])
        self.student_joint_pos_noise_widths = self._make_env_buffer_dict(
            [
                "base_xy_joint_pos_noise",
                "base_rot_joint_pos_noise",
                "arm_joint_pos_noise",
                "finger_joint_pos_noise",
            ]
        )
        self.student_joint_pos_biases = self._make_env_buffer_dict(
            [
                "base_xy_joint_pos_bias",
                "base_rot_joint_pos_bias",
                "arm_joint_pos_bias",
                "finger_joint_pos_bias",
            ]
        )
        # Action latency: the PD target applied on a step is the one computed `_action_latency_buf`
        # env/control steps earlier. The per-env latency (in steps) is sampled at reset and ramps
        # from 1 up to the ADR max. The history ring stores the most recent past targets, with
        # index 0 = the target from the previous step (lag 1) and the last index = the oldest (lag max).
        self._max_action_latency = max(
            1, int(round(float(self.cfg.adr_custom_cfg_dict["action_latency"]["latency_steps"][1])))
        )
        self._action_latency_buf = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._action_target_history = torch.zeros(
            (self.num_envs, self._max_action_latency, self.num_robot_actions), device=self.device
        )
        self._door_nominal_joint_stiffness = self.door.data.joint_stiffness.clone()
        self._door_nominal_joint_damping = self.door.data.joint_damping.clone()
        self._door_handle_effort_limits = torch.full(
            (self.num_envs, 1),
            float(self.cfg.door_handle_effort_limit_sim),
            device=self.device,
        )
        # Per-env panel-swing (joint_1) effort-limit cap, sampled at reset and read every step by
        # edit_door_articulation (which caps the unlatched restoring torque and switches to 1e6 while
        # latched). Seeded to the ADR-start upper bound; overwritten on the first reset.
        self._door_panel_effort_limits = torch.full(
            (self.num_envs, 1),
            float(self.cfg.door_panel_effort_limit_start_range_nm[1]),
            device=self.device,
        )
        # Per-env handle (joint_2) unlatch angle threshold (radians), sampled at reset and read every
        # step by edit_door_articulation. 1-D (num_envs,) so it broadcasts against door.data.joint_pos
        # [:, joint_2]; a (num_envs, 1) shape would broadcast the wrong way in the lock comparison.
        self._door_latch_thresholds = torch.full(
            (self.num_envs,),
            float(self.cfg.door_latch_threshold_start_range_rad[1]),
            device=self.device,
        )
        self._dr_metrics_interval = max(int(self.cfg.dr_metrics_interval), 1)
        self._log_verbose_dr_metrics = bool(self.cfg.log_verbose_dr_metrics)
        self._init_viser_pointcloud_recording()

    def _get_filtered_contact_force_w(self, sensor, expected_num_envs=None, filter_indices=None) -> torch.Tensor:
        return get_filtered_contact_force_w(
            sensor,
            expected_num_envs=expected_num_envs,
            filter_indices=filter_indices,
        )

    def _stacked_self_contact_force_norm(self, sensor_names) -> torch.Tensor:
        """Per-body net filtered contact-force magnitude, stacked across single-body sensors.

        Each ``sensor_names`` entry is a single-body contact sensor (filtered contacts require a
        single-body ``prim_path``), so ``get_self_contact_body_force_norm`` returns ``[N, 1]``. We
        stack them to recover the ``[N, num_bodies]`` tensor the old multi-body sensor produced.
        """
        per_body = [
            get_self_contact_body_force_norm(self.scene.sensors[name], expected_num_envs=self.num_envs)[:, 0]
            for name in sensor_names
        ]
        return torch.stack(per_body, dim=-1)

    def _get_x5_body_contact_force_norm(self, filter_indices=None, include_franka_box=True) -> torch.Tensor:
        sensor_names = [
            "contact_forces_door_x5_link2",
            "contact_forces_door_x5_link3",
            "contact_forces_door_x5_link4",
            "contact_forces_door_x5_link5",
            "contact_forces_door_x5_camera",
        ]
        if include_franka_box:
            # Legacy path (default callers now pass include_franka_box=False): the franka control
            # box is handled by its own graded penalty in _get_rewards, not the harsh x5 penalty or
            # a hard termination. Kept only for callers that still want the box lumped in.
            sensor_names.append("contact_forces_door_franka_box")
        per_body_force_norms = []
        for sensor_name in sensor_names:
            force_w = self._get_filtered_contact_force_w(
                self.scene.sensors[sensor_name],
                expected_num_envs=self.num_envs,
                filter_indices=filter_indices,
            )
            per_body_force_norms.append(torch.linalg.vector_norm(force_w, dim=-1))
        return torch.stack(per_body_force_norms, dim=-1).max(dim=-1).values

    def _get_franka_box_contact_force_norm(self, filter_indices=None) -> torch.Tensor:
        force_w = self._get_filtered_contact_force_w(
            self.scene.sensors["contact_forces_door_franka_box"],
            expected_num_envs=self.num_envs,
            filter_indices=filter_indices,
        )
        return torch.linalg.vector_norm(force_w, dim=-1)

    def _get_x5_body_frame_contact_force_norm(self) -> torch.Tensor:
        return self._get_x5_body_contact_force_norm(filter_indices=(DOOR_FRAME_FILTER_INDEX,))

    def _get_franka_arx_contact_force_norm(self) -> torch.Tensor:
        """Total contact-force magnitude between the franka arm and the arx/x5 camera arm, per env.

        The franka self-collision sensor filters against [x5 bodies, base bodies, door frame]; the
        arx (x5) bodies are the FIRST ``len(SELF_COLLISION_X5_BODIES)`` filters. Sum the franka<->arx
        contact forces (over all franka bodies + those arx filters) and return the magnitude.
        """
        arx_filter_indices = tuple(range(len(SELF_COLLISION_X5_BODIES)))
        force_w = self._get_filtered_contact_force_w(
            self.scene.sensors["contact_forces_self_collision_franka"],
            expected_num_envs=self.num_envs,
            filter_indices=arx_filter_indices,
        )
        return torch.linalg.vector_norm(force_w, dim=-1)

    def _get_self_collision_body_count(self) -> torch.Tensor:
        """Number of robot links in self-collision, per env (fine-grained, per-link).

        This is the r_contact term in r = r_target - lambda_l*r_limit - lambda_c*r_contact.
        We count the franka arm, x5/arx camera arm, and mobile-base chassis links whose net
        self-contact force with any other non-adjacent link exceeds the threshold. Finger<->finger
        self-collision is still excluded (it drove finger poses conservative), but the fingers
        ARE checked against the panda_link7 flange (contact_forces_self_collision_hand): the fingers
        curling back into the flange is a real collision we penalize.
        """
        threshold = self.cfg.self_collision_force_threshold
        # Each group sensor is multi-body: get_self_contact_body_force_norm returns
        # [N, group_bodies] (net filtered force per body). Concatenate the groups, then
        # count the bodies whose self-contact force exceeds the threshold.
        force_norm = torch.cat(
            [
                get_self_contact_body_force_norm(self.scene.sensors[name], expected_num_envs=self.num_envs)
                for name in SELF_COLLISION_SENSOR_NAMES
            ],
            dim=-1,
        )
        count = (force_norm > threshold).sum(dim=-1)
        return count.to(dtype=force_norm.dtype)

    def set_train_info(self, env_frames: int, algo=None, **kwargs):
        self._rlgames_env_frames = int(env_frames)

    def _get_curriculum_step_count(self) -> int:
        # Curriculum/reset scheduling should follow actual env progress, not logging side effects.
        return int(self.common_step_counter)

    def _init_viser_pointcloud_recording(self):
        """Initialize optional teacher-training point-cloud replay dumps."""

        self._reset_viser_pointcloud_runtime_state()

        record_cfg = dict(getattr(self.cfg, "viser_pointcloud", {}) or {})
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0 or not bool(record_cfg.get("enabled", False)):
            return

        self.viser_pointcloud_enabled = True
        self._configure_viser_pointcloud_from_cfg(record_cfg)

        self._init_viser_pointcloud_asset_samplers()
        self._init_viser_pointcloud_metadata()

        print(f"Viser .pt point-cloud capture enabled for env {self._viser_pointcloud_env_id}.")
        print(
            "Replay chunks will be written as "
            f"{format_iterated_record_path(self.viser_pointcloud_path, '<chunk_tag>')}"
        )
        print(f"Replay chunks will flush every {self._viser_pointcloud_save_interval} iterations.")
        if self._viser_pointcloud_max_frames > 0:
            print(f"Each replay chunk will keep at most {self._viser_pointcloud_max_frames} frames.")

    def _reset_viser_pointcloud_runtime_state(self):
        self.viser_pointcloud_enabled = False
        self.viser_pointcloud_path = ""
        self.viser_pointcloud_frames = []
        self.viser_pointcloud_frame_count = 0
        self.viser_pointcloud_chunk_index = 0
        self.viser_pointcloud_chunk_start_iteration = None
        self.viser_pointcloud_latest_iteration = None
        self._viser_pointcloud_metadata = {}
        self._viser_robot_sampler = None
        self._viser_door_samplers = {}
        self._viser_env_asset_idx = None

    def _configure_viser_pointcloud_from_cfg(self, record_cfg: dict):
        self._viser_pointcloud_env_id = max(0, min(int(record_cfg["env_id"]), self.num_envs - 1))
        self._viser_pointcloud_capture_interval = max(
            1,
            int(record_cfg.get("capture_interval", record_cfg.get("record_interval", 1))),
        )
        self._viser_pointcloud_save_interval = max(
            1,
            int(record_cfg.get("save_interval", record_cfg.get("raw_interval", 5000))),
        )
        self._viser_pointcloud_max_frames = max(0, int(record_cfg.get("max_frames", 2000)))
        self._viser_pointcloud_max_points = int(record_cfg.get("max_points", 6000))
        self._viser_robot_num_points = int(record_cfg.get("robot_num_points", record_cfg.get("num_points", 4096)))
        self._viser_door_num_points = int(record_cfg.get("door_num_points", record_cfg.get("num_points", 4096)))

        configured_path = Path(str(record_cfg.get("path", record_cfg.get("raw_path", "teacher_viser_replay.pt")))).expanduser()
        if not configured_path.is_absolute():
            log_dir = Path(str(getattr(self.cfg, "log_dir", os.getcwd())))
            configured_path = log_dir / configured_path
        self.viser_pointcloud_path = str(configured_path)
        raw_dir = os.path.dirname(self.viser_pointcloud_path)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)

    def _init_viser_pointcloud_asset_samplers(self):
        asset_index_by_dir = {Path(asset_path).resolve().parent: idx for idx, asset_path in enumerate(door_asset_paths)}
        if self.ref_motion_lib is None:
            self._viser_env_asset_idx = self.env_asset_indices.to(device=self.device, dtype=torch.long)
        else:
            motion_to_asset_idx = []
            for motion_path in motion_traj_paths:
                motion_dir = Path(motion_path).resolve().parent
                if motion_dir not in asset_index_by_dir:
                    raise KeyError(f"Could not map motion file '{motion_path}' to a door asset path.")
                motion_to_asset_idx.append(asset_index_by_dir[motion_dir])
            motion_to_asset_idx = torch.tensor(motion_to_asset_idx, device=self.device, dtype=torch.long)
            env_motion_idx = self.ref_motion_lib.env_to_file_map.to(device=self.device, dtype=torch.long)
            self._viser_env_asset_idx = motion_to_asset_idx[env_motion_idx]

        selected_asset_idx = int(self._viser_env_asset_idx[self._viser_pointcloud_env_id].detach().cpu().item())
        self._viser_door_samplers = {
            selected_asset_idx: FrankaGripperSampler(
                door_asset_paths[selected_asset_idx],
                device=self.device,
                num_points=self._viser_door_num_points,
            )
        }

        self._viser_robot_sampler = FrankaGripperSampler(
            glorbot_urdf_path,
            device=self.device,
            num_points=self._viser_robot_num_points,
        )
        robot_sampler_joint_names = list(self._viser_robot_sampler.robot.actuated_joint_names)
        robot_joint_ids, robot_joint_names = self.robot.find_joints(robot_sampler_joint_names)
        if len(robot_joint_ids) != len(robot_sampler_joint_names):
            raise ValueError("Could not map every robot sampler joint to the IsaacLab robot articulation.")
        self._viser_robot_sampler_joint_ids = torch.tensor(robot_joint_ids, device=self.device, dtype=torch.long)
        robot_joint_name_to_idx = {name: idx for idx, name in enumerate(robot_joint_names)}
        self._viser_robot_sampler_joint_reorder = [robot_joint_name_to_idx[name] for name in robot_sampler_joint_names]
        self._viser_robot_root_body_idx = int(self._robot_base_link_idx[0])

    def _init_viser_pointcloud_metadata(self):
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_dt = float(getattr(sim_cfg, "dt", self.cfg.sim_dt))
        env_step_dt = max(float(getattr(self, "dt", sim_dt * int(getattr(self.cfg, "decimation", 1)))), 1e-6)
        self._viser_pointcloud_metadata = {
            "format": "dooropening_viser_replay_v2",
            "capture_mode": "teacher_training_pointcloud_chunk",
            "pointcloud_frame": "world",
            "pointcloud_streams": [
                {"name": "robot", "label": "Robot", "color": (79, 195, 247), "point_size_scale": 1.0},
                {"name": "door", "label": "Door", "color": (255, 193, 7), "point_size_scale": 1.0},
            ],
            "pointcloud_source": "articulation_sampler",
            "sim_dt": sim_dt,
            "decimation": int(getattr(self.cfg, "decimation", 1)),
            "env_step_dt": env_step_dt,
            "env_step_fps": 1.0 / env_step_dt,
            "frame_dt": env_step_dt * self._viser_pointcloud_capture_interval,
            "frame_fps": 1.0 / (env_step_dt * self._viser_pointcloud_capture_interval),
            "pointcloud_sensor_dt": env_step_dt * self._viser_pointcloud_capture_interval,
            "pointcloud_sensor_fps": 1.0 / (env_step_dt * self._viser_pointcloud_capture_interval),
            "raw_cloud_config": {
                "max_points": self._viser_pointcloud_max_points,
                "robot_num_points": self._viser_robot_num_points,
                "door_num_points": self._viser_door_num_points,
                "capture_interval": self._viser_pointcloud_capture_interval,
            },
            "glorbot_urdf_path": str(glorbot_urdf_path),
            "robot_joint_names": list(getattr(self.robot, "joint_names", [])),
            "door_joint_names": list(getattr(self.door, "joint_names", [])),
        }

    def _sample_viser_robot_pointcloud_world(self, env_id: int) -> torch.Tensor:
        robot_joint_pos = self.robot.data.joint_pos[env_id : env_id + 1, self._viser_robot_sampler_joint_ids]
        robot_joint_pos = robot_joint_pos[:, self._viser_robot_sampler_joint_reorder]
        robot_local_pcd = self._viser_robot_sampler.sample(robot_joint_pos)
        robot_root_pos_w = self.robot.data.body_pos_w[env_id : env_id + 1, self._viser_robot_root_body_idx]
        robot_root_quat_w = self.robot.data.body_quat_w[env_id : env_id + 1, self._viser_robot_root_body_idx]
        quat = robot_root_quat_w.unsqueeze(1).expand(-1, robot_local_pcd.shape[1], -1)
        return (quat_apply(quat, robot_local_pcd) + robot_root_pos_w.unsqueeze(1))[0]

    def _sample_viser_door_pointcloud_world(self, env_id: int) -> torch.Tensor:
        asset_idx = int(self._viser_env_asset_idx[env_id].detach().cpu().item())
        sampler = self._viser_door_samplers.get(asset_idx)
        if sampler is None:
            sampler = FrankaGripperSampler(
                door_asset_paths[asset_idx],
                device=self.device,
                num_points=self._viser_door_num_points,
            )
            self._viser_door_samplers[asset_idx] = sampler

        door_joint_pos = self.door.data.joint_pos[env_id : env_id + 1]
        local_pcd = sampler.sample(door_joint_pos)
        door_base_pos_w = self.door.data.body_pos_w[env_id : env_id + 1, self._door_base_link_idx]
        door_base_quat_w = self.door.data.body_quat_w[env_id : env_id + 1, self._door_base_link_idx]
        quat = door_base_quat_w.unsqueeze(1).expand(-1, local_pcd.shape[1], -1)
        return (quat_apply(quat, local_pcd) + door_base_pos_w.unsqueeze(1))[0]

    def _trim_viser_pointcloud_frames(self):
        if (
            self._viser_pointcloud_max_frames > 0
            and len(self.viser_pointcloud_frames) > self._viser_pointcloud_max_frames
        ):
            self.viser_pointcloud_frames = self.viser_pointcloud_frames[-self._viser_pointcloud_max_frames :]
        self.viser_pointcloud_frame_count = len(self.viser_pointcloud_frames)
        if self.viser_pointcloud_frame_count > 0:
            self.viser_pointcloud_chunk_start_iteration = int(self.viser_pointcloud_frames[0]["iteration"])
        else:
            self.viser_pointcloud_chunk_start_iteration = None

    def _build_viser_pointcloud_payload(self, chunk_complete: bool) -> dict:
        latest_iteration = int(self.viser_pointcloud_latest_iteration or 0)
        payload = {
            **self._viser_pointcloud_metadata,
            "chunk_index": int(self.viser_pointcloud_chunk_index),
            "chunk_env_id": int(self._viser_pointcloud_env_id),
            "chunk_start_iteration": None
            if self.viser_pointcloud_chunk_start_iteration is None
            else int(self.viser_pointcloud_chunk_start_iteration),
            "chunk_end_iteration": latest_iteration,
            "chunk_complete": bool(chunk_complete),
            "chunk_frame_count": int(self.viser_pointcloud_frame_count),
            "episode_index": int(self.viser_pointcloud_chunk_index),
            "episode_env_id": int(self._viser_pointcloud_env_id),
            "episode_start_iteration": None
            if self.viser_pointcloud_chunk_start_iteration is None
            else int(self.viser_pointcloud_chunk_start_iteration),
            "episode_end_iteration": latest_iteration,
            "episode_complete": bool(chunk_complete),
            "episode_frame_count": int(self.viser_pointcloud_frame_count),
            "frames": self.viser_pointcloud_frames,
        }
        return payload

    def _flush_viser_pointcloud_recording(self, chunk_complete: bool, reason: str):
        if not self.viser_pointcloud_enabled or self.viser_pointcloud_frame_count <= 0:
            return

        latest_iteration = int(self.viser_pointcloud_latest_iteration or 0)
        record_tag = f"chunk_{self.viser_pointcloud_chunk_index:04d}_iter_{latest_iteration}"
        torch.save(
            self._build_viser_pointcloud_payload(chunk_complete=chunk_complete),
            format_iterated_record_path(self.viser_pointcloud_path, record_tag),
        )
        status = "complete" if chunk_complete else "partial"
        print(
            "Saved {} Viser .pt chunk {} for env {} with {} frames ({}).".format(
                status,
                self.viser_pointcloud_chunk_index,
                self._viser_pointcloud_env_id,
                self.viser_pointcloud_frame_count,
                reason,
            )
        )

        self.viser_pointcloud_frames = []
        self.viser_pointcloud_frame_count = 0
        self.viser_pointcloud_chunk_start_iteration = None
        self.viser_pointcloud_latest_iteration = None

    def _maybe_flush_viser_pointcloud_snapshot(self, iteration: int):
        if not self.viser_pointcloud_enabled or self.viser_pointcloud_frame_count <= 0:
            return
        if (int(iteration) + 1) % self._viser_pointcloud_save_interval != 0:
            return
        self._flush_viser_pointcloud_recording(
            chunk_complete=True,
            reason=f"save_interval {self._viser_pointcloud_save_interval} reached at iteration {int(iteration)}",
        )

    def _maybe_record_viser_pointcloud(self):
        if not self.viser_pointcloud_enabled or self._viser_env_asset_idx is None:
            return
        iteration = int(self.common_step_counter)
        if iteration % self._viser_pointcloud_capture_interval != 0:
            return

        env_id = self._viser_pointcloud_env_id
        robot_points_world = self._sample_viser_robot_pointcloud_world(env_id)
        door_points_world = self._sample_viser_door_pointcloud_world(env_id)
        robot_points_world = prepare_pointcloud(
            robot_points_world,
            max_points=self._viser_pointcloud_max_points,
        ).to(dtype=torch.float16)
        door_points_world = prepare_pointcloud(
            door_points_world,
            max_points=self._viser_pointcloud_max_points,
        ).to(dtype=torch.float16)
        door_asset_idx = int(self._viser_env_asset_idx[env_id].detach().cpu().item())

        if self.viser_pointcloud_chunk_start_iteration is None:
            self.viser_pointcloud_chunk_index += 1
            self.viser_pointcloud_chunk_start_iteration = iteration
        self.viser_pointcloud_latest_iteration = iteration

        frame_record = {
            "iteration": iteration,
            "sim_frame": int(self._rlgames_env_frames if self._rlgames_env_frames > 0 else iteration),
            "env_id": int(env_id),
            "pointcloud_source": "articulation_sampler",
            "robot_points_world": robot_points_world,
            "door_points_world": door_points_world,
            "pointclouds": {
                "robot": robot_points_world,
                "door": door_points_world,
            },
            "rlgames_env_frames": int(self._rlgames_env_frames),
            "robot_joint_pos": self.robot.data.joint_pos[env_id].detach().cpu().to(dtype=torch.float32),
            "door_joint_pos": self.door.data.joint_pos[env_id].detach().cpu().to(dtype=torch.float32),
            "robot_base_pos_w": self.robot.data.body_pos_w[env_id, self._robot_base_body_link_idx].detach().cpu().to(dtype=torch.float32),
            "robot_base_quat_w": self.robot.data.body_quat_w[env_id, self._robot_base_body_link_idx].detach().cpu().to(dtype=torch.float32),
            "door_base_pos_w": self.door.data.body_pos_w[env_id, self._door_base_link_idx].detach().cpu().to(dtype=torch.float32),
            "door_base_quat_w": self.door.data.body_quat_w[env_id, self._door_base_link_idx].detach().cpu().to(dtype=torch.float32),
            "door_asset_idx": door_asset_idx,
            "door_asset_path": str(door_asset_paths[door_asset_idx]),
        }
        self.viser_pointcloud_frames.append(frame_record)
        self._trim_viser_pointcloud_frames()
        self._maybe_flush_viser_pointcloud_snapshot(iteration)

    def _initialize_runtime_event_terms(self):
        if not self.cfg.events:
            return

        # IsaacLab's EventManager initializes class-based terms eagerly for "prestartup",
        # but reset-time class terms still need their play-time processing once physics is live.
        for mode in self.event_manager.available_modes:
            if mode == "prestartup":
                continue
            term_names = self.event_manager._mode_term_names.get(mode, [])
            term_cfgs = self.event_manager._mode_term_cfgs.get(mode, [])
            for term_name, term_cfg in zip(term_names, term_cfgs):
                if inspect.isclass(term_cfg.func):
                    self.event_manager._process_term_cfg_at_play(term_name, term_cfg)

    def _activate_door_contact_reporters(self):
        try:
            from isaaclab.sim.schemas import schemas as sim_schemas
        except ImportError:
            from isaaclab.sim.schemas.schemas import activate_contact_sensors
        else:
            activate_contact_sensors = sim_schemas.activate_contact_sensors

        for env_id in range(self.num_envs):
            activate_contact_sensors(f"/World/envs/env_{env_id}/Door", 1.0)

    def _setup_scene(self):
        configure_multi_door_assets_for_rank(self.cfg.door_cfg, int(self.cfg.scene.num_envs))
        self.robot = Articulation(self.cfg.robot_cfg)
        # Make the robot's per-link `collisions` scopes renderable by the collider debug viz
        # (green overlay); the converter marks them instanceable and the GUI skips instanced
        # colliders. Runs on the spawn source before cloning, so it's free at runtime.
        if self.sim.has_gui():
            disable_collision_scope_instancing(self.cfg.robot_cfg.prim_path)
        self.door = Articulation(self.cfg.door_cfg)
        # Same instancing fix for the door's converted USD, so its collider boxes render too.
        # GUI-only cosmetics: on the heterogeneous multi-door set each SetInstanceable() triggers a
        # USD recomposition (O(N^2) over thousands of unique door colliders) and stalls headless
        # training for many minutes, so skip it entirely when there is no viewer.
        if self.sim.has_gui():
            disable_collision_scope_instancing(self.cfg.door_cfg.prim_path)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Do not clone env_0 over the other envs for heterogeneous multi-door
        # scenes. The MultiAssetSpawner has already populated each env.
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["door"] = self.door
        self._activate_door_contact_reporters()
        self.pointcloud_camera = None
        pointcloud_render_mode = str(getattr(self.cfg, "pointcloud_render_mode", "none")).lower()
        if pointcloud_render_mode not in {"none", "depth", "lidar"}:
            raise ValueError(
                "Unsupported pointcloud_render_mode "
                f"'{pointcloud_render_mode}'. Expected one of ['none', 'depth', 'lidar']."
            )
        # The render mode is the source of truth: only depth mode enables the x5 pointcloud camera.
        enable_pointcloud_camera = pointcloud_render_mode == "depth"
        if enable_pointcloud_camera:
            self.pointcloud_camera = Camera(self.cfg.pointcloud_camera_cfg)
            self.scene.sensors["pointcloud_camera"] = self.pointcloud_camera
        self.scene.sensors["contact_forces_door2"] = ContactSensor(self.cfg.contact_forces_door2)
        self.scene.sensors["contact_forces_door2_palm"] = ContactSensor(self.cfg.contact_forces_door2_palm)
        self.scene.sensors["contact_forces_door_panel"] = ContactSensor(self.cfg.contact_forces_door_panel)
        # base<->door split one-sensor-per-body (filtered contacts need a single-body prim_path)
        for _name in BASE_DOOR_SENSOR_NAMES:
            self.scene.sensors[_name] = ContactSensor(getattr(self.cfg, _name))
        self.scene.sensors["contact_forces_door_franka_box"] = ContactSensor(self.cfg.contact_forces_door_franka_box)
        self.scene.sensors["contact_forces_door_x5_link2"] = ContactSensor(self.cfg.contact_forces_door_x5_link2)
        self.scene.sensors["contact_forces_door_x5_link3"] = ContactSensor(self.cfg.contact_forces_door_x5_link3)
        self.scene.sensors["contact_forces_door_x5_link4"] = ContactSensor(self.cfg.contact_forces_door_x5_link4)
        self.scene.sensors["contact_forces_door_x5_link5"] = ContactSensor(self.cfg.contact_forces_door_x5_link5)
        self.scene.sensors["contact_forces_door_x5_camera"] = ContactSensor(self.cfg.contact_forces_door_x5_camera)
        # Self-collision penalty over the franka arm, x5/arx arm, and base -- grouped into 3
        # multi-body sensors (franka / x5 / base) to keep the PhysX contact-view count low.
        for _name in SELF_COLLISION_SENSOR_NAMES:
            self.scene.sensors[_name] = ContactSensor(getattr(self.cfg, _name))
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)    

    def _make_env_buffer_dict(self, keys):
        return {key: torch.zeros((self.num_envs, 1), device=self.device) for key in keys}

    def _custom_param_upper_limit(self, group: str, name: str) -> float:
        return float(self.cfg.adr_custom_cfg_dict[group][name][1])

    def _current_custom_param(self, group: str, name: str) -> float:
        if not self._adr_enabled:
            return 0.0
        return self.dooropening_adr.get_custom_param_value(group, name)

    def _current_event_param(self, term_name: str, param_name: str):
        return self.event_manager.get_term_cfg(term_name).params[param_name]

    def _current_door_handle_effort_limit_range(self) -> tuple[float, float]:
        if not self._adr_enabled:
            effort = float(self.cfg.door_handle_effort_limit_sim)
            return effort, effort
        effort_limits = self._current_event_param(
            "door_hinge_joint_effort_limit", "effort_limit_distribution_params"
        )
        return float(effort_limits[0]), float(effort_limits[1])

    def _current_action_latency_bounds(self) -> tuple[int, int]:
        """Current per-env action-latency sampling bounds, in env/control steps.

        The lower bound is fixed at the configured starting latency; the upper bound ramps with
        ADR from the starting latency up to the configured max. With ADR disabled the latency is
        pinned to the starting value.
        """
        latency_range = self.cfg.adr_custom_cfg_dict["action_latency"]["latency_steps"]
        min_lag = max(1, int(round(float(latency_range[0]))))
        current_max = self._current_custom_param("action_latency", "latency_steps")
        max_lag = max(min_lag, int(round(current_max)))
        max_lag = min(max_lag, self._max_action_latency)
        return min_lag, max_lag

    def _refresh_nominal_door_joint_gains(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._door_nominal_joint_stiffness.copy_(self.door.data.joint_stiffness)
            self._door_nominal_joint_damping.copy_(self.door.data.joint_damping)
            return
        self._door_nominal_joint_stiffness[env_ids] = self.door.data.joint_stiffness[env_ids]
        self._door_nominal_joint_damping[env_ids] = self.door.data.joint_damping[env_ids]

    def _sample_door_handle_effort_limits(self, env_ids: torch.Tensor):
        effort_min, effort_max = self._current_door_handle_effort_limit_range()
        if effort_max < effort_min:
            raise ValueError(
                f"Door handle effort ADR range must satisfy min <= max, got min={effort_min}, max={effort_max}."
            )
        if effort_max == effort_min:
            self._door_handle_effort_limits[env_ids, 0] = effort_min
            return
        self._door_handle_effort_limits[env_ids, 0] = effort_min + (effort_max - effort_min) * torch.rand(
            len(env_ids), device=self.device
        )

    def _apply_door_handle_effort_limits(self, env_ids: torch.Tensor):
        self.door.write_joint_effort_limit_to_sim(
            self._door_handle_effort_limits[env_ids],
            joint_ids=[self._door_hinge_joint_idx],
            env_ids=env_ids,
        )

    def _current_door_panel_effort_limit_range(self) -> tuple[float, float]:
        start_range = self.cfg.door_panel_effort_limit_start_range_nm
        if not self._adr_enabled:
            return float(start_range[0]), float(start_range[1])
        interpolated = self.dooropening_adr.get_interpolated_range(
            start_range, self.cfg.door_panel_effort_limit_range_nm
        )
        return float(interpolated[0]), float(interpolated[1])

    def _sample_door_panel_effort_limits(self, env_ids: torch.Tensor):
        # No separate _apply: the sampled buffer is read every step by edit_door_articulation.
        effort_min, effort_max = self._current_door_panel_effort_limit_range()
        if effort_max < effort_min:
            raise ValueError(
                f"Door panel effort ADR range must satisfy min <= max, got min={effort_min}, max={effort_max}."
            )
        if effort_max == effort_min:
            self._door_panel_effort_limits[env_ids, 0] = effort_min
            return
        self._door_panel_effort_limits[env_ids, 0] = effort_min + (effort_max - effort_min) * torch.rand(
            len(env_ids), device=self.device
        )

    def _current_door_latch_threshold_range(self) -> tuple[float, float]:
        start_range = self.cfg.door_latch_threshold_start_range_rad
        if not self._adr_enabled:
            return float(start_range[0]), float(start_range[1])
        interpolated = self.dooropening_adr.get_interpolated_range(
            start_range, self.cfg.door_latch_threshold_range_rad
        )
        return float(interpolated[0]), float(interpolated[1])

    def _sample_door_latch_thresholds(self, env_ids: torch.Tensor):
        # No separate _apply: the sampled buffer is read every step by edit_door_articulation.
        low, high = self._current_door_latch_threshold_range()
        if high < low:
            raise ValueError(
                f"Door latch threshold ADR range must satisfy min <= max, got min={low}, max={high}."
            )
        if high == low:
            self._door_latch_thresholds[env_ids] = low
            return
        self._door_latch_thresholds[env_ids] = low + (high - low) * torch.rand(
            len(env_ids), device=self.device
        )

    def _compute_curriculum_progress(self, progress_total: float) -> float:
        progress_total = max(float(progress_total), 1.0)
        return min(float(self._get_curriculum_step_count()) / progress_total, 1.0)

    def _mean_completed_metric(self, values) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    def _update_last_frame_tracker(self):
        if self.ref_motion_lib is None:
            return
        current_frame_idx = torch.clamp(
            self.ref_motion_lib.frame_idx.to(device=self.device, dtype=torch.float32),
            max=self.success_frame_idx,
        )
        self.episode_reached_last_frame |= current_frame_idx >= self.success_frame_idx

    def _get_reached_last_frame_mask(self) -> torch.Tensor:
        if self.ref_motion_lib is None:
            return torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        current_frame_idx = torch.clamp(
            self.ref_motion_lib.frame_idx.to(device=self.device, dtype=torch.float32),
            max=self.success_frame_idx,
        )
        return current_frame_idx >= self.success_frame_idx

    def _base_dist_past_door(self, base_x_joint: torch.Tensor) -> torch.Tensor:
        """Signed distance the base has moved PAST the door plane, from the base_x JOINT value.
        base_x GROWS as the robot drives toward/through the door (verified from traj.pkl: ~0 at spawn
        -> ~1.5 for pull, ~2.5 for push), i.e. base world x = ROBOT_INITIAL_POS.x - base_x_joint. So
        dist_past = DOOR_INITIAL_POS.x - world_x = base_x_joint + (DOOR_INITIAL_POS.x - ROBOT_INITIAL_POS.x)
        = base_x_joint - 1.0 here (positive => past the door on the far side)."""
        return base_x_joint + self._door_minus_robot_x

    def _get_is_push_env(self) -> torch.Tensor:
        """Per-env push flag as float [num_envs] (1.0 = push, 0.0 = pull), inferred the same way as
        _far_side_target (push refs traverse far past the door). Static per assignment -> cached.

        Gates the direction-specific handle rewards: the new palm-handle reward is push-only, the old
        finger-inclusive hinge_contact reward is pull-only, so the pull pipeline is unchanged.
        """
        cached = getattr(self, "_is_push_env_cached", None)
        if cached is not None:
            return cached
        if self.motion_final_base_rel_x is None:
            is_push = torch.zeros(self.num_envs, device=self.device)
        else:
            env_motion_idx = self.ref_motion_lib.env_to_file_map.to(device=self.device, dtype=torch.long)
            ref_final_base_x_joint = self.motion_final_base_rel_x[env_motion_idx]
            ref_dist_past = self._base_dist_past_door(ref_final_base_x_joint)
            is_push = (ref_dist_past > float(self.cfg.success_far_push_ref_dist)).to(dtype=torch.float32)
        self._is_push_env_cached = is_push.reshape(self.num_envs)
        return self._is_push_env_cached

    def _update_success_metrics(self):
        self._update_last_frame_tracker()

        done_mask = torch.nonzero(self.reset_buf, as_tuple=False).squeeze(-1)
        if done_mask.numel() > 0:
            # Task success = the robot reached the last reference frame during the episode.
            episode_successes_tensor = self.episode_reached_last_frame[done_mask].to(dtype=torch.float32)
            episode_successes = episode_successes_tensor.detach().cpu().tolist()
            self.completed_successes.extend(float(value) for value in episode_successes)
            # Persist for a distillation/eval loop to read after env.step() (survives _reset_idx).
            self.last_success[done_mask] = episode_successes_tensor
            done_family_ids = self.env_family_ids[done_mask].detach().cpu().tolist()
            for family_id, value in zip(done_family_ids, episode_successes):
                family_name = DOOR_FAMILY_NAMES[int(family_id)]
                self.completed_successes_by_family[family_name].append(float(value))
            # Record whether each COMPLETED rollout had any x5-arm / franka-box door collision.
            self.completed_x5_collisions.extend(
                self.episode_x5_collided[done_mask].to(dtype=torch.float32).detach().cpu().tolist()
            )
            self.completed_franka_box_collisions.extend(
                self.episode_franka_box_collided[done_mask].to(dtype=torch.float32).detach().cpu().tolist()
            )

        success_rate = self._mean_completed_metric(self.completed_successes)
        if success_rate is not None:
            self.extras["success/success_rate"] = success_rate
        # Fraction of completed rollouts that experienced ANY x5-arm / franka-box door collision.
        x5_collision_rate = self._mean_completed_metric(self.completed_x5_collisions)
        if x5_collision_rate is not None:
            self.extras["fail/x5_collision_episode_rate"] = x5_collision_rate
        franka_box_collision_rate = self._mean_completed_metric(self.completed_franka_box_collisions)
        if franka_box_collision_rate is not None:
            self.extras["fail/franka_box_collision_episode_rate"] = franka_box_collision_rate

        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            family_success_rate = self._mean_completed_metric(self.completed_successes_by_family[family_name])
            if family_success_rate is not None:
                self.extras[f"success/family_success_rate/{family_name}"] = family_success_rate

    def _log_dr_metrics(self):
        step_count = self._get_curriculum_step_count()
        if step_count % self._dr_metrics_interval != 0:
            return

        progress = self._compute_curriculum_progress(self.adr_reset_progress_total)
        scheduled_increment = int(progress * self.cfg.num_adr_increments)
        scheduled_increment = max(self.cfg.starting_adr_increments, scheduled_increment)
        scheduled_increment = min(self.cfg.num_adr_increments, scheduled_increment)

        robot_stiffness = self._current_event_param(
            "robot_joint_stiffness_and_damping", "stiffness_distribution_params"
        )
        robot_damping = self._current_event_param(
            "robot_joint_stiffness_and_damping", "damping_distribution_params"
        )
        robot_finger_armature = self._current_event_param(
            "robot_finger_armature", "armature_distribution_params"
        )
        board_stiffness = self._current_event_param(
            "door_board_joint_stiffness_and_damping", "stiffness_distribution_params"
        )
        board_damping = self._current_event_param(
            "door_board_joint_stiffness_and_damping", "damping_distribution_params"
        )
        hinge_stiffness = self._current_event_param(
            "door_hinge_joint_stiffness_and_damping", "stiffness_distribution_params"
        )
        hinge_damping = self._current_event_param(
            "door_hinge_joint_stiffness_and_damping", "damping_distribution_params"
        )
        board_friction = self._current_event_param(
            "door_board_joint_friction", "friction_distribution_params"
        )
        handle_effort_min, handle_effort_max = self._current_door_handle_effort_limit_range()
        panel_effort_min, panel_effort_max = self._current_door_panel_effort_limit_range()
        latch_min, latch_max = self._current_door_latch_threshold_range()

        self.extras["dr/increment"] = float(self.dooropening_adr.increment_counter)
        self.extras["dr/robot_stiffness_min"] = float(robot_stiffness[0])
        self.extras["dr/robot_stiffness_max"] = float(robot_stiffness[1])
        self.extras["dr/robot_damping_min"] = float(robot_damping[0])
        self.extras["dr/robot_damping_max"] = float(robot_damping[1])
        self.extras["dr/robot_finger_armature_min"] = float(robot_finger_armature[0])
        self.extras["dr/robot_finger_armature_max"] = float(robot_finger_armature[1])
        self.extras["dr/door_board_stiffness_min"] = float(board_stiffness[0])
        self.extras["dr/door_board_stiffness_max"] = float(board_stiffness[1])
        self.extras["dr/door_board_damping_min"] = float(board_damping[0])
        self.extras["dr/door_board_damping_max"] = float(board_damping[1])
        self.extras["dr/door_hinge_stiffness_min"] = float(hinge_stiffness[0])
        self.extras["dr/door_hinge_stiffness_max"] = float(hinge_stiffness[1])
        self.extras["dr/door_hinge_damping_min"] = float(hinge_damping[0])
        self.extras["dr/door_hinge_damping_max"] = float(hinge_damping[1])
        self.extras["dr/door_handle_effort_limit_min"] = float(handle_effort_min)
        self.extras["dr/door_handle_effort_limit_max"] = float(handle_effort_max)
        self.extras["dr/door_board_friction_min"] = float(board_friction[0])
        self.extras["dr/door_board_friction_max"] = float(board_friction[1])
        self.extras["dr/door_panel_effort_limit_min"] = float(panel_effort_min)
        self.extras["dr/door_panel_effort_limit_max"] = float(panel_effort_max)
        self.extras["dr/door_latch_threshold_min"] = float(latch_min)
        self.extras["dr/door_latch_threshold_max"] = float(latch_max)

        self.extras["dr_limit/spawn_arm_joint_pos_noise"] = self._current_custom_param("robot_spawn", "arm_joint_pos_noise")
        self.extras["dr_limit/spawn_finger_joint_pos_noise"] = self._current_custom_param(
            "robot_spawn", "finger_joint_pos_noise"
        )
        self.extras["dr_limit/obs_arm_joint_pos_noise"] = self._current_custom_param(
            "robot_state_noise", "arm_joint_pos_noise"
        )
        self.extras["dr_limit/obs_finger_joint_pos_noise"] = self._current_custom_param(
            "robot_state_noise", "finger_joint_pos_noise"
        )
        self.extras["dr_limit/obs_arm_joint_vel_noise"] = self._current_custom_param(
            "robot_state_noise", "arm_joint_vel_noise"
        )
        self.extras["dr_limit/obs_finger_joint_vel_noise"] = self._current_custom_param(
            "robot_state_noise", "finger_joint_vel_noise"
        )
        self.extras["dr_limit/door_handle_effort_limit_min"] = handle_effort_min
        self.extras["dr_limit/door_handle_effort_limit_max"] = handle_effort_max
        action_latency_min, action_latency_max = self._current_action_latency_bounds()
        self.extras["dr_limit/action_latency_steps_min"] = float(action_latency_min)
        self.extras["dr_limit/action_latency_steps_max"] = float(action_latency_max)
        self.extras["dr_limit/action_latency_steps_mean"] = self._action_latency_buf.float().mean().item()

        if not self._log_verbose_dr_metrics:
            return

        board_nominal_stiffness = self._door_nominal_joint_stiffness[:, self._door_board_joint_idx]
        board_nominal_damping = self._door_nominal_joint_damping[:, self._door_board_joint_idx]
        hinge_nominal_stiffness = self._door_nominal_joint_stiffness[:, self._door_hinge_joint_idx]
        hinge_nominal_damping = self._door_nominal_joint_damping[:, self._door_hinge_joint_idx]

        # self.extras["dr_sample/spawn_arm_joint_pos_noise_mean"] = self.robot_spawn_noise_widths["arm_joint_pos_noise"].mean().item()
        # self.extras["dr_sample/spawn_finger_joint_pos_noise_mean"] = self.robot_spawn_noise_widths[
        #     "finger_joint_pos_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_arm_joint_pos_noise_mean"] = self.robot_state_noise_widths[
        #     "arm_joint_pos_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_finger_joint_pos_noise_mean"] = self.robot_state_noise_widths[
        #     "finger_joint_pos_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_arm_joint_vel_noise_mean"] = self.robot_state_noise_widths[
        #     "arm_joint_vel_noise"
        # ].mean().item()
        # self.extras["dr_sample/obs_finger_joint_vel_noise_mean"] = self.robot_state_noise_widths[
        #     "finger_joint_vel_noise"
        # ].mean().item()
        self.extras["dr_sample/door_board_stiffness_mean"] = board_nominal_stiffness.mean().item()
        self.extras["dr_sample/door_board_stiffness_min"] = board_nominal_stiffness.min().item()
        self.extras["dr_sample/door_board_stiffness_max"] = board_nominal_stiffness.max().item()
        self.extras["dr_sample/door_board_damping_mean"] = board_nominal_damping.mean().item()
        self.extras["dr_sample/door_board_damping_min"] = board_nominal_damping.min().item()
        self.extras["dr_sample/door_board_damping_max"] = board_nominal_damping.max().item()
        self.extras["dr_sample/door_hinge_stiffness_mean"] = hinge_nominal_stiffness.mean().item()
        self.extras["dr_sample/door_hinge_stiffness_min"] = hinge_nominal_stiffness.min().item()
        self.extras["dr_sample/door_hinge_stiffness_max"] = hinge_nominal_stiffness.max().item()
        self.extras["dr_sample/door_hinge_damping_mean"] = hinge_nominal_damping.mean().item()
        self.extras["dr_sample/door_hinge_damping_min"] = hinge_nominal_damping.min().item()
        self.extras["dr_sample/door_hinge_damping_max"] = hinge_nominal_damping.max().item()
        self.extras["dr_sample/door_handle_effort_limit_mean"] = self._door_handle_effort_limits.mean().item()
        self.extras["dr_sample/door_handle_effort_limit_min"] = self._door_handle_effort_limits.min().item()
        self.extras["dr_sample/door_handle_effort_limit_max"] = self._door_handle_effort_limits.max().item()
        # Actually-applied panel (joint_1) Coulomb friction coefficient in the sim -- proves the
        # door_board_joint_friction EventTerm is taking effect (nonzero once ADR ramps in).
        board_applied_friction = self.door.data.joint_friction_coeff[:, self._door_board_joint_idx]
        self.extras["dr_sample/door_board_friction_mean"] = board_applied_friction.mean().item()
        self.extras["dr_sample/door_board_friction_min"] = board_applied_friction.min().item()
        self.extras["dr_sample/door_board_friction_max"] = board_applied_friction.max().item()
        # Per-env panel effort-limit cap actually handed to edit_door_articulation each step.
        self.extras["dr_sample/door_panel_effort_limit_mean"] = self._door_panel_effort_limits.mean().item()
        self.extras["dr_sample/door_panel_effort_limit_min"] = self._door_panel_effort_limits.min().item()
        self.extras["dr_sample/door_panel_effort_limit_max"] = self._door_panel_effort_limits.max().item()
        self.extras["dr_sample/door_latch_threshold_mean"] = self._door_latch_thresholds.mean().item()
        self.extras["dr_sample/door_latch_threshold_min"] = self._door_latch_thresholds.min().item()
        self.extras["dr_sample/door_latch_threshold_max"] = self._door_latch_thresholds.max().item()

    def _update_adr_ranges(self):
        if not self._adr_enabled:
            return

        progress = self._compute_curriculum_progress(self.adr_reset_progress_total)
        target_increment = int(progress * self.cfg.num_adr_increments)
        target_increment = max(self.cfg.starting_adr_increments, target_increment)
        target_increment = min(self.cfg.num_adr_increments, target_increment)
        if target_increment != self.dooropening_adr.increment_counter:
            self.dooropening_adr.set_num_increments(target_increment)

    def _sample_reset_randomization(self, env_ids: torch.Tensor):
        num_ids = len(env_ids)

        # Sample per-env widths and biases once at reset, then reuse them during the whole episode.
        for key in self.robot_spawn_noise_widths:
            value = self._current_custom_param("robot_spawn", key)
            self.robot_spawn_noise_widths[key][env_ids, 0] = value * torch.rand(num_ids, device=self.device)

        for key in self.robot_state_noise_widths:
            value = self._current_custom_param("robot_state_noise", key)
            self.robot_state_noise_widths[key][env_ids, 0] = value * torch.rand(num_ids, device=self.device)

        for key in self.robot_state_biases:
            value = self._current_custom_param("robot_state_noise", key)
            width = value * torch.rand(num_ids, device=self.device)
            self.robot_state_biases[key][env_ids, 0] = width * (torch.rand(num_ids, device=self.device) - 0.5)

        for key in self.door_state_noise_widths:
            value = self._current_custom_param("door_state_noise", key)
            self.door_state_noise_widths[key][env_ids, 0] = value * torch.rand(num_ids, device=self.device)

        for key in self.door_state_biases:
            value = self._current_custom_param("door_state_noise", key)
            width = value * torch.rand(num_ids, device=self.device)
            self.door_state_biases[key][env_ids, 0] = width * (torch.rand(num_ids, device=self.device) - 0.5)

        for key in self.student_joint_pos_noise_widths:
            self.student_joint_pos_noise_widths[key][env_ids, 0] = self._custom_param_upper_limit(
                "robot_state_noise", key
            )

        for key in self.student_joint_pos_biases:
            bias_limit = self._custom_param_upper_limit("robot_state_noise", key)
            self.student_joint_pos_biases[key][env_ids, 0] = bias_limit * (
                torch.rand(num_ids, device=self.device) - 0.5
            )

        min_lag, max_lag = self._current_action_latency_bounds()
        self._action_latency_buf[env_ids] = torch.randint(
            min_lag, max_lag + 1, (num_ids,), device=self.device
        )

    def _uniform_noise_from_buffers(
        self,
        values: torch.Tensor,
        width_buffers: dict[str, torch.Tensor],
        width_key: str,
        bias_buffers: dict[str, torch.Tensor] | None = None,
        bias_key: str | None = None,
    ):
        width = width_buffers[width_key]
        while width.dim() < values.dim():
            width = width.unsqueeze(-1)
        noise = width * 2.0 * (torch.rand_like(values) - 0.5)
        if bias_key is None:
            return values + noise
        if bias_buffers is None:
            raise RuntimeError("Bias buffers must be provided when bias_key is set.")
        bias = bias_buffers[bias_key]
        while bias.dim() < values.dim():
            bias = bias.unsqueeze(-1)
        return values + noise + bias

    def _uniform_noise_like(self, values: torch.Tensor, width_key: str, bias_key: str | None = None):
        return self._uniform_noise_from_buffers(
            values,
            width_buffers=self.robot_state_noise_widths,
            width_key=width_key,
            bias_buffers=self.robot_state_biases,
            bias_key=bias_key,
        )

    def _uniform_door_noise_like(self, values: torch.Tensor, width_key: str, bias_key: str | None = None):
        return self._uniform_noise_from_buffers(
            values,
            width_buffers=self.door_state_noise_widths,
            width_key=width_key,
            bias_buffers=self.door_state_biases,
            bias_key=bias_key,
        )

    @staticmethod
    def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
        quat_norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)
        return quat / quat_norm

    def get_student_joint_pos_obs(self, use_noise: bool = False) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos
        if not use_noise:
            return joint_pos

        student_joint_pos = joint_pos.clone()
        student_joint_pos[:, self._robot_base_rot_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_base_rot_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="base_rot_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="base_rot_joint_pos_bias",
        )
        student_joint_pos[:, self._robot_base_xy_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_base_xy_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="base_xy_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="base_xy_joint_pos_bias",
        )
        student_joint_pos[:, self._robot_arm_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_arm_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="arm_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="arm_joint_pos_bias",
        )
        student_joint_pos[:, self._robot_arx_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_arx_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="arm_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="arm_joint_pos_bias",
        )
        student_joint_pos[:, self._robot_finger_dof_idx] = self._uniform_noise_from_buffers(
            student_joint_pos[:, self._robot_finger_dof_idx],
            width_buffers=self.student_joint_pos_noise_widths,
            width_key="finger_joint_pos_noise",
            bias_buffers=self.student_joint_pos_biases,
            bias_key="finger_joint_pos_bias",
        )
        return student_joint_pos

    def get_student_base_joint_vel_obs(self, use_noise: bool = False) -> torch.Tensor:
        base_joint_vel = self.robot.data.joint_vel[:, self._robot_base_dof_idx]
        if not use_noise:
            return base_joint_vel

        noisy_base_joint_vel = base_joint_vel.clone()
        noisy_base_joint_vel[:, self._target_base_rot_slice] = self._uniform_noise_from_buffers(
            noisy_base_joint_vel[:, self._target_base_rot_slice],
            width_buffers=self.robot_state_noise_widths,
            width_key="base_rot_joint_vel_noise",
            bias_buffers=self.robot_state_biases,
            bias_key="base_rot_joint_vel_bias",
        )
        noisy_base_joint_vel[:, self._target_base_xy_slice] = self._uniform_noise_from_buffers(
            noisy_base_joint_vel[:, self._target_base_xy_slice],
            width_buffers=self.robot_state_noise_widths,
            width_key="base_xy_joint_vel_noise",
            bias_buffers=self.robot_state_biases,
            bias_key="base_xy_joint_vel_bias",
        )
        return noisy_base_joint_vel

    def _apply_spawn_noise(self, env_ids: torch.Tensor):
        # Reset disturbance is applied around the reference motion state, with separate scales for base, arm, and fingers.
        base_xy_noise = self.robot_spawn_noise_widths["base_xy_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), len(self._robot_base_xy_dof_idx)), device=self.device) - 1.0
        )
        base_rot_noise = self.robot_spawn_noise_widths["base_rot_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), 1), device=self.device) - 1.0
        )
        arm_noise = self.robot_spawn_noise_widths["arm_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), len(self._robot_arm_dof_idx)), device=self.device) - 1.0
        )
        finger_noise = self.robot_spawn_noise_widths["finger_joint_pos_noise"][env_ids] * (
            2.0 * torch.rand((len(env_ids), len(self._robot_finger_dof_idx)), device=self.device) - 1.0
        )

        self.joint_pos[env_ids[:, None], self._robot_base_xy_dof_idx[None, :]] += base_xy_noise
        self.joint_pos[env_ids[:, None], self._robot_base_rot_dof_idx[None, :]] += base_rot_noise
        self.joint_pos[env_ids[:, None], self._robot_arm_dof_idx[None, :]] += arm_noise
        if not self.fixed_arx_pose:
            arx_noise = self.robot_spawn_noise_widths["arm_joint_pos_noise"][env_ids] * (
                2.0 * torch.rand((len(env_ids), len(self._robot_arx_dof_idx)), device=self.device) - 1.0
            )
            self.joint_pos[env_ids[:, None], self._robot_arx_dof_idx[None, :]] += arx_noise
        self.joint_pos[env_ids[:, None], self._robot_finger_dof_idx[None, :]] += finger_noise

        self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]] = torch.clamp(
            self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]],
            self.robot_dof_lower_limits[None, :],
            self.robot_dof_upper_limits[None, :],
        )

    def _get_policy_target_noise(self):
        target_noise = torch.zeros_like(self.robot_dof_targets)
        base_xy_target_noise = self._current_custom_param("pd_targets", "base_xy_target_noise")
        base_rot_target_noise = self._current_custom_param("pd_targets", "base_rot_target_noise")
        arm_target_noise = self._current_custom_param("pd_targets", "arm_target_noise")
        finger_target_noise = self._current_custom_param("pd_targets", "finger_target_noise")

        target_noise[:, self._target_base_xy_slice] = base_xy_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_base_xy_slice]) - 1.0
        )
        target_noise[:, self._target_base_rot_slice] = base_rot_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_base_rot_slice]) - 1.0
        )
        target_noise[:, self._target_arm_slice] = arm_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_arm_slice]) - 1.0
        )
        target_noise[:, self._target_finger_slice] = finger_target_noise * (
            2.0 * torch.rand_like(target_noise[:, self._target_finger_slice]) - 1.0
        )
        return target_noise

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != self.num_policy_actions:
            raise RuntimeError(
                f"Expected policy action shape [N, {self.num_policy_actions}], got {tuple(actions.shape)}."
            )
        clamped_actions = actions.clamp(-1.0, 1.0)
        scaled_actions = torch.zeros(
            (actions.shape[0], self.num_robot_actions),
            device=actions.device,
            dtype=actions.dtype,
        )
        scaled_actions[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop] = (
            clamped_actions[:, self._policy_base_rot_slice.start : self._policy_base_xy_slice.stop]
            * self.cfg.base_action_scale
        )
        scaled_actions[:, self._target_arm_slice] = (
            clamped_actions[:, self._policy_arm_slice] * self.cfg.arm_action_scale
        )
        scaled_actions[:, self._target_finger_slice] = (
            clamped_actions[:, self._policy_finger_slice] * self.cfg.finger_action_scale
        )
        return scaled_actions

    def _pin_arx_targets_to_fixed_pose(self, target_tensor: torch.Tensor) -> torch.Tensor:
        if not self.fixed_arx_pose or self.num_arx_joints <= 0:
            return target_tensor
        pinned_targets = target_tensor.clone()
        pinned_targets[:, self._target_arx_slice] = self._robot_arx_tuck_joint_pos_target.to(target_tensor).unsqueeze(0)
        return pinned_targets

    def _enforce_fixed_arx_joint_state(self):
        if not self.fixed_arx_pose or self.num_arx_joints <= 0:
            return
        arx_joint_pos = self._robot_arx_tuck_joint_pos_target.to(self.joint_pos).unsqueeze(0).expand(self.num_envs, -1)
        arx_joint_vel = torch.zeros_like(arx_joint_pos)
        self.joint_pos[:, self._robot_arx_dof_idx] = arx_joint_pos
        self.joint_vel[:, self._robot_arx_dof_idx] = arx_joint_vel
        self.robot_dof_targets[:, self._target_arx_slice] = arx_joint_pos
        self.applied_robot_dof_targets[:, self._target_arx_slice] = arx_joint_pos
        self.robot.write_joint_state_to_sim(arx_joint_pos, arx_joint_vel, joint_ids=self._robot_arx_dof_idx)

    def _enforce_fixed_camera_joint_state(self):
        # Hold the non-arx camera (x5) joints rigidly at their default pose so the wrist-mounted
        # camera viewpoint stays steady. These joints are outside the policy action/observation
        # accounting, so this never touches joint actuation commanding for the policy.
        if not self.fixed_arx_pose or self.num_extra_camera_joints <= 0:
            return
        cam_joint_pos = self._robot_extra_camera_default_pos.to(self.joint_pos).unsqueeze(0).expand(self.num_envs, -1)
        cam_joint_vel = torch.zeros_like(cam_joint_pos)
        self.joint_pos[:, self._robot_extra_camera_dof_idx] = cam_joint_pos
        self.joint_vel[:, self._robot_extra_camera_dof_idx] = cam_joint_vel
        self.robot.write_joint_state_to_sim(cam_joint_pos, cam_joint_vel, joint_ids=self._robot_extra_camera_dof_idx)
        self.robot.set_joint_position_target(cam_joint_pos, joint_ids=self._robot_extra_camera_dof_idx)

    def _apply_action_latency(self, new_targets: torch.Tensor) -> torch.Tensor:
        """Return the target to apply this step after a per-env action delay, then record the new one.

        ``_action_target_history[:, k]`` holds the target computed ``k + 1`` steps ago, so the target
        applied for an env with latency ``L`` is ``_action_target_history[:, L - 1]``. After reading it
        out, the freshly computed ``new_targets`` is pushed to the front of the history and the oldest
        entry is dropped.
        """
        lag_idx = (self._action_latency_buf - 1).clamp(0, self._max_action_latency - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)
        applied = self._action_target_history[env_idx, lag_idx]
        self._action_target_history = torch.cat(
            [new_targets.unsqueeze(1), self._action_target_history[:, :-1]], dim=1
        )
        return applied

    def _pre_physics_step(self, actions: torch.Tensor):
        # delta actions
        self.scaled_actions = self._scale_actions(actions)
        targets = self.robot_dof_targets + self.dt * self.scaled_actions
        targets = self._pin_arx_targets_to_fixed_pose(targets)
        # NOTE: no explicit contact-sensor update() here. This runs BEFORE the physics step, so it
        # could only ever refresh last step's contacts, and scene.update() (called by
        # DirectRLEnv.step on every decimation substep) re-marks them outdated immediately after.
        # The reward/termination code reads `sensor.data.*` post-step, which pulls fresh buffers on
        # demand. Updating here just bought an extra PhysX readback + CUDA sync per sensor per step.
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.robot_dof_targets[:] = self._pin_arx_targets_to_fixed_pose(self.robot_dof_targets)
        # Action latency: apply the target the policy produced `_action_latency_buf` env steps ago
        # (sampled per-env at reset, ramping from 1 step up to the ADR max). This replaces the old
        # EMA lag filter with a true delayed-action model of the real robot's control pipeline.
        self.applied_robot_dof_targets[:] = self._apply_action_latency(self.robot_dof_targets)
        self.applied_robot_dof_targets[:] = self._pin_arx_targets_to_fixed_pose(self.applied_robot_dof_targets)
        # Advance the reference once per RL/env step. IsaacLab will call _apply_action()
        # decimation times, so stepping here keeps replay duration tied to env dt instead
        # of being multiplied by decimation again.
        if self.ref_motion_lib is not None:
            self.ref_motion_lib.step()


    def _apply_action(self):
        # self.ref_motion_lib.step()
        # joint_pos = self.robot.data.default_joint_pos.clone()
        # ref_robot_joint_pos = self.ref_motion_lib.get_robot_joint_pos().to(joint_pos)
        # joint_pos[:, self._robot_dof_idx] = ref_robot_joint_pos[:, self.ref_robot_dof_idx]
        # self.robot.write_joint_position_to_sim(joint_pos)
        # self.robot_dof_targets[:] = joint_pos[:, self._robot_dof_idx]
        # self.applied_robot_dof_targets[:] = self.robot_dof_targets
        # door_pos = self.door.data.joint_pos.clone()
        # door_pos[:] = self.ref_motion_lib.get_door_joint_pos()
        # self.door.write_joint_position_to_sim(door_pos)

        edit_door_articulation(
            self.door,
            nominal_joint_stiffness=self._door_nominal_joint_stiffness,
            nominal_joint_damping=self._door_nominal_joint_damping,
            # Per-env panel-swing restoring-torque cap (sampled at reset, ADR-ramped) so the high
            # stiffness doesn't make the door impossibly heavy at large angles; latch lock stays 1e6.
            unlocked_panel_effort_limit=self._door_panel_effort_limits,
            # Per-env handle unlatch angle threshold (radians), sampled at reset, ADR-ramped.
            hinge_range=self._door_latch_thresholds,
        )
        # applied_robot_dof_targets already carries the lag-filtered target from _pre_physics_step.
        # Add per-substep controller noise on top without polluting the lag history.
        applied_targets = self.applied_robot_dof_targets.clone()
        applied_targets += self._get_policy_target_noise()
        applied_targets = self._pin_arx_targets_to_fixed_pose(applied_targets)
        applied_targets = torch.clamp(applied_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        applied_targets = self._pin_arx_targets_to_fixed_pose(applied_targets)
        self._enforce_fixed_arx_joint_state()
        self._enforce_fixed_camera_joint_state()
        self.robot.set_joint_position_target(applied_targets, joint_ids=self._robot_dof_idx)

    def _build_observations(
        self,
        record_viser: bool = True,
    ) -> dict:
        self._get_intermediate_values()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        door_to_base_link_pos = world_to_local(
            self.door_link_pos,
            self.robot_base_body_pos,
            self.robot_base_body_quat,
        ).reshape(self.num_envs, 1, -1)
        door_twist_in_robot_base_frame = world_to_local(
            self.ref_door_body_pos_twist,
            self.robot_base_body_pos,
            self.robot_base_body_quat,
        ).reshape(self.num_envs, 1, -1)
        robot_key_body_pos, robot_key_body_euler, base_lin_vel_local, base_ang_vel_local = self.transform_key_bodies_to_base_frame(
            self.robot_key_body_pos,
            self.robot_key_body_quat,
            self.robot_body_lin_vel,
            self.robot_body_ang_vel,
            self._robot_base_body_link_idx,
        )
        key_pos_err = (self.ref_robot_key_body_pos).to(self.robot_key_body_pos) - self.robot_key_body_pos
        key_pos_err = world_to_local(key_pos_err, None, self.robot_base_body_quat).reshape(self.num_envs, 1, -1)

        clean_joint_pos = self.joint_pos[:, self._robot_dof_idx]
        clean_joint_vel = self.joint_vel[:, self._robot_dof_idx]
        policy_joint_pos = clean_joint_pos.clone()
        policy_joint_vel = clean_joint_vel.clone()
        policy_robot_key_body_pos = self._uniform_noise_like(
            self.robot_key_body_pos.clone(),
            "key_body_pos_noise",
            "key_body_pos_bias",
        )
        policy_robot_key_body_quat = self._normalize_quat(
            self._uniform_noise_like(
                self.robot_key_body_quat.clone(),
                "key_body_quat_noise",
                "key_body_quat_bias",
            )
        )
        policy_robot_body_lin_vel = self._uniform_noise_like(
            self.robot_body_lin_vel.clone(),
            "body_lin_vel_noise",
            "body_lin_vel_bias",
        )
        policy_robot_body_ang_vel = self._uniform_noise_like(
            self.robot_body_ang_vel.clone(),
            "body_ang_vel_noise",
            "body_ang_vel_bias",
        )

        policy_joint_pos[:, self._target_base_rot_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_base_rot_slice], "base_rot_joint_pos_noise", "base_rot_joint_pos_bias"
        )
        policy_joint_pos[:, self._target_base_xy_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_base_xy_slice], "base_xy_joint_pos_noise", "base_xy_joint_pos_bias"
        )
        policy_joint_pos[:, self._target_arm_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_arm_slice], "arm_joint_pos_noise", "arm_joint_pos_bias"
        )
        policy_joint_pos[:, self._target_finger_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_finger_slice], "finger_joint_pos_noise", "finger_joint_pos_bias"
        )
        policy_joint_pos[:, self._target_arx_slice] = self._uniform_noise_like(
            policy_joint_pos[:, self._target_arx_slice], "arm_joint_pos_noise", "arm_joint_pos_bias"
        )

        policy_joint_vel[:, self._target_base_rot_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_base_rot_slice], "base_rot_joint_vel_noise", "base_rot_joint_vel_bias"
        )
        policy_joint_vel[:, self._target_base_xy_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_base_xy_slice], "base_xy_joint_vel_noise", "base_xy_joint_vel_bias"
        )
        policy_joint_vel[:, self._target_arm_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_arm_slice], "arm_joint_vel_noise", "arm_joint_vel_bias"
        )
        policy_joint_vel[:, self._target_finger_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_finger_slice], "finger_joint_vel_noise", "finger_joint_vel_bias"
        )
        policy_joint_vel[:, self._target_arx_slice] = self._uniform_noise_like(
            policy_joint_vel[:, self._target_arx_slice], "arm_joint_vel_noise", "arm_joint_vel_bias"
        )
        policy_robot_key_body_pos_local, policy_robot_key_body_euler, policy_base_lin_vel_local, policy_base_ang_vel_local = self.transform_key_bodies_to_base_frame(
            policy_robot_key_body_pos,
            policy_robot_key_body_quat,
            policy_robot_body_lin_vel,
            policy_robot_body_ang_vel,
            self._robot_base_body_link_idx,
        )
        policy_robot_base_body_pos = policy_robot_key_body_pos[:, self._robot_base_id_in_key_body_idx]
        policy_robot_base_body_quat = policy_robot_key_body_quat[:, self._robot_base_id_in_key_body_idx]
        policy_key_pos_err = (self.ref_robot_key_body_pos).to(policy_robot_key_body_pos) - policy_robot_key_body_pos
        policy_key_pos_err = world_to_local(policy_key_pos_err, None, policy_robot_base_body_quat).reshape(
            self.num_envs, 1, -1
        )
        policy_door_to_base_link_pos = world_to_local(
            self.door_link_pos,
            policy_robot_base_body_pos,
            policy_robot_base_body_quat,
        ).reshape(self.num_envs, 1, -1)
        policy_door_to_base_link_pos = self._uniform_door_noise_like(
            policy_door_to_base_link_pos,
            "door_pos_noise",
            "door_pos_bias",
        )
        policy_door_joint_pos = self._uniform_door_noise_like(
            self.door_joint_pos[:, self._door_joint_idx].clone(),
            "door_joint_pos_noise",
            "door_joint_pos_bias",
        ).unsqueeze(dim=1)

        # twist_obs = torch.cat(
        #     (
        #         self.ref_robot_key_body_pos_twist.reshape(self.num_envs, 1, -1),
        #         self.ref_robot_key_body_quat_twist.reshape(self.num_envs, 1, -1),
        #         self.ref_door_joint_pos_twist.reshape(self.num_envs, 1, -1),
        #         (
        #             self.ref_robot_base_joint_pos_twist
        #             - clean_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop].unsqueeze(dim=1)
        #         ).reshape(self.num_envs, 1, -1),
        #         (
        #             self.ref_robot_arm_joint_pos_twist - clean_joint_pos[:, self._target_arm_slice].unsqueeze(dim=1)
        #         ).reshape(self.num_envs, 1, -1),
        #         (
        #             self.ref_robot_arx_joint_pos_twist - clean_joint_pos[:, self._target_arx_slice].unsqueeze(dim=1)
        #         ).reshape(self.num_envs, 1, -1),
        #         door_twist_in_robot_base_frame,
        #     ),
        #     dim=-1,
        # )

        # Reference joint-angle error fed to the policy (DISABLED -- found not very useful; kept
        # commented for easy re-enable). Must stay in sync with the critic term and the
        # joint_reference_error_observation_space in the cfg.
        # policy_joint_ref_err = torch.cat(
        #     (
        #         policy_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop]
        #         - self.ref_robot_base_joint_pos.to(policy_joint_pos),
        #         policy_joint_pos[:, self._target_arm_slice] - self.ref_robot_arm_joint_pos.to(policy_joint_pos),
        #         policy_joint_pos[:, self._target_finger_slice] - self.ref_robot_finger_joint_pos.to(policy_joint_pos),
        #     ),
        #     dim=-1,
        # ).unsqueeze(dim=1)
        # policy_joint_ref_err = torch.cat(
        #     (
        #         clean_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop]
        #         - self.ref_robot_base_joint_pos.to(clean_joint_pos),
        #         clean_joint_pos[:, self._target_arm_slice] - self.ref_robot_arm_joint_pos.to(clean_joint_pos),
        #         clean_joint_pos[:, self._target_finger_slice] - self.ref_robot_finger_joint_pos.to(clean_joint_pos),
        #     ),
        #     dim=-1,
        # ).unsqueeze(dim=1)

        policy_obs = torch.cat(
            (
                policy_joint_pos.unsqueeze(dim=1),
                policy_joint_vel.unsqueeze(dim=1),
                self.robot_dof_targets.unsqueeze(dim = 1),
                # policy_joint_ref_err,
                policy_key_pos_err,
                policy_robot_key_body_pos_local.reshape(self.num_envs, 1, -1),
                policy_robot_key_body_euler.reshape(self.num_envs, 1, -1),
                policy_base_lin_vel_local.reshape(self.num_envs, 1, -1),
                policy_base_ang_vel_local.reshape(self.num_envs, 1, -1),
                policy_door_to_base_link_pos,
                policy_door_joint_pos,
                self.ref_door_joint_pos[:, self._door_joint_idx].to(self.door_joint_pos).unsqueeze(dim = 1),
                self.ref_robot_arx_joint_pos.to(self.robot_arx_joint_pos).unsqueeze(dim=1),
                # twist_obs,
            ),
            dim=-1,
        )

        # Reference joint-angle error fed to the critic (DISABLED -- found not very useful; kept
        # commented for easy re-enable). Must stay in sync with the policy term and the cfg.
        # clean_joint_ref_err = torch.cat(
        #     (
        #         clean_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop]
        #         - self.ref_robot_base_joint_pos.to(clean_joint_pos),
        #         clean_joint_pos[:, self._target_arm_slice] - self.ref_robot_arm_joint_pos.to(clean_joint_pos),
        #         clean_joint_pos[:, self._target_finger_slice] - self.ref_robot_finger_joint_pos.to(clean_joint_pos),
        #     ),
        #     dim=-1,
        # ).unsqueeze(dim=1)

        critic_obs = torch.cat(
            (
                clean_joint_pos.unsqueeze(dim=1),
                clean_joint_vel.unsqueeze(dim=1),
                self.robot_dof_targets.unsqueeze(dim=1),
                # clean_joint_ref_err,
                key_pos_err,
                robot_key_body_pos.reshape(self.num_envs, 1, -1),
                robot_key_body_euler.reshape(self.num_envs, 1, -1),
                base_lin_vel_local.reshape(self.num_envs, 1, -1),
                base_ang_vel_local.reshape(self.num_envs, 1, -1),
                door_to_base_link_pos,
                self.door_joint_pos[:, self._door_joint_idx].unsqueeze(dim = 1),
                self.ref_door_joint_pos[:, self._door_joint_idx].to(self.door_joint_pos).unsqueeze(dim = 1),
                self.ref_robot_arx_joint_pos.to(self.robot_arx_joint_pos).unsqueeze(dim=1),
                # twist_obs,
            ),
            dim=-1,
        )

        # The actor sees noisy deployment-like inputs; the critic keeps the clean privileged state.
        policy_obs = policy_obs.squeeze(1)
        critic_obs = critic_obs.squeeze(1)

        if record_viser:
            self._maybe_record_viser_pointcloud()

        observations = {"policy": policy_obs, "critic": critic_obs}
        return observations

    def _get_observations(self) -> dict:
        return self._build_observations()


    def normalize_to_base_frame(
        self, 
        base_pos: torch.Tensor,       # (N, 3) Current base position
        base_quat: torch.Tensor,      # (N, 4) Current base orientation
        current_pos: torch.Tensor,    # (N, B, 3) Current key link positions
        current_quat: torch.Tensor,   # (N, B, 4) Current key link orientations
        target_pos: torch.Tensor,     # (N, T, B, 3) Future target positions (Twist)
        target_quat: torch.Tensor     # (N, T, B, 4) Future target orientations (Twist)
    ):
        """
        Calculates the positional and rotational errors between future targets 
        and current link states, expressed in the robot's base frame.
        """
        N, T, B, _ = target_pos.shape

        # ==========================================
        # 1. POSITION ERROR (Expressed in Base Frame)
        # ==========================================
        # World-space error: P_err_world = P_target - P_current
        curr_pos_exp = current_pos.unsqueeze(1)            # (N, 1, B, 3)
        pos_err_world = target_pos - curr_pos_exp          # (N, T, B, 3)

        # Rotate the world-space error vector into the base frame
        base_quat_inv = quat_conjugate(base_quat).view(N, 1, 1, 4)
        base_quat_inv_exp = base_quat_inv.expand(N, T, B, 4)

        pos_err_base = quat_apply(
            base_quat_inv_exp.reshape(-1, 4), 
            pos_err_world.reshape(-1, 3)
        )
        pos_err_base = pos_err_base.view(N, T, B, 3)

        # ==========================================
        # 2. ORIENTATION ERROR
        # ==========================================
        # The relative rotation required to get from current_quat to target_quat.
        # Q_err = Q_current_inv * Q_target
        curr_quat_inv = quat_conjugate(current_quat).unsqueeze(1) # (N, 1, B, 4)
        curr_quat_inv_exp = curr_quat_inv.expand(N, T, B, 4)

        quat_err = quat_mul(
            curr_quat_inv_exp.reshape(-1, 4),
            target_quat.reshape(-1, 4)
        )
        quat_err = quat_err.view(N, T, B, 4)

        return pos_err_base, quat_err

    def transform_key_bodies_to_base_frame(
        self,
        robot_key_body_pos: torch.Tensor,       # (num_envs, num_bodies, 3)
        robot_key_body_quat: torch.Tensor,      # (num_envs, num_bodies, 4)
        robot_body_lin_vel: torch.Tensor,       # (num_envs, num_bodies, 3)
        robot_body_ang_vel: torch.Tensor,       # (num_envs, num_bodies, 3)
        robot_base_id_in_key_body_idx: int,     # index of tidybot_base_link
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform robot key body observations from World Frame to Robot Base Frame.
        
        Args:
            robot_key_body_pos: World frame positions of key bodies
            robot_key_body_quat: World frame orientations (quaternions, xyzw)
            robot_body_lin_vel: World frame linear velocities
            robot_body_ang_vel: World frame angular velocities
            robot_base_id_in_key_body_idx: Index of base link in key body list
        
        Returns:
            Tuple containing:
                - body_pos_rel: (num_envs, num_bodies, 3) Positions in Base Frame
                - body_euler_rel: (num_envs, num_bodies, 3) Orientations in Base Frame
                - base_lin_vel_local: (num_envs, 3) Base linear velocity in Base Frame
                - base_ang_vel_local: (num_envs, 3) Base angular velocity in Base Frame
        """
        N = robot_key_body_pos.shape[0]
        B = robot_key_body_pos.shape[1]

        # ---------------------------------------------------
        # 1. Extract base pose
        # ---------------------------------------------------
        base_pos = robot_key_body_pos[:, robot_base_id_in_key_body_idx, :]        # (N,3)
        base_quat = robot_key_body_quat[:, robot_base_id_in_key_body_idx, :]      # (N,4)

        base_quat_inv = quat_conjugate(base_quat)                        # (N,4)

        # ---------------------------------------------------
        # 2. Position transform
        # P_local = R^T (P_world - P_base)
        # ---------------------------------------------------
        pos_diff = robot_key_body_pos - base_pos.unsqueeze(1)      # (N,B,3)

        base_quat_inv_exp = base_quat_inv.unsqueeze(1).expand(-1, B, -1)
        body_pos_rel = quat_apply(base_quat_inv_exp, pos_diff)

        # ---------------------------------------------------
        # 3. Orientation transform
        # Q_local = Q_base_inv * Q_body
        # ---------------------------------------------------
        body_quat_rel = quat_mul(base_quat_inv_exp, robot_key_body_quat)

        # ---------------------------------------------------
        # 4. Base velocities (express in base frame)
        # ---------------------------------------------------
        base_lin_vel_world = robot_body_lin_vel[:, robot_base_id_in_key_body_idx, :]
        base_ang_vel_world = robot_body_ang_vel[:, robot_base_id_in_key_body_idx, :]

        base_lin_vel_local = quat_apply(base_quat_inv, base_lin_vel_world)
        base_ang_vel_local = quat_apply(base_quat_inv, base_ang_vel_world)

        # ---------------------------------------------------
        # 5. Remove base link from outputs
        # ---------------------------------------------------
        keep_mask = torch.ones(B, dtype=torch.bool, device=robot_key_body_pos.device)
        keep_mask[robot_base_id_in_key_body_idx] = False

        body_pos_rel = body_pos_rel[:, keep_mask, :]
        body_quat_rel = body_quat_rel[:, keep_mask, :]

        return (
            body_pos_rel,          # (N, B-1, 3)
            quat_to_6d(body_quat_rel),         # (N, B-1, 6)
            base_lin_vel_local,    # (N, 3)
            base_ang_vel_local,    # (N, 3)
        )

    def compute_door_keypoints(
        self,
    ) -> torch.Tensor:
        """
        Compute 5 keypoints:
            - 2 from joint 0 (handle offsets)
            - 3 from joint 1 (board offsets)

        Returns:
            keypoints_w: [B, 5, 3]
        """

        # ---- Joint 0 (handle) ----
        pos0 = self.door_link_pos[:, self.door_body_names.index("link_2"), :]         # [B, 3]
        quat0 = self.door_link_quat[:, self.door_body_names.index("link_2"), :]       # [B, 4]

        # reshape for batched quat_apply
        handle_offsets_flat = self.handle_offsets.reshape(-1, 3)           # [B*2, 3]
        quat0_rep = quat0.unsqueeze(1).repeat(1, 2, 1).reshape(-1, 4) # [B*2, 4]

        handle_rot = quat_apply(quat0_rep.float(), handle_offsets_flat.float())       # [B*2, 3]
        handle_rot = handle_rot.reshape(self.num_envs, 2, 3)

        handle_kpts = pos0.unsqueeze(1) + handle_rot                  # [B, 2, 3]

        # ---- Joint 1 (board) ----
        pos1 = self.door_link_pos[:, self.door_body_names.index("link_1"), :]
        quat1 = self.door_link_quat[:, self.door_body_names.index("link_1"), :]

        board_offsets_flat = self.board_offsets.reshape(-1, 3)             # [B*3, 3]
        quat1_rep = quat1.unsqueeze(1).repeat(1, 3, 1).reshape(-1, 4) # [B*3, 4]

        board_rot = quat_apply(quat1_rep.float(), board_offsets_flat.float())         # [B*3, 3]
        board_rot = board_rot.reshape(self.num_envs, 3, 3)

        board_kpts = pos1.unsqueeze(1) + board_rot                    # [B, 3, 3]

        # ---- Concatenate ----
        keypoints_w = torch.cat([handle_kpts, board_kpts], dim=1)     # [self.num_envs, 5, 3]

        return keypoints_w

    def get_handle_position_in_base_frame(self) -> torch.Tensor:
        """Return the door handle center position in the robot base frame.

        The handle geometry lives on door body ``link_2``. We approximate the
        handle center using the midpoint of the two precomputed handle offsets,
        then transform that point into the robot base frame.

        Returns:
            torch.Tensor: handle_pos_base with shape (num_envs, 3).
        """
        robot_base_pos_w = self.robot.data.body_pos_w[:, self._robot_base_body_link_idx]
        robot_base_quat_w = self.robot.data.body_quat_w[:, self._robot_base_body_link_idx]

        handle_body_local_idx = self.door_body_names.index("link_2")
        handle_body_idx = int(self._door_body_idx[handle_body_local_idx])
        handle_body_pos_w = self.door.data.body_pos_w[:, handle_body_idx]
        handle_body_quat_w = self.door.data.body_quat_w[:, handle_body_idx]

        handle_center_offset = self.handle_offsets.mean(dim=1)
        handle_center_pos_w = quat_apply(handle_body_quat_w.float(), handle_center_offset.float()) + handle_body_pos_w
        handle_center_pos_base = world_to_local(
            handle_center_pos_w.unsqueeze(1),
            robot_base_pos_w,
            robot_base_quat_w,
        ).squeeze(1)

        return handle_center_pos_base

    def get_closed_handle_position_in_base_frame(self) -> torch.Tensor:
        """Closed-door (joints=0) handle position expressed in the CURRENT robot base frame.

        Simulates a one-shot SAM3 detection of the closed handle in the world, re-expressed in the
        robot base frame as the base moves. The closed-handle position in the door "base" link frame
        is a precomputed URDF constant (``self.closed_handle_pos_door_base``); we compose it with the
        runtime door-base world pose (static, since fix_base) and transform into the robot base
        frame. No simulation capture needed. Shape (num_envs, 3).
        """
        door_base_pos_w = self.door.data.body_pos_w[:, self._door_base_link_idx]
        door_base_quat_w = self.door.data.body_quat_w[:, self._door_base_link_idx]
        handle_center_pos_w = (
            quat_apply(door_base_quat_w.float(), self.closed_handle_pos_door_base.float())
            + door_base_pos_w
        )

        robot_base_pos_w = self.robot.data.body_pos_w[:, self._robot_base_body_link_idx]
        robot_base_quat_w = self.robot.data.body_quat_w[:, self._robot_base_body_link_idx]
        return world_to_local(
            handle_center_pos_w.unsqueeze(1),
            robot_base_pos_w,
            robot_base_quat_w,
        ).squeeze(1)

    def _set_current_state_as_reference(self):
        """Populate reference tensors without loading demonstration trajectories."""

        ref_joint_pos = torch.zeros(
            (self.num_envs, len(FULL_JOINT_NAMES)),
            device=self.device,
            dtype=self.robot.data.joint_pos.dtype,
        )
        ref_joint_vel = torch.zeros_like(ref_joint_pos)
        ref_joint_pos[:, self.ref_robot_dof_idx] = self.robot.data.joint_pos[:, self._robot_dof_idx]
        ref_joint_vel[:, self.ref_robot_dof_idx] = self.robot.data.joint_vel[:, self._robot_dof_idx]

        self.ref_robot_key_body_pos = self.robot_key_body_pos
        self.ref_robot_key_body_quat = self.robot_key_body_quat
        self.ref_robot_reset_key_body_pos = self.robot_reset_key_body_pos
        self.ref_robot_joint_pos = ref_joint_pos
        self.ref_robot_base_joint_pos = ref_joint_pos[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos = ref_joint_pos[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_pos = ref_joint_pos[:, self.ref_finger_joint_idx]
        self.ref_robot_arx_joint_pos = ref_joint_pos[:, self.ref_arx_joint_idx]
        self.ref_joint_vel = ref_joint_vel
        self.ref_robot_base_joint_vel = ref_joint_vel[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_vel = ref_joint_vel[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_vel = ref_joint_vel[:, self.ref_finger_joint_idx]
        self.ref_robot_arx_joint_vel = ref_joint_vel[:, self.ref_arx_joint_idx]
        self.ref_door_joint_pos = self.door_joint_pos.clone()
        self.ref_hinge_contact_mask = torch.zeros(self.num_envs, device=self.device, dtype=self.door_joint_pos.dtype)
        self.ref_panel_contact_mask = torch.zeros(self.num_envs, device=self.device, dtype=self.door_joint_pos.dtype)
        self.ref_grasp_stage_mask = torch.zeros(self.num_envs, device=self.device, dtype=self.door_joint_pos.dtype)
        self.ref_robot_body_lin_vel = self.robot_body_lin_vel
        self.ref_robot_body_ang_vel = self.robot_body_ang_vel
        # Diagnostic contact-force buffers (populated each step in _get_rewards); pre-init so the
        # eval/play scripts can read them before the first reward computation.
        self.franka_arx_contact_force_norm = torch.zeros(self.num_envs, device=self.device)
        self.finger_panel_contact_force_norm = torch.zeros(self.num_envs, device=self.device)
        self.finger_handle_contact_force_norm = torch.zeros(self.num_envs, device=self.device)

        twist_len = len(self.twist_indices) if self.twist_indices is not None else 0
        twist_shape = (self.num_envs, twist_len)
        self.ref_robot_key_body_pos_twist = torch.zeros(
            (*twist_shape, len(self.ref_key_body_idx), 3),
            device=self.device,
            dtype=self.robot_key_body_pos.dtype,
        )
        self.ref_robot_key_body_quat_twist = torch.zeros(
            (*twist_shape, len(self.ref_key_body_idx), 6),
            device=self.device,
            dtype=self.robot_key_body_quat.dtype,
        )
        self.ref_robot_joint_pos_twist = ref_joint_pos.unsqueeze(1).expand(-1, twist_len, -1)
        self.ref_robot_base_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_arm_joint_idx]
        self.ref_robot_arx_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_arx_joint_idx]
        self.ref_door_joint_pos_twist = self.ref_door_joint_pos.unsqueeze(1).expand(-1, twist_len, -1)
        self.ref_door_body_pos_twist = torch.zeros(
            (*twist_shape, 3),
            device=self.device,
            dtype=self.door_link_pos.dtype,
        )
        self._override_reference_arx_pose_for_fixed_mode()

    def _override_reference_arx_pose_for_fixed_mode(self):
        if not self.fixed_arx_pose or self.num_arx_joints <= 0:
            return
        fixed_arx_pose = self._robot_arx_tuck_joint_pos_target.to(self.robot.data.joint_pos).unsqueeze(0).expand(
            self.num_envs, -1
        )
        zero_arx_vel = torch.zeros_like(fixed_arx_pose)
        self.ref_robot_arx_joint_pos = fixed_arx_pose
        self.ref_robot_arx_joint_vel = zero_arx_vel
        self.ref_robot_joint_pos[:, self.ref_arx_joint_idx] = fixed_arx_pose
        self.ref_joint_vel[:, self.ref_arx_joint_idx] = zero_arx_vel
        fixed_arx_pose_twist = fixed_arx_pose.unsqueeze(1).expand(-1, self.ref_robot_joint_pos_twist.shape[1], -1)
        self.ref_robot_joint_pos_twist[:, :, self.ref_arx_joint_idx] = fixed_arx_pose_twist
        self.ref_robot_arx_joint_pos_twist = fixed_arx_pose_twist

    def _get_intermediate_values(self):
        self.robot_key_body_pos = self.robot.data.body_pos_w[:, self._robot_key_body_idx]\
             - self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)
        self.robot_key_body_quat = self.robot.data.body_quat_w[:, self._robot_key_body_idx]

        self.robot_base_body_pos = self.robot_key_body_pos[:, self._robot_base_id_in_key_body_idx]
        self.robot_base_body_quat = self.robot_key_body_quat[:, self._robot_base_id_in_key_body_idx]

        self.robot_palm_body_pos = self.robot_key_body_pos[:, self._robot_palm_id_in_key_body_idx]
        self.robot_palm_body_quat = self.robot_key_body_quat[:, self._robot_palm_id_in_key_body_idx]

        self.robot_reset_key_body_pos = self.robot.data.body_pos_w[:, self._robot_reset_key_body_idx]\
             - self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)

        self.robot_base_joint_pos = self.robot.data.joint_pos[:, self._robot_base_dof_idx]
        self.robot_arm_joint_pos = self.robot.data.joint_pos[:, self._robot_arm_dof_idx]
        self.robot_finger_joint_pos = self.robot.data.joint_pos[:, self._robot_finger_dof_idx]
        self.robot_arx_joint_pos = self.robot.data.joint_pos[:, self._robot_arx_dof_idx]
        self.door_joint_pos = self.door.data.joint_pos
        # self.door_joint_vel = self.door.data.joint_vel
        self.robot_base_joint_vel = self.robot.data.joint_vel[:, self._robot_base_dof_idx]
        self.robot_arm_joint_vel = self.robot.data.joint_vel[:, self._robot_arm_dof_idx]
        self.robot_finger_joint_vel = self.robot.data.joint_vel[:, self._robot_finger_dof_idx]
        self.robot_arx_joint_vel = self.robot.data.joint_vel[:, self._robot_arx_dof_idx]

        self.door_base_link_pos = self.door.data.body_pos_w[:, self._door_base_link_idx]
        self.door_base_link_quat = self.door.data.body_quat_w[:, self._door_base_link_idx]
        self.door_link_pos = self.door.data.body_pos_w[:, self._door_body_idx]
        self.door_link_pos -= self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)
        self.door_link_quat = self.door.data.body_quat_w[:, self._door_body_idx]
        # self.door_keypoints = self.compute_door_keypoints()
        self.robot_body_lin_vel = self.robot.data.body_link_lin_vel_w[:, self._robot_key_body_idx]
        self.robot_body_ang_vel = self.robot.data.body_link_ang_vel_w[:, self._robot_key_body_idx]

        if self.ref_motion_lib is None:
            self._set_current_state_as_reference()
            return

        ref_robot_body_pos_twist = self.ref_motion_lib.get_robot_body_pos_twist()
        ref_key_body_idx = self.ref_key_body_idx.to(device=ref_robot_body_pos_twist.device, dtype=torch.long)
        ref_reset_key_body_idx = self.ref_reset_key_body_idx.to(device=ref_robot_body_pos_twist.device, dtype=torch.long)
        if ref_robot_body_pos_twist.shape[-1] != 3 and ref_robot_body_pos_twist.shape[-2] == 3:
            ref_robot_body_pos_twist = ref_robot_body_pos_twist.transpose(-1, -2)
        if ref_key_body_idx.numel() > 0:
            if int(ref_key_body_idx.min().item()) < 0 or int(ref_key_body_idx.max().item()) >= ref_robot_body_pos_twist.shape[2]:
                raise ValueError(
                    f"ref_key_body_idx out of range for robot_body_pos_twist: "
                    f"max_idx={int(ref_key_body_idx.max().item())}, num_bodies={ref_robot_body_pos_twist.shape[2]}."
                )
        self.ref_robot_key_body_pos_twist = ref_robot_body_pos_twist.index_select(2, ref_key_body_idx)
        # It is a misnomer, we are actually sending euler angles as it might be more friendly to MLP
        ref_robot_body_quat_twist = self.ref_motion_lib.get_robot_body_quat_twist()
        if ref_robot_body_quat_twist.shape[-1] != 4 and ref_robot_body_quat_twist.shape[-2] == 4:
            ref_robot_body_quat_twist = ref_robot_body_quat_twist.transpose(-1, -2)
        ref_key_body_idx = ref_key_body_idx.to(device=ref_robot_body_quat_twist.device)
        self.ref_robot_key_body_quat_twist = ref_robot_body_quat_twist.index_select(2, ref_key_body_idx)
        if self.ref_robot_key_body_pos_twist.shape[:3] != self.ref_robot_key_body_quat_twist.shape[:3]:
            raise ValueError(
                "Reference body twist tensors disagree on (env, twist, body) dimensions: "
                f"pos={tuple(self.ref_robot_key_body_pos_twist.shape)}, "
                f"quat={tuple(self.ref_robot_key_body_quat_twist.shape)}."
            )
        self.ref_robot_key_body_pos_twist, self.ref_robot_key_body_quat_twist = self.normalize_to_base_frame(
            self.robot_base_body_pos.squeeze(),
            self.robot_base_body_quat.squeeze(),
            self.robot_key_body_pos, 
            self.robot_key_body_quat, 
            self.ref_robot_key_body_pos_twist, 
            self.ref_robot_key_body_quat_twist
        )
        # self.ref_robot_key_body_pos_twist, self.ref_robot_key_body_quat_twist = self.normalize_to_base_frame(self.robot_base_body_pos, self.robot_base_body_quat, self.ref_robot_key_body_pos_twist, self.ref_robot_key_body_quat_twist)
        self.ref_robot_key_body_quat_twist = quat_to_6d(self.ref_robot_key_body_quat_twist)
        self.ref_robot_joint_pos_twist = self.ref_motion_lib.get_robot_joint_pos_twist()
        self.ref_robot_base_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_arm_joint_idx]
        self.ref_robot_arx_joint_pos_twist = self.ref_robot_joint_pos_twist[:, :, self.ref_arx_joint_idx]
        self.ref_door_joint_pos_twist = self.ref_motion_lib.get_door_joint_pos_twist()

        # self.ref_robot_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_key_body_idx]
        # self.ref_robot_key_body_quat = self.ref_motion_lib.get_robot_body_quat()[:, self._robot_key_body_idx]
        # self.ref_robot_reset_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_reset_key_body_idx]
        ref_robot_body_pos = self.ref_motion_lib.get_robot_body_pos()
        if ref_robot_body_pos.shape[-1] != 3 and ref_robot_body_pos.shape[-2] == 3:
            ref_robot_body_pos = ref_robot_body_pos.transpose(-1, -2)
        ref_key_body_idx = ref_key_body_idx.to(device=ref_robot_body_pos.device)
        ref_reset_key_body_idx = ref_reset_key_body_idx.to(device=ref_robot_body_pos.device)
        self.ref_robot_key_body_pos = ref_robot_body_pos.index_select(1, ref_key_body_idx)
        ref_robot_body_quat = self.ref_motion_lib.get_robot_body_quat()
        if ref_robot_body_quat.shape[-1] != 4 and ref_robot_body_quat.shape[-2] == 4:
            ref_robot_body_quat = ref_robot_body_quat.transpose(-1, -2)
        ref_key_body_idx = ref_key_body_idx.to(device=ref_robot_body_quat.device)
        self.ref_robot_key_body_quat = ref_robot_body_quat.index_select(1, ref_key_body_idx)
        self.ref_robot_reset_key_body_pos = ref_robot_body_pos.index_select(1, ref_reset_key_body_idx)
        self.ref_robot_joint_pos = self.ref_motion_lib.get_robot_joint_pos()
        self.ref_robot_base_joint_pos = self.ref_robot_joint_pos[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos = self.ref_robot_joint_pos[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_pos = self.ref_robot_joint_pos[:, self.ref_finger_joint_idx]
        self.ref_robot_arx_joint_pos = self.ref_robot_joint_pos[:, self.ref_arx_joint_idx]
        self.ref_joint_vel = self.ref_motion_lib.get_robot_joint_vel()
        self.ref_robot_base_joint_vel = self.ref_joint_vel[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_vel = self.ref_joint_vel[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_vel = self.ref_joint_vel[:, self.ref_finger_joint_idx]
        self.ref_robot_arx_joint_vel = self.ref_joint_vel[:, self.ref_arx_joint_idx]
        self.ref_door_joint_pos = self.ref_motion_lib.get_door_joint_pos()
        self.ref_hinge_contact_mask = self.ref_motion_lib.get_hinge_contact_mask()
        self.ref_panel_contact_mask = self.ref_motion_lib.get_panel_contact_mask()
        self.ref_grasp_stage_mask = self.ref_motion_lib.get_grasp_stage_mask()
        self.ref_door_body_pos_twist = self.ref_motion_lib.get_door_body_pos_twist()
        ref_motion_dt = max(float(self.ref_motion_lib.frame_dt), 1e-6)
        ref_robot_body_lin_vel = self.ref_motion_lib.get_robot_body_lin_vel()
        if ref_robot_body_lin_vel.shape[-1] != 3 and ref_robot_body_lin_vel.shape[-2] == 3:
            ref_robot_body_lin_vel = ref_robot_body_lin_vel.transpose(-1, -2)
        ref_robot_body_ang_vel = self.ref_motion_lib.get_robot_body_ang_vel()
        if ref_robot_body_ang_vel.shape[-1] != 3 and ref_robot_body_ang_vel.shape[-2] == 3:
            ref_robot_body_ang_vel = ref_robot_body_ang_vel.transpose(-1, -2)
        ref_key_body_idx = ref_key_body_idx.to(device=ref_robot_body_lin_vel.device)
        self.ref_robot_body_lin_vel = ref_robot_body_lin_vel.index_select(1, ref_key_body_idx) / ref_motion_dt
        ref_key_body_idx = ref_key_body_idx.to(device=ref_robot_body_ang_vel.device)
        self.ref_robot_body_ang_vel = ref_robot_body_ang_vel.index_select(1, ref_key_body_idx) / ref_motion_dt
        self._override_reference_arx_pose_for_fixed_mode()

    def _compute_penalties(self) -> torch.Tensor:
        """Compute every per-step penalty (each >= 0) and return their sum, to be SUBTRACTED from the
        reward in _get_rewards. Grouped here (out of _get_rewards, which handles tracking + contact
        rewards) for clarity. Each block also logs its own error/stats and updates its episode latch;
        this must be called exactly once per _get_rewards so those latches/logs stay correct.

        Terms: joint-limit, non-finger self-collision, base<->door, x5/arx-arm<->door, and franka
        control-box<->door contact. The finger<->door protection penalty is intentionally NOT here --
        it was removed from the reward (hackable; pull never used it, push no longer uses it)."""
        # x5/arx camera arm <-> door contact penalty (frame/panel/handle). Harsh: the slender
        # camera arm striking the door is a serious real-world failure.
        x5_body_force_norm = self._get_x5_body_contact_force_norm(include_franka_box=False)
        unsafe_x5_body_contact = x5_body_force_norm > self.cfg.x5_body_contact_force_threshold
        # Latch: this episode had at least one x5-arm door collision (reported per-rollout at done).
        self.episode_x5_collided |= unsafe_x5_body_contact
        self.extras["stats/x5_body_contact_force_norm_max"] = float(x5_body_force_norm.max().detach().cpu().item())
        self.extras["stats/x5_body_unsafe_contact_frac"] = float(
            unsafe_x5_body_contact.float().mean().detach().cpu().item()
        )
        weighted_x5_door_contact_penalty = self.x5_door_contact_penalty_w * unsafe_x5_body_contact.to(dtype=x5_body_force_norm.dtype)
        self.extras["error/x5_door_contact_penalty"] = weighted_x5_door_contact_penalty.mean().item()

        # Self-collision penalty over non-finger links (franka arm + x5/arx arm + base chassis).
        # The hand is excluded so finger poses are not driven conservative.
        self_collision_body_count = self._get_self_collision_body_count()
        weighted_self_collision_penalty = self.self_collision_penalty_w * self_collision_body_count
        self.extras["error/self_collision_penalty"] = weighted_self_collision_penalty.mean().item()
        self.extras["stats/self_collision_body_count_mean"] = self_collision_body_count.mean().item()

        # Base<->door contact penalty: no base face should touch the door. Count the four
        # vertical faces (front/back/left/right) whose contact force with any door body exceeds
        # the threshold and penalize per face in contact.
        base_door_force_norm = self._stacked_self_contact_force_norm(BASE_DOOR_SENSOR_NAMES)
        base_door_contact_count = (base_door_force_norm > self.cfg.base_door_contact_force_threshold).sum(dim=-1)
        weighted_base_door_contact_penalty = self.base_door_contact_penalty_w * base_door_contact_count.to(dtype=base_door_force_norm.dtype)
        self.extras["error/base_door_contact_penalty"] = weighted_base_door_contact_penalty.mean().item()
        self.extras["stats/base_door_contact_frac"] = float(
            (base_door_contact_count > 0).float().mean().detach().cpu().item()
        )

        # Franka control-box <-> door contact: no longer episode-ending (the box is sturdy). A
        # graded penalty ramps linearly from 0 at min_force (25 N -- light taps are free) to the
        # full weight at max_force (75 N and above). Replaces the old hard-termination check.
        franka_box_force_norm = self._get_franka_box_contact_force_norm()
        # Latch: this episode had at least one franka-box door collision (> min_force).
        self.episode_franka_box_collided |= franka_box_force_norm > self.cfg.franka_box_contact_penalty_min_force
        franka_box_penalty_frac = torch.clamp(
            (franka_box_force_norm - self.cfg.franka_box_contact_penalty_min_force)
            / (self.cfg.franka_box_contact_penalty_max_force - self.cfg.franka_box_contact_penalty_min_force),
            min=0.0,
            max=1.0,
        )
        weighted_franka_box_contact_penalty = self.franka_box_contact_penalty_w * franka_box_penalty_frac
        self.extras["error/franka_box_contact_penalty"] = weighted_franka_box_contact_penalty.mean().item()
        self.extras["stats/franka_box_contact_force_norm_max"] = float(franka_box_force_norm.max().detach().cpu().item())
        self.extras["fail/franka_box_contact_frac"] = float(
            (franka_box_force_norm > self.cfg.franka_box_contact_penalty_min_force).float().mean().detach().cpu().item()
        )

        # Joint-limit penalty on the actuated robot DOFs (soft margin near each limit).
        joint_limit_penalty, joint_limit_active_fraction = compute_joint_limit_penalty(
            joint_pos=self.robot.data.joint_pos[:, self._robot_dof_idx],
            joint_lower_limits=self.robot_dof_lower_limits,
            joint_upper_limits=self.robot_dof_upper_limits,
            soft_ratio=self.joint_limit_penalty_margin_ratio,
        )
        weighted_joint_limit_penalty = self.joint_limit_penalty_w * joint_limit_penalty
        self.extras["error/joint_limit_penalty"] = weighted_joint_limit_penalty.reshape(self.num_envs, -1).mean().item()
        self.extras["stats/joint_limit_active_fraction"] = (
            joint_limit_active_fraction.reshape(self.num_envs, -1).mean().item()
        )

        return (
            weighted_joint_limit_penalty
            + weighted_self_collision_penalty
            + weighted_base_door_contact_penalty
            + weighted_x5_door_contact_penalty
            + weighted_franka_box_contact_penalty
        )

    def _get_rewards(self) -> torch.Tensor:
        self._get_intermediate_values()
        self._log_dr_metrics()

        # key_body_pos_err, key_body_quat_err, door_err, root_pos_err, root_rot_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err, door_pos_err = compute_tracking_error(
        key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, arx_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err = compute_tracking_error(
            robot_key_body_pos = self.robot_key_body_pos,
            robot_key_body_quat = self.robot_key_body_quat,
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos,
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_arx_joint_pos = self.robot_arx_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,

            ref_robot_key_body_pos = self.ref_robot_key_body_pos,
            ref_robot_key_body_quat = self.ref_robot_key_body_quat,
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
            ref_robot_arx_joint_pos = self.ref_robot_arx_joint_pos,
            ref_robot_base_joint_vel = self.ref_robot_base_joint_vel,
            ref_robot_arm_joint_vel = self.ref_robot_arm_joint_vel,
            ref_robot_finger_joint_vel = self.ref_robot_finger_joint_vel,
        )

        self.extras["error/key_body_pos_err"] = math.sqrt(max(key_body_pos_err.reshape(self.num_envs, -1).mean().item(), 0.0))
        self.extras["error/key_body_quat_err"] = math.sqrt(max(key_body_quat_err.reshape(self.num_envs, -1).mean().item(), 0.0))
        self.extras["error/door_err"] = math.sqrt(max(door_err.reshape(self.num_envs, -1).mean().item(), 0.0))
        self.extras["error/base_joint_pos_err"] = math.sqrt(
            max(base_joint_pos_err.reshape(self.num_envs, -1).mean().item() / len(self.cfg.base_joints), 0.0)
        )
        self.extras["error/arm_joint_pos_err"] = math.sqrt(
            max(arm_joint_pos_err.reshape(self.num_envs, -1).mean().item() / len(self.cfg.arm_joints), 0.0)
        )
        self.extras["error/finger_joint_pos_err"] = math.sqrt(
            max(finger_joint_pos_err.reshape(self.num_envs, -1).mean().item() / len(self.cfg.finger_joints), 0.0)
        )
        self.extras["error/arx_joint_pos_err"] = math.sqrt(
            max(arx_joint_pos_err.reshape(self.num_envs, -1).mean().item() / max(len(self.cfg.arx_joints), 1), 0.0)
        )
        # self.extras["error/base_joint_vel_err"] = base_joint_vel_err.mean()
        # self.extras["error/arm_joint_vel_err"] = arm_joint_vel_err.mean()
        # self.extras["error/finger_joint_vel_err"] = finger_joint_vel_err.mean()

        if self.prob_get_first_key_frame is not None:
            self.extras["reset/prob_get_first_key_frame"] = float(self.prob_get_first_key_frame)

        # Use filtered handle-hand force. Do not use net_forces_w here because
        # net_forces_w includes all contacts acting on Door/link_2.
        contact_forces_door2 = self._get_filtered_contact_force_w(
            self.scene.sensors["contact_forces_door2"],
            expected_num_envs=self.num_envs,
        )
        # finger<->handle force. Fed (together with the finger<->panel force below) into the unified
        # finger<->door contact PROTECTION penalty. The hinge contact REWARD still uses this force but
        # is binary (rewards ANY contact above 1 N), so it alone never stops the fingers from crushing
        # the handle -- the protection penalty below does.
        handle_force_norm = torch.linalg.vector_norm(contact_forces_door2, dim=-1)
        self.extras["stats/filtered_handle_force_norm_mean"] = float(handle_force_norm.mean().detach().cpu().item())
        self.extras["stats/filtered_handle_force_norm_max"] = float(handle_force_norm.max().detach().cpu().item())

        # --- PUSH-only palm-handle contact reward ---------------------------------------------------
        # Reward panda_hand pressing the handle (link_2) during the grasp->push-open window
        # (ref_hinge_contact_mask == keyframes 2..5). Binary bonus, push envs only. On PUSH the old
        # finger-inclusive hinge_contact reward is gated off (see the compute_reward call below), so
        # fingers touching the handle are no longer rewarded for pushing. Pull is untouched.
        is_push_env = self._get_is_push_env()  # [num_envs] float (1 push / 0 pull)
        contact_forces_door2_palm = self._get_filtered_contact_force_w(
            self.scene.sensors["contact_forces_door2_palm"],
            expected_num_envs=self.num_envs,
        )
        palm_handle_force_norm = torch.linalg.vector_norm(contact_forces_door2_palm, dim=-1)
        palm_handle_contact = (palm_handle_force_norm > self.cfg.handle_contact_force_threshold).to(
            dtype=palm_handle_force_norm.dtype
        )
        weighted_palm_handle_reward = (
            self.palm_handle_reward_w
            * self.ref_hinge_contact_mask.squeeze()
            * is_push_env
            * palm_handle_contact
        )
        self.extras["reward/palm_handle_push_reward"] = weighted_palm_handle_reward.mean().item()
        self.extras["stats/palm_handle_force_norm_max"] = float(palm_handle_force_norm.max().detach().cpu().item())

        # --- PULL-only hinge (handle) contact reward -----------------------------------------------
        # Separated out of compute_deep_mimic_rewards (this is a task/contact reward, not a tracking
        # term). Binary bonus for the hand contacting the handle (link_2) during the grasp/pull window
        # (ref_hinge_contact_mask), gated to PULL envs (1 - is_push); PUSH uses the palm-only reward
        # above instead. Reuses handle_force_norm computed above (||contact_forces_door2||), whose filter
        # is the hand body + the two fingers (
        # fingers 1/2/3, mcp excluded); behavior is identical to the old contact_force_w * contact_reward
        # term inside deep-mimic.
        hinge_contact = (handle_force_norm > self.cfg.handle_contact_force_threshold).to(
            dtype=handle_force_norm.dtype
        )
        weighted_hinge_contact_reward = (
            self.hinge_contact_reward_w
            * self.ref_hinge_contact_mask.squeeze()
            * (1.0 - is_push_env)
            * hinge_contact
        )
        self.extras["reward/hinge_contact_pull_reward"] = weighted_hinge_contact_reward.mean().item()

        # finger<->panel force (gripper fingers vs door panel Door/link_1).
        contact_forces_door_panel = self._get_filtered_contact_force_w(
            self.scene.sensors["contact_forces_door_panel"],
            expected_num_envs=self.num_envs,
        )
        panel_force_norm = torch.linalg.vector_norm(contact_forces_door_panel, dim=-1)
        self.extras["stats/panel_contact_force_norm_max"] = float(panel_force_norm.max().detach().cpu().item())

        # Finger<->door contact PROTECTION penalty: finger<->panel and finger<->handle are processed
        # TOGETHER to protect the fingers. Whenever the fingers press EITHER door body
        # (panel link_1 OR handle link_2) harder than finger_door_contact_force_threshold, a strong
        # penalty is applied -- but ONLY while ref_panel_contact_mask is on (push: open-door +
        # base-forward, keyframes 3..5; pull: retract-arm + push-panel + hold-traverse, keyframes
        # 6..9). Off elsewhere so it never fights grasping/rotating the handle. Replaces the old weak
        # graded panel + handle penalties.
        combined_finger_door_force_norm = torch.maximum(panel_force_norm, handle_force_norm)
        finger_door_over_threshold = combined_finger_door_force_norm > self.cfg.finger_door_contact_force_threshold
        finger_door_penalty_active = (self.ref_panel_contact_mask.squeeze() > 0).to(dtype=combined_finger_door_force_norm.dtype)
        weighted_finger_door_contact_penalty = (
            self.finger_door_contact_penalty_w
            * finger_door_over_threshold.to(dtype=combined_finger_door_force_norm.dtype)
            * finger_door_penalty_active
        )
        self.extras["error/finger_door_contact_penalty"] = weighted_finger_door_contact_penalty.mean().item()
        self.extras["stats/finger_door_contact_force_norm_max"] = float(combined_finger_door_force_norm.max().detach().cpu().item())
        self.extras["stats/finger_door_contact_frac"] = float(finger_door_over_threshold.float().mean().detach().cpu().item())

        # Diagnostic contact forces surfaced for the eval/play scripts to print: (1) franka<->arx
        # (self-collision of the franka arm against the arx/x5 camera arm) and (2) the fingers
        # <->panel. Store per-env tensors as attributes + max/mean in extras.
        franka_arx_force_norm = self._get_franka_arx_contact_force_norm()
        self.franka_arx_contact_force_norm = franka_arx_force_norm
        self.finger_panel_contact_force_norm = panel_force_norm
        self.finger_handle_contact_force_norm = handle_force_norm
        self.extras["stats/franka_arx_contact_force_norm_max"] = float(franka_arx_force_norm.max().detach().cpu().item())
        self.extras["stats/franka_arx_contact_force_norm_mean"] = float(franka_arx_force_norm.mean().detach().cpu().item())
        self.extras["stats/finger_panel_contact_force_norm_mean"] = float(panel_force_norm.mean().detach().cpu().item())
        # Also REPORT the finger<->panel normal force during the PREGRASP->GRASP stage
        # (ref_grasp_stage_mask, keyframes 1..3) for debugging. Masked mean = 0 when no env is there.
        grasp_stage = (self.ref_grasp_stage_mask.squeeze() > 0).to(panel_force_norm.dtype)
        grasp_denom = grasp_stage.sum().clamp(min=1.0)
        self.extras["finger_panel_normal_forces_mean"] = float(
            ((panel_force_norm * grasp_stage).sum() / grasp_denom).detach().cpu().item()
        )

        deep_mimic_reward = compute_deep_mimic_rewards(
            robot_key_body_pos = self.robot_key_body_pos, 
            robot_key_body_quat = self.robot_key_body_quat, 
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos, 
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_arx_joint_pos = self.robot_arx_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,
            robot_body_lin_vel = self.robot_body_lin_vel,
            robot_body_ang_vel = self.robot_body_ang_vel,

            ref_robot_key_body_pos = self.ref_robot_key_body_pos, 
            ref_robot_key_body_quat = self.ref_robot_key_body_quat, 
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
            ref_robot_arx_joint_pos = self.ref_robot_arx_joint_pos,
            ref_robot_base_joint_vel = self.ref_robot_base_joint_vel,
            ref_robot_arm_joint_vel = self.ref_robot_arm_joint_vel,
            ref_robot_finger_joint_vel = self.ref_robot_finger_joint_vel,
            ref_robot_body_lin_vel = self.ref_robot_body_lin_vel,
            ref_robot_body_ang_vel = self.ref_robot_body_ang_vel,

            robot_key_body_pos_scale = self.robot_key_body_pos_scale, 
            robot_key_body_quat_scale = self.robot_body_quat_scale,
            door_joint_pos_scale = self.door_joint_pos_scale,
            robot_base_joint_pos_scale = self.robot_base_joint_pos_scale,
            robot_arm_joint_pos_scale = self.robot_arm_joint_pos_scale,
            robot_finger_joint_pos_scale = self.robot_finger_joint_pos_scale,
            robot_base_joint_vel_scale = self.robot_base_joint_vel_scale,
            robot_arm_joint_vel_scale = self.robot_arm_joint_vel_scale,
            robot_finger_joint_vel_scale = self.robot_finger_joint_vel_scale,
            robot_body_lin_vel_scale = self.robot_body_lin_vel_scale,
            robot_body_ang_vel_scale = self.robot_body_ang_vel_scale,

            robot_key_body_pos_w = self.robot_key_body_pos_w, 
            robot_key_body_quat_w = self.robot_body_quat_w,
            door_joint_pos_w = self.door_joint_pos_w,
            robot_base_joint_pos_w = self.robot_base_joint_pos_w,
            robot_arm_joint_pos_w = self.robot_arm_joint_pos_w,
            robot_finger_joint_pos_w = self.robot_finger_joint_pos_w,
            robot_arx_joint_pos_w = self.robot_arx_joint_pos_w,
            robot_base_joint_vel_w = self.robot_base_joint_vel_w,
            robot_arm_joint_vel_w = self.robot_arm_joint_vel_w,
            robot_finger_joint_vel_w = self.robot_finger_joint_vel_w,
            robot_body_lin_vel_w = self.robot_body_lin_vel_w,
            robot_body_ang_vel_w = self.robot_body_ang_vel_w,
        )

        arx_tuck_joint_pos_diff = hinge_angle_diff(
            self._robot_arx_tuck_joint_pos_target.unsqueeze(0),
            self.robot_arx_joint_pos,
        )
        arx_tuck_joint_pos_err = torch.sum(arx_tuck_joint_pos_diff * arx_tuck_joint_pos_diff, dim=-1)
        arx_tuck_reward = torch.exp(-self.robot_arx_tuck_joint_pos_scale * arx_tuck_joint_pos_err)
        weighted_arx_tuck_reward = self.robot_arx_tuck_joint_pos_w * arx_tuck_reward
        self.extras["reward/arx_tuck_reward"] = weighted_arx_tuck_reward.mean().item()

        # All per-step penalties (joint-limit, self-collision, base/x5/franka-box door contact),
        # summed. Grouped in one method so the reward assembly below reads as reward - penalties.
        total_penalty = self._compute_penalties()
        self._update_success_metrics()

        # 1. Base alive reward: small constant for remaining active
        alive_base = self.alive_base 
        
        # 2. Difficulty Bonus: Extra points for staying alive during contact
        # self.ref_hinge_contact_mask is 1.0 when grasping/pulling
        alive_bonus = self.alive_bonus * self.ref_hinge_contact_mask.squeeze()
        
        total_alive_reward = alive_base + alive_bonus

        # 3. Combine with tracking reward and termination penalty
        is_killed = self.reset_terminated
        termination_penalty = self.termination_penalty
        
        final_reward = (
            deep_mimic_reward
            + weighted_arx_tuck_reward
            + total_alive_reward
            + weighted_palm_handle_reward    # PUSH-only palm-handle contact reward
            + weighted_hinge_contact_reward  # PULL-only handle contact reward (was inside deep-mimic)
            - total_penalty
        )
        final_reward = torch.where(is_killed, final_reward + termination_penalty, final_reward)

        return final_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        reached_last_frame = self._get_reached_last_frame_mask()
        time_out = (self.episode_length_buf >= self.max_trial_steps - 1) | reached_last_frame
        if not self.early_stopping:
            return torch.zeros_like(time_out), time_out
        self._get_intermediate_values()
        progress = min(self._get_curriculum_step_count() / self.reset_progress_total, 1.0)
        # Diagnostic override: force the drift-threshold curriculum to a fixed progress in [0, 1]
        # (e.g. 1.0 = loosest/highest thresholds) so a teacher test isn't killed by the tight
        # early-schedule deltas. Set via base_env.drift_threshold_progress_override; None = normal.
        _progress_override = getattr(self, "drift_threshold_progress_override", None)
        if _progress_override is not None:
            progress = min(max(float(_progress_override), 0.0), 1.0)
        reset_key_body_pos_delta = self.reset_key_body_pos_delta_min + (self.reset_key_body_pos_delta_max - self.reset_key_body_pos_delta_min) * progress
        reset_key_body_quat_delta = self.reset_key_body_quat_delta_min + (self.reset_key_body_quat_delta_max - self.reset_key_body_quat_delta_min) * progress
        reset_door_joint_pos_delta = self.reset_door_joint_pos_delta_min + (self.reset_door_joint_pos_delta_max - self.reset_door_joint_pos_delta_min) * progress
        self.extras["reset/reset_key_body_pos_delta"] = reset_key_body_pos_delta
        self.extras["reset/reset_key_body_quat_delta"] = reset_key_body_quat_delta
        self.extras["reset/reset_door_joint_pos_delta"] = reset_door_joint_pos_delta
        # reset_key_body_pos_delta = reset_key_body_pos_delta ** 2 * len(self.cfg.robot_reset_key_bodies)
        # reset_key_body_quat_delta = reset_key_body_quat_delta ** 2 * len(self.cfg.robot_reset_key_bodies)
        # reset_door_joint_pos_delta = reset_door_joint_pos_delta ** 2 * len(self.cfg.door_joint_names)
        reset_key_body_pos_delta = reset_key_body_pos_delta ** 2
        reset_key_body_quat_delta = reset_key_body_quat_delta ** 2
        reset_door_joint_pos_delta = reset_door_joint_pos_delta ** 2
        # Termination: robot/door tracking drift only. Neither the x5 arm nor the franka control
        # box ends the episode anymore -- both are discouraged by (high) contact penalties in
        # _get_rewards (x5: x5_door_contact_penalty_w) instead of hard termination. The x5-arm
        # contact is still computed here PURELY to keep logging its fraction to wandb.
        x5_arm_force_norm = self._get_x5_body_contact_force_norm(include_franka_box=False)
        unsafe_x5_arm_contact = x5_arm_force_norm > self.cfg.x5_body_contact_force_threshold
        # key_body_pos_err, key_body_quat_err, door_err, root_pos_err, root_rot_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err, door_pos_err = compute_tracking_error(
        key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, arx_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err = compute_tracking_error(
            robot_key_body_pos = self.robot_reset_key_body_pos,
            robot_key_body_quat = self.robot_key_body_quat,
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos,
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_arx_joint_pos = self.robot_arx_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,

            ref_robot_key_body_pos = self.ref_robot_reset_key_body_pos,
            ref_robot_key_body_quat = self.ref_robot_key_body_quat,
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
            ref_robot_arx_joint_pos = self.ref_robot_arx_joint_pos,
            ref_robot_base_joint_vel = self.ref_robot_base_joint_vel,
            ref_robot_arm_joint_vel = self.ref_robot_arm_joint_vel,
            ref_robot_finger_joint_vel = self.ref_robot_finger_joint_vel,
        )
        # Hard-termination failure modes are now robot/door tracking drift only; log each fraction
        # for diagnosis. x5-arm and franka-box door contact are (high) contact penalties in
        # _get_rewards, not terminations (x5 contact fraction is logged there as
        # stats/x5_body_unsafe_contact_frac).
        fail_robot_drift = (key_body_pos_err > reset_key_body_pos_delta) | (key_body_quat_err > reset_key_body_quat_delta)
        fail_door_drift = door_err > reset_door_joint_pos_delta
        self.extras["fail/robot_drift_frac"] = float(fail_robot_drift.float().mean().detach().cpu().item())
        self.extras["fail/door_drift_frac"] = float(fail_door_drift.float().mean().detach().cpu().item())
        # Logged-only (NOT a termination): fraction of envs with unsafe x5-arm door contact.
        self.extras["fail/x5_collision_frac"] = float(unsafe_x5_arm_contact.float().mean().detach().cpu().item())
        # Per-env termination-reason masks + the raw drift magnitudes vs their (curriculum) thresholds.
        # These let an offline diagnostic (scripts/rl_games/test_teacher_diagnose.py) attribute each
        # kill to robot-pose drift vs door-angle drift and see how far over threshold it went.
        self.extras["fail/robot_drift"] = fail_robot_drift.detach()
        self.extras["fail/door_drift"] = fail_door_drift.detach()
        self.extras["fail/key_body_pos_err"] = key_body_pos_err.detach()
        self.extras["fail/key_body_quat_err"] = key_body_quat_err.detach()
        self.extras["fail/door_err"] = door_err.detach()
        self.extras["fail/reset_key_body_pos_delta"] = float(reset_key_body_pos_delta)
        self.extras["fail/reset_key_body_quat_delta"] = float(reset_key_body_quat_delta)
        self.extras["fail/reset_door_joint_pos_delta"] = float(reset_door_joint_pos_delta)
        # Per-body / per-joint drift breakdown (BEFORE the max-reduction that compute_tracking_error
        # applies) so a single-env trace can name WHICH body/joint drifted and by how much. Errors are
        # squared (matching the thresholds above): pos in m^2, door in rad^2.
        pos_diff = self.ref_robot_reset_key_body_pos - self.robot_reset_key_body_pos          # [B, N, 3]
        self.extras["diag/key_body_pos_err_per_body"] = (pos_diff * pos_diff).sum(dim=-1).detach()  # [B, N]
        # Actual (measured) vs desired (reference) world positions per reset key body, so a trace can
        # print exactly where the body IS vs where the reference wants it.
        self.extras["diag/key_body_pos"] = self.robot_reset_key_body_pos.detach()             # [B, N, 3]
        self.extras["diag/ref_key_body_pos"] = self.ref_robot_reset_key_body_pos.detach()     # [B, N, 3]
        quat_diff = quat_diff_angle(self.robot_key_body_quat, self.ref_robot_key_body_quat)   # [B, M]
        self.extras["diag/key_body_quat_err_per_body"] = (quat_diff * quat_diff).detach()     # [B, M]
        door_diff = self.ref_door_joint_pos - self.door_joint_pos                             # [B, D]
        self.extras["diag/door_joint_pos"] = self.door_joint_pos.detach()
        self.extras["diag/ref_door_joint_pos"] = self.ref_door_joint_pos.detach()
        self.extras["diag/door_joint_err_per_joint"] = (door_diff * door_diff).detach()       # [B, D]
        self.extras["diag/reset_key_body_names"] = list(self._robot_reset_key_body_names)
        self.extras["diag/key_body_names"] = list(self._robot_key_body_names)
        return fail_robot_drift | fail_door_drift, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._update_adr_ranges()
        if not self.use_motion_ref:
            self.prob_get_first_key_frame = None
            default_trial_steps = max(1, math.ceil(float(self.cfg.episode_length_s) / max(float(self.dt), 1e-6)))
            self.max_trial_steps[env_ids] = default_trial_steps
            self._sample_reset_randomization(env_ids)

            default_root_state = self.robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self.scene.env_origins[env_ids]

            self.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
            self.joint_vel[env_ids] = self.robot.data.default_joint_vel[env_ids]
            self._apply_spawn_noise(env_ids)
            if self.fixed_arx_pose and self.num_arx_joints > 0:
                self.joint_pos[env_ids[:, None], self._robot_arx_dof_idx[None, :]] = (
                    self._robot_arx_tuck_joint_pos_target.to(self.joint_pos).unsqueeze(0)
                )
                self.joint_vel[env_ids[:, None], self._robot_arx_dof_idx[None, :]] = 0.0
            if self.fixed_arx_pose and self.num_extra_camera_joints > 0:
                self.joint_pos[env_ids[:, None], self._robot_extra_camera_dof_idx[None, :]] = (
                    self._robot_extra_camera_default_pos.to(self.joint_pos).unsqueeze(0)
                )
                self.joint_vel[env_ids[:, None], self._robot_extra_camera_dof_idx[None, :]] = 0.0

            self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], None, env_ids)
            self.robot.set_joint_position_target(self.joint_pos[env_ids], env_ids=env_ids)

            door_joint_pos = self.door.data.default_joint_pos[env_ids].clone()
            door_joint_vel = self.door.data.default_joint_vel[env_ids].clone()
            self.door.write_joint_state_to_sim(door_joint_pos, door_joint_vel, None, env_ids)

            self.robot_dof_targets[env_ids, :] = self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]]
            self.applied_robot_dof_targets[env_ids, :] = self.robot_dof_targets[env_ids, :]
            self._action_target_history[env_ids] = self.robot_dof_targets[env_ids].unsqueeze(1)
            self.episode_reached_last_frame[env_ids] = False
            self.episode_x5_collided[env_ids] = False
            self.episode_franka_box_collided[env_ids] = False
            super()._reset_idx(env_ids)
            self._sample_door_handle_effort_limits(env_ids)
            self._apply_door_handle_effort_limits(env_ids)
            self._sample_door_panel_effort_limits(env_ids)
            self._sample_door_latch_thresholds(env_ids)
            self._refresh_nominal_door_joint_gains(env_ids)
            return

        reset_frame_idx, self.prob_get_first_key_frame = self.ref_motion_lib.reset(
            env_ids,
            step_count=self._get_curriculum_step_count(),
            reset_progress_total=self.reset_progress_total,
        )
        remaining_frames = torch.clamp(
            float(self.ref_motion_lib.num_frames - 1) - reset_frame_idx.to(dtype=torch.float32),
            min=0.0,
        )
        required_steps = torch.ceil(remaining_frames / max(float(self.ref_motion_lib.frame_step), 1e-6)).long() + 1
        self.max_trial_steps[env_ids] = torch.clamp(required_steps, min=1)
        self._sample_reset_randomization(env_ids)

        deep_mimic_initial_joint_pos = self.ref_motion_lib.get_robot_joint_pos(env_ids)
        deep_mimic_initial_joint_vel = torch.zeros_like(deep_mimic_initial_joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.default_joint_vel[env_ids]
        self.joint_vel[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_vel.to(self.joint_vel)[..., self.ref_robot_dof_idx]
        self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_pos.to(self.joint_pos)[..., self.ref_robot_dof_idx]
        self._apply_spawn_noise(env_ids)
        if self.fixed_arx_pose and self.num_arx_joints > 0:
            self.joint_pos[env_ids[:, None], self._robot_arx_dof_idx[None, :]] = (
                self._robot_arx_tuck_joint_pos_target.to(self.joint_pos).unsqueeze(0)
            )
            self.joint_vel[env_ids[:, None], self._robot_arx_dof_idx[None, :]] = 0.0
        if self.fixed_arx_pose and self.num_extra_camera_joints > 0:
            self.joint_pos[env_ids[:, None], self._robot_extra_camera_dof_idx[None, :]] = (
                self._robot_extra_camera_default_pos.to(self.joint_pos).unsqueeze(0)
            )
            self.joint_vel[env_ids[:, None], self._robot_extra_camera_dof_idx[None, :]] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], None, env_ids)
        self.robot.set_joint_position_target(self.joint_pos[env_ids], env_ids=env_ids)

        door_joint_pos = self.ref_motion_lib.get_door_joint_pos(env_ids).to(self.door.data.joint_pos)
        door_joint_vel = self.door.data.default_joint_vel[env_ids].clone()

        self.door.write_joint_state_to_sim(door_joint_pos, door_joint_vel, None, env_ids)
        # self.door.set_joint_position_target(door_joint_pos, None, env_ids)

        # self.last_actions[env_ids] = 0.0
        self.robot_dof_targets[env_ids, :] = self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]]
        self.applied_robot_dof_targets[env_ids, :] = self.robot_dof_targets[env_ids, :]
        self._action_target_history[env_ids] = self.robot_dof_targets[env_ids].unsqueeze(1)
        self.episode_reached_last_frame[env_ids] = False
        self.episode_x5_collided[env_ids] = False
        self.episode_franka_box_collided[env_ids] = False
        super()._reset_idx(env_ids)
        self._sample_door_handle_effort_limits(env_ids)
        self._apply_door_handle_effort_limits(env_ids)
        self._sample_door_panel_effort_limits(env_ids)
        self._sample_door_latch_thresholds(env_ids)
        self._refresh_nominal_door_joint_gains(env_ids)

    def close(self):
        if getattr(self, "viser_pointcloud_enabled", False):
            self._flush_viser_pointcloud_recording(chunk_complete=False, reason="env close")
        return super().close()

@torch.jit.script
def compute_deep_mimic_rewards(
    robot_key_body_pos: torch.Tensor,
    robot_key_body_quat: torch.Tensor,
    door_joint_pos: torch.Tensor,
    robot_base_joint_pos: torch.Tensor,
    robot_arm_joint_pos: torch.Tensor,
    robot_finger_joint_pos: torch.Tensor,
    robot_arx_joint_pos: torch.Tensor,
    robot_base_joint_vel: torch.Tensor,
    robot_arm_joint_vel: torch.Tensor,
    robot_finger_joint_vel: torch.Tensor,
    robot_body_lin_vel: torch.Tensor,
    robot_body_ang_vel: torch.Tensor,

    ref_robot_key_body_pos: torch.Tensor,
    ref_robot_key_body_quat: torch.Tensor,
    ref_door_joint_pos: torch.Tensor,
    ref_robot_base_joint_pos: torch.Tensor,
    ref_robot_arm_joint_pos: torch.Tensor,
    ref_robot_finger_joint_pos: torch.Tensor,
    ref_robot_arx_joint_pos: torch.Tensor,
    ref_robot_base_joint_vel: torch.Tensor,
    ref_robot_arm_joint_vel: torch.Tensor,
    ref_robot_finger_joint_vel: torch.Tensor,
    ref_robot_body_lin_vel: torch.Tensor,
    ref_robot_body_ang_vel: torch.Tensor,

    robot_key_body_pos_scale: float,
    robot_key_body_quat_scale: float,
    door_joint_pos_scale: float,
    robot_base_joint_pos_scale: float,
    robot_arm_joint_pos_scale: float,
    robot_finger_joint_pos_scale: float,
    robot_base_joint_vel_scale: float,
    robot_arm_joint_vel_scale: float,
    robot_finger_joint_vel_scale: float,
    robot_body_lin_vel_scale: float,
    robot_body_ang_vel_scale: float,

    robot_key_body_pos_w: float,
    robot_key_body_quat_w: float,
    door_joint_pos_w: float,
    robot_base_joint_pos_w: float,
    robot_arm_joint_pos_w: float,
    robot_finger_joint_pos_w: float,
    robot_arx_joint_pos_w: float,
    robot_base_joint_vel_w: float,
    robot_arm_joint_vel_w: float,
    robot_finger_joint_vel_w: float,
    robot_body_lin_vel_w: float,
    robot_body_ang_vel_w: float,
) -> torch.Tensor:
    # ----------------------------------
    # Robot body position error
    # ----------------------------------
    # [B, N, 3]
    key_body_pos_diff = ref_robot_key_body_pos - robot_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # [B, N]
    key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)

    if robot_body_lin_vel_w != 0:
        robot_body_lin_vel_diff = ref_robot_body_lin_vel - robot_body_lin_vel
        robot_body_lin_vel_err = torch.sum(robot_body_lin_vel_diff * robot_body_lin_vel_diff, dim=-1)  # [B, N]
        robot_body_lin_vel_err = torch.sum(robot_body_lin_vel_err, dim=-1)
    else:
        robot_body_lin_vel_err = None
    if robot_body_ang_vel_w != 0:
        robot_body_ang_vel_diff = ref_robot_body_ang_vel - robot_body_ang_vel
        robot_body_ang_vel_err = torch.sum(robot_body_ang_vel_diff * robot_body_ang_vel_diff, dim=-1)  # [B, N]
        robot_body_ang_vel_err = torch.sum(robot_body_ang_vel_err, dim=-1)
    else:
        robot_body_ang_vel_err = None
    # ----------------------------------
    # Robot body orientation error
    # ----------------------------------
    # [B, N]
    key_body_quat_diff = quat_diff_angle(robot_key_body_quat, ref_robot_key_body_quat)
    key_body_quat_err = torch.sum(key_body_quat_diff * key_body_quat_diff, dim=-1)  # [B]
    # ----------------------------------
    # Door joint error
    # ----------------------------------
    door_diff = hinge_angle_diff(ref_door_joint_pos, door_joint_pos)
    door_err = torch.sum(door_diff * door_diff, dim=-1)  # [B]
    # ----------------------------------
    # Robot joint position error
    # ----------------------------------
    base_joint_pos_diff = ref_robot_base_joint_pos - robot_base_joint_pos
    base_joint_pos_err = torch.sum(base_joint_pos_diff * base_joint_pos_diff, dim=-1)  # [B]
    arm_joint_pos_diff = hinge_angle_diff(ref_robot_arm_joint_pos, robot_arm_joint_pos)
    arm_joint_pos_err = torch.sum(arm_joint_pos_diff * arm_joint_pos_diff, dim=-1)  # [B]
    finger_joint_pos_diff = hinge_angle_diff(ref_robot_finger_joint_pos, robot_finger_joint_pos)
    finger_joint_pos_err = torch.sum(finger_joint_pos_diff * finger_joint_pos_diff, dim=-1)  # [B]
    arx_joint_pos_diff = hinge_angle_diff(ref_robot_arx_joint_pos, robot_arx_joint_pos)
    arx_joint_pos_err = torch.sum(arx_joint_pos_diff * arx_joint_pos_diff, dim=-1)  # [B]

    base_joint_vel_diff = ref_robot_base_joint_vel - robot_base_joint_vel
    base_joint_vel_err = torch.sum(base_joint_vel_diff * base_joint_vel_diff, dim=-1)  # [B]
    arm_joint_vel_diff = ref_robot_arm_joint_vel - robot_arm_joint_vel
    arm_joint_vel_err = torch.sum(arm_joint_vel_diff * arm_joint_vel_diff, dim=-1)  # [B]
    finger_joint_vel_diff = ref_robot_finger_joint_vel - robot_finger_joint_vel
    finger_joint_vel_err = torch.sum(finger_joint_vel_diff * finger_joint_vel_diff, dim=-1)  # [B]

    # ----------------------------------
    # Exponential rewards (DeepMimic style)
    # ----------------------------------
    key_body_pos_r = torch.exp(-robot_key_body_pos_scale * key_body_pos_err)
    key_body_quat_r = torch.exp(-robot_key_body_quat_scale * key_body_quat_err)
    door_r = torch.exp(-door_joint_pos_scale * door_err)
    base_joint_pos_r = torch.exp(-robot_base_joint_pos_scale * base_joint_pos_err)
    arm_joint_pos_r = torch.exp(-robot_arm_joint_pos_scale * arm_joint_pos_err)
    finger_joint_pos_r = torch.exp(-robot_finger_joint_pos_scale * finger_joint_pos_err)
    arx_joint_pos_r = torch.exp(-robot_arm_joint_pos_scale * arx_joint_pos_err)
    base_joint_vel_r = torch.exp(-robot_base_joint_vel_scale * base_joint_vel_err)
    arm_joint_vel_r = torch.exp(-robot_arm_joint_vel_scale * arm_joint_vel_err)
    finger_joint_vel_r = torch.exp(-robot_finger_joint_vel_scale * finger_joint_vel_err)
    if robot_body_lin_vel_err is not None:
        robot_body_lin_vel_r = torch.exp(-robot_body_lin_vel_scale * robot_body_lin_vel_err)
    else:
        robot_body_lin_vel_r = torch.zeros_like(key_body_pos_r)
    if robot_body_ang_vel_err is not None:
        robot_body_ang_vel_r = torch.exp(-robot_body_ang_vel_scale * robot_body_ang_vel_err)
    else:
        robot_body_ang_vel_r = torch.zeros_like(key_body_pos_r)

    # ----------------------------------
    # Final reward (pure DeepMimic tracking; contact/handle rewards live in _get_rewards)
    # ----------------------------------
    reward = robot_key_body_pos_w * key_body_pos_r\
         + robot_key_body_quat_w * key_body_quat_r\
         + door_joint_pos_w * door_r\
         + robot_base_joint_pos_w * base_joint_pos_r\
         + robot_arm_joint_pos_w * arm_joint_pos_r\
         + robot_finger_joint_pos_w * finger_joint_pos_r\
         + robot_arx_joint_pos_w * arx_joint_pos_r\
         + robot_base_joint_vel_w * base_joint_vel_r\
         + robot_arm_joint_vel_w * arm_joint_vel_r\
         + robot_finger_joint_vel_w * finger_joint_vel_r\
         + robot_body_lin_vel_w * robot_body_lin_vel_r\
         + robot_body_ang_vel_w * robot_body_ang_vel_r
    return reward

def compute_tracking_error(
    robot_key_body_pos: torch.Tensor,
    robot_key_body_quat: torch.Tensor,
    door_joint_pos: torch.Tensor,
    robot_base_joint_pos: torch.Tensor,
    robot_arm_joint_pos: torch.Tensor,
    robot_finger_joint_pos: torch.Tensor,
    robot_arx_joint_pos: torch.Tensor,
    robot_base_joint_vel: torch.Tensor,
    robot_arm_joint_vel: torch.Tensor,
    robot_finger_joint_vel: torch.Tensor,

    ref_robot_key_body_pos: torch.Tensor,
    ref_robot_key_body_quat: torch.Tensor,
    ref_door_joint_pos: torch.Tensor,
    ref_robot_base_joint_pos: torch.Tensor,
    ref_robot_arm_joint_pos: torch.Tensor,
    ref_robot_finger_joint_pos: torch.Tensor,
    ref_robot_arx_joint_pos: torch.Tensor,
    ref_robot_base_joint_vel: torch.Tensor,
    ref_robot_arm_joint_vel: torch.Tensor,
    ref_robot_finger_joint_vel: torch.Tensor,
) -> torch.Tensor:
    # ----------------------------------
    # Robot body position error
    # ----------------------------------
    # [B, N, 3]
    key_body_pos_diff = ref_robot_key_body_pos - robot_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # [B, N]
    key_body_pos_err = torch.max(key_body_pos_err, dim=-1).values
    # ----------------------------------
    # Robot body orientation error
    # ----------------------------------
    # [B, N]
    key_body_quat_diff = quat_diff_angle(robot_key_body_quat, ref_robot_key_body_quat)
    # key_body_quat_err = torch.sum(key_body_quat_diff * key_body_quat_diff, dim=-1)  # [B]
    key_body_quat_err = torch.max(key_body_quat_diff * key_body_quat_diff, dim=-1).values
    # ----------------------------------
    # Door joint error
    # ----------------------------------
    door_diff = ref_door_joint_pos - door_joint_pos
    # door_err = torch.sum(door_diff * door_diff, dim=-1)  # [B]
    door_err = torch.max(door_diff * door_diff, dim=-1).values
    # ----------------------------------
    # Robot joint position error
    # ----------------------------------
    base_joint_pos_diff = ref_robot_base_joint_pos - robot_base_joint_pos
    base_joint_pos_err = torch.sum(base_joint_pos_diff * base_joint_pos_diff, dim=-1)  # [B]
    # root_pos_diff = ref_robot_base_joint_pos[:, :2] - robot_base_joint_pos[:, :2]
    # root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)  # [B]
    # root_rot_diff = hinge_angle_diff(ref_robot_base_joint_pos[:, 2:], robot_base_joint_pos[:, 2:])
    # root_rot_err = torch.sum(root_rot_diff * root_rot_diff, dim=-1)  # [B]
    arm_joint_pos_diff = ref_robot_arm_joint_pos - robot_arm_joint_pos
    arm_joint_pos_err = torch.sum(arm_joint_pos_diff * arm_joint_pos_diff, dim=-1)  # [B]
    finger_joint_pos_diff = ref_robot_finger_joint_pos - robot_finger_joint_pos
    finger_joint_pos_err = torch.sum(finger_joint_pos_diff * finger_joint_pos_diff, dim=-1)  # [B]
    arx_joint_pos_diff = ref_robot_arx_joint_pos - robot_arx_joint_pos
    arx_joint_pos_err = torch.sum(arx_joint_pos_diff * arx_joint_pos_diff, dim=-1)  # [B]
    base_joint_vel_diff = ref_robot_base_joint_vel - robot_base_joint_vel
    base_joint_vel_err = torch.sum(base_joint_vel_diff * base_joint_vel_diff, dim=-1)  # [B]
    arm_joint_vel_diff = ref_robot_arm_joint_vel - robot_arm_joint_vel
    arm_joint_vel_err = torch.sum(arm_joint_vel_diff * arm_joint_vel_diff, dim=-1)  # [B]
    finger_joint_vel_diff = ref_robot_finger_joint_vel - robot_finger_joint_vel
    finger_joint_vel_err = torch.sum(finger_joint_vel_diff * finger_joint_vel_diff, dim=-1)  # [B]
    return (
        key_body_pos_err,
        key_body_quat_err,
        door_err,
        base_joint_pos_err,
        arm_joint_pos_err,
        finger_joint_pos_err,
        arx_joint_pos_err,
        base_joint_vel_err,
        arm_joint_vel_err,
        finger_joint_vel_err,
    )

def compute_joint_limit_penalty(
    joint_pos: torch.Tensor,
    joint_lower_limits: torch.Tensor,
    joint_upper_limits: torch.Tensor,
    soft_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
    if soft_ratio <= 0.0:
        return zeros, zeros

    finite_limit_mask = torch.isfinite(joint_lower_limits) & torch.isfinite(joint_upper_limits)
    valid_limit_mask = finite_limit_mask & (joint_upper_limits > joint_lower_limits)

    joint_range = torch.where(valid_limit_mask, joint_upper_limits - joint_lower_limits, torch.ones_like(joint_lower_limits))
    soft_zone = torch.clamp(joint_range * soft_ratio, min=1e-6)

    dist_to_lower = joint_pos - joint_lower_limits.unsqueeze(0)
    dist_to_upper = joint_upper_limits.unsqueeze(0) - joint_pos
    dist_to_limit = torch.minimum(dist_to_lower, dist_to_upper)

    normalized_penalty = torch.clamp(
        (soft_zone.unsqueeze(0) - dist_to_limit) / soft_zone.unsqueeze(0),
        min=0.0,
        max=1.0,
    )
    valid_limit_mask = valid_limit_mask.unsqueeze(0)
    normalized_penalty = torch.where(valid_limit_mask, normalized_penalty, torch.zeros_like(normalized_penalty))

    num_tracked_joints = valid_limit_mask.to(joint_pos.dtype).sum(dim=-1).clamp(min=1.0)
    joint_limit_penalty = normalized_penalty.square().sum(dim=-1) / num_tracked_joints
    active_joint_fraction = (normalized_penalty > 0.0).to(joint_pos.dtype).sum(dim=-1) / num_tracked_joints
    return joint_limit_penalty, active_joint_fraction
