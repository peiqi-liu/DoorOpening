import inspect
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from DoorOpening.utils.quat_utils import quat_diff_angle, hinge_angle_diff
from DoorOpening.motion.motion_lib import ReferenceMotionManager
from DoorOpening.assets.door.door_cfg import edit_door_articulation
from DoorOpening.utils.finger_utils import joint_angle_to_tendon_utils, tendon_to_joint_angle_utils, leap_joints_to_tendon
from .dooropening_adr import DoorOpeningADR
from .dooropening_env_cfg import DooropeningEnvCfg
from DoorOpening.assets.door.door_cfg import motion_traj_paths, handle_offsets, board_offsets
from isaaclab.sensors import Camera, ContactSensor
from DoorOpening.constants.robot_constants import FULL_JOINT_NAMES, ROBOT_KEY_BODY_NAMES
from DoorOpening.utils.pose_utils import normalize_to_center_frame, world_to_local
from isaaclab.utils.math import quat_conjugate, quat_apply, quat_mul
from DoorOpening.utils.quat_utils import quat_to_euler, quat_to_6d
from typing import Tuple

import pickle as pkl
import math


class DooropeningEnv(DirectRLEnv):
    cfg: DooropeningEnvCfg

    def __init__(self, cfg: DooropeningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._initialize_runtime_event_terms()
        self.early_stopping = True

        self.num_base_joints = len(self.cfg.base_joints)
        self.num_arm_joints = len(self.cfg.arm_joints)

        actuated_joints = self.cfg.base_joints + self.cfg.arm_joints + self.cfg.finger_joints
        self._robot_dof_idx, joint_names = self.robot.find_joints(actuated_joints)
        self._robot_dof_idx = torch.tensor(self._robot_dof_idx, device=self.device)
        self.ref_robot_dof_idx = torch.tensor([FULL_JOINT_NAMES.index(name) for name in joint_names], device=self.device)

        self._robot_key_body_idx, robot_key_body_names = self.robot.find_bodies(self.cfg.robot_key_bodies)
        self._robot_reset_key_body_idx, robot_reset_key_body_names = self.robot.find_bodies(self.cfg.robot_reset_key_bodies)
        self._robot_base_id_in_key_body_idx = robot_key_body_names.index(self.cfg.robot_base_body_link_name)
        self._robot_palm_id_in_key_body_idx = robot_key_body_names.index(self.cfg.robot_palm_link_name)

        self.ref_key_body_idx = [ROBOT_KEY_BODY_NAMES.index(name) for name in robot_key_body_names]
        self.ref_reset_key_body_idx = [ROBOT_KEY_BODY_NAMES.index(name) for name in robot_reset_key_body_names]

        self._robot_base_dof_idx, base_joint_names = self.robot.find_joints(self.cfg.base_joints)
        self._robot_arm_dof_idx, arm_joint_names = self.robot.find_joints(self.cfg.arm_joints)
        self._robot_finger_dof_idx, finger_joint_names = self.robot.find_joints(self.cfg.finger_joints)
        self._robot_base_dof_idx = torch.tensor(self._robot_base_dof_idx, device=self.device)
        self._robot_arm_dof_idx = torch.tensor(self._robot_arm_dof_idx, device=self.device)
        self._robot_finger_dof_idx = torch.tensor(self._robot_finger_dof_idx, device=self.device)
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

        robot_abduction_dof_idx, abduction_joint_names = self.robot.find_joints(self.cfg.abduction_joints)
        self.robot_abduction_default_pos = self.robot.data.default_joint_pos[..., robot_abduction_dof_idx]
        self.finger_dof_names_to_id = {name: idx for idx, name in enumerate(finger_joint_names)}
        # self.robot_abduction_dof_idx_in_targets = [self.finger_dof_names_to_id[name] + self.num_base_joints + self.num_arm_joints for name in self.cfg.abduction_joints]
        # self.close_finger_joints = torch.tensor([self.cfg.close_finger_joints[name] for name in finger_joint_names], device=self.device)
        # self.open_finger_joints = torch.tensor([self.cfg.open_finger_joints[name] for name in finger_joint_names], device=self.device)

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
        self._target_finger_slice = slice(self.num_base_joints + self.num_arm_joints, self.robot_dof_targets.shape[1])

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
        self.robot_base_joint_vel_w = self.cfg.robot_base_joint_vel_w
        self.robot_arm_joint_vel_w = self.cfg.robot_arm_joint_vel_w
        self.robot_finger_joint_vel_w = self.cfg.robot_finger_joint_vel_w
        self.hinge_contact_reward_w = self.cfg.hinge_contact_reward_w
        self.robot_body_lin_vel_w = self.cfg.robot_body_lin_vel_w
        self.robot_body_ang_vel_w = self.cfg.robot_body_ang_vel_w
        self.joint_limit_penalty_w = self.cfg.joint_limit_penalty_w
        self.joint_limit_penalty_margin_ratio = self.cfg.joint_limit_penalty_margin_ratio

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

        # self.ref_motion_lib = ReferenceMotionManager(self.cfg.motion_file, self.num_envs, self.device, velocity=self.cfg.velocity, reset_from_start = True)
        self.num_door_assets = len(handle_offsets)
        self.handle_offsets = [handle_offsets[i % len(handle_offsets)] for i in range(self.num_envs)]
        self.board_offsets = [board_offsets[i % len(board_offsets)] for i in range(self.num_envs)]
        self.handle_offsets = torch.stack(self.handle_offsets).to(self.device)
        self.board_offsets = torch.stack(self.board_offsets).to(self.device)
        env_to_file_map = [i % len(motion_traj_paths) for i in range(self.num_envs)]
        self.ref_motion_lib = ReferenceMotionManager(num_envs=self.num_envs, device=self.device, velocity=self.cfg.velocity, reset_from_start = False, env_to_file_map=env_to_file_map, twist_indices=self.twist_indices)
        self.prob_get_first_key_frame = None
        self.max_trial_steps = self.ref_motion_lib.num_frames * torch.ones_like(self.episode_length_buf, device=self.device)

        torch.set_printoptions(precision=4, sci_mode=False)

        self.step_count = 0
        self.reset_progress_total = self.cfg.reset_progress_total
        self.adr_reset_progress_total = self.cfg.adr_reset_progress_total

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
        self.robot_state_noise_widths = self._make_env_buffer_dict(
            [key for key in robot_state_cfg if key.endswith("_noise")]
        )
        self.robot_state_biases = self._make_env_buffer_dict([key for key in robot_state_cfg if key.endswith("_bias")])
        self._door_nominal_joint_stiffness = self.door.data.joint_stiffness.clone()
        self._door_nominal_joint_damping = self.door.data.joint_damping.clone()
        self._dr_metrics_interval = max(int(self.cfg.dr_metrics_interval), 1)
        self._log_verbose_dr_metrics = bool(self.cfg.log_verbose_dr_metrics)

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
            for link_name in ("link_1", "link_2"):
                activate_contact_sensors(f"/World/envs/env_{env_id}/Door/{link_name}", True)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.door = Articulation(self.cfg.door_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["door"] = self.door
        self._activate_door_contact_reporters()
        self.pointcloud_camera = None
        if self.cfg.enable_pointcloud_camera:
            self.pointcloud_camera = Camera(self.cfg.pointcloud_camera_cfg)
            self.scene.sensors["pointcloud_camera"] = self.pointcloud_camera
        # self.scene.sensors["contact_forces_door1"] = ContactSensor(self.cfg.contact_forces_door1)
        self.scene.sensors["contact_forces_door2"] = ContactSensor(self.cfg.contact_forces_door2)
        # self.scene.sensors["contact_forces_robot_palm_center"] = ContactSensor(self.cfg.contact_forces_robot_palm_center)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)    

    def _make_env_buffer_dict(self, keys):
        return {key: torch.zeros((self.num_envs, 1), device=self.device) for key in keys}

    def _current_custom_param(self, group: str, name: str) -> float:
        if not self._adr_enabled:
            return 0.0
        return self.dooropening_adr.get_custom_param_value(group, name)

    def _current_event_param(self, term_name: str, param_name: str):
        return self.event_manager.get_term_cfg(term_name).params[param_name]

    def _refresh_nominal_door_joint_gains(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._door_nominal_joint_stiffness.copy_(self.door.data.joint_stiffness)
            self._door_nominal_joint_damping.copy_(self.door.data.joint_damping)
            return
        self._door_nominal_joint_stiffness[env_ids] = self.door.data.joint_stiffness[env_ids]
        self._door_nominal_joint_damping[env_ids] = self.door.data.joint_damping[env_ids]

    def _compute_curriculum_progress(self, progress_total: float) -> float:
        progress_total = max(float(progress_total), 1.0)
        return min(float(self.step_count) / progress_total, 1.0)

    def _log_dr_metrics(self):
        if self.step_count % self._dr_metrics_interval != 0:
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

        self.extras["dr/increment"] = float(self.dooropening_adr.increment_counter)
        # self.extras["dr/fraction"] = self.dooropening_adr.get_increment_fraction()
        # self.extras["dr/scheduled_increment_from_step"] = float(scheduled_increment)
        # self.extras["dr/step_count"] = int(self.step_count)
        # self.extras["dr/common_env_step_count"] = int(self.common_step_counter)
        # self.extras["dr/rlgames_frame_equivalent_from_sim_steps"] = float(
        #     self.step_count * self.num_envs / max(self.cfg.decimation, 1)
        # )
        # self.extras["dr/frame_per_sim_step_expected"] = float(self.num_envs / max(self.cfg.decimation, 1))
        # self.extras["dr/scheduled_fraction_from_step"] = progress
        self.extras["dr/robot_stiffness_min"] = float(robot_stiffness[0])
        self.extras["dr/robot_stiffness_max"] = float(robot_stiffness[1])
        self.extras["dr/robot_damping_min"] = float(robot_damping[0])
        self.extras["dr/robot_damping_max"] = float(robot_damping[1])
        self.extras["dr/door_board_stiffness_min"] = float(board_stiffness[0])
        self.extras["dr/door_board_stiffness_max"] = float(board_stiffness[1])
        self.extras["dr/door_board_damping_min"] = float(board_damping[0])
        self.extras["dr/door_board_damping_max"] = float(board_damping[1])
        self.extras["dr/door_hinge_stiffness_min"] = float(hinge_stiffness[0])
        self.extras["dr/door_hinge_stiffness_max"] = float(hinge_stiffness[1])
        self.extras["dr/door_hinge_damping_min"] = float(hinge_damping[0])
        self.extras["dr/door_hinge_damping_max"] = float(hinge_damping[1])

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
        self.extras["dr_limit/target_lag_alpha"] = self._current_custom_param("pd_targets", "target_lag_alpha")

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

    def _uniform_noise_like(self, values: torch.Tensor, width_key: str, bias_key: str | None = None):
        width = self.robot_state_noise_widths[width_key]
        while width.dim() < values.dim():
            width = width.unsqueeze(-1)
        noise = width * 2.0 * (torch.rand_like(values) - 0.5)
        if bias_key is None:
            return values + noise
        bias = self.robot_state_biases[bias_key]
        while bias.dim() < values.dim():
            bias = bias.unsqueeze(-1)
        return values + noise + bias

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

    def _pre_physics_step(self, actions: torch.Tensor):
        self.step_count = self._sim_step_counter
        # delta actions
        self.scaled_actions = actions.clamp(-1.0, 1.0)
        self.scaled_actions[:, :self.num_base_joints] = self.scaled_actions[:, :self.num_base_joints] * self.cfg.base_action_scale
        self.scaled_actions[:, self.num_base_joints:self.num_base_joints + self.num_arm_joints] = self.scaled_actions[:, self.num_base_joints:self.num_base_joints + self.num_arm_joints] * self.cfg.arm_action_scale
        self.scaled_actions[:, self.num_base_joints + self.num_arm_joints:] = self.scaled_actions[:, self.num_base_joints + self.num_arm_joints:] * self.cfg.finger_action_scale
        targets = self.robot_dof_targets + self.dt * self.scaled_actions
        # Optional: lock the abduction joints
        # targets[..., self.robot_abduction_dof_idx_in_targets] = self.robot_abduction_default_pos
        # targets[..., self.num_base_joints + self.num_arm_joints:] = torch.where( \
        #     (torch.linalg.norm(targets[..., self.num_base_joints + self.num_arm_joints:] - self.close_finger_joints, dim=-1) < \
        #     torch.linalg.norm(targets[..., self.num_base_joints + self.num_arm_joints:] - self.open_finger_joints, dim=-1)).unsqueeze(-1), \
        #     self.close_finger_joints[None, :], \
        #     self.open_finger_joints[None, :] \
        # )
        # Optional: use tendon actions to control the finger joints
        # tendon_actions = leap_joints_to_tendon(targets[..., self.num_base_joints + self.num_arm_joints:], self.finger_dof_names_to_id, device=self.device)
        # targets[..., self.num_base_joints + self.num_arm_joints:] = tendon_to_joint_angle_utils(self.robot, tendon_actions)[..., self._robot_finger_dof_idx]

        # self.scene.sensors["contact_forces_door1"].update(self.cfg.sim_dt, force_recompute=True)
        self.scene.sensors["contact_forces_door2"].update(self.cfg.sim_dt)
        # self.scene.sensors["contact_forces_robot_palm_center"].update(self.cfg.sim_dt, force_recompute=True)

        # print("robot body lin vel: ", self.robot.data.body_link_lin_vel_w[0, self._robot_key_body_idx])
        # print("robot body ang vel: ", self.robot.data.body_link_ang_vel_w[0, self._robot_key_body_idx])
        # print("ref robot body lin vel: ", self.ref_robot_body_lin_vel[0, self.ref_key_body_idx] / self.cfg.sim_dt)
        # print("ref robot body ang vel: ", self.ref_robot_body_ang_vel[0, self.ref_key_body_idx] / self.cfg.sim_dt)
        # print("joint_vel: ", self.robot.data.joint_vel[:, self._robot_base_dof_idx])
        # print("ref joint vel: ", self.ref_robot_base_joint_vel)

        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # self.last_actions[:] = self.scaled_actions

    def _apply_action(self):
        edit_door_articulation(
            self.door,
            nominal_joint_stiffness=self._door_nominal_joint_stiffness,
            nominal_joint_damping=self._door_nominal_joint_damping,
        )
        self.ref_motion_lib.step()
        lag_alpha = self._current_custom_param("pd_targets", "target_lag_alpha")
        applied_targets = self.robot_dof_targets.clone()
        if lag_alpha > 0.0:
            applied_targets = (1.0 - lag_alpha) * applied_targets + lag_alpha * self.applied_robot_dof_targets
        # Controller-side DR is applied on the position targets after action integration.
        applied_targets += self._get_policy_target_noise()
        self.applied_robot_dof_targets[:] = torch.clamp(
            applied_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.robot.set_joint_position_target(self.applied_robot_dof_targets, joint_ids=self._robot_dof_idx)
        # joint_pos = self.robot.data.default_joint_pos.clone()
        # joint_pos[:] = self.ref_motion_lib.get_robot_joint_pos()
        # self.robot.write_joint_position_to_sim(joint_pos)
        # door_pos = self.door.data.joint_pos.clone()
        # door_pos[:] = self.ref_motion_lib.get_door_joint_pos()
        # self.door.write_joint_position_to_sim(door_pos)

    def _get_observations(self) -> dict:
        self._get_intermediate_values()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        door_to_base_link_pos = world_to_local(self.door_link_pos, self.robot_base_body_pos, self.robot_base_body_quat).reshape(self.num_envs, 1, -1)
        door_twist_in_robot_base_frame = world_to_local(
            self.ref_door_body_pos_twist,
            self.robot_base_body_pos,
            self.robot_base_body_quat,
        ).reshape(self.num_envs, 1, -1)
        robot_key_body_pos, robot_key_body_euler, base_lin_vel_local, base_ang_vel_local = self.transform_key_bodies_to_base_frame(self.robot_key_body_pos, self.robot_key_body_quat, self.robot_body_lin_vel, self.robot_body_ang_vel, self._robot_base_body_link_idx)
        key_pos_err = (self.ref_robot_key_body_pos).to(self.robot_key_body_pos) - self.robot_key_body_pos
        key_pos_err = world_to_local(key_pos_err, None, self.robot_base_body_quat)
        key_pos_err = key_pos_err.reshape(self.num_envs, 1, -1)

        clean_joint_pos = self.joint_pos[:, self._robot_dof_idx]
        clean_joint_vel = self.joint_vel[:, self._robot_dof_idx]
        policy_joint_pos = clean_joint_pos.clone()
        policy_joint_vel = clean_joint_vel.clone()

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

        clean_base_joint_ref_err = (
            self.ref_robot_base_joint_pos
            - clean_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop]
        ).unsqueeze(dim=1)
        clean_arm_joint_ref_err = (self.ref_robot_arm_joint_pos - clean_joint_pos[:, self._target_arm_slice]).unsqueeze(
            dim=1
        )
        policy_base_joint_ref_err = (
            self.ref_robot_base_joint_pos
            - policy_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop]
        ).unsqueeze(dim=1)
        policy_arm_joint_ref_err = (self.ref_robot_arm_joint_pos - policy_joint_pos[:, self._target_arm_slice]).unsqueeze(
            dim=1
        )

        twist_obs = torch.cat(
            (
                self.ref_robot_key_body_pos_twist.reshape(self.num_envs, 1, -1),
                self.ref_robot_key_body_quat_twist.reshape(self.num_envs, 1, -1),
                self.ref_door_joint_pos_twist.reshape(self.num_envs, 1, -1),
                (
                    self.ref_robot_base_joint_pos_twist - clean_joint_pos[:, self._target_base_rot_slice.start : self._target_base_xy_slice.stop].unsqueeze(dim=1)
                ).reshape(self.num_envs, 1, -1),
                (
                    self.ref_robot_arm_joint_pos_twist - clean_joint_pos[:, self._target_arm_slice].unsqueeze(dim=1)
                ).reshape(self.num_envs, 1, -1),
                door_twist_in_robot_base_frame,
            ),
            dim=-1,
        )

        policy_obs = torch.cat(
            (
                policy_joint_pos.unsqueeze(dim=1),
                policy_joint_vel.unsqueeze(dim=1),
                self.robot_dof_targets.unsqueeze(dim = 1),
                policy_base_joint_ref_err,
                policy_arm_joint_ref_err,
                key_pos_err,
                robot_key_body_pos.reshape(self.num_envs, 1, -1),
                robot_key_body_euler.reshape(self.num_envs, 1, -1),
                base_lin_vel_local.reshape(self.num_envs, 1, -1),
                base_ang_vel_local.reshape(self.num_envs, 1, -1),
                door_to_base_link_pos,
                self.door_joint_pos[:, self._door_joint_idx].unsqueeze(dim = 1),
                self.ref_door_joint_pos[:, self._door_joint_idx].to(self.door_joint_pos).unsqueeze(dim = 1),
                twist_obs,
            ),
            dim=-1,
        )

        critic_obs = torch.cat(
            (
                clean_joint_pos.unsqueeze(dim=1),
                clean_joint_vel.unsqueeze(dim=1),
                self.robot_dof_targets.unsqueeze(dim=1),
                clean_base_joint_ref_err,
                clean_arm_joint_ref_err,
                key_pos_err,
                robot_key_body_pos.reshape(self.num_envs, 1, -1),
                robot_key_body_euler.reshape(self.num_envs, 1, -1),
                base_lin_vel_local.reshape(self.num_envs, 1, -1),
                base_ang_vel_local.reshape(self.num_envs, 1, -1),
                door_to_base_link_pos,
                # door_to_palm_link_pos,
                # self.door_link_pos.reshape(self.num_envs, 1, -1),
                self.door_joint_pos[:, self._door_joint_idx].unsqueeze(dim = 1),
                # self.door_joint_vel[:, self._door_joint_idx].unsqueeze(dim = 1),
                self.ref_door_joint_pos[:, self._door_joint_idx].to(self.door_joint_pos).unsqueeze(dim = 1),
                twist_obs,
            ),
            dim=-1,
        )

        # The actor sees noisy deployment-like inputs; the critic keeps the clean privileged state.
        policy_obs = policy_obs.squeeze(1)
        critic_obs = critic_obs.squeeze(1)

        observations = {"policy": policy_obs, "critic": critic_obs}
        return observations


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
        self.door_joint_pos = self.door.data.joint_pos
        # self.door_joint_vel = self.door.data.joint_vel
        self.robot_base_joint_vel = self.robot.data.joint_vel[:, self._robot_base_dof_idx]
        self.robot_arm_joint_vel = self.robot.data.joint_vel[:, self._robot_arm_dof_idx]
        self.robot_finger_joint_vel = self.robot.data.joint_vel[:, self._robot_finger_dof_idx]

        self.door_base_link_pos = self.door.data.body_pos_w[:, self._door_base_link_idx]
        self.door_base_link_quat = self.door.data.body_quat_w[:, self._door_base_link_idx]
        self.door_link_pos = self.door.data.body_pos_w[:, self._door_body_idx]
        self.door_link_pos -= self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)
        self.door_link_quat = self.door.data.body_quat_w[:, self._door_body_idx]
        # self.door_keypoints = self.compute_door_keypoints()
        self.robot_body_lin_vel = self.robot.data.body_link_lin_vel_w[:, self._robot_key_body_idx]
        self.robot_body_ang_vel = self.robot.data.body_link_ang_vel_w[:, self._robot_key_body_idx]
        # print("door keypoints: ", self.compute_door_keypoints())
        # print("door link pos: ", self.door_link_pos)

        self.ref_robot_key_body_pos_twist = self.ref_motion_lib.get_robot_body_pos_twist()[:, :, self.ref_key_body_idx]
        # It is a misnomer, we are actually sending euler angles as it might be more friendly to MLP
        self.ref_robot_key_body_quat_twist = self.ref_motion_lib.get_robot_body_quat_twist()[:, :, self.ref_key_body_idx]
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
        self.ref_door_joint_pos_twist = self.ref_motion_lib.get_door_joint_pos_twist()

        # self.ref_robot_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_key_body_idx]
        # self.ref_robot_key_body_quat = self.ref_motion_lib.get_robot_body_quat()[:, self._robot_key_body_idx]
        # self.ref_robot_reset_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_reset_key_body_idx]
        self.ref_robot_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self.ref_key_body_idx]
        self.ref_robot_key_body_quat = self.ref_motion_lib.get_robot_body_quat()[:, self.ref_key_body_idx]
        self.ref_robot_reset_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self.ref_reset_key_body_idx]
        self.ref_robot_joint_pos = self.ref_motion_lib.get_robot_joint_pos()
        self.ref_robot_base_joint_pos = self.ref_robot_joint_pos[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_pos = self.ref_robot_joint_pos[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_pos = self.ref_robot_joint_pos[:, self.ref_finger_joint_idx]
        self.ref_joint_vel = self.ref_motion_lib.get_robot_joint_vel()
        self.ref_robot_base_joint_vel = self.ref_joint_vel[:, self.ref_base_joint_idx]
        self.ref_robot_arm_joint_vel = self.ref_joint_vel[:, self.ref_arm_joint_idx]
        self.ref_robot_finger_joint_vel = self.ref_joint_vel[:, self.ref_finger_joint_idx]
        self.ref_door_joint_pos = self.ref_motion_lib.get_door_joint_pos()
        self.ref_hinge_contact_mask = self.ref_motion_lib.get_hinge_contact_mask()
        self.ref_door_body_pos_twist = self.ref_motion_lib.get_door_body_pos_twist()
        self.ref_robot_body_lin_vel = self.ref_motion_lib.get_robot_body_lin_vel()[:, self.ref_key_body_idx] / self.cfg.sim_dt
        self.ref_robot_body_ang_vel = self.ref_motion_lib.get_robot_body_ang_vel()[:, self.ref_key_body_idx] / self.cfg.sim_dt

    def _get_rewards(self) -> torch.Tensor:
        self._get_intermediate_values()
        self._log_dr_metrics()

        # key_body_pos_err, key_body_quat_err, door_err, root_pos_err, root_rot_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err, door_pos_err = compute_tracking_error(
        key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err = compute_tracking_error(
            robot_key_body_pos = self.robot_key_body_pos,
            robot_key_body_quat = self.robot_key_body_quat,
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos,
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,

            ref_robot_key_body_pos = self.ref_robot_key_body_pos,
            ref_robot_key_body_quat = self.ref_robot_key_body_quat,
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
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
        # self.extras["error/base_joint_vel_err"] = base_joint_vel_err.mean()
        # self.extras["error/arm_joint_vel_err"] = arm_joint_vel_err.mean()
        # self.extras["error/finger_joint_vel_err"] = finger_joint_vel_err.mean()

        # progress = min(self.step_count / self.reset_progress_total, 1.0)
        # alpha = 1 - 0.1**(2 ** (2.0 - 4.0 * progress))
        # probs = torch.tensor(
        #     [(1 - alpha) * (alpha ** i) for i in range(self.ref_motion_lib.key_indices.shape[1])],
        #     device=self.ref_motion_lib.key_indices.device
        # )
        # probs = probs / probs.sum()
        # self.extras["reset/prob_get_first_key_frame"] = probs[0]
        if self.prob_get_first_key_frame is not None:
            self.extras["reset/prob_get_first_key_frame"] = float(self.prob_get_first_key_frame)

        # contact_forces_robot_palm_center = self.scene.sensors["contact_forces_robot_palm_center"].data.net_forces_w
        contact_forces_door2 = self.scene.sensors["contact_forces_door2"].data.net_forces_w

        deep_mimic_reward = compute_deep_mimic_rewards(
            robot_key_body_pos = self.robot_key_body_pos, 
            robot_key_body_quat = self.robot_key_body_quat, 
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos, 
            robot_finger_joint_pos = self.robot_finger_joint_pos,
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
            robot_base_joint_vel_w = self.robot_base_joint_vel_w,
            robot_arm_joint_vel_w = self.robot_arm_joint_vel_w,
            robot_finger_joint_vel_w = self.robot_finger_joint_vel_w,
            robot_body_lin_vel_w = self.robot_body_lin_vel_w,
            robot_body_ang_vel_w = self.robot_body_ang_vel_w,

            contact_forces = contact_forces_door2,
            contact_force_w = self.hinge_contact_reward_w * self.ref_hinge_contact_mask,
        )

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

        # 1. Base Alive Reward: Small constant for staying in the safety tunnel
        alive_base = self.alive_base 
        
        # 2. Difficulty Bonus: Extra points for staying alive during contact
        # self.ref_hinge_contact_mask is 1.0 when grasping/pulling
        alive_bonus = self.alive_bonus * self.ref_hinge_contact_mask.squeeze()
        
        total_alive_reward = alive_base + alive_bonus

        # 3. Combine with tracking reward and termination penalty
        is_killed, _ = self._get_dones()
        termination_penalty = self.termination_penalty
        
        final_reward = deep_mimic_reward + total_alive_reward - weighted_joint_limit_penalty
        final_reward = torch.where(is_killed, final_reward + termination_penalty, final_reward)

        return final_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_trial_steps - 1
        if not self.early_stopping:
            return False, time_out
        self._get_intermediate_values()
        progress = min(self.step_count / self.reset_progress_total, 1.0)
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
        # key_body_pos_err, key_body_quat_err, door_err, root_pos_err, root_rot_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err, door_pos_err = compute_tracking_error(
        key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err = compute_tracking_error(
            robot_key_body_pos = self.robot_reset_key_body_pos,
            robot_key_body_quat = self.robot_key_body_quat,
            door_joint_pos = self.door_joint_pos,
            robot_base_joint_pos = self.robot_base_joint_pos,
            robot_arm_joint_pos = self.robot_arm_joint_pos,
            robot_finger_joint_pos = self.robot_finger_joint_pos,
            robot_base_joint_vel = self.robot_base_joint_vel,
            robot_arm_joint_vel = self.robot_arm_joint_vel,
            robot_finger_joint_vel = self.robot_finger_joint_vel,

            ref_robot_key_body_pos = self.ref_robot_reset_key_body_pos,
            ref_robot_key_body_quat = self.ref_robot_key_body_quat,
            ref_door_joint_pos = self.ref_door_joint_pos,
            ref_robot_base_joint_pos = self.ref_robot_base_joint_pos,
            ref_robot_arm_joint_pos = self.ref_robot_arm_joint_pos,
            ref_robot_finger_joint_pos = self.ref_robot_finger_joint_pos,
            ref_robot_base_joint_vel = self.ref_robot_base_joint_vel,
            ref_robot_arm_joint_vel = self.ref_robot_arm_joint_vel,
            ref_robot_finger_joint_vel = self.ref_robot_finger_joint_vel,
        )
        return \
            (key_body_pos_err > reset_key_body_pos_delta) | \
            (key_body_quat_err > reset_key_body_quat_delta) | \
            (door_err > reset_door_joint_pos_delta), \
            time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._update_adr_ranges()
        reset_frame_idx, self.prob_get_first_key_frame = self.ref_motion_lib.reset(env_ids, step_count=self.step_count, reset_progress_total=self.reset_progress_total)
        self.max_trial_steps[env_ids] = ((self.ref_motion_lib.num_frames - reset_frame_idx) // self.ref_motion_lib.velocity).long()
        self._sample_reset_randomization(env_ids)

        deep_mimic_initial_joint_pos = self.ref_motion_lib.get_robot_joint_pos(env_ids)
        deep_mimic_initial_joint_vel = self.ref_motion_lib.get_robot_joint_vel(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.default_joint_vel[env_ids]
        self.joint_vel[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_vel.to(self.joint_vel)[..., self.ref_robot_dof_idx]
        self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_pos.to(self.joint_pos)[..., self.ref_robot_dof_idx]
        self._apply_spawn_noise(env_ids)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], None, env_ids)
        self.robot.set_joint_position_target(self.joint_pos[env_ids], env_ids=env_ids)

        door_joint_pos = self.ref_motion_lib.get_door_joint_pos(env_ids).to(self.door.data.joint_pos)
        door_joint_vel = self.door.data.default_joint_vel[env_ids].clone()

        self.door.write_joint_state_to_sim(door_joint_pos, door_joint_vel, None, env_ids)
        self.door.set_joint_position_target(door_joint_pos, None, env_ids)

        # self.last_actions[env_ids] = 0.0
        self.robot_dof_targets[env_ids, :] = self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]]
        self.applied_robot_dof_targets[env_ids, :] = self.robot_dof_targets[env_ids, :]
        super()._reset_idx(env_ids)
        self._refresh_nominal_door_joint_gains(env_ids)

@torch.jit.script
def compute_deep_mimic_rewards(
    robot_key_body_pos: torch.Tensor,
    robot_key_body_quat: torch.Tensor,
    door_joint_pos: torch.Tensor,
    robot_base_joint_pos: torch.Tensor,
    robot_arm_joint_pos: torch.Tensor,
    robot_finger_joint_pos: torch.Tensor,
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
    robot_base_joint_vel_w: float,
    robot_arm_joint_vel_w: float,
    robot_finger_joint_vel_w: float,
    robot_body_lin_vel_w: float,
    robot_body_ang_vel_w: float,

    contact_force_w: torch.Tensor,
    contact_forces: torch.Tensor,
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

    contact_reward = torch.where(torch.norm(contact_forces, dim=-1) > 1, 1.0, 0.0).squeeze()
    # ----------------------------------
    # Final reward
    # ----------------------------------
    reward = robot_key_body_pos_w * key_body_pos_r\
         + robot_key_body_quat_w * key_body_quat_r\
         + door_joint_pos_w * door_r\
         + robot_base_joint_pos_w * base_joint_pos_r\
         + robot_arm_joint_pos_w * arm_joint_pos_r\
         + robot_finger_joint_pos_w * finger_joint_pos_r\
         + robot_base_joint_vel_w * base_joint_vel_r\
         + robot_arm_joint_vel_w * arm_joint_vel_r\
         + robot_finger_joint_vel_w * finger_joint_vel_r\
         + robot_body_lin_vel_w * robot_body_lin_vel_r\
         + robot_body_ang_vel_w * robot_body_ang_vel_r\
         + contact_force_w * contact_reward
    return reward

def compute_tracking_error(
    robot_key_body_pos: torch.Tensor,
    robot_key_body_quat: torch.Tensor,
    door_joint_pos: torch.Tensor,
    robot_base_joint_pos: torch.Tensor,
    robot_arm_joint_pos: torch.Tensor,
    robot_finger_joint_pos: torch.Tensor,
    robot_base_joint_vel: torch.Tensor,
    robot_arm_joint_vel: torch.Tensor,
    robot_finger_joint_vel: torch.Tensor,

    ref_robot_key_body_pos: torch.Tensor,
    ref_robot_key_body_quat: torch.Tensor,
    ref_door_joint_pos: torch.Tensor,
    ref_robot_base_joint_pos: torch.Tensor,
    ref_robot_arm_joint_pos: torch.Tensor,
    ref_robot_finger_joint_pos: torch.Tensor,
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
    # print("key_body_pos_diff: ", key_body_pos_err)
    key_body_pos_err = torch.max(key_body_pos_err, dim=-1).values
    # print("key_body_pos_err: ", key_body_pos_err)
    # ----------------------------------
    # Robot body orientation error
    # ----------------------------------
    # [B, N]
    key_body_quat_diff = quat_diff_angle(robot_key_body_quat, ref_robot_key_body_quat)
    # print("key_body_quat_diff: ", key_body_quat_diff)
    # key_body_quat_err = torch.sum(key_body_quat_diff * key_body_quat_diff, dim=-1)  # [B]
    key_body_quat_err = torch.max(key_body_quat_diff * key_body_quat_diff, dim=-1).values
    # print("key_body_quat_err: ", key_body_quat_err)
    # ----------------------------------
    # Door joint error
    # ----------------------------------
    door_diff = ref_door_joint_pos - door_joint_pos
    # print("door_diff: ", door_diff)
    # door_err = torch.sum(door_diff * door_diff, dim=-1)  # [B]
    door_err = torch.max(door_diff * door_diff, dim=-1).values
    # print("door_err: ", door_err)
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
    base_joint_vel_diff = ref_robot_base_joint_vel - robot_base_joint_vel
    base_joint_vel_err = torch.sum(base_joint_vel_diff * base_joint_vel_diff, dim=-1)  # [B]
    arm_joint_vel_diff = ref_robot_arm_joint_vel - robot_arm_joint_vel
    arm_joint_vel_err = torch.sum(arm_joint_vel_diff * arm_joint_vel_diff, dim=-1)  # [B]
    finger_joint_vel_diff = ref_robot_finger_joint_vel - robot_finger_joint_vel
    finger_joint_vel_err = torch.sum(finger_joint_vel_diff * finger_joint_vel_diff, dim=-1)  # [B]
    return (key_body_pos_err, key_body_quat_err, door_err, base_joint_pos_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err)

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
