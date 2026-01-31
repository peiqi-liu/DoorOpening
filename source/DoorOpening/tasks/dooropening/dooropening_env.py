# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
from .dooropening_env_cfg import DooropeningEnvCfg
from DoorOpening.assets.door.door_cfg import motion_traj_paths

import pickle as pkl
import math


class DooropeningEnv(DirectRLEnv):
    cfg: DooropeningEnvCfg

    def __init__(self, cfg: DooropeningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.early_stopping = True

        self.num_base_joints = len(self.cfg.base_joints)
        self.num_arm_joints = len(self.cfg.arm_joints)

        actuated_joints = self.cfg.base_joints + self.cfg.arm_joints + self.cfg.finger_joints
        self._robot_dof_idx, joint_names = self.robot.find_joints(actuated_joints)

        deep_mimic_joints = self.cfg.base_joints + self.cfg.arm_joints
        self._robot_deep_mimic_dof_idx, _ = self.robot.find_joints(deep_mimic_joints)

        self._robot_key_body_idx, _ = self.robot.find_bodies(self.cfg.robot_key_bodies)
        self._robot_reset_key_body_idx, _ = self.robot.find_bodies(self.cfg.robot_reset_key_bodies)

        self._robot_base_dof_idx, base_joint_names = self.robot.find_joints(self.cfg.base_joints)
        self._robot_arm_dof_idx, arm_joint_names = self.robot.find_joints(self.cfg.arm_joints)
        self._robot_finger_dof_idx, finger_joint_names = self.robot.find_joints(self.cfg.finger_joints)
        self._robot_abduction_dof_idx, abduction_joint_names = self.robot.find_joints(self.cfg.abduction_joints)
        self.robot_abduction_default_pos = self.robot.data.default_joint_pos[..., self._robot_abduction_dof_idx]
        self.finger_dof_names_to_id = {name: idx for idx, name in enumerate(finger_joint_names)}

        self._robot_base_link_idx, self.robot_base_link_name = self.robot.find_bodies(self.cfg.base_link_name)
        self._door_body_idx, _ = self.door.find_bodies(self.cfg.door_body_names)
        self._door_joint_idx, _ = self.door.find_joints(self.cfg.door_joint_names)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 1].to(device=self.device)

        # We are going to update this variables to control the robot
        self.robot_dof_targets = torch.zeros((self.num_envs, len(self._robot_dof_idx)), device=self.device)

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

        self.robot_key_body_pos_w = self.cfg.robot_key_body_pos_w
        self.robot_body_quat_w = self.cfg.robot_body_quat_w
        self.door_joint_pos_w = self.cfg.door_joint_pos_w
        self.robot_base_joint_pos_w = self.cfg.robot_base_joint_pos_w
        self.robot_arm_joint_pos_w = self.cfg.robot_arm_joint_pos_w
        self.robot_finger_joint_pos_w = self.cfg.robot_finger_joint_pos_w
        self.robot_base_joint_vel_w = self.cfg.robot_base_joint_vel_w
        self.robot_arm_joint_vel_w = self.cfg.robot_arm_joint_vel_w
        self.robot_finger_joint_vel_w = self.cfg.robot_finger_joint_vel_w

        # self.reset_base_pos_delta = (self.cfg.reset_base_pos_delta ** 2) * len(self.cfg.base_joints)
        # self.reset_key_body_pos_delta = (self.cfg.reset_key_body_pos_delta ** 2) * len(self.cfg.robot_reset_key_bodies)
        # self.reset_key_body_quat_delta = (self.cfg.reset_key_body_quat_delta ** 2) * len(self.cfg.robot_reset_key_bodies)

        # self.reset_base_pos_delta_min = (self.cfg.reset_base_pos_delta_min ** 2) * len(self.cfg.base_joints)
        # self.reset_key_body_pos_delta_min = (self.cfg.reset_key_body_pos_delta_min ** 2) * len(self.cfg.robot_reset_key_bodies)
        # self.reset_key_body_quat_delta_min = (self.cfg.reset_key_body_quat_delta_min ** 2) * len(self.cfg.robot_reset_key_bodies)
        # self.reset_base_pos_delta_max = (self.cfg.reset_base_pos_delta_max ** 2) * len(self.cfg.base_joints)
        # self.reset_key_body_pos_delta_max = (self.cfg.reset_key_body_pos_delta_max ** 2) * len(self.cfg.robot_reset_key_bodies)
        # self.reset_key_body_quat_delta_max = (self.cfg.reset_key_body_quat_delta_max ** 2) * len(self.cfg.robot_reset_key_bodies)
        self.reset_base_pos_delta_min = self.cfg.reset_base_pos_delta_min
        self.reset_key_body_pos_delta_min = self.cfg.reset_key_body_pos_delta_min
        self.reset_key_body_quat_delta_min = self.cfg.reset_key_body_quat_delta_min
        self.reset_base_pos_delta_max = self.cfg.reset_base_pos_delta_max
        self.reset_key_body_pos_delta_max = self.cfg.reset_key_body_pos_delta_max
        self.reset_key_body_quat_delta_max = self.cfg.reset_key_body_quat_delta_max
        self.reset_door_joint_pos_delta_min = self.cfg.reset_door_joint_pos_delta_min
        self.reset_door_joint_pos_delta_max = self.cfg.reset_door_joint_pos_delta_max

        # self.ref_motion_lib = ReferenceMotionManager(self.cfg.motion_file, self.num_envs, self.device, velocity=self.cfg.velocity, reset_from_start = True)
        env_to_file_map = [i % len(motion_traj_paths) for i in range(self.num_envs)]
        self.ref_motion_lib = ReferenceMotionManager(num_envs=self.num_envs, device=self.device, velocity=self.cfg.velocity, reset_from_start = True, env_to_file_map=env_to_file_map)
        self.max_trial_steps = self.ref_motion_lib.num_frames * torch.ones_like(self.episode_length_buf, device=self.device)

        torch.set_printoptions(precision=4, sci_mode=False)

        self.step_count = 0
        self.reset_progress_total = self.cfg.reset_progress_total

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
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)    

    def _pre_physics_step(self, actions: torch.Tensor):
        self.step_count = self._sim_step_counter
        # delta actions
        self.scaled_actions = actions.clone().clamp(-1.0, 1.0)
        # targets = self.robot_dof_targets + self.dt * self.actions * self.cfg.action_scale
        self.scaled_actions[:, :self.num_base_joints] = self.scaled_actions[:, :self.num_base_joints] * self.cfg.base_action_scale
        self.scaled_actions[:, self.num_base_joints:self.num_base_joints + self.num_arm_joints] = self.scaled_actions[:, self.num_base_joints:self.num_base_joints + self.num_arm_joints] * self.cfg.arm_action_scale
        self.scaled_actions[:, self.num_base_joints + self.num_arm_joints:] = self.scaled_actions[:, self.num_base_joints + self.num_arm_joints:] * self.cfg.finger_action_scale
        targets = self.robot_dof_targets + self.dt * self.scaled_actions
        # Optional: lock the abduction joints
        targets[..., self._robot_abduction_dof_idx] = self.robot_abduction_default_pos
        # Optional: use tendon actions to control the finger joints
        # tendon_actions = leap_joints_to_tendon(targets[..., self.num_base_joints + self.num_arm_joints:], self.finger_dof_names_to_id, device=self.device)
        # targets[..., self.num_base_joints + self.num_arm_joints:] = tendon_to_joint_angle_utils(self.robot, tendon_actions)[..., self._robot_finger_dof_idx]
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        edit_door_articulation(self.door)
        self.ref_motion_lib.step()
        self.robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self._robot_dof_idx)
        # joint_pos = self.robot.data.joint_pos.clone()
        # joint_pos[:] = self.ref_motion_lib.get_robot_joint_pos()
        # self.robot.set_joint_position_target(joint_pos)
        # door_pos = self.door.data.joint_pos.clone()
        # door_pos[:] = self.ref_motion_lib.get_door_joint_pos()
        # self.door.write_joint_position_to_sim(door_pos)

    def _get_observations(self) -> dict:
        self._get_intermediate_values()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        base_link_pos = self.robot.data.body_pos_w[:, self._robot_base_link_idx]
        base_link_pos -= self.scene.env_origins.repeat((1, 1
            )).reshape(self.num_envs, 1, 3) 
        
        door_to_base_link_pos = (self.door_link_pos - base_link_pos).reshape(self.num_envs, 1, -1)

        rel_robot_key_body_pos = (self.robot_key_body_pos - base_link_pos).reshape(self.num_envs, 1, -1)

        key_pos_err = self.robot_key_body_pos - (self.ref_robot_key_body_pos).to(self.robot_key_body_pos)
        key_pos_err = key_pos_err.reshape(self.num_envs, 1, -1)

        joint_err = self.joint_pos[:, self._robot_dof_idx] - (self.ref_robot_joint_pos[:, self._robot_dof_idx]).to(self.joint_pos)
        joint_err = joint_err.reshape(self.num_envs, 1, -1)

        door_joint_err = self.door_joint_pos[:, self._door_joint_idx] - (self.ref_door_joint_pos[:, self._door_joint_idx]).to(self.door_joint_pos)

        obs = torch.cat(
            (
                self.joint_pos[:, self._robot_dof_idx].unsqueeze(dim = 1),
                self.joint_vel[:, self._robot_dof_idx].unsqueeze(dim = 1),
                # joint_err,
                key_pos_err,
                door_to_base_link_pos,
                rel_robot_key_body_pos,
                self.door_joint_pos[:, self._door_joint_idx].unsqueeze(dim = 1),
                # self.door_joint_vel[:, self._door_joint_idx].unsqueeze(dim = 1),
                door_joint_err.unsqueeze(dim = 1),
            ),
            dim=-1,
        )
        observations = {"policy": obs.squeeze()}
        return observations

    def _get_intermediate_values(self):
        self.robot_key_body_pos = self.robot.data.body_pos_w[:, self._robot_key_body_idx]\
             - self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)
        self.robot_key_body_quat = self.robot.data.body_quat_w[:, self._robot_key_body_idx]
        self.robot_reset_key_body_pos = self.robot.data.body_pos_w[:, self._robot_reset_key_body_idx]\
             - self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)
        self.robot_base_joint_pos = self.robot.data.joint_pos[:, self._robot_base_dof_idx]
        self.robot_arm_joint_pos = self.robot.data.joint_pos[:, self._robot_arm_dof_idx]
        self.robot_finger_joint_pos = self.robot.data.joint_pos[:, self._robot_finger_dof_idx]
        self.door_joint_pos = self.door.data.joint_pos
        self.robot_base_joint_vel = self.robot.data.joint_vel[:, self._robot_base_dof_idx]
        self.robot_arm_joint_vel = self.robot.data.joint_vel[:, self._robot_arm_dof_idx]
        self.robot_finger_joint_vel = self.robot.data.joint_vel[:, self._robot_finger_dof_idx]
        self.door_link_pos = self.door.data.body_pos_w[:, self._door_body_idx]
        self.door_link_pos -= self.scene.env_origins.repeat((1, 1)).reshape(self.num_envs, 1, 3)

        self.ref_robot_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_key_body_idx]
        self.ref_robot_key_body_quat = self.ref_motion_lib.get_robot_body_quat()[:, self._robot_key_body_idx]
        self.ref_robot_reset_key_body_pos = self.ref_motion_lib.get_robot_body_pos()[:, self._robot_reset_key_body_idx]
        self.ref_robot_joint_pos = self.ref_motion_lib.get_robot_joint_pos()
        self.ref_robot_base_joint_pos = self.ref_robot_joint_pos[:, self._robot_base_dof_idx]
        self.ref_robot_arm_joint_pos = self.ref_robot_joint_pos[:, self._robot_arm_dof_idx]
        self.ref_robot_finger_joint_pos = self.ref_robot_joint_pos[:, self._robot_finger_dof_idx]
        self.ref_door_joint_pos = self.ref_motion_lib.get_door_joint_pos()
        self.ref_robot_base_joint_vel = self.ref_motion_lib.get_robot_joint_vel()[:, self._robot_base_dof_idx]
        self.ref_robot_arm_joint_vel = self.ref_motion_lib.get_robot_joint_vel()[:, self._robot_arm_dof_idx]
        self.ref_robot_finger_joint_vel = self.ref_motion_lib.get_robot_joint_vel()[:, self._robot_finger_dof_idx]
        self.ref_door_link_pos = self.ref_motion_lib.get_door_pos()[:, self._door_body_idx]

    def _get_rewards(self) -> torch.Tensor:
        self._get_intermediate_values()

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

        self.extras["error/key_body_pos_err"] = math.sqrt(key_body_pos_err.mean().item() / len(self.cfg.robot_reset_key_bodies))
        self.extras["error/key_body_quat_err"] = math.sqrt(key_body_quat_err.mean().item() / len(self.cfg.robot_reset_key_bodies))
        self.extras["error/door_err"] = math.sqrt(door_err.mean().item() / len(self.cfg.door_body_names))
        self.extras["error/base_joint_pos_err"] = math.sqrt(base_joint_pos_err.mean().item() / len(self.cfg.base_joints))
        # self.extras["error/root_pos_err"] = math.sqrt(root_pos_err.mean().item())
        # self.extras["error/root_rot_err"] = math.sqrt(root_rot_err.mean().item())
        self.extras["error/arm_joint_pos_err"] = math.sqrt(arm_joint_pos_err.mean().item() / len(self.cfg.arm_joints))
        self.extras["error/finger_joint_pos_err"] = math.sqrt(finger_joint_pos_err.mean().item() / len(self.cfg.finger_joints))
        # self.extras["error/door_pos_err"] = math.sqrt(door_pos_err.max() / len(self.cfg.door_body_names))
        # self.extras["error/base_joint_vel_err"] = base_joint_vel_err.mean()
        # self.extras["error/arm_joint_vel_err"] = arm_joint_vel_err.mean()
        # self.extras["error/finger_joint_vel_err"] = finger_joint_vel_err.mean()

        progress = min(self.step_count / self.reset_progress_total, 1.0)
        alpha = 0.9 - 0.7 * progress  # from 0.9 → 0.2
        probs = torch.tensor(
            [(1 - alpha) * (alpha ** i) for i in range(len(self.ref_motion_lib.key_indices))]
        )
        probs = probs / probs.sum()
        self.extras["reset/prob_get_first_key_frame"] = probs[0]

        return compute_deep_mimic_rewards(
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
            ref_door_body_pos = self.ref_door_link_pos,

            robot_key_body_pos_scale = self.robot_key_body_pos_scale, 
            robot_key_body_quat_scale = self.robot_body_quat_scale,
            door_joint_pos_scale = self.door_joint_pos_scale,
            robot_base_joint_pos_scale = self.robot_base_joint_pos_scale,
            robot_arm_joint_pos_scale = self.robot_arm_joint_pos_scale,
            robot_finger_joint_pos_scale = self.robot_finger_joint_pos_scale,
            robot_base_joint_vel_scale = self.robot_base_joint_vel_scale,
            robot_arm_joint_vel_scale = self.robot_arm_joint_vel_scale,
            robot_finger_joint_vel_scale = self.robot_finger_joint_vel_scale,

            robot_key_body_pos_w = self.robot_key_body_pos_w, 
            robot_key_body_quat_w = self.robot_body_quat_w,
            door_joint_pos_w = self.door_joint_pos_w,
            robot_base_joint_pos_w = self.robot_base_joint_pos_w,
            robot_arm_joint_pos_w = self.robot_arm_joint_pos_w,
            robot_finger_joint_pos_w = self.robot_finger_joint_pos_w,
            robot_base_joint_vel_w = self.robot_base_joint_vel_w,
            robot_arm_joint_vel_w = self.robot_arm_joint_vel_w,
            robot_finger_joint_vel_w = self.robot_finger_joint_vel_w,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_trial_steps - 1
        if not self.early_stopping:
            return False, time_out
        self._get_intermediate_values()
        progress = min(self.step_count / self.reset_progress_total, 1.0)
        # reset_base_pos_delta = self.reset_base_pos_delta_min + (self.reset_base_pos_delta_max - self.reset_base_pos_delta_min) * progress
        # reset_key_body_pos_delta = self.reset_key_body_pos_delta_min + (self.reset_key_body_pos_delta_max - self.reset_key_body_pos_delta_min) * progress
        # reset_key_body_quat_delta = self.reset_key_body_quat_delta_min + (self.reset_key_body_quat_delta_max - self.reset_key_body_quat_delta_min) * progress
        reset_base_pos_delta = self.reset_base_pos_delta_min + (self.reset_base_pos_delta_max - self.reset_base_pos_delta_min) * progress
        reset_key_body_pos_delta = self.reset_key_body_pos_delta_min + (self.reset_key_body_pos_delta_max - self.reset_key_body_pos_delta_min) * progress
        reset_key_body_quat_delta = self.reset_key_body_quat_delta_min + (self.reset_key_body_quat_delta_max - self.reset_key_body_quat_delta_min) * progress
        reset_door_joint_pos_delta = self.reset_door_joint_pos_delta_min + (self.reset_door_joint_pos_delta_max - self.reset_door_joint_pos_delta_min) * progress
        # self.extras["reset/reset_base_pos_delta"] = math.sqrt(reset_base_pos_delta / len(self.cfg.base_joints))
        # self.extras["reset/reset_key_body_pos_delta"] = math.sqrt(reset_key_body_pos_delta / len(self.cfg.robot_reset_key_bodies))
        # self.extras["reset/reset_key_body_quat_delta"] = math.sqrt(reset_key_body_quat_delta / len(self.cfg.robot_reset_key_bodies))
        self.extras["reset/reset_base_pos_delta"] = reset_base_pos_delta
        self.extras["reset/reset_key_body_pos_delta"] = reset_key_body_pos_delta
        self.extras["reset/reset_key_body_quat_delta"] = reset_key_body_quat_delta
        self.extras["reset/reset_door_joint_pos_delta"] = reset_door_joint_pos_delta
        reset_base_pos_delta = reset_base_pos_delta ** 2 * len(self.cfg.base_joints)
        reset_key_body_pos_delta = reset_key_body_pos_delta ** 2 * len(self.cfg.robot_reset_key_bodies)
        reset_key_body_quat_delta = reset_key_body_quat_delta ** 2 * len(self.cfg.robot_reset_key_bodies)
        reset_door_joint_pos_delta = reset_door_joint_pos_delta ** 2 * len(self.cfg.door_joint_names)
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
            (base_joint_pos_err > reset_base_pos_delta) | \
            (key_body_pos_err > reset_key_body_pos_delta) | \
            (key_body_quat_err > reset_key_body_quat_delta) | \
            (door_err > reset_door_joint_pos_delta), \
            time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Optional: change static friction and dynamic friction of the robot
        # if not hasattr(self, "_initialized_materials"):
        #     props = self.robot.root_physx_view.get_material_properties().to(self.device)
        #     print("material properties: ", props)
        #     props[..., 0] = 3.0
        #     props[..., 1] = 2.5
        #     self.robot.root_physx_view.set_material_properties(props.cpu(), torch.arange(self.num_envs, device="cpu"))
        #     self._initialized_materials = True

        if not hasattr(self, "_initialized_materials"):
            props = self.robot.root_physx_view.get_material_properties().to(self.device)
            props[..., 0] = 4.0
            props[..., 1] = 2.5
            self.robot.root_physx_view.set_material_properties(props.cpu(), torch.arange(self.num_envs, device="cpu"))
            self._initialized_materials = True

        reset_frame_idx = self.ref_motion_lib.reset(env_ids, step_count=self.step_count, reset_progress_total=self.reset_progress_total)
        self.max_trial_steps[env_ids] = ((self.ref_motion_lib.num_frames - reset_frame_idx) // self.ref_motion_lib.velocity).long()

        deep_mimic_initial_joint_pos = self.ref_motion_lib.get_robot_joint_pos(env_ids)
        deep_mimic_initial_joint_vel = self.ref_motion_lib.get_robot_joint_vel(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.joint_vel[env_ids] = deep_mimic_initial_joint_vel.to(self.joint_vel)
        # print(deep_mimic_initial_joint_pos.shape)
        self.joint_pos[env_ids] = deep_mimic_initial_joint_pos.to(self.joint_pos)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], None, env_ids)
        self.robot.set_joint_position_target(self.joint_pos[env_ids], env_ids=env_ids)

        door_joint_pos = self.ref_motion_lib.get_door_joint_pos(env_ids).to(self.door.data.joint_pos)

        self.door.write_joint_position_to_sim(door_joint_pos, None, env_ids)

        super()._reset_idx(env_ids)

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

    ref_robot_key_body_pos: torch.Tensor,
    ref_robot_key_body_quat: torch.Tensor,
    ref_door_joint_pos: torch.Tensor,
    ref_robot_base_joint_pos: torch.Tensor,
    ref_robot_arm_joint_pos: torch.Tensor,
    ref_robot_finger_joint_pos: torch.Tensor,
    ref_robot_base_joint_vel: torch.Tensor,
    ref_robot_arm_joint_vel: torch.Tensor,
    ref_robot_finger_joint_vel: torch.Tensor,
    ref_door_body_pos: torch.Tensor,

    robot_key_body_pos_scale: float,
    robot_key_body_quat_scale: float,
    door_joint_pos_scale: float,
    robot_base_joint_pos_scale: float,
    robot_arm_joint_pos_scale: float,
    robot_finger_joint_pos_scale: float,
    robot_base_joint_vel_scale: float,
    robot_arm_joint_vel_scale: float,
    robot_finger_joint_vel_scale: float,

    robot_key_body_pos_w: float,
    robot_key_body_quat_w: float,
    door_joint_pos_w: float,
    robot_base_joint_pos_w: float,
    robot_arm_joint_pos_w: float,
    robot_finger_joint_pos_w: float,
    robot_base_joint_vel_w: float,
    robot_arm_joint_vel_w: float,
    robot_finger_joint_vel_w: float,
) -> torch.Tensor:
    # ----------------------------------
    # Robot body position error
    # ----------------------------------
    # [B, N, 3]
    key_body_pos_diff = ref_robot_key_body_pos - robot_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # [B, N]
    key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)

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
    # base_joint_pos_r = torch.exp(-robot_base_joint_pos_scale * base_joint_pos_err)
    base_joint_pos_r = torch.exp(-robot_base_joint_pos_scale * base_joint_pos_err)
    arm_joint_pos_r = torch.exp(-robot_arm_joint_pos_scale * arm_joint_pos_err)
    finger_joint_pos_r = torch.exp(-robot_finger_joint_pos_scale * finger_joint_pos_err)
    # base_joint_vel_r = torch.exp(-robot_base_joint_vel_scale * base_joint_vel_err)
    base_joint_vel_r = torch.exp(-robot_base_joint_vel_scale * base_joint_vel_err)
    arm_joint_vel_r = torch.exp(-robot_arm_joint_vel_scale * arm_joint_vel_err)
    finger_joint_vel_r = torch.exp(-robot_finger_joint_vel_scale * finger_joint_vel_err)
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
         + robot_finger_joint_vel_w * finger_joint_vel_r

    restricted_reward = (
        robot_key_body_pos_w * key_body_pos_r\
        + door_joint_pos_w * door_r\
    ) * (robot_key_body_pos_w + door_joint_pos_w + robot_base_joint_pos_w + robot_arm_joint_pos_w + robot_finger_joint_pos_w + robot_base_joint_vel_w + robot_arm_joint_vel_w + robot_finger_joint_vel_w) / \
    (robot_key_body_pos_w + door_joint_pos_w)

    # special_env_mask = (ref_door_joint_pos[:, 1] > 0) & (ref_door_joint_pos[:, 0] < 0)
    special_env_mask = torch.linalg.norm(ref_door_body_pos[:, 1] - ref_robot_key_body_pos[:, -1], dim=-1) < 0.15

    reward = torch.where(
        special_env_mask,
        restricted_reward,
        reward
    )
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
    key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)
    # ----------------------------------
    # Robot body orientation error
    # ----------------------------------
    # [B, N]
    key_body_quat_diff = quat_diff_angle(robot_key_body_quat, ref_robot_key_body_quat)
    key_body_quat_err = torch.sum(key_body_quat_diff * key_body_quat_diff, dim=-1)  # [B]
    # ----------------------------------
    # Door joint error
    # ----------------------------------
    door_diff = ref_door_joint_pos - door_joint_pos
    door_err = torch.sum(door_diff * door_diff, dim=-1)  # [B]
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
    # return (key_body_pos_err, key_body_quat_err, door_err, root_pos_err, root_rot_err, arm_joint_pos_err, finger_joint_pos_err, base_joint_vel_err, arm_joint_vel_err, finger_joint_vel_err, door_pos_err)