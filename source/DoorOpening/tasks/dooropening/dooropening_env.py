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
from isaaclab.utils.math import sample_uniform

from .dooropening_env_cfg import DooropeningEnvCfg


class DooropeningEnv(DirectRLEnv):
    cfg: DooropeningEnvCfg

    def __init__(self, cfg: DooropeningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        actuated_joints = self.cfg.base_joints + self.cfg.arm_joints

        self._robot_dof_idx, _ = self.robot.find_joints(actuated_joints)
        self._robot_base_idx, _ = self.robot.find_joints(self.cfg.base_joints)
        self._hand_body_idx, self.body_names = self.robot.find_bodies(self.cfg.hand_body_name)
        self._handle_body_idx, _ = self.door.find_bodies(self.cfg.door_handle_body_name)

        self._base_link_idx, _ = self.robot.find_bodies(self.cfg.base_link_names)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self._robot_dof_idx, 1].to(device=self.device)

        # self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # self.robot_dof_speed_scales[self.robot.find_joints("base_rotation_joint")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, len(self._robot_dof_idx)), device=self.device)

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

    # def _pre_physics_step(self, actions: torch.Tensor) -> None:
    #     self.actions = actions.clone()

    # def _apply_action(self) -> None:
    #     self.robot.set_joint_position_target(self.actions * self.cfg.action_scale, joint_ids=self._robot_dof_idx)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        targets = self.robot_dof_targets + self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self._robot_dof_idx)

    def _get_observations(self) -> dict:
        self.compute_intermediate_reward_values()
        obs = torch.cat(
            (
                self.joint_pos[:, self._robot_dof_idx].unsqueeze(dim = 1),
                self.joint_vel[:, self._robot_dof_idx].unsqueeze(dim = 1),
                self.handle_pos - self.hand_pos
            ),
            dim=-1,
        )
        observations = {"policy": obs.squeeze()}
        return observations

    def compute_intermediate_reward_values(self):
        self.hand_pos = self.robot.data.body_pos_w[:, self._hand_body_idx]
        self.hand_pos -= self.scene.env_origins.repeat((1, 1
            )).reshape(self.num_envs, 1, 3)
        self.handle_pos = self.door.data.body_pos_w[:, self._handle_body_idx]
        self.handle_pos -= self.scene.env_origins.repeat((1, 1
            )).reshape(self.num_envs, 1, 3)
        
        self.handle_pos_error = torch.norm(self.hand_pos - self.handle_pos, dim=-1, p=1).squeeze()

        self.base_link_pos = self.robot.data.body_pos_w[:, self._base_link_idx]
        self.base_link_pos -= self.scene.env_origins.repeat((1, 1
            )).reshape(self.num_envs, 1, 3)
        self.base_link_pos = torch.mean(self.base_link_pos, dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate_reward_values()
        # self.rotation_action = self.actions[:, self.robot.find_joints("base_rotation_joint")[0]].reshape(self.num_envs)
        self.base_rotation = self.actions[:, self._robot_base_idx[0]].reshape(self.num_envs)
        return compute_rewards(
            self.cfg.handle_pos_error_scale, 
            self.handle_pos_error, 
            self.cfg.base_link_pos_error_scale, 
            self.base_link_pos - self.handle_pos.squeeze(), 
            # self.cfg.action_penalty_scale,
            # self.rotation_action
            self.cfg.base_rotation_error_scale,
            self.base_rotation
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.compute_intermediate_reward_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.handle_pos_error < 0.01
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    handle_pos_error_scale: float,
    handle_pos_error: torch.Tensor,
    base_link_pos_error_scale: float,
    base_link_pos_error: torch.Tensor,
    base_rotation_error_scale: float,
    base_rotation: torch.Tensor,
    # action_penalty_scale: float,
    # rotation_actions: torch.Tensor,
):
    # print(base_link_pos_error.shape)
    # print(rotation_actions.shape)
    base_penalty = - torch.clamp(torch.norm(base_link_pos_error[:, :2], dim=-1, p=1), 0.0, 1.5)
    # action_penalty = - rotation_actions**2
    base_rotation_penalty = - torch.abs(base_rotation)
    return \
        handle_pos_error_scale / (handle_pos_error + 0.25) \
        + base_link_pos_error_scale * base_penalty \
        + base_rotation_error_scale * base_rotation_penalty \
        # + action_penalty_scale * action_penalty