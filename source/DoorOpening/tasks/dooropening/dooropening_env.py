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

        self._robot_dof_idx, _ = self.robot.find_joints(self.cfg.actuated_joints)
        self._hand_body_idx = self.robot.find_bodies(self.cfg.hand_body_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._robot_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._robot_dof_idx].unsqueeze(dim=1),
                self.joint_vel[:, self._robot_dof_idx].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        print("joint_pos shape: ", self.joint_pos.shape)
        print("joint_vel shape: ", self.joint_vel.shape)
        print("robot_dof_idx shape: ", self._robot_dof_idx)
        observations = {"policy": obs}
        print("obs shape: ", obs.shape)
        return observations

    def compute_intermediate_reward_values(self):
        self.hand_pos = self.robot.data.body_pos_w[:, self._hand_body_idx]
        self.hand_pos -= self.scene.env_origins.repeat((1, 1
            )).reshape(self.num_envs, 1, 3)
        self.handle_pos = self.door.data.body_pos_w[:, self.door.data.body_names.index(self.cfg.door_handle_body_name)].reshape(self.num_envs, 1, 3)
        
        self.handle_pos_error = torch.norm(self.hand_pos - self.handle_pos, dim=-1, p=2)

    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate_reward_values()
        return compute_rewards(self.cfg.handle_pos_error_scale, self.handle_pos_error)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.compute_intermediate_reward_values()
        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel
        print("episode_length_buf: ", self.episode_length_buf)
        print("max_episode_length: ", self.max_episode_length)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        # return out_of_bounds, time_out
        return False, time_out

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
):
    return - handle_pos_error_scale * handle_pos_error