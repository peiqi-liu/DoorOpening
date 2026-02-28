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
from DoorOpening.assets.door.door_cfg import motion_traj_paths, handle_offsets, board_offsets
from isaaclab.sensors import ContactSensor
from DoorOpening.constants.robot_constants import FULL_JOINT_NAMES, ROBOT_KEY_BODY_NAMES
from DoorOpening.utils.pose_utils import normalize_to_center_frame, world_to_local
from isaaclab.utils.math import quat_apply
from DoorOpening.utils.quat_utils import quat_to_euler

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

        self.ref_base_joint_idx = [FULL_JOINT_NAMES.index(name) for name in base_joint_names]
        self.ref_arm_joint_idx = [FULL_JOINT_NAMES.index(name) for name in arm_joint_names]
        self.ref_finger_joint_idx = [FULL_JOINT_NAMES.index(name) for name in finger_joint_names]

        robot_abduction_dof_idx, abduction_joint_names = self.robot.find_joints(self.cfg.abduction_joints)
        self.robot_abduction_default_pos = self.robot.data.default_joint_pos[..., robot_abduction_dof_idx]
        self.finger_dof_names_to_id = {name: idx for idx, name in enumerate(finger_joint_names)}
        self.robot_abduction_dof_idx_in_targets = [self.finger_dof_names_to_id[name] + self.num_base_joints + self.num_arm_joints for name in self.cfg.abduction_joints]
        self.close_finger_joints = torch.tensor([self.cfg.close_finger_joints[name] for name in finger_joint_names], device=self.device)
        self.open_finger_joints = torch.tensor([self.cfg.open_finger_joints[name] for name in finger_joint_names], device=self.device)

        self._robot_base_link_idx, self.robot_base_link_name = self.robot.find_bodies(self.cfg.base_link_name)
        self._door_body_idx, self.door_body_names = self.door.find_bodies(self.cfg.door_body_names)
        self._door_joint_idx, self.door_joint_names = self.door.find_joints(self.cfg.door_joint_names)

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

        # self.reset_base_pos_delta = (self.cfg.reset_base_pos_delta ** 2) * len(self.cfg.base_joints)
        # self.reset_key_body_pos_delta = (self.cfg.reset_key_body_pos_delta ** 2) * len(self.cfg.robot_reset_key_bodies)
        # self.reset_key_body_quat_delta = (self.cfg.reset_key_body_quat_delta ** 2) * len(self.cfg.robot_reset_key_bodies)
        self.reset_base_pos_delta_min = self.cfg.reset_base_pos_delta_min
        self.reset_key_body_pos_delta_min = self.cfg.reset_key_body_pos_delta_min
        self.reset_key_body_quat_delta_min = self.cfg.reset_key_body_quat_delta_min
        self.reset_base_pos_delta_max = self.cfg.reset_base_pos_delta_max
        self.reset_key_body_pos_delta_max = self.cfg.reset_key_body_pos_delta_max
        self.reset_key_body_quat_delta_max = self.cfg.reset_key_body_quat_delta_max
        self.reset_door_joint_pos_delta_min = self.cfg.reset_door_joint_pos_delta_min
        self.reset_door_joint_pos_delta_max = self.cfg.reset_door_joint_pos_delta_max

        self.last_actions = torch.zeros(
            (self.num_envs, len(self._robot_dof_idx)),
            device=self.device
        )

        self.twist_indices = self.cfg.twist_indices

        # self.ref_motion_lib = ReferenceMotionManager(self.cfg.motion_file, self.num_envs, self.device, velocity=self.cfg.velocity, reset_from_start = True)
        self.handle_offsets = [handle_offsets[i % len(handle_offsets)] for i in range(self.num_envs)]
        self.board_offsets = [board_offsets[i % len(board_offsets)] for i in range(self.num_envs)]
        self.handle_offsets = torch.stack(self.handle_offsets).to(self.device)
        self.board_offsets = torch.stack(self.board_offsets).to(self.device)
        env_to_file_map = [i % len(motion_traj_paths) for i in range(self.num_envs)]
        self.ref_motion_lib = ReferenceMotionManager(num_envs=self.num_envs, device=self.device, velocity=self.cfg.velocity, reset_from_start = False, env_to_file_map=env_to_file_map, twist_indices=self.twist_indices)
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
        # self.scene.sensors["contact_forces_door1"] = ContactSensor(self.cfg.contact_forces_door1)
        self.scene.sensors["contact_forces_door2"] = ContactSensor(self.cfg.contact_forces_door2)
        # self.scene.sensors["contact_forces_robot_palm_center"] = ContactSensor(self.cfg.contact_forces_robot_palm_center)
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
        self.last_actions[:] = self.scaled_actions

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
        
        # door_to_base_link_pos = (self.door_link_pos - base_link_pos).reshape(self.num_envs, 1, -1)
        # door_to_base_link_pos = (self.door_keypoints - base_link_pos).reshape(self.num_envs, 1, -1)
        # door_twist_base_link_pos = (self.ref_door_body_pos_twist - base_link_pos).reshape(self.num_envs, 1, -1)
        # door_to_base_link_pos = world_to_local(self.door_link_pos, self.robot_base_body_pos, self.robot_base_body_quat).reshape(self.num_envs, 1, -1)
        # door_twist_palm_link_pos = world_to_local(self.ref_door_body_pos_twist, self.robot_palm_body_pos, self.robot_palm_body_quat).reshape(self.num_envs, 1, -1)
        # door_to_palm_link_pos = world_to_local(self.door_link_pos, self.robot_palm_body_pos, self.robot_palm_body_quat).reshape(self.num_envs, 1, -1)
        door_to_base_link_pos = (self.door_link_pos - self.robot_base_body_pos.unsqueeze(1))
        door_to_base_link_pos = world_to_local(door_to_base_link_pos, self.robot_base_body_pos, self.robot_base_body_quat)
        door_to_base_link_pos = door_to_base_link_pos.reshape(self.num_envs, 1, -1)
        door_twist_palm_link_pos = (self.ref_door_body_pos_twist - self.robot_palm_body_pos.unsqueeze(1))
        door_twist_palm_link_pos = world_to_local(door_twist_palm_link_pos, self.robot_base_body_pos, self.robot_base_body_quat)
        door_twist_palm_link_pos = door_twist_palm_link_pos.reshape(self.num_envs, 1, -1)
        door_to_palm_link_pos = (self.door_link_pos - self.robot_base_body_pos.unsqueeze(1))
        door_to_palm_link_pos = world_to_local(door_to_palm_link_pos, self.robot_base_body_pos, self.robot_base_body_quat)
        door_to_palm_link_pos = door_to_palm_link_pos.reshape(self.num_envs, 1, -1)

        # rel_robot_key_body_pos = (self.robot_key_body_pos - base_link_pos).reshape(self.num_envs, 1, -1)
        key_pos_err = self.robot_key_body_pos - (self.ref_robot_key_body_pos).to(self.robot_key_body_pos)
        key_pos_err = key_pos_err.reshape(self.num_envs, 1, -1)

        # door_joint_err = self.door_joint_pos[:, self._door_joint_idx] - (self.ref_door_joint_pos[:, self._door_joint_idx]).to(self.door_joint_pos)

        # contact_forces_door1 = self.scene.sensors["contact_forces_door1"].data.net_forces_w
        # contact_forces_door2 = self.scene.sensors["contact_forces_door2"].data.net_forces_w
        # contact_forces_robot_palm_center = self.scene.sensors["contact_forces_robot_palm_center"].data.net_forces_w
        # contact_forces_door1 = contact_forces_door1.reshape(self.num_envs, 1, -1)
        # contact_forces_door2 = contact_forces_door2.reshape(self.num_envs, 1, -1)
        # contact_forces_robot_palm_center = contact_forces_robot_palm_center.reshape(self.num_envs, 1, -1)

        # frame_idx = torch.ceil(self.ref_motion_lib.frame_idx).unsqueeze(dim = -1).to(self.device) // (self.ref_motion_lib.num_frames // 10)
        obs = torch.cat(
            (
                self.joint_pos[:, self._robot_dof_idx].unsqueeze(dim = 1),
                self.joint_vel[:, self._robot_dof_idx].unsqueeze(dim = 1),
                self.last_actions.unsqueeze(dim = 1),

                key_pos_err,
                # rel_robot_key_body_pos,
                self.robot_key_body_pos.reshape(self.num_envs, 1, -1),
                self.robot_key_body_euler.reshape(self.num_envs, 1, -1),
                self.robot_body_lin_vel.reshape(self.num_envs, 1, -1),
                self.robot_body_ang_vel.reshape(self.num_envs, 1, -1),

                door_to_base_link_pos,
                door_to_palm_link_pos,
                # self.door_link_pos.reshape(self.num_envs, 1, -1),
                self.door_joint_pos[:, self._door_joint_idx].unsqueeze(dim = 1),

                # door_joint_err.unsqueeze(dim = 1),
                self.ref_door_joint_pos[:, self._door_joint_idx].to(self.door_joint_pos).unsqueeze(dim = 1),
                self.ref_robot_key_body_pos_twist.reshape(self.num_envs, 1, -1),
                self.ref_robot_key_body_quat_twist.reshape(self.num_envs, 1, -1),
                self.ref_door_joint_pos_twist.reshape(self.num_envs, 1, -1),
                door_twist_palm_link_pos,
                # self.ref_door_body_pos_twist.reshape(self.num_envs, 1, -1),
                # frame_idx.unsqueeze(dim = -1),
                # contact_forces_door1,
                # contact_forces_door2,
                # contact_forces_robot_palm_center,
            ),
            dim=-1,
        )
        observations = {"policy": obs.squeeze()}
        return observations

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

        self.robot_key_body_euler = quat_to_euler(self.robot_key_body_quat)
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
        self.door_link_quat = self.door.data.body_quat_w[:, self._door_body_idx]
        # self.door_keypoints = self.compute_door_keypoints()
        self.robot_body_lin_vel = self.robot.data.body_link_lin_vel_w[:, self._robot_key_body_idx]
        self.robot_body_ang_vel = self.robot.data.body_link_ang_vel_w[:, self._robot_key_body_idx]
        # print("door keypoints: ", self.compute_door_keypoints())
        # print("door link pos: ", self.door_link_pos)

        self.ref_robot_key_body_pos_twist = self.ref_motion_lib.get_robot_body_pos_twist()[:, :, self.ref_key_body_idx]
        # It is a minomer, we are actually sending euler angles as it might be more friendly to MLP
        self.ref_robot_key_body_quat_twist = self.ref_motion_lib.get_robot_body_quat_twist()[:, :, self.ref_key_body_idx]
        self.ref_robot_joint_pos_twist = self.ref_motion_lib.get_robot_joint_pos_twist()
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

        # progress = min(self.step_count / self.reset_progress_total, 1.0)
        # alpha = 1.2 - 0.8 * progress  # from 1.2 → 0.4
        # probs = torch.tensor(
        #     [(1 - alpha) * (alpha ** i) for i in range(len(self.ref_motion_lib.key_indices))]
        # )
        # probs = probs / probs.sum()
        # self.extras["reset/prob_get_first_key_frame"] = probs[0]

        # contact_forces_robot_palm_center = self.scene.sensors["contact_forces_robot_palm_center"].data.net_forces_w
        contact_forces_door2 = self.scene.sensors["contact_forces_door2"].data.net_forces_w

        return compute_deep_mimic_rewards(
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

        self.joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.default_joint_vel[env_ids]
        self.joint_vel[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_vel.to(self.joint_vel)[..., self.ref_robot_dof_idx]
        self.joint_pos[env_ids[:, None], self._robot_dof_idx[None, :]] = deep_mimic_initial_joint_pos.to(self.joint_pos)[..., self.ref_robot_dof_idx]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], None, env_ids,)
        self.robot.set_joint_position_target(self.joint_pos[env_ids], env_ids=env_ids)

        door_joint_pos = self.ref_motion_lib.get_door_joint_pos(env_ids).to(self.door.data.joint_pos)

        self.door.write_joint_position_to_sim(door_joint_pos, None, env_ids)
        self.door.set_joint_position_target(torch.zeros_like(door_joint_pos), None, env_ids)

        self.last_actions[env_ids] = 0.0

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
    # print("contact_forces: ", contact_forces)
    # print("contact_reward: ", contact_reward)
    # print("contact_force_w: ", contact_force_w)
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

    # restricted_reward = (
    #     robot_key_body_pos_w * key_body_pos_r\
    #     + door_joint_pos_w * door_r\
    # ) * (robot_key_body_pos_w + door_joint_pos_w + robot_base_joint_pos_w + robot_arm_joint_pos_w + robot_finger_joint_pos_w + robot_base_joint_vel_w + robot_arm_joint_vel_w + robot_finger_joint_vel_w) / \
    # (robot_key_body_pos_w + door_joint_pos_w)

    # # special_env_mask = (ref_door_joint_pos[:, 1] > 0) & (ref_door_joint_pos[:, 0] < 0)
    # special_env_mask = torch.linalg.norm(door_body_pos[:, 1] - robot_key_body_pos[:, -1], dim=-1) < 0.2

    # reward = torch.where(
    #     special_env_mask,
    #     restricted_reward,
    #     reward
    # )
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