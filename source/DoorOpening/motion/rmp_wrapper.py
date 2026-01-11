import torch
from isaaclab.scene import InteractiveScene
import numpy as np
import torch
from isaaclab.utils.math import quat_apply

from DoorOpening.utils.point_utils import fit_plane_batch_torch
from DoorOpening.motion.glorbot_controller import GlorbotRMPController as GlorbotController
from DoorOpening.utils.extract_pointcloud_from_articulation import sample_pointcloud
from DoorOpening.utils.point_utils import tensor_to_ply
from isaaclab.utils.math import euler_xyz_from_quat, yaw_quat

import torch
from DoorOpening.utils.pose_utils import  world_to_local

class RMPWrapper:
    """
    Handles perception (finding door normal) and motion generation (base + arm).
    """
    def __init__(self, scene: InteractiveScene, device="cuda", handle_body_name = "link_1"):
        self.scene = scene
        self.device = device
        self.num_envs = scene.num_envs
        self.handle_body_name = handle_body_name
        
        # --- RMP Controller Setup ---
        self.glorbot_controller = GlorbotController(scene["robot"].cfg.spawn.asset_path, self.scene["robot"])
        self.joint_order = self.glorbot_controller.rmpflow.robot_world.joint_names_in_order
        print("joint_order: ", self.joint_order)
        q = self.get_joint_positions()
        # q = self.scene["robot"].data.joint_pos
        self.glorbot_controller.initialize(q=q.squeeze().cpu().numpy())
        torch.set_printoptions(precision=5, sci_mode=False)

    def get_joint_positions(self, q=None):
        if q is None:
            q = self.scene["robot"].data.joint_pos
        else:
            q = q.clone()
        q_out = q.clone()
        for i, name in enumerate(self.joint_order):
            joint_id = self.scene["robot"].find_joints(name)[0]
            q_out[..., i] = q[..., joint_id]
        return q_out

    def get_joint_velocities(self, qd=None):
        if qd is None:
            qd = self.scene["robot"].data.joint_vel
        else:
            qd = qd.clone()
        qd = self.scene["robot"].data.joint_vel
        qd_out = qd.clone()
        for i, name in enumerate(self.joint_order):
            joint_id = self.scene["robot"].find_joints(name)[0]
            qd_out[..., i] = qd[..., joint_id]
        return qd_out

    def map_joint_positions_to_isaaclab_ordering(self, q):
        joint_pos = self.scene["robot"].data.default_joint_pos.clone()
        for i, name in enumerate(self.joint_order):
            joint_id = self.scene["robot"].find_joints(name)[0]
            joint_pos[..., joint_id] = q[..., i].float()
        return joint_pos

    def map_joint_velocities_to_isaaclab_ordering(self, qd):
        joint_vel = self.scene["robot"].data.default_joint_vel.clone()
        for i, name in enumerate(self.joint_order):
            joint_id = self.scene["robot"].find_joints(name)[0]
            joint_vel[..., joint_id] = qd[..., i].float()
        return joint_vel

    def get_base_pos_and_quat(self, articulation, base_name="base_link"):
        base_idx, _ = articulation.find_bodies(base_name)
        base_id = base_idx[0]
        base_pos = articulation.data.body_pos_w[:, base_id]
        base_quat = articulation.data.body_quat_w[:, base_id]
        return base_pos, base_quat
    
    def reach_pose(self, ee_pos, ee_quat, num_steps=20):
        q = self.get_joint_positions()
        qd = self.get_joint_velocities()
        for _ in range(num_steps):
            q, qd = self.reach_pose_one_step(ee_pos, ee_quat, q, qd)
        q = self.map_joint_positions_to_isaaclab_ordering(q)
        qd = self.map_joint_velocities_to_isaaclab_ordering(qd)
        return q, qd

    def reach_pose_one_step(self, ee_pos, ee_quat, q, qd):
        robot_pos, robot_quat = self.get_base_pos_and_quat(self.scene["robot"])

        base_pose = q[...,0:3]
        base_velocity = qd[...,0:3]

        if isinstance(q, torch.Tensor):
            q = q.squeeze()
            q = q.cpu().numpy()
        
        if isinstance(qd, torch.Tensor):
            qd = qd.squeeze()
            qd = qd.cpu().numpy()

        if isinstance(base_pose, torch.Tensor):
            base_pose = base_pose.squeeze()
            base_pose = base_pose.cpu().numpy()

        if isinstance(base_velocity, torch.Tensor):
            base_velocity = base_velocity.squeeze()
            base_velocity = base_velocity.cpu().numpy()

        joint_angles = self.scene["door"].data.joint_pos
        door_pointcloud = sample_pointcloud(self.scene["door"].cfg.spawn.asset_path, joint_angles, device=self.device)
        door_pointcloud = quat_apply(self.scene["door"].data.body_quat_w[:, 0], door_pointcloud) + self.scene["door"].data.body_pos_w[:, 0]
        door_pointcloud = door_pointcloud.squeeze()

        door_pointcloud = world_to_local(door_pointcloud, robot_pos, robot_quat)

        if isinstance(door_pointcloud, torch.Tensor):
            door_pointcloud = door_pointcloud.cpu().numpy()

        ee_pos = world_to_local(ee_pos, robot_pos, robot_quat)

        ee_target_pose = np.concatenate((ee_pos.squeeze().detach().cpu().numpy(), ee_quat.squeeze().detach().cpu().numpy()))
        
        arm_pos_target, arm_vel_target, base_pos_target_world_frame, base_vel_target_world_frame, base_se2_plan, _ = self.glorbot_controller.get_action(
            base_pose=base_pose, base_velocity=base_velocity, 
            franka_joint_positions=q[3:3+7], franka_joint_velocities=qd[3:3+7],
            leap_joint_positions=q[10:10+16], leap_joint_velocities=qd[10:10+16],
            arx_joint_positions=q[26:26+6], arx_joint_velocities=qd[26:26+6],
            ee_target_pose=ee_target_pose, 
            global_point_cloud=door_pointcloud, 
            robot_point_cloud_world_frame=door_pointcloud,
        )
        joint_pos_target = np.concatenate((base_pos_target_world_frame, arm_pos_target))
        joint_vel_target = np.concatenate((base_vel_target_world_frame, arm_vel_target))
        joint_pos = self.scene["robot"].data.joint_pos
        joint_pos_target, joint_vel_target = torch.from_numpy(joint_pos_target).to(self.device), torch.from_numpy(joint_vel_target).to(self.device)

        # print("base_pos: ", base_pos_target_world_frame, robot_pos)
        # joint_pos_target[..., :3] = unbase_goal(joint_pos_target[..., :3], robot_pos, robot_quat, velocity = False)
        # joint_pos_target[..., :3] = unbase_goal(joint_pos_target[..., :3], torch.zeros_like(robot_pos), robot_quat, velocity = True)
        # print("base target: ", joint_pos_target[:3])

        # Normalize the angle to [-pi, pi]
        joint_pos_target[..., 2] = (joint_pos_target[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi
        joint_vel_target[..., 2] = (joint_vel_target[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi

        return joint_pos_target, joint_vel_target
        