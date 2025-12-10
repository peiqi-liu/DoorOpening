import torch
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_rotate

from isaaclab.utils import convert_dict_to_backend

import omni
from pxr import Usd, UsdGeom
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

class MotionGenerator:
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
        self.joint_order = self.glorbot_controller.rmp_configs.tidybot2_franka_joint_names.tolist()
        q = self.get_joint_positions()
        # q = self.scene["robot"].data.joint_pos
        self.glorbot_controller.initialize(q=q.squeeze().cpu().numpy())
        torch.set_printoptions(precision=5, sci_mode=False)

    def get_joint_positions(self):
        joint_ids, _ = self.scene["robot"].find_joints(self.joint_order)
        # print("joint_ids: ", joint_ids, len(joint_ids))
        q = self.scene["robot"].data.joint_pos
        q = q[..., joint_ids].float()
        # q = q.cpu().numpy()
        return q

    def get_joint_velocities(self):
        joint_ids, _ = self.scene["robot"].find_joints(self.joint_order)
        qd = self.scene["robot"].data.joint_vel
        qd = qd[..., joint_ids].float()
        # qd = qd.cpu().numpy()
        return qd

    def map_joint_positions_to_isaaclab_ordering(self, q):
        joint_ids, _ = self.scene["robot"].find_joints(self.joint_order)
        joint_pos = self.scene["robot"].data.default_joint_pos
        joint_pos[..., joint_ids] = q.float()
        return joint_pos

    def map_joint_velocities_to_isaaclab_ordering(self, qd):
        joint_ids, _ = self.scene["robot"].find_joints(self.joint_order)
        joint_vel = self.scene["robot"].data.default_joint_vel
        joint_vel[..., joint_ids] = qd.float()
        return joint_vel

    def get_door_knob_pos(self):
        door = self.scene["door"]
        # handle_body_name = self.scene.cfg.door.handle_body_name
        handle_body_name = self.handle_body_name
        handle_body_idx, _ = door.find_bodies(handle_body_name)
        handle_body_id = handle_body_idx[0]
        handle_pos = door.data.body_pos_w[:, handle_body_id]
        # door_pos, door_quat = self.get_base_pos_and_quat(door, base_name="base")
        # handle_pos = quat_apply(door_quat, handle_pos) + door_pos
        return handle_pos

    def get_base_pos_and_quat(self, articulation, base_name="base_link"):
        base_idx, _ = articulation.find_bodies(base_name)
        base_id = base_idx[0]
        base_pos = articulation.data.body_pos_w[:, base_id]
        base_quat = articulation.data.body_quat_w[:, base_id]
        return base_pos, base_quat

    def reach_door_knob(self, q, qd):
        q = self.get_joint_positions()
        qd = self.get_joint_velocities()
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

        ee_target_position = self.get_door_knob_pos().squeeze()
        ee_target_position = world_to_local(ee_target_position, robot_pos, robot_quat)

        if isinstance(ee_target_position, torch.Tensor):
            ee_target_position = ee_target_position.cpu().numpy()

        ee_target_orientation = robot_quat.squeeze().cpu().numpy()
        ee_target_pose = np.concatenate((ee_target_position, ee_target_orientation))

        # print("ee_target_pose: ", ee_target_pose)

        joint_angles = self.scene["door"].data.joint_pos
        door_pointcloud = sample_pointcloud(self.scene["door"].cfg.spawn.asset_path, joint_angles, device=self.device)
        door_pointcloud = quat_apply(self.scene["door"].data.body_quat_w[:, 0], door_pointcloud) + self.scene["door"].data.body_pos_w[:, 0]
        door_pointcloud = door_pointcloud.squeeze()

        # body_idx, _ = self.scene["robot"].find_bodies(["mcp_joint1"])
        # body_id = body_idx[0]
        # print(world_to_local(self.scene["robot"].data.body_pos_w[:, body_id], robot_pos, robot_quat))

        door_pointcloud = world_to_local(door_pointcloud, robot_pos, robot_quat)

        # tensor_to_ply(door_pointcloud, "door_pointcloud.ply")

        if isinstance(door_pointcloud, torch.Tensor):
            door_pointcloud = door_pointcloud.cpu().numpy()
        
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

        print(joint_pos_target[..., :10])
        print(self.scene["robot"].data.joint_pos[..., :10])

        # joint_pos_target = self.map_joint_positions_to_isaaclab_ordering(joint_pos_target)
        # joint_vel_target = self.map_joint_velocities_to_isaaclab_ordering(joint_vel_target)
        return joint_pos_target, joint_vel_target
        