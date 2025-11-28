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

def rebase_goal(rel_pos, orig_pos, orig_quat, velocity = False):
    """
    Transforms a relative pose (x, y, theta) to an absolute pose (x, y, theta) based on an
    original position (x, y, z) and orientation quaternion (w, x, y, z).

    Args:
        rel_pos (tensor-like): The relative pose as (x, y, theta).
        orig_pos (tensor-like): The original absolute position as (x, y, z).
        orig_quat (tensor-like): The original absolute orientation as (w, x, y, z) quaternion.
        velocity (bool): Whether to this function is used to rebase vel.

    Returns:
        torch.Tensor: The absolute pose as (x, y, theta).
    """

    rel_x, rel_y, rel_theta = rel_pos.unbind(dim = -1)
    orig_x, orig_y, orig_z = orig_pos.unbind(dim = -1)

    # --- Calculate absolute position (x, y) ---
    # Calculate the yaw angle from the original quaternion
    _, _, orig_yaw = euler_xyz_from_quat(orig_quat)

    cos_yaw_val = torch.cos(orig_yaw)
    sin_yaw_val = torch.sin(orig_yaw)

    # Rotate the relative x, y components by the original yaw
    rotated_rel_x = rel_x * cos_yaw_val - rel_y * sin_yaw_val
    rotated_rel_y = rel_x * sin_yaw_val + rel_y * cos_yaw_val

    # Calculate absolute x, y
    if not velocity:
        abs_x = orig_x + rotated_rel_x
        abs_y = orig_y + rotated_rel_y
    else:
        abs_x = rotated_rel_x
        abs_y = rotated_rel_y

    # --- Calculate absolute orientation theta ---
    # The absolute theta is the sum of the original theta and the relative theta
    abs_theta = orig_yaw + rel_theta

    # Optional: Normalize the angle to [-pi, pi]
    # abs_theta = torch.atan2(torch.sin(abs_theta), torch.cos(abs_theta))

    return torch.hstack([abs_x, abs_y, abs_theta]).to(rel_pos.device)

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
        self.glorbot_controller = GlorbotController()
        self.glorbot_controller.initialize()
        torch.set_printoptions(precision=3, sci_mode=False)

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
        robot_pos, robot_quat = self.get_base_pos_and_quat(self.scene["robot"])
        base_pose = q[...,0:3]
        base_velocity = qd[...,0:3]
        base_pose = torch.Tensor(list(rebase_goal(base_pose, robot_pos, robot_quat, velocity = False)))
        base_velocity = torch.Tensor(list(rebase_goal(base_velocity, torch.zeros(3), robot_quat, velocity = True)))
        # base_pose = quat_apply(robot_quat, base_pose) + robot_pos
        # base_velocity = quat_apply(robot_quat, base_velocity)

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

        if isinstance(ee_target_position, torch.Tensor):
            ee_target_position = ee_target_position.cpu().numpy()

        ee_target_orientation = robot_quat.squeeze().cpu().numpy()
        ee_target_pose = np.concatenate((ee_target_position, ee_target_orientation))

        joint_angles = self.scene["door"].data.joint_pos
        door_pointcloud = sample_pointcloud(self.scene["door"].cfg.spawn.asset_path, joint_angles, device=self.device)
        door_pointcloud = quat_apply(self.scene["door"].data.body_quat_w[:, 0], door_pointcloud) + self.scene["door"].data.body_pos_w[:, 0]
        door_pointcloud = door_pointcloud.squeeze()

        # print("ee_target_position: ", ee_target_position)

        # tensor_to_ply(door_pointcloud, "pointcloud.ply")

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
        # Normalize the angle to [-pi, pi]
        joint_pos_target[..., 2] = (joint_pos_target[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi
        joint_vel_target[..., 2] = (joint_vel_target[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi
        # pos, quat = self.get_base_pos_and_quat(self.scene["robot"], base_name="palm_lower")
        # print("pos, target", pos, ee_target_position)
        # print("quat, target", quat, ee_target_orientation)
        joint_pos = self.scene["robot"].data.joint_pos
        # print("joint_pos: ", joint_pos[..., :3 + 7])
        # print("joint_pos_target: ", joint_pos_target[..., :3 + 7])
        # print("joint_vel_target", joint_vel_target)
        return torch.from_numpy(joint_pos_target).to(self.device), torch.from_numpy(joint_vel_target).to(self.device)
        