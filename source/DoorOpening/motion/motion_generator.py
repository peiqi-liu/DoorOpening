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

from DoorOpening.utils.extract_pointcloud_from_articulation import sample_pointcloud


FRANKA_JOINT_NAMES = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
    ]

BASE_JOINT_NAMES = [
        'base_rotation_joint',
        'base_x_joint',
        'base_y_joint',
    ]

from isaaclab.utils.math import euler_xyz_from_quat, yaw_quat

def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    # keep angle in (-pi, pi]
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

def rebase_goal(
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_theta: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
):
    base_pos = base_pos.squeeze()[:2]
    _, _, yaw = euler_xyz_from_quat(base_quat)
    base_theta = yaw

    dx = target_x - base_pos[0]
    dy = target_y - base_pos[1]

    cos_t = torch.cos(base_theta)
    sin_t = torch.sin(base_theta)

    # rotate by -base_theta (frame change)
    local_x =  dx * cos_t + dy * sin_t
    local_y = -dx * sin_t + dy * cos_t

    local_theta = normalize_angle(target_theta - base_theta)
    return local_x, local_y, local_theta

class MotionGenerator:
    """
    Handles perception (finding door normal) and motion generation (base + arm).
    """
    def __init__(self, scene: InteractiveScene, device="cuda", handle_body_name = "link_1"):
        self.scene = scene
        self.device = device
        self.num_envs = scene.num_envs
        self.handle_body_name = handle_body_name
        # --- IK Controller Setup ---
        # Adjust 'command_type' based on your robot (position or velocity control)
        # Adjust 'target_link' to be your end-effector name
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position", 
            use_relative_mode=True,
            ik_method="dls",
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.device
        )
        self.ik_controller.reset()
        # hand_idx = self.scene["robot"].find_bodies("palm_lower")[0][0]
        # initial_ee_quat = self.scene["robot"].data.body_quat_w[:, hand_idx]
        # self.ik_controller.set_command(command=torch.zeros(self.num_envs, 3, device=self.device), ee_quat=initial_ee_quat)
        
        self.reset()

    def reset(self):
        self.base_pose = None
        self.prev_angles = self.scene["robot"].data.joint_pos

    def get_door_approach_pose(self, door_normal, door_handle_pos, robot_pos, offset=0.3):
        """
        Calculate target (x, y, theta) for robot to approach door from correct side
        
        Args:
            door_normal: [x, y, z] - Normal vector orthogonal to door plane
            door_handle_pos: [x, y, z] - Position of door handle
            robot_pos: [x, y, z] - Current robot position
            offset: Distance to stand from door handle (default 0.5m)
        
        Returns:
            x, y, theta: Target base position and orientation in radians
        """
        # Work in 2D (XY plane) for base navigation
        n = door_normal[..., :2]
        handle = door_handle_pos[..., :2]
        robot = robot_pos[..., :2]
        
        # Normalize door normal in XY plane
        n_norm = torch.norm(n)
        if n_norm < 1e-6:
            n = torch.tensor([0.0, 1.0], device=door_normal.device)  # Default normal if invalid
        else:
            n = n / n_norm
        
        # Determine which side of the door the robot is on
        vec_handle_to_robot = robot - handle
        side = (vec_handle_to_robot * n).sum(dim=-1)
        
        # Target position: offset from handle on the correct side
        approach_side = torch.sign(side)
        # print("n, handle, approach_side: ", n, handle, approach_side)
        target_pos = handle + approach_side * n * offset
        
        # TODO: Remove this once we can parellelize the computation
        target_pos = target_pos.squeeze()
        handle = handle.squeeze()
        
        # Target orientation: face along the door plane toward the handle
        # Door plane direction is perpendicular to normal
        door_dir = handle - target_pos   # Perpendicular vector
        
        # Calculate theta from direction vector
        theta = torch.atan2(door_dir[1], door_dir[0])

        return target_pos[0], target_pos[1], theta
    
    
    def get_door_knob_pos(self, use_handle_body_name = True):
        door = self.scene["door"]
        if not use_handle_body_name:
            return door.data.body_pos_w[:, 0]
        # handle_body_name = self.scene.cfg.door.handle_body_name
        handle_body_name = self.handle_body_name
        handle_body_idx, _ = door.find_bodies(handle_body_name)
        handle_body_id = handle_body_idx[0]
        handle_pos = door.data.body_pos_w[:, handle_body_id]
        return handle_pos

    def get_robot_base_pos(self):
        robot = self.scene["robot"]
        robot_base_name = "base_x_link"
        robot_base_idx, _ = robot.find_bodies(robot_base_name)
        robot_base_id = robot_base_idx[0]
        robot_pos = robot.data.body_pos_w[:, robot_base_id]
        robot_quat = robot.data.body_quat_w[:, robot_base_id]
        return robot_pos, robot_quat
        

    def get_door_normal(self, verbose = False):

        joint_angles = self.scene["door"].data.joint_pos
        door_pointcloud = sample_pointcloud(self.scene["door"].cfg.spawn.asset_path, joint_angles, device=self.device)
        door_pointcloud = quat_apply(self.scene["door"].data.body_quat_w[:, 0], door_pointcloud) + self.scene["door"].data.body_pos_w[:, 0]
        # joint_angles = self.scene["robot"].data.joint_pos
        # door_pointcloud = sample_pointcloud(self.scene["robot"].cfg.spawn.asset_path, joint_angles, device=self.device)
        # door_pointcloud = quat_rotate(self.scene["robot"].data.body_quat_w[:, 0], door_pointcloud) + self.scene["robot"].data.body_pos_w[:, 0]
        if verbose:
            from DoorOpening.utils.point_utils import tensor_to_ply
            tensor_to_ply(door_pointcloud[0], "pointcloud.ply")
        normals, centroids = fit_plane_batch_torch(door_pointcloud)
        return normals, centroids

    def compute_approach_target(self):
        # Perception Step
        door_normal, centroids = self.get_door_normal()
        # door_knob_pos = self.get_door_knob_pos(use_handle_body_name = False)
        robot_pos, robot_base_quat = self.get_robot_base_pos()
        x, y, theta = self.get_door_approach_pose(door_normal, centroids, robot_pos)
        x, y, theta = rebase_goal(x, y, theta, robot_pos, robot_base_quat)
        # print("door_knob_pos: ", door_knob_pos)
        # print("robot_pos: ", robot_pos)
        # print("xytheta: ", xytheta)
        self.base_pose = torch.stack([x, y, theta], dim=-1)
        return torch.stack([x, y, theta], dim=-1)
        
    def compute_arm_target(self):
        # Get current robot state
        ee_id = self.scene["robot"].find_bodies("palm_lower")[0][0]
        # print("ee_idx: ", ee_idx)
        ee_quat_w = self.scene["robot"].data.body_quat_w[:, ee_id]
        ee_pos = self.scene["robot"].data.body_pos_w[:, ee_id]
        ee_quat = self.scene["robot"].data.body_quat_w[:, ee_id]
        
        hand_idx = self.scene["robot"].find_joints(FRANKA_JOINT_NAMES + BASE_JOINT_NAMES)[0]
        hand_jac = self.scene["robot"].root_physx_view.get_jacobians()[:, ee_id, :, hand_idx]
        current_joint_pos = self.scene["robot"].data.joint_pos[:, hand_idx]
        robot_base_pos = self.scene["robot"].data.root_pos_w[:, :3]
        door_knob_pos = self.get_door_knob_pos()
        # print("door_base_pos: ", self.scene["door"].data.body_pos_w[:, 0])
        door_knob_pos = (torch.linalg.norm(door_knob_pos - ee_pos, dim=-1) - 0.1) * (door_knob_pos - ee_pos) / torch.linalg.norm(door_knob_pos - ee_pos, dim=-1)
        # print("door_knob_pos: ", door_knob_pos)
        # print("ee_pos: ", ee_pos)
        
        # self.ik_controller.reset()
        self.ik_controller.set_command(command=door_knob_pos, ee_pos=ee_pos, ee_quat=ee_quat)
        joint_pos_des = self.ik_controller.compute(
            ee_pos,
            ee_quat,
            hand_jac,
            current_joint_pos,
        )

        joint_pos = self.scene["robot"].data.joint_pos
        # print("joint_pos: ", current_joint_pos)
        joint_pos[:, hand_idx] = joint_pos_des

        # if self.base_pose is not None:
        #     base_idx = self.scene["robot"].find_joints(BASE_JOINT_NAMES)[0]
        #     joint_pos[:, base_idx] = self.base_pose
        return joint_pos


    def door_opening_motion(self):
        door_joint_pos = self.scene["door"].data.joint_pos