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
            use_relative_mode=False,
            ik_method="dls",
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.device
        )
        
        # State machine: 0 = Move Base, 1 = Move Arm, 2 = Done
        self.state = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def get_door_approach_pose(self, door_normal, door_handle_pos, robot_pos, offset=0.5):
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
    
    
    def get_door_knob_pos(self):
        door = self.scene["door"]
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
        return robot_pos
        

    def get_door_normal(self, verbose = False):
        """
        Perception Step: 
        1. Sample Point Cloud.
        2. Find Door Normal.
        3. Find Knob Position.
        """
        # --- A. Sample Point Cloud from Camera ---
        # camera_data shape: (num_envs, H, W, 3)
        # camera = self.scene["point_camera"]

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
        return normals

    def compute_approach_target(self):
        # Get current robot state
        ee_idx = self.scene["robot"].find_bodies("palm_lower")[0][0] # REPLACE with actual EE name
        # print("ee_idx: ", ee_idx)
        ee_jac = self.scene["robot"].root_physx_view.get_jacobians()[:, ee_idx, :, :]
        # print("ee_jac: ", ee_jac.shape)
        ee_pos = self.scene["robot"].data.body_pos_w[:, ee_idx, :]
        # print("ee_pos: ", ee_pos)
        ee_quat = self.scene["robot"].data.body_quat_w[:, ee_idx, :]
        # print("ee_quat: ", ee_quat)
        robot_base_pos = self.scene["robot"].data.root_pos_w[:, :3]
        # print("robot_base_pos: ", robot_base_pos)
        
        # Perception Step
        door_normal = self.get_door_normal()
        door_knob_pos = self.get_door_knob_pos()
        robot_pos = self.get_robot_base_pos()
        x, y, theta = self.get_door_approach_pose(door_normal, door_knob_pos, robot_pos)
        # print("door_knob_pos: ", door_knob_pos)
        # print("robot_pos: ", robot_pos)
        # print("xytheta: ", xytheta)
        return torch.stack([x, y, theta], dim=-1)
        
        # # --- Phase 1: Base Navigation (Stop 0.6m away) ---
        # target_stand_pos = door_knob_pos[0] + (door_normal * 0.6) # Target is 0.6m out along normal
        # target_stand_pos[2] = 0.0 # Keep target on ground
        
        # dist_to_target = torch.norm(robot_base_pos[0] - target_stand_pos)
        
        # if self.state[0] == 0:
        #     print(f"Approaching... Dist: {dist_to_target:.3f}")
            
        #     # Simple P-Controller for Base
        #     # Vector to target
        #     direction = target_stand_pos - robot_base_pos[0]
        #     direction[2] = 0 # Flatten
        #     direction = direction / torch.norm(direction)
            
        #     # Velocity command
        #     vel_cmd = direction * 1.0 # 1.0 m/s speed
            
        #     # This is pseudo-code mapping. You must map vel_cmd to your specific robot's wheel joints.
        #     # Example: If joints 0,1 are wheels:
        #     actions[:, 0] = vel_cmd[0] 
        #     actions[:, 1] = vel_cmd[1]
            
        #     # Transition condition
        #     if dist_to_target < 0.05:
        #         self.state[0] = 1 # Move to Arm Phase
                
        # # --- Phase 2: Arm Inverse Kinematics ---
        # elif self.state[0] == 1:
        #     print("Reaching for Knob...")
            
        #     # Set IK Target
        #     ik_commands = self.ik_controller.compute(
        #         ee_pos,
        #         ee_quat,
        #         ee_jac,
        #         door_knob_pos, # Target position
        #         torch.tensor([0, 0, 0, 1.0], device=self.device).repeat(self.num_envs, 1), # Target Rot (Identity for now)
        #     )
            
        #     # Apply to Arm Joints (assuming arm starts at index 2)
        #     # You need to map the IK result to the specific joint indices of your robot
        #     actions[:, 2:] = ik_commands[:, :] 

        # return actions