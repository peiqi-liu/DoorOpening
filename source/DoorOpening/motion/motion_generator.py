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
    def __init__(self, scene: InteractiveScene, device="cuda"):
        self.scene = scene
        self.device = device
        self.num_envs = scene.num_envs
        
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

    def get_door_normal_and_knob(self, verbose = False):
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
            print("door_pointcloud: ", door_pointcloud.shape)
            from DoorOpening.utils.point_utils import tensor_to_ply
            tensor_to_ply(door_pointcloud[0], "pointcloud.ply")
        # print("door_pos: ", self.scene["door"].data.body_pos_w[:, 0])
        # print("door_quat: ", self.scene["door"].data.body_quat_w[:, 0])
        # print("robot_pos: ", self.scene["robot"].data.body_pos_w[:, 0])
        # print("robot_quat: ", self.scene["robot"].data.body_quat_w[:, 0])
        normals, centroids = fit_plane_batch_torch(door_pointcloud)
        normal = normals[0]
        centroid = centroids[0]
        print("normal: ", normal)
        print("centroid: ", centroid)
        return normal

    def compute_action(self):
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
        door_normal = self.get_door_normal_and_knob()

        # num_dof = self.scene["robot"].num_joints
        # actions = torch.zeros((self.num_envs, num_dof), device=self.device)
        
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