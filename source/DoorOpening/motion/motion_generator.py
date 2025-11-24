import torch
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_rotate

from isaaclab.utils import convert_dict_to_backend

import omni
from pxr import Usd, UsdGeom
import numpy as np
import torch
from isaaclab.utils.math import quat_rotate_inverse

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

    def estimate_door_plane_pca(self, point_cloud):
        """
        Approximates RANSAC plane fitting using PCA/SVD.
        Takes a point cloud (N, 3) and returns the normal vector and centroid.
        """
        # 1. Calculate Centroid
        centroid = torch.mean(point_cloud, dim=0)
        
        # 2. Center the points
        centered_points = point_cloud - centroid
        
        # 3. Compute SVD (Singular Value Decomposition)
        # The normal of the plane is the singular vector corresponding to the smallest singular value
        try:
            u, s, vh = torch.linalg.svd(centered_points, full_matrices=False)
            normal = vh[-1, :] # The last row of Vh (or column of V) is the normal
        except:
            normal = torch.tensor([1.0, 0.0, 0.0], device=self.device) # Fallback

        return centroid, normal

    # def get_door_pointcloud(self, camera, verbose = False):
    #     camera_index = 0
    #     single_cam_data = convert_dict_to_backend(
    #         {k: v for k, v in camera.data.output.items()}, backend="numpy"
    #     )
    #     camera_K = camera.data.intrinsic_matrices[camera_index]
    #     depth = single_cam_data["distance_to_image_plane"]
    #     depth = depth.squeeze(-1)
    #     depth = torch.tensor(depth).to(self.device)

    #     if verbose:

    #         depth_draw = depth[0].clone().cpu().numpy()
    #         depth_draw[depth_draw < 0.1] = 0.1
    #         depth_draw[depth_draw > 3.0] = 3.0
        
    #         depth_draw = (depth_draw - depth_draw.min()) / (depth_draw.max() - depth_draw.min())
    #         import cv2
    #         cv2.imwrite("depth.png", np.array(depth_draw * 255.0).astype(np.uint8))
        
    #     pos = camera.data.pos_w
    #     quat = camera.data.quat_w_world
    #     print(depth.shape, pos.shape, quat.shape, camera_K.shape)
    #     points = depth_to_pointcloud(depth, camera_K, pos, quat)
    #     print(points.shape)
    #     tensor_to_ply(points[0], "points.ply")
    #     return points

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
        if verbose:
            print("door_pointcloud: ", door_pointcloud.shape)
            from DoorOpening.utils.point_utils import tensor_to_ply
            tensor_to_ply(door_pointcloud[0], "door_pointcloud.ply")
        centroid, normal = self.estimate_door_plane_pca(door_pointcloud)
        print("normal: ", normal)
        print("centroid: ", centroid)
        return normal, centroid
        

        # prim_path = self.scene.cfg.door.prim_path
        
        # points = extract_articulation_pointcloud(prim_path)
        # print("points: ", points.shape)
        
        # if len(points) > 100:
        #     # Transform points from Camera Frame to World Frame
        #     cam_pos = self.scene["camera"].data.pos_w[0]
        #     cam_quat = self.scene["camera"].data.quat_w_ros[0]
            
        #     # Rotate points
        #     points_w = quat_rotate(cam_quat.repeat(len(points), 1), points) + cam_pos
            
        #     # Fit Plane (Logic similiar to RANSAC)
        #     centroid, normal = self.estimate_door_plane_pca(points_w)
            
        #     # Ensure normal points towards robot (dot product with vector to robot)
        #     vec_to_robot = cam_pos - centroid
        #     if torch.dot(vec_to_robot, normal) < 0:
        #         normal = -normal
        # else:
        #     # Fallback if camera sees nothing
        #     normal = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        #     centroid = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # # --- B. Sample Knob Position ---
        # # Getting knob from raw point cloud is hard without a segmentation network.
        # # For this generator, we will use ground truth for the KNOB, but use calculated NORMAL for approach.
        # # Assume the knob is a rigid body named "knob" or similar in the door articulation
        # # Here we cheat and get the door link index. Replace '2' with actual knob link index.
        # knob_pos = self.scene["door"].data.body_pos_w[:, 1, :] # Assuming index 1 is knob
        
        # return normal, knob_pos

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
        door_normal, door_knob_pos = self.get_door_normal_and_knob()

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