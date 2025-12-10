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
from DoorOpening.utils.pose_utils import unbase_goal, world_to_local

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
            ik_params={"lambda_val": 0.2}
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.device
        )
        self.ik_controller.reset()
        
        # Perception Step
        self.door_normal, self.centroids = self.get_door_normal()
        
        self.count = 0
        self.prev_ee_pos = []

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
    
    
    def get_door_knob_pos(self, door_handle_body_name = None):
        door = self.scene["door"]
        if door_handle_body_name is None:
            handle_body_name = self.handle_body_name
        else:
            handle_body_name = door_handle_body_name
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
        if verbose:
            from DoorOpening.utils.point_utils import tensor_to_ply
            tensor_to_ply(door_pointcloud[0], "pointcloud.ply")
        normals, centroids = fit_plane_batch_torch(door_pointcloud)
        return normals, centroids

    def compute_approach_target(self):
        # Perception Step
        door_pos = self.scene["door"].data.body_pos_w[:, 0]
        robot_pos, robot_base_quat = self.get_robot_base_pos()
        x, y, theta = self.get_door_approach_pose(self.door_normal, door_pos, robot_pos)
        x, y, theta = unbase_goal(torch.stack([x, y, theta], dim=-1), robot_pos, robot_base_quat).unbind(dim=-1)
        joint_pos_des =self.scene["robot"].data.default_joint_pos.clone()
        joint_pos_des[..., :3] = torch.stack([x, y, theta], dim=-1)
        self.joint_pos_des = joint_pos_des
        return joint_pos_des
        
    def compute_arm_target(self, compute_base = True):
        if compute_base:
            target_joint_names = FRANKA_JOINT_NAMES + BASE_JOINT_NAMES
        else:
            target_joint_names = FRANKA_JOINT_NAMES
        # Get current robot state
        ee_id = self.scene["robot"].find_bodies("fingertip_1")[0][0]
        # print("ee_idx: ", ee_idx)
        ee_quat_w = self.scene["robot"].data.body_quat_w[:, ee_id]
        ee_pos = self.scene["robot"].data.body_pos_w[:, ee_id]
        ee_quat = self.scene["robot"].data.body_quat_w[:, ee_id]
        
        hand_idx = self.scene["robot"].find_joints(target_joint_names)[0]
        hand_jac = self.scene["robot"].root_physx_view.get_jacobians()[:, ee_id, :, hand_idx]
        current_joint_pos = self.scene["robot"].data.joint_pos[:, hand_idx].clone()
        door_knob_pos = self.get_door_knob_pos()
        # print("door_base_pos: ", self.scene["door"].data.body_pos_w[:, 0])
        door_knob_pos = door_knob_pos - ee_pos
        # print("ee_pos: ", ee_pos)
        
        # self.ik_controller.reset()
        self.ik_controller.set_command(command=door_knob_pos, ee_pos=ee_pos, ee_quat=ee_quat)
        joint_pos_des = self.ik_controller.compute(
            ee_pos,
            ee_quat,
            hand_jac,
            current_joint_pos,
        )

        joint_pos = self.scene["robot"].data.joint_pos.clone()
        # print("joint_pos: ", current_joint_pos)
        joint_pos[:, hand_idx] = joint_pos_des

        # if self.base_pose is not None:
        #     base_idx = self.scene["robot"].find_joints(BASE_JOINT_NAMES)[0]
        #     joint_pos[:, base_idx] = self.base_pose
        return joint_pos

    def open_door(self):
        step_size = (self.scene["door"].data.soft_joint_pos_limits[..., 1] - \
             self.scene["door"].data.soft_joint_pos_limits[..., 0]) * 0.1
        target_door_pos = self.scene["door"].data.joint_pos + step_size
        self.scene["door"].write_joint_position_to_sim(target_door_pos)
    
    def follow_door(self):
        ee_id = self.scene["robot"].find_bodies("fingertip_3")[0][0]
        ee_pos = self.scene["robot"].data.body_pos_w[:, ee_id]
        ee_quat = self.scene["robot"].data.body_quat_w[:, ee_id]

        door_knob_pos = self.get_door_knob_pos(door_handle_body_name = "link_1")
        door_knob_pos = door_knob_pos - ee_pos

        hand_idx = self.scene["robot"].find_joints(FRANKA_JOINT_NAMES + BASE_JOINT_NAMES)[0]
        hand_jac = self.scene["robot"].root_physx_view.get_jacobians()[:, ee_id, :, hand_idx]
        current_joint_pos = self.scene["robot"].data.joint_pos[:, hand_idx]

        self.ik_controller.set_command(command=door_knob_pos, ee_pos=ee_pos, ee_quat=ee_quat)
        joint_pos_des = self.ik_controller.compute(
            ee_pos,
            ee_quat,
            hand_jac,
            current_joint_pos,
        )

        joint_pos = self.scene["robot"].data.joint_pos.clone()
        joint_pos[:, hand_idx] = joint_pos_des
        return joint_pos

    def check_ee_pos(self):
        ee_id = self.scene["robot"].find_bodies("fingertip_3")[0][0]
        ee_pos = self.scene["robot"].data.body_pos_w[:, ee_id]
        door_knob_pos = self.get_door_knob_pos(door_handle_body_name = "link_1")
        door_knob_pos = door_knob_pos - ee_pos
        
        self.count += 1
        # print(self.prev_ee_pos)
        # print("pose_reached: ", torch.linalg.norm(door_knob_pos, dim=-1))
        # print("moved: ", torch.linalg.norm(self.prev_ee_pos[-1] - ee_pos, dim=-1), len(self.prev_ee_pos))

        _, centroids = self.get_door_normal()
        robot_pos, robot_base_quat = self.get_robot_base_pos()

        return (torch.linalg.norm(door_knob_pos, dim=-1) < 0.17).item(), (torch.linalg.norm(door_knob_pos, dim=-1) > 0.6).item()

    def door_opening_motion(self):
        pose_reached, far = self.check_ee_pos()
        if pose_reached:
            self.open_door()
            return None
        # elif far:
        #     print("jittering robot")
        #     return self.compute_approach_target()
        else:
            joint_pos_target = self.compute_arm_target(compute_base = True)
            # print("joint_pos_target: ", joint_pos_target[..., :10])
            # print("joint_pos: ", self.scene["robot"].data.joint_pos[..., :10])
            return joint_pos_target

    def move_away_from_door(self):
        # Assume:
        # - self.door_normal: (..., 3) — outward normal (from room A to room B) → points in traversal direction
        # - self.door_side: "left" or "right" — which side the hinges are on (from robot's approach view)
        door_normal_xy = self.door_normal[..., :2]  # direction through doorway
        door_normal_xy = door_normal_xy / torch.linalg.norm(door_normal_xy, dim=-1, keepdim=True).clamp_min(1e-8)
        # Perpendicular to door_normal (i.e., along door width)
        # e.g., for door_normal = [1, 0] (facing +X), perp = [0, 1] or [0, -1]
        perp = torch.stack([-door_normal_xy[..., 1], door_normal_xy[..., 0]], dim=-1)  # 90° CCW

        # Determine safe lateral offset direction:
        # If hinges on RIGHT (common case), door swings LEFT → robot should step LEFT (i.e., +perp if perp is leftward)
        # But sign depends on coordinate convention! Let’s define:
        #   self.door_hinge_side = +1 for right-hinged (door swings CCW), -1 for left-hinged (CW)
        # Then: safe_offset_dir = door_normal_xy + hinge_side_factor * perp
        hinge_side_factor = 0.5  # e.g., +0.5 for right-hinged (bias left), -0.5 for left-hinged

        # Desired direction: mostly forward, slightly sideways to clear door
        move_dir = door_normal_xy + hinge_side_factor * perp
        move_dir = move_dir / torch.linalg.norm(move_dir, dim=-1, keepdim=True).clamp_min(1e-8)

        # Step forward ~0.3–0.5m in that direction
        joint_pos = self.scene["robot"].data.joint_pos.clone()
        joint_pos[..., :2] += move_dir * 0.4  # tunable step size

        return joint_pos

    # def door_opening_motion(self):
    #     pose_reached, far = self.check_ee_pos()
    #     # print("count: ", self.count)
    #     # print("pose_reached: ", pose_reached)   
    #     # print("far: ", far)
    #     if pose_reached:
    #         self.count = 0
    #         self.open_door()
    #         return None
    #     elif self.count > 40:
    #         self.count = 0
    #         print("jittering robot")
    #         # return self.jitter_robot()
    #         return self.compute_approach_target()
    #     else:
    #         joint_pos_target = self.compute_arm_target(compute_base = far)
    #         # print("joint_pos_target: ", joint_pos_target[..., :10])
    #         # print("joint_pos: ", self.scene["robot"].data.joint_pos[..., :10])
    #         return joint_pos_target

    # def door_opening_motion(self, step):
    #     if step > 100:
    #         step = 100
    #     step_size = (self.scene["door"].data.soft_joint_pos_limits[..., 1] - \
    #          self.scene["door"].data.soft_joint_pos_limits[..., 0]) * int(step) / 100

    #     # step_size = (self.scene["door"].data.soft_joint_pos_limits[..., 1] - \
    #     #      self.scene["door"].data.soft_joint_pos_limits[..., 0]) * 0.02

    #     target_door_pos = self.scene["door"].data.soft_joint_pos_limits[..., 0] + step_size
    #     # target_door_pos = self.scene["door"].data.joint_pos + step_size
    #     print("target_door_pos: ", self.scene["door"].data.joint_pos)
    #     self.scene["door"].write_joint_position_to_sim(target_door_pos)
    #     print("door_pos: ", self.scene["door"].data.joint_pos)

    #     _, centroids = self.get_door_normal()
    #     joint_pos = self.scene["robot"].data.joint_pos.clone()
    #     robot_xy = joint_pos[..., :2]
    #     # push the robot away from the centroid of the door
    #     robot_xy[..., 1] = robot_xy[..., 1] + (centroids[..., 1] - robot_xy[..., 1]) * 0.003 / torch.norm(centroids[..., 1] - robot_xy[..., 1], dim = -1)
    #     robot_xy[..., 0] = robot_xy[..., 0] - (centroids[..., 0] - robot_xy[..., 0]) * 0.003 / torch.norm(centroids[..., 0] - robot_xy[..., 0], dim = -1)
    #     joint_pos[..., :2] = robot_xy
    #     self.scene["robot"].write_joint_position_to_sim(joint_pos)

    #     ee_id = self.scene["robot"].find_bodies("fingertip_1")[0][0]
    #     ee_pos = self.scene["robot"].data.body_pos_w[:, ee_id]
    #     ee_quat = self.scene["robot"].data.body_quat_w[:, ee_id]

    #     door_knob_pos = self.get_door_knob_pos(door_handle_body_name = "link_1")
    #     door_knob_pos = door_knob_pos - ee_pos

    #     hand_idx = self.scene["robot"].find_joints(FRANKA_JOINT_NAMES + BASE_JOINT_NAMES)[0]
    #     hand_jac = self.scene["robot"].root_physx_view.get_jacobians()[:, ee_id, :, hand_idx]
    #     current_joint_pos = self.scene["robot"].data.joint_pos[:, hand_idx]

    #     self.ik_controller.set_command(command=door_knob_pos, ee_pos=ee_pos, ee_quat=ee_quat)
    #     joint_pos_des = self.ik_controller.compute(
    #         ee_pos,
    #         ee_quat,
    #         hand_jac,
    #         current_joint_pos,
    #     )

    #     # print("ee_pos: ", ee_pos)
    #     # print("door_knob_pos: ", door_knob_pos)

    #     joint_pos = self.scene["robot"].data.joint_pos
    #     joint_pos[:, hand_idx] = joint_pos_des

    #     return joint_pos