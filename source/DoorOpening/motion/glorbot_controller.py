import time
import numpy as np
import threading

from pyRMP.rmp.rmp import RMPFlow
from pyRMP.global_planners.se2_planner import SE2Planner, SE2Planner_config
from pyRMP.rmp_configs.glorbot_rmp_configs import GlorbotRMPConfigs


def arc_length_xy(base_se2_plan: np.ndarray) -> float:
    """
    Arc length of the XY path only.
    base_se2_plan: (N,3) array [x, y, theta]
    """
    P = np.asarray(base_se2_plan, dtype=float)
    if P.shape[0] < 2: 
        return 0.0
    # (N-1, 2)
    dxy = np.diff(P[:, :2], axis=0)
    return np.linalg.norm(dxy, axis=1).sum()


class GlorbotRMPController:
    def __init__(self):
        self.rmp_configs = GlorbotRMPConfigs()
        # self.rmp_configs.speed_scalar = 1
        # self.rmp_configs.num_integration_steps = 3 #3 #4
        # creates combined RMP object
        self.rmpflow = RMPFlow(robot_urdf=self.rmp_configs.robot_urdf)
        # apply acceleration constraints
        self.rmpflow.add_acceleration_constraints(self.rmp_configs.acceleration_constraints)
        # apply jerk constraints
        self.rmpflow.add_jerk_constraints(self.rmp_configs.jerk_constraints)
        # creates collision model for the robot
        self.rmpflow.add_collision_model(
            collision_model_dict=self.rmp_configs.get_collision_model(),
            self_collision_pairs_dict=self.rmp_configs.get_self_collision_pairs(),
        )
        # se2 base global planner
        se2_planner_cfg = SE2Planner_config
        # se2_planner_cfg["verbose"] = True
        se2_planner_cfg["rate_hz"] = 10
        se2_planner_cfg["cell_size"] = 0.2
        se2_planner_cfg["robot_base_radius"] = 0.25
        se2_planner_cfg["step_length"] = 0.01
        se2_planner_cfg["skip_index"] = 20 #50
        self.se2_planner = SE2Planner(se2_planner_cfg)

        # --- thread-safety guards ---
        self._rmpflow_lock = threading.Lock()      # protects all rmpflow calls/state
        self._point_cloud_lock = threading.Lock()  # protects self.global_point_cloud

        # instantiate any useful buffers
        self._create_buffers()

    def _create_buffers(self):
        self.arm_posture_joint_pos = np.array([0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.0])
        self.global_point_cloud = np.zeros((0, 3))

    def initialize(self, q=None):
        tidybot2_franka_rmps = self.rmp_configs.get_rmps()
        for rmp in tidybot2_franka_rmps:
            self.rmpflow.add_rmp(rmp)
        if q is None:
            q = self.rmp_configs.default_joint_pos
        self.rmpflow.initialize(q=q)
        self.se2_planner.start()
    

    def update_global_point_cloud(self, global_point_cloud):
        global_point_cloud = np.ascontiguousarray(global_point_cloud)
        with self._point_cloud_lock:
            # atomic swap of the reference under lock
            self.global_point_cloud = global_point_cloud
    
    def update_robot_point_cloud_world_frame(self, robot_point_cloud_world_frame):
        robot_point_cloud_world_frame = np.ascontiguousarray(robot_point_cloud_world_frame)
        with self._rmpflow_lock:
            self.rmpflow.update_scene_point_cloud(robot_point_cloud_world_frame)
    

    def get_action(
        self, base_pose, base_velocity, 
        franka_joint_positions, franka_joint_velocities,
        leap_joint_positions, leap_joint_velocities,
        arx_joint_positions, arx_joint_velocities,
        ee_target_pose, global_point_cloud=None, robot_point_cloud_world_frame=None,
    ):
        q = np.concatenate((base_pose, franka_joint_positions, leap_joint_positions, arx_joint_positions))
        qd = np.concatenate((base_velocity, franka_joint_velocities, leap_joint_velocities, arx_joint_velocities))

        # update global point cloud if provided
        if global_point_cloud is not None:
            self.update_global_point_cloud(global_point_cloud)

        # if robot's local point cloud is provided, set it as the scene point cloud in RMP
        if robot_point_cloud_world_frame is not None:
            self.update_robot_point_cloud_world_frame(robot_point_cloud_world_frame)

        # snapshot the latest global map atomically
        with self._point_cloud_lock:
            # make a local copy so we can safely release the lock immediately
            if self.global_point_cloud.size == 0:
                global_point_cloud = None
            else:
                global_point_cloud = self.global_point_cloud.copy()
        
        # if no global map provided, snapshot rmpflow’s local point cloud under its lock
        if global_point_cloud is None:
            with self._rmpflow_lock:
                pc = self.rmpflow.get_point_cloud(filtered=False)
                global_point_cloud = np.asarray(pc, dtype=np.float32).copy()

        # update se2 planner with global scene point cloud
        self.se2_planner.update_input(global_point_cloud, base_pose[0:2], ee_target_pose[0:2]) 
        # get planner results
        base_se2_plan = self.se2_planner.get_full_plan()
        base_target_attractor_target = self.se2_planner.get_way_point()

        if base_se2_plan is not None:
            plan_arc_length = arc_length_xy(base_se2_plan)
            # print("Path Arc Length", plan_arc_length)
        else:
            plan_arc_length = None

        # set RMP objectives
        if base_target_attractor_target is not None:
            self.rmpflow.rmps["base_target_attractor"].update_guiding_waypoint(
                waypoint=base_target_attractor_target,
                guiding_path_length=plan_arc_length,
            )
            # self.rmpflow.rmps["base_axis_attractor"].update_guiding_waypoint(
            #     waypoint=base_target_attractor_target,
            #     guiding_path_length=plan_arc_length,
            # )
        
        self.rmpflow.set_rmp_objective("base_target_attractor", ee_target_pose[0:3])
        self.rmpflow.set_rmp_objective("base_axis_attractor", ee_target_pose)
        self.rmpflow.set_rmp_objective("ee_target_attractor", ee_target_pose)
        self.rmpflow.set_rmp_objective("joint_space_attractor", self.arm_posture_joint_pos)
        self.rmpflow.set_rmp_objective("camera_joint_space_attractor", np.array([0.0, 0.785, 0.785, 0.0, 0.0, 0.0]))

        # computes RMP joint acceleration targets
        speed_scalar = self.rmp_configs.speed_scalar
        num_integration_steps = self.rmp_configs.num_integration_steps
        robot_pos_target, robot_vel_target = self.rmpflow.compute_joint_targets(q, qd, num_integration_steps, speed_scalar=speed_scalar)
        # organize output
        arm_pos_target = robot_pos_target[3:]
        arm_vel_target = robot_vel_target[3:]
        base_vel_target_world_frame = robot_vel_target[0:3]
        base_pos_target_world_frame = robot_pos_target[0:3]
        return arm_pos_target, arm_vel_target, base_pos_target_world_frame, base_vel_target_world_frame, base_se2_plan, base_target_attractor_target
