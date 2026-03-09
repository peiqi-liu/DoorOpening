import torch
import omni.ui as ui
from functools import partial
import pickle as pkl
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from scipy.interpolate import CubicSpline
import numpy as np  
from DoorOpening.utils.finger_utils import tendon_to_joint_angle_utils, joint_angle_to_tendon_utils

class OmniJointController:
    def __init__(self, scene: InteractiveScene, joint_names):
        self.scene = scene

        self.joint_names = joint_names
        self.joint_ids, self.joint_names = scene["robot"].find_joints(joint_names)

        self.robot_joint_pos_upper_limit = scene["robot"].data.joint_pos_limits[..., self.joint_ids, 1]
        self.robot_joint_pos_lower_limit = scene["robot"].data.joint_pos_limits[..., self.joint_ids, 0]

        self.door_joint_names = scene["door"].data.joint_names
        self.door_joint_ids, self.door_joint_names = scene["door"].find_joints(self.door_joint_names)

        self.door_joint_pos_upper_limit = scene["door"].data.joint_pos_limits[..., self.door_joint_ids, 1]
        self.door_joint_pos_lower_limit = scene["door"].data.joint_pos_limits[..., self.door_joint_ids, 0]

        self.key_pose_idx, _ = scene["robot"].find_bodies("palm_center")
        self.key_pose_idx = self.key_pose_idx[0]
        self.pose_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        self.q_slider = scene["robot"].data.joint_pos.clone()
        self.door_q_slider = scene["door"].data.joint_pos.clone()
        self.finger_q_slider = torch.zeros(4)
        finger_joint_names = [f"finger_joint_{i}" for i in range(15)]
        self.finger_joint_ids, finger_joint_names = scene["robot"].find_joints(finger_joint_names)

        self.xyz = scene["robot"].data.body_pos_w[:, self.key_pose_idx].clone()
        quat = scene["robot"].data.body_quat_w[:, self.key_pose_idx].clone()
        self.euler_angles = torch.stack(euler_xyz_from_quat(quat), dim = -1)

        # ===== Keyframe + playback state =====
        self.key_poses = []          # list[Tensor (1, num_joints)]
        self.door_key_joint_angles = []     # list[Tensor (1, num_door_joints)]
        self.door_pos_traj = []            # list[Tensor (1, num_door_joints)]
        self.traj = None             # Tensor (T, num_joints)
        self.playback = False
        self.play_idx = 0

        self.total_steps = 1000

        self.pose_sliders = []
        self.joint_sliders = []

        self._initialize_trajectory()

        self._build_ik_controller()

        self._build_ui()

    def _build_ik_controller(self):
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05}
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.scene.num_envs, device=self.scene["robot"].data.joint_pos.device
        )
        self.ik_controller.reset()
        self.joint_pos_des = None

    def _build_ui(self):
        self.window = ui.Window(
            title="Joint Controller",
            width=350,
            height=800,
            dockPreference=ui.DockPreference.LEFT,
        )

        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Joint Position Control (rad)", height=30)

                # 6 end effector pose sliders
                for i, name in enumerate(self.pose_names):
                    ui.Label(name)

                    slider = ui.FloatSlider(
                        min=-3.14,
                        max=3.14,
                        step=0.01,
                        height=18,
                    )

                    slider.model.add_value_changed_fn(
                        partial(self._on_pose_changed, i)
                    )
                    slider.model.set_value(self.xyz[0, i] if i < 3 else self.euler_angles[0, i - 3])
                    self.pose_sliders.append(slider)
                
                ui.Button("Write XYZ to sim", clicked_fn=self._write_xyz_to_sim)

                # robot joint position sliders
                for i, name in enumerate(self.joint_names):
                    if name.startswith("finger_joint_"):
                        continue
                    ui.Label(name)

                    if name.startswith("base_"):
                        self.robot_joint_pos_lower_limit[0, i] = -4.5
                        self.robot_joint_pos_upper_limit[0, i] = 4.5

                    slider = ui.FloatSlider(
                        min=self.robot_joint_pos_lower_limit[0, i],
                        max=self.robot_joint_pos_upper_limit[0, i],
                        step=0.01,
                        height=18,
                    )

                    slider.model.set_value(self.q_slider[0, self.joint_ids[i]])
                    slider.model.add_value_changed_fn(
                        partial(self._on_slider_changed, i)
                    )
                    self.joint_sliders.append(slider)

                for i in range(4):
                    ui.Label(f"finger_joint_{i}")
                    slider = ui.FloatSlider(
                        min=0.0,
                        max=3.14,
                        step=0.01,
                        height=18,
                    )
                    slider.model.set_value(0.0)
                    slider.model.add_value_changed_fn(partial(self._on_finger_slider_changed, i))

                # door joint position sliders
                for i, name in enumerate(self.door_joint_names):
                    ui.Label("door_" + name)

                    slider = ui.FloatSlider(
                        min=self.door_joint_pos_lower_limit[0, i],
                        max=self.door_joint_pos_upper_limit[0, i],
                        step=0.01,
                        height=18,
                    )

                    slider.model.set_value(self.door_q_slider[0, self.door_joint_ids[i]])
                    slider.model.add_value_changed_fn(
                        partial(self._on_door_slider_changed, i)
                    )
                
                ui.Separator(height=10)

                ui.Button("Record Key Pose", clicked_fn=self._record_key_pose)
                ui.Button("Play Trajectory", clicked_fn=self._start_playback)
                ui.Button("Save Trajectory", clicked_fn=self._save_trajectory)

                ui.Label("Estimated Trajectory Length:", width=160)

                length_field = ui.IntField(
                    min=1,
                    max=1000,
                    step=1,
                    width=120,
                ) 
                length_field.model.add_value_changed_fn(
                    partial(self._on_length_changed)
                )

                ui.Separator(height=10)

    def _on_length_changed(self, model):
        self.total_steps = model.get_value_as_int()

    def _record_key_pose(self):
        # q = self.q_slider.clone()
        q = self.scene["robot"].data.joint_pos.clone()
        self.key_poses.append(q)
        print(f"[KEYPOSE] Recorded #{len(self.key_poses)}")
        door_q = self.scene["door"].data.joint_pos.clone()
        self.door_key_joint_angles.append(door_q)
        print(f"[DOOR KEYPOSE] Recorded #{len(self.door_key_joint_angles)}")

    def _build_trajectory(self):
        assert len(self.key_poses) >= 2, "Need at least 2 key poses"
        assert len(self.door_key_joint_angles) >= 2, "Need at least 2 door key poses"

        num_key_poses = self.key_poses[0].shape[-1]
        num_door_poses = self.door_key_joint_angles[0].shape[-1]

        combined_poses = [torch.cat((i, j), dim = -1) for i, j in zip(self.key_poses, self.door_key_joint_angles)]
        qs = torch.stack([kp[0] for kp in combined_poses]).detach().cpu().numpy()

        # automatic timing
        dq = np.linalg.norm(qs[1:] - qs[:-1], axis=1)
        dt = np.maximum(dq / 1.0, 0.1)
        t_key = np.concatenate([[0], np.cumsum(dt)])
        t_key /= t_key[-1]

        cs = CubicSpline(t_key, qs, axis=0, bc_type="clamped")

        print("self.total_steps: ", self.total_steps)
        t = np.linspace(0, 1, self.total_steps)
        self.traj = torch.tensor(cs(t))
        self.qd = torch.tensor(cs(t, 1))
        self.qdd = torch.tensor(cs(t, 2))

        key_indices = np.searchsorted(t, t_key)
        key_indices = np.clip(key_indices, 0, len(t) - 1)

        self.door_traj = self.traj[:, num_key_poses:]
        self.traj = self.traj[:, :num_key_poses]
        self.qd = self.qd[:, :num_key_poses]
        self.qdd = self.qdd[:, :num_key_poses]

        self.key_indices = torch.from_numpy(key_indices)

        self.play_idx = 0

    def _initialize_trajectory(self):
        self.robot_body_pos_traj = []
        self.robot_body_quat_traj = []
        self.door_pos_traj = []

    def _start_playback(self):
        self._initialize_trajectory()
        self._build_trajectory()
        self.playback = True
        print(f"[PLAYBACK] Trajectory length: {len(self.traj)}")

    def _save_trajectory(self, path="trajectory.pkl"):
        if self.traj is None or self.door_traj is None:
            self._build_trajectory()
        data = {
            "robot_joint_pos_traj": self.traj, 
            "robot_joint_vel_traj": self.qd, 
            "robot_joint_acc_traj": self.qdd, 
            "door_traj": self.door_traj, 
            "key_indices": self.key_indices
        }
        if len(self.robot_body_pos_traj) > 0:
            self.robot_body_pos_traj = torch.stack(self.robot_body_pos_traj, dim = 0)
            data["robot_body_pos_traj"] = self.robot_body_pos_traj
        if len(self.robot_body_quat_traj) > 0:
            self.robot_body_quat_traj = torch.stack(self.robot_body_quat_traj, dim = 0)
            data["robot_body_quat_traj"] = self.robot_body_quat_traj
        if len(self.door_pos_traj) > 0:
            self.door_pos_traj = torch.stack(self.door_pos_traj, dim = 0)
            data["door_pos_traj"] = self.door_pos_traj

        self._initialize_trajectory()

        with open(path, "wb") as f:
            pkl.dump(data, f)

        print(f"[SAVE] Trajectory saved to {path}")

    def _on_finger_slider_changed(self, idx, model):
        self._initialize_trajectory()
        value = model.get_value_as_float()
        self.finger_q_slider[idx] = value
        new_q_value = tendon_to_joint_angle_utils(self.scene["robot"], self.finger_q_slider)
        self.q_slider[..., self.finger_joint_ids] = new_q_value[..., self.finger_joint_ids]

        test_tendon_value =joint_angle_to_tendon_utils(self.scene["robot"])

    def _on_slider_changed(self, idx, model):
        self._initialize_trajectory()
        # Single env → env index = 0
        joint_id = self.joint_ids[idx]
        value = model.get_value_as_float()

        self.q_slider[0, joint_id] = value

    def _on_door_slider_changed(self, idx, model):
        self._initialize_trajectory()
        joint_id = self.door_joint_ids[idx]
        value = model.get_value_as_float()

        self.door_q_slider[0, joint_id] = value
        self._sync_pose_from_sim()

    def _on_pose_changed(self, idx, model):
        self._initialize_trajectory()
        value = model.get_value_as_float()

        if idx <= 2:
            self.xyz[..., idx] = value
        else:
            self.euler_angles[..., idx - 3] = value

        roll, pitch, yaw = self.euler_angles.unbind(dim = -1)
        quat = quat_from_euler_xyz(roll, pitch, yaw)

        if hasattr(self, "goal_marker"):
            self.goal_marker.visualize(
                translations=self.xyz,
                orientations=quat,
            )

    def _write_xyz_to_sim(self):
        roll, pitch, yaw = self.euler_angles.unbind(dim = -1)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        ee_pos = self.scene["robot"].data.body_pos_w[:, self.key_pose_idx]
        ee_quat = self.scene["robot"].data.body_quat_w[:, self.key_pose_idx]
        hand_jac = self.scene["robot"].root_physx_view.get_jacobians()[:, self.key_pose_idx, :, self.joint_ids[3:]]
        current_joint_pos = self.scene["robot"].data.joint_pos[:, self.joint_ids[3:]]

        self.ik_controller.reset()

        self.ik_controller.set_command(command=torch.cat((self.xyz, quat), dim=-1), ee_pos=ee_pos, ee_quat=ee_quat)
        self.joint_pos_des = self.ik_controller.compute(
            ee_pos,
            ee_quat,
            hand_jac,
            current_joint_pos,
        )

        joint_pos_des = torch.clamp(self.joint_pos_des, self.robot_joint_pos_lower_limit[..., 3:], self.robot_joint_pos_upper_limit[..., 3:])

        # Write joint positions to sim
        # The robot might not reach the pose in the first shot, keep pressing the button to keep reaching the pose
        if joint_pos_des is not None:
            self.q_slider[0, self.joint_ids[3:]] = joint_pos_des
        self._apply_joint_positions(self.q_slider, sync_pose=False)


    def _sync_pose_from_sim(self):
        pos = self.scene["robot"].data.body_pos_w[:, self.key_pose_idx]
        quat = self.scene["robot"].data.body_quat_w[:, self.key_pose_idx]

        self.xyz[:] = pos
        roll, pitch, yaw = euler_xyz_from_quat(quat)
        self.euler_angles[:] = torch.stack((roll, pitch, yaw), -1)

        for i in range(3):
            self.pose_sliders[i].model.set_value(float(self.xyz[0, i]))
        for i in range(3):
            self.pose_sliders[i + 3].model.set_value(float(self.euler_angles[0, i]))

    def _sync_joint_sliders(self):
        for i, jid in enumerate(self.joint_ids):
            self.joint_sliders[i].model.set_value(float(self.q_slider[0, jid]))

    def _apply_joint_positions(self, q, sync_pose = True):
        self.q_slider[:] = q

        # write to sim
        self.scene["robot"].set_joint_position_target(q)

        # FK → pose
        if sync_pose:
            self._sync_pose_from_sim()
        self._sync_joint_sliders()
