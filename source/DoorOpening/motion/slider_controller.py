import torch
import omni.ui as ui
from functools import partial
import pickle as pkl
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

class OmniJointController:
    def __init__(self, scene: InteractiveScene, joint_names):
        self.scene = scene
        self.joint_names = joint_names
        self.joint_ids, self.joint_names = scene["robot"].find_joints(joint_names)

        self.door_joint_names = scene["door"].data.joint_names
        self.door_joint_ids, self.door_joint_names = scene["door"].find_joints(self.door_joint_names)

        self.q_slider = scene["robot"].data.default_joint_pos.clone()
        self.door_q_slider = scene["door"].data.default_joint_pos.clone()

        # ===== Keyframe + playback state =====
        self.key_poses = [scene["robot"].data.default_joint_pos.clone()]          # list[Tensor (1, num_joints)]
        self.door_key_joint_angles = [scene["door"].data.default_joint_pos.clone()]     # list[Tensor (1, num_door_joints)]
        self.traj = None             # Tensor (T, num_joints)
        self.playback = False
        self.play_idx = 0
        self.steps_per_segments = []

        self.step_temp = 60

        self._initialize_trajectory()

        self._build_ui()

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

                for i, name in enumerate(self.joint_names):
                    ui.Label(name)

                    slider = ui.FloatSlider(
                        min=-3.14,
                        max=3.14,
                        step=0.01,
                        height=18,
                    )

                    slider.model.add_value_changed_fn(
                        partial(self._on_slider_changed, i)
                    )

                for i, name in enumerate(self.door_joint_names):
                    ui.Label("door_" + name)

                    slider = ui.FloatSlider(
                        min=-3.14,
                        max=3.14,
                        step=0.01,
                        height=18,
                    )

                    slider.model.add_value_changed_fn(
                        partial(self._on_door_slider_changed, i)
                    )
                
                ui.Separator(height=10)

                ui.Button("Record Key Pose", clicked_fn=self._record_key_pose)
                ui.Button("Play Trajectory", clicked_fn=self._start_playback)
                ui.Button("Save Trajectory", clicked_fn=self._save_trajectory)

                ui.Label("Steps per key pose:", width=160)

                steps_field = ui.IntField(
                    min=1,
                    max=150,
                    step=10,
                    width=120,
                )
                steps_field.model.set_value(self.step_temp)

    def _record_key_pose(self):
        q = self.q_slider.clone()
        self.key_poses.append(q)
        print(f"[KEYPOSE] Recorded #{len(self.key_poses)}")
        self.door_key_joint_angles.append(self.door_q_slider.clone())
        print(f"[DOOR KEYPOSE] Recorded #{len(self.door_key_joint_angles)}")
        self.steps_per_segments.append(self.step_temp)

    def _build_trajectory(self):
        assert len(self.key_poses) >= 2, "Need at least 2 key poses"
        assert len(self.door_key_joint_angles) >= 2, "Need at least 2 door key poses"
        assert len(self.steps_per_segments) >= 1, "Need at least 1 steps per segment"

        traj = []
        for i in range(len(self.key_poses) - 1):
            q0 = self.key_poses[i][0]
            q1 = self.key_poses[i + 1][0]
            steps = self.steps_per_segments[i]

            for a in torch.linspace(0, 1, steps):
                traj.append((1 - a) * q0 + a * q1)

        self.traj = torch.stack(traj)   # (T, num_joints)

        door_traj = []
        for i in range(len(self.door_key_joint_angles) - 1):
            q0 = self.door_key_joint_angles[i][0]
            q1 = self.door_key_joint_angles[i + 1][0]
            steps = self.steps_per_segments[i]

            for a in torch.linspace(0, 1, steps):
                door_traj.append((1 - a) * q0 + a * q1)

        self.door_traj = torch.stack(door_traj)   # (T, num_joints)
        self.play_idx = 0

    def _initialize_trajectory(self):
        self.robot_body_pos_traj = []
        self.robot_body_quat_traj = []

    def _start_playback(self):
        self._initialize_trajectory()
        self._build_trajectory()
        self.playback = True
        print(f"[PLAYBACK] Trajectory length: {len(self.traj)}")

    def _save_trajectory(self, path="trajectory.pkl"):
        if self.traj is None or self.door_traj is None:
            self._build_trajectory()
        data = {"robot_joint_pos_traj": self.traj, "door_traj": self.door_traj}
        if len(self.robot_body_pos_traj) > 0:
            self.robot_body_pos_traj = torch.stack(self.robot_body_pos_traj, dim = 0)
            data["robot_body_pos_traj"] = self.robot_body_pos_traj
        if len(self.robot_body_quat_traj) > 0:
            self.robot_body_quat_traj = torch.stack(self.robot_body_quat_traj, dim = 0)
            data["robot_body_quat_traj"] = self.robot_body_quat_traj

        self._initialize_trajectory()

        # data = {
        #     "joint_names": self.joint_names,
        #     "q": self.traj.cpu(),
        #     "dt": self.scene.sim.get_physics_dt(),
        #     "source": "ui_keyframes",
        # }

        with open(path, "wb") as f:
            pkl.dump(data, f)

        print(f"[SAVE] Trajectory saved to {path}")

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
