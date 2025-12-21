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
        self.joint_ids, _ = scene["robot"].find_joints(joint_names)

        self.q_slider = scene["robot"].data.default_joint_pos.clone()

        # ===== Keyframe + playback state =====
        self.key_poses = []          # list[Tensor (1, num_joints)]
        self.traj = None             # Tensor (T, num_joints)
        self.playback = False
        self.play_idx = 0
        self.steps_per_segment = 60

        self._initialize_trajectory()

        self._build_ui()

    def _build_ui(self):
        self.window = ui.Window(
            title="Robot Joint Controller",
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

                    # ✅ IMPORTANT: use partial (NO lambda)
                    slider.model.add_value_changed_fn(
                        partial(self._on_slider_changed, i)
                    )
                
                ui.Separator(height=10)

                ui.Button("Record Key Pose", clicked_fn=self._record_key_pose)
                ui.Button("Play Trajectory", clicked_fn=self._start_playback)
                ui.Button("Save Trajectory", clicked_fn=self._save_trajectory)

    def _record_key_pose(self):
        q = self.q_slider.clone()
        self.key_poses.append(q)
        print(f"[KEYPOSE] Recorded #{len(self.key_poses)}")

    def _build_trajectory(self):
        assert len(self.key_poses) >= 2, "Need at least 2 key poses"

        traj = []
        for i in range(len(self.key_poses) - 1):
            q0 = self.key_poses[i][0]
            q1 = self.key_poses[i + 1][0]

            for a in torch.linspace(0, 1, self.steps_per_segment):
                traj.append((1 - a) * q0 + a * q1)

        self.traj = torch.stack(traj)   # (T, num_joints)
        self.play_idx = 0

    def _initialize_trajectory(self):
        self.door_joint_pos_traj = []
        self.robot_body_pos_traj = []
        self.robot_body_quat_traj = []

    def _start_playback(self):
        self._initialize_trajectory()
        self._build_trajectory()
        self.playback = True
        print(f"[PLAYBACK] Trajectory length: {len(self.traj)}")

    def _save_trajectory(self, path="trajectory.pkl"):
        if self.traj is None:
            self._build_trajectory()

        data = {"robot_joint_pos_traj": self.traj}
        if len(self.door_joint_pos_traj) > 0:
            self.door_joint_pos_traj = torch.stack(self.door_joint_pos_traj, dim = 0)
            data["door_joint_pos_traj"] = self.door_joint_pos_traj
        if len(self.robot_body_pos_traj) > 0:
            self.robot_body_pos_traj = torch.stack(self.robot_body_pos_traj, dim = 0)
            data["robot_body_pos_traj"] = self.robot_body_pos_traj
        if len(self.robot_body_quat_traj) > 0:
            self.robot_body_quat_traj = torch.stack(self.robot_body_quat_traj, dim = 0)
            data["robot_body_quat_traj"] = self.robot_body_quat_traj

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
        # Single env → env index = 0
        joint_id = self.joint_ids[idx]
        value = model.get_value_as_float()

        self.q_slider[0, joint_id] = value
