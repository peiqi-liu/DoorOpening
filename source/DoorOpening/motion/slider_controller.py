import torch
import omni.ui as ui
from functools import partial

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

class OmniJointController:
    def __init__(self, scene: InteractiveScene, joint_names):
        self.scene = scene
        self.joint_names = joint_names

        # Find joint indices in articulation
        self.joint_ids, _ = scene["robot"].find_joints(joint_names)

        # IsaacLab tensors are (num_envs, num_joints)
        self.q_slider = scene["robot"].data.default_joint_pos.clone()

        self._build_ui()

    def _build_ui(self):
        self.window = ui.Window(
            title="Robot Joint Controller",
            width=350,
            height=600,
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

    def _on_slider_changed(self, idx, model):
        # Single env → env index = 0
        joint_id = self.joint_ids[idx]
        value = model.get_value_as_float()

        self.q_slider[0, joint_id] = value
