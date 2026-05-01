import numpy as np
import torch

from DoorOpening.utils.extract_pointcloud_from_articulation import TorchURDF, TorchSpheres



class GlorbotCollisionChecker:
    """Spherical collision representation for the 32-DoF mobile manipulator."""

    DEFAULT_JOINT_ORDER = [
        "base_x_joint",
        "base_y_joint",
        "base_rotation_joint",
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "finger_joint_1",
        "finger_joint_0",
        "finger_joint_2",
        "finger_joint_3",
        "finger_joint_12",
        "finger_joint_13",
        "finger_joint_14",
        "finger_joint_15",
        "finger_joint_5",
        "finger_joint_4",
        "finger_joint_6",
        "finger_joint_7",
        "finger_joint_9",
        "finger_joint_8",
        "finger_joint_10",
        "finger_joint_11",
        "x5_joint1",
        "x5_joint2",
        "x5_joint3",
        "x5_joint4",
        "x5_joint5",
        "x5_joint6",
    ]

    def __init__(self, urdf_path: str, device, input_joint_names=None):
        self.device = device
        self.robot = TorchURDF.load(urdf_path, lazy_load_meshes=True, device=self.device)

        self.input_joint_names = (
            list(input_joint_names) if input_joint_names is not None else list(self.DEFAULT_JOINT_ORDER)
        )
        self.robot_joint_names = [joint.name for joint in self.robot.actuated_joints]
        self.default_joint_mapping = self._build_joint_mapping()
        self.collision_model = self.get_collision_model()

        self._link_to_geometry = {}
        self._link_visual_origin_inv = {}
        for link in self.robot.links:
            if not link.visuals:
                continue
            visual = link.visuals[0]
            self._link_to_geometry[link.name] = visual.geometry
            self._link_visual_origin_inv[link.name] = torch.linalg.inv(
                visual.origin.to(self.device, dtype=torch.float32)
            )
        self._setup_sphere_buffers()

    def _build_joint_mapping(self):
        missing = [name for name in self.robot_joint_names if name not in self.input_joint_names]
        if missing:
            return None
        return [self.input_joint_names.index(name) for name in self.robot_joint_names]

    def _prepare_joint_angles(self, joint_angles, joint_mapping_list=None):
        if isinstance(joint_angles, np.ndarray):
            joint_angles = torch.from_numpy(joint_angles)
        if not isinstance(joint_angles, torch.Tensor):
            joint_angles = torch.tensor(joint_angles, dtype=torch.float32)
        joint_angles = joint_angles.to(self.device, dtype=torch.float32)
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.unsqueeze(0)
        if joint_angles.ndim != 2:
            raise ValueError(f"joint_angles must have shape (B, C) or (C,), got {joint_angles.shape}.")

        if joint_mapping_list is not None:
            joint_angles = joint_angles[:, joint_mapping_list]
        elif self.default_joint_mapping is not None and joint_angles.shape[1] == len(self.input_joint_names):
            joint_angles = joint_angles[:, self.default_joint_mapping]

        expected_dim = len(self.robot_joint_names)
        if joint_angles.shape[1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} joints in URDF order {self.robot_joint_names}, "
                f"got {joint_angles.shape[1]}. Pass joint_mapping_list if your order differs."
            )
        return joint_angles

    def _setup_sphere_buffers(self):
        link_names = []
        link_to_idx = {}
        sphere_link_idxs = []
        sphere_centers = []
        sphere_radii = []
        missing_links = []

        for link_name, spheres in self.collision_model.items():
            if link_name not in self._link_to_geometry:
                missing_links.append((link_name, link_name))
                continue

            if link_name not in link_to_idx:
                link_to_idx[link_name] = len(link_names)
                link_names.append(link_name)
            link_idx = link_to_idx[link_name]

            for center, radius in spheres:
                sphere_link_idxs.append(link_idx)
                sphere_centers.append(center)
                sphere_radii.append([radius])

        if missing_links:
            missing_str = ", ".join([f"{src}->{resolved}" for src, resolved in missing_links])
            raise ValueError(f"Collision model references missing URDF links: {missing_str}.")

        self._fk_links = link_names
        self._sphere_link_idxs = torch.tensor(sphere_link_idxs, dtype=torch.long, device=self.device)
        self._sphere_centers_local = torch.tensor(sphere_centers, dtype=torch.float32, device=self.device)
        ones = torch.ones((self._sphere_centers_local.shape[0], 1), device=self.device)
        self._sphere_centers_local_h = torch.cat([self._sphere_centers_local, ones], dim=1)
        self._sphere_radii = torch.tensor(sphere_radii, dtype=torch.float32, device=self.device)

    def get_collision_model(self):
        model = dict()
        model["tidybot2_base_link"] = [
            ([0.17, 0.15, 0.17], 0.17),
            ([0.17, 0.15, 0.34], 0.17),
            ([0.17, -0.15, 0.17], 0.17),
            ([0.17, -0.15, 0.34], 0.17),
            ([0.17, 0.0, 0.17], 0.17),
            ([0.17, 0.0, 0.34], 0.17),
            ([-0.17, 0.15, 0.17], 0.17),
            ([-0.17, 0.15, 0.34], 0.17),
            ([-0.17, -0.15, 0.17], 0.17),
            ([-0.17, -0.15, 0.34], 0.17),
            ([-0.17, 0.0, 0.17], 0.17),
            ([-0.17, 0.0, 0.34], 0.17),
            ([0.0, 0.15, 0.17], 0.17),
            ([0.0, 0.15, 0.34], 0.17),
            ([0.0, -0.15, 0.17], 0.17),
            ([0.0, -0.15, 0.34], 0.17),
            ([-0.0335, 0.17, 0.51], 0.12),
            ([-0.0335, 0.06, 0.51], 0.12),
            ([-0.0335, -0.06, 0.51], 0.12),
            ([-0.0335, -0.17, 0.51], 0.12),
            ([-0.0335 - 0.23, 0.17, 0.51], 0.12),
            ([-0.0335 - 0.23, 0.06, 0.51], 0.12),
            ([-0.0335 - 0.23, -0.06, 0.51], 0.12),
            ([-0.0335 - 0.23, -0.17, 0.51], 0.12),
            ([-0.0335 - 0.115, 0.17, 0.51], 0.12),
            ([-0.0335 - 0.115, 0.06, 0.51], 0.12),
            ([-0.0335 - 0.115, -0.06, 0.51], 0.12),
            ([-0.0335 - 0.115, -0.17, 0.51], 0.12),
            ([-0.0335 - 0.02, 0.0, 0.65], 0.1),
            ([-0.0335 - 0.19 / 2.0, 0.0, 0.65], 0.1),
            ([-0.0335 - 0.17, 0.0, 0.65], 0.1),
            *[([-0.26, 0.0, 0.75 + 0.07 * i], 0.05) for i in range(12)],
            ([-0.2, 0.0, 0.75 + 0.07 * 11.4], 0.07),
        ]

        model["panda_link0"] = [([-0.02, 0.0, 0.05], 0.12)]
        model["panda_link1"] = [
            ([0.0, -0.08, 0.0], 0.06),
            ([0.0, -0.03, 0.0], 0.06),
            ([0.0, 0.0, -0.12], 0.06),
            ([0.0, 0.0, -0.17], 0.06),
        ]
        model["panda_link2"] = [
            ([0.0, 0.0, 0.03], 0.06),
            ([0.0, 0.0, 0.08], 0.06),
            ([0.0, -0.12, 0.0], 0.06),
            ([0.0, -0.17, 0.0], 0.06),
        ]
        model["panda_link3"] = [
            ([0.0, 0.0, -0.06], 0.05),
            ([0.0, 0.0, -0.1], 0.06),
            ([0.08, 0.06, 0.0], 0.055),
            ([0.08, 0.02, 0.0], 0.055),
        ]
        model["panda_link4"] = [
            ([0.0, 0.0, 0.02], 0.055),
            ([0.0, 0.0, 0.06], 0.055),
            ([-0.08, 0.095, 0.0], 0.06),
            ([-0.08, 0.06, 0.0], 0.055),
        ]
        model["panda_link5"] = [
            ([0.0, 0.055, 0.0], 0.06),
            ([0.0, 0.075, 0.0], 0.06),
            ([0.0, 0.0, -0.22], 0.06),
            ([0.0, 0.05, -0.18], 0.05),
            ([0.01, 0.08, -0.14], 0.025),
            ([0.01, 0.085, -0.11], 0.025),
            ([0.01, 0.09, -0.08], 0.025),
            ([0.01, 0.095, -0.05], 0.025),
            ([-0.01, 0.08, -0.14], 0.025),
            ([-0.01, 0.085, -0.11], 0.025),
            ([-0.01, 0.09, -0.08], 0.025),
            ([-0.01, 0.095, -0.05], 0.025),
        ]
        model["panda_link6"] = [
            ([0.0, 0.0, 0.0], 0.06),
            ([0.08, 0.03, 0.0], 0.06),
            ([0.08, -0.01, 0.0], 0.06),
        ]
        model["panda_link7"] = [
            ([0.0, 0.0, 0.07], 0.05),
            ([0.02, 0.04, 0.08], 0.025),
            ([0.04, 0.02, 0.08], 0.025),
            ([0.04, 0.06, 0.085], 0.02),
            ([0.06, 0.04, 0.085], 0.02),
        ]

        model["x5_base_link"] = [([0.0, 0.0, 0.03], 0.05)]
        model["link1"] = [([0.01, 0.0, 0.03], 0.052)]
        model["link2"] = [
            ([-0.05, 0.0, 0.0], 0.04),
            ([-0.05 - 0.05 * 1, 0.0, 0.0], 0.04),
            ([-0.05 - 0.05 * 2, 0.0, 0.0], 0.04),
            ([-0.05 - 0.05 * 3, 0.0, 0.0], 0.04),
            ([-0.05 - 0.051 * 4, 0.0, 0.0], 0.05),
        ]
        model["link3"] = [
            ([0.03, 0.0, -0.02], 0.05),
            ([0.06, 0.0, -0.05], 0.038),
            ([0.1, 0.0, -0.055], 0.035),
            ([0.15, 0.0, -0.055], 0.035),
            ([0.2, 0.0, -0.055], 0.035),
            ([0.25, 0.0, -0.055], 0.042),
        ]
        model["link4"] = [
            ([0.072, 0.0, 0.0], 0.04),
            ([0.068, 0.0, -0.06], 0.04),
        ]
        model["x5_camera_link"] = [
            ([0.016, 0.0, 0.0], 0.03),
            ([0.016, -0.025, 0.025], 0.03),
            ([0.016, 0.025, -0.025], 0.03),
        ]
        model["palm_lower"] = [
            ([-0.04, -0.035, 0.01], 0.035),
            ([-0.04, -0.070, 0.01], 0.035),
            ([-0.04, 0.0, 0.01], 0.035),
            ([-0.070, -0.060, 0.01], 0.032),
            ([-0.070, -0.010, 0.01], 0.032),
        ]

        for i in [1, 2, 3]:
            model[f"mcp_joint_{i}"] = [([-0.025, 0.04, 0.015], 0.025)]
            model[f"pip_{i}"] = [([0.01, 0.0, -0.01], 0.02)]
            model[f"dip_{i}"] = [([0.01, -0.035, 0.015], 0.02)]
            model[f"fingertip_{i}"] = [([0.0, -0.035, 0.015], 0.02)]

        model["pip_4"] = [([0.0, 0.0, 0.0], 0.02)]
        model["dip_4"] = [([0.001, 0.01, -0.02], 0.02)]
        model["fingertip_4"] = [
            ([0.0, -0.045, -0.01], 0.02),
            ([0.0, -0.02, -0.01], 0.02),
            ([0.0, 0.0, -0.01], 0.02),
        ]
        return model

    def torch_spheres(self, joint_angles, joint_mapping_list=None):
        joint_angles = self._prepare_joint_angles(joint_angles=joint_angles, joint_mapping_list=joint_mapping_list)
        fk = self.robot.visual_geometry_fk_batch(joint_angles)
        link_transforms_list = []
        for link_name in self._fk_links:
            geom_tf = fk[self._link_to_geometry[link_name]]
            origin_inv = self._link_visual_origin_inv[link_name].type_as(geom_tf)
            link_transforms_list.append(torch.matmul(geom_tf, origin_inv))
        link_transforms = torch.stack(link_transforms_list, dim=1)

        batch_size = joint_angles.shape[0]
        sphere_link_transforms = link_transforms[:, self._sphere_link_idxs]
        local_offsets = (
            self._sphere_centers_local_h.to(joint_angles.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .unsqueeze(-1)
        )
        centers = torch.matmul(sphere_link_transforms, local_offsets)[:, :, :3, 0]
        radii = self._sphere_radii.to(joint_angles.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return TorchSpheres(centers=centers, radii=radii)

    def filter_pointcloud_outside_spheres(
        self,
        pointclouds: torch.Tensor,
        joint_angles: torch.Tensor,
        sdf_cutoff: float = 0.02,
        joint_mapping_list=None,
        pad_value: float = torch.nan,
        max_points_per_process: int = 5000,
    ):
        spheres = self.torch_spheres(joint_angles, joint_mapping_list=joint_mapping_list)
        if spheres.centers.shape[0] != pointclouds.shape[0]:
            raise ValueError(
                f"Batch size mismatch: pointclouds batch={pointclouds.shape[0]}, "
                f"joint batch={spheres.centers.shape[0]}."
            )

        _, num_points, _ = pointclouds.shape
        if num_points <= max_points_per_process:
            sdf = spheres.sdf(pointclouds)
            outside_mask = sdf >= sdf_cutoff
            filtered = pointclouds.clone()
            filtered[~outside_mask] = pad_value
        else:
            chunks = []
            for start_idx in range(0, num_points, max_points_per_process):
                end_idx = min(start_idx + max_points_per_process, num_points)
                pointcloud_chunk = pointclouds[:, start_idx:end_idx, :]
                sdf_chunk = spheres.sdf(pointcloud_chunk)
                outside_mask_chunk = sdf_chunk >= sdf_cutoff
                filtered_chunk = pointcloud_chunk.clone()
                filtered_chunk[~outside_mask_chunk] = pad_value
                chunks.append(filtered_chunk)
            filtered = torch.cat(chunks, dim=1)

        return filtered.to(pointclouds.device, dtype=pointclouds.dtype)
