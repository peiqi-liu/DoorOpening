"""Same-frame visualization probe for the on-robot depth camera model.

This wraps a *running* multi-door play environment and, for one tracked env, produces at each call:

    ground_truth : the dense door(link_1/link_2) + robot mesh cloud placed by the live body poses,
                   i.e. the geometry the distillation renderer takes as input.
    sim_roundtrip: that geometry pushed through multi_pcd_dagger's exact depth-camera model
                   (``render_depth_roundtrip_from_pose`` at the D435 spec + the x5 mount offset).
    rgb          : the IsaacLab GPU-rendered RGB frame at the same camera pose -- a *reference image*
                   to eyeball the simulated point cloud against (NOT back-projected into a cloud).

The point cloud is produced purely by the training-time renderer; the IsaacLab camera contributes
only the RGB image. All clouds and the camera pose are returned in the tracked env's *env-relative*
world frame (origin subtracted) so a viser replay lands near the world origin, exactly like
scripts/replay_viser_pt.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import quat_mul

from DoorOpening.assets.door.multi_door_cfg import asset_paths as door_asset_paths
from DoorOpening.assets.glorbot.glorbot_cfg import glorbot_urdf_path
from DoorOpening.utils.camera_utils import build_realsense_sampler_spec, render_depth_roundtrip_from_pose
from DoorOpening.utils.extract_pointcloud_from_articulation import (
    FrankaGripperSampler,
    build_first_visual_link_pointcloud_cache,
    compose_cached_link_pointcloud_world,
)


class PointcloudCameraProbe:
    """Build once, then :meth:`capture` per env step to record the overlay frames."""

    DOOR_LINK_NAMES = ("link_1", "link_2")

    def __init__(
        self,
        play_env,
        door_num_points: int = 4096,
        robot_num_points: int = 16384,
        render_num_points: int | None = None,
        inflate_px: int = 0,
        occluder_inflate_px: int = 0,
        camera_body_name: str = "x5_camera_link",
    ):
        self.env = play_env
        self.device = play_env.device
        self.door_num_points = int(door_num_points)
        # The robot mesh (Franka arm + base + gripper + x5 arm) covers far more surface than the
        # door panel+handle, so it needs more points than the door to reach a comparable per-area
        # density -- otherwise background points leak through the sparse robot in the depth z-buffer.
        self.robot_num_points = int(robot_num_points)
        # Keep every rendered point by default (the round-trip yields <= door+robot valid pixels).
        self.render_num_points = int(render_num_points) if render_num_points else (self.door_num_points + self.robot_num_points)
        self.inflate_px = int(inflate_px)
        self.occluder_inflate_px = int(occluder_inflate_px)

        camera = getattr(play_env, "pointcloud_camera", None)
        if camera is None:
            raise AttributeError(
                "PointcloudCameraProbe requires env.pointcloud_camera; set "
                "env_cfg.pointcloud_render_mode='depth' before creating the env."
            )
        self.camera = camera

        # --- Per-asset door mesh caches (moving panel + handle only, matching the renderer input) ---
        self.env_asset_idx = play_env.env_asset_indices.to(device=self.device, dtype=torch.long)
        unique_asset_idx = sorted(set(self.env_asset_idx.detach().cpu().tolist()))
        self.door_link_pointclouds = {}
        for idx in unique_asset_idx:
            sampler = FrankaGripperSampler(door_asset_paths[idx], device=self.device, num_points=self.door_num_points)
            self.door_link_pointclouds[idx] = build_first_visual_link_pointcloud_cache(
                sampler, link_names=self.DOOR_LINK_NAMES, device=self.device
            )
        self.door_link_body_indices = {
            link_name: int(play_env._door_body_idx[play_env.door_body_names.index(link_name)])
            for link_name in self.DOOR_LINK_NAMES
        }

        # --- Robot mesh cache (all links, placed by live body poses -- no joint remap needed) ---
        robot_sampler = FrankaGripperSampler(glorbot_urdf_path, device=self.device, num_points=self.robot_num_points)
        self.robot_link_pointclouds = build_first_visual_link_pointcloud_cache(robot_sampler, device=self.device)
        self.robot_sampler_body_indices = {}
        for link_name in self.robot_link_pointclouds.keys():
            body_ids = play_env.robot.find_bodies(link_name)[0]
            if len(body_ids) == 0:
                continue
            self.robot_sampler_body_indices[link_name] = int(body_ids[0])

        # --- Simulated D435 camera spec (matches the IsaacLab camera we drive below) + mount offset ---
        self.camera_body_idx = int(play_env.robot.find_bodies(camera_body_name)[0][0])
        self.sampler_camera_spec = build_realsense_sampler_spec(
            int(self.camera.cfg.height), int(self.camera.cfg.width), device=self.device
        )
        self.realized_intrinsics = None  # set by configure_isaac_camera_intrinsics()
        # The optical frame = x5_camera_link composed with the CameraCfg mount offset (-45deg bracket roll).
        self._mount_offset_quat_wxyz = torch.tensor(
            tuple(play_env.cfg.pointcloud_camera_cfg.offset.rot), device=self.device, dtype=torch.float32
        )

    # ------------------------------------------------------------------------------------------
    def configure_isaac_camera_intrinsics(self):
        """Drive the live IsaacLab camera toward the simulated D435 intrinsics (fx, fy, cx, cy).

        The env cfg already sets the resolution + clipping range; this pins the FOV. NOTE: Omniverse
        cameras only support square pixels, so ``set_intrinsic_matrices`` averages fx and fy -- the
        D435 model is fx!=fy (80 deg x 55 deg). The realized (rendered) intrinsics are read back and
        reported so the FOV gap vs the training spec is explicit; ``self.realized_intrinsics`` holds
        the actual K used by the GPU render.
        """
        num_cams = int(self.camera.data.intrinsic_matrices.shape[0])
        K = self.sampler_camera_spec["intrinsics"].to(dtype=torch.float32)
        self.camera.set_intrinsic_matrices(K.unsqueeze(0).repeat(num_cams, 1, 1))
        self.realized_intrinsics = self.camera.data.intrinsic_matrices[0].detach().cpu().clone()

        H, W = int(self.sampler_camera_spec["H"]), int(self.sampler_camera_spec["W"])
        nom_fx, nom_fy = float(K[0, 0]), float(K[1, 1])
        rl_fx, rl_fy = float(self.realized_intrinsics[0, 0]), float(self.realized_intrinsics[1, 1])

        def _fov(f, n):
            return math.degrees(2.0 * math.atan((n * 0.5) / f))

        print(
            "[PROBE] D435 spec fx=%.1f fy=%.1f (FOV %.1f x %.1f deg) -> Omniverse realized "
            "fx=%.1f fy=%.1f (FOV %.1f x %.1f deg). Square-pixel averaging: %s"
            % (
                nom_fx, nom_fy, _fov(nom_fx, W), _fov(nom_fy, H),
                rl_fx, rl_fy, _fov(rl_fx, W), _fov(rl_fy, H),
                "YES (fx!=fy collapsed)" if abs(nom_fx - nom_fy) > 1.0 else "no",
            )
        )

    # ------------------------------------------------------------------------------------------
    def _ground_truth_cloud_world(self, env_id: int) -> torch.Tensor:
        sl = slice(env_id, env_id + 1)
        asset_idx = int(self.env_asset_idx[env_id].item())
        door_cloud = compose_cached_link_pointcloud_world(
            link_points_by_name=self.door_link_pointclouds[asset_idx],
            link_pos_w_by_name={
                link_name: self.env.door.data.body_pos_w[sl, body_idx]
                for link_name, body_idx in self.door_link_body_indices.items()
            },
            link_quat_w_by_name={
                link_name: self.env.door.data.body_quat_w[sl, body_idx]
                for link_name, body_idx in self.door_link_body_indices.items()
            },
            num_points=self.door_num_points,
        )
        robot_cloud = compose_cached_link_pointcloud_world(
            link_points_by_name=self.robot_link_pointclouds,
            link_pos_w_by_name={
                link_name: self.env.robot.data.body_pos_w[sl, body_idx]
                for link_name, body_idx in self.robot_sampler_body_indices.items()
            },
            link_quat_w_by_name={
                link_name: self.env.robot.data.body_quat_w[sl, body_idx]
                for link_name, body_idx in self.robot_sampler_body_indices.items()
            },
            num_points=self.robot_num_points,
        )
        return door_cloud, robot_cloud  # each (1, N, 3) world frame

    def _camera_pose_xyzw(self, env_id: int) -> torch.Tensor:
        sl = slice(env_id, env_id + 1)
        link_pos_w = self.env.robot.data.body_pos_w[sl, self.camera_body_idx]  # (1, 3)
        link_quat_w = self.env.robot.data.body_quat_w[sl, self.camera_body_idx]  # (1, 4) wxyz
        cam_quat_wxyz = quat_mul(link_quat_w, self._mount_offset_quat_wxyz.unsqueeze(0))
        cam_quat_xyzw = cam_quat_wxyz[:, [1, 2, 3, 0]]
        return torch.cat([link_pos_w, cam_quat_xyzw], dim=-1)  # (1, 7)

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def capture(self, env_id: int) -> dict:
        """Return one comparison frame (env-relative world coords) for ``env_id``."""
        env_origin = self.env.scene.env_origins[env_id].detach()  # (3,)

        door_cloud, robot_cloud = self._ground_truth_cloud_world(env_id)
        gt_cloud = torch.cat([door_cloud, robot_cloud], dim=1)  # (1, N, 3)
        camera_pose = self._camera_pose_xyzw(env_id)  # (1, 7)

        sim_cloud, _ = render_depth_roundtrip_from_pose(
            pcd=gt_cloud,
            camera_pose=camera_pose,
            num_points=self.render_num_points,
            cam_spec_dict=self.sampler_camera_spec,
            inflate_px=self.inflate_px,
            occluder_pcd=door_cloud if self.occluder_inflate_px > 0 else None,
            occluder_inflate_px=self.occluder_inflate_px,
            use_compile=False,
        )  # (1, render_num_points, 3), NaN-padded

        frame = {
            "ground_truth_points_world": (gt_cloud[0] - env_origin).detach().cpu(),
            "sim_roundtrip_points_world": (sim_cloud[0] - env_origin).detach().cpu(),
            "camera_pos_w": (camera_pose[0, :3] - env_origin).detach().cpu(),
            "camera_quat_xyzw": camera_pose[0, 3:7].detach().cpu(),
        }

        # The IsaacLab camera is used ONLY for the reference RGB image; the pointcloud comes purely
        # from multi_pcd_dagger's renderer (render_depth_roundtrip_from_pose above).
        rgb = self.camera.data.output.get("rgb")
        if rgb is not None:
            rgb_i = rgb[env_id].detach().cpu()
            if rgb_i.dtype != torch.uint8:
                rgb_i = rgb_i.clamp(0, 255).to(torch.uint8)
            frame["rgb"] = rgb_i[..., :3].contiguous()

        return frame

    def payload(self, frames: list, frame_dt: float) -> dict:
        spec = self.sampler_camera_spec
        return {
            "frames": frames,
            "frame_dt": float(frame_dt),
            "camera_spec": {
                "H": int(spec["H"]),
                "W": int(spec["W"]),
                "intrinsics": spec["intrinsics"].detach().cpu(),
                "realized_intrinsics": (
                    self.realized_intrinsics if self.realized_intrinsics is not None else spec["intrinsics"].detach().cpu()
                ),
                "near_m": float(spec["near_m"]),
                "far_m": float(spec["far_m"]),
            },
            "pointcloud_streams": [
                {"name": "sim_roundtrip", "label": "Sim roundtrip (dagger)", "color": [255, 140, 0]},
                {"name": "ground_truth", "label": "Ground-truth mesh", "color": [120, 120, 120]},
            ],
        }
