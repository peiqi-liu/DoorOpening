# This module was split out of multi_pcd_dagger.py to keep that file manageable.
# It defines a mixin that is composed into the Dagger class; it is not a standalone
# class and relies on attributes/methods provided by Dagger and the other mixins.
import os
from collections import OrderedDict
from pathlib import Path

import torch

from DoorOpening.assets.door.multi_door_cfg import DOOR_FAMILY_NAMES
from DoorOpening.utils.viser_pt import (
    format_iterated_record_path,
    prepare_pointcloud,
    prepare_world_points_from_local,
)


class ViserDebugMixin:
    """Viser-based point cloud debug streaming/recording helpers for Dagger."""

    def _init_viser_debug_tools(self):
        """Initialize optional raw replay capture state used for point-cloud debugging."""
        self._viser_cached_ground_truth_pcd_world = None
        self._viser_cached_sensor_obs_pcd_base = OrderedDict()
        self._viser_pending_debug_frame = None
        self._viser_raw_streams = OrderedDict()

        # Keep replay outputs next to checkpoints by default so a run's artifacts stay together.
        default_record_dir = Path(self.nn_dir).parent if self.nn_dir is not None else Path(os.getcwd())
        configured_raw_path = self.viser_raw_cfg.get("path", self.viser_raw_cfg.get("raw_path"))
        if configured_raw_path is None:
            self.viser_raw_path = str(default_record_dir / "viser_replay.pt")
        else:
            configured_raw_path = Path(str(configured_raw_path))
            if not configured_raw_path.is_absolute():
                configured_raw_path = default_record_dir / configured_raw_path
            self.viser_raw_path = str(configured_raw_path)

        sim_cfg = getattr(self.ov_env.cfg, "sim", None)
        sim_dt = getattr(sim_cfg, "dt", None)
        self.viser_sim_dt = float(sim_dt) if sim_dt is not None else 1.0 / 30.0
        self.viser_env_step_dt = max(float(getattr(self.ov_env, "dt", self.viser_sim_dt)), 1e-6)
        if not self.viser_raw_enabled:
            return
        if self.viser_raw_max_points <= 0:
            raise ValueError("viser.raw.max_points must be positive when raw Viser capture is enabled.")

        raw_dir = os.path.dirname(self.viser_raw_path)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
        self._init_viser_raw_streams()

    def _init_viser_raw_streams(self):
        """Select one random env per door family for multi-door replay export."""
        self._viser_raw_streams = OrderedDict()
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            self._viser_raw_streams[family_name] = {
                "family_id": int(family_id),
                "family_name": family_name,
                "env_id": None,
                "frames": [],
                "frame_count": 0,
                "chunk_index": 0,
                "latest_iteration": None,
                "resample_env_each_chunk": True,
            }
            self._resample_viser_raw_stream_env(self._viser_raw_streams[family_name])

        if not self._viser_raw_streams:
            self._viser_raw_streams["env"] = {
                "family_id": None,
                "family_name": "env",
                "env_id": None,
                "frames": [],
                "frame_count": 0,
                "chunk_index": 0,
                "latest_iteration": None,
                "resample_env_each_chunk": True,
            }
            self._resample_viser_raw_stream_env(self._viser_raw_streams["env"])

    def _sample_random_env_id_for_family(self, family_id=None):
        if self.num_envs <= 0:
            return 0
        if family_id is None:
            return int(torch.randint(self.num_envs, (1,), device=self.device).detach().cpu().item())
        matching_envs = self.family_env_ids.get(int(family_id))
        if matching_envs is None:
            matching_envs = torch.nonzero(self.env_family_ids == int(family_id), as_tuple=False).squeeze(-1)
        if matching_envs.numel() == 0:
            return int(torch.randint(self.num_envs, (1,), device=self.device).detach().cpu().item())
        env_offset = int(torch.randint(matching_envs.numel(), (1,), device=self.device).detach().cpu().item())
        return int(matching_envs[env_offset].detach().cpu().item())

    def _resample_viser_raw_stream_env(self, stream):
        family_id = stream.get("family_id")
        stream["env_id"] = self._sample_random_env_id_for_family(family_id=family_id)

    def _viser_stream_env_ids(self):
        return [int(stream["env_id"]) for stream in self._viser_raw_streams.values()]

    def _select_viser_ground_truth_points(self, pointcloud_world):
        if not self.viser_raw_enabled or pointcloud_world is None:
            return None
        return {
            int(env_id): prepare_pointcloud(
                pointcloud_world,
                env_id=int(env_id),
                max_points=self.viser_raw_max_points,
            )
            for env_id in self._viser_stream_env_ids()
        }

    def _viser_raw_pointcloud_stream_specs(self):
        streams = [
            {
                "name": "ground_truth",
                "label": "GT",
                "color": (120, 120, 120),
                "point_size_scale": 1.0,
            }
        ]
        if self.pointcloud_source in {"lidar", "both"}:
            streams.append(
                {
                    "name": "robot_lidar_obs",
                    "label": "Robot Lidar Obs",
                    "color": (255, 140, 0),
                    "point_size_scale": 1.0,
                }
            )
        if self.pointcloud_source in {"sampler", "depth", "both"}:
            streams.append(
                {
                    "name": "robot_depth_cam_obs",
                    "label": "Robot Depth Cam Obs",
                    "color": (79, 195, 247),
                    "point_size_scale": 1.0,
                }
            )
        streams.append(
            {
                "name": "policy_input",
                "label": "Policy Input",
                "color": (0, 170, 120),
                "point_size_scale": 1.0,
            }
        )
        return streams

    def _build_viser_raw_payload(self, stream):
        frame_dt = float(self.viser_env_step_dt * self.viser_raw_capture_interval)
        return {
            "format": "dooropening_viser_replay_v1",
            "pointcloud_frame": "world",
            "pointcloud_source": self.pointcloud_source,
            "pointcloud_streams": self._viser_raw_pointcloud_stream_specs(),
            "sim_dt": float(self.viser_sim_dt),
            "decimation": int(getattr(self.ov_env.cfg, "decimation", 1)),
            "env_step_dt": float(self.viser_env_step_dt),
            "env_step_fps": 1.0 / float(self.viser_env_step_dt),
            "frame_dt": frame_dt,
            "frame_fps": 1.0 / frame_dt,
            "pointcloud_sensor_dt": frame_dt,
            "pointcloud_sensor_fps": 1.0 / frame_dt,
            "frames": stream["frames"],
        }

    def _trim_viser_raw_frames(self, stream):
        if self.viser_raw_max_frames > 0 and len(stream["frames"]) > self.viser_raw_max_frames:
            stream["frames"] = stream["frames"][-self.viser_raw_max_frames :]
        stream["frame_count"] = len(stream["frames"])

    def _maybe_flush_viser_raw_snapshot(self, iteration):
        if not self.viser_raw_enabled or self.rank != 0:
            return
        if self.viser_raw_save_interval <= 0:
            return
        if (int(iteration) + 1) % self.viser_raw_save_interval != 0:
            return
        for stream in self._viser_raw_streams.values():
            self._flush_viser_raw_stream(
                stream,
                chunk_complete=True,
                reason=f"save_interval {self.viser_raw_save_interval} reached at iteration {int(iteration)}",
            )

    def _flush_viser_raw_recording(self, chunk_complete, reason):
        if not self.viser_raw_enabled or self.rank != 0:
            return
        for stream in self._viser_raw_streams.values():
            self._flush_viser_raw_stream(stream, chunk_complete=chunk_complete, reason=reason)

    def _flush_viser_raw_stream(self, stream, chunk_complete, reason):
        if not self.viser_raw_enabled or self.rank != 0:
            return
        if stream["frame_count"] <= 0:
            return

        latest_iteration = int(stream["latest_iteration"])
        record_tag = f"{stream['family_name']}_chunk_{stream['chunk_index']:04d}_iter_{latest_iteration}"
        payload = self._build_viser_raw_payload(stream)
        torch.save(payload, self._format_iterated_record_path(self.viser_raw_path, record_tag))

        if self.rank == 0:
            status = "complete" if chunk_complete else "partial"
            print(
                "Saved {} Viser raw chunk {} for {} env {} with {} frames ({}).".format(
                    status,
                    stream["chunk_index"],
                    stream["family_name"],
                    stream["env_id"],
                    stream["frame_count"],
                    reason,
                )
            )

        stream["frames"] = []
        stream["frame_count"] = 0
        stream["latest_iteration"] = None
        if bool(stream.get("resample_env_each_chunk", False)):
            self._resample_viser_raw_stream_env(stream)

    def _prepare_viser_world_points_from_local(
        self,
        pointcloud_local,
        base_pos_w,
        base_quat_w,
        env_id,
        max_points,
        drop_zero_rows=False,
    ):
        """Convert a base-frame point cloud for one env into world coordinates for raw replay output."""
        return prepare_world_points_from_local(
            pointcloud_local,
            base_pos_w,
            base_quat_w,
            env_id=int(env_id),
            max_points=max_points,
            drop_zero_rows=drop_zero_rows,
        )

    def _prepare_viser_points(self, pointcloud, env_id, max_points, drop_zero_rows=False):
        """Extract one env's point cloud, sanitize it, and downsample it on CPU for replay export."""
        return prepare_pointcloud(
            pointcloud,
            env_id=int(env_id),
            max_points=max_points,
            drop_zero_rows=drop_zero_rows,
        )

    def _format_iterated_record_path(self, path_str, iteration):
        """Append a record-specific suffix to replay filenames so each saved capture gets its own file."""
        return format_iterated_record_path(path_str, iteration)

    def _is_viser_capture_iteration(self, iteration):
        # Mirrors the capture gate in _maybe_update_viser_debug so callers can avoid composing a
        # ground-truth cloud on iterations that would be skipped anyway.
        interval = max(1, int(self.viser_raw_capture_interval))
        return int(iteration) % interval == 0

    def _maybe_update_viser_debug(
        self,
        iteration,
        robot_base_pos_w,
        robot_base_quat_w,
        ground_truth_pcd_world,
        sensor_obs_pcd_base_by_name,
        policy_input_pcd_base,
        aux_prediction=None,
        aux_input=None,
    ):
        """Append one replay frame for each selected multi-door family env."""
        if not self.viser_raw_enabled:
            return
        if int(iteration) % self.viser_raw_capture_interval != 0:
            # Save cadence is iteration-based, not capture-based. Still check whether an
            # existing chunk should flush on iterations where we intentionally skip capture.
            self._maybe_flush_viser_raw_snapshot(iteration)
            return

        for stream in self._viser_raw_streams.values():
            env_id = int(stream["env_id"])
            if stream["frame_count"] == 0:
                stream["chunk_index"] += 1
            stream["latest_iteration"] = int(iteration)

            selected_ground_truth = None
            if isinstance(ground_truth_pcd_world, dict):
                selected_ground_truth = ground_truth_pcd_world.get(env_id)
            else:
                selected_ground_truth = ground_truth_pcd_world

            ground_truth_points_world = self._prepare_viser_points(
                selected_ground_truth,
                env_id,
                self.viser_raw_max_points,
            ).to(dtype=torch.float16)

            pointclouds = {
                "ground_truth": ground_truth_points_world,
            }
            for name, pointcloud_base in (sensor_obs_pcd_base_by_name or {}).items():
                if pointcloud_base is None:
                    continue
                pointclouds[name] = self._prepare_viser_world_points_from_local(
                    pointcloud_base,
                    robot_base_pos_w,
                    robot_base_quat_w,
                    env_id,
                    self.viser_raw_max_points,
                    drop_zero_rows=True,
                ).to(dtype=torch.float16)

            policy_input_points_world = None
            if policy_input_pcd_base is not None:
                policy_input_points_world = self._prepare_viser_world_points_from_local(
                    policy_input_pcd_base,
                    robot_base_pos_w,
                    robot_base_quat_w,
                    env_id,
                    self.viser_raw_max_points,
                    drop_zero_rows=True,
                ).to(dtype=torch.float16)
            pointclouds["policy_input"] = policy_input_points_world

            aux_prediction_base = (
                None
                if aux_prediction is None
                else self._aux_to_2d(aux_prediction)[env_id].detach().cpu().to(dtype=torch.float32)
            )
            aux_input_base = (
                None
                if aux_input is None
                else self._aux_to_2d(aux_input)[env_id].detach().cpu().to(dtype=torch.float32)
            )
            frame_record = {
                "pointclouds": pointclouds,
            }
            if aux_prediction_base is not None or aux_input_base is not None:
                frame_record["robot_base_pos_w"] = robot_base_pos_w[env_id].detach().cpu().to(dtype=torch.float32)
                frame_record["robot_base_quat_w"] = robot_base_quat_w[env_id].detach().cpu().to(dtype=torch.float32)
            if aux_prediction_base is not None:
                frame_record["aux_prediction"] = aux_prediction_base
            if aux_input_base is not None:
                frame_record["aux_input"] = aux_input_base
            stream["frames"].append(frame_record)
            self._trim_viser_raw_frames(stream)
        self._maybe_flush_viser_raw_snapshot(iteration)

    def _close_viser_debug_tools(self):
        """Flush any in-progress raw replay chunk on shutdown."""
        self._flush_viser_raw_recording(chunk_complete=False, reason="shutdown")
