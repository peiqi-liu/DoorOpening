#!/usr/bin/env python3
"""Replay a DoorOpening raw Viser .pt payload in a live Viser server."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_CLOUD_COLORS = {
    "ground_truth": (120, 120, 120),
    "robot_lidar_obs": (255, 140, 0),
    "robot_depth_cam_obs": (79, 195, 247),
    "robot_obs": (79, 195, 247),
    "policy_input": (0, 170, 120),
    "robot": (79, 195, 247),
    "door": (255, 193, 7),
}
PREFERRED_STREAM_ORDER = (
    "ground_truth",
    "robot_lidar_obs",
    "robot_depth_cam_obs",
    "policy_input",
    "robot_obs",
    "robot",
    "door",
)

# DoorOpening/scripts/ -> DoorOpening/, where the glorbot assets live.
DEFAULT_URDF = (
    Path(__file__).resolve().parents[1] / "source" / "DoorOpening" / "assets" / "glorbot" / "glorbot.urdf"
)

# Joint-state streams in a compact payload, in display order. The first source
# present in the frames becomes the default for the robot viewer.
COMPACT_JOINT_SOURCES = (
    ("compact_q", "Measured (compact_q)"),
    ("compact_target", "Target (compact_target)"),
    ("policy_action", "Policy Action"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a DoorOpening raw .pt Viser dump in a browser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recording", type=Path, help="Path to a raw .pt replay file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the Viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the Viser server.")
    parser.add_argument(
        "--clouds",
        nargs="*",
        default=None,
        help="Optional point-cloud stream names to show. Defaults to every stream in the payload.",
    )
    parser.add_argument(
        "--hide-clouds",
        nargs="*",
        default=[],
        help="Point-cloud stream names to load initially hidden.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback speed in frames per second. Defaults to recorded metadata when available.",
    )
    parser.add_argument("--point-size", type=float, default=0.004, help="Rendered point size.")
    parser.add_argument("--start-paused", action="store_true", help="Load the replay without auto-playing.")
    parser.add_argument("--no-grid", action="store_true", help="Hide the ground grid.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="URDF used to render the robot meshes when the payload carries compact joint state.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Skip rendering the URDF robot even when compact joint state is present.",
    )
    return parser.parse_args()


def _to_numpy_points(value: object) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("points")
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        points = value.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        points = np.asarray(value, dtype=np.float32)
    points = points.reshape(-1, 3)
    finite = np.isfinite(points).all(axis=-1)
    return points[finite]


def _to_numpy_vector(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        vector = value.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        vector = np.asarray(value, dtype=np.float32)
    vector = vector.reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    return vector


def _quat_rotate_points(quat_wxyz: np.ndarray, points: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    quat_norm = np.linalg.norm(quat)
    if quat_norm <= 0.0:
        return points
    quat = quat / quat_norm
    w = quat[0]
    q_xyz = quat[1:]
    uv = np.cross(q_xyz[None, :], points)
    uuv = np.cross(q_xyz[None, :], uv)
    return points + 2.0 * (w * uv + uuv)


def _aux_prediction_world_points(
    frame: dict, base_pose: tuple[np.ndarray, np.ndarray] | None = None
) -> np.ndarray:
    aux_prediction = _to_numpy_vector(frame.get("aux_prediction"))
    if aux_prediction is None or aux_prediction.size != 3:
        return np.zeros((0, 3), dtype=np.float32)

    if base_pose is not None:
        base_pos, base_quat = base_pose
    else:
        base_pos = _to_numpy_vector(frame.get("robot_base_pos_w"))
        base_quat = _to_numpy_vector(frame.get("robot_base_quat_w"))
        if base_pos is None or base_pos.size != 3:
            return np.zeros((0, 3), dtype=np.float32)
        if base_quat is None or base_quat.size != 4:
            return np.zeros((0, 3), dtype=np.float32)

    aux_base_points = aux_prediction.reshape(1, 3)
    aux_world_points = _quat_rotate_points(base_quat, aux_base_points) + base_pos.reshape(1, 3)
    return aux_world_points.astype(np.float32, copy=False)


def _identity_quat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _base_compact_layout(payload: dict) -> dict[str, int] | None:
    """Compact-vector indices of the base pseudo-joints, or ``None`` if absent.

    The deploy recorder stores the base as three name-less entries in
    ``compact_target_joint_names`` ordered ``[yaw, x, y]`` (see
    ``door_policy_node._current_base_state_compact``). Reconstructing the base
    world pose from these is robust to the recorder's historical
    ``robot_base_pos_w``/``quat`` index scramble.
    """
    names = payload.get("compact_target_joint_names")
    if not isinstance(names, (list, tuple)):
        return None
    base_indices = [idx for idx, name in enumerate(names) if name is None]
    if len(base_indices) != 3 or base_indices != list(range(3)):
        return None
    return {"yaw": base_indices[0], "x": base_indices[1], "y": base_indices[2]}


def _compact_base_pose(
    compact_vec: np.ndarray | None, base_layout: dict[str, int] | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """World pose (xyz, wxyz) from compact base dims, or ``None`` if unavailable."""
    if compact_vec is None or base_layout is None:
        return None
    if compact_vec.size <= max(base_layout.values()):
        return None
    pos = np.array(
        [float(compact_vec[base_layout["x"]]), float(compact_vec[base_layout["y"]]), 0.0],
        dtype=np.float32,
    )
    return pos, _quat_from_yaw(compact_vec[base_layout["yaw"]])


def _robot_base_pose(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """World pose (position xyz, quaternion wxyz) of the robot base from recorded fields."""
    pos = _to_numpy_vector(frame.get("robot_base_pos_w"))
    if pos is None or pos.size != 3:
        pos = np.zeros(3, dtype=np.float32)
    quat = _to_numpy_vector(frame.get("robot_base_quat_w"))
    if quat is None or quat.size != 4:
        quat = _identity_quat()
    return pos.astype(np.float32, copy=False), quat.astype(np.float32, copy=False)


def _measured_base_pose(frame: dict, base_layout: dict[str, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """Measured base world pose, preferring the reliable compact ``compact_q`` dims."""
    pose = _compact_base_pose(_to_numpy_vector(frame.get("compact_q")), base_layout)
    if pose is not None:
        return pose
    return _robot_base_pose(frame)


def _available_joint_sources(frames: list[dict]) -> list[tuple[str, str]]:
    """Compact joint-state streams that are actually populated in the frames."""
    sources = []
    for key, label in COMPACT_JOINT_SOURCES:
        if any(_to_numpy_vector(frame.get(key)) is not None for frame in frames):
            sources.append((key, label))
    return sources


def _build_joint_index_pairs(payload: dict, actuated_names: list[str]) -> list[tuple[int, int]]:
    """Map compact-vector indices to actuated-joint indices via joint names.

    Compact base dims (name ``None``) are skipped; the base is placed in the world
    via :func:`_robot_base_pose` instead of through the prismatic/revolute base joints.
    """
    compact_names = payload.get("compact_target_joint_names")
    if not isinstance(compact_names, (list, tuple)):
        return []
    actuated_index = {name: idx for idx, name in enumerate(actuated_names)}
    pairs: list[tuple[int, int]] = []
    for compact_idx, joint_name in enumerate(compact_names):
        if joint_name is None:
            continue
        actuated_idx = actuated_index.get(str(joint_name))
        if actuated_idx is not None:
            pairs.append((compact_idx, actuated_idx))
    return pairs


def _compact_to_cfg(compact_vec: np.ndarray | None, pairs: list[tuple[int, int]], n_actuated: int) -> np.ndarray:
    cfg = np.zeros(n_actuated, dtype=np.float32)
    if compact_vec is None:
        return cfg
    for compact_idx, actuated_idx in pairs:
        if compact_idx < compact_vec.size:
            cfg[actuated_idx] = compact_vec[compact_idx]
    return cfg


def _cloud_key_to_name(key: str) -> str | None:
    suffix = "_points_world"
    if not key.endswith(suffix):
        return None
    return key[: -len(suffix)]


def _coerce_color(value: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    try:
        color = tuple(int(x) for x in value)
    except TypeError:
        return default
    return color if len(color) == 3 else default


def _stream_label(name: str) -> str:
    labels = {
        "ground_truth": "GT",
        "robot_lidar_obs": "Robot Lidar Obs",
        "robot_depth_cam_obs": "Robot Depth Cam Obs",
        "robot_obs": "Robot Obs",
        "policy_input": "Policy Input",
        "robot": "Robot",
        "door": "Door",
    }
    return labels.get(name, name.replace("_", " ").title())


def _add_stream(streams: dict[str, dict], name: str, raw_spec: object | None = None) -> None:
    if not name:
        return
    spec = dict(raw_spec) if isinstance(raw_spec, dict) else {}
    default_color = DEFAULT_CLOUD_COLORS.get(name, (200, 200, 200))
    streams[name] = {
        "name": name,
        "label": str(spec.get("label", _stream_label(name))),
        "key": str(spec.get("key", f"{name}_points_world")),
        "color": _coerce_color(spec.get("color", spec.get("colors")), default_color),
        "point_size_scale": float(spec.get("point_size_scale", 1.0)),
    }


def _discover_streams(payload: dict, frames: list[dict]) -> list[dict]:
    streams: dict[str, dict] = {}
    payload_streams = payload.get("pointcloud_streams", [])
    if isinstance(payload_streams, dict):
        payload_streams = list(payload_streams.values())
    if isinstance(payload_streams, list):
        for raw_spec in payload_streams:
            if isinstance(raw_spec, str):
                _add_stream(streams, raw_spec)
            elif isinstance(raw_spec, dict):
                _add_stream(streams, str(raw_spec.get("name", "")), raw_spec)

    for frame in frames:
        pointclouds = frame.get("pointclouds")
        if isinstance(pointclouds, dict):
            for name in pointclouds:
                if name not in streams:
                    _add_stream(streams, str(name))
        for key in frame:
            name = _cloud_key_to_name(str(key))
            if name is not None and name not in streams:
                _add_stream(streams, name, {"key": key})

    ordered_names = [name for name in PREFERRED_STREAM_ORDER if name in streams]
    ordered_names.extend(name for name in streams if name not in ordered_names)
    return [streams[name] for name in ordered_names]


def _frame_cloud_points(frame: dict, stream: dict) -> np.ndarray:
    name = stream["name"]
    pointclouds = frame.get("pointclouds")
    if isinstance(pointclouds, dict) and name in pointclouds:
        return _to_numpy_points(pointclouds[name])
    for key in (stream.get("key"), f"{name}_points_world"):
        if key and key in frame:
            return _to_numpy_points(frame.get(key))
    return np.zeros((0, 3), dtype=np.float32)


def _first_nonempty_cloud(frames: list[dict], streams: list[dict]) -> np.ndarray:
    for frame in frames:
        for stream in streams:
            pts = _frame_cloud_points(frame, stream)
            if pts.shape[0] > 0:
                return pts
    return np.zeros((0, 3), dtype=np.float32)


def _aux_input_world_points(frame: dict) -> np.ndarray:
    aux_input = _to_numpy_vector(frame.get("aux_input"))
    if aux_input is None or aux_input.size != 3:
        return np.zeros((0, 3), dtype=np.float32)

    robot_base_pos_w = _to_numpy_vector(frame.get("robot_base_pos_w"))
    robot_base_quat_w = _to_numpy_vector(frame.get("robot_base_quat_w"))
    if robot_base_pos_w is None or robot_base_pos_w.size != 3:
        return np.zeros((0, 3), dtype=np.float32)
    if robot_base_quat_w is None or robot_base_quat_w.size != 4:
        return np.zeros((0, 3), dtype=np.float32)

    aux_base_points = aux_input.reshape(1, 3)
    aux_world_points = _quat_rotate_points(robot_base_quat_w, aux_base_points) + robot_base_pos_w.reshape(1, 3)
    return aux_world_points.astype(np.float32, copy=False)


def _has_aux_prediction(frames: list[dict]) -> bool:
    return any(_to_numpy_vector(frame.get("aux_prediction")) is not None for frame in frames)


def _has_aux_input(frames: list[dict]) -> bool:
    return any(_to_numpy_vector(frame.get("aux_input")) is not None for frame in frames)


def _door_joint_prediction(frame: dict) -> np.ndarray | None:
    return _to_numpy_vector(frame.get("door_joint_prediction"))


def _has_door_joint_prediction(frames: list[dict]) -> bool:
    return any(_door_joint_prediction(frame) is not None for frame in frames)


def _positive_float(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _infer_payload_fps(payload: dict) -> tuple[float, str]:
    for key in ("frame_fps", "env_step_fps", "replay_fps"):
        value = _positive_float(payload.get(key))
        if value is not None:
            return value, key

    for key in ("frame_dt", "env_step_dt", "replay_frame_dt"):
        value = _positive_float(payload.get(key))
        if value is not None:
            return 1.0 / value, key

    sim_dt = _positive_float(payload.get("sim_dt"))
    if sim_dt is not None:
        decimation = _positive_float(payload.get("decimation")) or 1.0
        return 1.0 / (sim_dt * decimation), "sim_dt"

    # Legacy DoorOpening payloads predate timing metadata; the control step is 40 Hz.
    return 40.0, "legacy_dooropening_default"


def _format_fps_source_label(source: str) -> str:
    if source == "legacy_dooropening_default":
        return "legacy DoorOpening default (40 Hz)"
    return f"payload {source}"


def main() -> None:
    args = _parse_args()
    recording = args.recording.expanduser().resolve()
    if not recording.exists():
        raise SystemExit(f"Replay file does not exist: {recording}")

    try:
        import viser
    except ImportError as exc:
        raise SystemExit("This script requires `viser`. Install it first with `pip install viser`.") from exc

    payload = torch.load(recording, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "frames" not in payload:
        raise SystemExit(f"Unexpected replay payload format in {recording}")

    frames = payload["frames"]
    if not isinstance(frames, list) or len(frames) == 0:
        raise SystemExit(f"Replay payload has no frames: {recording}")

    streams = _discover_streams(payload, frames)
    if args.clouds is not None:
        requested_clouds = set(args.clouds)
        streams = [stream for stream in streams if stream["name"] in requested_clouds]
    hidden_clouds = set(args.hide_clouds)
    has_aux_prediction = _has_aux_prediction(frames)
    has_aux_input = _has_aux_input(frames)
    has_door_joint_prediction = _has_door_joint_prediction(frames)
    has_compact_joint_state = any(
        _to_numpy_vector(frame.get("compact_q")) is not None
        or _to_numpy_vector(frame.get("compact_target")) is not None
        for frame in frames
    )

    joint_sources = [] if args.no_robot else _available_joint_sources(frames)
    robot_requested = bool(joint_sources) and isinstance(payload.get("compact_target_joint_names"), (list, tuple))
    base_layout = _base_compact_layout(payload)
    if not streams and not robot_requested:
        raise SystemExit("Replay payload has no point-cloud streams matching the requested filters.")

    payload_fps, payload_fps_source = _infer_payload_fps(payload)
    initial_fps = float(args.fps) if args.fps is not None else payload_fps
    fps_slider_max = max(60.0, initial_fps * 2.0)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_width="medium")
    server.scene.add_frame("/world", show_axes=True, axes_length=0.2, axes_radius=0.01)
    if not args.no_grid:
        server.scene.add_grid("/grid", width=6, height=6, position=(0.0, 0.0, 0.0), shadow_opacity=0.1)

    first_cloud = _first_nonempty_cloud(frames, streams)
    if first_cloud.shape[0] > 0:
        center = first_cloud.mean(axis=0)
    elif robot_requested:
        center = _measured_base_pose(frames[0], base_layout)[0]
    else:
        center = None
    if center is not None:
        server.initial_camera.look_at = tuple(float(x) for x in center)
        server.initial_camera.position = tuple(
            float(x) for x in (center + np.array([1.5, -1.5, 1.0], dtype=np.float32))
        )

    cloud_handles = {}
    for stream in streams:
        cloud_handles[stream["name"]] = server.scene.add_point_cloud(
            f"/{stream['name']}_points",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=stream["color"],
            point_size=args.point_size * float(stream.get("point_size_scale", 1.0)),
        )

    aux_handle = None
    if has_aux_prediction:
        aux_handle = server.scene.add_point_cloud(
            "/aux_prediction",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=(255, 140, 0),
            point_size=args.point_size * 3.0,
        )

    aux_input_handle = None
    if has_aux_input:
        aux_input_handle = server.scene.add_point_cloud(
            "/aux_input",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=(0, 200, 255),
            point_size=args.point_size * 3.0,
        )

    robot_urdf = None
    robot_frame = None
    robot_joint_pairs: list[tuple[int, int]] = []
    if robot_requested:
        urdf_path = args.urdf.expanduser().resolve()
        if not urdf_path.exists():
            print(f"Robot URDF not found at {urdf_path}; skipping robot meshes.", file=sys.stderr)
        else:
            try:
                from viser.extras import ViserUrdf

                robot_frame = server.scene.add_frame("/robot", show_axes=False)
                robot_urdf = ViserUrdf(server, urdf_path, root_node_name="/robot")
                robot_joint_pairs = _build_joint_index_pairs(payload, robot_urdf.get_actuated_joint_names())
                if not robot_joint_pairs:
                    print("No compact joint names matched the URDF; robot will stay at its default pose.")
            except Exception as exc:  # noqa: BLE001 - degrade gracefully to pointcloud-only replay.
                print(f"Failed to load robot URDF ({exc}); continuing without robot meshes.", file=sys.stderr)
                robot_urdf = None

    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("Play", initial_value=not args.start_paused)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        fps = server.gui.add_slider("FPS", min=0.25, max=fps_slider_max, step=0.25, initial_value=initial_fps)
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(frames) - 1, step=1, initial_value=0)
        prev_button = server.gui.add_button("Prev")
        next_button = server.gui.add_button("Next")

    show_clouds = {}
    with server.gui.add_folder("Pointclouds"):
        for stream in streams:
            show_clouds[stream["name"]] = server.gui.add_checkbox(
                f"Show {stream['label']}",
                initial_value=stream["name"] not in hidden_clouds,
            )
        show_aux = None
        if has_aux_prediction:
            show_aux = server.gui.add_checkbox("Show Aux Prediction (output)", initial_value=True)
        show_aux_input = None
        if has_aux_input:
            show_aux_input = server.gui.add_checkbox("Show Aux Input (to network)", initial_value=True)

    show_robot = None
    robot_source = None
    if robot_urdf is not None:
        with server.gui.add_folder("Robot"):
            show_robot = server.gui.add_checkbox("Show Robot", initial_value=True)
            robot_source = server.gui.add_dropdown(
                "Joint Source",
                options=[label for _, label in joint_sources],
                initial_value=joint_sources[0][1],
            )
    robot_label_to_key = {label: key for key, label in joint_sources}
    n_actuated = len(robot_urdf.get_actuated_joint_names()) if robot_urdf is not None else 0

    def _update_robot(frame: dict) -> None:
        if robot_urdf is None or robot_frame is None or show_robot is None:
            return
        robot_frame.visible = bool(show_robot.value)
        if not show_robot.value:
            return
        source_key = robot_label_to_key.get(robot_source.value, joint_sources[0][0]) if robot_source else joint_sources[0][0]
        compact_vec = _to_numpy_vector(frame.get(source_key))
        # Place the base from the compact base dims of the selected source (reliable,
        # and lets the Target view actually move the base); fall back to recorded fields.
        base_pose = _compact_base_pose(compact_vec, base_layout) or _robot_base_pose(frame)
        robot_frame.position = base_pose[0]
        robot_frame.wxyz = base_pose[1]
        robot_urdf.update_cfg(_compact_to_cfg(compact_vec, robot_joint_pairs, n_actuated))

    def _apply_frame(frame_idx: int) -> None:
        frame = frames[frame_idx]
        for stream in streams:
            points = _frame_cloud_points(frame, stream)
            handle = cloud_handles[stream["name"]]
            handle.points = points if show_clouds[stream["name"]].value else np.zeros((0, 3), dtype=np.float32)
        if aux_handle is not None and show_aux is not None:
            # aux_prediction is in the measured robot-base frame, so use the measured pose.
            aux_points = _aux_prediction_world_points(frame, _measured_base_pose(frame, base_layout))
            aux_handle.points = aux_points if show_aux.value else np.zeros((0, 3), dtype=np.float32)
        if aux_input_handle is not None and show_aux_input is not None:
            aux_input_points = _aux_input_world_points(frame)
            aux_input_handle.points = (
                aux_input_points if show_aux_input.value else np.zeros((0, 3), dtype=np.float32)
            )
        _update_robot(frame)
        if has_door_joint_prediction:
            door_joint_pred = _door_joint_prediction(frame)
            if door_joint_pred is not None:
                print(
                    "[frame {}] door_joint_prediction: {}".format(
                        frame_idx,
                        np.array2string(door_joint_pred, precision=4, suppress_small=True),
                    )
                )
        if has_compact_joint_state:
            # Print saved joint angles + PD targets per frame (in the payload's saved order, i.e.
            # [base_x, base_y, base_rotation, panda_1..7, finger_0..N]), like door_joint_prediction.
            compact_q_vec = _to_numpy_vector(frame.get("compact_q"))
            if compact_q_vec is not None:
                print(
                    "[frame {}] joint_angles: {}".format(
                        frame_idx, np.array2string(compact_q_vec, precision=4, suppress_small=True)
                    )
                )
            compact_target_vec = _to_numpy_vector(frame.get("compact_target"))
            if compact_target_vec is not None:
                print(
                    "[frame {}] joint_targets: {}".format(
                        frame_idx, np.array2string(compact_target_vec, precision=4, suppress_small=True)
                    )
                )

    @frame_slider.on_update
    def _(_event):
        _apply_frame(int(frame_slider.value))

    @prev_button.on_click
    def _(_event):
        frame_slider.value = max(0, int(frame_slider.value) - 1)

    @next_button.on_click
    def _(_event):
        frame_slider.value = min(len(frames) - 1, int(frame_slider.value) + 1)

    for checkbox in show_clouds.values():
        @checkbox.on_update
        def _(_event):
            _apply_frame(int(frame_slider.value))

    if show_aux is not None:
        @show_aux.on_update
        def _(_event):
            _apply_frame(int(frame_slider.value))

    if show_aux_input is not None:
        @show_aux_input.on_update
        def _(_event):
            _apply_frame(int(frame_slider.value))

    for robot_control in (show_robot, robot_source):
        if robot_control is not None:
            @robot_control.on_update
            def _(_event):
                _apply_frame(int(frame_slider.value))

    _apply_frame(0)
    print(f"Loaded {len(frames)} frames from {recording}")
    if streams:
        print("Point-cloud streams: " + ", ".join(stream["name"] for stream in streams))
    if robot_urdf is not None:
        print(
            "Robot URDF: {} ({} joints driven from {})".format(
                args.urdf,
                len(robot_joint_pairs),
                ", ".join(label for _, label in joint_sources),
            )
        )
    if has_door_joint_prediction:
        dj0 = _door_joint_prediction(frames[0])
        dj_dim = 0 if dj0 is None else int(dj0.size)
        print(f"Door joint prediction present ({dj_dim} dims) — printing per frame to this console.")
    if has_compact_joint_state:
        _cnames = payload.get("compact_target_joint_names")
        _order = (
            [str(n) if n is not None else "<base>" for n in _cnames]
            if isinstance(_cnames, (list, tuple))
            else "unknown"
        )
        print(f"Saved joint state present — printing joint_angles/joint_targets per frame. Order: {_order}")
    if args.fps is None:
        print(f"Playback FPS: {initial_fps:.2f} (from {_format_fps_source_label(payload_fps_source)})")
    else:
        print(f"Playback FPS: {initial_fps:.2f} (from --fps override)")
    frame_dt = _positive_float(payload.get("frame_dt")) or _positive_float(payload.get("env_step_dt"))
    sensor_dt = _positive_float(payload.get("pointcloud_sensor_dt"))
    if frame_dt is not None and sensor_dt is not None and sensor_dt > frame_dt * 1.5:
        print(
            "Pointcloud sensor dt is {:.4f}s ({:.2f} FPS), so clouds may repeat between replay frames.".format(
                sensor_dt,
                1.0 / sensor_dt,
            )
        )
    if args.host == "0.0.0.0":
        print(f"For SSH use: ssh -L {args.port}:127.0.0.1:{args.port} <remote-host>")

    last_tick = time.perf_counter()
    while True:
        time.sleep(0.01)
        if not play.value:
            last_tick = time.perf_counter()
            continue

        frame_interval = 1.0 / max(float(fps.value), 1e-6)
        now = time.perf_counter()
        if now - last_tick < frame_interval:
            continue

        steps = max(1, int((now - last_tick) / frame_interval))
        last_tick += steps * frame_interval
        next_frame = int(frame_slider.value) + steps
        if next_frame >= len(frames):
            if loop.value:
                next_frame %= len(frames)
            else:
                next_frame = len(frames) - 1
                play.value = False
        frame_slider.value = next_frame


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping replay.")
    except Exception as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        raise
