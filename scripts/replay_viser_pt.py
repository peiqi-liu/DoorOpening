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


def _aux_prediction_world_points(frame: dict) -> np.ndarray:
    aux_prediction = _to_numpy_vector(frame.get("aux_prediction"))
    if aux_prediction is None or aux_prediction.size != 3:
        return np.zeros((0, 3), dtype=np.float32)

    robot_base_pos_w = _to_numpy_vector(frame.get("robot_base_pos_w"))
    robot_base_quat_w = _to_numpy_vector(frame.get("robot_base_quat_w"))
    if robot_base_pos_w is None or robot_base_pos_w.size != 3:
        return np.zeros((0, 3), dtype=np.float32)
    if robot_base_quat_w is None or robot_base_quat_w.size != 4:
        return np.zeros((0, 3), dtype=np.float32)

    aux_base_points = aux_prediction.reshape(1, 3)
    aux_world_points = _quat_rotate_points(robot_base_quat_w, aux_base_points) + robot_base_pos_w.reshape(1, 3)
    return aux_world_points.astype(np.float32, copy=False)


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
    if not streams:
        raise SystemExit("Replay payload has no point-cloud streams matching the requested filters.")
    hidden_clouds = set(args.hide_clouds)
    has_aux_prediction = _has_aux_prediction(frames)
    has_aux_input = _has_aux_input(frames)

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
            colors=(0, 220, 180),
            point_size=args.point_size * 3.0,
        )

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

    def _apply_frame(frame_idx: int) -> None:
        frame = frames[frame_idx]
        for stream in streams:
            points = _frame_cloud_points(frame, stream)
            handle = cloud_handles[stream["name"]]
            handle.points = points if show_clouds[stream["name"]].value else np.zeros((0, 3), dtype=np.float32)
        if aux_handle is not None and show_aux is not None:
            aux_points = _aux_prediction_world_points(frame)
            aux_handle.points = aux_points if show_aux.value else np.zeros((0, 3), dtype=np.float32)
        if aux_input_handle is not None and show_aux_input is not None:
            aux_input_points = _aux_input_world_points(frame)
            aux_input_handle.points = aux_input_points if show_aux_input.value else np.zeros((0, 3), dtype=np.float32)

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

    _apply_frame(0)
    print(f"Loaded {len(frames)} frames from {recording}")
    print("Point-cloud streams: " + ", ".join(stream["name"] for stream in streams))
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
