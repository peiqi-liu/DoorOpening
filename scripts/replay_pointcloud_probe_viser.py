#!/usr/bin/env python3
"""Replay a --probe-pointcloud-camera payload from scripts/rl_games/play.py in Viser.

Overlays, per recorded frame and in the same env-relative world frame:

    sim_roundtrip (orange) -- multi_pcd_dagger's simulated depth round-trip (the training renderer).
    ground_truth  (grey)   -- the dense door + robot mesh cloud fed into the simulated renderer.

The IsaacLab RGB frame is shown both as a GUI panel and as a camera frustum placed at the physical
camera pose, so you can eyeball the simulated point cloud against the reference image.

Usage (SSH: forward the port):  python scripts/replay_pointcloud_probe_viser.py pointcloud_probe.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_STREAMS = [
    {"name": "sim_roundtrip", "label": "Sim roundtrip (dagger)", "color": [255, 140, 0]},
    {"name": "ground_truth", "label": "Ground-truth mesh", "color": [120, 120, 120]},
]
# Cyclic axis permutation (x->y, y->z, z->x) as a wxyz quaternion. Converts our "x-forward" optical
# orientation into viser's camera-frustum convention (forward +Z, right +X, down +Y).
_XFWD_TO_VISER_FRUSTUM = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recording", type=Path, help="Path to a --probe-pointcloud-camera .pt file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the Viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the Viser server.")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (defaults to recorded frame_dt).")
    parser.add_argument("--point-size", type=float, default=0.006, help="Rendered point size.")
    parser.add_argument(
        "--hide-clouds", nargs="*", default=[], help="Stream names to load initially hidden."
    )
    parser.add_argument("--start-paused", action="store_true", help="Load without auto-playing.")
    parser.add_argument("--no-grid", action="store_true", help="Hide the ground grid.")
    parser.add_argument("--no-rgb-frustum", action="store_true", help="Do not draw the RGB camera frustum.")
    parser.add_argument("--frustum-scale", type=float, default=0.25, help="RGB camera frustum size.")
    return parser.parse_args()


def _to_numpy_points(value: object) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        points = value.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        points = np.asarray(value, dtype=np.float32)
    points = points.reshape(-1, 3)
    return points[np.isfinite(points).all(axis=-1)]


def _to_numpy_vector(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        vector = value.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        vector = np.asarray(value, dtype=np.float32)
    vector = vector.reshape(-1)
    return vector if vector.size and np.isfinite(vector).all() else None


def _to_numpy_rgb(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        rgb = value.detach().cpu().numpy()
    else:
        rgb = np.asarray(value)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return None
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def _frustum_pose(frame: dict) -> tuple[np.ndarray, np.ndarray] | None:
    pos = _to_numpy_vector(frame.get("camera_pos_w"))
    quat_xyzw = _to_numpy_vector(frame.get("camera_quat_xyzw"))
    if pos is None or pos.size != 3 or quat_xyzw is None or quat_xyzw.size != 4:
        return None
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return pos, _quat_mul_wxyz(quat_wxyz, _XFWD_TO_VISER_FRUSTUM)


def _stream_points(frame: dict, name: str) -> np.ndarray:
    return _to_numpy_points(frame.get(f"{name}_points_world"))


def main() -> None:
    args = _parse_args()
    recording = args.recording.expanduser().resolve()
    if not recording.exists():
        raise SystemExit(f"Replay file does not exist: {recording}")

    try:
        import viser
    except ImportError as exc:
        raise SystemExit("This script requires `viser` (pip install viser).") from exc

    payload = torch.load(recording, map_location="cpu", weights_only=False)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not frames:
        raise SystemExit(f"No frames in {recording}")

    streams = payload.get("pointcloud_streams") or DEFAULT_STREAMS
    hidden = set(args.hide_clouds)

    spec = payload.get("camera_spec", {})
    intr = _to_numpy_vector(spec.get("intrinsics"))
    frustum_fov, frustum_aspect = math.radians(55.0), 4.0 / 3.0
    if intr is not None and intr.size == 9:
        K = intr.reshape(3, 3)
        H, W = int(spec.get("H", 240)), int(spec.get("W", 320))
        frustum_fov = 2.0 * math.atan((H * 0.5) / float(K[1, 1]))
        frustum_aspect = float(W) / float(H)

    frame_dt = float(payload.get("frame_dt") or 0.025)
    initial_fps = float(args.fps) if args.fps is not None else (1.0 / frame_dt if frame_dt > 0 else 40.0)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world", show_axes=True, axes_length=0.2, axes_radius=0.01)
    if not args.no_grid:
        server.scene.add_grid("/grid", width=6, height=6, shadow_opacity=0.1)

    # Center the camera on the first non-empty cloud.
    center = None
    for frame in frames:
        for stream in streams:
            pts = _stream_points(frame, stream["name"])
            if pts.shape[0] > 0:
                center = pts.mean(axis=0)
                break
        if center is not None:
            break
    if center is not None:
        server.initial_camera.look_at = tuple(float(x) for x in center)
        server.initial_camera.position = tuple(float(x) for x in center + np.array([1.5, -1.5, 1.0], np.float32))

    cloud_handles = {}
    for stream in streams:
        color = tuple(int(c) for c in stream.get("color", (200, 200, 200)))
        cloud_handles[stream["name"]] = server.scene.add_point_cloud(
            f"/{stream['name']}",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=color,
            point_size=args.point_size,
        )

    frustum_handle = None
    rgb_panel = None
    has_rgb = any(_to_numpy_rgb(frame.get("rgb")) is not None for frame in frames)

    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("Play", initial_value=not args.start_paused)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        fps = server.gui.add_slider("FPS", min=0.25, max=max(60.0, initial_fps * 2), step=0.25, initial_value=initial_fps)
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(frames) - 1, step=1, initial_value=0)
        prev_button = server.gui.add_button("Prev")
        next_button = server.gui.add_button("Next")

    show_clouds = {}
    with server.gui.add_folder("Pointclouds"):
        for stream in streams:
            show_clouds[stream["name"]] = server.gui.add_checkbox(
                f"Show {stream.get('label', stream['name'])}",
                initial_value=stream["name"] not in hidden,
            )
        show_frustum = None
        if has_rgb and not args.no_rgb_frustum:
            show_frustum = server.gui.add_checkbox("Show RGB frustum", initial_value=True)

    if has_rgb:
        with server.gui.add_folder("IsaacLab RGB"):
            first_rgb = next(_to_numpy_rgb(f.get("rgb")) for f in frames if _to_numpy_rgb(f.get("rgb")) is not None)
            rgb_panel = server.gui.add_image(first_rgb, label="IsaacLab camera")
        if not args.no_rgb_frustum:
            frustum_handle = server.scene.add_camera_frustum(
                "/rgb_frustum",
                fov=frustum_fov,
                aspect=frustum_aspect,
                scale=args.frustum_scale,
                image=first_rgb,
            )

    def _apply_frame(idx: int) -> None:
        frame = frames[idx]
        for stream in streams:
            name = stream["name"]
            pts = _stream_points(frame, name) if show_clouds[name].value else np.zeros((0, 3), dtype=np.float32)
            cloud_handles[name].points = pts
        rgb = _to_numpy_rgb(frame.get("rgb"))
        if rgb_panel is not None and rgb is not None:
            rgb_panel.image = rgb
        if frustum_handle is not None:
            visible = show_frustum is None or bool(show_frustum.value)
            frustum_handle.visible = visible
            if visible:
                if rgb is not None:
                    frustum_handle.image = rgb
                pose = _frustum_pose(frame)
                if pose is not None:
                    frustum_handle.position = pose[0]
                    frustum_handle.wxyz = pose[1]

    @frame_slider.on_update
    def _(_event):
        _apply_frame(int(frame_slider.value))

    @prev_button.on_click
    def _(_event):
        frame_slider.value = max(0, int(frame_slider.value) - 1)

    @next_button.on_click
    def _(_event):
        frame_slider.value = min(len(frames) - 1, int(frame_slider.value) + 1)

    for control in list(show_clouds.values()) + ([show_frustum] if show_frustum is not None else []):
        @control.on_update
        def _(_event):
            _apply_frame(int(frame_slider.value))

    _apply_frame(0)
    print(f"Loaded {len(frames)} frames from {recording}")
    print("Streams: " + ", ".join(s["name"] for s in streams))
    print(f"Playback FPS: {initial_fps:.2f}")
    if args.host == "0.0.0.0":
        print(f"For SSH use: ssh -L {args.port}:127.0.0.1:{args.port} <remote-host>")

    last_tick = time.perf_counter()
    while True:
        time.sleep(0.01)
        if not play.value:
            last_tick = time.perf_counter()
            continue
        interval = 1.0 / max(float(fps.value), 1e-6)
        now = time.perf_counter()
        if now - last_tick < interval:
            continue
        steps = max(1, int((now - last_tick) / interval))
        last_tick += steps * interval
        nxt = int(frame_slider.value) + steps
        if nxt >= len(frames):
            nxt = (nxt % len(frames)) if loop.value else len(frames) - 1
            if not loop.value:
                play.value = False
        frame_slider.value = nxt


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping replay.")
