#!/usr/bin/env python3
"""Replay a DoorOpening raw Viser .pt payload in a live Viser server."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a DoorOpening raw .pt Viser dump in a browser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recording", type=Path, help="Path to a raw .pt replay file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the Viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the Viser server.")
    parser.add_argument("--fps", type=float, default=2.0, help="Playback speed in frames per second.")
    parser.add_argument("--point-size", type=float, default=0.004, help="Rendered point size.")
    return parser.parse_args()


def _to_numpy_points(value: object) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        points = value.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        points = np.asarray(value, dtype=np.float32)
    points = points.reshape(-1, 3)
    finite = np.isfinite(points).all(axis=-1)
    return points[finite]


def _first_nonempty_cloud(frames: list[dict]) -> np.ndarray:
    for frame in frames:
        for key in ("ground_truth_points_world", "robot_obs_points_world", "policy_input_points_world"):
            pts = _to_numpy_points(frame.get(key))
            if pts.shape[0] > 0:
                return pts
    return np.zeros((0, 3), dtype=np.float32)


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

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_width="medium")
    server.scene.add_frame("/world", show_axes=True, axes_length=0.2, axes_radius=0.01)
    server.scene.add_grid("/grid", width=6, height=6, position=(0.0, 0.0, 0.0), shadow_opacity=0.1)

    first_cloud = _first_nonempty_cloud(frames)
    if first_cloud.shape[0] > 0:
        center = first_cloud.mean(axis=0)
        server.initial_camera.look_at = tuple(float(x) for x in center)
        server.initial_camera.position = tuple(
            float(x) for x in (center + np.array([1.5, -1.5, 1.0], dtype=np.float32))
        )

    gt_handle = server.scene.add_point_cloud(
        "/ground_truth_points",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=(120, 120, 120),
        point_size=args.point_size,
    )
    obs_handle = server.scene.add_point_cloud(
        "/robot_obs_points",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=(79, 195, 247),
        point_size=args.point_size,
    )
    policy_handle = server.scene.add_point_cloud(
        "/policy_input_points",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=(0, 170, 120),
        point_size=args.point_size,
    )

    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("Play", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        fps = server.gui.add_slider("FPS", min=0.25, max=30.0, step=0.25, initial_value=args.fps)
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(frames) - 1, step=1, initial_value=0)
        prev_button = server.gui.add_button("Prev")
        next_button = server.gui.add_button("Next")

    with server.gui.add_folder("Pointclouds"):
        show_gt = server.gui.add_checkbox("Show GT", initial_value=True)
        show_obs = server.gui.add_checkbox("Show Robot Obs", initial_value=True)
        show_policy = server.gui.add_checkbox("Show Policy Input", initial_value=True)

    def _apply_frame(frame_idx: int) -> None:
        frame = frames[frame_idx]
        gt_points = _to_numpy_points(frame.get("ground_truth_points_world"))
        obs_points = _to_numpy_points(frame.get("robot_obs_points_world"))
        policy_points = _to_numpy_points(frame.get("policy_input_points_world"))

        gt_handle.points = gt_points if show_gt.value else np.zeros((0, 3), dtype=np.float32)
        obs_handle.points = obs_points if show_obs.value else np.zeros((0, 3), dtype=np.float32)
        policy_handle.points = policy_points if show_policy.value else np.zeros((0, 3), dtype=np.float32)

    @frame_slider.on_update
    def _(_event):
        _apply_frame(int(frame_slider.value))

    @prev_button.on_click
    def _(_event):
        frame_slider.value = max(0, int(frame_slider.value) - 1)

    @next_button.on_click
    def _(_event):
        frame_slider.value = min(len(frames) - 1, int(frame_slider.value) + 1)

    @show_gt.on_update
    def _(_event):
        _apply_frame(int(frame_slider.value))

    @show_obs.on_update
    def _(_event):
        _apply_frame(int(frame_slider.value))

    @show_policy.on_update
    def _(_event):
        _apply_frame(int(frame_slider.value))

    _apply_frame(0)
    print(f"Loaded {len(frames)} frames from {recording}")
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
