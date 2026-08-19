#!/usr/bin/env python3
"""Browse door URDFs ONE AT A TIME in viser, with autoplay + a tunable glorbot.

Loads a single ``mobility.urdf`` at a time and swaps it in place, so you can page through a large
asset directory (e.g. the 512 doors in PartNetv5_plusplus) one by one without drowning the browser.

The glorbot robot is loaded once at the origin next to the door, with one slider per actuated
joint so you can pose it joint-by-joint (base x/y/yaw, panda arm, gripper, x5).

Each door is shown at the CLOSED pose (all joints = 0) with:
    - the door mesh (frame + panel + handle),
    - a coordinate frame at every door link (base, link_0, link_1, link_2), and
    - the board keypoints (blue) and handle keypoints (red) the teacher sees.

GUI (top-right panel in the viser browser tab):
    - Prev / Next buttons, a slider, and a dropdown to jump to any door,
    - Autoplay checkbox + FPS slider to page through doors automatically,
    - checkboxes to toggle the link frames, keypoints, and robot,
    - a "Robot joints" folder with one slider per glorbot joint (+ reset).

Example:

    PYTHONPATH=source python scripts/tools/browse_doors_viser.py \
        --asset-dir source/DoorOpening/assets/door/PartNetv5_plusplus
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from DoorOpening.utils.urdf_utils import compute_exact_door_keypoints

DOOR_LINKS = ["base", "link_0", "link_1", "link_2"]
DEFAULT_GLORBOT = SOURCE_ROOT / "DoorOpening/assets/glorbot/glorbot.urdf"

# Joints whose URDF limits are effectively unbounded -- clamp the slider to something usable.
SLIDER_CLAMP = {
    "base_x_joint": (-3.0, 3.0),
    "base_y_joint": (-3.0, 3.0),
    "base_rotation_joint": (-math.pi, math.pi),
    "x5_joint1": (-math.pi, math.pi),
}
# Group joints into GUI folders by name prefix for readability.
JOINT_GROUPS = [
    ("Base", ("base_",)),
    ("Panda arm", ("panda_",)),
    ("Gripper", ("panda_finger_joint",)),
    ("X5", ("x5_",)),
]


def rotmat_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return (float(w), float(x), float(y), float(z))


def transform_points(points_local, transform_4x4):
    pts = np.asarray(points_local, dtype=np.float64)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (homog @ np.asarray(transform_4x4, dtype=np.float64).T)[:, :3]


def slider_range(joint) -> tuple[float, float]:
    lo = getattr(joint.limit, "lower", None) if joint.limit else None
    hi = getattr(joint.limit, "upper", None) if joint.limit else None
    if lo is None or hi is None:
        return (-math.pi, math.pi)
    return (float(lo), float(hi))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--asset-dir",
        type=Path,
        default=SOURCE_ROOT / "DoorOpening/assets/door/PartNetv5_plusplus",
    )
    p.add_argument("--robot-urdf", type=Path, default=DEFAULT_GLORBOT, help="glorbot URDF (empty to skip).")
    p.add_argument("--axes-length", type=float, default=0.18, help="Length (m) of the link coordinate frames.")
    p.add_argument("--kpt-size", type=float, default=0.03, help="Rendered size (m) of the keypoint markers.")
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args()

    urdfs = sorted(glob.glob(os.path.join(str(args.asset_dir), "**/mobility.urdf"), recursive=True))
    if not urdfs:
        raise SystemExit(f"No mobility.urdf found under {args.asset_dir}")
    names = [Path(u).parent.name for u in urdfs]
    n = len(urdfs)

    server = viser.ViserServer(port=args.port) if args.port else viser.ViserServer()
    server.scene.add_grid("/grid", width=4.0, height=4.0)

    # --- door browsing controls ---
    server.gui.add_markdown(f"### `{args.asset_dir.name}`  ({n} doors)")
    gui_index = server.gui.add_text("Index", initial_value="", disabled=True)
    gui_name = server.gui.add_text("Door", initial_value="", disabled=True)
    gui_prev = server.gui.add_button("◀ Prev")
    gui_next = server.gui.add_button("Next ▶")
    gui_slider = server.gui.add_slider("Go to", min=0, max=n - 1, step=1, initial_value=0)
    gui_dropdown = server.gui.add_dropdown("Jump to name", options=names, initial_value=names[0])
    gui_play = server.gui.add_checkbox("▶ Autoplay", initial_value=False)
    gui_fps = server.gui.add_slider("Autoplay FPS", min=0.5, max=10.0, step=0.5, initial_value=2.0)
    gui_loop = server.gui.add_checkbox("Loop at end", initial_value=True)
    gui_show_frames = server.gui.add_checkbox("Show link frames", initial_value=True)
    gui_show_kpts = server.gui.add_checkbox("Show keypoints", initial_value=True)

    state = {"i": -1, "urdf_handle": None, "extras": []}

    def clear_scene():
        if state["urdf_handle"] is not None:
            try:
                state["urdf_handle"].remove()
            except Exception:
                pass
            state["urdf_handle"] = None
        for h in state["extras"]:
            try:
                h.remove()
            except Exception:
                pass
        state["extras"] = []

    def load_door(i: int):
        i = int(np.clip(i, 0, n - 1))
        clear_scene()
        state["i"] = i
        urdf_path = urdfs[i]
        name = names[i]
        gui_index.value = f"{i} / {n - 1}"
        gui_name.value = name

        urdf_abs = os.path.abspath(urdf_path)
        urdf = yourdfpy.URDF.load(urdf_abs, build_scene_graph=True, load_meshes=True)
        cfg = np.zeros(len(urdf.actuated_joint_names))
        urdf.update_cfg(cfg)

        viser_urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/door", load_meshes=True)
        try:
            viser_urdf.update_cfg(cfg)
        except Exception:
            pass
        state["urdf_handle"] = viser_urdf

        # --- link poses at the closed config ---
        link_T = {}
        for link in DOOR_LINKS:
            try:
                T = np.asarray(urdf.get_transform(link, "base"), dtype=np.float64)
            except Exception:
                continue
            link_T[link] = T
            state["extras"].append(
                server.scene.add_frame(
                    f"/frames/{link}",
                    position=tuple(T[:3, 3]),
                    wxyz=rotmat_to_wxyz(T[:3, :3]),
                    axes_length=args.axes_length,
                    axes_radius=args.axes_length * 0.04,
                    visible=gui_show_frames.value,
                )
            )

        # --- teacher keypoints (board = blue on link_1, handle = red on link_2) ---
        try:
            kp = compute_exact_door_keypoints(urdf_abs)
            if "link_1" in link_T and kp.get("link_1"):
                board = transform_points(kp["link_1"], link_T["link_1"]).astype(np.float32)
                state["extras"].append(
                    server.scene.add_point_cloud(
                        "/kpts/board", points=board,
                        colors=np.tile(np.array([40, 120, 255], np.uint8), (board.shape[0], 1)),
                        point_size=args.kpt_size, visible=gui_show_kpts.value,
                    )
                )
            if "link_2" in link_T and kp.get("link_2"):
                handle = transform_points(kp["link_2"], link_T["link_2"]).astype(np.float32)
                state["extras"].append(
                    server.scene.add_point_cloud(
                        "/kpts/handle", points=handle,
                        colors=np.tile(np.array([255, 60, 60], np.uint8), (handle.shape[0], 1)),
                        point_size=args.kpt_size, visible=gui_show_kpts.value,
                    )
                )
        except Exception as exc:
            print(f"[warn] keypoints failed for {name}: {exc}")

        if gui_slider.value != i:
            gui_slider.value = i
        if gui_dropdown.value != name:
            gui_dropdown.value = name
        print(f"[{i}/{n - 1}] {name} | links {sorted(link_T)}")

    @gui_prev.on_click
    def _(_) -> None:
        gui_play.value = False
        load_door(state["i"] - 1)

    @gui_next.on_click
    def _(_) -> None:
        gui_play.value = False
        load_door(state["i"] + 1)

    @gui_slider.on_update
    def _(_) -> None:
        if gui_slider.value != state["i"]:
            load_door(gui_slider.value)

    @gui_dropdown.on_update
    def _(_) -> None:
        i = names.index(gui_dropdown.value)
        if i != state["i"]:
            load_door(i)

    @gui_show_frames.on_update
    def _(_) -> None:
        for h in state["extras"]:
            if getattr(h, "name", "").startswith("/frames/"):
                h.visible = gui_show_frames.value

    @gui_show_kpts.on_update
    def _(_) -> None:
        for h in state["extras"]:
            if getattr(h, "name", "").startswith("/kpts/"):
                h.visible = gui_show_kpts.value

    # --- glorbot robot with per-joint sliders ---
    if str(args.robot_urdf):
        robot_urdf = yourdfpy.URDF.load(
            os.path.abspath(str(args.robot_urdf)), build_scene_graph=True, load_meshes=True
        )
        joint_names = list(robot_urdf.actuated_joint_names)
        robot_cfg = np.zeros(len(joint_names))
        robot_viser = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot", load_meshes=True)
        robot_viser.update_cfg(robot_cfg)

        with server.gui.add_folder("Robot"):
            gui_show_robot = server.gui.add_checkbox("Show robot", initial_value=True)
            gui_reset_robot = server.gui.add_button("Reset joints")
            joint_sliders = {}
            for group_label, prefixes in JOINT_GROUPS:
                group_joints = [jn for jn in joint_names if jn.startswith(prefixes)]
                if not group_joints:
                    continue
                with server.gui.add_folder(group_label):
                    for jn in group_joints:
                        lo, hi = SLIDER_CLAMP.get(jn, slider_range(robot_urdf.joint_map[jn]))
                        init = float(np.clip(0.0, lo, hi))
                        s = server.gui.add_slider(jn, min=lo, max=hi, step=(hi - lo) / 200.0, initial_value=init)
                        joint_sliders[jn] = s

        def apply_robot_cfg() -> None:
            cfg = np.array([joint_sliders[jn].value for jn in joint_names], dtype=np.float64)
            robot_viser.update_cfg(cfg)

        for jn, s in joint_sliders.items():
            s.on_update(lambda _evt: apply_robot_cfg())

        @gui_show_robot.on_update
        def _(_) -> None:
            robot_viser.show_visual = gui_show_robot.value

        @gui_reset_robot.on_click
        def _(_) -> None:
            for s in joint_sliders.values():
                s.value = float(np.clip(0.0, s.min, s.max))
            apply_robot_cfg()

        apply_robot_cfg()

    load_door(0)
    print(f"\nServing {n} doors from {args.asset_dir}. Open the viser URL; Prev/Next/Autoplay. Ctrl-C to stop.")

    last_advance = time.time()
    while True:
        if gui_play.value:
            now = time.time()
            if now - last_advance >= 1.0 / max(gui_fps.value, 1e-3):
                last_advance = now
                nxt = state["i"] + 1
                if nxt >= n:
                    if gui_loop.value:
                        nxt = 0
                    else:
                        gui_play.value = False
                        nxt = state["i"]
                if nxt != state["i"]:
                    load_door(nxt)
        time.sleep(0.02)


if __name__ == "__main__":
    main()
