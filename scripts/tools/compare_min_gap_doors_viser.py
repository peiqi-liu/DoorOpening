#!/usr/bin/env python3
"""Side-by-side viser comparison of the SMALLEST-underside-gap door from two families.

Loads the two real (textured) door URDFs with ViserUrdf, aligns their handles at a common point and
separates them in x, so you can directly compare the finger-clearance gap under the lever AND the lever
bar's own thickness. A translucent sphere shows the Franka finger's 21 mm cross-section for scale (does
the finger fit the gap / can it wrap the bar?), and a label reports each door's gap + lever thickness.

Defaults to each family's minimum-underside-gap variant; override with --door-a/--door-b.

    PYTHONPATH=source python scripts/tools/compare_min_gap_doors_viser.py
    # then open the printed viser URL in a browser
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

from DoorOpening.utils.urdf_utils import compute_exact_door_keypoints

REPO_ROOT = Path(__file__).resolve().parents[2]
DOOR_ROOT = REPO_ROOT / "source" / "DoorOpening" / "assets" / "door"
# Franka finger width across the lever axis, i.e. the dimension that has to fit the gap under the bar.
# Measured from the collision mesh (meshes/franka/collision/finger.obj), whose bounds are
# 21.0 x 26.5 x 53.7 mm in (x, y, z) = (across the bar, closing direction, reach).
FRANKA_FINGER_WIDTH_M = 0.021


def _underside_gap(handle: dict) -> float:
    if handle.get("panel_underside_gap_m") is not None:
        return float(handle["panel_underside_gap_m"])
    lever = next((c for c in handle["collision_primitives"] if c["name"] == "handle_main_lever"), None)
    return float(lever["origin_xyz"][2] - lever["size"][2] / 2.0)


def _lever_thickness(handle: dict) -> float:
    return float(handle.get("lever_thickness_m", 2.0 * handle["radius_m"]))


def find_min_gap_door(family: str) -> Path:
    best_dir, best_gap = None, None
    for meta_path in sorted(glob.glob(str(DOOR_ROOT / family / "scratch_door__rnd_*" / "variant_meta.json"))):
        gap = _underside_gap(json.load(open(meta_path))["handle"])
        if best_gap is None or gap < best_gap:
            best_gap, best_dir = gap, Path(meta_path).parent
    if best_dir is None:
        raise FileNotFoundError(f"No scratch_door variants found under {DOOR_ROOT / family}")
    return best_dir


def _handle_pos_in_base(urdf: yourdfpy.URDF, urdf_abs: str) -> np.ndarray:
    """Closed-door lever position in the base frame (mean of the link_2 handle keypoints)."""
    T_link2 = np.asarray(urdf.get_transform("link_2", "base"), dtype=np.float64)
    kp = compute_exact_door_keypoints(urdf_abs)
    pts = kp.get("link_2")
    if pts:
        local = np.asarray(pts, dtype=np.float64)
        world = local @ T_link2[:3, :3].T + T_link2[:3, 3]
        return world.mean(axis=0)
    return T_link2[:3, 3]


def add_door(server, door_dir, x_off, tag, target_z, joint_cfg_fn):
    meta = json.load(open(door_dir / "variant_meta.json"))["handle"]
    gap, lever_th = _underside_gap(meta), _lever_thickness(meta)

    urdf_abs = os.path.abspath(str(door_dir / "mobility.urdf"))
    urdf = yourdfpy.URDF.load(urdf_abs, build_scene_graph=True, load_meshes=True)
    urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
    handle_pos = _handle_pos_in_base(urdf, urdf_abs)

    # Offset the whole door so its lever lands at (x_off, 0, target_z) -> both handles line up.
    frame_pos = np.array([x_off, 0.0, target_z]) - handle_pos
    server.scene.add_frame(f"/{tag}", position=tuple(frame_pos), wxyz=(1.0, 0.0, 0.0, 0.0), axes_length=0.01, axes_radius=0.001)
    viser_urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name=f"/{tag}/door", load_meshes=True)

    # Translucent Franka finger cross-section at the lever, for scale (fit the gap? wrap the bar?).
    finger = trimesh.creation.uv_sphere(radius=FRANKA_FINGER_WIDTH_M / 2.0)
    finger.apply_translation(np.array([x_off, 0.09, target_z]))
    server.scene.add_mesh_simple(
        f"/{tag}_finger", vertices=np.asarray(finger.vertices, np.float32),
        faces=np.asarray(finger.faces, np.uint32), color=(250, 220, 60), opacity=0.55,
    )
    server.scene.add_label(
        f"/{tag}_label",
        text=f"{door_dir.parent.name}  |  gap {gap*1000:.1f} mm  |  lever {lever_th*1000:.1f} mm",
        position=(x_off, 0.0, target_z + 0.18),
    )
    return viser_urdf, list(urdf.actuated_joint_names), gap, lever_th


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family-a", default="PartNetv5_plusplus_v1")
    p.add_argument("--family-b", default="PartNetv5_plusplus")
    p.add_argument("--door-a", type=Path, default=None, help="Explicit door dir (overrides --family-a).")
    p.add_argument("--door-b", type=Path, default=None, help="Explicit door dir (overrides --family-b).")
    p.add_argument("--separation", type=float, default=0.6, help="X-gap (m) between the two doors.")
    p.add_argument("--handle-z", type=float, default=1.0, help="World z the two levers are aligned to.")
    p.add_argument("--port", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    door_a = args.door_a or find_min_gap_door(args.family_a)
    door_b = args.door_b or find_min_gap_door(args.family_b)

    server = viser.ViserServer(port=args.port) if args.port else viser.ViserServer()
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    handle_open = server.gui.add_slider("Handle turn (rad)", min=0.0, max=1.4, step=0.05, initial_value=0.0)

    urdf_a, ajn_a, gap_a, lev_a = add_door(server, door_a, -args.separation / 2, "door_a", args.handle_z, None)
    urdf_b, ajn_b, gap_b, lev_b = add_door(server, door_b, +args.separation / 2, "door_b", args.handle_z, None)

    server.gui.add_markdown(
        f"### Smallest-gap door comparison\n"
        f"- **A (left):** `{door_a.parent.name}` — gap **{gap_a*1000:.1f} mm**, lever **{lev_a*1000:.1f} mm**\n"
        f"- **B (right):** `{door_b.parent.name}` — gap **{gap_b*1000:.1f} mm**, lever **{lev_b*1000:.1f} mm**\n"
        f"- Yellow sphere = Franka finger width Ø{FRANKA_FINGER_WIDTH_M*1000:.0f} mm (scale)."
    )

    @handle_open.on_update
    def _(_) -> None:
        for urdf_h, ajn in ((urdf_a, ajn_a), (urdf_b, ajn_b)):
            cfg = np.zeros(len(ajn))
            if "joint_2" in ajn:  # the handle (lever) joint
                cfg[ajn.index("joint_2")] = handle_open.value
            try:
                urdf_h.update_cfg(cfg)
            except Exception:
                pass

    print(f"[A] {door_a.parent.name}/{door_a.name}: gap {gap_a*1000:.1f} mm, lever {lev_a*1000:.1f} mm")
    print(f"[B] {door_b.parent.name}/{door_b.name}: gap {gap_b*1000:.1f} mm, lever {lev_b*1000:.1f} mm")
    print("[INFO] Open the viser URL above; levers are aligned so you can eyeball gap + bar thickness. Ctrl+C to quit.")
    try:
        import time
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
