#!/usr/bin/env python3
"""Pack surface-vs-solid wall occlusion outputs into one compact Viser replay."""

from pathlib import Path
import argparse
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("surface", type=Path)
    p.add_argument("solid", type=Path)
    p.add_argument("output", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    surface = torch.load(args.surface, map_location="cpu", weights_only=False)
    solid = torch.load(args.solid, map_location="cpu", weights_only=False)
    sf = surface["frames"][0]["pointclouds"]
    so = solid["frames"][0]["pointclouds"]
    streams = [
        {"name": "surface_render", "label": "Surface walls: rendered cloud", "color": (79, 195, 247)},
        {"name": "solid_render", "label": "Solid walls: rendered cloud", "color": (0, 230, 140)},
        {"name": "surface_walls", "label": "Surface wall samples", "color": (100, 100, 255)},
        {"name": "solid_walls", "label": "Solid-box wall samples", "color": (255, 170, 20)},
        {"name": "surface_residual", "label": "Surface residual behind door", "color": (255, 40, 40)},
        {"name": "solid_residual", "label": "Solid residual behind door", "color": (255, 255, 0)},
        {"name": "solid_door_panel", "label": "Door panel", "color": (220, 220, 220)},
        {"name": "robot", "label": "Robot input surface", "color": (0, 170, 120)},
    ]
    out = {
        "format": "dooropening_viser_replay_v1",
        "pointcloud_frame": "world",
        "pointcloud_source": "wall_occlusion_surface_vs_solid_box",
        "sampler_ab": {
            "surface_wall_residual": surface.get("sampler_ab", {}).get("wall_residual_behind_door"),
            "surface_wall_residual_fraction": surface.get("sampler_ab", {}).get("wall_residual_fraction"),
            "solid_wall_residual": solid.get("sampler_ab", {}).get("wall_residual_behind_door"),
            "solid_wall_residual_fraction": solid.get("sampler_ab", {}).get("wall_residual_fraction"),
            "surface_wall_points": int(sf["walls"].shape[0]),
            "solid_wall_points": int(so["walls"].shape[0]),
            "same_camera": "160x120, identical z-buffer inflate_px=2",
        },
        "pointcloud_streams": streams,
        "frame_dt": 0.5,
        "frame_fps": 2.0,
        "frames": [{"pointclouds": {
            "surface_render": sf["robot_depth_cam_obs"],
            "solid_render": so["robot_depth_cam_obs"],
            "surface_walls": sf["walls"],
            "solid_walls": so["walls"],
            "surface_residual": sf["wall_residual_behind_door"],
            "solid_residual": so["wall_residual_behind_door"],
            "solid_door_panel": so["solid_door_panel"],
            "robot": so["robot"],
        }}],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"replay={args.output}")
    print(f"surface_residual={out['sampler_ab']['surface_wall_residual']} ({out['sampler_ab']['surface_wall_residual_fraction']:.6f})")
    print(f"solid_residual={out['sampler_ab']['solid_wall_residual']} ({out['sampler_ab']['solid_wall_residual_fraction']:.6f})")


if __name__ == "__main__":
    main()
