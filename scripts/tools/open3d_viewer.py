#!/usr/bin/env python3
"""Keep an Open3D point-cloud window alive for interactive inspection."""
import sys
import time

import open3d as o3d

path = sys.argv[1]
cloud = o3d.io.read_point_cloud(path)
print(f"loaded {len(cloud.points)} points from {path}", flush=True)
vis = o3d.visualization.Visualizer()
ok = vis.create_window(window_name="DoorOpening mesh-depth composite", width=1280, height=800, visible=True)
print(f"create_window={ok}", flush=True)
vis.add_geometry(cloud)
render = vis.get_render_option()
render.point_size = 2.0
render.background_color = [0.04, 0.04, 0.04]
vis.reset_view_point(True)
try:
    while vis.poll_events():
        vis.update_renderer()
        time.sleep(0.02)
finally:
    vis.destroy_window()
