#!/usr/bin/env python3
"""True per-frame wall-config sequence: resample walls and rerender the camera cloud each frame."""
from pathlib import Path
import sys, json, argparse
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source")); sys.path.insert(0, str(ROOT / "scripts" / "tools"))
import render_dex_style_wall_scene as rd
from DoorOpening.utils.camera_utils import rasterize_depth_zbuffer_from_pose, backproject_depth_to_world_from_pose
from DoorOpening.utils.wall_distractors import WallDistractorParams, compute_wall_bbox_ordering, sample_wall_points_local


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--door", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--wall-points", type=int, default=120000)
    p.add_argument("--width", type=int, default=320); p.add_argument("--height", type=int, default=240)
    args = p.parse_args()
    dev = torch.device("cpu")
    init = rd.load_initial_state(args.door)
    full_bbox, panel_bbox, *_ = rd.load_door_asset(args.door, 30000, dev)
    from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler
    sampler = FrankaGripperSampler(str(args.door), device=dev, num_points=16000)
    q = torch.zeros((1, len(sampler.robot.actuated_joint_names)), device=dev)
    door_world_pos_np, door_world_R_np = rd._pose_from_initial_state(init, "door_world_pose")
    door_pos = torch.tensor(door_world_pos_np, dtype=torch.float32)
    door_R = torch.tensor(door_world_R_np, dtype=torch.float32)
    robot, camera_pose, *_ = rd.load_ready_robot(dev, -0.62, init, base_approach_m=0.25, arm_down_rad=0.15)
    cam_pos_base = (camera_pose[0, :3] - door_pos) @ door_R
    door_base = rd.sample_camera_facing_visuals(sampler, ("link_0", "link_1", "link_2"), cam_pos_base.numpy(), 16000)
    door_base = door_base @ door_R.T + door_pos
    panel_base = rd.sample_camera_facing_visuals(sampler, ("link_1",), cam_pos_base.numpy(), 12000)
    panel_world = panel_base @ door_R.T + door_pos
    axis, fmin, fmax = compute_wall_bbox_ordering(full_bbox)
    pmin = torch.gather(panel_bbox[:, 0], 1, axis); pmax = torch.gather(panel_bbox[:, 1], 1, axis)
    cfg = {"enabled": True, "num_points": args.wall_points, "side_margin_m": [0.35, 0.80],
           "edge_gap_m": [0.015, 0.04], "depth_m": [0.10, 0.80], "center_offset_m": [-0.30, 0.30],
           "face_jitter_m": 0.004, "flush_prob": 0.7, "flush_point_fraction": 0.5,
           "flush_extent_m": [0.6, 1.6], "detached_within_flush_frac": [0.0, 1.0]}
    params = WallDistractorParams.from_cfg(cfg, args.wall_points)
    cam = rd.build_camera_spec(args.width, args.height, 0.25, 3.0, dev)
    frames=[]
    for i in range(args.frames):
        torch.manual_seed(17000+i)
        walls = sample_wall_points_local(axis, fmin, fmax, args.wall_points, params, dev, pmin, pmax)[0]
        # Preserve the training distribution, including flush walls.  The sampler's
        # physical-overlap filter removes only points inside the full door AABB; do not
        # apply a broad front/back half-space cut here, since that deletes valid flush
        # distractors beside the panel.
        walls = walls @ door_R.T + door_pos
        scene = torch.cat((door_base, walls), dim=0).unsqueeze(0)
        door_depth, intr = rasterize_depth_zbuffer_from_pose(door_base.unsqueeze(0), camera_pose, cam, inflate_px=2, clip_mode="post")
        wall_depth, _ = rasterize_depth_zbuffer_from_pose(walls.unsqueeze(0), camera_pose, cam, inflate_px=2, clip_mode="post")
        robot_depth, _ = rasterize_depth_zbuffer_from_pose(robot, camera_pose, cam, inflate_px=2, clip_mode="post")
        panel_depth, _ = rasterize_depth_zbuffer_from_pose(panel_world.unsqueeze(0), camera_pose, cam, inflate_px=2, clip_mode="post")
        panel_mask = torch.isfinite(panel_depth[0])
        protected_walls = torch.where(panel_mask & (wall_depth[0] < panel_depth[0]), torch.full_like(wall_depth[0], float("inf")), wall_depth[0])
        depth = torch.minimum(torch.minimum(door_depth[0], protected_walls), robot_depth[0]).unsqueeze(0)
        rendered, valid = backproject_depth_to_world_from_pose(depth, camera_pose, intr)
        policy = rendered[0][valid[0]].detach().cpu().to(torch.float16)
        frames.append({"pointclouds": {
            "policy_input": policy,
            "ground_truth_door": door_base.detach().cpu().to(torch.float16),
            "ground_truth_walls": walls.detach().cpu().to(torch.float16),
            "ground_truth_robot": robot[0].detach().cpu().to(torch.float16),
        }})
    payload={"format":"dooropening_viser_replay_v1","pointcloud_frame":"world","pointcloud_source":"per_frame_exact_training_wall_zbuffer",
      "pointcloud_streams":[
        {"name":"policy_input","label":"Per-frame camera-visible policy cloud","color":(120,190,230)},
        {"name":"ground_truth_door","label":"Ground-truth door: panel + frame + handle","color":(255,193,7)},
        {"name":"ground_truth_walls","label":"Per-frame solid wall distractors","color":(100,100,255)},
        {"name":"ground_truth_robot","label":"Ground-truth robot geometry","color":(0,170,120)}],
      "frame_dt":0.1,"frame_fps":10.0,"frames":frames,"wall_sequence":{"num_frames":args.frames,"wall_points":args.wall_points,"seed_base":17000}}
    torch.save(payload,args.output); print(f"saved={args.output} frames={len(frames)} wall_points={args.wall_points}")

if __name__ == "__main__": main()
