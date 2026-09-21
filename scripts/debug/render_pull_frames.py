"""Offscreen render of pull-door plan frames, so the motion can actually be LOOKED at.

viser needs a browser. This draws the same thing with matplotlib and writes PNGs: the door
(frame / panel / handle point clouds at that frame's joint angles) plus the robot arm as a
stick figure with the gripper jaws, from a top-down and an isometric view.

It also measures what the eye is being asked to judge: the signed distance from the gripper to
the PANEL PLANE, so "is the arm going through the door" is a number, not an impression.
Negative = the gripper is on the far side of the panel, i.e. penetrating it.

    python scripts/debug/render_pull_frames.py --door-urdf <path> --phase block

Use --keyframes to render exactly the planner's mark_keyframe=True waypoints (mapped through
collocate_and_playback, i.e. what the env actually tracks), one PNG per keyframe, named so they
sort in trajectory order and identify the step, e.g. /tmp/viser_debug/k03_grasp_open.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from DoorOpening.constants.env_constants import (
    DOOR_INITIAL_POS, DOOR_INITIAL_ROT, ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT,
)
from DoorOpening.constants.robot_constants import BASE_JOINT_NAMES, FRANKA_DEFAULT_JOINT_POS, FRANKA_JOINT_NAMES
from DoorOpening.utils.extract_pointcloud_from_articulation import sample_pointcloud_from_link_name
from DoorOpening.utils.pose_utils import base_to_world_frame
from DoorOpening.utils.state_machine.compute_waypoint import (
    collocate_and_playback, get_robot_constants, resolve_planner_options,
)
from DoorOpening.utils.state_machine.offline_pull_door import (
    GRIPPER_Q_IDX, state_machine_offline_pull_door,
)
# GRIPPER_TCP_OFFSET was removed as a named constant from offline_pull_door.py; the physical
# value (palm_center offset off panda_hand, see offline_pull_door.py) is unchanged at 0.1034 m.
GRIPPER_TCP_OFFSET = 0.1034
from DoorOpening.utils.state_machine.pin import PinocchioIKSolver
from isaaclab.utils.math import quat_apply

REPO = Path(__file__).resolve().parents[2]
ARM_LINKS = ["panda_link0", "panda_link2", "panda_link4", "panda_link6", "panda_hand"]


def _wxyz(q):
    return torch.tensor([[q[3], q[0], q[1], q[2]]], dtype=torch.float32)


def _door_cloud(door_urdf, door_pose, q_door, link):
    pc = sample_pointcloud_from_link_name(door_urdf, q_door.unsqueeze(0), link, device="cpu")
    return (quat_apply(door_pose[..., 3:], pc) + door_pose[..., :3]).squeeze(0).numpy()


def _keyframe_tags(handle_side: str, n_keys: int) -> list[str] | None:
    """Step names for each RAW planner keyframe, in the order offline_pull_door.py appends them.

    Mirrors the fixed sequence of _append_state(..., mark_keyframe=True) calls in
    state_machine_offline_{left,right}_pull_door: 5 single keyframes (start / pregrasp /
    grasp_open / grasp_closed / unlatch), then the pull sweep's periodic keyframes, then
    release_open / release_retreat / retract_translate / retract_reorient (the retract is now
    TWO stages: translate+lift holding grasp_rot, then re-orient to block_rot in place), the 4
    block-approach stages / block_contact / block_push_open, and traverse_mid / traverse_end.
    Returns None (caller falls back to plain indices) if the constants below drift from the
    source and the count no longer matches -- this function must never be trusted over the
    actual key_idx_in_key_indices length.
    """
    pull_theta_start, pull_theta_stop, pull_theta_step = 0.0, 1.25, 0.025
    pull_keyframe_every = 2
    block_approach_clearances = (0.45, 0.30, 0.18, 0.10)

    n_theta = int(round((pull_theta_stop - pull_theta_start) / pull_theta_step)) + 1
    pull_tags = [
        f"pull_{pull_theta_start + i * pull_theta_step:.3f}"
        for i in range(n_theta)
        if i > 0 and i % pull_keyframe_every == 0
    ]
    if (n_theta - 1) % pull_keyframe_every != 0:
        pull_tags.append(f"pull_{pull_theta_start + (n_theta - 1) * pull_theta_step:.3f}")

    tags = (
        ["start", "pregrasp", "grasp_open", "grasp_closed", "unlatch"]
        + pull_tags
        + ["release_open", "release_retreat", "retract_translate", "retract_reorient", "retract_block_bridge"]
        + [f"block_approach_{c:.2f}" for c in block_approach_clearances]
        + ["block_contact", "block_push_open", "traverse_mid", "traverse_end"]
    )
    if len(tags) != n_keys:
        return None
    return tags


def _frame_geometry(i, ri, di, args, robot_pose, door_pose, fk):
    """FK + door point clouds for playback frame i. Shared by the PNG and video renderers."""
    q = ri[i]
    pts = []
    for ln in ARM_LINKS:
        pb, qb = fk.compute_fk(q[:10].numpy(), link_name=ln)
        pw, qw = base_to_world_frame(robot_pose[:, :3], robot_pose[:, 3:],
                                     torch.tensor([pb], dtype=torch.float32), _wxyz(qb))
        pts.append((pw.squeeze(0).numpy(), qw))
    hand_p, hand_q = pts[-1]
    approach = quat_apply(hand_q, torch.tensor([[0., 0., 1.]])).squeeze(0).numpy()
    jaw = quat_apply(hand_q, torch.tensor([[0., 1., 0.]])).squeeze(0).numpy()
    tcp = hand_p + GRIPPER_TCP_OFFSET * approach
    w = float(q[GRIPPER_Q_IDX])
    fing = [tcp + s * w * jaw for s in (1, -1)]

    panel = _door_cloud(args.door_urdf, door_pose, di[i], "link_1")
    handle = _door_cloud(args.door_urdf, door_pose, di[i], "link_2")
    frame_pc = _door_cloud(args.door_urdf, door_pose, di[i], "link_0")

    # panel plane by PCA; normal = smallest singular direction
    c = panel.mean(0)
    n = np.linalg.svd(panel - c)[2][-1]
    # Orient the normal by the HANDLE, not by the robot's fixed start pose. The old test
    # (dot(ROBOT_INITIAL_POS - c, n) < 0) is only meaningful while the panel still faces where the
    # robot started: once the door swings past ~70 deg that reference direction becomes nearly
    # PARALLEL to the panel plane, the test's |cos| collapses to 0.09-0.29, and the sign it picks
    # is decided by noise -- which flipped `gap` negative for the whole block phase and read as
    # the gripper diving through the panel when it was on the correct side the entire time.
    # link_2 is mounted on the face the robot works from, so this is well-defined at any angle.
    if np.dot(handle.mean(0) - c, n) < 0:
        n = -n
    gap = float(np.dot(tcp - c, n))
    board = float(di[i, 0])
    return dict(pts=pts, tcp=tcp, approach=approach, fing=fing, panel=panel, handle=handle,
                frame_pc=frame_pc, gap=gap, board=board, w=w)


def _render_one(i, name, ri, di, args, robot_pose, door_pose, fk, outdir):
    """FK + door clouds + top-down/iso plot for playback frame i. Returns the panel gap (m)."""
    g = _frame_geometry(i, ri, di, args, robot_pose, door_pose, fk)
    pts, tcp, approach, fing = g["pts"], g["tcp"], g["approach"], g["fing"]
    panel, handle, frame_pc = g["panel"], g["handle"], g["frame_pc"]
    gap, board, w = g["gap"], g["board"], g["w"]
    print(f"{i:>6} {board:>6.2f} {w:>5.3f} {gap:>10.3f}  {name}")

    fig = plt.figure(figsize=(13, 6))
    for k, (view, title) in enumerate(((dict(elev=90, azim=-90), "top-down"),
                                       (dict(elev=22, azim=-70), "iso"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for pc, col, sz, lbl in ((frame_pc, "0.6", 1, "frame"), (panel, "tab:blue", 1, "panel"),
                                 (handle, "tab:orange", 4, "handle")):
            s = pc[:: max(1, len(pc) // 1500)]
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=sz, c=col, label=lbl, depthshade=False)
        arm = np.array([p for p, _ in pts])
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", c="tab:red", lw=2, ms=3, label="arm")
        f = np.array(fing)
        ax.plot(f[:, 0], f[:, 1], f[:, 2], "-", c="k", lw=3, label="jaws")
        ax.quiver(*tcp, *(0.15 * approach), color="tab:green", lw=2)
        ax.view_init(**view)
        ax.set_title(f"{title}  {name}  frame {i}  board={board:.2f}  gap={gap:+.3f} m")
        ax.set_xlim(-1.2, 1.6); ax.set_ylim(-1.4, 1.4); ax.set_zlim(0, 2.0)
        ax.set_box_aspect((2.8, 2.8, 2.0))
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", dpi=80)
    plt.close(fig)
    return gap


def _render_video_frame(i, ri, di, args, robot_pose, door_pose, fk, view):
    """Single-view (fast) render of playback frame i as an RGB array, for --video."""
    g = _frame_geometry(i, ri, di, args, robot_pose, door_pose, fk)
    pts, tcp, approach, fing = g["pts"], g["tcp"], g["approach"], g["fing"]
    panel, handle, frame_pc = g["panel"], g["handle"], g["frame_pc"]
    gap, board, w = g["gap"], g["board"], g["w"]

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for pc, col, sz, lbl in ((frame_pc, "0.6", 1, "frame"), (panel, "tab:blue", 1, "panel"),
                             (handle, "tab:orange", 5, "handle")):
        s = pc[:: max(1, len(pc) // 1200)]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=sz, c=col, label=lbl, depthshade=False)
    arm = np.array([p for p, _ in pts])
    ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", c="tab:red", lw=2, ms=3, label="arm")
    f = np.array(fing)
    ax.plot(f[:, 0], f[:, 1], f[:, 2], "-", c="k", lw=3, label="jaws")
    ax.quiver(*tcp, *(0.15 * approach), color="tab:green", lw=2)
    ax.view_init(**view)
    ax.set_title(f"frame {i:04d}  board={board:+.2f}  grip={w:.3f}  gap={gap:+.3f} m")
    ax.set_xlim(-1.2, 1.6); ax.set_ylim(-1.4, 1.4); ax.set_zlim(0, 2.0)
    ax.set_box_aspect((2.8, 2.8, 2.0))
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--door-urdf", required=True)
    ap.add_argument("--robot-urdf", default=str(REPO / "source/DoorOpening/assets/glorbot/glorbot.urdf"))
    ap.add_argument("--outdir", default="/tmp/pull_frames")
    ap.add_argument("--n", type=int, default=6, help="frames to render across the chosen phase")
    ap.add_argument("--phase", choices=["all", "pull", "block"], default="block")
    ap.add_argument("--min-board", type=float, default=None,
                    help="only frames after the door has opened this far (rad)")
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--keyframes", action="store_true",
                    help="ignore --phase/--n/--min-board; render exactly the planner's "
                         "mark_keyframe=True waypoints, one PNG per keyframe, named by step")
    ap.add_argument("--video", type=str, default=None,
                    help="write an MP4 of the WHOLE collocated playback (every frame the env "
                         "actually tracks, not just keyframes) to this path, e.g. /tmp/pull.mp4. "
                         "Overrides --keyframes/--phase/--n.")
    ap.add_argument("--video-stride", type=int, default=2,
                    help="render 1 in N collocated frames into the video")
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--video-view", choices=["iso", "top"], default="iso")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    robot_pose = torch.tensor([[*ROBOT_INITIAL_POS, *ROBOT_INITIAL_ROT]], dtype=torch.float32)
    door_pose = torch.tensor([[*DOOR_INITIAL_POS, *DOOR_INITIAL_ROT]], dtype=torch.float32)
    _, q0 = get_robot_constants()
    side, direction = resolve_planner_options(args.door_urdf, "auto", "auto")
    print(f"{Path(args.door_urdf).parent.name}: {direction} / {side}-side handle")

    rt, dt, keys = state_machine_offline_pull_door(
        args.robot_urdf, args.door_urdf, robot_pose, door_pose, q0,
        torch.tensor([0.0, 0.0]), handle_side=side, device="cpu")
    n_raw_keys = len(keys)
    ri, di, _, _, keys = collocate_and_playback(rt, dt, keys, length=args.length)

    fk = PinocchioIKSolver(urdf_path=args.robot_urdf, ee_link_name="panda_hand",
                           controlled_joints=BASE_JOINT_NAMES + FRANKA_JOINT_NAMES,
                           reference_joint_pos=FRANKA_DEFAULT_JOINT_POS)

    if args.video is not None:
        import imageio.v3 as iio
        view = dict(elev=90, azim=-90) if args.video_view == "top" else dict(elev=22, azim=-70)
        idxs = list(range(0, ri.shape[0], args.video_stride))
        # Also dump the individual PNGs (not just the MP4 container): I have no video-playback
        # tool, so this is how I actually flip through the whole motion myself, frame by frame,
        # instead of judging it from a handful of keyframes.
        # Sit the PNG folder next to the MP4, NOT under --outdir: --outdir belongs to the
        # keyframe/phase renderers, and pointing it at a "<name>_frames" folder (the natural
        # thing to do) made this append "_frames" a second time and bury the PNGs one level
        # deeper than the printed path claimed.
        video_path = Path(args.video)
        pngs_dir = video_path.parent / (video_path.stem + "_frames")
        pngs_dir.mkdir(parents=True, exist_ok=True)
        print(f"rendering {len(idxs)} frames (stride {args.video_stride}) -> {args.video} "
              f"and {pngs_dir} ...")
        frames = []
        for n, i in enumerate(idxs):
            img = _render_video_frame(i, ri, di, args, robot_pose, door_pose, fk, view)
            frames.append(img)
            iio.imwrite(pngs_dir / f"v{n:04d}_f{i:04d}.png", img)
            if n % 25 == 0:
                print(f"  {n}/{len(idxs)}", flush=True)
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.video, frames, fps=args.video_fps, codec="libx264",
                    macro_block_size=None)
        print(f"wrote {args.video} ({len(frames)} frames @ {args.video_fps} fps) and "
              f"{len(frames)} PNGs to {pngs_dir}")
        return

    if args.keyframes:
        tags = _keyframe_tags(side, n_raw_keys)
        if tags is None:
            print(f"WARNING: _keyframe_tags() step count doesn't match {n_raw_keys} raw "
                  f"keyframes -- the planner's structure has drifted from what this script "
                  f"assumes. Falling back to plain indices; re-derive the tag list.")
            tags = [f"{i:02d}" for i in range(n_raw_keys)]
        print(f"\n{'frame':>6} {'board':>6} {'grip':>5} {'panel gap':>10}  step "
              f"  (negative gap = gripper THROUGH the panel)")
        for idx, (i, name) in enumerate(zip(keys, tags)):
            # collocate_and_playback's last key index is a cumulative-length total (==
            # len(ri)), one past the last valid sample -- clip it back onto the array.
            i = min(i, ri.shape[0] - 1)
            _render_one(i, f"k{idx:02d}_{name}", ri, di, args, robot_pose, door_pose, fk, outdir)
        print(f"\nwrote {len(keys)} keyframe PNGs to {outdir}")
        return

    grip = ri[:, GRIPPER_Q_IDX].numpy()
    board = di[:, 0].numpy()
    if args.phase == "pull":
        sel = np.where(grip < 0.02)[0]
    elif args.phase == "block":
        # after release: the retract + swing-around + panel contact
        rel = np.where(grip < 0.02)[0]
        sel = np.arange(rel[-1] if len(rel) else 0, len(grip))
    else:
        sel = np.arange(len(grip))
    if args.min_board is not None:
        sel = np.array([i for i in sel if board[i] >= args.min_board]) if len(sel) else sel
    if len(sel) == 0:
        print("no frames matched"); return
    idxs = sel[np.linspace(0, len(sel) - 1, min(args.n, len(sel))).astype(int)]

    print(f"\n{'frame':>6} {'board':>6} {'grip':>5} {'panel gap':>10}   (negative = gripper THROUGH the panel)")
    for i in idxs:
        _render_one(i, f"f{i:04d}", ri, di, args, robot_pose, door_pose, fk, outdir)
    print(f"\nwrote {len(idxs)} PNGs to {outdir}")


if __name__ == "__main__":
    main()
