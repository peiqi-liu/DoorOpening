"""Render the pull-door plan's KEYFRAMES from the real URDF meshes, via viser.

This is the mesh-accurate counterpart to render_pull_frames.py. That script draws the arm as a
5-point stick figure and the jaws as a single line, which is fine for reading gross posture but
CANNOT show mesh clipping or true hand-to-handle contact -- the two things that actually matter
when judging a grasp. Here the real robot and door URDFs are loaded with their meshes and posed
at each keyframe's q_robot / q_door, and the frame is captured with viser's own get_render().

Loading follows compute_waypoint.py's play_trajectories_in_viser exactly (yourdfpy URDF.load ->
viser.extras.ViserUrdf under a world-posed root frame), so what is drawn here is the same scene
that script shows, just stepped keyframe by keyframe instead of played back.

viser renders IN THE BROWSER, so this needs a browser pointed at the printed URL.

USE HEADLESS CHROME. Do not rely on a normal tab: Chrome only runs a tab's render loop while that
tab is actually VISIBLE, so a backgrounded (or merely not-frontmost) tab silently stops rendering
and every get_render times out -- while the websocket still reports "Connected", which makes it
look like the renderer is broken when it is not. Headless has no such throttling and needs nobody
watching it.

    python scripts/debug/render_keyframes_viser.py --door-urdf <path> --outdir /tmp/kf &
    # wait for the "open http://localhost:PORT" line, then:
    google-chrome --headless=new --disable-gpu --use-gl=swiftshader \\
        --enable-unsafe-swiftshader --no-sandbox --window-size=1200,900 \\
        --user-data-dir=/tmp/chrome-viser http://localhost:8080 &

Software GL makes headless ~7 s/frame versus ~0.3 s on a real visible tab, so pick --stride to
suit. Kill the chrome process when the run finishes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from DoorOpening.constants.env_constants import (
    DOOR_INITIAL_POS, DOOR_INITIAL_ROT, ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT,
)
from DoorOpening.constants.robot_constants import (
    BASE_JOINT_NAMES, FRANKA_DEFAULT_JOINT_POS, FRANKA_JOINT_NAMES,
)
from DoorOpening.utils.pose_utils import base_to_world_frame
from DoorOpening.utils.state_machine.compute_waypoint import (
    collocate_and_playback, get_robot_constants, resolve_planner_options,
)
from DoorOpening.utils.state_machine.offline_pull_door import (
    GRIPPER_Q_IDX, GRIPPER_TCP_OFFSET, state_machine_offline_pull_door,
)
from DoorOpening.utils.state_machine.pin import PinocchioIKSolver
from isaaclab.utils.math import quat_apply

REPO = Path(__file__).resolve().parents[2]

# Whole-scene viewpoints. Positions are world-frame; the door stands at the origin and the robot
# starts ~1 m out at +x and traverses to -1 m, so these hold both for every frame.
SCENE_VIEWS = {
    "iso":  dict(position=(3.6, -3.0, 2.4), look_at=(0.2, 0.0, 1.0)),
    "top":  dict(position=(0.6, 0.0, 4.6),  look_at=(0.2, 0.0, 1.0)),
    "side": dict(position=(0.4, -4.2, 1.6), look_at=(0.2, 0.0, 1.1)),
}
# The close-up sits this far from the TCP, along these world directions. Two opposed diagonals so
# a contact that one view leaves depth-ambiguous can be settled from the other.
HAND_VIEW_DIRS = {
    "hand_a": (0.75, -0.62, 0.25),
    "hand_b": (0.30, 0.80, 0.35),
}
HAND_VIEW_DIST = 0.55


def _wxyz(q):
    return torch.tensor([[q[3], q[0], q[1], q[2]]], dtype=torch.float32)


def _aim_and_render(client, position, look_at, settle, height, width):
    """Point the camera, then render -- both off the main thread, on purpose.

    Every one of these is a round-trip to the browser, and a tab that is throttled or has its
    viser message stream PAUSED simply never answers. On the main thread that wedges the whole
    run silently: a hang is not an exception, so wrapping the calls in try/except catches
    nothing. Run here, the caller's fut.result(timeout=...) can give up and move on.
    """
    client.camera.position = position
    client.camera.look_at = look_at
    time.sleep(settle)  # let the browser actually draw the new pose before grabbing it
    return client.get_render(height=height, width=width, transport_format="png")


def _tcp_world(fk, q_robot, robot_pose):
    """World-frame grasp center for a planned configuration (same convention as the planner)."""
    pb, qb = fk.compute_fk(q_robot[:10].numpy(), link_name="panda_hand")
    pw, qw = base_to_world_frame(
        robot_pose[:, :3], robot_pose[:, 3:],
        torch.tensor(np.asarray([pb]), dtype=torch.float32), _wxyz(qb),
    )
    hand_p = pw.squeeze(0).numpy()
    approach = quat_apply(qw, torch.tensor([[0.0, 0.0, 1.0]])).squeeze(0).numpy()
    return hand_p + GRIPPER_TCP_OFFSET * approach


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--door-urdf", required=True)
    ap.add_argument("--robot-urdf", default=str(REPO / "source/DoorOpening/assets/glorbot/glorbot.urdf"))
    ap.add_argument("--outdir", default="/tmp/viser_keyframes")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--views", default="iso,hand_a,hand_b",
                    help=f"comma-separated: {','.join(list(SCENE_VIEWS) + list(HAND_VIEW_DIRS))}")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=750)
    ap.add_argument("--settle", type=float, default=0.35,
                    help="seconds to let the browser draw a new pose before capturing it")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for a browser")
    ap.add_argument("--render-timeout", type=float, default=8.0,
                    help="seconds to wait on ONE client for ONE frame before trying the next "
                         "client. Short on purpose: an unresponsive (backgrounded) tab never "
                         "answers at all, so waiting longer just wastes the run.")
    ap.add_argument("--all-frames", action="store_true",
                    help="render the FULL collocated playback (what the env tracks), not just the "
                         "planner's keyframes -- a bad pose between two good keyframes only shows "
                         "up here")
    ap.add_argument("--length", type=int, default=1440,
                    help="collocated playback length for --all-frames (pull uses 1440)")
    ap.add_argument("--stride", type=int, default=1, help="render 1 in N frames with --all-frames")
    ap.add_argument("--video", type=str, default=None,
                    help="also mux the rendered frames into this .mp4 (one view only)")
    ap.add_argument("--video-fps", type=int, default=30)
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf
    from yourdfpy import URDF
    import imageio.v3 as iio

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = [v for v in views if v not in SCENE_VIEWS and v not in HAND_VIEW_DIRS]
    if unknown:
        raise SystemExit(f"unknown view(s): {unknown}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    robot_pose = torch.tensor([[*ROBOT_INITIAL_POS, *ROBOT_INITIAL_ROT]], dtype=torch.float32)
    door_pose = torch.tensor([[*DOOR_INITIAL_POS, *DOOR_INITIAL_ROT]], dtype=torch.float32)
    _, q0 = get_robot_constants()
    side, direction = resolve_planner_options(args.door_urdf, "auto", "auto")
    print(f"{Path(args.door_urdf).parent.name}: {direction} / {side}-side handle")

    robot_traj, door_traj, key_indices = state_machine_offline_pull_door(
        args.robot_urdf, args.door_urdf, robot_pose, door_pose, q0,
        torch.tensor([0.0, 0.0]), handle_side=side, device="cpu")

    if args.all_frames:
        # The COLLOCATED playback -- what the env actually tracks. Keyframes alone can all look
        # right while the spline between two of them swings the arm through the panel, so the
        # only way to clear the motion is to look at every frame of it.
        ri, di, _, _, _ = collocate_and_playback(robot_traj, door_traj, key_indices,
                                                 length=args.length)
        frames = list(range(0, ri.shape[0], args.stride))
        get_q = lambda k: (ri[k], di[k])  # noqa: E731
        print(f"{ri.shape[0]} collocated frames, rendering {len(frames)} (stride {args.stride})")
    else:
        # The planner's own keyframes, un-collocated: the poses it actually authored, so a bad
        # offset shows up undiluted by interpolation with its neighbours.
        n_raw = len(robot_traj)
        frames = [k for k in key_indices if 0 <= k < n_raw]
        if len(frames) != len(key_indices):
            # Guards a known off-by-one: the last key index can come back == len(robot_traj).
            print(f"note: dropped {len(key_indices) - len(frames)} out-of-range key index/indices "
                  f"(trajectory has {n_raw} frames)")
        get_q = lambda k: (robot_traj[k], door_traj[k])  # noqa: E731
        print(f"{len(frames)} keyframes")

    fk = PinocchioIKSolver(urdf_path=args.robot_urdf, ee_link_name="panda_hand",
                           controlled_joints=BASE_JOINT_NAMES + FRANKA_JOINT_NAMES,
                           reference_joint_pos=FRANKA_DEFAULT_JOINT_POS)

    # ---- scene: identical construction to play_trajectories_in_viser ----
    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/robot_root", position=robot_pose[0, :3].numpy(),
                           wxyz=robot_pose[0, 3:].numpy(), show_axes=False)
    server.scene.add_frame("/door_root", position=door_pose[0, :3].numpy(),
                           wxyz=door_pose[0, 3:].numpy(), show_axes=False)
    viser_robot = ViserUrdf(server, urdf_or_path=URDF.load(args.robot_urdf),
                            root_node_name="/robot_root", load_meshes=True)
    viser_door = ViserUrdf(server, urdf_or_path=URDF.load(args.door_urdf),
                           root_node_name="/door_root", load_meshes=True)

    print(f"open http://localhost:{args.port} and KEEP THE TAB IN FRONT ...", flush=True)
    deadline = time.time() + args.timeout
    while time.time() < deadline and not server.get_clients():
        time.sleep(0.5)
    if not server.get_clients():
        raise SystemExit("no browser connected; nothing to render from.")
    print("browser connected; letting meshes load ...", flush=True)
    time.sleep(4.0)

    from concurrent.futures import ThreadPoolExecutor
    # Generous worker count on purpose. fut.result(timeout=) stops the CALLER waiting, but the
    # worker itself stays blocked forever on a browser that never replies, so every timeout
    # permanently burns a thread. With a small pool a couple of early hangs wedge the executor and
    # every later frame then "times out" even after the browser comes back healthy -- which looks
    # exactly like a broken renderer and is not.
    pool = ThreadPoolExecutor(max_workers=32)

    written = 0
    video_frames = []
    t_start = time.time()
    for n, k in enumerate(frames):
        q_robot, q_door = get_q(k)
        viser_robot.update_cfg(q_robot.numpy())
        viser_door.update_cfg(q_door.numpy())
        tcp = _tcp_world(fk, q_robot, robot_pose)
        tag = (f"f{n:05d}_i{k:04d}_board{float(q_door[0]):.2f}"
               f"_grip{float(q_robot[GRIPPER_Q_IDX]):.3f}")

        for view in views:
            if view in SCENE_VIEWS:
                pos, look = SCENE_VIEWS[view]["position"], SCENE_VIEWS[view]["look_at"]
            else:
                d = np.asarray(HAND_VIEW_DIRS[view], dtype=float)
                pos = tuple(tcp + HAND_VIEW_DIST * d / np.linalg.norm(d))
                look = tuple(tcp)
            # Re-pick a live client each time and keep every browser round-trip inside the
            # try: a throttled tab simply never answers, and losing one frame beats hanging.
            clients = list(server.get_clients().values())
            if not clients:
                print(f"  {tag} [{view}]: no client, skipping", flush=True)
                continue
            # Try EVERY connected client, newest first, and keep the first that actually answers.
            # More than one browser can be attached (a tab you left open plus one opened here),
            # and only a VISIBLE tab renders: Chrome throttles requestAnimationFrame in background
            # tabs, so their viser render loop stops and get_render never returns even though the
            # websocket still reports "Connected". Betting on a single client picked at random is
            # why whole runs came back with every frame skipped.
            img = None
            for client in reversed(clients):
                try:
                    fut = pool.submit(_aim_and_render, client, pos, look,
                                      args.settle, args.height, args.width)
                    img = fut.result(timeout=float(args.render_timeout))
                    break
                except Exception:
                    continue
            if img is None:
                print(f"  {tag} [{view}]: no client rendered "
                      f"({len(clients)} attached, all unresponsive -- is a viser tab VISIBLE?)",
                      flush=True)
                continue
            iio.imwrite(outdir / f"{tag}_{view}.png", img)
            if args.video and view == views[0]:
                video_frames.append(img[..., :3])
            written += 1
        # Progress with an ETA: a full 1440-frame pass is long enough that "is this still moving"
        # is a real question.
        if n % 20 == 0 or n == len(frames) - 1:
            done = n + 1
            rate = (time.time() - t_start) / max(done, 1)
            print(f"  {done}/{len(frames)} {tag}  ({rate:.2f}s/frame, "
                  f"~{rate * (len(frames) - done) / 60:.1f} min left)", flush=True)

    if args.video and video_frames:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.video, np.stack(video_frames), fps=args.video_fps,
                    codec="libx264", macro_block_size=None)
        print(f"wrote {args.video} ({len(video_frames)} frames @ {args.video_fps} fps)")

    print(f"\nwrote {written} PNGs to {outdir}")
    server.stop()


if __name__ == "__main__":
    main()
