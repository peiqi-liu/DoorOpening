"""Collision-check a pull-door plan against the door, frame by frame.

Uses the repo's own GlorbotCollisionChecker sphere model for the robot, and the door's link
point clouds sampled at each frame's joint angles. A door point whose signed distance to the
robot's spheres is negative is INSIDE the robot: that frame is in collision.

Checks the COLLOCATED playback (what the env actually tracks), not the planner's raw frames,
and reports per door link so panel / frame / handle contact can be told apart -- handle contact
during the grasp is expected, panel and door-frame contact is not.

    python scripts/debug/check_pull_collisions.py --doors 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from DoorOpening.constants.env_constants import (
    DOOR_INITIAL_POS, DOOR_INITIAL_ROT, ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT,
)
from DoorOpening.constants.robot_constants import (
    ALL_DOF_NAMES, DRIVEN_FINGER_JOINT_NAME, FULL_JOINT_NAMES, MIMIC_FINGER_JOINT_NAME,
)
from DoorOpening.utils.extract_pointcloud_from_articulation import sample_pointcloud_from_link_name
from DoorOpening.utils.glorbot_collision_checker import GlorbotCollisionChecker
from DoorOpening.utils.state_machine.compute_waypoint import (
    collocate_and_playback, get_robot_constants, resolve_planner_options,
)
from DoorOpening.utils.state_machine.offline_pull_door import (
    GRIPPER_Q_IDX, state_machine_offline_pull_door,
)
from isaaclab.utils.math import quat_apply, quat_apply_inverse

REPO = Path(__file__).resolve().parents[2]
ROBOT_URDF = str(REPO / "source/DoorOpening/assets/glorbot/glorbot.urdf")
DOORS = REPO / "source/DoorOpening/assets/door/PartNetv5_plusplus"
# link_0 = door frame/jamb, link_1 = panel, link_2 = handle
DOOR_LINKS = {"frame": "link_0", "panel": "link_1", "handle": "link_2"}


def check_door(door_urdf: str, length: int, margin: float, stride: int):
    robot_pose = torch.tensor([[*ROBOT_INITIAL_POS, *ROBOT_INITIAL_ROT]], dtype=torch.float32)
    door_pose = torch.tensor([[*DOOR_INITIAL_POS, *DOOR_INITIAL_ROT]], dtype=torch.float32)
    _, q0 = get_robot_constants()
    side, direction = resolve_planner_options(door_urdf, "auto", "auto")
    if direction != "pull":
        return None

    rt, dt, keys = state_machine_offline_pull_door(
        ROBOT_URDF, door_urdf, robot_pose, door_pose, q0,
        torch.tensor([0.0, 0.0]), handle_side=side, device="cpu")
    ri, di, _, _, _ = collocate_and_playback(rt, dt, keys, length=length)

    # The trajectory carries only COMMANDED joints (17); the collision model wants every DOF
    # (18). The follower finger tracks the driven one, so fill it in here.
    src = {n: k for k, n in enumerate(FULL_JOINT_NAMES)}
    cols = [src.get(n, src[DRIVEN_FINGER_JOINT_NAME]) if n != MIMIC_FINGER_JOINT_NAME
            else src[DRIVEN_FINGER_JOINT_NAME] for n in ALL_DOF_NAMES]
    ri = ri[:, cols]
    checker = GlorbotCollisionChecker(ROBOT_URDF, device="cpu", input_joint_names=ALL_DOF_NAMES)
    idxs = list(range(0, ri.shape[0], stride))
    hits = {k: [] for k in DOOR_LINKS}

    for i in idxs:
        q = ri[i : i + 1]
        # GlorbotCollisionChecker.torch_spheres() returns sphere centers in the robot's OWN
        # root-link frame (it never applies robot_pose) -- e.g. the tidybot2_base_link spheres
        # come back as the literal local offsets from the collision model, [0.17, 0.15, 0.17]
        # etc, regardless of where the robot is actually spawned. ROBOT_INITIAL_POS is (1, 0, 0),
        # a full metre away from DOOR_INITIAL_POS -- comparing those local-frame spheres directly
        # against the door's WORLD-frame point cloud (as this used to) manufactures a bogus,
        # near-constant ~-0.15 m "penetration" on every single frame of every door, from frame 0
        # (the robot's un-moved home pose) onward, which is exactly what a plumbing bug looks
        # like: transform the door cloud INTO the robot's frame instead, so both sides agree.
        spheres = checker.torch_spheres(q)
        for label, link in DOOR_LINKS.items():
            pc = sample_pointcloud_from_link_name(door_urdf, di[i : i + 1], link, device="cpu")
            pc_w = quat_apply(door_pose[..., 3:], pc) + door_pose[..., :3]
            pc_robot = quat_apply_inverse(robot_pose[..., 3:], pc_w - robot_pose[..., :3])
            sdf = spheres.sdf(pc_robot.to(torch.float32))
            worst = float(sdf.min())
            if worst < -margin:
                hits[label].append((i, float(di[i][0]), float(ri[i][GRIPPER_Q_IDX]), worst))
    return dict(side=side, frames=len(idxs), hits=hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doors", type=int, default=5)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.005,
                    help="penetration depth (m) past the sphere surface before it counts")
    args = ap.parse_args()

    urdfs = sorted(DOORS.glob("*/mobility.urdf"))[: args.doors]
    print(f"{'door':>22} {'frames':>7} {'frame':>8} {'panel':>8} {'handle':>8}   worst penetration")
    totals = {k: 0 for k in DOOR_LINKS}
    for u in urdfs:
        res = check_door(str(u), args.length, args.margin, args.stride)
        if res is None:
            continue
        h = res["hits"]
        worst = min([x[3] for v in h.values() for x in v], default=0.0)
        for k in h:
            totals[k] += len(h[k])
        print(f"{u.parent.name:>22} {res['frames']:>7} "
              f"{len(h['frame']):>8} {len(h['panel']):>8} {len(h['handle']):>8}   {worst:+.3f} m")
        for k in ("frame", "panel"):
            if h[k]:
                b = [x[1] for x in h[k]]
                print(f"{'':>22}   {k} hits at board {min(b):.2f}..{max(b):.2f} rad, "
                      f"deepest {min(x[3] for x in h[k]):+.3f} m")
    print(f"\ntotals over {len(urdfs)} doors: " +
          "  ".join(f"{k}={v}" for k, v in totals.items()))
    print("handle contact during the grasp is expected; frame/panel contact is not.")


if __name__ == "__main__":
    main()
