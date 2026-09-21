"""Numeric check of the pull-door plan: is the gripper actually ON the handle?

Runs the same planner compute_waypoint.py runs, with the same inputs, then does FORWARD
kinematics on every planned frame and compares where the gripper ended up against where the
handle actually is. This answers by measurement the questions a 3D view only hints at:

  * grip distance -- how far the grasp center (TCP) is from the handle, per frame. If the
    gripper "never touches the handle" this column says so, and says from which frame.
  * jaw vs bar     -- angle between the jaw-travel axis and vertical. The jaws must close
    ACROSS a horizontal lever bar, so this should stay near 0 deg while gripping.
  * approach       -- world direction the gripper points, to check it keeps facing the door.
  * ik err         -- FK(planned q) vs the pose the planner asked for. Large values mean
    solve_ik did not converge and returned a best-effort pose.

Usage:
    python scripts/debug/dump_pull_geometry.py --door-urdf <path/to/mobility.urdf>
    python scripts/debug/dump_pull_geometry.py            # picks the first door in the folder

Writes a JSON next to the door with every per-frame quantity, for plotting or diffing runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from DoorOpening.constants.env_constants import (
    DOOR_INITIAL_POS,
    DOOR_INITIAL_ROT,
    ROBOT_INITIAL_POS,
    ROBOT_INITIAL_ROT,
)
from DoorOpening.constants.robot_constants import (
    BASE_JOINT_NAMES,
    FRANKA_DEFAULT_JOINT_POS,
    FRANKA_JOINT_NAMES,
)
from DoorOpening.utils.pose_utils import base_to_world_frame
from DoorOpening.utils.state_machine.api import get_hinge_pos
from DoorOpening.utils.state_machine.compute_waypoint import (
    collocate_and_playback,
    get_robot_constants,
    resolve_planner_options,
)
from DoorOpening.utils.state_machine.offline_pull_door import (
    GRIPPER_Q_IDX,
    GRIPPER_TCP_OFFSET,
    state_machine_offline_pull_door,
)
from DoorOpening.utils.state_machine.pin import PinocchioIKSolver

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_URDF = REPO / "source/DoorOpening/assets/glorbot/glorbot.urdf"
DEFAULT_DOOR_FOLDER = REPO / "source/DoorOpening/assets/door/PartNetv5_plusplus"


def _quat_xyzw_to_wxyz(q):
    return torch.tensor([[q[3], q[0], q[1], q[2]]], dtype=torch.float32)


def _rotate(quat_wxyz: torch.Tensor, vec) -> np.ndarray:
    from isaaclab.utils.math import quat_apply

    v = torch.tensor([vec], dtype=torch.float32)
    return quat_apply(quat_wxyz, v).squeeze(0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--door-urdf", type=str, default=None)
    ap.add_argument("--robot-urdf", type=str, default=str(DEFAULT_ROBOT_URDF))
    ap.add_argument("--handle-side", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--raw", action="store_true",
                    help="measure the planner's own frames instead of the collocated playback")
    ap.add_argument("--length", type=int, default=400, help="collocated playback length")
    args = ap.parse_args()

    door_urdf = args.door_urdf
    if door_urdf is None:
        candidates = sorted(p for p in DEFAULT_DOOR_FOLDER.glob("*/mobility.urdf"))
        if not candidates:
            raise SystemExit(f"no doors under {DEFAULT_DOOR_FOLDER}; pass --door-urdf")
        door_urdf = str(candidates[0])

    # ---- exactly the inputs compute_waypoint.py builds -------------------------------------
    robot_initial_pose = torch.tensor([[*ROBOT_INITIAL_POS, *ROBOT_INITIAL_ROT]], dtype=torch.float32)
    door_initial_pose = torch.tensor([[*DOOR_INITIAL_POS, *DOOR_INITIAL_ROT]], dtype=torch.float32)
    _, robot_initial_q = get_robot_constants()
    door_initial_q = torch.tensor([0.0, 0.0])

    handle_side, opening_direction = resolve_planner_options(door_urdf, args.handle_side, "auto")
    print(f"door           : {door_urdf}")
    print(f"planner        : {opening_direction} / {handle_side}-side handle")
    if opening_direction != "pull":
        raise SystemExit(f"this dump is for PULL doors; that one resolves to '{opening_direction}'")

    robot_traj, door_traj, key_indices = state_machine_offline_pull_door(
        args.robot_urdf,
        door_urdf,
        robot_initial_pose,
        door_initial_pose,
        robot_initial_q,
        door_initial_q,
        handle_side=handle_side,
        device="cpu",
    )
    print(f"planner frames : {len(robot_traj)}  (keyframes at {key_indices})")

    if not args.raw:
        # The env never replays the planner's frames -- collocate_and_playback splines through
        # them (chord-length parameterised, bc_type="clamped") and THAT is what gets tracked.
        # Measure the splined motion, so overshoot between knots shows up.
        robot_i, door_i, _, _, key_indices = collocate_and_playback(
            robot_traj, door_traj, key_indices, length=args.length
        )
        robot_traj = [robot_i[i] for i in range(robot_i.shape[0])]
        door_traj = [door_i[i] for i in range(door_i.shape[0])]
        print(f"collocated     : {len(robot_traj)} frames  (keyframes at {key_indices})")

    # ---- where the handle actually is, per frame -------------------------------------------
    door_stack = torch.stack(door_traj)
    handle_w = get_hinge_pos(door_urdf, door_initial_pose, door_stack).to(torch.float32)

    # ---- where the gripper actually ended up, per frame -------------------------------------
    fk = PinocchioIKSolver(
        urdf_path=args.robot_urdf,
        ee_link_name="panda_hand",
        controlled_joints=BASE_JOINT_NAMES + FRANKA_JOINT_NAMES,
        reference_joint_pos=FRANKA_DEFAULT_JOINT_POS,
    )
    base_pos = robot_initial_pose[:, :3]
    base_quat = robot_initial_pose[:, 3:]

    # door swing sign, read from the planner so the diagnostic cannot disagree with it
    import DoorOpening.utils.state_machine.offline_pull_door as _pull
    _src = Path(_pull.__file__).read_text()
    _fn = "state_machine_offline_right_pull_door" if handle_side == "right" else "state_machine_offline_left_pull_door"
    _b = _src[_src.index(f"def {_fn}"):]
    import re as _re
    _sign = float(_re.search(r"pull_swing_sign = (-?[\d.]+)", _b).group(1))
    def swing_of(board):
        return _sign * float(board)

    rows = []
    for i, q in enumerate(robot_traj):
        pos_b, quat_b_xyzw = fk.compute_fk(q[:10].numpy())
        quat_b = _quat_xyzw_to_wxyz(quat_b_xyzw)
        hand_w, hand_quat_w = base_to_world_frame(
            base_pos, base_quat, torch.tensor([pos_b], dtype=torch.float32), quat_b
        )
        hand_w = hand_w.squeeze(0).numpy()
        approach = _rotate(hand_quat_w, [0.0, 0.0, 1.0])
        jaw = _rotate(hand_quat_w, [0.0, 1.0, 0.0])
        tcp = hand_w + GRIPPER_TCP_OFFSET * approach

        handle = handle_w[i].numpy()
        delta = tcp - handle
        rows.append(
            dict(
                frame=i,
                key=i in key_indices,
                board=float(door_traj[i][0]),
                lever=float(door_traj[i][1]),
                grip=float(q[GRIPPER_Q_IDX]),
                tcp=tcp.tolist(),
                handle=handle.tolist(),
                dist=float(np.linalg.norm(delta)),
                d_xyz=delta.tolist(),
                # jaws must close ACROSS a horizontal bar -> jaw axis near vertical while gripping
                jaw_from_vertical_deg=float(math.degrees(math.acos(min(1.0, abs(jaw[2]))))),
                approach=approach.tolist(),
                # panda_joint7 is the wrist roll about the tool axis -- the joint that has to
                # absorb the door's swing, and the one that saturates first.
                # "pointing at the handle": angle between the gripper approach axis and the
                # door's OUTWARD NORMAL, which rotates with the panel. Position error can look
                # fine while this is 50 deg off, i.e. the jaws twisted off the bar.
                approach_vs_door_normal_deg=float(
                    math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(
                        approach,
                        [-math.cos(swing_of(door_traj[i][0])), -math.sin(swing_of(door_traj[i][0])), 0.0],
                    ))))))
                ),
                wrist7=float(q[9]),
                arm=[float(v) for v in q[3:10]],
            )
        )

    # ---- report ------------------------------------------------------------------------------
    print()
    print(f"{'frm':>4} {'k':>1} {'board':>6} {'lever':>6} {'grip':>5} "
          f"{'dist':>7} {'dx':>7} {'dy':>7} {'dz':>7} {'jaw°':>5}  approach")
    for r in rows:
        print(
            f"{r['frame']:>4} {'K' if r['key'] else ' ':>1} {r['board']:>6.3f} {r['lever']:>6.3f} "
            f"{r['grip']:>5.3f} {r['dist']:>7.3f} "
            f"{r['d_xyz'][0]:>7.3f} {r['d_xyz'][1]:>7.3f} {r['d_xyz'][2]:>7.3f} "
            f"{r['jaw_from_vertical_deg']:>5.1f}  "
            f"({r['approach'][0]:+.2f},{r['approach'][1]:+.2f},{r['approach'][2]:+.2f})"
        )

    def _bucket_report(rows_subset, label):
        if not rows_subset:
            return
        d = np.array([r["dist"] for r in rows_subset])
        j = np.array([r["jaw_from_vertical_deg"] for r in rows_subset])
        print(f"   {label:>18}: n={len(rows_subset):>4}  dist mean {d.mean():.3f} max {d.max():.3f}"
              f"   jaw mean {j.mean():4.1f} max {j.max():4.1f}")

    closed = [r for r in rows if r["grip"] < 0.02]
    if closed:
        print()
        print("holding the handle, bucketed by how far the door has opened:")
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.6]
        for lo, hi in zip(edges[:-1], edges[1:]):
            _bucket_report([r for r in closed if lo <= r["board"] < hi], f"{lo:.1f}-{hi:.1f} rad")
        # where does it first lose the bar?
        LOSS = 0.05
        lost = [r for r in closed if r["dist"] > LOSS]
        if lost:
            first = min(lost, key=lambda r: r["board"])
            print(f"   first exceeds {LOSS} m at board={first['board']:.2f} rad "
                  f"({math.degrees(first['board']):.0f} deg), dist={first['dist']:.3f}")
        else:
            print(f"   never exceeds {LOSS} m while closed")
        n = np.array([r["approach_vs_door_normal_deg"] for r in closed])
        print(f"   approach vs door normal: mean {n.mean():4.1f}  max {n.max():4.1f} deg "
              f"(0 = gripper square to the door face, i.e. still pointing at the handle)")
        w = np.array([r["wrist7"] for r in closed])
        print(f"   panda_joint7 over the hold: {w.min():+.2f} .. {w.max():+.2f} rad "
              f"(range {w.max()-w.min():.2f}, limit +-2.97)")

    gripping = [r for r in rows if r["grip"] < 0.02]
    if gripping:
        d = np.array([r["dist"] for r in gripping])
        print()
        print(f"while the gripper is CLOSED ({len(gripping)} frames):")
        print(f"   TCP-to-handle distance  min {d.min():.3f}  mean {d.mean():.3f}  max {d.max():.3f} m")
        j = np.array([r["jaw_from_vertical_deg"] for r in gripping])
        print(f"   jaw axis off vertical   min {j.min():.1f}  mean {j.mean():.1f}  max {j.max():.1f} deg")
        worst = max(gripping, key=lambda r: r["dist"])
        print(f"   worst frame {worst['frame']} at board={worst['board']:.2f}: "
              f"{worst['dist']:.3f} m away, d_xyz={np.round(worst['d_xyz'], 3).tolist()}")
        print("   (a lever bar is ~0.02 m across; anything past ~0.05 m is not holding it)")

    out = args.out or str(Path(door_urdf).with_name("pull_geometry.json"))
    Path(out).write_text(json.dumps(dict(door=door_urdf, handle_side=handle_side,
                                         key_indices=list(key_indices), rows=rows), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
