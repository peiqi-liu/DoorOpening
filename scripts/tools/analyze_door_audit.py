#!/usr/bin/env python3
"""Explain which door parameters separate the policy's failures from its successes.

Consumes the JSON written by `scripts/rl_games/play.py --door-audit-out`, joins each episode row
against that door's `variant_meta.json`, and reports every geometry/dynamics parameter ranked by how
strongly it discriminates failure from success.

    python scripts/rl_games/play.py --task DooropeningMulti --num_envs 512 \
        --checkpoint <ckpt> --headless --door-audit-out logs/door_audit.json
    python scripts/tools/analyze_door_audit.py logs/door_audit.json --csv logs/door_audit.csv

Ranking uses a rank-biserial correlation (equivalently, normalized Mann-Whitney U): +1 means the
parameter is always higher on failing doors, -1 always higher on succeeding doors, 0 no separation.
It is rank-based, so it does not assume normality and is not distorted by the long-tailed frame /
handle distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


# Door parameters pulled from variant_meta.json. (label, dotted path into the meta dict).
META_FIELDS = [
    ("panel_width", "panel.width_m"),
    ("panel_height", "panel.height_m"),
    ("panel_thickness", "panel.thickness_m"),
    ("frame_depth", "frame.depth_m"),
    ("frame_post_width", "frame.post_width_m"),
    ("frame_head_height", "frame.head_height_m"),
    ("frame_clearance", "frame.clearance_m"),
    ("handle_height", "actual_properties.handle_height_m"),
    ("handle_edge_distance", "actual_properties.handle_to_edge_distance_m"),
    ("lever_length", "handle.lever_length_m"),
    ("stem_length", "handle.stem_length_m"),
    ("lever_thickness", "handle.lever_thickness_m"),
    ("handle_radius", "handle.radius_m"),
    ("return_length", "handle.return_length_m"),
    ("bump_protrusion", "handle.bump_protrusion_m"),
    ("bump_height", "handle.bump_height_m"),
    ("bump_width", "handle.bump_width_m"),
]

# Parameters recorded per EPISODE by the audit (ADR draws, so they vary per rollout, not per door).
EPISODE_FIELDS = [
    "hinge_max_rad",
    "latch_max_rad",
    "dist_past_door_max_m",
    "ref_frame_reached",
    "is_push",
    "board_stiffness",
    "board_damping",
    "hinge_stiffness",
    "hinge_damping",
    "hinge_effort_limit",
    "episode_steps",
    "x5_door_collision",
    "franka_box_collision",
    "killed_robot_drift",
    "killed_door_drift",
]

# Recorded per episode by the audit, not read from variant_meta.
EPISODE_CATEGORICAL_FIELDS = ["term_reason"]

CATEGORICAL_FIELDS = [
    ("opening_direction", "actual_properties.opening_direction"),
    ("handle_side", "actual_properties.handle_side"),
    ("handle_type", "handle.type"),
    ("has_bump", "handle.has_bump"),
]


def dig(meta, dotted):
    cur = meta
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def derived(meta):
    """Parameters that are not stored directly but drive graspability."""
    out = {}
    handle = meta.get("handle") or {}
    panel = meta.get("panel") or {}
    frame = meta.get("frame") or {}
    stem = handle.get("stem_length_m")
    lever_half = handle.get("lever_thickness_m")
    bump = handle.get("bump_protrusion_m", 0.0) if handle.get("has_bump") else 0.0
    if stem is not None and lever_half is not None:
        # Clear finger space under the lever, measured from whatever surface is nearest (plate or panel).
        out["clear_finger_gap"] = stem - lever_half - (bump or 0.0)
    depth, thick = frame.get("depth_m"), panel.get("thickness_m")
    if depth is not None and thick is not None:
        # How far the casing stands proud of each panel face (frame box is panel-centered).
        proud = 0.5 * (max(depth, thick + 0.01) - thick)
        out["frame_proud_per_face"] = proud
        if stem is not None and lever_half is not None:
            # >0 means the casing sticks out further than the lever, i.e. handle recessed in the reveal.
            out["reveal_minus_handle"] = proud - (stem - lever_half)
    return out


def rank_biserial(fail_vals, ok_vals):
    """Normalized Mann-Whitney U in [-1, 1]: +1 = always higher on failures."""
    n_f, n_o = len(fail_vals), len(ok_vals)
    if n_f == 0 or n_o == 0:
        return None
    pooled = sorted([(v, 0) for v in fail_vals] + [(v, 1) for v in ok_vals])
    ranks = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks.setdefault(k, avg)
        i = j + 1
    rank_sum_fail = sum(ranks[k] for k, (_, grp) in enumerate(pooled) if grp == 0)
    u_fail = rank_sum_fail - n_f * (n_f + 1) / 2.0
    return 2.0 * u_fail / (n_f * n_o) - 1.0


def summarize(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"n": n, "min": s[0], "p25": s[max(0, n // 4)], "median": median,
            "p75": s[min(n - 1, 3 * n // 4)], "max": s[-1], "mean": mean}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_json", type=Path)
    ap.add_argument("--csv", type=Path, default=None, help="Also write the joined per-door table here.")
    ap.add_argument("--top", type=int, default=12, help="How many discriminating parameters to print.")
    ap.add_argument("--list-failures", type=int, default=20, help="Print this many worst doors individually.")
    args = ap.parse_args()

    payload = json.loads(args.audit_json.read_text())
    rows = payload["rows"]
    if not rows:
        raise SystemExit("No rows in audit file.")

    meta_cache = {}
    records = []
    for row in rows:
        urdf = Path(row["asset_path"])
        meta_path = urdf.parent / "variant_meta.json"
        if meta_path not in meta_cache:
            meta_cache[meta_path] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta = meta_cache[meta_path]
        rec = {"asset_idx": row["asset_idx"], "door": urdf.parent.name,
               "family": urdf.parent.parent.name, "success": row["success"]}
        for key in EPISODE_FIELDS:
            if key in row:
                rec[key] = row[key]
        for key in EPISODE_CATEGORICAL_FIELDS:
            if key in row:
                rec[key] = row[key]
        for label, path in META_FIELDS:
            v = dig(meta, path)
            if isinstance(v, (int, float)):
                rec[label] = float(v)
        rec.update(derived(meta))
        for label, path in CATEGORICAL_FIELDS:
            v = dig(meta, path)
            if v is not None:
                rec[label] = v
        records.append(rec)

    fails = [r for r in records if r["success"] <= 0.5]
    oks = [r for r in records if r["success"] > 0.5]
    print(f"\n{len(records)} episodes over {len({r['door'] for r in records})} doors: "
          f"{len(oks)} success ({100.0*len(oks)/len(records):.1f}%), {len(fails)} fail.\n")
    if not fails or not oks:
        print("All episodes fell on one side; nothing to contrast.")
        return

    numeric = [k for k in records[0] if isinstance(records[0].get(k), float) and k != "success"]
    numeric = [k for k in numeric if any(k in r for r in records)]

    scored = []
    for key in numeric:
        f = [r[key] for r in fails if key in r]
        o = [r[key] for r in oks if key in r]
        rb = rank_biserial(f, o)
        if rb is None:
            continue
        scored.append((abs(rb), rb, key, summarize(f), summarize(o)))
    scored.sort(reverse=True)

    print("Parameters ranked by how strongly they separate FAIL from SUCCESS")
    print("(effect > 0 -> higher values fail; effect < 0 -> higher values succeed)\n")
    hdr = f"{'parameter':24s} {'effect':>7s} | {'FAIL median':>11s} {'[p25, p75]':>18s} | {'OK median':>10s} {'[p25, p75]':>18s}"
    print(hdr)
    print("-" * len(hdr))
    for _, rb, key, sf, so in scored[: args.top]:
        print(f"{key:24s} {rb:+7.3f} | {sf['median']:11.4f} [{sf['p25']:7.4f},{sf['p75']:7.4f}] | "
              f"{so['median']:10.4f} [{so['p25']:7.4f},{so['p75']:7.4f}]")

    for label in EPISODE_CATEGORICAL_FIELDS + [lbl for lbl, _ in CATEGORICAL_FIELDS]:
        vals = {r.get(label) for r in records if label in r}
        if len(vals) < 2:
            continue
        print(f"\n{label}:")
        for v in sorted(vals, key=str):
            nf = sum(1 for r in fails if r.get(label) == v)
            no = sum(1 for r in oks if r.get(label) == v)
            tot = nf + no
            if tot:
                print(f"   {str(v):12s} success {no:4d}/{tot:4d} = {100.0*no/tot:5.1f}%")

    if args.list_failures:
        print(f"\nWorst {args.list_failures} failing doors:")
        cols = [c for c in ("clear_finger_gap", "lever_length", "frame_proud_per_face",
                            "reveal_minus_handle", "handle_edge_distance", "hinge_stiffness",
                            "hinge_damping") if c in records[0] or any(c in r for r in records)]
        print(f"   {'door':26s} {'family':22s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
        for r in fails[: args.list_failures]:
            print(f"   {r['door']:26s} {r['family']:22s} " +
                  " ".join(f"{r.get(c, float('nan')):14.4f}" for c in cols))

    if args.csv:
        import csv as _csv

        keys = sorted({k for r in records for k in r})
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(records)
        print(f"\nWrote joined per-episode table to {args.csv}")


if __name__ == "__main__":
    main()
