"""Attach step-aligned attention and exact base velocities to eval Viser replays."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("base_replay", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--attention-env-id", type=int, default=1)
    args = parser.parse_args()

    base = torch.load(args.base_replay, map_location="cpu", weights_only=False)
    run = base["runs"][0]
    attention_path = args.eval_dir / "attn_viz" / "attn_mass_run1.json"
    attention_by_step = {}
    if attention_path.exists():
        attention_by_step = {
            int(row["step"]): {key: float(row[key]) for key in (
                "pointcloud", "q_arm_history", "base_vel_history", "aux_handle_pos", "tracking_error"
            )}
            for row in json.loads(attention_path.read_text())
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((args.eval_dir / "viser").glob("*.pt")):
        match = re.search(r"_env(\d+)_", source.name)
        if match is None:
            continue
        env_id = int(match.group(1))
        payload = torch.load(source, map_location="cpu", weights_only=False)
        frames = payload["frames"]
        for step, frame in enumerate(frames):
            for key in (
                "base_velocity_measured_robot",
                "base_velocity_policy_input_robot",
                "base_velocity_commanded_robot",
                "base_effort_applied_estimate",
                "base_effort_computed_estimate",
            ):
                if key in run:
                    frame[key] = run[key][step, env_id].clone()
            if env_id == args.attention_env_id and step in attention_by_step:
                frame["proprio_attention"] = attention_by_step[step]
        payload["base_velocity_schema"] = base["base_velocity_schema"]
        if "base_effort_schema" in base:
            payload["base_effort_schema"] = base["base_effort_schema"]
        payload["base_damping_translation"] = 4000.0
        payload["base_damping_rotation"] = 400.0
        target = args.output_dir / f"env{env_id:03d}_damping_div3.pt"
        torch.save(payload, target)
        print(f"{target}: {len(frames)} frames, attention={env_id == args.attention_env_id}")


if __name__ == "__main__":
    main()
