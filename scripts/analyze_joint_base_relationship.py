from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualizations"


def onset(signal: np.ndarray, smooth: int = 7) -> int:
    x = np.asarray(signal, dtype=float)
    if len(x) < 12:
        return 0
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        x = np.convolve(x, kernel, mode="same")
        x[: smooth // 2] = x[smooth // 2]
        x[-smooth // 2 :] = x[-smooth // 2 - 1]
    baseline = np.median(x[: min(20, len(x) // 4)])
    scale = np.percentile(np.abs(x - baseline), 90)
    if scale < 1e-8:
        return 0
    # First sustained departure from the initial plateau. This is intentionally
    # permissive: failed episodes remain useful as long as the signal moves.
    for i in range(3, len(x) - 5):
        delta = abs(x[i] - baseline)
        if delta > 0.10 * scale and np.mean(np.abs(x[i : i + 5] - baseline)) > 0.07 * scale:
            return i
    return int(np.argmax(np.abs(x - baseline)))


def load_sim_episode(path: Path, batch: str, env: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    frames = payload["frames"]
    aux = np.stack([np.asarray(f["aux_prediction_world"], dtype=float) for f in frames])
    joint = np.stack([np.asarray(f["door_joint_prediction"], dtype=float) for f in frames])
    action = np.stack([np.asarray(f["policy_action"], dtype=float)[:3] for f in frames])
    dt = float(payload.get("frame_dt", 1 / 15))
    handle_motion = np.linalg.norm(aux - aux[0], axis=1)
    # Config/source order is [hinge (panel), latch (handle)] in radians.
    panel_pred = joint[:, 0]
    handle_pred = joint[:, 1]
    h_on = onset(handle_motion)
    p_on = onset(panel_pred)
    if p_on <= h_on + 4:
        p_on = min(len(frames) - 1, max(h_on + 20, int(0.55 * len(frames))))
    lo, hi = h_on, min(len(frames), max(p_on + 1, h_on + 8))
    return {
        "id": f"{batch}/env{env:03d}",
        "batch": batch,
        "env": env,
        "n": len(frames),
        "dt": dt,
        "handle_onset": h_on,
        "panel_onset": p_on,
        "handle_motion": handle_motion[lo:hi].tolist(),
        "panel_prediction_abs": np.abs(panel_pred[lo:hi]).tolist(),
        "handle_prediction_abs": np.abs(handle_pred[lo:hi]).tolist(),
        "base_command_abs": np.abs(action[lo:hi]).tolist(),
        "initial_joint_prediction": joint[0].tolist(),
        "initial_joint_prediction_abs": np.abs(joint[0]).tolist(),
        "full_handle_joint_abs": np.abs(handle_pred).tolist(),
        "full_panel_joint_abs": np.abs(panel_pred).tolist(),
    }


def main() -> None:
    sim = []
    for batch in [
        "04-30-46_doorjoint",
        "04-47-47_doorjoint",
        "05-03-26_doorjoint",
    ]:
        d = ROOT / "runs" / f"DoorOpening-Distillation-Eval_2026-10-11-{batch[:8]}"
        for env, p in enumerate(sorted((d / "viser").glob("*.pt"))):
            sim.append(load_sim_episode(p, batch, env))

    real_csv = ROOT / "runs/vision5_base_velocity_replay_20261011/base_velocity_compare_sim080_120_real090_160.csv"
    import csv

    with real_csv.open() as f:
        rows = list(csv.DictReader(f))
    real = {
        "source": str(real_csv),
        "sim_step": [int(r["sim_step"]) for r in rows],
        "real_step_coordinate": [float(r["real_step_coordinate"]) for r in rows],
        "measured": [[float(r[f"real_measured_{c}_interp"]) for c in ("vx", "vy", "yaw_rate")] for r in rows],
        "commanded": [[float(r[f"real_commanded_{c}_interp"]) for c in ("vx", "vy", "yaw_rate")] for r in rows],
        "sim_commanded": [[float(r[f"sim_commanded_{c}"]) for c in ("vx", "vy", "yaw_rate")] for r in rows],
    }
    real["measured_abs"] = [[abs(float(x)) for x in v] for v in real["measured"]]
    real["commanded_abs"] = [[abs(float(x)) for x in v] for v in real["commanded"]]
    data = {
        "sim": sim,
        "real_status": "Glorbo4 PT recordings could not be read: SSH at 128.2.213.13 timed out; no local real prediction PT was found.",
        "notes": {
            "joint_order": "door_joint_prediction[0]=hinge/panel angle, [1]=latch/handle angle; radians",
            "base_command": "policy_action[:3], normalized robot-frame [vx, vy, yaw_rate]",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "joint-base-relationship-data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(json.dumps({"sim_episodes": len(sim), "real_rows": len(rows), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
