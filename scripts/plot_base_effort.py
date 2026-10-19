"""Plot per-environment base actuator effort from a saved distillation evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.replay, map_location="cpu", weights_only=False)
    run = payload["runs"][0]
    hz = float(payload["control_frequency_hz"])
    active = run["active_mask"].numpy()
    applied = run["base_effort_applied_estimate"].numpy()
    computed = run["base_effort_computed_estimate"].numpy()
    measured_velocity = run["base_velocity_measured_robot"].numpy()
    commanded_velocity = run["base_velocity_commanded_robot"].numpy()
    axes = (("base_x_force", "Base x force", "N"), ("base_y_force", "Base y force", "N"),
            ("base_yaw_torque", "Base yaw torque", "N·m"))

    for env_id in range(active.shape[1]):
        n = int(active[:, env_id].sum())
        time_s = np.arange(n) / hz
        csv_path = args.output_dir / f"env{env_id:03d}_base_effort.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "step", "time_s", "applied_x_force_n", "applied_y_force_n", "applied_yaw_torque_nm",
                "computed_x_force_n", "computed_y_force_n", "computed_yaw_torque_nm",
                "measured_vx_m_s", "measured_vy_m_s", "measured_yaw_rad_s",
                "commanded_vx_m_s", "commanded_vy_m_s", "commanded_yaw_rad_s",
            ])
            for step in range(n):
                writer.writerow([
                    step, time_s[step], *applied[step, env_id], *computed[step, env_id],
                    *measured_velocity[step, env_id], *commanded_velocity[step, env_id],
                ])

        fig, panels = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        for axis_idx, (key, label, unit) in enumerate(axes):
            ax = panels[axis_idx]
            ax.plot(time_s, computed[:n, env_id, axis_idx], color="#a0a0a0", linewidth=1,
                    alpha=0.65, label="PD estimate before clipping")
            ax.plot(time_s, applied[:n, env_id, axis_idx], color=("#e07a5f", "#3d9970", "#8267be")[axis_idx],
                    linewidth=1.5, label="Clipped applied estimate")
            ax.axhline(1000, color="#aa3333", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axhline(-1000, color="#aa3333", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylabel(f"{label} ({unit})")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        panels[-1].set_xlabel("Time (s), 15 Hz policy steps")
        fig.suptitle(
            f"Env {env_id}: estimated base actuator effort, damping x/y=4000 and yaw=400\n"
            "IsaacLab implicit-actuator estimate at last physics substep; not direct PhysX drive force"
        )
        figure_path = args.output_dir / f"env{env_id:03d}_base_effort.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        print(f"env {env_id}: {n} active steps -> {figure_path}, {csv_path}")


if __name__ == "__main__":
    main()
