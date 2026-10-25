"""Measure free-space base velocity tracking with the current door environment.

Each trial applies a constant physical base velocity request through the normal
environment action path. This intentionally avoids loading a policy checkpoint.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--settle-step", type=int, default=27)
parser.add_argument("--base-action-scale", type=float, default=4.0)
parser.add_argument("--translation-request", type=float, default=0.1, help="m/s")
parser.add_argument("--yaw-request", type=float, default=0.1, help="rad/s")
parser.add_argument("--seed", type=int, default=85000)
parser.add_argument("--controller", choices=("position", "velocity", "native-velocity"), default="position")
parser.add_argument("--limit-acceleration", action="store_true", help="Apply deployed 0.2 m/s² and 0.79 rad/s² limits")
parser.add_argument("--trace-json", type=Path, help="Replay recorded robot-frame raw base actions at their physical scale")
parser.add_argument("--door-families", default="ScratchDoors_left_pull_current_20261023")
parser.add_argument("--translation-k", type=float, help="Fixed position-drive stiffness in N/m")
parser.add_argument("--translation-d", type=float, help="Fixed position-drive damping in Ns/m")
parser.add_argument("--yaw-k", type=float, help="Fixed position-drive stiffness in Nm/rad")
parser.add_argument("--yaw-d", type=float, help="Fixed position-drive damping in Nms/rad")
parser.add_argument("--action-latency-steps", type=int, help="Override sampled target latency for a controlled probe")
parser.add_argument("--position-target-no-dt", action="store_true", help="Interpret scaled base action as a direct position displacement")
parser.add_argument("--trace-raw-actions", action="store_true", help="Send recorded raw policy actions through the configured base_action_scale")
parser.add_argument("--output-dir", type=Path, default=Path("runs/base_velocity_probe_20261019"))
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
os.environ["DOOROPENING_MULTI_DOOR_FAMILIES"] = args.door_families
os.environ["DOOROPENING_MULTI_DOOR_ASSET_LIMIT"] = str(args.num_envs)
sys.argv = [sys.argv[0], *hydra_args]
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import DoorOpening.tasks  # noqa: F401
from DoorOpening.assets.cache_utils import preconvert_shared_urdf_assets


@hydra_task_config("DooropeningMulti", "rl_games_cfg_entry_point")
def main(env_cfg, _agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.use_motion_ref = True
    env_cfg.base_action_scale = args.base_action_scale
    env_cfg.base_control_mode = "velocity" if args.controller == "native-velocity" else "position"
    gains = (args.translation_k, args.translation_d, args.yaw_k, args.yaw_d)
    if any(value is not None for value in gains):
        if any(value is None or value <= 0 for value in gains):
            raise ValueError("All four fixed position-drive gains must be positive and specified together")
        params = env_cfg.events.base_impedance_correlated.params
        params["translation_damping_range"] = (args.translation_d, args.translation_d)
        params["translation_k_over_d_range"] = (args.translation_k / (1.5 * args.translation_d),) * 2
        params["yaw_damping_range"] = (args.yaw_d, args.yaw_d)
        params["yaw_k_over_d_range"] = (args.yaw_k / (1.5 * args.yaw_d),) * 2
    preconvert_shared_urdf_assets(door_configs=env_cfg.door_cfg, verbose=False)
    env = gym.make("DooropeningMulti", cfg=env_cfg)
    base_env = env.unwrapped
    base_env.early_stopping = False
    base_env.ref_motion_lib.reset_from_start = True
    device = base_env.device
    if args.position_target_no_dt:
        if args.controller != "position":
            raise ValueError("--position-target-no-dt requires --controller position")

        def base_target_without_dt(scaled_base_actions):
            base_stop = base_env._target_base_xy_slice.stop
            q_base = base_env.robot.data.joint_pos[:, base_env._robot_base_dof_idx]
            return torch.clamp(
                q_base + scaled_base_actions,
                base_env.robot_dof_lower_limits[:base_stop],
                base_env.robot_dof_upper_limits[:base_stop],
            )

        base_env.base_target_from_scaled_actions = base_target_without_dt
    joint_ids, joint_names = base_env.robot.find_joints(
        ["base_x_joint", "base_y_joint", "base_rotation_joint"], preserve_order=True
    )
    joint_ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
    joint_names = list(joint_names)
    indices = {name: joint_names.index(name) for name in joint_names}
    if args.controller == "velocity":
        # Experimental base-only controller: preserve the environment action semantics,
        # its sampled action latency, and all other joints. A zero-stiffness implicit
        # drive then tracks a velocity target with the randomized damping as its servo gain.
        original_pre_physics_step = base_env._pre_physics_step
        original_apply_action = base_env._apply_action
        velocity_history = torch.zeros(
            (args.num_envs, base_env._max_action_latency, base_env.num_base_joints), device=device
        )
        velocity_target = torch.zeros((args.num_envs, base_env.num_base_joints), device=device)
        speed_limits = torch.tensor((0.7, 0.2, 0.2), device=device)
        accel_limits = torch.tensor((0.79, 0.2, 0.2), device=device)
        env_index = torch.arange(args.num_envs, device=device)

        def velocity_pre_physics_step(actions):
            nonlocal velocity_history, velocity_target
            original_pre_physics_step(actions)
            desired = base_env.scaled_actions[:, : base_env.num_base_joints].clamp(-speed_limits, speed_limits)
            lag_index = (base_env._action_latency_buf - 1).clamp(0, base_env._max_action_latency - 1)
            delayed = velocity_history[env_index, lag_index]
            velocity_history = torch.cat((desired.unsqueeze(1), velocity_history[:, :-1]), dim=1)
            if args.limit_acceleration:
                max_change = accel_limits * float(base_env.dt)
                velocity_target = velocity_target + (delayed - velocity_target).clamp(-max_change, max_change)
            else:
                velocity_target = delayed

        def velocity_apply_action():
            original_apply_action()
            base_env.robot.set_joint_velocity_target(velocity_target, joint_ids=base_env._robot_base_dof_idx)

        base_env._pre_physics_step = velocity_pre_physics_step
        base_env._apply_action = velocity_apply_action
    trace = json.loads(args.trace_json.read_text()) if args.trace_json else None
    trials = (("recorded", -1, None, None),) if trace else (
        ("forward", 1, "base_x_joint", args.translation_request),
        ("lateral", 2, "base_y_joint", args.translation_request),
        ("yaw", 0, "base_rotation_joint", args.yaw_request),
    )
    rows = []
    summary = {
        "seed": args.seed,
        "base_action_scale": args.base_action_scale,
        "controller": args.controller,
        "limit_acceleration": args.limit_acceleration,
        "trace_json": str(args.trace_json) if args.trace_json else None,
        "door_families": args.door_families,
        "fixed_gains": dict(zip(("translation_k", "translation_d", "yaw_k", "yaw_d"), gains)),
        "action_latency_steps_override": args.action_latency_steps,
        "position_target_no_dt": args.position_target_no_dt,
        "trace_raw_actions": args.trace_raw_actions,
        "steps": args.steps,
        "settle_step": args.settle_step,
        "dt_s": float(base_env.dt),
        "trials": {},
    }
    try:
        for name, action_idx, joint_name, request in trials:
            env.reset()
            if args.action_latency_steps is not None:
                if not 1 <= args.action_latency_steps <= base_env._max_action_latency:
                    raise ValueError("action latency override is outside the environment range")
                base_env._action_latency_buf.fill_(args.action_latency_steps)
            sampled_latency = base_env._action_latency_buf.detach().cpu().tolist()
            if args.controller == "velocity":
                velocity_history.zero_()
                velocity_target.zero_()
                # The reset event resamples base position gains. Remove their spring
                # so only the velocity error produces base force/torque.
                base_env.robot.write_joint_stiffness_to_sim(0.0, joint_ids=base_env._robot_base_dof_idx)
                for actuator in base_env.robot.actuators.values():
                    for local_idx, global_idx in enumerate(actuator.joint_indices):
                        if int(global_idx) in set(base_env._robot_base_dof_idx.tolist()):
                            actuator.stiffness[:, local_idx] = 0.0
            gains_k = base_env.robot.data.joint_stiffness[:, joint_ids].detach().cpu()
            gains_d = base_env.robot.data.joint_damping[:, joint_ids].detach().cpu()
            if trace is None:
                print(f"[PROBE] {name} request={request:.4f} action={request / args.base_action_scale:.4f}", flush=True)
            else:
                print(f"[PROBE] replaying {len(trace['frames'])} recorded robot-frame base commands", flush=True)
            trial_vel = []
            for step in range(args.steps):
                actions = torch.zeros((args.num_envs, base_env.num_policy_actions), device=device)
                if trace is None:
                    actions[:, action_idx] = request / args.base_action_scale
                else:
                    frame = trace["frames"][step]
                    raw_robot = torch.tensor(frame["raw_action"], device=device).clamp(-1.0, 1.0)
                    request_robot = raw_robot * torch.tensor(trace["scale"], device=device)
                    yaw = base_env.robot.data.joint_pos[:, joint_ids[indices["base_rotation_joint"]]]
                    policy_robot = raw_robot if args.trace_raw_actions else request_robot / args.base_action_scale
                    actions[:, 0] = policy_robot[2]
                    actions[:, 1] = yaw.cos() * policy_robot[0] - yaw.sin() * policy_robot[1]
                    actions[:, 2] = yaw.sin() * policy_robot[0] + yaw.cos() * policy_robot[1]
                _, _, terminated, truncated, _ = env.step(actions)
                if bool((terminated | truncated).any()):
                    raise RuntimeError(f"Environment reset during {name} probe at step {step}")
                measured = base_env.robot.data.joint_vel[:, joint_ids].detach().cpu()
                effort = base_env.robot.data.applied_torque[:, joint_ids].detach().cpu()
                if trace is None:
                    trial_vel.append(measured[:, indices[joint_name]].clone())
                    for env_id in range(args.num_envs):
                        rows.append({
                            "trial": name,
                            "step": step,
                            "time_s": (step + 1) * float(base_env.dt),
                            "env_id": env_id,
                            "request": request,
                            "measured": float(measured[env_id, indices[joint_name]]),
                            "effort": float(effort[env_id, indices[joint_name]]),
                        })
                else:
                    yaw = base_env.robot.data.joint_pos[:, joint_ids[indices["base_rotation_joint"]]].detach().cpu()
                    vx, vy, wz = measured.unbind(-1)
                    measured_robot = torch.stack(
                        (yaw.cos() * vx + yaw.sin() * vy, -yaw.sin() * vx + yaw.cos() * vy, wz), dim=-1
                    )
                    trial_vel.append(measured_robot.clone())
                    for env_id in range(args.num_envs):
                        row = {
                            "trial": name,
                            "step": step,
                            "time_s": (step + 1) * float(base_env.dt),
                            "env_id": env_id,
                            "sim_base_yaw": float(yaw[env_id]),
                        }
                        for i, axis in enumerate(("vx", "vy", "yaw")):
                            row[f"request_{axis}"] = float(request_robot[i])
                            row[f"measured_{axis}"] = float(measured_robot[env_id, i])
                            row[f"real_measured_{axis}"] = float(frame["measured_robot"][i])
                            row[f"sim_effort_{axis}"] = float(effort[env_id, i])
                        rows.append(row)
            velocity = torch.stack(trial_vel)
            if trace is None:
                steady = velocity[args.settle_step :]
                ratio = steady.mean(dim=0) / request
                summary["trials"][name] = {
                    "requested": request,
                    "mean_measured_steady_by_env": steady.mean(dim=0).tolist(),
                    "mean_ratio_steady_by_env": ratio.tolist(),
                    "last_measured_by_env": velocity[-1].tolist(),
                    "stiffness_by_env": gains_k.tolist(),
                    "damping_by_env": gains_d.tolist(),
                    "action_latency_by_env": sampled_latency,
                }
                print(f"[PROBE] {name} steady_ratio={ratio.tolist()}", flush=True)
            else:
                real = torch.tensor([f["measured_robot"] for f in trace["frames"][: args.steps]])
                requested = torch.tensor(
                    [torch.tensor(f["raw_action"]).clamp(-1.0, 1.0).tolist() for f in trace["frames"][: args.steps]]
                ) * torch.tensor(trace["scale"])
                summary["trials"][name] = {
                    "mean_request_steady": requested[args.settle_step :].mean(0).tolist(),
                    "mean_real_steady": real[args.settle_step :].mean(0).tolist(),
                    "mean_sim_steady_by_env": velocity[args.settle_step :].mean(0).tolist(),
                    "mae_sim_vs_real_by_env": (velocity - real[:, None, :]).abs().mean(0).tolist(),
                    "mae_sim_vs_real_steady_by_env": (
                        velocity[args.settle_step :] - real[args.settle_step :, None, :]
                    ).abs().mean(0).tolist(),
                    "stiffness_by_env": gains_k.tolist(),
                    "damping_by_env": gains_d.tolist(),
                    "action_latency_by_env": sampled_latency,
                }
                print(f"[PROBE] recorded mean sim steady={velocity[args.settle_step :].mean(0).tolist()}", flush=True)
    finally:
        env.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tracking.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[PROBE] Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
