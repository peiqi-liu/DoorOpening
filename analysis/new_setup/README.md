# Shared `11_12/new_setup`

This branch combines Glorbot2 laptop's 20-degree reference-generation code with
the Franka effort-limit and PD-target ADR W&B logging, and uses the student
configuration without `target_err_arm`.

## Reference motions

Run `generate_refs.sh` on Glorbot2 laptop with `PYTHON_BIN` set to the Isaac Lab
Python environment. It calls the laptop's `generate_refs.py`, which invokes
`compute_waypoint.py` for all 1,024 doors and records the planner and trajectory
hashes. The asset family is
`PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated`.

## RL teacher

Run `launch_teacher.sh` on two GPUs. It launches distributed PPO with 2,048
environments per GPU (4,096 total), W&B tracking, and video recording. Set
`TEACHER_CHECKPOINT` only when resuming; leave it unset for a scratch run.

## Distillation

Run `launch_distillation.sh` on two GPUs with `TEACHER_CHECKPOINT` set. It uses
300 environments per GPU, W&B tracking, and video recording on both ranks. The
shared `pcd_transformer_dagger_cfg.yaml` omits `target_err_arm`.

Set `REPO_ROOT` and `PYTHON_BIN` when the checkout or Isaac Lab environment is
at a different path. These launchers require the same two-GPU setup on each
host.
