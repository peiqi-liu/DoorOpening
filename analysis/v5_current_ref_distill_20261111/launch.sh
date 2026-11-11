#!/usr/bin/env bash
set -euo pipefail

repo=/home/peiqi/peiqi/DoorOpening
out="$repo/analysis/v5_current_ref_distill_20261111"
python_bin=/home/peiqi/miniconda3/envs/dooropening/bin/python
asset_family=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
asset_dir="$repo/source/DoorOpening/assets/door/$asset_family"
teacher="$repo/checkpoints/current_ref_20261111/v9_mesh_best.pth"

cd "$repo"
test "$(git rev-parse --short HEAD)" = 59a9a1b6
test "$(sha256sum "$teacher" | cut -d ' ' -f 1)" = 1eb81ed8bd89a39cf9806320210fd39c9caec06e23e274f762b54d81d2c277a9
test "$(find "$asset_dir" -name traj.pkl | wc -l)" -eq 1024
test "$(cd "$asset_dir" && find . -name traj.pkl -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1)" = b3802e167509640ebfbce7c6907bfd5ddd02e7c740a275548be06743d8866a42
test "$(nvidia-smi -L | grep -c '^GPU ')" -eq 2

export PYTHONPATH="/tmp/vision5_eval_pydeps:$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG=v5_current_ref_distill
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/v5_current_ref_distill
export SLURM_JOB_ID=codex_20261111_v5_current_ref_distill

exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/distillation/run_multi_distillation.py \
  --distributed --headless --task DooropeningMulti --num_envs 300 \
  --door-families "$asset_family" --teacher "$teacher" \
  --teacher_cfg source/DoorOpening/tasks/dooropening/agents/rl_games_ppo_cfg.yaml \
  --seed 42 --triton-depth \
  --video --video_ranks all --video_length 600 --video_interval 5000 \
  --track true --wandb-project-name dooropeningv22-distill --wandb-entity peiqiliu \
  --wandb-name v5_current_ref_distill \
  > "$out/train.log" 2>&1
