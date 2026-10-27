#!/usr/bin/env bash
set -euo pipefail
repo=/home/peiqiliu/DoorOpening
python=/home/peiqiliu/miniforge3/envs/DoorOpening/bin/python
run_name=vision9_base_position_teacher_scale1_highbase_panel40_margin004_20261026
cd "$repo"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/rl_games/train.py \
  --distributed --headless --video --video_length 600 --video_interval 10000 \
  --task DooropeningMulti --num_envs 2048 \
  --door-families ScratchDoors_left_pull_current_20261023 \
  --base-action-scale 1.0 \
  --base-translation-stiffness 80000 --base-translation-damping 12000 \
  --base-yaw-stiffness 11000 --base-yaw-damping 1200 \
  --door-panel-effort-max 40 \
  --door-handle-limit-margin-range-rad 0.01 0.04 \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu \
  --wandb-name "$run_name" --exp-name "$run_name" \
  --seed 42 --max_iterations 30000
