#!/usr/bin/env bash
set -euo pipefail

repo=/home/laptop/peiqi/DoorOpening
python=/home/laptop/anaconda3/envs/dooropening/bin/python
run_name=glorbot2_teleop_ref_box10_40_w5_2048env_scratch_20261103
asset_family=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261101
log_file="$repo/analysis/glorbot2_box_penalty_2048_20261103/logs/train.log"

cd "$repo"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run_name"
export SIM_INIT_DIR="/tmp/DoorOpening/.sim_init/$run_name"

# One RTX 5090 Laptop GPU; 2,048 here is the total environment count.
exec "$python" scripts/rl_games/train.py \
  --headless --video --video_length 900 --video_interval 10000 \
  --task DooropeningMulti --num_envs 2048 \
  --door-families "$asset_family" \
  --track true --wandb-project-name dooropeningv22 --wandb-entity peiqiliu \
  --wandb-name "$run_name" --exp-name "$run_name" \
  --seed 42 --max_iterations 30000 > "$log_file" 2>&1
