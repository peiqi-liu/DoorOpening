#!/usr/bin/env bash
set -euo pipefail

repo=/home/laptop/peiqi/DoorOpening
python=/home/laptop/anaconda3/envs/dooropening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
run=gl2_bias_015

cd "$repo"
test "$(git branch --show-current)" = 11_10/laptop_torque_bias
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l)" -eq 1024
mkdir -p "analysis/$run"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"

exec "$python" scripts/rl_games/train.py --headless \
  --video --video_length 300 --video_interval 10000 \
  --task DooropeningMulti --num_envs 4096 --door-families "$asset" \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu --wandb-no-config-upload \
  --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000 \
  > "analysis/$run/train.log" 2>&1
