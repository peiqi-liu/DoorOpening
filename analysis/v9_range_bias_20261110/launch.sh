#!/usr/bin/env bash
set -euo pipefail

repo=/home/peiqiliu/DoorOpening
python=/home/peiqiliu/miniforge3/envs/DoorOpening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
run=v9_bias_noise

cd "$repo"
test "$(git branch --show-current)" = 11_10/measured_range_bias
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l)" -eq 1024
mkdir -p "analysis/v9_range_bias_20261110"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/v9_bias_noise

exec "$python" scripts/rl_games/train.py --headless \
  --video --video_length 600 --video_interval 10000 \
  --task DooropeningMulti --num_envs 2048 --door-families "$asset" \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu --wandb-no-config-upload \
  --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000 \
  > analysis/v9_range_bias_20261110/train.log 2>&1
