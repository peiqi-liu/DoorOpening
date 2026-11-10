#!/usr/bin/env bash
set -euo pipefail

repo=/home/peiqiliu/DoorOpening
python=/home/peiqiliu/miniforge3/envs/DoorOpening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated_laptop_exact_20261109
run=v9_qnom_06full_1110

cd "$repo"
test "$(git branch --show-current)" = 11_10/q_nomial
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l)" -eq 1024
mkdir -p "analysis/$run"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR="/tmp/DoorOpening/.sim_init/$run"

exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/rl_games/train.py --distributed --headless \
  --video --video_length 600 --video_interval 10000 \
  --task DooropeningMulti --num_envs 2048 --door-families "$asset" \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu --wandb-no-config-upload \
  --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000 \
  > "analysis/$run/train.log" 2>&1
