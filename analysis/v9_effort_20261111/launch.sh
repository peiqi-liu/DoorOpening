#!/usr/bin/env bash
set -euo pipefail

repo=/home/peiqiliu/DoorOpening
python=/home/peiqiliu/miniforge3/envs/DoorOpening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
run=v9_effort

cd "$repo"
test "$(git branch --show-current)" = 11_11/vision9_effort_limit
git merge-base --is-ancestor a314e681 HEAD
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l)" -eq 1024
mkdir -p "analysis/v9_effort_20261111"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/v9_effort
export SLURM_JOB_ID=codex_20261111_effort1

exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/rl_games/train.py \
  --distributed --headless \
  --video --video_length 600 --video_interval 10000 \
  --task DooropeningMulti --num_envs 2048 --door-families "$asset" \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu --wandb-no-config-upload \
  --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000 \
  > analysis/v9_effort_20261111/train.log 2>&1
