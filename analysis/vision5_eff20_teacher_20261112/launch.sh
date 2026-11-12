#!/usr/bin/env bash
set -euo pipefail
repo=/home/peiqi/peiqi/DoorOpening
python=/home/peiqi/miniconda3/envs/dooropening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
teacher="$repo/checkpoints/glorbot2_eff20_best_ep2300.pth"
run=v5_eff20_teacher
cd "$repo"
test "$(git branch --show-current)" = 11_11/vision5_eff20_teacher
git merge-base --is-ancestor f7c78a8 HEAD
test -s "$teacher"
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l | tr -d  )" -eq 1024
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/v5_eff20_teacher
exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/rl_games/train.py \
  --distributed --headless --video --video_length 900 --video_interval 10000 \
  --task DooropeningMulti --num_envs 300 --door-families "$asset" \
  --checkpoint "$teacher" --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu \
  --wandb-no-config-upload --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000 \
  > analysis/vision5_eff20_teacher_20261112/train.log 2>&1
