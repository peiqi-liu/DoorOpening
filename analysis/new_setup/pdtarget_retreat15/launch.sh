#!/usr/bin/env bash
set -euo pipefail
repo=/home/laptop/peiqi/DoorOpening_vision9_pre_effort
python=/home/laptop/anaconda3/envs/dooropening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261114_pdtarget_retreat15
run=teacher_pdtarget_retreat15_4096
cd "$repo"
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l | tr -d ' ')" -eq 1024
test -s analysis/new_setup/pdtarget_retreat15/trajectory_manifest_sha256.txt
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/$run
exec "$python" scripts/rl_games/train.py \
  --headless --video --video_length 900 --video_interval 10000 \
  --task DooropeningMulti --num_envs 4096 --door-families "$asset" \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu --wandb-no-config-upload \
  --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000
