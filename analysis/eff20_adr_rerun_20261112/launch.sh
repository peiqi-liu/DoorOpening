#!/usr/bin/env bash
set -euo pipefail
repo=/home/laptop/peiqi/DoorOpening_vision9_pre_effort
python=/home/laptop/anaconda3/envs/dooropening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
run=g2_eff20_adr
cd "$repo"
git merge-base --is-ancestor f7c78a8 HEAD
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | awk 'END {print NR}')" -eq 1024
test "$(cat analysis/glorbot2_effort_tilt20_20261111/trajectory_manifest_sha256.txt)" = b55b2fd95683cb57fa2161b48e835d5337e73109f2c2f8d2eff5e2a9dcff08cf
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/g2_eff20_adr
exec "$python" scripts/rl_games/train.py \
  --headless --video --video_length 900 --video_interval 10000 \
  --task DooropeningMulti --num_envs 4096 --door-families "$asset" \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu --wandb-no-config-upload \
  --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000 \
  > analysis/eff20_adr_rerun_20261112/train.log 2>&1
