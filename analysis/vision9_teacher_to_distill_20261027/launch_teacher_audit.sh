#!/usr/bin/env bash
set -euo pipefail
repo=/home/peiqiliu/DoorOpening
out="$repo/analysis/vision9_teacher_to_distill_20261027"
checkpoint="$out/teacher_ep3200.pth"
cd "$repo"
source /home/peiqiliu/miniforge3/etc/profile.d/conda.sh
conda activate DoorOpening
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/rl_games/play.py \
  --headless --video --video_length 600 \
  --task DooropeningMulti --num_envs 32 \
  --door-families ScratchDoors_left_pull_current_20261023 \
  --checkpoint "$checkpoint" --seed 42 \
  --base-action-scale 1.0 \
  --base-translation-stiffness 80000 --base-translation-nominal-damping 12000 \
  --base-yaw-stiffness 11000 --base-yaw-nominal-damping 1200 \
  --door-panel-effort-max 40 \
  --door-handle-limit-margin-range-rad 0.01 0.04 \
  --door-audit-out "$out/teacher_audit_32env.json" --door-audit-episodes 1 \
  --save-viser-pt "$out/teacher_env00_replay.pt" --viser-env-id 0 \
  --save-viser-num-rollouts 1
