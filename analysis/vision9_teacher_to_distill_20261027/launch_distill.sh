#!/usr/bin/env bash
set -eo pipefail
repo=/home/peiqiliu/DoorOpening
out="$repo/analysis/vision9_teacher_to_distill_20261027"
checkpoint="$out/teacher_ep3200.pth"
cd "$repo"
source /home/peiqiliu/miniforge3/etc/profile.d/conda.sh
conda activate DoorOpening
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export DOOROPENING_WANDB_SAVE_CODE=0
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/distillation/run_multi_distillation.py \
  --distributed --headless --task DooropeningMulti --num_envs 400 \
  --door-families ScratchDoors_left_pull_current_20261023 \
  --teacher "$checkpoint" --seed 42 \
  --base-action-scale 1.0 \
  --base-translation-stiffness 80000 --base-translation-damping 12000 \
  --base-yaw-stiffness 11000 --base-yaw-damping 1200 \
  --door-panel-effort-max 40 \
  --door-handle-limit-margin-range-rad 0.01 0.04 \
  --triton-depth \
  --video --video_length 600 --video_interval 5000 \
  --viser --viser-raw-path "$out/student_viser_replay.pt" \
  --track --wandb-project-name dooropeningv22-distill --wandb-entity peiqiliu \
  --wandb-name vision9_scratchdoor_teacher3200_triton_400gpu_20261027
