#!/usr/bin/env bash
set -eo pipefail
cd /home/peiqiliu/DoorOpening
source /home/peiqiliu/miniforge3/etc/profile.d/conda.sh
conda activate DoorOpening
export DOOROPENING_WANDB_SAVE_CODE=0
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/rl_games/train.py \
  --distributed --headless --video --video_length 600 --video_interval 10000 \
  --task DooropeningMulti --num_envs 2048 \
  --door-families ScratchDoors_left_pull_current_20261023 \
  --base-action-scale 1.0 \
  --base-translation-stiffness 80000 --base-translation-damping 12000 \
  --base-yaw-stiffness 11000 --base-yaw-damping 1200 \
  --door-panel-effort-max 40 \
  --door-handle-limit-margin-range-rad 0.01 0.04 \
  --checkpoint /home/peiqiliu/DoorOpening/logs/rl_games/door_opening/vision9_base_position_teacher_scale1_highbase_panel40_margin004_20261026_rank000_local000/nn/door_opening.pth \
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu \
  --wandb-name vision9_base_position_teacher_sharedadr_resume487_20261026 \
  --exp-name vision9_base_position_teacher_sharedadr_resume487_20261026 \
  --seed 42 --max_iterations 30000
