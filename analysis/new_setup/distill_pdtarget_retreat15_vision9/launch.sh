#!/usr/bin/env bash
set -euo pipefail
repo=/home/peiqiliu/DoorOpening
python=/home/peiqiliu/miniforge3/envs/DoorOpening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261114_pdtarget_retreat15
teacher="$repo/checkpoints/new_setup/teacher_pdtarget_retreat15_best_ep2400.pth"
run=distill_pdtarget_retreat15_best2400_320
cd "$repo"
test -s "$teacher"
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l)" -eq 1024
test "$(nvidia-smi -L | grep -c '^GPU ')" -eq 2
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR="/tmp/DoorOpening/.sim_init/$run"
exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/distillation/run_multi_distillation.py \
  --distributed --headless --task DooropeningMulti --num_envs 320 \
  --door-families "$asset" --teacher "$teacher" \
  --teacher_cfg source/DoorOpening/tasks/dooropening/agents/rl_games_ppo_cfg.yaml \
  --seed 42 --triton-depth --video --video_ranks all --video_length 600 --video_interval 5000 \
  --track --wandb-project-name dooropeningv22-distill --wandb-entity peiqiliu \
  --wandb-name "$run"
