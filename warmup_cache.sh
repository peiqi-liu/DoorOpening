#!/bin/bash
cd ~/DoorOpening
~/miniforge3/envs/DoorOpening/bin/python ./scripts/rl_games/train.py \
  --task DooropeningMulti \
  --num_envs 512 \
  --max_iterations 1 \
  --headless \
  --seed 42 \
  --door_families PartNetv5_plus \
  2>&1 | tee ~/DoorOpening/warmup_cache_$(date +%Y%m%d_%H%M%S).log
