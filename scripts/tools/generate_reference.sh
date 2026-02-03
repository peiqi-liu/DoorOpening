#!/usr/bin/env bash

set -e  # stop if any command fails

for door in $(seq 0 20); do
    echo "=============================="
    echo "Running door_number=$door"
    echo "=============================="

    python scripts/llm_agent.py \
        --enable_cameras \
        --door_number "$door"
done

python scripts/rl_games/train.py --task Dooropening --num_envs 1024 agent.params.config.minibatch_size=4096 --max_iterations 20000 --video --wandb-project-name dooropeningv11 --wandb-entity peiqiliu --seed 1 --headless --track