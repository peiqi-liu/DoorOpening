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