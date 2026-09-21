#!/bin/bash
set -x
cd ~/DoorOpening

docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v ~/DoorOpening:/workspace/DoorOpening \
  -v ~/.netrc:/root/.netrc:ro \
  dooropening:latest \
  python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    ./scripts/distillation/run_multi_distillation.py \
    --task DooropeningMulti \
    --headless \
    --num_envs 300 \
    --distributed \
    --teacher_partnetv5 source/DoorOpening/assets/door/PartNetv5/door_opening.pth \
    --door-families PartNetv5 \
    --video --video_interval 5000 --video_length 1000 \
    --viser \
    --track \
    --wandb-entity peiqiliu \
    --wandb-project-name dooropeningv23 \
    --wandb-name PartNetv5_docker
