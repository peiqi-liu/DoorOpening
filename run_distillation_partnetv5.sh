#!/bin/bash
set -x
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate DoorOpening
export ISAACSIM_PATH="$HOME/isaacsim"
export ISAACSIM_PYTHON_EXE="$ISAACSIM_PATH/python.sh"
# torch.compile's inductor backend fails to compile the depth-render function on this
# torch 2.5.1/triton build (TypeError: Signature keys must be string). Fall back to eager
# for that graph instead of crashing -- numerically identical, just not compiled.
export TORCHDYNAMO_SUPPRESS_ERRORS=1
cd "$HOME/DoorOpening"

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
  --wandb-name PartNetv5_vision9
