#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python="${PYTHON_BIN:-python}"
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
run=teacher_4096
cd "$repo"

if ! command -v nvidia-smi >/dev/null || [[ "$(nvidia-smi -L | grep -c '^GPU ')" -lt 2 ]]; then
  echo "Expected at least two visible GPUs for the 4,096-environment DDP run." >&2
  exit 1
fi
if [[ "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l | tr -d ' ')" -ne 1024 ]]; then
  echo "Expected 1,024 generated trajectories in $asset." >&2
  exit 1
fi

export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR="${SIM_INIT_DIR:-/tmp/DoorOpening/.sim_init/$run}"

args=(--distributed --headless --video --video_length 900 --video_interval 10000
  --task DooropeningMulti --num_envs 2048 --door-families "$asset"
  --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu
  --wandb-no-config-upload --wandb-name "$run" --exp-name "$run"
  --seed 42 --max_iterations 30000)
if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "$TEACHER_CHECKPOINT")
fi

exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/rl_games/train.py "${args[@]}"
