#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python="${PYTHON_BIN:-python}"
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
run=distill_300
out="$repo/analysis/new_setup/$run"
cd "$repo"

if [[ -z "${TEACHER_CHECKPOINT:-}" || ! -s "$TEACHER_CHECKPOINT" ]]; then
  echo "Set TEACHER_CHECKPOINT to the teacher checkpoint to distill." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null || [[ "$(nvidia-smi -L | grep -c '^GPU ')" -lt 2 ]]; then
  echo "Expected at least two visible GPUs for the 300-environment-per-GPU DDP run." >&2
  exit 1
fi
if rg -q '^  target_err_arm:' source/DoorOpening/distillation/pcd_transformer_dagger_cfg.yaml; then
  echo "Student config unexpectedly includes target_err_arm." >&2
  exit 1
fi
if [[ "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l | tr -d ' ')" -ne 1024 ]]; then
  echo "Expected 1,024 generated trajectories in $asset." >&2
  exit 1
fi

mkdir -p "$out"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR="${SIM_INIT_DIR:-/tmp/DoorOpening/.sim_init/$run}"

exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/distillation/run_multi_distillation.py \
  --distributed --headless --task DooropeningMulti --num_envs 300 \
  --door-families "$asset" --teacher "$TEACHER_CHECKPOINT" \
  --teacher_cfg source/DoorOpening/tasks/dooropening/agents/rl_games_ppo_cfg.yaml \
  --seed 42 --triton-depth --video --video_ranks all --video_length 600 --video_interval 5000 \
  --track --wandb-project-name dooropeningv22-distill --wandb-entity peiqiliu \
  --wandb-name "$run" > "$out/train.log" 2>&1
