#!/usr/bin/env bash
set -euo pipefail
repo=/home/peiqiliu/DoorOpening
out="$repo/analysis/vision9_effort_tilt20_distill_20261111"
python=/home/peiqiliu/miniforge3/envs/DoorOpening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
teacher="$repo/logs/rl_games/door_opening/v9_effort3_rank000_local000/nn/last_door_opening_ep_2300_rew_20453.068.pth"
cd "$repo"
test "$(git branch --show-current)" = 11_11/vision9_effort_distill
git merge-base --is-ancestor dceee01e HEAD
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l | tr -d ' ')" -eq 1024
test "$(python - "$asset" <<'PY'
from pathlib import Path
import hashlib,sys
family=Path('source/DoorOpening/assets/door')/sys.argv[1]
h=hashlib.sha256()
for p in sorted(family.glob('*/traj.pkl')):
    h.update(p.parent.name.encode())
    h.update(hashlib.sha256(p.read_bytes()).digest())
print(h.hexdigest())
PY
)" = b55b2fd95683cb57fa2161b48e835d5337e73109f2c2f8d2eff5e2a9dcff08cf
test "$(sha256sum "$teacher" | cut -d ' ' -f 1)" = ba6410ef554029d5985c2fd642ced22282c41321edb50ea78b2d2a649478acd1
mkdir -p "$out"
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG=v9_eff20_distill
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/v9_eff20_distill
export SLURM_JOB_ID=codex_20261111_v9_eff20_distill
exec "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/distillation/run_multi_distillation.py \
  --distributed --headless --task DooropeningMulti --num_envs 400 \
  --door-families "$asset" --teacher "$teacher" \
  --teacher_cfg source/DoorOpening/tasks/dooropening/agents/rl_games_ppo_cfg.yaml \
  --seed 42 --triton-depth \
  --video --video_ranks all --video_length 600 --video_interval 5000 \
  --track --wandb-project-name dooropeningv22-distill --wandb-entity peiqiliu \
  --wandb-name v9_eff20_distill \
  > "$out/train.log" 2>&1
