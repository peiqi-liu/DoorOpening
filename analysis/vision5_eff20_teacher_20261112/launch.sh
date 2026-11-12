#!/usr/bin/env bash
set -euo pipefail
repo=/home/peiqi/peiqi/DoorOpening
python=/home/peiqi/miniconda3/envs/dooropening/bin/python
asset=PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated
teacher="$repo/checkpoints/glorbot2_eff20_best_ep2300.pth"
run=v5_eff20_teacher
cd "$repo"
test "$(git branch --show-current)" = 11_12/vision5_eff20_teacher
git merge-base --is-ancestor f7c78a8 HEAD
test -s "$teacher"
test "$(find "source/DoorOpening/assets/door/$asset" -name traj.pkl | wc -l | tr -d ' ')" -eq 1024
test "$("$python" - "$asset" <<'PYHASH'
from pathlib import Path
import hashlib,sys
family=Path("source/DoorOpening/assets/door")/sys.argv[1]
h=hashlib.sha256()
for p in sorted(family.glob("*/traj.pkl")):
    h.update(p.parent.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
print(h.hexdigest())
PYHASH
)" = b55b2fd95683cb57fa2161b48e835d5337e73109f2c2f8d2eff5e2a9dcff08cf
export PYTHONPATH="$repo/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DOOROPENING_WANDB_SAVE_CODE=0
export DOOROPENING_DISTRIBUTED_RUN_TAG="$run"
export SIM_INIT_DIR=/tmp/DoorOpening/.sim_init/v5_eff20_teacher
exec "$python" -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29579 \
  scripts/rl_games/train.py \
  --distributed --headless --video --video_length 900 --video_interval 10000 \
  --task DooropeningMulti --num_envs 300 --door-families "$asset" \
  --checkpoint "$teacher" --track --wandb-project-name dooropeningv22 --wandb-entity peiqiliu \
  --wandb-no-config-upload --wandb-name "$run" --exp-name "$run" --seed 42 --max_iterations 30000 \
  --minibatch-size 7200 \
  > analysis/vision5_eff20_teacher_20261112/train.log 2>&1
