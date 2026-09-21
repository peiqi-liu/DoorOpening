#!/bin/bash
# Runs the repo-local editable install of `source` against whatever DoorOpening checkout is
# bind-mounted at /workspace/DoorOpening, then execs the container's CMD. This is what lets the
# image stay stable while the repo keeps changing underneath it. pointnet2_ops is NOT installed
# here -- it's baked into the image at build time (see Dockerfile) since it's a vendored
# third-party CUDA extension that essentially never changes, so there's no reason to recompile it
# on every container start.
set -eo pipefail

# -u is deliberately not set: conda's own gxx_linux-64 activation script references
# SYS_SYSROOT without a default and crashes under `set -u` (unbound variable).
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate DoorOpening

cd /workspace/DoorOpening

# pip install -e source
pip install -e source

exec "$@"
