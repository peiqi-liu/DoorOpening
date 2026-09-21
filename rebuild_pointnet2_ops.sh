#!/bin/bash
set -exo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate DoorOpening
cd "$HOME/DoorOpening"

pip uninstall -y pointnet2_ops || true
rm -rf third_party/pointnet2_ops/build third_party/pointnet2_ops/*.egg-info third_party/pointnet2_ops/pointnet2_ops.egg-info
find third_party/pointnet2_ops -name "*.so" -delete

pip install -e third_party/pointnet2_ops --no-build-isolation --no-deps

echo "POINTNET2_REBUILD_DONE"
