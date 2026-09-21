#!/bin/bash
set -exo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate DoorOpening
cd "$HOME/DoorOpening"

pip install -e source --no-deps
pip install psutil urdf_parser_py urchin open3d geometrout trimesh usd_core scipy viser yourdfpy

pip install -e third_party/pointnet2_ops --no-build-isolation

pip install --no-binary=pinocchio pin

echo "DOOROPENING_INSTALL_DONE"
