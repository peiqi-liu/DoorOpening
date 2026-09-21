#!/bin/bash
set -exo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate DoorOpening

pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

pip install -U torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com

pip install git+https://github.com/isaac-sim/rl_games.git@python3.11

cd "$HOME/DoorOpening"
pip install -e source

pip install -e third_party/pointnet2_ops --no-build-isolation

pip install --no-binary=pinocchio pin

pip install viser

echo "REMOTE_ENV_INSTALL_DONE"
