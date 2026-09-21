#!/bin/bash
set -exo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate DoorOpening
mamba install -y -c "nvidia/label/cuda-12.4.x" cuda-toolkit=12.4
echo "CUDA_TOOLKIT_REINSTALL_DONE"
