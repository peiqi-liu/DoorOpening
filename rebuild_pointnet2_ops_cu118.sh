#!/bin/bash
set -exo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate DoorOpening

# Build against Isaac Sim's bundled torch (2.5.1+cu118) -- confirmed via diagnostic
# that Kit force-loads this after AppLauncher boots, regardless of PYTHONPATH order.
export PYTHONPATH="/home/peiqiliu/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle:${PYTHONPATH}"

# Real CUDA 11.8 toolkit (matches torch.version.cuda), not the conda env's CUDA 12.4.
export CUDA_HOME="$HOME/cuda-11.8"
export PATH="$HOME/cuda-11.8/bin:${PATH}"

# nvcc 11.8 rejects the conda env's gcc 12.4 host compiler (>11 unsupported). Use the
# system gcc 9.4 (Ubuntu 20.04 default) for BOTH the .cpp and .cu compile steps to
# avoid mixing ABIs between two different compiler majors.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export NVCC_PREPEND_FLAGS=" -ccbin=/usr/bin/g++"

python -c "import torch; print('Build-time torch:', torch.__version__, torch.__file__)"
nvcc --version
g++ --version | head -1

cd "$HOME/DoorOpening"
pip uninstall -y pointnet2_ops || true
rm -rf third_party/pointnet2_ops/build third_party/pointnet2_ops/*.egg-info third_party/pointnet2_ops/pointnet2_ops.egg-info
find third_party/pointnet2_ops -name "*.so" -delete

pip install -e third_party/pointnet2_ops --no-build-isolation --no-deps

echo "POINTNET2_REBUILD_CU118_DONE"
