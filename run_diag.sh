#!/bin/bash
set -x
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate DoorOpening
cd "$HOME/DoorOpening"
python diag_sys_path.py
