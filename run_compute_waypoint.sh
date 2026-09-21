#!/bin/bash
# PYTHONPATH fix: vision9's isaaclab install is NVIDIA's newer split package where
# isaaclab.utils only lives in the nested source/isaaclab/isaaclab dir, not the outer
# site-packages/isaaclab shim (which only exposes isaaclab.app). Prepending this path lets
# compute_waypoint.py import isaaclab.utils.math without needing a SimulationApp bootstrap.
export PYTHONPATH=/home/peiqiliu/miniforge3/envs/DoorOpening/lib/python3.11/site-packages/isaaclab/source/isaaclab:$PYTHONPATH
cd ~/DoorOpening
~/miniforge3/envs/DoorOpening/bin/python source/DoorOpening/utils/state_machine/compute_waypoint.py "$@"
