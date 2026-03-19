# mamba create -n DoorOpening python=3.11
# mamba activate DoorOpening

# mamba install -c nvidia cuda-toolkit=12.4

pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

pip install -U torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
# git clone git@github.com:isaac-sim/IsaacLab.git
# cd ../IsaacLab
# ./isaaclab.sh --install
pip install git+https://github.com/isaac-sim/rl_games.git@python3.11

cd ../DoorOpening
pip install -e source

pip install -e third_party/pointnet2_ops