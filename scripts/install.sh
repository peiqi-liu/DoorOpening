mamba create -n DoorOpening python=3.11
mamba activate DoorOpening

mamba install -c nvidia cuda-toolkit=12.8

pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install