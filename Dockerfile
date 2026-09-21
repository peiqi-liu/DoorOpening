# Mirrors scripts/install.sh. Builds the base toolchain (CUDA, Isaac Sim, torch, IsaacLab,
# rl_games, pinocchio, viser) into the image. The DoorOpening repo itself is NOT copied in --
# it's actively developed, so it's bind-mounted at `docker run` time and the two repo-local
# editable installs (source, third_party/pointnet2_ops) run at container start via
# scripts/docker_entrypoint.sh, so they always match whatever repo state is mounted.
#
# Build:
#   docker build -t dooropening:latest .
# Run (repo bind-mounted at /workspace/DoorOpening, matching install.sh's `cd ../DoorOpening`):
#   docker run --rm -it --gpus all \
#     -v /home/peiqiliu/DoorOpening:/workspace/DoorOpening \
#     dooropening:latest bash

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
# Isaac Sim's first bootstrap prompts interactively for EULA acceptance (isaacsim/kit/kit_app.py
# check_eula()), which hangs forever in a non-interactive container. This is the officially
# supported non-interactive accept mechanism it checks for.
ENV OMNI_KIT_ACCEPT_EULA=Y

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates git build-essential cmake ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (conda + mamba), matching the conda-based flow install.sh assumes.
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/miniforge3 \
    && rm -f /tmp/miniforge.sh
ENV PATH=/opt/miniforge3/bin:${PATH}

# mamba create -n DoorOpening python=3.11 / mamba activate DoorOpening
RUN mamba create -y -n DoorOpening python=3.11 \
    && echo "conda activate DoorOpening" >> /etc/bash.bashrc
SHELL ["mamba", "run", "-n", "DoorOpening", "/bin/bash", "-c"]

# conda install -y -c "nvidia/label/cuda-12.4.x" cuda-toolkit=12.4
RUN mamba install -y -c "nvidia/label/cuda-12.4.x" cuda-toolkit=12.4

# pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
RUN pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

# pip install -U torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install -U torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com
RUN pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com

# pip install git+https://github.com/isaac-sim/rl_games.git@python3.11
RUN pip install git+https://github.com/isaac-sim/rl_games.git@python3.11

# pip install --no-binary=pinocchio pin
RUN pip install --no-binary=pinocchio pin

# pip install viser
RUN pip install viser

# pointnet2_ops is a vendored third-party CUDA extension that essentially never changes, unlike
# source/ which is under active development -- bake its build into the image so `docker run`
# doesn't recompile it every single container start (see docker_entrypoint.sh for why: pip's own
# install bookkeeping lives in the container's site-packages, not the bind-mounted repo, so a
# fresh --rm container has no record of a prior install regardless of leftover build artifacts).
COPY third_party/pointnet2_ops /opt/pointnet2_ops_build
RUN pip install -e /opt/pointnet2_ops_build --no-build-isolation

WORKDIR /workspace/DoorOpening
COPY scripts/docker_entrypoint.sh /usr/local/bin/docker_entrypoint.sh
RUN chmod +x /usr/local/bin/docker_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker_entrypoint.sh"]
CMD ["bash"]
