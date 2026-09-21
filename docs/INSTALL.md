# DoorOpening — fresh machine install

Reference environment (the machine this was written from, everything below is verified against it):

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, glibc **2.35** |
| GPU / driver | RTX 3090, driver 550.144.03 (CUDA 12.4) |
| Python | 3.11 (conda env `dooropening`) |
| isaacsim | 5.1.0.0 (pip wheels) |
| IsaacLab | editable clone, commit `f4aa17f87e2e5db5484f0b5974918573e8918ce2` |
| torch | 2.7.0+cu128 |
| numpy | 1.26.0 (must stay `<2`) |
| CUDA toolkit for building | `cuda-toolkit` 12.4.1 + `gxx_linux-64` 11.2 from conda |

## 0. Hard OS requirement (read this first)

Isaac Sim 5.1 pip wheels are tagged `manylinux_2_35`, i.e. **glibc >= 2.35 → Ubuntu 22.04 or 24.04**.
On Ubuntu 20.04 (glibc 2.31, often with a 5.15 HWE kernel so `uname -r` looks modern) the install
fails with `RuntimeError: Didn't find wheel for isaacsim 5.1.0.0` — see Troubleshooting.

```bash
ldd --version | head -1     # want 2.35+
nvidia-smi                  # want driver >= 535; 550+ matches the reference box
```

## 1. System packages

```bash
sudo apt update
sudo apt install -y build-essential git git-lfs cmake \
    libglu1-mesa libxi6 libxrandr2 libxcursor1 libxinerama1 libgl1
git lfs install
```

`git-lfs` is **not optional**: `.gitattributes` puts `*.obj`, `*.mtl`, `*.dae`, `*.pt`, `*.hdf5` in LFS,
so the robot meshes under `source/DoorOpening/assets/glorbot/meshes/` arrive as text pointers without it.

## 2. Conda env + torch

```bash
conda create -n dooropening python=3.11 -y
conda activate dooropening
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

## 3. Isaac Sim

```bash
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -c "import isaacsim; print('isaacsim ok')"
```

## 4. IsaacLab (editable, pinned)

Clone **outside** this repo, then pin to the commit this project was last run against:

```bash
cd ~/peiqi
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout f4aa17f87e2e5db5484f0b5974918573e8918ce2
./isaaclab.sh --install          # installs isaaclab* packages editable + rl_games, rsl_rl, ...
```

That single command is what produces the editable `isaaclab`, `isaaclab_assets`, `isaaclab_rl`,
`isaaclab_tasks` installs plus `rl_games==1.6.1` and `warp-lang`.

Sanity check:

```bash
python -c "import isaaclab, isaaclab_rl, rl_games; print(isaaclab.__file__)"
```

## 5. Clone DoorOpening (with submodule + LFS)

```bash
cd ~/peiqi
git clone --recurse-submodules <this-repo-url> DoorOpening
cd DoorOpening
git lfs pull
git submodule update --init --recursive     # third_party/pointnet2_ops
```

Verify LFS actually resolved (should be a binary mesh, not a ~130 byte pointer):

```bash
find source/DoorOpening/assets/glorbot/meshes -name '*.obj' | head -1 | xargs ls -l
```

## 6. Install the project package

```bash
pip install -e source/DoorOpening
pip install "numpy==1.26.0"     # re-pin: some deps try to pull numpy 2.x, isaacsim needs <2
```

`source/setup.py` covers `psutil urdf_parser_py urchin open3d geometrout trimesh usd_core bpy scipy viser yourdfpy`.

Extras used by scripts but **not** declared in `setup.py`:

```bash
pip install wandb prettytable websockets matplotlib
# only for the offline state-machine / IK tooling (source/DoorOpening/utils/state_machine/pin.py):
pip install pin pin-pink
```

`source/DoorOpening/motion/glorbot_controller.py` imports `pyRMP`, which is not installed here and
not imported by anything else — it is dead code, skip it.

## 7. Build `pointnet2_ops` (needed for the point-cloud student model)

Requires a real `nvcc` and a gcc that torch accepts. The reference box gets both from conda rather
than system CUDA (there is no `/usr/local/cuda`):

```bash
conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit=12.4.1
conda install -y -c conda-forge gxx_linux-64=11.2
export CUDA_HOME=$CONDA_PREFIX
pip install -e third_party/pointnet2_ops
python -c "from pointnet2_ops.pointnet2_utils import furthest_point_sample; print('pointnet2 ok')"
```

The setup builds for `TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"` — add your arch (e.g. `8.9` for 4090,
`9.0` for H100) to `third_party/pointnet2_ops/setup.py` if the target GPU is newer than Ampere.
The checked-in `_ext.cpython-311-*.so` only works for cp311 and is not in git anyway, so always rebuild.

## 8. Door assets — the part git does not give you

**32 GB of door assets are not in the repository.** `.gitignore` excludes `**/PartNet*/*` (and all
`*.usd`), so `source/DoorOpening/assets/door/` arrives with only `door_cfg.py` and `multi_door_cfg.py`.
Doors are loaded from `mobility.urdf` at runtime, so without them every run dies with
`FileNotFoundError: No mobility.urdf files found under ...`.

The defaults in `source/DoorOpening/assets/door/multi_door_cfg.py:171` are:

```
PartNetv5_plusplus  PartNetv6_plusplus  PartNetv7_plusplus  PartNetv8_plusplus
```

(495M / 495M / 415M / 415M — all four families must contain the **same number** of assets or
`multi_door_cfg.py` raises `Door families must contain the same number of assets`.)

**Option A (recommended) — copy from a working machine:**

```bash
rsync -avh --progress \
  <user>@<working-host>:~/peiqi/DoorOpening/source/DoorOpening/assets/door/PartNetv{5,6,7,8}_plusplus \
  source/DoorOpening/assets/door/
```

Add `PartNetv4` (48M, the base set the generators clone from) plus any `v*_test` split you need for eval.

**Option B — regenerate:** the generators are clone-and-edit, not from-scratch, so you still need the
`PartNetv4` base doors first (originally derived from PartNet-Mobility). Then:

```bash
python scripts/tools/generate_randomized_doors_v2.py \
  --asset-root source/DoorOpening/assets/door/PartNetv4 \
  --output-dir source/DoorOpening/assets/door/PartNetv5_plusplus \
  --variants-per-source 4 --seed 7
python scripts/tools/copy_pull_to_push.py --help    # v5 -> v8 style push variants
```

Note: per project convention, changes to generation parameters belong in the `DEFAULT_*` constants at
the top of the generator, not passed as one-off CLI flags.

## 9. First run

`IsaacLab_tmp/` (see `source/DoorOpening/assets/cache_utils.py:15`) is an auto-created URDF→USD
conversion cache. It is gitignored, so **the first launch is slow** while every door URDF is converted;
later runs reuse it. Don't copy it between machines.

```bash
python scripts/list_envs.py                     # should list DooropeningMulti

python scripts/rl_games/train.py \
  --task DooropeningMulti --num_envs 16 --headless --max_iterations 2
```

Then the real thing:

```bash
python scripts/rl_games/train.py \
  --task DooropeningMulti --num_envs 2048 agent.params.config.minibatch_size=8192 \
  --max_iterations 1000 --headless --track \
  --wandb-project-name dooropening --wandb-entity <entity>
```

(`wandb login` first if using `--track`.)

## Troubleshooting

**`RuntimeError: Didn't find wheel for isaacsim 5.1.0.0` / `metadata-generation-failed`**
The `isaacsim` PyPI package is a stub that downloads the real wheel from `pypi.nvidia.com`. If the log
lists the wheel names (`...manylinux_2_35_x86_64.whl`) and *then* fails, the index was reachable and the
**platform tag did not match** — almost always glibc < 2.35 (Ubuntu 20.04). Check:

```bash
ldd --version | head -1
python -c "from packaging.tags import sys_tags; print(any('manylinux_2_35_x86_64' in str(t) for t in sys_tags()))"
```

If that prints `False`, no pip flag helps. Options: upgrade to Ubuntu 22.04/24.04, or run the
Isaac Sim 5.1 NGC container (`nvcr.io/nvidia/isaac-sim:5.1.0`) and install steps 4–8 inside it.
If instead the log shows a connection/404 before listing wheels, that is a genuine network/proxy issue —
retry with `--extra-index-url https://pypi.nvidia.com` and no local index mirror.

**Meshes load as garbage / tiny files** — `git lfs pull` was skipped.

> On a glibc-2.31 host (Ubuntu 20.04) see [Appendix A](#appendix-a--running-on-an-ubuntu-2004--glibc-231-host).

**`ModuleNotFoundError: pointnet2_ops`** — submodule not initialized, or the CUDA extension failed to
build; re-run step 7 with `CUDA_HOME` set and check `nvcc --version` matches torch's CUDA major version.

**`numpy.dtype size changed` / numpy 2.x errors** — something upgraded numpy; `pip install numpy==1.26.0`.

**`No mobility.urdf files found`** — door assets missing, step 8.

**`Door families must contain the same number of assets`** — an incomplete rsync of one family.

---

# Appendix A — running on an Ubuntu 20.04 / glibc 2.31 host

## What does *not* work

- **Retagging or force-installing the wheels.** The tag is not over-cautious: the shipped Isaac Sim
  binaries reference `GLIBC_2.32`, `GLIBC_2.33` and `GLIBC_2.34` symbols (verified with
  `objdump -T` over `site-packages/isaacsim*/**/*.so` on the 22.04 reference box). Bypassing the tag
  check just moves the failure to import time: `version 'GLIBC_2.34' not found`.
- **An older `isaacsim` version.** `pypi.nvidia.com` currently serves only `5.1.0.0` and `5.0.0.0`;
  both are built on 22.04, so the glibc floor is identical. The 4.x pip wheels are no longer indexed,
  and pairing a 4.5-era Isaac Sim with this project would mean rolling IsaacLab back ~1 year from the
  pinned `f4aa17f` commit — a far larger port than the OS problem it avoids.
- **Hand-upgrading system glibc on 20.04.** Replacing the loader out-of-distro is a good way to brick
  a machine. Don't.

The fix is always: keep the 20.04 *host*, give Isaac Sim a 22.04+ *userspace*. Only the NVIDIA driver
is shared with the host, and a driver bump is a normal 20.04 operation if kit complains about it —
no OS change needed.

## Option 1 — Docker (most predictable)

Build a 22.04 image and run the exact verified pip install (steps 1–8) inside it. Do **not** start from
the NGC `isaac-sim` image unless you want the binary/`python.sh` layout instead of the pip layout this
doc is written against.

```dockerfile
# docker/Dockerfile.u2204
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y build-essential git git-lfs cmake wget \
    libglu1-mesa libxi6 libxrandr2 libxcursor1 libxinerama1 libgl1 && git lfs install
# then: miniforge + python 3.11 + steps 2-7
```

Host side (works on 20.04):

```bash
sudo apt install -y docker.io nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

docker run --gpus all -it --rm \
  -v ~/peiqi/DoorOpening:/workspace/DoorOpening \
  -v ~/peiqi/IsaacLab:/workspace/IsaacLab \
  -v ~/.cache/ov:/root/.cache/ov -v ~/.nvidia-omniverse:/root/.nvidia-omniverse \
  dooropening:u2204 bash
```

Mount the repo and IsaacLab rather than copying them — the 32 GB of door assets should live on the host
once. Persist the Omniverse caches too, or every container start re-downloads shaders. Training is
`--headless`, so no X plumbing is required; for a viewport use Isaac Sim's WebRTC livestream, and note
that `viser`-based tools need their port published (`-p 8080:8080`).

## Option 2 — Distrobox / Podman (keeps the normal workflow)

Lighter than Docker for interactive work: the container shares your `$HOME`, so paths, the repo, and the
asset tree are identical to the host.

```bash
sudo apt install -y podman   # or docker
# distrobox from https://github.com/89luca89/distrobox (single script, no root needed)
distrobox create --name u2204 --image nvidia/cuda:12.4.1-devel-ubuntu22.04 --nvidia \
  --home ~/containers/u2204
distrobox enter u2204
# inside: glibc is 2.35 -> run steps 1-8 normally
```

Use the separate `--home` (or at minimum a container-specific conda prefix such as
`~/miniforge3-u2204`). A conda env is compiled against the glibc it was created on, so sharing one
`~/miniconda3` between the 20.04 host and the 22.04 container will eventually break both.

## Option 3 — in-place release upgrade

`sudo do-release-upgrade` from 20.04 LTS to 22.04 LTS is an in-place upgrade, not a reinstall: packages
and `$HOME` survive. This is the least-effort permanent fix if the box can take an afternoon of downtime,
and it removes the container layer entirely. Snapshot or back up first, and expect to reinstall the
NVIDIA driver afterwards.

## Verifying, whichever option

Inside the container/upgraded host:

```bash
ldd --version | head -1     # expect 2.35+
python -c "from packaging.tags import sys_tags; print(any('manylinux_2_35_x86_64' in str(t) for t in sys_tags()))"
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -c "import isaacsim; print('ok')"
```

