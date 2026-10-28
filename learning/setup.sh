#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p tools logs .cache/pip
export MAMBA_ROOT_PREFIX="$PWD/.mamba"
export CONDA_PKGS_DIRS="$PWD/.mamba/pkgs"
if [ ! -x tools/bin/micromamba ]; then
  curl -fL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o tools/micromamba.tar.bz2
  tar -xjf tools/micromamba.tar.bz2 -C tools bin/micromamba
fi
if [ -f conda-explicit.txt ]; then
  tools/bin/micromamba --no-rc create -y -p "$PWD/.env" -f conda-explicit.txt
else
  tools/bin/micromamba --no-rc create -y -p "$PWD/.env" -f environment.yml
fi
if [ ! -x tools/cuda-gdb/bin/cuda-gdb-python3.11-tui ]; then
  curl -fL --retry 3 https://developer.download.nvidia.com/compute/cuda/redist/cuda_gdb/linux-x86_64/cuda_gdb-linux-x86_64-12.8.90-archive.tar.xz -o tools/cuda-gdb.tar.xz
  printf '%s\n' 'f10953b7bb795b1ff59f769da8d2e776f5b0f9e4bf5daf2a6c3bd301250321ff  tools/cuda-gdb.tar.xz' | sha256sum -c -
  mkdir -p tools/cuda-gdb
  tar -xJf tools/cuda-gdb.tar.xz --strip-components=1 -C tools/cuda-gdb
fi
if compgen -G '.cache/wheels/torch-*.whl' > /dev/null; then
  ./run python -m pip install .cache/wheels/*.whl --index-url https://pypi.org/simple
else
  ./run python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
fi
if [ -f requirements-lock.txt ]; then
  ./run python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
else
  ./run python -m pip install -r requirements-bootstrap.txt
fi
./run python export_environment.py
./verify.sh
