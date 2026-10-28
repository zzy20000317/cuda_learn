#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
./run python -c 'import torch; print("PyTorch",torch.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(),torch.cuda.get_device_capability())'
./run nvcc --version
mkdir -p build
./run nvcc -O3 -lineinfo -arch=sm_86 exercises/probe.cu -o build/probe
./run ./build/probe
./run python exercises/project_smoke.py
./run python exercises/check_benchmark_tools.py
./run python exercises/bench_add.py
./run python -m pip check
echo "PASS: core development checks (Windows-gated profiling/debugging checked separately)"
