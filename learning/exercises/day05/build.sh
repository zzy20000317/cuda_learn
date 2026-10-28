#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
../../run nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 reduce.cu -o reduce_bench
