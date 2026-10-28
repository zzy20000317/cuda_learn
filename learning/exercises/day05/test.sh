#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
./build.sh
for block in 128 256 512 1024; do
  # Grid 1 stresses per-thread loops; 17 is deliberately not a multiple of SMs.
  for grid in 1 17 256; do
    ../../run ./reduce_bench --check-only --n 65539 --block "$block" --grid "$grid"
  done
done
../../run ./reduce_bench --check-only --n 1048579
