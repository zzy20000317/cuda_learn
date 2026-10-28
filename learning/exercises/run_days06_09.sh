#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for item in day06:test_reduce_shfl.py day07:test_review.py day08:test_transpose.py day09:test_padding.py; do
  day="${item%%:*}"; script="${item#*:}"
  cd "$ROOT/$day"
  ../../run python "$script" > results/test.log 2>&1
  printf '%s PASS\n' "$day"
done

cd "$ROOT"
../run python render_days06_09_reports.py
