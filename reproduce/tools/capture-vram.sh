#!/usr/bin/env bash
# Capture nvidia-smi + free into a rung artifact directory.
set -euo pipefail
OUT=${1:?usage: capture-vram.sh <results-subdir>}
mkdir -p "$OUT"
ts=$(date -u +%Y%m%dT%H%M%SZ)
{
  echo "ts=$ts"
  nvidia-smi
  echo "---"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
  echo "---"
  free -h
} | tee "$OUT/vram-$ts.txt"
