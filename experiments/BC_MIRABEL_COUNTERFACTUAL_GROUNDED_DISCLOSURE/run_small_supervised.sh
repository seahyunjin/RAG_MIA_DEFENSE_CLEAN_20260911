#!/usr/bin/env bash
set -u

ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/BC_MIRABEL_COUNTERFACTUAL_GROUNDED_DISCLOSURE"
PYTHON="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
LOG="$ROOT/logs/SMALL_E2E.log"

mkdir -p "$ROOT/logs"
attempt=1
maximum_attempts=3
while [ "$attempt" -le "$maximum_attempts" ]; do
  printf '%s supervisor attempt=%s/%s\n' "$(date -u +%FT%TZ)" "$attempt" "$maximum_attempts" >> "$LOG"
  "$PYTHON" -u "$ROOT/code/run_small.py" >> "$LOG" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    printf '%s supervisor completed status=0\n' "$(date -u +%FT%TZ)" >> "$LOG"
    exit 0
  fi
  printf '%s supervisor failure status=%s\n' "$(date -u +%FT%TZ)" "$status" >> "$LOG"
  attempt=$((attempt + 1))
done

printf '%s supervisor exhausted retries\n' "$(date -u +%FT%TZ)" >> "$LOG"
exit 1
