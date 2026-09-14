#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_CLEAN_V1"
PYTHON="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT/code"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
"$PYTHON" code/verify_frozen.py 2>&1 | tee -a logs/supervisor.log
attempt=1
while [ "$attempt" -le 3 ]; do
  if "$PYTHON" code/run_phase1.py 2>&1 | tee -a logs/phase1_gpu.log; then
    "$PYTHON" code/analyze_phase1.py 2>&1 | tee -a logs/phase1_analysis.log
    exit 0
  else
    status=$?
  fi
  printf 'Phase1 attempt %s failed with status %s; resumable retry follows.\n' "$attempt" "$status" | tee -a logs/supervisor.log
  attempt=$((attempt + 1))
  sleep 15
done
printf 'Phase1 failed after three resumable attempts.\n' | tee -a logs/supervisor.log
exit 1
