#!/usr/bin/env bash
set -euo pipefail

campaign=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/CLEAN_CORE3_DEV_V1
python_bin=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python

while tmux has-session -t clean-core3-generation 2>/dev/null; do
  sleep 20
done

if [[ ! -f "$campaign/runtime/CLEAN_CORE3_GENERATION_MANIFEST.json" ]]; then
  printf '%s\n' 'GENERATION_INCOMPLETE: final evaluation was not started.'
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$python_bin" \
  "$campaign/code/run_final_evaluation.py"
