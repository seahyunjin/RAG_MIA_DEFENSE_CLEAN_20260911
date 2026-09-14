#!/usr/bin/env bash
set -euo pipefail
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$EXP"
exec >>"$EXP/logs/RUN.log" 2>&1
echo "[$(date -u +%FT%TZ)] chain start"
"$PY" code/prepare_phase_a.py
"$PY" code/generate_standardized_queries.py
"$PY" code/run_phase_a_detection.py
if "$PY" - <<'PY'
import json,sys
r=json.load(open('PHASE_A_RESULT.json'))
sys.exit(0 if r['verdict']=='CORE6_DETECTION_PASS' else 1)
PY
then
  while [[ ! -x code/run_phase_b.sh ]]; do
    echo "[$(date -u +%FT%TZ)] Phase A passed; waiting for precommitted Phase-B runner"
    sleep 30
  done
  code/run_phase_b.sh
else
  echo "[$(date -u +%FT%TZ)] Phase A failed; mandatory stop"
fi
"$PY" code/finalize_chain.py
echo "[$(date -u +%FT%TZ)] chain end"
