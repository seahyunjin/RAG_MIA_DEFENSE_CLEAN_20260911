#!/usr/bin/env bash
set -euo pipefail
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$EXP"
"$PY" code/prepare_phase_b.py
"$PY" code/run_phase_b_generation.py
"$PY" code/run_phase_b_scoring.py
if "$PY" - <<'PY'
import json,sys
sys.exit(0 if json.load(open('PHASE_B_RESULT.json'))['verdict']=='CORE6_MATCHED_BUDGET_E2E_PASS' else 1)
PY
then
  while [[ ! -x code/run_phase_c.sh ]]; do
    echo "[$(date -u +%FT%TZ)] Phase B passed; waiting for precommitted Phase-C runner" >> logs/RUN.log
    sleep 30
  done
  code/run_phase_c.sh
else
  echo "[$(date -u +%FT%TZ)] Phase B failed; mandatory stop" >> logs/RUN.log
fi
