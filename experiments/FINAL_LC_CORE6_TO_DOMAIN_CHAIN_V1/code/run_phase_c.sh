#!/usr/bin/env bash
set -euo pipefail
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$EXP"
"$PY" code/prepare_phase_c.py
"$PY" code/run_phase_c_gold.py
if "$PY" - <<'PY'
import json,sys
sys.exit(0 if json.load(open('PHASE_C_RESULT.json'))['verdict']=='GOLD_QA_UTILITY_PASS' else 1)
PY
then
  while [[ ! -x code/run_phase_d.sh ]]; do sleep 30; done
  code/run_phase_d.sh
else
  echo "[$(date -u +%FT%TZ)] Phase C failed; domain stage prohibited" >> logs/RUN.log
fi
