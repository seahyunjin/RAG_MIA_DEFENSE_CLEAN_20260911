#!/usr/bin/env bash
set -euo pipefail
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$EXP"
"$PY" code/prepare_phase_d.py
"$PY" code/run_phase_d_queries.py --scope screen
"$PY" code/run_phase_d.py --scope screen
if "$PY" - <<'PY'
import json,sys
sys.exit(0 if json.load(open('PHASE_D_SCREEN_DETECTION_RESULT.json'))['verdict']=='PHASE_D_SCREEN_DETECTION_PASS' else 1)
PY
then
  "$PY" code/run_phase_d_queries.py --scope all
  "$PY" code/run_phase_d.py --scope all
else
  echo "[$(date -u +%FT%TZ)] Phase D screen detection failed; mandatory stop" >> logs/RUN.log
  exit 0
fi
if "$PY" - <<'PY'
import json,sys
sys.exit(0 if json.load(open('PHASE_D_ALL_DETECTION_RESULT.json'))['verdict']=='PHASE_D_ALL_DETECTION_PASS' else 1)
PY
then
  "$PY" code/run_phase_d_e2e.py
else
  echo "[$(date -u +%FT%TZ)] Phase D full detection failed; E2E prohibited" >> logs/RUN.log
fi
