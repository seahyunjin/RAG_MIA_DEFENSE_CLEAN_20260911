#!/usr/bin/env bash
set -euo pipefail

EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/ORTHOGONAL_EXPOSURE_SEARCH_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$EXP"

"$PY" code/prepare.py
"$PY" code/run_phase_a.py
"$PY" code/run_phase_b.py

if "$PY" - "$EXP/PHASE_B_RESULT.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["screen_passed"] else 1)
PY
then
  while [[ ! -x "$EXP/code/run_confirmation_chain.sh" ]]; do
    sleep 30
  done
  exec "$EXP/code/run_confirmation_chain.sh"
fi

