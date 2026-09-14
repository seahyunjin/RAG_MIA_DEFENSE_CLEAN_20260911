#!/usr/bin/env bash
set -euo pipefail
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/ORTHOGONAL_EXPOSURE_SEARCH_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$EXP"
"$PY" code/run_confirmation.py prepare
"$PY" code/run_confirmation.py local
export OPENAI_API_KEY="$(tr -d '\r\n' < /home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key)"
"$PY" code/run_confirmation.py menta
unset OPENAI_API_KEY
"$PY" code/run_confirmation.py score
if "$PY" - "$EXP/confirmation/CONFIRMATION_RESULT.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["e2e_allowed"] else 1)
PY
then
  exec "$PY" code/run_e2e.py
fi

