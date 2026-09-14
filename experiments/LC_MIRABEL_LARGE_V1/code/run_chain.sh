#!/usr/bin/env bash
set -euo pipefail
cd /home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/LC_MIRABEL_LARGE_V1
PY=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python
SECRET=/home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key

if [[ ! -f configs/LC_MIRABEL_LARGE_V1_PRECOMMIT.json ]]; then
  "$PY" code/prepare_large.py
fi
if [[ ! -f audits/LARGE_MENTA_INPUT_AUDIT.json ]] || ! grep -q 'LARGE_MENTA_INPUT_PASS' audits/LARGE_MENTA_INPUT_AUDIT.json; then
  if [[ ! -f "$SECRET" ]]; then
    echo "OPENAI key file missing" >&2
    exit 20
  fi
  export OPENAI_API_KEY="$(<"$SECRET")"
  "$PY" code/generate_queries.py menta
  unset OPENAI_API_KEY
fi
if [[ ! -f audits/LARGE_LOCAL_ATTACK_INPUT_AUDIT.json ]] || ! grep -q 'LARGE_LOCAL_ATTACK_INPUT_PASS' audits/LARGE_LOCAL_ATTACK_INPUT_AUDIT.json; then
  "$PY" code/generate_queries.py local
fi
if [[ ! -f manifests/LARGE_QUERY_MANIFEST.json ]] || ! grep -q 'LARGE_QUERY_MANIFEST_PASS' manifests/LARGE_QUERY_MANIFEST.json; then
  "$PY" code/generate_queries.py freeze
fi
if [[ ! -f PHASE1_RESULT.json ]]; then
  "$PY" code/run_detection.py
fi
if ! grep -q 'LC_MIRABEL_DETECTION_PHASE1_PASS' PHASE1_RESULT.json; then
  exit 0
fi
if [[ ! -f runtime/LARGE_GENERATION_MANIFEST.json ]]; then
  "$PY" code/run_e2e_generation.py
fi
if [[ ! -f E2E_RESULT.json ]]; then
  "$PY" code/run_e2e_scoring.py
fi
if ! grep -q 'LC_MIRABEL_E2E_PASS' E2E_RESULT.json; then
  exit 0
fi
if [[ ! -f RESULT.json ]]; then
  "$PY" code/run_churn.py
fi

