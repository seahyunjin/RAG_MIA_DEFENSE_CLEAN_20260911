#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_8ATTACK_E2E_FRAMEWORK_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
export PYTHONPATH="${ROOT}/code"

cd "${ROOT}"
"${PY}" code/phase_a_protocol_audit.py
"${PY}" code/prepare_freeze.py
"${PY}" code/run_retrieval.py
"${PY}" code/run_generation.py
"${PY}" code/run_scoring.py

