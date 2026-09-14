#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/DV_LOO_V2_PHASE1
PY=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python
cd "$ROOT"
mkdir -p logs checkpoints private tables reports audits configs
exec "$PY" -u code/run_phase1.py --device cuda >> logs/phase1.log 2>&1
