#!/usr/bin/env bash
set -euo pipefail

EXP=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CHURN_AND_FP_SAFE_V1
PY=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python
cd "$EXP"

if [[ ! -f configs/DB_CHURN_V2_PRECOMMIT.json ]]; then
  "$PY" -u code/prepare_campaign.py
fi

if [[ ! -f audits/CURRENT_HIDE_PACKING_POLICY.json ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u code/audit_hide_packing.py >> logs/PACKING_AUDIT.log 2>&1 &
  packing_pid=$!
else
  packing_pid=""
fi

if [[ ! -f cache/BGE_EMBEDDING_UNION_MANIFEST.json ]]; then
  "$PY" -u code/embed_missing_union.py >> logs/BGE_EMBEDDING.log 2>&1
fi

if [[ ! -f DB_CHURN_V2_RESULT.json ]]; then
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    "$PY" -u code/run_churn_scoring.py >> logs/CHURN_SCORING.log 2>&1
fi

if [[ -n "$packing_pid" ]]; then
  wait "$packing_pid"
fi

"$PY" -u code/finalize_campaign.py >> logs/FINALIZE.log 2>&1

