#!/usr/bin/env bash
set -euo pipefail
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
cd "$EXP"
exec >>"$EXP/logs/CORE5_RESUME.log" 2>&1
echo "[$(date -u +%FT%TZ)] Core5 resume start"
[[ -f configs/CORE5_RESUME_PRECOMMIT.json ]] || "$PY" code/prepare_core5_resume.py
[[ -f PHASE_A_RESULT.json ]] || "$PY" code/run_core5_detection.py
[[ -f configs/CORE5_MATCHED_BUDGET_E2E_PRECOMMIT.json ]] || "$PY" code/prepare_core5_phase_b.py
[[ -f runtime/PHASE_B_GENERATION_MANIFEST.json ]] || "$PY" code/run_core5_generation.py
[[ -f PHASE_B_RESULT.json ]] || "$PY" code/run_core5_scoring.py
[[ -f configs/GOLD_QA_PRECOMMIT.json ]] || "$PY" code/prepare_phase_c.py
[[ -f PHASE_C_RESULT.json ]] || "$PY" code/run_phase_c_gold.py
echo "[$(date -u +%FT%TZ)] Core5 resume complete"
