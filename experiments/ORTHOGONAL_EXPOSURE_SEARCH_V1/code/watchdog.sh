#!/usr/bin/env bash
set -u

EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/ORTHOGONAL_EXPOSURE_SEARCH_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
SESSION="orthogonal-exposure-v1"

while true; do
  if [[ -f "$EXP/confirmation/e2e/E2E_RESULT.json" ]]; then
    exit 0
  fi
  if [[ -f "$EXP/confirmation/CONFIRMATION_RESULT.json" ]]; then
    if "$PY" - "$EXP/confirmation/CONFIRMATION_RESULT.json" <<'PY'
import json,sys
raise SystemExit(0 if not json.load(open(sys.argv[1]))["e2e_allowed"] else 1)
PY
    then
      exit 0
    fi
  fi
  if [[ -f "$EXP/PHASE_B_RESULT.json" ]]; then
    if "$PY" - "$EXP/PHASE_B_RESULT.json" <<'PY'
import json,sys
raise SystemExit(0 if not json.load(open(sys.argv[1]))["screen_passed"] else 1)
PY
    then
      exit 0
    fi
  fi
  if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    if [[ -f "$EXP/confirmation/CONFIRMATION_RESULT.json" ]]; then
      command="cd '$EXP' && set -o pipefail && '$PY' code/run_e2e.py 2>&1 | tee -a logs/RUN.log"
    elif [[ -f "$EXP/PHASE_B_RESULT.json" ]]; then
      command="cd '$EXP' && set -o pipefail && ./code/run_confirmation_chain.sh 2>&1 | tee -a logs/RUN.log"
    elif [[ -f "$EXP/PHASE_A_RESULT.json" ]]; then
      command="cd '$EXP' && set -o pipefail && '$PY' code/run_phase_b.py 2>&1 | tee -a logs/RUN.log"
    else
      command="cd '$EXP' && set -o pipefail && '$PY' code/run_phase_a.py 2>&1 | tee -a logs/RUN.log && '$PY' code/run_phase_b.py 2>&1 | tee -a logs/RUN.log"
    fi
    tmux new-session -d -s "$SESSION" "$command"
  fi
  sleep 30
done
