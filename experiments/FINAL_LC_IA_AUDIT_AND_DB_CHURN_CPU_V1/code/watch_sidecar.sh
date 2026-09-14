#!/usr/bin/env bash
set -u

EXP=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_IA_AUDIT_AND_DB_CHURN_CPU_V1
PYTHON=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python
SESSION=final-lc-cpu-sidecar-v1
LOG="$EXP/logs/SIDECAR.log"
WATCH="$EXP/logs/WATCHDOG.log"
restarts=0

worker_exists() {
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fqx "$SESSION"
}

launch() {
  tmux new-session -d -s "$SESSION" "cd '$EXP' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-3 '$PYTHON' -u code/run_sidecar.py >> '$LOG' 2>&1"
  printf '%s launch restart=%s\n' "$(date -u +%FT%TZ)" "$restarts" >> "$WATCH"
}

while true; do
  if [[ -f "$EXP/FINAL_RESULT.json" ]]; then
    printf '%s terminal result reached\n' "$(date -u +%FT%TZ)" >> "$WATCH"
    exit 0
  fi
  if ! worker_exists; then
    if (( restarts >= 3 )); then
      printf '%s restart limit reached\n' "$(date -u +%FT%TZ)" >> "$WATCH"
      exit 1
    fi
    restarts=$((restarts + 1)); launch
  fi
  sleep 30
done
