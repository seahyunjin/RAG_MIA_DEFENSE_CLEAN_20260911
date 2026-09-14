#!/usr/bin/env bash
set -u

V6=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1/IA_STD_Q15_V6_DUPLICATE_CONSTRAINED
PYTHON=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python
SESSION=final-lc-ia-v6
LOG="$V6/logs/IA_V6.log"
WATCH="$V6/logs/IA_V6_WATCHDOG.log"
restarts=0

worker_exists() {
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fqx "$SESSION"
}

launch() {
  tmux new-session -d -s "$SESSION" "cd '$V6' && '$PYTHON' -u code/run_ia_v6.py >> '$LOG' 2>&1"
  printf '%s launch restart=%s\n' "$(date -u +%FT%TZ)" "$restarts" >> "$WATCH"
}

while true; do
  if [[ -f "$V6/FINAL_RESULT.json" ]]; then
    printf '%s terminal result reached\n' "$(date -u +%FT%TZ)" >> "$WATCH"
    exit 0
  fi
  if ! worker_exists; then
    if (( restarts >= 3 )); then
      printf '%s restart limit reached\n' "$(date -u +%FT%TZ)" >> "$WATCH"
      exit 1
    fi
    restarts=$((restarts + 1))
    launch
  fi
  sleep 20
done
