#!/usr/bin/env bash
set -u

V5=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1/IA_STD_Q15_V5_SLOTTED_CONSTRAINED
PYTHON=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python
SESSION=final-lc-ia-v5
LOG="$V5/logs/IA_V5_R1.log"
WATCH="$V5/logs/IA_V5_R1_WATCHDOG.log"
restarts=0

launch() {
  tmux new-session -d -s "$SESSION" "cd '$V5' && '$PYTHON' -u code/run_ia_v5.py >> '$LOG' 2>&1"
  printf '%s launch restart=%s\n' "$(date -u +%FT%TZ)" "$restarts" >> "$WATCH"
}

worker_exists() {
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fqx "$SESSION"
}

while true; do
  if [[ -f "$V5/FINAL_RESULT.json" ]]; then
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
