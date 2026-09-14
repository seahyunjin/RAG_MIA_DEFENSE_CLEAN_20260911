#!/usr/bin/env bash
set -u

EXP=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CHURN_AND_FP_SAFE_V1
SESSION=final-lc-churn-fp-safe-v1
WATCH="$EXP/logs/WATCHDOG.log"
restarts=0

worker_exists() {
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fqx "$SESSION"
}

launch() {
  tmux new-session -d -s "$SESSION" "cd '$EXP' && bash code/run_chain.sh >> '$EXP/logs/CHAIN.log' 2>&1"
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
    restarts=$((restarts + 1))
    launch
  fi
  sleep 20
done
