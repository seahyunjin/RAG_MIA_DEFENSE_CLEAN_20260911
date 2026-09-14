#!/usr/bin/env bash
set -u
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
SESSION="final-lc-core5-resume"
while true; do
  if [[ -f "$EXP/PHASE_C_RESULT.json" ]]; then
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" "cd '$EXP' && bash code/run_core5_resume.sh"
    printf '[%s] restarted %s\n' "$(date -u +%FT%TZ)" "$SESSION" >> "$EXP/logs/CORE5_WATCHDOG.log"
  fi
  sleep 30
done
