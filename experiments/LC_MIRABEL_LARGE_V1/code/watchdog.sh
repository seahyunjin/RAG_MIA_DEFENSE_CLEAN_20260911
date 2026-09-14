#!/usr/bin/env bash
set -u
EXP=/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/LC_MIRABEL_LARGE_V1
SESSION=lc-mirabel-large-v1
attempts=0
while true; do
  if [[ -f "$EXP/RESULT.json" ]] || grep -qsE 'LC_MIRABEL_DETECTION_NOT_SUPPORTED|LC_MIRABEL_PRIVACY_FAILED|LARGE_.*INCOMPATIBLE|OPENAI_API_KEY_REQUIRED' "$EXP"/PHASE1_RESULT.json "$EXP"/E2E_RESULT.json "$EXP"/audits/*.json; then
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    attempts=$((attempts+1))
    if (( attempts > 3 )); then
      printf '%s watchdog stopped after 3 restart attempts\n' "$(date -u +%FT%TZ)" >> "$EXP/logs/WATCHDOG.log"
      exit 1
    fi
    tmux new-session -d -s "$SESSION" "cd '$EXP' && ./code/run_chain.sh >> logs/RUN.log 2>&1"
    printf '%s restarted main chain attempt=%s\n' "$(date -u +%FT%TZ)" "$attempts" >> "$EXP/logs/WATCHDOG.log"
  fi
  sleep 30
done

