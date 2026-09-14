#!/usr/bin/env bash
set -u

# Operational watchdog only. This file is deliberately outside the frozen
# IA protocol/code manifest and never changes prompts, parsing, scoring, or
# already-issued API calls.
EXP_DIR="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/IA_STEALTH_CONFIRMATION_V1"
RUNNER="$EXP_DIR/code/run_ia_st1_full.py"
PYTHON_BIN="/usr/local/bin/python"
WATCH_LOG="$EXP_DIR/logs/IA_ST1_FULL_WATCHDOG.log"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

while true; do
  if [[ -f "$EXP_DIR/FINAL_RESULT.json" ]]; then
    printf '[%s] final result present; watchdog exiting\n' "$(date -u +%FT%TZ)" >> "$WATCH_LOG"
    exit 0
  fi

  if ! pgrep -f "[r]un_ia_st1_full.py" >/dev/null 2>&1; then
    printf '[%s] full runner absent; restarting frozen runner\n' "$(date -u +%FT%TZ)" >> "$WATCH_LOG"
    "$PYTHON_BIN" "$RUNNER" >> "$EXP_DIR/logs/IA_ST1.log" 2>&1
    rc=$?
    printf '[%s] restarted runner exited rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$WATCH_LOG"
  fi

  sleep 15
done
