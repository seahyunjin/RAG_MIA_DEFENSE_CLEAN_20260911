#!/usr/bin/env bash
set -u

EXP_DIR="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/IA_STEALTH_CONFIRMATION_V1"
PYTHON_BIN="/usr/local/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

while true; do
  "$PYTHON_BIN" "$EXP_DIR/code/run_ia_st1.py" >> "$EXP_DIR/logs/IA_ST1_SUPERVISOR.log" 2>&1
  rc=$?
  if [[ -f "$EXP_DIR/FINAL_RESULT.json" ]]; then
    exit "$rc"
  fi
  if [[ -f "$EXP_DIR/IA_ST1_PREFLIGHT_RESULT.json" ]] && grep -q 'IA_STD_Q15_ST1_UNAVAILABLE' "$EXP_DIR/IA_ST1_PREFLIGHT_RESULT.json"; then
    exit "$rc"
  fi
  printf '[%s] runner exit=%s; restarting after 10 seconds\n' "$(date -u +%FT%TZ)" "$rc" >> "$EXP_DIR/logs/IA_ST1_SUPERVISOR.log"
  sleep 10
done
