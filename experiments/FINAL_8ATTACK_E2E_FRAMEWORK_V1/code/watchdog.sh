#!/usr/bin/env bash
set -u

ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_8ATTACK_E2E_FRAMEWORK_V1"
LOG="${ROOT}/logs/watchdog.log"
MAX_RESTARTS=20
attempt=0

mkdir -p "${ROOT}/logs"
while (( attempt < MAX_RESTARTS )); do
    attempt=$((attempt + 1))
    printf '%s attempt=%d\n' "$(date -u +%FT%TZ)" "${attempt}" >> "${LOG}"
    bash "${ROOT}/code/run_chain.sh" >> "${ROOT}/logs/run.log" 2>&1
    status=$?
    if (( status == 0 )); then
        printf '%s complete\n' "$(date -u +%FT%TZ)" >> "${LOG}"
        exit 0
    fi
    printf '%s exit=%d; retrying in 30s\n' "$(date -u +%FT%TZ)" "${status}" >> "${LOG}"
    sleep 30
done
printf '%s restart_limit_reached\n' "$(date -u +%FT%TZ)" >> "${LOG}"
exit 1

