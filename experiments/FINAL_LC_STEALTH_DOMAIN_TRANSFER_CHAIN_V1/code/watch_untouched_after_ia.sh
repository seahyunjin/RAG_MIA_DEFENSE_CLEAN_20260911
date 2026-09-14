#!/usr/bin/env bash
set -u
PARENT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_STEALTH_DOMAIN_TRANSFER_CHAIN_V1"
CHILD="$PARENT/UNTOUCHED_FINQA_V1"
IA="$PARENT/IA_STD_Q15_API1"
LOG="$CHILD/logs/GATE_MONITOR.log"
STATUS="$CHILD/GATE_STATUS.md"
LOCK="$CHILD/runtime/gate-monitor.lock"
exec 9>"$LOCK"; flock -n 9 || exit 0
stamp(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }
verdict(){ local path="$1"; [[ -f "$path" ]] || { printf 'PENDING'; return; }; /home/traffic_3/workspace/miniconda3/envs/torch/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict","MISSING_VERDICT"))' "$path" 2>/dev/null || printf 'UNREADABLE'; }
write_status(){
  v1="$1"; v2="$2"; v3="$3"
  printf '# Untouched-domain gate monitor\n\n- Updated UTC: `%s`\n- IA full validity: `%s` (required `IA_STD_Q15_API1_READY`)\n- Same-FPR 2.5%% detection: `%s` (required `IA_STEALTH_DETECTION_PASS`)\n- IA E2E privacy: `%s` (required `IA_STEALTH_CONFIRMATION_PASS`)\n- Child: `FinQA / screen 100 member + 100 nonmember / strict and benign-only calibration`\n' "$(stamp)" "$v1" "$v2" "$v3" > "$STATUS"
}
printf '[%s] gate monitor started pid=%s\n' "$(stamp)" "$$" >> "$LOG"
while true; do
  v1="$(verdict "$IA/FINAL_RESULT.json")"
  v2="$(verdict "$IA/post_ready/IA_API1_DETECTION_RESULT.json")"
  v3="$(verdict "$IA/post_ready/IA_API1_E2E_RESULT.json")"
  write_status "$v1" "$v2" "$v3"
  if [[ "$v1" == "IA_STD_Q15_API1_READY" && "$v2" == "IA_STEALTH_DETECTION_PASS" && "$v3" == "IA_STEALTH_CONFIRMATION_PASS" ]]; then
    printf '[%s] all three gates PASS; launching untouched FinQA chain\n' "$(stamp)" >> "$LOG"
    exec bash "$CHILD/code/run_untouched_chain.sh"
  fi
  if ! pgrep -f "[w]atch_phase_a_resume.sh" >/dev/null 2>&1; then
    printf '[%s] parent chain ended without all three PASS; untouched domain NOT opened: %s | %s | %s\n' "$(stamp)" "$v1" "$v2" "$v3" >> "$LOG"
    printf '\n- Final action: `UNTOUCHED_DOMAIN_NOT_OPENED`\n' >> "$STATUS"
    exit 0
  fi
  sleep 60
done
