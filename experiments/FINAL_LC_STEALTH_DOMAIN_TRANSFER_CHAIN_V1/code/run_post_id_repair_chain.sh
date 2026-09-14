#!/usr/bin/env bash
set -u
ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911"
EXP="$ROOT/experiments/FINAL_LC_STEALTH_DOMAIN_TRANSFER_CHAIN_V1"
IA="$EXP/IA_STD_Q15_API1";CODE="$EXP/code";CHILD="$EXP/UNTOUCHED_FINQA_V1";PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python";LOG="$EXP/logs/ID_REPAIR_CHAIN.log"
export PYTHONPATH="$CODE";export CUDA_VISIBLE_DEVICES=0
exec 9>"$EXP/runtime/id-repair-chain.lock";flock -n 9 || exit 0
stamp(){ date -u +"[%Y-%m-%dT%H:%M:%SZ]"; };note(){ printf "%s %s\n" "$(stamp)" "$*" >> "$LOG"; };run(){ s="$1";shift;note "START $s";"$@" >> "$LOG" 2>&1;rc=$?;note "END $s rc=$rc";return "$rc"; }
run ID_REPAIR_PRECOMMIT "$PY" "$CODE/prepare_phase_c_detection_id_repair.py" || exit 51
run IA_DETECTION_ID_REPAIR "$PY" "$CODE/run_phase_c_detection_id_repair.py" || exit 52
grep -q '"verdict": "IA_STEALTH_DETECTION_PASS"' "$IA/post_ready/IA_API1_DETECTION_RESULT.json" || { note "IA_STEALTH_DETECTION_NOT_SUPPORTED; stop";exit 0; }
[[ -f "$IA/post_ready/configs/IA_API1_E2E_PRECOMMIT.json" ]] || run IA_E2E_PREPARE "$PY" "$CODE/prepare_phase_c_e2e.py" || exit 53
[[ -f "$IA/post_ready/runtime/IA_ST1_GENERATION_MANIFEST.json" ]] || run IA_E2E_GENERATION "$PY" "$CODE/run_phase_c_e2e_generation.py" || exit 54
[[ -f "$IA/post_ready/IA_API1_E2E_RESULT.json" ]] || run IA_E2E_SCORING "$PY" "$CODE/run_phase_c_e2e_scoring.py" || exit 55
grep -q '"verdict": "IA_STEALTH_CONFIRMATION_PASS"' "$IA/post_ready/IA_API1_E2E_RESULT.json" || { note "IA_STEALTH_E2E_FAILED; untouched domain not opened";exit 0; }
note "ALL THREE IA GATES PASS; FinQA remains sealed pending SciDocs/TREC-COVID, conformal control, few-shot adaptation, k sensitivity, and deployment precommit"
NEXT="$EXP/PRE_FINQA_DEPLOYMENT_PROTOCOL_V1/code/run_pre_finqa_chain.sh"
[[ -f "$NEXT" ]] || { note "PRE_FINQA_CHAIN_NOT_READY; stop safely without opening FinQA"; exit 0; }
exec bash "$NEXT"
