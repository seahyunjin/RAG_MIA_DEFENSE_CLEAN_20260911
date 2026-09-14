#!/usr/bin/env bash
set -euo pipefail

BASE="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"

while true; do
  if "$PY" - "$BASE" <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
local_path = base / "audits/FRESH_LOCAL_ATTACK_INPUT_AUDIT.json"
menta_path = base / "audits/FRESH_MENTA_INPUT_AUDIT.json"
if not local_path.is_file() or not menta_path.is_file():
    raise SystemExit(1)
local = json.loads(local_path.read_text(encoding="utf-8"))
menta = json.loads(menta_path.read_text(encoding="utf-8"))
if local.get("verdict") != "FRESH_MBA_RAG_MIA_INPUT_PASS":
    raise SystemExit(1)
if menta.get("verdict") != "FRESH_MENTA_INPUT_PASS":
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 30
done

cd "/home/traffic_3/workspace/workspace/SH"
exec "$PY" "$BASE/code/run_phase1_replication.py"
