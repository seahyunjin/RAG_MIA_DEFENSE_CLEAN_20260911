#!/usr/bin/env bash
set -euo pipefail

BASE="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
SECRET_FILE="/home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"

# Do not touch the active first pass. Wait until its detached session exits.
while tmux has-session -t '=dualtail-v2-menta' 2>/dev/null; do
  sleep 30
done

for attempt in 1 2 3; do
  if "$PY" - "$BASE" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1]) / "audits/FRESH_MENTA_INPUT_AUDIT.json"
if not path.is_file():
    raise SystemExit(1)
value = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if value.get("verdict") == "FRESH_MENTA_INPUT_PASS" else 1)
PY
  then
    exit 0
  fi

  "$PY" - "$BASE" "$attempt" <<'PY'
import csv
import hashlib
import json
import pathlib
import shutil
import sys

base = pathlib.Path(sys.argv[1])
attempt = sys.argv[2]
audit_path = base / "audits/FRESH_MENTA_INPUT_AUDIT.json"
if not audit_path.is_file():
    raise SystemExit("MEntA audit missing after first pass")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
target_ids = [row["target_id"] for row in audit.get("invalid", [])]
history = base / "runtime/menta_api/history"
history.mkdir(parents=True, exist_ok=True)
for target_id in target_ids:
    tag = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:16]
    # Preserve the invalid response, then permit an exact same-protocol retry.
    source = base / "runtime/menta_api" / f"{tag}_questions.json"
    if source.is_file():
        destination = history / f"{tag}_questions.retry{attempt}.invalid.json"
        shutil.move(source, destination)
print(json.dumps({"retry": attempt, "invalid_targets": target_ids}, ensure_ascii=False))
PY

  export OPENAI_API_KEY="$(<"$SECRET_FILE")"
  "$PY" "$BASE/code/generate_fresh_queries.py" menta >> "$BASE/logs/menta_query_generation_retry.log" 2>&1 || true
done

exit 1
