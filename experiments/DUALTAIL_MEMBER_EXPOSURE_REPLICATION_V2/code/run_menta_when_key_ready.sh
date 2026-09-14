#!/usr/bin/env bash
set -euo pipefail

SECRET_FILE="/home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key"
BASE="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"

while [[ ! -s "$SECRET_FILE" ]]; do
  sleep 30
done

if [[ "$(stat -c '%a' "$SECRET_FILE")" != "600" ]]; then
  chmod 600 "$SECRET_FILE"
fi
export OPENAI_API_KEY="$(<"$SECRET_FILE")"
cd "/home/traffic_3/workspace/workspace/SH"
exec "$PY" "$BASE/code/generate_fresh_queries.py" menta
