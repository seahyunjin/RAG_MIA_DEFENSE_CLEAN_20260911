#!/usr/bin/env bash
set -euo pipefail

campaign_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
audit_tmp="$(mktemp -d)"
cleanup() {
  rm -rf -- "${audit_tmp}"
}
trap cleanup EXIT

git clone --quiet https://github.com/Xinyu140203/RAG_MIA.git "${audit_tmp}/dcmi"
git clone --quiet https://github.com/ali7naseh/RAG_MIA.git "${audit_tmp}/ia"

python "${campaign_dir}/code/run_protocol_recovery.py" \
  --dcmi-repo "${audit_tmp}/dcmi" \
  --ia-repo "${audit_tmp}/ia"
python "${campaign_dir}/code/test_protocol_recovery.py"
