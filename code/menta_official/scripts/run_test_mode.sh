#!/usr/bin/env bash
# Smoke-test all configured attack pipelines with a tiny balanced dataset copy.
#
# This wrapper mirrors scripts/run_all.sh: it accepts only --online/--offline,
# builds an env file from .env, then runs the same jobs sequentially. The only
# difference is that it points DATA_DIR/RESULTS_DIR at smoke-test locations and
# forces all attack modes on a small corpus copy.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_test_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_test_mode.sh --online
  bash scripts/run_test_mode.sh --offline

Run --online first on a networked node, then --offline inside the GPU allocation.
Smoke-test settings are read from .env TEST_* variables, but Python entry points
receive normal data/result paths only. No Python --test_mode flags are used.

Defaults:
  TEST_RESULTS_DIR=results/test
  TEST_DATA_DIR=data/test
  TEST_TOTAL_DOCS=20
  TEST_DOCS_PER_SPLIT=10
  TEST_RAG_GENERATOR_MODEL=google/gemma-2b-it
  TEST_FORCE=1
EOF
}

parse_env_only_pipeline_args print_test_usage "$@"
TEST_PIPELINE_PHASE="${PIPELINE_PHASE_ARG}"

BASE_ENV_PATH="${RUN_TEST_ENV_PATH:-${REPO_ROOT}/.env}"
if [[ ! -f "${BASE_ENV_PATH}" ]]; then
    echo "Error: ${BASE_ENV_PATH} does not exist. Copy .env.example to .env first." >&2
    exit 2
fi

TMP_ENV="$(mktemp "${TMPDIR:-/tmp}/menta_test_env.XXXXXX")"
cleanup() {
    rm -f "${TMP_ENV}"
}
trap cleanup EXIT

cat > "${TMP_ENV}" <<EOF
source "${BASE_ENV_PATH}"

ORIGINAL_DATA_DIR="\${DATA_DIR:-data}"
_test_dataset_list="\${DATASETS:-BeIR_nfcorpus BeIR_scidocs BeIR_trec-covid}"
_test_dataset_list="\${_test_dataset_list//,/ }"
read -r _test_dataset _test_dataset_rest <<< "\${_test_dataset_list}"
DATASETS="\${_test_dataset:-BeIR_nfcorpus}"

RESULTS_DIR="\${TEST_RESULTS_DIR:-results/test}"
DATA_DIR="\${TEST_DATA_DIR:-data/test}"
TEST_TOTAL_DOCS="\${TEST_TOTAL_DOCS:-20}"
TEST_DOCS_PER_SPLIT="\${TEST_DOCS_PER_SPLIT:-\$(( (TEST_TOTAL_DOCS + 1) / 2 ))}"
RAG_GENERATOR_MODEL="\${TEST_RAG_GENERATOR_MODEL:-google/gemma-2b-it}"
RETRIEVER_MODEL="\${RETRIEVER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
QUERY_PARAPHRASE_METHOD="\${TEST_QUERY_PARAPHRASE_METHOD:-openai}"
QUERY_PARAPHRASE_TRANSFORMERS_MODEL="\${TEST_QUERY_PARAPHRASE_TRANSFORMERS_MODEL:-\${RAG_GENERATOR_MODEL}}"
QUERY_PARAPHRASE_MODEL="\${QUERY_PARAPHRASE_MODEL:-\${OPENAI_QUERY_MODEL:-gpt-4.1-nano}}"
QUERY_PARAPHRASE_BATCH_SIZE="\${TEST_QUERY_PARAPHRASE_BATCH_SIZE:-1}"
NUM_QUERIES="\${NUM_QUERIES:-5}"
TOP_K="\${TOP_K:-3}"
NUM_MASKS="\${NUM_MASKS:-1}"
PIPELINE_PHASE="${TEST_PIPELINE_PHASE}"
PIPELINE_SMOKE_TEST=1
FORCE="\${TEST_FORCE:-1}"

MENTA_MODES="default dp rerank prompt_instruction paraphrase similarity generic summary_only"
IA_MODES="default dp rerank prompt_instruction paraphrase"
DCMI_MODES="default dp rerank prompt_instruction paraphrase"
MBA_MODES="default dp rerank prompt_instruction paraphrase"
S2MIA_MODES="default dp rerank prompt_instruction paraphrase"
EOF

ENV_PATH="${TMP_ENV}"
export ENV_PATH
init_pipeline_env
set_hf_cache_args
# shellcheck disable=SC1090
source "${TMP_ENV}"
apply_pipeline_phase_arg

prepare_small_dataset() {
    python3 - "${ORIGINAL_DATA_DIR}" "${DATA_DIR}" "${DATASETS}" "${TEST_DOCS_PER_SPLIT}" "${TEST_TOTAL_DOCS}" <<'PY'
import json
import sys
from pathlib import Path

source_base = Path(sys.argv[1])
test_base = Path(sys.argv[2])
dataset = sys.argv[3]
docs_per_split = max(1, int(sys.argv[4]))
test_total_docs = max(1, int(sys.argv[5]))

source_ds = source_base / dataset
test_ds = test_base / dataset
if not source_ds.exists():
    raise SystemExit(f"Missing source dataset: {source_ds}")

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

member_rows = read_jsonl(source_ds / "corpus_member.jsonl")[:docs_per_split]
nonmember_rows = read_jsonl(source_ds / "corpus_nonmember.jsonl")[:docs_per_split]
if not member_rows or not nonmember_rows:
    raise SystemExit("Smoke test requires at least one member and one nonmember document")

write_jsonl(test_ds / "corpus_member.jsonl", member_rows)
write_jsonl(test_ds / "corpus_nonmember.jsonl", nonmember_rows)

member_ids = {row["_id"] for row in member_rows}
selected_ids = member_ids | {row["_id"] for row in nonmember_rows}
source_summary = source_ds / "summary.jsonl"
summary_rows = []
if source_summary.exists():
    for row in read_jsonl(source_summary):
        if row.get("_id") in selected_ids:
            summary_rows.append(row)
seen_summary_ids = {row.get("_id") for row in summary_rows}
for row in member_rows + nonmember_rows:
    doc_id = row["_id"]
    if doc_id in seen_summary_ids:
        continue
    text = " ".join(str(row.get(key, "")) for key in ("title", "text") if row.get(key)).strip()
    summary_rows.append({"_id": doc_id, "summary": text[:512]})
write_jsonl(test_ds / "summary.jsonl", summary_rows)

benign_rows = []

def benign_source_id(row):
    return str(row.get("source_id") or row.get("_id") or row.get("query_id") or "").split("::benign_", 1)[0]

source_benign = source_ds / "benign_queries.jsonl"
if source_benign.exists():
    for row in read_jsonl(source_benign):
        raw_id = benign_source_id(row)
        if raw_id not in member_ids:
            continue
        benign_rows.append({
            "_id": raw_id,
            "text": row.get("text", row.get("query", "")),
        })
        if len(benign_rows) >= test_total_docs:
            break

source_queries_dataset = source_ds / "queries_dataset.jsonl"
if len(benign_rows) < test_total_docs and source_queries_dataset.exists():
    for row in read_jsonl(source_queries_dataset):
        attack_type = str(row.get("attack_type", "")).lower()
        if row.get("is_attack") is not False and attack_type not in ("", "benign"):
            continue
        raw_id = benign_source_id(row)
        if raw_id not in member_ids:
            continue
        benign_rows.append({
            "_id": raw_id,
            "text": row.get("text", row.get("query", "")),
        })
        if len(benign_rows) >= test_total_docs:
            break

idx = 0
while len(benign_rows) < test_total_docs:
    row = member_rows[idx % len(member_rows)]
    text = " ".join(str(row.get(key, "")) for key in ("title", "text") if row.get(key)).strip()
    benign_rows.append({"_id": row["_id"], "text": text[:512]})
    idx += 1

write_jsonl(test_ds / "benign_queries.jsonl", benign_rows[:test_total_docs])

print(
    f"Prepared {test_ds} with {len(member_rows)} member and "
    f"{len(nonmember_rows)} nonmember documents"
)
PY
}

prepare_test_data() {
    local markers=(
        "${DATA_DIR}/${DATASETS}/corpus_member.jsonl"
        "${DATA_DIR}/${DATASETS}/corpus_nonmember.jsonl"
        "${DATA_DIR}/${DATASETS}/summary.jsonl"
        "${DATA_DIR}/${DATASETS}/benign_queries.jsonl"
    )
    if skip_if_all_exist "Preparing small smoke-test dataset" "${markers[@]}"; then
        return
    fi
    prepare_small_dataset
}

prepare_offline_index() {
    local retriever_tag="${RETRIEVER_MODEL//\//--}"
    local index_path="${DATA_DIR}/${DATASETS}/faiss_indices/${retriever_tag}.faiss"
    if skip_if_exists "${index_path}" "Building smoke-test FAISS index"; then
        return
    fi
    python utils/create_index.py \
        --model "${RETRIEVER_MODEL}" \
        --data_base "${DATA_DIR}" \
        --datasets "${DATASETS}" \
        "${HF_CACHE_ARGS[@]}" \
        --use_gpu
}

run_job() {
    local label="$1"
    local script_path="$2"

    echo ""
    echo "=================================================="
    echo "Smoke test: ${label} (${PIPELINE_PHASE})"
    echo "=================================================="

    ENV_PATH="${TMP_ENV}" bash "${script_path}" "--${PIPELINE_PHASE}"
}

echo "Smoke-test env: ${TMP_ENV}"
echo "Dataset: ${DATASETS}"
echo "Small data: ${DATA_DIR}/${DATASETS} (${TEST_DOCS_PER_SPLIT} member + ${TEST_DOCS_PER_SPLIT} nonmember docs)"
echo "Results: ${RESULTS_DIR}"
print_pipeline_phase_summary

prepare_test_data
if [[ "${PIPELINE_PHASE}" == "offline" ]]; then
    prepare_offline_index
fi

run_job "MEntA" "${REPO_ROOT}/scripts/run_menta_pipeline.sh"
run_job "IA-MIA" "${REPO_ROOT}/scripts/run_ia_pipeline.sh"
run_job "DCMI" "${REPO_ROOT}/scripts/run_dcmi_pipeline.sh"
run_job "MBA" "${REPO_ROOT}/scripts/run_mba_pipeline.sh"
run_job "S2-MIA" "${REPO_ROOT}/scripts/run_s2mia_pipeline.sh"
run_job "Query detection defenses" "${REPO_ROOT}/scripts/run_input_detection_defenses.sh"

echo ""
echo "Smoke-test ${PIPELINE_PHASE} phase finished."
