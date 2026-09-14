#!/usr/bin/env bash
# Download HuggingFace models configured in .env into ${HF_HOME}/hub.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup_model.sh

Reads model choices from .env and downloads HuggingFace models into HF_HUB_CACHE,
which defaults to ${HF_HOME}/hub. Run this on a networked node before running
attack scripts with --offline inside a GPU allocation.

Models included:
  RAG_GENERATOR_MODEL
  RETRIEVER_MODEL
  QUERY_PARAPHRASE_TRANSFORMERS_MODEL
  TEST_RAG_GENERATOR_MODEL when set
  TEST_QUERY_PARAPHRASE_TRANSFORMERS_MODEL when set
  MEntA entailment helper models
EOF
}

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    print_usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Error: runtime parameters must be set in .env; this script only accepts --help." >&2
    print_usage >&2
    exit 2
fi

init_pipeline_env
mkdir -p "${HF_HUB_CACHE_DIR}"
export HF_HOME HF_HUB_CACHE="${HF_HUB_CACHE_DIR}" TRANSFORMERS_CACHE="${HF_HUB_CACHE_DIR}"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

RAG_GENERATOR_MODEL="${RAG_GENERATOR_MODEL:-google/gemma-2b-it}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
QUERY_PARAPHRASE_METHOD="${QUERY_PARAPHRASE_METHOD:-openai}"
QUERY_PARAPHRASE_TRANSFORMERS_MODEL="${QUERY_PARAPHRASE_TRANSFORMERS_MODEL:-${RAG_GENERATOR_MODEL}}"

models=()
tasks=()
add_model() {
    local model="$1"
    local task="$2"
    if [[ -z "${model}" ]]; then
        return
    fi
    case "${model}" in
        gpt-*|o1*|o3*|o4*|text-*|davinci*|curie*|babbage*|ada*) return ;;
        /*|./*|../*) return ;;
    esac
    local i
    for i in "${!models[@]}"; do
        if [[ "${models[$i]}" == "${model}" && "${tasks[$i]}" == "${task}" ]]; then
            return
        fi
    done
    models+=("${model}")
    tasks+=("${task}")
}

model_cache_dir() {
    local model="$1"
    printf '%s/models--%s' "${HF_HUB_CACHE_DIR}" "${model//\//--}"
}

model_is_cached() {
    local model="$1"
    local repo_dir
    repo_dir="$(model_cache_dir "${model}")"
    if [[ ! -d "${repo_dir}" ]]; then
        return 1
    fi
    if [[ -f "${repo_dir}/config.json" ]]; then
        return 0
    fi
    if [[ -d "${repo_dir}/snapshots" ]]; then
        local snapshot
        for snapshot in "${repo_dir}/snapshots"/*; do
            if [[ -d "${snapshot}" ]] && find "${snapshot}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
                return 0
            fi
        done
    fi
    return 1
}

add_model "${RAG_GENERATOR_MODEL}" "text-generation"
add_model "${RETRIEVER_MODEL}" "embedding"
add_model "${QUERY_PARAPHRASE_TRANSFORMERS_MODEL}" "text-generation"
if [[ -n "${TEST_RAG_GENERATOR_MODEL:-}" ]]; then
    add_model "${TEST_RAG_GENERATOR_MODEL}" "text-generation"
fi
if [[ -n "${TEST_QUERY_PARAPHRASE_TRANSFORMERS_MODEL:-}" ]]; then
    add_model "${TEST_QUERY_PARAPHRASE_TRANSFORMERS_MODEL}" "text-generation"
fi

# Used by MEntA/compute_entailment.py.
add_model "Babelscape/t5-base-summarization-claim-extractor" "seq2seq"
add_model "tasksource/deberta-base-long-nli" "sequence-classification"

if [[ ${#models[@]} -eq 0 ]]; then
    echo "No HuggingFace models selected for download."
    exit 0
fi

pending_models=()
pending_tasks=()
for i in "${!models[@]}"; do
    model="${models[$i]}"
    task="${tasks[$i]}"
    if model_is_cached "${model}"; then
        echo "[SKIP] ${model} (${task}) already exists in $(model_cache_dir "${model}")"
        continue
    fi
    pending_models+=("${model}")
    pending_tasks+=("${task}")
done

if [[ ${#pending_models[@]} -eq 0 ]]; then
    echo "All configured HuggingFace models are already available in the hub cache."
    exit 0
fi

echo "HF_HOME: ${HF_HOME:-}"
echo "HF hub cache: ${HF_HUB_CACHE_DIR}"
echo "Downloading ${#pending_models[@]} model(s) ..."

python3 - "${HF_HUB_CACHE_DIR}" "${pending_models[@]}" -- "${pending_tasks[@]}" <<'PYDL'
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1]).expanduser().resolve()
sep = sys.argv.index("--")
models = sys.argv[2:sep]
tasks = sys.argv[sep + 1:]

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'huggingface_hub'. Run this inside the project "
        "Apptainer/container environment or install the HuggingFace dependencies."
    ) from exc

for model, task in zip(models, tasks):
    print(f"[download] {model} ({task}) -> {cache_dir}", flush=True)
    snapshot_download(
        repo_id=model,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )

print("All configured HuggingFace models are available in the hub cache.")
PYDL
