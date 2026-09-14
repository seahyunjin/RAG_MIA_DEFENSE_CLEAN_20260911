#!/usr/bin/env bash
# Run all non-test pipelines configured in .env.
#
# This wrapper intentionally ignores smoke-test controls. Use
# scripts/run_test_mode.sh for small test runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_run_all_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_all.sh --online
  bash scripts/run_all.sh --offline

Run --online first on a networked node. It runs all OpenAI/API stages for every
attack configured in .env, then builds/runs the GPT query detector.

Run --offline inside the GPU allocation. It resumes all local/GPU stages for
every attack configured in .env, then runs the Mirabel query detector.

All datasets, modes, models, and defense knobs are read from .env. TEST_*
settings are ignored; use scripts/run_test_mode.sh for smoke tests.
Set RUN_ALL_ENV_PATH=/path/to/env only if you intentionally want a non-default env file.
EOF
}

clear_run_all_test_env() {
    unset PIPELINE_PHASE_OVERRIDE PIPELINE_SMOKE_TEST
    unset TEST_RESULTS_DIR TEST_DATA_DIR TEST_TOTAL_DOCS TEST_DOCS_PER_SPLIT
    unset TEST_RAG_GENERATOR_MODEL TEST_FORCE TEST_DCMI_PERTURBED_DATA_DIR
    unset TEST_QUERY_PARAPHRASE_METHOD TEST_QUERY_PARAPHRASE_MODEL
    unset TEST_QUERY_PARAPHRASE_TRANSFORMERS_MODEL TEST_QUERY_PARAPHRASE_BATCH_SIZE
}

parse_env_only_pipeline_args print_run_all_usage "$@"
clear_run_all_test_env
ENV_PATH="${RUN_ALL_ENV_PATH:-${REPO_ROOT}/.env}"
if [[ ! -f "${ENV_PATH}" ]]; then
    echo "Error: run_all env file not found: ${ENV_PATH}" >&2
    exit 2
fi
export ENV_PATH
init_pipeline_env
PIPELINE_PHASE="${PIPELINE_PHASE:-all}"
apply_pipeline_phase_arg

if [[ "${PIPELINE_PHASE}" != "online" && "${PIPELINE_PHASE}" != "offline" ]]; then
    echo "Error: run_all.sh requires --online or --offline." >&2
    print_run_all_usage >&2
    exit 2
fi

run_job() {
    local label="$1"
    local script_path="$2"

    echo ""
    echo "=================================================="
    echo "${label} (${PIPELINE_PHASE})"
    echo "=================================================="

    (
        clear_run_all_test_env
        export ENV_PATH
        bash "${script_path}" "--${PIPELINE_PHASE}"
    )
}

echo "Running all configured non-test pipelines from ${ENV_PATH:-.env}."
print_pipeline_phase_summary

run_job "MEntA" "${REPO_ROOT}/scripts/run_menta_pipeline.sh"
run_job "IA-MIA" "${REPO_ROOT}/scripts/run_ia_pipeline.sh"
run_job "DCMI" "${REPO_ROOT}/scripts/run_dcmi_pipeline.sh"
run_job "MBA" "${REPO_ROOT}/scripts/run_mba_pipeline.sh"
run_job "S2-MIA" "${REPO_ROOT}/scripts/run_s2mia_pipeline.sh"
run_job "Query detection defenses" "${REPO_ROOT}/scripts/run_input_detection_defenses.sh"

echo ""
echo "run_all.sh ${PIPELINE_PHASE} phase finished."
