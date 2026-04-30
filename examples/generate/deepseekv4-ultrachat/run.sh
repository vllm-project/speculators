#!/usr/bin/env bash
# =============================================================================
# run.sh — DeepSeek-V4-Flash UltraChat response regeneration
#
# Handles GPU reservation, server lifecycle, and response regeneration.
# Reuses scripts/response_regeneration/script.py for the actual generation.
#
# Usage:
#   bash run.sh [OPTIONS]
#
# Options:
#   --mode STR            native | docker  (default: native)
#   --hardware STR        h100 | b200      (default: h100; docker mode only)
#   --dataset STR         ultrachat | magpie (default: ultrachat)
#   --limit N             Stop after N rows — use for test runs (default: all)
#   --concurrency N       Concurrent HTTP requests (default: 64)
#   --max-tokens N        Max tokens per response (default: 8192)
#   --resume              Skip rows already written to the output file
#   --output-dir DIR      Results directory (default: results_TIMESTAMP)
#   --max-runtime N       Global deadline in hours (default: 12)
#   --reserve-duration S  chg reserve duration string (default: 2h)
#
# Quick test run — native vllm, 50 samples:
#   bash run.sh --limit 50
#
# Quick test run — Docker on H100, 50 samples:
#   bash run.sh --mode docker --hardware h100 --limit 50
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_START=$(date +%s)

# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="native"
HARDWARE="h100"
LIMIT=""
CONCURRENCY=64
MAX_TOKENS=8192
RESUME=false
OUTPUT_DIR=""  # Set after MODE and HARDWARE are parsed
MAX_RUNTIME_HOURS=12
RESERVE_DURATION="2h"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)             MODE="$2";               shift 2 ;;
        --hardware)         HARDWARE="$2";           shift 2 ;;
        --limit)            LIMIT="$2";              shift 2 ;;
        --concurrency)      CONCURRENCY="$2";        shift 2 ;;
        --max-tokens)       MAX_TOKENS="$2";         shift 2 ;;
        --resume)           RESUME=true;             shift   ;;
        --output-dir)       OUTPUT_DIR="$2";         shift 2 ;;
        --max-runtime)      MAX_RUNTIME_HOURS="$2";  shift 2 ;;
        --reserve-duration) RESERVE_DURATION="$2";   shift 2 ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

# Set default output directory after MODE and HARDWARE are parsed
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="/mnt/data/engine/rahul-tuli/deepseekv4-${HARDWARE}-${MODE}-$(date +%Y%m%d_%H%M%S)"
fi

# ---------------------------------------------------------------------------
# Validate mode / hardware and pick serve script + GPU count
# ---------------------------------------------------------------------------
case "${MODE}" in
    native)
        NUM_GPUS=8
        SERVE_SCRIPT="${SCRIPT_DIR}/serve.sh"
        ;;
    docker)
        case "${HARDWARE}" in
            h100) NUM_GPUS=8 ;;
            b200) NUM_GPUS=4 ;;
            h200) NUM_GPUS=8 ;;
            *) error "Unknown hardware: ${HARDWARE}. Valid: h100, b200, h200"; exit 1 ;;
        esac
        SERVE_SCRIPT="${SCRIPT_DIR}/serve_docker.sh"
        : "${HF_HUB_CACHE:?HF_HUB_CACHE must be set for docker mode}"
        ;;
    *) error "Unknown mode: ${MODE}. Valid: native, docker"; exit 1 ;;
esac

# Expose to lib.sh stop_server
export SERVE_MODE="${MODE}"

# ---------------------------------------------------------------------------
# Computed values
# ---------------------------------------------------------------------------
SCRIPT_DEADLINE=$(( SCRIPT_START + MAX_RUNTIME_HOURS * 3600 ))
GPU_IDS=""
RESERVATION_EXPIRY=0

SERVER_LOG="${OUTPUT_DIR}/server.log"

REGEN_SCRIPT="${SCRIPT_DIR}/../../../scripts/response_regeneration/script.py"
if [[ ! -f "${REGEN_SCRIPT}" ]]; then
    error "Regeneration script not found: ${REGEN_SCRIPT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

DEADLINE_TS=$(date -d "@${SCRIPT_DEADLINE}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
    || date -r "${SCRIPT_DEADLINE}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
    || echo "start + ${MAX_RUNTIME_HOURS}h")

banner "DeepSeek-V4-Flash Response Regeneration"
info "Mode          : ${MODE}$([[ ${MODE} == docker ]] && echo " (${HARDWARE})" || true)"
info "Datasets      : ultrachat, magpie (sequential)"
info "Output dir    : ${OUTPUT_DIR}"
info "Concurrency   : ${CONCURRENCY}"
info "Max tokens    : ${MAX_TOKENS}"
info "Reserve dur   : ${RESERVE_DURATION}"
info "Max runtime   : ${MAX_RUNTIME_HOURS}h (deadline: ${DEADLINE_TS})"
[[ -n "${LIMIT}" ]]       && info "Limit         : ${LIMIT} rows per dataset"
[[ "${RESUME}" == true ]] && info "Resume        : enabled"

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    info "Cleaning up (total elapsed: $(elapsed))..."
    stop_server 2>/dev/null || true
    release_gpus
    info "Done."
}
trap cleanup EXIT
trap exit_after_cleanup INT TERM HUP

# ---------------------------------------------------------------------------
# Acquire GPUs
# ---------------------------------------------------------------------------
banner "Acquiring ${NUM_GPUS} GPUs"
if ! wait_for_gpus "${NUM_GPUS}"; then
    error "Could not acquire GPUs before deadline."
    exit 1
fi
if ! reserve_gpus "${NUM_GPUS}" "${RESERVE_DURATION}"; then
    error "GPU reservation failed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
banner "Starting vLLM Server (${MODE})"

PORT=$(find_free_port 8000)
ENDPOINT="http://localhost:${PORT}"

info "Starting server on port ${PORT}..."
GPU_IDS="${GPU_IDS}" PORT="${PORT}" SERVER_LOG="${SERVER_LOG}" \
    HARDWARE="${HARDWARE}" \
    bash "${SERVE_SCRIPT}" > /dev/null 2>&1 &
SERVER_PID=$!
info "Server PID: ${SERVER_PID}"

if ! wait_for_server "${ENDPOINT}" 3600; then
    error "Server failed to start."
    exit 1
fi

# ---------------------------------------------------------------------------
# Run response regeneration for both datasets
# ---------------------------------------------------------------------------
for DATASET in ultrachat magpie; do
    banner "Running Response Regeneration: ${DATASET}"

    DATASET_OUTDIR="${OUTPUT_DIR}/${DATASET}"
    mkdir -p "${DATASET_OUTDIR}"
    OUTFILE="${DATASET_OUTDIR}/${DATASET}_DeepSeek-V4-Flash.jsonl"
    DATASET_LOG="${DATASET_OUTDIR}/generation.log"

    REGEN_ARGS=(
        --endpoint "${ENDPOINT}/v1/chat/completions"
        --dataset "${DATASET}"
        --concurrency "${CONCURRENCY}"
        --max-tokens "${MAX_TOKENS}"
        --outfile "${OUTFILE}"
    )
    [[ -n "${LIMIT}" ]]       && REGEN_ARGS+=(--limit "${LIMIT}")
    [[ "${RESUME}" == true ]] && REGEN_ARGS+=(--resume)

    info "Command: python script.py ${REGEN_ARGS[*]}"
    python "${REGEN_SCRIPT}" "${REGEN_ARGS[@]}" 2>&1 | tee "${DATASET_LOG}"

    info "Dataset ${DATASET} complete. Lines written: $(wc -l < "${OUTFILE}" 2>/dev/null || echo "0")"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "Complete! (total: $(elapsed))"
info "Output dir       : ${OUTPUT_DIR}"
info "UltraChat results: ${OUTPUT_DIR}/ultrachat/ultrachat_DeepSeek-V4-Flash.jsonl"
info "Magpie results   : ${OUTPUT_DIR}/magpie/magpie_DeepSeek-V4-Flash.jsonl"
info "Server log       : ${SERVER_LOG}"
