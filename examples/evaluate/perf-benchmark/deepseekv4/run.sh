#!/usr/bin/env bash
# =============================================================================
# run.sh — DeepSeek-V4-Flash acceptance rate evaluation controller
#
# Kick off and forget.  Handles GPU reservation, server lifecycle, and
# per-subset acceptance rate capture with automatic retries until a global
# deadline.
#
# Usage:
#   bash run.sh --hardware h100 [OPTIONS]
#   bash run.sh --hardware b200 --subsets "HumanEval,qa" --max-runtime 4
#
# Options:
#   --hardware TYPE         Required. h100 | b200
#   --subsets A,B,...        default: all 9 benchmark subsets
#   --output-dir DIR        default: results_TIMESTAMP
#   --max-requests N        default: 200 (requests per subset)
#   --max-concurrency N     default: 128
#   --max-runtime N         default: 8 (hours; global deadline)
#   --reserve-duration STR  default: 2h (chg reserve duration)
#   --gen-kwargs JSON       e.g. '{"temperature":0.6}'
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_START=$(date +%s)

# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HARDWARE=""
DATASET="${DATASET:-RedHatAI/speculator_benchmarks}"
SUBSETS="${SUBSETS:-HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing}"
OUTPUT_DIR="${OUTPUT_DIR:-results_$(date +%Y%m%d_%H%M%S)}"
MAX_REQUESTS="${MAX_REQUESTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-8}"
RESERVE_DURATION="${RESERVE_DURATION:-2h}"
GEN_KWARGS="${GEN_KWARGS:-}"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --hardware)         HARDWARE="$2";           shift 2 ;;
        --subsets)          SUBSETS="$2";             shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";         shift 2 ;;
        --max-requests)     MAX_REQUESTS="$2";       shift 2 ;;
        --max-concurrency)  MAX_CONCURRENCY="$2";    shift 2 ;;
        --max-runtime)      MAX_RUNTIME_HOURS="$2";  shift 2 ;;
        --reserve-duration) RESERVE_DURATION="$2";   shift 2 ;;
        --gen-kwargs)       GEN_KWARGS="$2";         shift 2 ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

: "${HARDWARE:?--hardware is required (h100 or b200)}"

# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------
case "${HARDWARE}" in
    h100)  NUM_GPUS=8; SERVE_SCRIPT="${SCRIPT_DIR}/serve_h100.sh" ;;
    b200)  NUM_GPUS=4; SERVE_SCRIPT="${SCRIPT_DIR}/serve_b200.sh" ;;
    *)     error "Unknown hardware: ${HARDWARE}. Valid: h100, b200"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Computed values
# ---------------------------------------------------------------------------
SCRIPT_DEADLINE=$(( SCRIPT_START + MAX_RUNTIME_HOURS * 3600 ))
ACCEPTANCE_DIR="${OUTPUT_DIR}/acceptance"
ACCEPTANCE_RATES_FILE="${OUTPUT_DIR}/acceptance_rates.json"
SERVER_LOG="${OUTPUT_DIR}/server.log"
PORT=8000
GPU_IDS=""
RESERVATION_EXPIRY=0

mkdir -p "${ACCEPTANCE_DIR}"

IFS=',' read -ra SUBSET_ARRAY <<< "${SUBSETS}"
N_SUBSETS=${#SUBSET_ARRAY[@]}

DEADLINE_TS=$(date -d "@${SCRIPT_DEADLINE}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
    || date -r "${SCRIPT_DEADLINE}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
    || echo "start + ${MAX_RUNTIME_HOURS}h")

banner "DeepSeek-V4-Flash Acceptance Rate Eval"
info "Hardware       : ${HARDWARE} (${NUM_GPUS} GPUs)"
info "Output dir     : ${OUTPUT_DIR}"
info "Subsets (${N_SUBSETS})   : ${SUBSETS}"
info "Max requests   : ${MAX_REQUESTS} per subset"
info "Reserve duration: ${RESERVE_DURATION}"
info "Max runtime    : ${MAX_RUNTIME_HOURS}h (deadline: ${DEADLINE_TS})"
[[ -n "${GEN_KWARGS}" ]] && info "Gen kwargs     : ${GEN_KWARGS}"

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    info "Cleaning up (total elapsed: $(elapsed))..."
    stop_server "${PORT}" 2>/dev/null || true
    release_gpus
    info "Done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Acquire GPUs
# ---------------------------------------------------------------------------
acquire_gpus() {
    banner "Acquiring ${NUM_GPUS} GPUs"
    if ! wait_for_gpus "${NUM_GPUS}"; then
        error "Could not acquire GPUs before deadline."
        exit 1
    fi
    if ! reserve_gpus "${NUM_GPUS}" "${RESERVE_DURATION}"; then
        error "GPU reservation failed."
        exit 1
    fi
}

acquire_gpus

# ---------------------------------------------------------------------------
# Main loop: per-subset server restart + eval
# ---------------------------------------------------------------------------
banner "Running acceptance eval [${N_SUBSETS} subsets]"

idx=0
for subset in "${SUBSET_ARRAY[@]}"; do
    idx=$(( idx + 1 ))
    output_file="${ACCEPTANCE_DIR}/after_${subset}.json"

    if [[ -f "${output_file}" ]]; then
        info "[${idx}/${N_SUBSETS}] ${subset} -- already done, skipping."
        continue
    fi

    sub_banner "Subset ${idx}/${N_SUBSETS}: ${subset}"

    # Check / renew reservation
    if ! renew_reservation "${NUM_GPUS}" "${RESERVE_DURATION}"; then
        if past_deadline; then
            error "Deadline reached during reservation renewal."
            exit 1
        fi
        warn "Reservation lost -- re-acquiring GPUs..."
        acquire_gpus
    fi

    # Retry loop for this subset (deadline-bounded)
    attempt=0
    success=false
    while ! past_deadline; do
        attempt=$(( attempt + 1 ))
        ATTEMPT_START=$(date +%s)
        info "Attempt ${attempt} for ${subset} (elapsed: $(elapsed))"

        # Find a free port
        PORT=$(find_free_port 8000)
        ENDPOINT="http://localhost:${PORT}"

        # Start server
        info "Starting server on port ${PORT}..."
        SERVER_LOG="${OUTPUT_DIR}/server_${subset}_attempt${attempt}.log"
        GPU_IDS="${GPU_IDS}" PORT="${PORT}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
            SERVER_LOG="${SERVER_LOG}" \
            bash "${SERVE_SCRIPT}" > /dev/null 2>&1 &
        SERVER_PID=$!
        info "Server PID: ${SERVER_PID}"

        # Wait for server health
        if ! wait_for_server "${ENDPOINT}" 1800; then
            warn "Server failed to start -- retrying..."
            stop_server "${PORT}"
            continue
        fi

        # Run acceptance eval
        info "Running acceptance eval for ${subset}..."
        EVAL_ARGS=(
            --endpoint "${ENDPOINT}"
            --dataset "${DATASET}"
            --subset "${subset}"
            --output "${output_file}"
            --max-requests "${MAX_REQUESTS}"
            --max-concurrency "${MAX_CONCURRENCY}"
        )
        [[ -n "${GEN_KWARGS}" ]] && EVAL_ARGS+=(--gen-kwargs "${GEN_KWARGS}")

        if bash "${SCRIPT_DIR}/eval_acceptance.sh" "${EVAL_ARGS[@]}"; then
            info "Subset ${idx}/${N_SUBSETS} (${subset}) complete (attempt ${attempt}, took $(elapsed_since "${ATTEMPT_START}"))"
            success=true
            stop_server "${PORT}"
            break
        else
            warn "Eval failed for ${subset} (attempt ${attempt}) -- retrying..."
            stop_server "${PORT}"
        fi
    done

    if [[ "${success}" != "true" ]]; then
        error "Deadline reached -- eval never completed for: ${subset}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
banner "Aggregating acceptance rates"

PARENT_SCRIPTS="${SCRIPT_DIR}/../scripts"

python "${PARENT_SCRIPTS}/compute_acceptance_rates.py" \
    --acceptance-dir "${ACCEPTANCE_DIR}" \
    --subsets "${SUBSETS}" \
    --output "${ACCEPTANCE_RATES_FILE}"

info "Acceptance rates: ${ACCEPTANCE_RATES_FILE}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "Complete! (total: $(elapsed))"
info "Output dir      : ${OUTPUT_DIR}"
info "  snapshots     : ${ACCEPTANCE_DIR}/"
info "  rates         : ${ACCEPTANCE_RATES_FILE}"
