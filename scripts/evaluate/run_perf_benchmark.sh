#!/usr/bin/env bash
# Performance benchmarking pipeline with noise-reduced output length capping.
#
# Workflow:
#   1. Run guidellm in throughput mode to estimate output token distributions
#   2. Derive per-subset max_tokens (first power-of-2 >= median output length)
#   3. Run guidellm sweep with those max_tokens caps
#   4. Parse sweep results into a consolidated CSV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Configuration
# ==============================================================================

TARGET=""
DATASET=""
SUBSETS=""
OUTPUT_DIR=""
MAX_CONCURRENCY=""
MAX_REQUESTS=""
GEN_KWARGS=""
GEN_LEN_RATE=""
DATA_COLUMN_MAPPER=""

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 --target URL [OPTIONS]

Required:
  --target URL               vLLM server endpoint (e.g. http://localhost:8000/v1)

Optional:
  --dataset DATASET          HF dataset ID or local dir (default: RedHatAI/speculator_benchmarks)
  --subsets LIST             Comma-separated subset names
                             (default: HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing)
  --output-dir DIR           Output directory (default: perf_results_TIMESTAMP)
  --max-concurrency N        Max concurrent requests for guidellm (default: 128)
  --max-requests N           Max requests per sweep point (default: 200)
  --gen-len-rate N           Request rate for gen-len estimation (default: 128)
  --gen-kwargs JSON          Flat JSON with generation kwargs, e.g.
                             '{"temperature":0.6, "top_p":0.95, "top_k":20}'
  --data-column-mapper JSON  Column mapping for guidellm
                             (default: '{"text_column":"prompt"}')
  -h, --help                 Show this help message

Examples:
  $0 --target http://localhost:8000/v1
  $0 --target http://localhost:8000/v1 --subsets "HumanEval,qa" \\
     --gen-kwargs '{"temperature":0.6, "top_p":0.95}'
EOF
}

check_dependencies() {
    local missing_deps=()

    for cmd in guidellm python curl; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    # Check for required Python modules
    if ! python -c "import requests" &> /dev/null; then
        missing_deps+=("python-requests")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "[ERROR] Missing required dependencies: ${missing_deps[*]}" >&2
        echo "[ERROR] Install with: pip install -r requirements.txt" >&2
        return 1
    fi

    return 0
}

build_backend_args() {
    local gen_kwargs="${1:-}"
    local max_tokens="$2"

    python -c "
import json, sys
gen_kwargs = sys.argv[1]
max_tokens = int(sys.argv[2])
body = json.loads(gen_kwargs) if gen_kwargs else {}
body['max_tokens'] = max_tokens
print(json.dumps({'extras': {'body': body}}))
" "${gen_kwargs}" "${max_tokens}"
}

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --subsets)
            SUBSETS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS="$2"
            shift 2
            ;;
        --gen-len-rate)
            GEN_LEN_RATE="$2"
            shift 2
            ;;
        --gen-kwargs)
            GEN_KWARGS="$2"
            shift 2
            ;;
        --data-column-mapper)
            DATA_COLUMN_MAPPER="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

# ==============================================================================
# Apply Defaults
# ==============================================================================

DATASET="${DATASET:-RedHatAI/speculator_benchmarks}"
SUBSETS="${SUBSETS:-HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing}"
OUTPUT_DIR="${OUTPUT_DIR:-perf_results_$(date +%Y%m%d_%H%M%S)}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
MAX_REQUESTS="${MAX_REQUESTS:-200}"
GEN_LEN_RATE="${GEN_LEN_RATE:-128}"
DATA_COLUMN_MAPPER="${DATA_COLUMN_MAPPER:-{\"text_column\":\"prompt\"}}"

# Derive vLLM URL from target if not provided (strip /v1 suffix)
if [[ -z "${VLLM_URL:-}" ]]; then
    VLLM_URL="${TARGET%/}"
    VLLM_URL="${VLLM_URL%/v1}"
fi

# ==============================================================================
# Validate
# ==============================================================================

if [[ -z "${TARGET}" ]]; then
    echo "[ERROR] --target is required" >&2
    show_usage
    exit 1
fi

if ! check_dependencies; then
    exit 1
fi

# ==============================================================================
# Setup Output Directory
# ==============================================================================

GEN_LEN_DIR="${OUTPUT_DIR}/gen_len"
SWEEP_DIR="${OUTPUT_DIR}/sweeps"

mkdir -p "${GEN_LEN_DIR}" "${SWEEP_DIR}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"

# Split subsets into array
IFS=',' read -ra SUBSET_ARRAY <<< "${SUBSETS}"

# ==============================================================================
# Main Pipeline: Process Each Subset Sequentially
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Processing ${#SUBSET_ARRAY[@]} subsets sequentially"
echo "[INFO] ============================================"

PARTIAL_CSV_FILES=()
ALL_MAX_TOKENS=()

for subset in "${SUBSET_ARRAY[@]}"; do
    echo ""
    echo "[INFO] ============================================"
    echo "[INFO] Processing subset: ${subset}"
    echo "[INFO] ============================================"

    # Step 1: Gen-len estimation for this subset
    echo "[INFO] [${subset}] Step 1/3: Estimating output token distribution"

    GEN_LEN_OUTPUT="${GEN_LEN_DIR}/gen_len_${subset}.json"
    BACKEND_ARGS=$(build_backend_args "${GEN_KWARGS}" "4096")

    GUIDELLM__MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
    guidellm benchmark \
        --target "${TARGET}" \
        --data "${DATASET}" \
        --data-args "{\"data_files\": \"${subset}.jsonl\"}" \
        --data-column-mapper "${DATA_COLUMN_MAPPER}" \
        --profile throughput \
        --rate "${GEN_LEN_RATE}" \
        --output-path "${GEN_LEN_OUTPUT}" \
        --backend-args "${BACKEND_ARGS}"

    echo "[INFO] [${subset}] Gen-len estimation complete"

    # Step 2: Parse gen-len to get max_tokens for this subset
    echo "[INFO] [${subset}] Step 2/3: Computing max_tokens"

    SUBSET_MAX_TOKENS_FILE="${GEN_LEN_DIR}/max_tokens_${subset}.json"
    python "${SCRIPT_DIR}/parse_gen_len.py" \
        --output "${SUBSET_MAX_TOKENS_FILE}" \
        "${GEN_LEN_OUTPUT}"

    MAX_TOKENS=$(python -c "import json; print(json.load(open('${SUBSET_MAX_TOKENS_FILE}'))['${subset}'])")
    ALL_MAX_TOKENS+=("\"${subset}\": ${MAX_TOKENS}")
    echo "[INFO] [${subset}] max_tokens=${MAX_TOKENS}"

    # Step 3: Run sweep immediately (server state is fresh from gen-len)
    echo "[INFO] [${subset}] Step 3/3: Running performance sweep"

    # Capture baseline vLLM metrics BEFORE the sweep (if vLLM URL provided)
    BASELINE_METRICS_FILE=""
    if [[ -n "${VLLM_URL}" ]]; then
        BASELINE_METRICS_FILE="${OUTPUT_DIR}/${subset}_baseline_metrics.txt"
        METRICS_URL="${VLLM_URL%/}/metrics"
        if curl -s -f --max-time 10 "${METRICS_URL}" > "${BASELINE_METRICS_FILE}"; then
            echo "[INFO] [${subset}] Captured baseline metrics"
        else
            echo "[WARN] Failed to fetch baseline metrics, proceeding without delta" >&2
            BASELINE_METRICS_FILE=""
        fi
    fi

    SWEEP_OUTPUT="${SWEEP_DIR}/sweep_${subset}.json"
    BACKEND_ARGS=$(build_backend_args "${GEN_KWARGS}" "${MAX_TOKENS}")

    GUIDELLM__MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
    guidellm benchmark \
        --target "${TARGET}" \
        --data "${DATASET}" \
        --data-args "{\"data_files\": \"${subset}.jsonl\"}" \
        --data-column-mapper "${DATA_COLUMN_MAPPER}" \
        --profile sweep \
        --max-requests "${MAX_REQUESTS}" \
        --output-path "${SWEEP_OUTPUT}" \
        --backend-args "${BACKEND_ARGS}"

    echo "[INFO] [${subset}] Sweep complete"

    # Capture current vLLM metrics AFTER sweep (if vLLM URL provided)
    CURRENT_METRICS_FILE=""
    if [[ -n "${VLLM_URL}" ]]; then
        CURRENT_METRICS_FILE="${OUTPUT_DIR}/${subset}_current_metrics.txt"
        METRICS_URL="${VLLM_URL%/}/metrics"
        if curl -s -f --max-time 10 "${METRICS_URL}" > "${CURRENT_METRICS_FILE}"; then
            echo "[INFO] [${subset}] Captured current metrics"
        else
            echo "[WARN] Failed to fetch current metrics, proceeding without spec decode data" >&2
            CURRENT_METRICS_FILE=""
        fi
    fi

    # Parse sweep results to capture metrics (delta from baseline)
    PARTIAL_CSV="${OUTPUT_DIR}/${subset}_partial.csv"

    PARSE_ARGS=(
        --output "${PARTIAL_CSV}"
    )
    [[ -n "${CURRENT_METRICS_FILE}" && -f "${CURRENT_METRICS_FILE}" ]] && PARSE_ARGS+=(--current-metrics "${CURRENT_METRICS_FILE}")
    [[ -n "${BASELINE_METRICS_FILE}" && -f "${BASELINE_METRICS_FILE}" ]] && PARSE_ARGS+=(--baseline-metrics "${BASELINE_METRICS_FILE}")

    python "${SCRIPT_DIR}/parse_sweep.py" \
        "${PARSE_ARGS[@]}" \
        "${SWEEP_OUTPUT}"

    # Only add to the array if the file was created successfully
    if [[ -f "${PARTIAL_CSV}" ]]; then
        PARTIAL_CSV_FILES+=("${PARTIAL_CSV}")
        echo "[INFO] [${subset}] ✓ Complete"
    else
        echo "[WARN] [${subset}] Parsing failed, skipping" >&2
    fi
done

# ==============================================================================
# Save Combined max_tokens.json
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Saving consolidated max_tokens mapping"
echo "[INFO] ============================================"

MAX_TOKENS_FILE="${OUTPUT_DIR}/max_tokens.json"
printf "{\n  %s\n}\n" "$(IFS=','; echo "${ALL_MAX_TOKENS[*]}" | sed 's/,/,\n  /g')" > "${MAX_TOKENS_FILE}"
echo "[INFO] max_tokens mapping saved to: ${MAX_TOKENS_FILE}"

# ==============================================================================
# Step 4: Combine Per-Subset CSVs
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Step 4: Combining subset results into final CSV"
echo "[INFO] ============================================"

CSV_FILE="${OUTPUT_DIR}/perf_results.csv"

# Combine all partial CSVs (header from first, data from all)
if [[ ${#PARTIAL_CSV_FILES[@]} -gt 0 ]]; then
    head -n 1 "${PARTIAL_CSV_FILES[0]}" > "${CSV_FILE}"
    for partial_csv in "${PARTIAL_CSV_FILES[@]}"; do
        tail -n +2 "${partial_csv}" >> "${CSV_FILE}"
    done
    echo "[INFO] Combined ${#PARTIAL_CSV_FILES[@]} subset results"
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Performance benchmarking complete!"
echo "[INFO] ============================================"
echo "[INFO] Results saved to: ${OUTPUT_DIR}"
echo "[INFO]   - Gen-len outputs:  ${GEN_LEN_DIR}/"
echo "[INFO]   - Max tokens map:   ${MAX_TOKENS_FILE}"
echo "[INFO]   - Sweep outputs:    ${SWEEP_DIR}/"
echo "[INFO]   - CSV results:      ${CSV_FILE}"
