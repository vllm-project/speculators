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

    for cmd in guidellm python; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "[ERROR] Missing required dependencies: ${missing_deps[*]}" >&2
        echo "[ERROR] Install with: pip install guidellm" >&2
        return 1
    fi

    return 0
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
# Step 1: Estimate Output Token Distributions
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Step 1: Estimating output token distributions"
echo "[INFO] ============================================"

for subset in "${SUBSET_ARRAY[@]}"; do
    echo "[INFO] Running gen-len estimation for subset: ${subset}"

    GEN_LEN_ARGS=(
        --target "${TARGET}"
        --dataset "${DATASET}"
        --subset "${subset}"
        --output-file "${GEN_LEN_DIR}/gen_len_${subset}.json"
        --rate "${GEN_LEN_RATE}"
        --max-concurrency "${MAX_CONCURRENCY}"
    )
    [[ -n "${GEN_KWARGS}" ]] && GEN_LEN_ARGS+=(--gen-kwargs "${GEN_KWARGS}")
    [[ -n "${DATA_COLUMN_MAPPER}" ]] && GEN_LEN_ARGS+=(--data-column-mapper "${DATA_COLUMN_MAPPER}")

    bash "${SCRIPT_DIR}/scripts/run_gen_len_estimation.sh" "${GEN_LEN_ARGS[@]}"

    echo "[INFO] Gen-len estimation complete for: ${subset}"
done

# ==============================================================================
# Step 2: Parse Gen-Len Results -> max_tokens.json
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Step 2: Computing max_tokens per subset"
echo "[INFO] ============================================"

GEN_LEN_FILES=()
for subset in "${SUBSET_ARRAY[@]}"; do
    GEN_LEN_FILES+=("${GEN_LEN_DIR}/gen_len_${subset}.json")
done

MAX_TOKENS_FILE="${OUTPUT_DIR}/max_tokens.json"

python "${SCRIPT_DIR}/scripts/parse_gen_len.py" \
    --output "${MAX_TOKENS_FILE}" \
    "${GEN_LEN_FILES[@]}"

echo "[INFO] max_tokens mapping saved to: ${MAX_TOKENS_FILE}"

# ==============================================================================
# Step 3: Run Sweeps
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Step 3: Running performance sweeps"
echo "[INFO] ============================================"

for subset in "${SUBSET_ARRAY[@]}"; do
    MAX_TOKENS=$(python -c "import json; print(json.load(open('${MAX_TOKENS_FILE}'))['${subset}'])")

    echo "[INFO] Running sweep for subset: ${subset} (max_tokens=${MAX_TOKENS})"

    SWEEP_ARGS=(
        --target "${TARGET}"
        --dataset "${DATASET}"
        --subset "${subset}"
        --output-file "${SWEEP_DIR}/sweep_${subset}.json"
        --max-tokens "${MAX_TOKENS}"
        --max-requests "${MAX_REQUESTS}"
        --max-concurrency "${MAX_CONCURRENCY}"
    )
    [[ -n "${GEN_KWARGS}" ]] && SWEEP_ARGS+=(--gen-kwargs "${GEN_KWARGS}")
    [[ -n "${DATA_COLUMN_MAPPER}" ]] && SWEEP_ARGS+=(--data-column-mapper "${DATA_COLUMN_MAPPER}")

    bash "${SCRIPT_DIR}/scripts/run_sweep.sh" "${SWEEP_ARGS[@]}"

    echo "[INFO] Sweep complete for: ${subset}"
done

# ==============================================================================
# Step 4: Parse Sweep Results -> CSV
# ==============================================================================

echo ""
echo "[INFO] ============================================"
echo "[INFO] Step 4: Extracting sweep metrics to CSV"
echo "[INFO] ============================================"

SWEEP_FILES=()
for subset in "${SUBSET_ARRAY[@]}"; do
    SWEEP_FILES+=("${SWEEP_DIR}/sweep_${subset}.json")
done

CSV_FILE="${OUTPUT_DIR}/perf_results.csv"

python "${SCRIPT_DIR}/scripts/parse_sweep_results.py" \
    --output "${CSV_FILE}" \
    "${SWEEP_FILES[@]}"

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
