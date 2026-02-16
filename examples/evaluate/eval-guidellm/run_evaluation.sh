#!/usr/bin/env bash
# Main controller script for evaluating speculator models with guidellm

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""

# Variables (precedence: CLI args > config file > defaults)
BASE_MODEL=""
SPECULATOR_MODEL=""
NUM_SPEC_TOKENS=""
METHOD=""
DATASET=""
TENSOR_PARALLEL_SIZE=""
GPU_MEMORY_UTILIZATION=""
PORT=""
HEALTH_CHECK_TIMEOUT=""
OUTPUT_DIR=""
TEMPERATURE=""
TOP_P=""
TOP_K=""
EVAL_MODE=""

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required (use one):
  -c, --config FILE                    Config file (e.g., configs/llama-eagle3.env)
  -b BASE_MODEL -s SPECULATOR_MODEL    Base model, speculator, and dataset via command line
     -d DATASET

Optional:
  -o OUTPUT_DIR                        Output directory (default: eval_results_TIMESTAMP)
  --eval-mode MODE                     Evaluation mode: text or vl (default: text)
  -h, --help                           Show this help message

Examples:
  $0 -c configs/llama-3.3-70b-eagle3.env                      # Use config file
  $0 -b "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \\
     -s "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3" \\
     -d "emulated"                                             # Use command line
  $0 -c configs/llama-eagle3.env -d "different-dataset"       # Override dataset
EOF
}

check_dependencies() {
    local missing_deps=()

    for cmd in vllm guidellm python curl hf; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "[ERROR] Missing required dependencies: ${missing_deps[*]}" >&2
        echo "[ERROR] Install with: pip install vllm guidellm huggingface-hub" >&2
        return 1
    fi

    return 0
}

cleanup() {
    local exit_code=$?

    echo "[INFO] Cleaning up..."
    "${SCRIPT_DIR}/scripts/vllm_stop.sh" --pid-file "${OUTPUT_DIR}/vllm_server.pid" 2>/dev/null || true

    exit "${exit_code}"
}

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -b)
            BASE_MODEL="$2"
            shift 2
            ;;
        -s)
            SPECULATOR_MODEL="$2"
            shift 2
            ;;
        -d)
            DATASET="$2"
            shift 2
            ;;
        -o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval-mode)
            EVAL_MODE="$2"
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
# Load Configuration
# ==============================================================================

if [[ -n "${CONFIG_FILE}" ]]; then
    if [[ -f "${CONFIG_FILE}" ]]; then
        echo "[INFO] Loading configuration from: ${CONFIG_FILE}"
        # Source config file, but only if variables are not already set
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue

            # Remove quotes from value
            value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

            # Only set if not already set via command line
            if [[ -z "${!key:-}" ]]; then
                eval "${key}=\"${value}\""
            fi
        done < "${CONFIG_FILE}"
    else
        echo "[ERROR] Config file not found: ${CONFIG_FILE}" >&2
        exit 1
    fi
fi

# ==============================================================================
# Apply Defaults
# ==============================================================================

# Apply defaults for any variables not set by CLI args or config file
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
METHOD="${METHOD:-eagle3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
PORT="${PORT:-8000}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results_$(date +%Y%m%d_%H%M%S)}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
EVAL_MODE="${EVAL_MODE:-text}"

# ==============================================================================
# Validate Configuration
# ==============================================================================

if [[ -z "${BASE_MODEL}" ]]; then
    echo "[ERROR] BASE_MODEL is required (set in config file or via -b)" >&2
    show_usage
    exit 1
fi

if [[ -z "${SPECULATOR_MODEL}" ]]; then
    echo "[ERROR] SPECULATOR_MODEL is required (set in config file or via -s)" >&2
    show_usage
    exit 1
fi

if [[ -z "${DATASET}" ]]; then
    echo "[ERROR] DATASET is required (set in config file or via -d)" >&2
    show_usage
    exit 1
fi

if [[ "${EVAL_MODE}" != "text" ]] && [[ "${EVAL_MODE}" != "vl" ]]; then
    echo "[ERROR] EVAL_MODE must be one of: text, vl. Got: ${EVAL_MODE}" >&2
    exit 1
fi

if ! check_dependencies; then
    exit 1
fi

# Setup cleanup handler
trap cleanup EXIT INT TERM

# ==============================================================================
# Setup Output Directory
# ==============================================================================

if ! mkdir -p "${OUTPUT_DIR}"; then
    echo "[ERROR] Failed to create output directory: ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "[INFO] Evaluation mode: ${EVAL_MODE}"

# Define output file paths
SERVER_LOG="${OUTPUT_DIR}/vllm_server.log"
SERVER_PID="${OUTPUT_DIR}/vllm_server.pid"
GUIDELLM_LOG="${OUTPUT_DIR}/guidellm_output.log"
GUIDELLM_RESULTS="${OUTPUT_DIR}/guidellm_results.json"
ACCEPTANCE_RESULTS="${OUTPUT_DIR}/acceptance_analysis.txt"

# ==============================================================================
# Start vLLM Server
# ==============================================================================

echo "[INFO] Starting vLLM server..."

IMAGE_ROOT="${IMAGE_ROOT:-}" "${SCRIPT_DIR}/scripts/vllm_serve.sh" \
    -b "${BASE_MODEL}" \
    -s "${SPECULATOR_MODEL}" \
    --num-spec-tokens "${NUM_SPEC_TOKENS}" \
    --method "${METHOD}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --port "${PORT}" \
    --health-check-timeout "${HEALTH_CHECK_TIMEOUT}" \
    --log-file "${SERVER_LOG}" \
    --pid-file "${SERVER_PID}"

# ==============================================================================
# Run GuideLLM Benchmark
# ==============================================================================

echo "[INFO] Running benchmark..."

if [[ "${EVAL_MODE}" == "vl" ]]; then
    IMAGE_ROOT="${IMAGE_ROOT:-}" bash "${SCRIPT_DIR}/scripts/run_guidellm_vl.sh" \
        -d "${DATASET}" \
        --target "http://localhost:${PORT}/v1" \
        --output-file "${GUIDELLM_RESULTS}" \
        --log-file "${GUIDELLM_LOG}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}"
else
    bash "${SCRIPT_DIR}/scripts/run_guidellm.sh" \
        -d "${DATASET}" \
        --target "http://localhost:${PORT}/v1" \
        --output-file "${GUIDELLM_RESULTS}" \
        --log-file "${GUIDELLM_LOG}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}"
fi

# ==============================================================================
# Parse Acceptance Lengths
# ==============================================================================

echo "[INFO] Parsing acceptance lengths..."

PARSER_SCRIPT="${SCRIPT_DIR}/scripts/parse_logs.py"

if [[ ! -f "${PARSER_SCRIPT}" ]]; then
    echo "[ERROR] Parser script not found: ${PARSER_SCRIPT}" >&2
    exit 1
fi

if ! python "${PARSER_SCRIPT}" "${SERVER_LOG}" -o "${ACCEPTANCE_RESULTS}"; then
    echo "[ERROR] Failed to parse acceptance lengths" >&2
    exit 1
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "[INFO] Evaluation complete!"
echo "[INFO] Results saved to: ${OUTPUT_DIR}"
echo "[INFO]   - Server log:        ${SERVER_LOG}"
echo "[INFO]   - GuideLLM output:   ${GUIDELLM_LOG}"
if [[ -f "${GUIDELLM_RESULTS}" ]]; then
    echo "[INFO]   - GuideLLM results:  ${GUIDELLM_RESULTS}"
fi
if ls "${OUTPUT_DIR}"/vl_eval_summary*.json >/dev/null 2>&1; then
    for one in "${OUTPUT_DIR}"/vl_eval_summary*.json; do
        echo "[INFO]   - VL summary:        ${one}"
    done
fi
if ls "${OUTPUT_DIR}"/vl_eval_results*.jsonl >/dev/null 2>&1; then
    for one in "${OUTPUT_DIR}"/vl_eval_results*.jsonl; do
        echo "[INFO]   - VL records:        ${one}"
    done
fi
echo "[INFO]   - Acceptance stats:  ${ACCEPTANCE_RESULTS}"
