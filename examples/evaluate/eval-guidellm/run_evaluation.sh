#!/usr/bin/env bash
# Main controller script for evaluating speculator models with guidellm

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""

# Default values (can be overridden by config file or command line)
MODEL=""
DATASET=""
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.8
PORT=8000
HEALTH_CHECK_TIMEOUT=300
OUTPUT_DIR="eval_results_$(date +%Y%m%d_%H%M%S)"

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required (use one):
  -c, --config FILE         Config file (e.g., configs/llama-eagle3.env)
  -m MODEL -d DATASET       Model and dataset via command line

Optional:
  -o OUTPUT_DIR             Output directory (default: eval_results_TIMESTAMP)
  -h, --help                Show this help message

Examples:
  $0 -c configs/llama-eagle3.env                              # Use config file
  $0 -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \\
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
        -m)
            MODEL="$2"
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
# Validate Configuration
# ==============================================================================

if [[ -z "${MODEL}" ]]; then
    echo "[ERROR] MODEL is required (set in config file or via -m)" >&2
    show_usage
    exit 1
fi

if [[ -z "${DATASET}" ]]; then
    echo "[ERROR] DATASET is required (set in config file or via -d)" >&2
    show_usage
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

"${SCRIPT_DIR}/scripts/vllm_serve.sh" \
    -m "${MODEL}" \
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

"${SCRIPT_DIR}/scripts/run_guidellm.sh" \
    -d "${DATASET}" \
    --target "http://localhost:${PORT}/v1" \
    --output-file "${GUIDELLM_RESULTS}" \
    --log-file "${GUIDELLM_LOG}"

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
echo "[INFO]   - GuideLLM results:  ${GUIDELLM_RESULTS}"
echo "[INFO]   - Acceptance stats:  ${ACCEPTANCE_RESULTS}"
