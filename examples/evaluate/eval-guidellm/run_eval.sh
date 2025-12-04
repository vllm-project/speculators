#!/usr/bin/env bash
# Evaluate speculator models with guidellm and extract acceptance lengths
#
# Usage: ./run_eval.sh -m MODEL -d DATASET [OPTIONS]
# Example: ./run_eval.sh -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" -d emulated

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ==============================================================================
# Configuration
# ==============================================================================

# Required arguments (set via command line)
MODEL=""
DATASET=""

# Optional arguments with defaults
readonly DEFAULT_TENSOR_PARALLEL_SIZE=2
readonly DEFAULT_GPU_MEMORY_UTILIZATION=0.8
readonly DEFAULT_PORT=8000
readonly DEFAULT_MAX_SECONDS=600
readonly DEFAULT_HEALTH_CHECK_TIMEOUT=300  # 5 minutes

TENSOR_PARALLEL_SIZE="${DEFAULT_TENSOR_PARALLEL_SIZE}"
GPU_MEMORY_UTILIZATION="${DEFAULT_GPU_MEMORY_UTILIZATION}"
PORT="${DEFAULT_PORT}"
MAX_SECONDS="${DEFAULT_MAX_SECONDS}"
OUTPUT_DIR="eval_results_$(date +%Y%m%d_%H%M%S)"

# Process ID for cleanup
VLLM_PID=""

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

show_usage() {
    cat << EOF
Usage: $0 -m MODEL -d DATASET [OPTIONS]

Required:
  -m MODEL          Speculator model path or HuggingFace ID
  -d DATASET        Dataset for guidellm benchmarking

Optional:
  --tensor-parallel-size SIZE    Number of GPUs (default: ${DEFAULT_TENSOR_PARALLEL_SIZE})
  --gpu-memory-utilization UTIL  GPU memory fraction (default: ${DEFAULT_GPU_MEMORY_UTILIZATION})
  --port PORT                    Server port (default: ${DEFAULT_PORT})
  --max-seconds SECONDS          Benchmark duration (default: ${DEFAULT_MAX_SECONDS})
  -o OUTPUT_DIR                  Output directory (default: eval_results_TIMESTAMP)
  -h, --help                     Show this help message

Example:
  $0 -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" -d emulated
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
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_error "Install with: pip install vllm guidellm huggingface-hub"
        return 1
    fi

    return 0
}

cleanup() {
    local exit_code=$?

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log_info "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill -TERM "${VLLM_PID}" 2>/dev/null || true

        # Wait for graceful shutdown (max 5 seconds)
        for i in {1..5}; do
            if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            log_info "Force killing vLLM server..."
            kill -KILL "${VLLM_PID}" 2>/dev/null || true
        fi
    fi

    exit "${exit_code}"
}

wait_for_server() {
    local port=$1
    local timeout=$2
    local elapsed=0
    local sleep_interval=5

    log_info "Waiting for server to be ready (timeout: ${timeout}s)..."

    while [[ ${elapsed} -lt ${timeout} ]]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            log_info "Server ready!"
            return 0
        fi

        # Check if server process is still running
        if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log_error "vLLM server died during startup"
            log_error "Check logs: ${SERVER_LOG}"
            tail -n 50 "${SERVER_LOG}" >&2
            return 1
        fi

        sleep ${sleep_interval}
        elapsed=$((elapsed + sleep_interval))
    done

    log_error "Server failed to start within ${timeout}s"
    return 1
}

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -m)
            MODEL="$2"
            shift 2
            ;;
        -d)
            DATASET="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max-seconds)
            MAX_SECONDS="$2"
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
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ==============================================================================
# Validate Arguments and Environment
# ==============================================================================

# Check required arguments
if [[ -z "${MODEL}" ]] || [[ -z "${DATASET}" ]]; then
    log_error "Missing required arguments"
    show_usage
    exit 1
fi

# Check dependencies
if ! check_dependencies; then
    exit 1
fi

# Setup cleanup handler
trap cleanup EXIT INT TERM

# ==============================================================================
# Handle Dataset Download
# ==============================================================================

# TODO: Once guidellm adds support for dataset:config syntax (e.g., "dataset:config"),
# we can remove this download logic and pass HuggingFace datasets directly to guidellm

# If dataset looks like a HuggingFace dataset (contains "/" and doesn't exist as a file),
# download it and use the first data file
if [[ "${DATASET}" == */* ]] && [[ ! -f "${DATASET}" ]]; then
    log_info "Detected HuggingFace dataset: ${DATASET}"

    # Download the dataset using hf download and capture the download path
    dataset_dir=$(hf download "${DATASET}" --repo-type dataset 2>&1 | tail -1)

    if [[ $? -ne 0 ]] || [[ -z "${dataset_dir}" ]]; then
        log_error "Failed to download dataset: ${DATASET}"
        exit 1
    fi

    log_info "Dataset downloaded to: ${dataset_dir}"

    # Find the first .jsonl or .json file (follow symlinks with -L)
    data_file=$(find -L "${dataset_dir}" -type f \( -name "*.jsonl" -o -name "*.json" \) | head -1)

    if [[ -z "${data_file}" ]]; then
        log_error "No data files found in dataset"
        exit 1
    fi

    log_info "Using dataset file: ${data_file}"
    DATASET="${data_file}"
fi

# ==============================================================================
# Setup Output Directory
# ==============================================================================

if ! mkdir -p "${OUTPUT_DIR}"; then
    log_error "Failed to create output directory: ${OUTPUT_DIR}"
    exit 1
fi

# Define output file paths
readonly SERVER_LOG="${OUTPUT_DIR}/vllm_server.log"
readonly GUIDELLM_LOG="${OUTPUT_DIR}/guidellm_output.log"
readonly GUIDELLM_RESULTS="${OUTPUT_DIR}/guidellm_results.json"
readonly ACCEPTANCE_RESULTS="${OUTPUT_DIR}/acceptance_analysis.txt"

log_info "Output directory: ${OUTPUT_DIR}"

# ==============================================================================
# Start vLLM Server
# ==============================================================================

log_info "Starting vLLM server: ${MODEL}"
log_info "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
log_info "  GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
log_info "  Port: ${PORT}"

vllm serve "${MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --port "${PORT}" \
    > "${SERVER_LOG}" 2>&1 &

VLLM_PID=$!
log_info "vLLM server started (PID: ${VLLM_PID})"

# ==============================================================================
# Wait for Server to be Ready
# ==============================================================================

if ! wait_for_server "${PORT}" "${DEFAULT_HEALTH_CHECK_TIMEOUT}"; then
    exit 1
fi

# ==============================================================================
# Run GuideLLM Benchmark
# ==============================================================================

log_info "Running guidellm benchmark..."
log_info "  Dataset: ${DATASET}"
log_info "  Max seconds: ${MAX_SECONDS}"

GUIDELLM__PREFERRED_ROUTE="chat_completions" \
guidellm benchmark \
  --target "http://localhost:${PORT}/v1" \
  --data "${DATASET}" \
  --rate-type throughput \
  --max-seconds "${MAX_SECONDS}" \
  --output-path "${GUIDELLM_RESULTS}" \
  | tee "${GUIDELLM_LOG}"

log_info "Benchmark complete"

# ==============================================================================
# Parse Acceptance Lengths from Server Logs
# ==============================================================================

log_info "Parsing acceptance lengths..."

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PARSER_SCRIPT="${SCRIPT_DIR}/parse_acceptance_lengths.py"

if [[ ! -f "${PARSER_SCRIPT}" ]]; then
    log_error "Parser script not found: ${PARSER_SCRIPT}"
    exit 1
fi

if ! python "${PARSER_SCRIPT}" "${SERVER_LOG}" -o "${ACCEPTANCE_RESULTS}"; then
    log_error "Failed to parse acceptance lengths"
    exit 1
fi

# ==============================================================================
# Summary
# ==============================================================================

log_info ""
log_info "Evaluation complete!"
log_info "Results saved to: ${OUTPUT_DIR}"
log_info "  - Server log:        ${SERVER_LOG}"
log_info "  - GuideLLM output:   ${GUIDELLM_LOG}"
log_info "  - GuideLLM results:  ${GUIDELLM_RESULTS}"
log_info "  - Acceptance stats:  ${ACCEPTANCE_RESULTS}"
