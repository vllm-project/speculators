#!/usr/bin/env bash
# Start vLLM server for speculator model evaluation

set -euo pipefail

# ==============================================================================
# Configuration Variables
# ==============================================================================

BASE_MODEL=""
SPECULATOR_MODEL=""
NUM_SPEC_TOKENS=""
METHOD=""
TENSOR_PARALLEL_SIZE=""
MAX_MODEL_LEN=""
GPU_MEMORY_UTILIZATION=""
PORT=""
HEALTH_CHECK_TIMEOUT=""
SERVER_LOG=""
PID_FILE=""

readonly SLEEP_INTERVAL=5

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 -b BASE_MODEL -s SPECULATOR_MODEL [OPTIONS]

Required:
  -b BASE_MODEL              Base model path or HuggingFace ID
  -s SPECULATOR_MODEL        Speculator model path or HuggingFace ID

Optional:
  --num-spec-tokens TOKENS       Number of speculative tokens (default: 3)
  --method METHOD                Speculative decoding method (default: eagle3)
  --tensor-parallel-size SIZE    Number of GPUs (default: 1)
  --max-model-len LENGTH      Max model length (default: 24000)
  --gpu-memory-utilization UTIL  GPU memory fraction (default: 0.85)
  --port PORT                    Server port (default: 8000)
  --health-check-timeout SECS    Health check timeout (default: 300)
  --log-file FILE               Log file path (default: vllm_server.log)
  --pid-file FILE               PID file path (default: vllm_server.pid)
  -h, --help                    Show this help message

Example:
  $0 -b "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \\
     -s "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3" \\
     --num-spec-tokens 3 \\
     --method eagle3
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -b)
            BASE_MODEL="$2"
            shift 2
            ;;
        -s)
            SPECULATOR_MODEL="$2"
            shift 2
            ;;
        --num-spec-tokens)
            NUM_SPEC_TOKENS="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
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
        --health-check-timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --log-file)
            SERVER_LOG="$2"
            shift 2
            ;;
        --pid-file)
            PID_FILE="$2"
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

# Apply defaults for any arguments not provided
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
METHOD="${METHOD:-eagle3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PORT="${PORT:-8000}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
SERVER_LOG="${SERVER_LOG:-vllm_server.log}"
PID_FILE="${PID_FILE:-vllm_server.pid}"

# ==============================================================================
# Validate Arguments
# ==============================================================================

if [[ -z "${BASE_MODEL}" ]]; then
    echo "[ERROR] Missing required argument: -b BASE_MODEL" >&2
    show_usage
    exit 1
fi

if [[ -z "${SPECULATOR_MODEL}" ]]; then
    echo "[ERROR] Missing required argument: -s SPECULATOR_MODEL" >&2
    show_usage
    exit 1
fi

# ==============================================================================
# Start Server
# ==============================================================================

echo "[INFO] Starting vLLM server with speculative decoding"
echo "[INFO]   Base model: ${BASE_MODEL}"
echo "[INFO]   Speculator model: ${SPECULATOR_MODEL}"
echo "[INFO]   Num speculative tokens: ${NUM_SPEC_TOKENS}"
echo "[INFO]   Method: ${METHOD}"
echo "[INFO]   Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "[INFO]   Max model length: ${MAX_MODEL_LEN}"
echo "[INFO]   GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "[INFO]   Port: ${PORT}"
echo "[INFO]   Log file: ${SERVER_LOG}"

vllm serve "${BASE_MODEL}" \
    --seed 42 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --port "${PORT}" \
    --speculative-config "{\"model\": \"${SPECULATOR_MODEL}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}, \"method\": \"${METHOD}\", \"max_model_len\": ${MAX_MODEL_LEN}}" \
    > "${SERVER_LOG}" 2>&1 &

VLLM_PID=$!
echo "${VLLM_PID}" > "${PID_FILE}"
echo "[INFO] vLLM server started (PID: ${VLLM_PID})"

# ==============================================================================
# Wait for Server to be Ready
# ==============================================================================

echo "[INFO] Waiting for server to be ready (timeout: ${HEALTH_CHECK_TIMEOUT}s)..."

elapsed=0

while [[ ${elapsed} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] Server ready!"
        exit 0
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM server died during startup" >&2
        echo "[ERROR] Check logs: ${SERVER_LOG}" >&2
        tail -n 50 "${SERVER_LOG}" >&2
        rm -f "${PID_FILE}"
        exit 1
    fi

    sleep "${SLEEP_INTERVAL}"
    elapsed=$((elapsed + SLEEP_INTERVAL))
done

echo "[ERROR] Server failed to start within ${HEALTH_CHECK_TIMEOUT}s" >&2
kill -TERM "${VLLM_PID}" 2>/dev/null || true
rm -f "${PID_FILE}"
exit 1
