#!/usr/bin/env bash
# Start vLLM server for speculator model evaluation

set -euo pipefail

# ==============================================================================
# Default Configuration
# ==============================================================================

MODEL=""
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.8
PORT=8000
HEALTH_CHECK_TIMEOUT=300
SERVER_LOG="vllm_server.log"
PID_FILE="vllm_server.pid"

readonly SLEEP_INTERVAL=5

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 -m MODEL [OPTIONS]

Required:
  -m MODEL          Speculator model path or HuggingFace ID

Optional:
  --tensor-parallel-size SIZE    Number of GPUs (default: 2)
  --gpu-memory-utilization UTIL  GPU memory fraction (default: 0.8)
  --port PORT                    Server port (default: 8000)
  --health-check-timeout SECS    Health check timeout (default: 300)
  --log-file FILE               Log file path (default: vllm_server.log)
  --pid-file FILE               PID file path (default: vllm_server.pid)
  -h, --help                    Show this help message

Example:
  $0 -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m)
            MODEL="$2"
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
# Validate Arguments
# ==============================================================================

if [[ -z "${MODEL}" ]]; then
    echo "[ERROR] Missing required argument: -m MODEL" >&2
    show_usage
    exit 1
fi

# ==============================================================================
# Start Server
# ==============================================================================

echo "[INFO] Starting vLLM server: ${MODEL}"
echo "[INFO]   Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "[INFO]   GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "[INFO]   Port: ${PORT}"
echo "[INFO]   Log file: ${SERVER_LOG}"

vllm serve "${MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --port "${PORT}" \
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
