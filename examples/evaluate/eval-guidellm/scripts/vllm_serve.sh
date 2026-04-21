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
TOKENIZER_MODE=""
NO_CHUNKED_PREFILL=""

readonly SLEEP_INTERVAL=5

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 -b BASE_MODEL [OPTIONS]

Required:
  -b BASE_MODEL              Base model path or HuggingFace ID

Optional:
  -s SPECULATOR_MODEL            Speculator model path or HuggingFace ID
                                 (omit for built-in MTP heads)
  --num-spec-tokens TOKENS       Number of speculative tokens (default: 3)
  --method METHOD                Speculative decoding method (default: eagle3)
  --tensor-parallel-size SIZE    Number of GPUs (default: 1)
  --max-model-len LENGTH         Max model length (default: 24000)
  --gpu-memory-utilization UTIL  GPU memory fraction (default: 0.85)
  --port PORT                    Server port (default: 8000)
  --health-check-timeout SECS    Health check timeout (default: 300)
  --log-file FILE                Log file path (default: vllm_server.log)
  --pid-file FILE                PID file path (default: vllm_server.pid)
  --tokenizer-mode MODE          Tokenizer mode passed to vllm (e.g. auto)
  --no-enable-chunked-prefill    Pass --no-enable-chunked-prefill to vllm
  -h, --help                     Show this help message

Examples:
  # Eagle3 (separate speculator)
  $0 -b "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \\
     -s "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3" \\
     --num-spec-tokens 3 --method eagle3

  # MTP (built-in head)
  $0 -b "Qwen/Qwen3-Next-80B-A3B-Instruct" \\
     --num-spec-tokens 2 --method mtp \\
     --tokenizer-mode auto --no-enable-chunked-prefill
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
        --tokenizer-mode)
            TOKENIZER_MODE="$2"
            shift 2
            ;;
        --no-enable-chunked-prefill)
            NO_CHUNKED_PREFILL="true"
            shift
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

# SPECULATOR_MODEL is optional: omit for built-in MTP heads

# ==============================================================================
# Start Server
# ==============================================================================

echo "[INFO] Starting vLLM server with speculative decoding"
echo "[INFO]   Base model: ${BASE_MODEL}"
echo "[INFO]   Speculator model: ${SPECULATOR_MODEL:-(built-in MTP head)}"
echo "[INFO]   Num speculative tokens: ${NUM_SPEC_TOKENS}"
echo "[INFO]   Method: ${METHOD}"
echo "[INFO]   Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "[INFO]   Max model length: ${MAX_MODEL_LEN}"
echo "[INFO]   GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "[INFO]   Port: ${PORT}"
echo "[INFO]   Log file: ${SERVER_LOG}"
[[ -n "${TOKENIZER_MODE}" ]] && echo "[INFO]   Tokenizer mode: ${TOKENIZER_MODE}"
[[ "${NO_CHUNKED_PREFILL}" == "true" ]] && echo "[INFO]   Chunked prefill: disabled"

# Build speculative-config JSON:
#   With external speculator (Eagle): include model + max_model_len fields
#   Without speculator (MTP built-in): method + num_speculative_tokens only
if [[ -n "${SPECULATOR_MODEL}" ]]; then
    SPEC_CONFIG="{\"model\": \"${SPECULATOR_MODEL}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}, \"method\": \"${METHOD}\", \"max_model_len\": ${MAX_MODEL_LEN}}"
else
    SPEC_CONFIG="{\"method\": \"${METHOD}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}"
fi

# Build optional extra flags
EXTRA_FLAGS=()
[[ -n "${TOKENIZER_MODE}" ]] && EXTRA_FLAGS+=(--tokenizer-mode "${TOKENIZER_MODE}")
[[ "${NO_CHUNKED_PREFILL}" == "true" ]] && EXTRA_FLAGS+=(--no-enable-chunked-prefill)

# Fail fast if the port is already in use rather than burying the error in vLLM logs
if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || nc -z 127.0.0.1 "${PORT}" 2>/dev/null; then
    echo "[ERROR] Port ${PORT} is already in use. Stop the existing process or set PORT to a different value." >&2
    exit 1
fi

vllm serve "${BASE_MODEL}" \
    --seed 42 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --port "${PORT}" \
    --speculative-config "${SPEC_CONFIG}" \
    "${EXTRA_FLAGS[@]}" \
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
