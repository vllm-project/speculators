#!/usr/bin/env bash
# Stop vLLM server gracefully

set -euo pipefail

# ==============================================================================
# Default Configuration
# ==============================================================================

PID_FILE="vllm_server.pid"

readonly GRACEFUL_SHUTDOWN_TIMEOUT=5

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Optional:
  --pid-file FILE    PID file path (default: vllm_server.pid)
  -h, --help         Show this help message

Example:
  $0
EOF
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
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
# Stop Server
# ==============================================================================

if [[ ! -f "${PID_FILE}" ]]; then
    echo "[INFO] No PID file found at ${PID_FILE}, server may not be running"
    exit 0
fi

VLLM_PID=$(cat "${PID_FILE}")

if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[INFO] Server (PID: ${VLLM_PID}) is not running"
    rm -f "${PID_FILE}"
    exit 0
fi

echo "[INFO] Stopping vLLM server (PID: ${VLLM_PID})..."
kill -TERM "${VLLM_PID}" 2>/dev/null || true

# Wait for graceful shutdown
for ((i=1; i<=GRACEFUL_SHUTDOWN_TIMEOUT; i++)); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[INFO] Server stopped successfully"
        rm -f "${PID_FILE}"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[INFO] Force killing server..."
    kill -KILL "${VLLM_PID}" 2>/dev/null || true
    sleep 1
fi

rm -f "${PID_FILE}"
echo "[INFO] Server stopped"
