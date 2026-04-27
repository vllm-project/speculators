#!/usr/bin/env bash
# Wait for the vLLM server to be ready, then run the perf benchmark.
# Usage: bash wait_and_run_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="http://localhost:8000/v1"

echo "[INFO] Waiting for vLLM server at ${TARGET%/v1} ..."
until curl -sf "${TARGET%/v1}/health" > /dev/null 2>&1; do
    echo "[INFO]   Still waiting..."
    sleep 150
done
echo "[INFO] Server is up!"

# max-requests 50: keeps the synchronous sweep phase to ~3 min/subset
# rather than the default 200 (~16 min/subset) which caused the apparent hang.
bash "${SCRIPT_DIR}/run_perf_benchmark.sh" \
    --target "${TARGET}" \
    --max-requests 50 \
    --capture-acceptance-rate \
    2>&1 | tee "${SCRIPT_DIR}/client-log.log"
