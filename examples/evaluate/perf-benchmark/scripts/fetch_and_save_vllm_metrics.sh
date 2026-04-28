#!/usr/bin/env bash
# Fetch vLLM metrics and save to file

set -euo pipefail

VLLM_URL="$1"
OUTPUT_FILE="$2"

METRICS_URL="${VLLM_URL%/}/metrics"

echo "[INFO] Fetching vLLM metrics from ${METRICS_URL}"

if curl -s -f --max-time 10 "${METRICS_URL}" > "${OUTPUT_FILE}"; then
    echo "[INFO] Saved metrics to ${OUTPUT_FILE}"
else
    echo "[ERROR] Failed to fetch metrics from ${METRICS_URL}" >&2
    exit 1
fi
