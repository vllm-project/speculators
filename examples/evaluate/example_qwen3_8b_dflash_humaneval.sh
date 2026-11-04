#!/bin/bash
# Example: Speculative decoding evaluation with Qwen3-8B dflash
#
# Launch a server and collect native AIPerf acceptance reports.
# Prerequisites: pip install -r scripts/evaluate/requirements.txt
# Usage: bash examples/evaluate/example_qwen3_8b_dflash_humaneval.sh
# Output: profile_export_aiperf.{json,csv}, per-request JSONL, and provenance.

set -euo pipefail

# ============ Configuration ============
MODEL="nm-testing/dflash-qwen3-8b-speculators"
DATASET="spec_al_humaneval"
VLLM_PORT=8108
SERVER_URL="http://localhost:${VLLM_PORT}"
# Uses CUDA_VISIBLE_DEVICES from environment (or set it before running this script)
# =======================================

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$EXAMPLE_DIR/../../scripts/evaluate" && pwd)"

# Check for AIPerf
if ! command -v aiperf &> /dev/null; then
    echo "ERROR: aiperf not found. Install dependencies first:"
    echo "  pip install -r $SCRIPT_DIR/requirements.txt"
    exit 1
fi

# Step 1: Launch vLLM server
echo "=== Step 1: Launching vLLM server ==="
vllm serve "$MODEL" --port "$VLLM_PORT" \
    --per-request-spec-decode-metrics summary &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 2: Run acceptance on the native HumanEval dataset.
python "$SCRIPT_DIR/evaluate.py" acceptance \
    --url "$SERVER_URL" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --request-count 80 \
    --output-artifact-dir "output/humaneval-$(date +%Y%m%d_%H%M%S)"
