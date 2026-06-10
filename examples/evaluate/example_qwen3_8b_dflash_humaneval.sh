#!/bin/bash
# Example: Speculative decoding evaluation with Qwen3-8B dflash
#
# This example launches a vLLM server and runs an acceptance rate
# evaluation. Change "throughput" to "sweep" for a full performance
# benchmark with gen-len estimation and multi-rate sweep.
#
# Prerequisites:
#   pip install -r ../../scripts/evaluate/requirements.txt
#
# Usage:
#   bash examples/evaluate/example_qwen3_8b_dflash_humaneval.sh
#
# Output Results:
#   Creates a timestamped directory (<model_name>_YYYYMMDD_HHMMSS/) containing:
#     acceptance.csv                   - Per-position acceptance rates CSV
#     artifacts/
#       run_HumanEval.json             - Raw GuideLLM output
#
#   Printed report includes:
#     num_drafts, num_draft_tokens, num_accepted_tokens,
#     acceptance_length, acceptance_at_pos_0..N

set -euo pipefail

# ============ Configuration ============
MODEL="nm-testing/dflash-qwen3-8b-speculators"
DATASET="RedHatAI/speculator_benchmarks"
VLLM_PORT=8108
SERVER_URL="http://localhost:${VLLM_PORT}"
# Uses CUDA_VISIBLE_DEVICES from environment (or set it before running this script)
# =======================================

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$EXAMPLE_DIR/../../scripts/evaluate" && pwd)"

# Check for guidellm
if ! command -v guidellm &> /dev/null; then
    echo "ERROR: guidellm not found. Install dependencies first:"
    echo "  pip install -r $SCRIPT_DIR/requirements.txt"
    exit 1
fi

# Step 1: Launch vLLM server
echo "=== Step 1: Launching vLLM server ==="
vllm serve "$MODEL" --port "$VLLM_PORT" &
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

# Step 2: Run acceptance rate only (no gen-len estimation or sweep)
echo ""
echo "=== Step 2: Running acceptance rate evaluation ==="
python "$SCRIPT_DIR/evaluate.py" \
    --target "${SERVER_URL}/v1" \
    --dataset "$DATASET" \
    throughput \
    --subsets "HumanEval" \
    --max-requests 80

echo ""
echo "Done. Check the output directory for results."
