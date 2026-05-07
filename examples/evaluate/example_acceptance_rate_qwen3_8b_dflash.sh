#!/bin/bash
# Example: Per-position acceptance rate (no performance sweep)
#
# This example sends a small batch of requests through vLLM and reports
# per-position speculative decoding acceptance rates. It skips the
# expensive gen-len estimation and sweep steps from the full benchmark.
#
# Prerequisites:
#   cd examples/evaluate
#   pip install -r ../../scripts/evaluate/requirements.txt
#
# Usage:
#   bash examples/evaluate/example_acceptance_rate_qwen3_8b_dflash.sh
#
# Output Results:
#   Creates a timestamped directory (perf_results_YYYYMMDD_HHMMSS/) containing:
#     acceptance.csv                   - Per-position acceptance rates CSV
#     .artifacts/acceptance/
#       baseline_metrics.txt           - vLLM metrics snapshot before requests
#       current_metrics.txt            - vLLM metrics snapshot after requests
#       run_HumanEval.json             - Raw GuideLLM output
#
#   Printed report includes:
#     num_drafts, num_draft_tokens, num_accepted_tokens,
#     acceptance_length, acceptance_at_pos_0..N

set -euo pipefail

# ============ Configuration ============
MODEL="nm-testing/dflash-qwen3-8b-speculators"
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
export VLLM_URL="${SERVER_URL}"
"$SCRIPT_DIR/run_perf_benchmark.sh" \
    --target "${SERVER_URL}/v1" \
    --subsets "HumanEval" \
    --max-requests 80 \
    --acceptance-only

echo ""
echo "Done. Check the output directory for results."
