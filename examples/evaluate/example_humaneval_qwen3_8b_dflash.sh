#!/bin/bash
# Example: Performance benchmark with vLLM speculative decoding metrics
#
# This example shows how to run the full performance benchmark pipeline
# and extract vLLM speculative decoding metrics.
#
# Prerequisites:
#   cd examples/evaluate
#   pip install -r ../../scripts/evaluate/requirements.txt
#
# Usage:
#   bash examples/evaluate/example_humaneval_qwen3_8b_dflash.sh
#
# Expected File Structure:
#   The script uses the HuggingFace dataset RedHatAI/speculator_benchmarks,
#   which contains HumanEval.jsonl - a code completion benchmark with 164
#   programming problems. Each entry has a "prompt" column containing a
#   function signature and docstring to complete. For custom datasets,
#   ensure your dataset directory contains:
#     - <subset_name>.jsonl files
#     - Each file should have a "prompt" column (or use --data-column-mapper)
#
# Output Results:
#   Creates a timestamped directory (perf_results_YYYYMMDD_HHMMSS/) containing:
#     gen_len/
#       gen_len_HumanEval.json     - Raw output length distribution data
#       max_tokens_HumanEval.json  - Computed max_tokens cap (power-of-2)
#     sweeps/
#       sweep_HumanEval.json       - Full GuideLLM sweep results (11MB)
#     HumanEval_baseline_metrics.txt   - vLLM metrics snapshot before sweep
#     HumanEval_current_metrics.txt    - vLLM metrics snapshot after sweep
#     HumanEval_partial.csv            - Performance + spec decode metrics
#     perf_results.csv                 - Final consolidated CSV with columns:
#       Performance: subset, strategy, target_rate, rps_median, latency_median_s,
#                    itl_median_ms, ttft_median_ms, output_tps_median
#       Spec Decode: num_drafts, num_draft_tokens, num_accepted_tokens,
#                    acceptance_length, acceptance_at_pos_0..N

set -euo pipefail

# ============ Configuration ============
MODEL="nm-testing/dflash-qwen3-8b-speculators"
VLLM_PORT=8000
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

# Step 2: Run the main performance benchmark script
echo ""
echo "=== Step 2: Running performance benchmark ==="
export VLLM_URL="${SERVER_URL}"
"$SCRIPT_DIR/run_perf_benchmark.sh" \
    --target "${SERVER_URL}/v1" \
    --subsets "HumanEval" \
    --max-requests 80

echo ""
echo "Done. Check the output directory for results."
