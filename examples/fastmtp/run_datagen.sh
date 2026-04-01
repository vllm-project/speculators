#!/usr/bin/env bash
# Steps 2 & 3: Tokenize the response JSONL, then extract last-layer hidden states.
#
# Prerequisites:
#   - JSONL file with 'conversations' field (from run_response_regen.sh or existing)
#   - vLLM server running (launch_server.sh)
#
# Usage:
#   MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct \
#   DATA=/mnt/data/gsm8k_responses.jsonl \
#   OUTPUT=/mnt/data/gsm8k_hidden_states \
#   bash examples/fastmtp/run_datagen.sh
#
# Optional overrides:
#   PREPROCESSED=./gsm8k_preprocessed   (Arrow dataset cache dir)
#   ENDPOINT=http://localhost:8000/v1   (vLLM endpoint)
#   CONCURRENCY=32                      (parallel vLLM requests)
#   MAX_SAMPLES=100                     (limit samples; useful for testing)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
DATA="${DATA:?Set DATA to the path of your response-regeneration JSONL file}"
OUTPUT="${OUTPUT:-./gsm8k_hidden_states}"
PREPROCESSED="${PREPROCESSED:-${OUTPUT}_preprocessed}"
ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # leave empty to process all samples

echo "=== FastMTP Hidden States Generation ==="
echo "  Model:         $MODEL"
echo "  Input JSONL:   $DATA"
echo "  Preprocessed:  $PREPROCESSED"
echo "  Output:        $OUTPUT"
echo "  Endpoint:      $ENDPOINT"
echo "  Concurrency:   $CONCURRENCY"
[ -n "$MAX_SAMPLES" ] && echo "  Max samples:   $MAX_SAMPLES"
echo ""

# Step 2: Tokenize JSONL → Arrow dataset (mtp mode skips token_freq.pt)
echo "--- Step 2: Preprocessing (tokenization) ---"
python "$REPO_ROOT/scripts/prepare_data.py" \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$PREPROCESSED" \
    --method mtp

# Step 3: Extract hidden states via the running vLLM server
echo ""
echo "--- Step 3: Hidden States Extraction ---"
echo "Waiting for vLLM server at $ENDPOINT..."
until curl -sf "${ENDPOINT}/models" > /dev/null 2>&1; do
    sleep 5
done
echo "Server ready."
echo ""

python "$REPO_ROOT/scripts/data_generation_offline2.py" \
    --preprocessed-data "$PREPROCESSED" \
    --output "$OUTPUT" \
    --endpoint "$ENDPOINT" \
    --concurrency "$CONCURRENCY" \
    --validate-outputs \
    ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}

echo ""
echo "Done. Hidden states saved to: $OUTPUT"
echo "Total files: $(ls "$OUTPUT"/hs_*.safetensors 2>/dev/null | wc -l) safetensors files"
