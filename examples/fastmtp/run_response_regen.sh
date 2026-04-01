#!/usr/bin/env bash
# Step 1 (optional): Regenerate model responses for GSM8K.
# Skip if your JSONL file already exists.
#
# Prerequisites: a vLLM server running at $ENDPOINT serving $MODEL
#
# Usage:
#   MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct \
#   ENDPOINT=http://localhost:8000 \
#   OUTFILE=/mnt/data/gsm8k_responses.jsonl \
#   bash examples/fastmtp/run_response_regen.sh
#
# Optional overrides:
#   CONCURRENCY=64  (parallel requests, default 64)
#   LIMIT=100       (process only first N rows; useful for testing)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
OUTFILE="${OUTFILE:-gsm8k_${MODEL##*/}.jsonl}"
CONCURRENCY="${CONCURRENCY:-64}"
LIMIT="${LIMIT:-}"  # leave empty to process all rows

echo "=== GSM8K Response Regeneration ==="
echo "  Model:       $MODEL"
echo "  Endpoint:    $ENDPOINT"
echo "  Output:      $OUTFILE"
echo "  Concurrency: $CONCURRENCY"
[ -n "$LIMIT" ] && echo "  Limit:       $LIMIT rows"
echo ""

python "$REPO_ROOT/scripts/response_regeneration/script.py" \
    --dataset gsm8k \
    --model "$MODEL" \
    --endpoint "$ENDPOINT" \
    --outfile "$OUTFILE" \
    --concurrency "$CONCURRENCY" \
    ${LIMIT:+--limit "$LIMIT"}

echo ""
echo "Done: $OUTFILE"
