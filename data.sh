#!/bin/bash
#
# Run response regeneration for all 4 English language splits of Nemotron dataset
# Splits: chat, stem, math, code
#

MODEL="Qwen/Qwen3-8B"
TP_SIZE=1
DP_SIZE=8
CONCURRENCY=512
MAX_MODEL_LEN=16384

# Array of all nemotron splits
SPLITS=("chat" "stem" "math" "code")

echo "========================================="
echo "Nemotron Dataset Processing - All Splits"
echo "========================================="
echo ""

for split in "${SPLITS[@]}"; do
    echo "Processing split: $split"
    ./scripts/response_regeneration/run_all.sh \
        --model "$MODEL" \
        --tp-size "$TP_SIZE" \
        --dp-size "$DP_SIZE" \
        --dataset nemotron \
        --split "$split" \
        --concurrency "$CONCURRENCY" \
        --resume \
        --max-model-len "$MAX_MODEL_LEN"

    echo ""
    echo "Completed split: $split"
    echo "----------------------------------------"
    echo ""
done

echo "========================================="
echo "All Nemotron splits processed!"
echo "========================================="
