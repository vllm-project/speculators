#!/bin/bash
# Continue the DFlash Qwen3-8B example on a second target-generated dataset.
#
# First complete dflash_qwen3_8b_sharegpt_online_5k.sh, then point PROMPT_FILE
# at a local JSON/JSONL prompt set. The stage-one architecture is restored from
# its checkpoint; this script starts a fresh optimizer and output directory.

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
PROMPT_FILE="./stage2_prompts.jsonl"
STAGE1_DIR="./output/dflash_qwen3_8b_sharegpt"
STAGE1_CHECKPOINT="$STAGE1_DIR/checkpoints/checkpoint_best"
STAGE2_DIR="./output/dflash_qwen3_8b_stage2"
RESPONSES="$STAGE2_DIR/responses.jsonl"
STAGE2_DATA="$STAGE2_DIR/data"

VLLM_PORT=8000
SEQ_LENGTH=8192
MAX_RESPONSE_TOKENS=4096
EPOCHS=5
LR=3e-4
MAX_ANCHORS=3072
TARGET_LAYER_IDS="2 18 33"

VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =======================================

[[ -f "$PROMPT_FILE" ]] || {
    echo "Prompt file not found: $PROMPT_FILE" >&2
    exit 1
}
[[ -d "$STAGE1_CHECKPOINT" ]] || {
    echo "Stage-one checkpoint not found: $STAGE1_CHECKPOINT" >&2
    exit 1
}
for mapping in "$STAGE1_DIR/d2t.npy" "$STAGE1_DIR/t2d.npy"; do
    [[ -f "$mapping" ]] || {
        echo "Stage-one vocabulary mapping not found: $mapping" >&2
        exit 1
    }
done

# Step 1: Generate second-stage responses with the target model.
echo "=== Step 1: Regenerating responses ==="
./scripts/response_regeneration/run_all.sh \
    --model "$MODEL" \
    --gpus "$VLLM_GPUS" \
    --dp-size 2 \
    --dataset-file "$PROMPT_FILE" \
    --outfile "$RESPONSES" \
    --sampling-params '{"temperature": 0}' \
    --max-tokens "$MAX_RESPONSE_TOKENS"

# Step 2: Prepare the target-generated data.
echo "=== Step 2: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$RESPONSES" \
    --output "$STAGE2_DATA" \
    --seq-length "$SEQ_LENGTH"

# Step 3: Launch the target for online hidden-state generation.
echo "=== Step 3: Launching vLLM ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done

# Step 4: Warm-start the draft. Reuse stage one's vocabulary mapping.
echo "=== Step 4: Training stage two ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --speculator-type dflash \
    --from-pretrained "$STAGE1_CHECKPOINT" \
    --data-path "$STAGE2_DATA" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$STAGE2_DIR/checkpoints" \
    --d2t-path "$STAGE1_DIR/d2t.npy" \
    --t2d-path "$STAGE1_DIR/t2d.npy" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --max-anchors "$MAX_ANCHORS" \
    --loss-fn kl_div \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $STAGE2_DIR/checkpoints/"
