#!/bin/bash
# DSpark Warm-Start Training Script (from DFlash checkpoint)
#
# Warm-starts a DSpark model from the published DFlash checkpoint for
# google/gemma-4-31B-it.  The shared backbone (decoder layers, fc, norms,
# embeddings, lm_head) is copied from the DFlash checkpoint; the DSpark-
# specific heads (MarkovHead, ConfidenceHead) are randomly initialized and
# trained from scratch.
#
# The DFlash checkpoint's config (block_size, draft_vocab_size, mask_token_id,
# transformer_layer_config, target_layer_ids) is inherited automatically via
# the cross-type warm-start path in train.py.
#
# Usage:
#   bash examples/train/dspark_gemma4_31b_warmstart_from_dflash.sh

set -euo pipefail

# ============ Configuration ============
MODEL="google/gemma-4-31B-it"
DFLASH_CHECKPOINT="RedHatAI/gemma-4-31B-it-speculator.dflash"
DATASET="hf:inference-optimization/gemma4-31B-responses"
OUTPUT_DIR="./output/dspark_gemma4_31b_warmstart"
VLLM_PORT=8000
MAX_SAMPLES=50000
SEQ_LENGTH=8192
EPOCHS=10
LR=1e-4

# Target layer IDs — must match the DFlash checkpoint and the vLLM server.
TARGET_LAYER_IDS="1 17 29 47 58"

# DSpark-specific parameters
MARKOV_RANK=256              # low-rank dim for Markov logit-bias head (0 disables)
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
MAX_ANCHORS=3072

# GPU assignments — online training needs separate GPUs for vLLM and training.
# gemma-4-31B-it requires TP>=4 for vLLM with default max-model-len; adjust as needed.
VLLM_GPUS="0,1,2,3"
VLLM_TP=4
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --tensor-parallel-size "$VLLM_TP" --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

# Step 3: Train DSpark, warm-started from the DFlash checkpoint
echo "=== Step 3: Training DSpark (warm start from DFlash) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --speculator-type dspark \
    --from-pretrained "$DFLASH_CHECKPOINT" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --max-anchors "$MAX_ANCHORS" \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --on-missing generate \
    --on-generate delete

echo "Done. DSpark checkpoints saved to $OUTPUT_DIR/checkpoints/"
