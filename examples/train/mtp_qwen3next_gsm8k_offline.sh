#!/bin/bash
# Offline MTP Finetuning Script
#
# Runs the full MTP finetuning pipeline: data preparation, vLLM server launch,
# hidden states generation, model conversion, training, and stitching.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/mtp_qwen3next_gsm8k_offline.sh
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_mtp_offline/

### Example E2E run for Qwen3-Next-80B on GSM8K ###

# Note: This script covers stages 2-6 of the MTP finetuning pipeline.
# Stage 1 (response regeneration) and stage 7 (deployment) are left to
# the user since they require a running vLLM server.

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
DATASET="gsm8k"
OUTPUT_DIR="./output"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
MTP_HEAD_DIR="$OUTPUT_DIR/mtp_head"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
STITCHED_DIR="$OUTPUT_DIR/stitched_model"
VLLM_PORT=8000
MAX_SAMPLES=10000
SEQ_LENGTH=8192
EPOCHS=3
LR=1e-4
NUM_SPECULATIVE_STEPS=3
STEP_WEIGHT_BETA=0.6
CONCURRENCY=32

# GPU assignments (offline reuses the same GPUs sequentially)
VLLM_GPUS="0,1,2,3,4,5,6,7"
VLLM_TP_SIZE=8
TRAIN_GPUS="0,1"
NUM_TRAIN_GPUS=2

# For MTP, only the last hidden layer is needed.
# Qwen3-Next-80B has 64 layers, so the last layer ID is 64.
TARGET_LAYER_ID=64
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server for hidden states extraction
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids "$TARGET_LAYER_ID" \
    -- --tensor-parallel-size "$VLLM_TP_SIZE" --port "$VLLM_PORT" &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Generate hidden states
echo "=== Step 3: Generating hidden states ==="
python scripts/data_generation_offline.py \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --validate-outputs

# Step 4: Stop vLLM server to free GPU memory
echo "=== Step 4: Stopping vLLM server ==="
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM server stopped. GPUs freed for training."

# Step 5: Convert -- extract MTP head from native checkpoint
echo "=== Step 5: Extracting MTP head ==="
python -c "
from speculators.convert.mtp import MTPConverter

converter = MTPConverter()
converter.convert(
    input_path='$MODEL',
    output_path='$MTP_HEAD_DIR',
    base_model='$MODEL',
    num_speculative_steps=$NUM_SPECULATIVE_STEPS,
)
"

# Step 6: Train the MTP head
echo "=== Step 6: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --speculator-type mtp \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --num-speculative-steps "$NUM_SPECULATIVE_STEPS" \
    --step-weight-beta "$STEP_WEIGHT_BETA" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise

# Step 7: Stitch -- reintegrate finetuned weights into verifier checkpoint
echo "=== Step 7: Stitching finetuned weights ==="
python -c "
from speculators.convert.mtp import MTPStitcher

stitcher = MTPStitcher()
stitcher.stitch(
    finetuned_checkpoint='$CHECKPOINT_DIR/checkpoint_best',
    verifier_path='$MODEL',
    output_path='$STITCHED_DIR',
)
"

echo "Done. Stitched checkpoint saved to $STITCHED_DIR/"
echo ""
echo "To deploy with vLLM:"
echo "  vllm serve $STITCHED_DIR \\"
echo "    --tensor-parallel-size 8 \\"
echo "    --speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}' \\"
echo "    --no-enable-chunked-prefill"
