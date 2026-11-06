#!/bin/bash
# Online LiLiCorr Training Script -- Paper Recipe
#
# Trains a LiLiCorr drafter (arXiv:2608.20530) for Qwen/Qwen3-4B: a DFlash
# backbone plus a candidate-lattice head that rescores the backbone's per-slot
# top-k candidates, so serving can commit a whole greedy path through the
# lattice instead of only top-1 tokens. Settings below follow the paper's main
# model: K=8 candidates, block_size=16 (15 speculative tokens), 5 draft layers
# fed by verifier layers 2/10/18/26/34, 512 anchors per ~3k-token sequence,
# fixed-exponential-decay per-position weighting with gamma=7, lr 6e-4, no
# weight decay, cosine schedule with 4% warmup (ModelOpt's recipe), and the
# base objective (DFlash loss + 0.25 * lattice CE + 0.25 * distractor penalty).
#
# The full vocab is kept (no --draft-vocab-size): vLLM's LiLiCorr runtime
# cannot serve a reduced draft vocabulary. Draft sampling at serve time is
# greedy (draft_sample_method "greedy" in the speculative config), which is
# what the paper reports its main numbers with.
#
# The dataset below is UltraChat prompts with every assistant turn regenerated
# on-policy (pretokenized, with input_ids + loss_mask).
#
# Usage:
#   bash examples/train/lilicorr_qwen3_4b_online.sh                 # full run, 100k samples
#   RUN_MODE=smoke bash examples/train/lilicorr_qwen3_4b_online.sh  # 5-sample smoke test
#
# For a detailed walkthrough of the online pipeline, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/

### Example E2E run for LiLiCorr Qwen3-4B on 100k regenerated UltraChat samples ###

# Smoke mode (RUN_MODE=smoke) uses 5 samples and a single training GPU to
# verify the pipeline end-to-end in minutes. Even with 100k samples the
# drafter will not reach the paper's acceptance numbers (the paper trains
# with a global batch of 64 sequences on 64 GPUs; this script uses 2
# training GPUs, so scale lr/steps accordingly if you chase that regime),
# but the lattice head learns: watch for lilicorr_ce dropping below
# ln(8) ~= 2.08 and lilicorr_eal rising above lilicorr_top1_eal. Run your
# own eval afterward (see examples/evaluate/).

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-4B"
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"  # on-policy regenerated data from UltraChat prompts (pretokenized); or a preset/path to custom data
VLLM_PORT=8000
LR=6e-4
EPOCHS=6

# Run mode: "smoke" (5 samples, 1 training GPU, minutes) or "full" (100k
# samples, 2 training GPUs).
RUN_MODE="${RUN_MODE:-full}"
if [[ "$RUN_MODE" == "smoke" ]]; then
    OUTPUT_DIR="./output/lilicorr_qwen3_4b_smoke"
    MAX_SAMPLES=5
    EPOCHS=1
else
    OUTPUT_DIR="./output/lilicorr_qwen3_4b_ultrachat_100k"
    MAX_SAMPLES=100000
fi

# LiLiCorr parameters (paper's main model)
SPECULATOR_TYPE="lilicorr"
BLOCK_SIZE=16
MAX_ANCHORS=512
SEQ_LENGTH=3072  # paper used ~3120; 3072 keeps the paper's 512 anchors per sequence
NUM_LAYERS=5
PER_POSITION_LOSS_WEIGHT="fixed-exp-decay"
DFLASH_DECAY_GAMMA=7.0  # paper value; the lilicorr default is 4.0
LOSS_FN="ce"
TARGET_LAYER_IDS="2 10 18 26 34"  # Must match vLLM's eagle_aux_hidden_state_layer_ids
MASK_TOKEN_ID=151669  # unused Qwen3 embedding slot, the ModelOpt/paper choice

# Optimizer (paper / ModelOpt recipe)
WEIGHT_DECAY=0
SCHEDULER_TYPE="cosine"
SCHEDULER_WARMUP_RATIO=0.04

# Note: no DRAFT_VOCAB_SIZE -- LiLiCorr requires the full verifier vocabulary.

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_DP_SIZE=2
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
if [[ "$RUN_MODE" == "smoke" ]]; then
    VLLM_DP_SIZE=1
    VLLM_GPUS="0"
    TRAIN_GPUS="1"
    NUM_TRAIN_GPUS=1
fi
# =======================================

# Step 1: Prepare data
# The dataset is on-policy regenerated: assistant turns were produced by the
# target model and ship pretokenized (input_ids + loss_mask). prepare-data just
# packages them -- no --render-endpoint (or running vLLM server) needed here.
echo "=== Step 1: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$VLLM_DP_SIZE" --port "$VLLM_PORT" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
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
echo "vLLM server ready."

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --scheduler-type "$SCHEDULER_TYPE" \
    --scheduler-warmup-ratio "$SCHEDULER_WARMUP_RATIO" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --per-position-loss-weight "$PER_POSITION_LOSS_WEIGHT" \
    --dflash-decay-gamma "$DFLASH_DECAY_GAMMA" \
    --loss-fn "$LOSS_FN" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --mask-token-id "$MASK_TOKEN_ID" \
    --on-missing generate

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
