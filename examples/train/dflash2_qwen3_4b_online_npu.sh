#!/usr/bin/env bash
# Online DFlash2 training for Qwen3-4B on Ascend NPU.
#
# Runs the full online pipeline: data preparation, vllm-ascend server launch,
# and training with hidden states generated on-the-fly from the live server.
#
# Usage:
#   bash examples/train/dflash2_qwen3_4b_online_npu.sh
#
# Prerequisites:
#   1. torch + torch_npu installed (CANN toolkit set up, e.g. via
#      `source /usr/local/Ascend/ascend-toolkit/set_env.sh`).
#   2. vllm-ascend installed for the verifier server; the same environment
#      must provide the hs_connectors "file" backend (default). The mooncake
#      backend is not NPU-ready yet (its vLLM-side connector uses CUDA
#      streams directly).
#   3. VLLM_NPUS / TRAIN_NPUS partition the visible Ascend devices between
#      the verifier server and the trainer; a 4B verifier fits on one NPU.
#      Multi-NPU training uses HCCL through torch.distributed.
#
# NPU-specific switches (do not drop them):
#   --draft-attn-impl sdpa   flex_attention is CUDA/CPU/HPU-only upstream
#                            (vllm-project/speculators#531); sdpa uses a dense
#                            eager mask instead of a compiled BlockMask.
#   --loss-implementation eager
#                            the default fused loss is a Triton kernel, which
#                            is unavailable on NPU.

set -euo pipefail

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET="${DATASET:-sharegpt}"      # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash2_qwen3_4b_online_npu}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"    # lower this first on OOM (eager loss + dense mask)
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-4}"

# DFlash2-specific parameters
BLOCK_SIZE=8
MAX_ANCHORS=256
NUM_LAYERS=5
TARGET_LAYER_IDS=(1 9 17 25 33)     # Must match vLLM's eagle_aux_hidden_state_layer_ids
DRAFT_VOCAB_SIZE=151936
CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16
SELECTOR_RANK=256
SELECTOR_TOP_K=16

# NPU assignments (online training needs separate devices for vLLM and training)
VLLM_NPUS="${VLLM_NPUS:-0}"
TRAIN_NPUS="${TRAIN_NPUS:-1,2}"
MASTER_PORT="${MASTER_PORT:-29501}"
# =======================================

die() {
    echo "error: $*" >&2
    exit 1
}

IFS=',' read -r -a TRAIN_NPU_IDS <<< "$TRAIN_NPUS"
NUM_TRAIN_NPUS="${#TRAIN_NPU_IDS[@]}"

# Optional availability preflight; npu-smi is not present in every container.
if command -v npu-smi > /dev/null 2>&1; then
    for npu in ${VLLM_NPUS//,/ } "${TRAIN_NPU_IDS[@]}"; do
        npu-smi info -i "$npu" > /dev/null 2>&1 ||
            die "NPU $npu is not visible (check ASCEND_RT_VISIBLE_DEVICES)"
    done
else
    echo "warning: npu-smi not found; skipping NPU availability preflight" >&2
fi

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch the vllm-ascend server in the background
echo "=== Step 2: Launching vLLM server on NPU $VLLM_NPUS ==="
ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    -- --port "$VLLM_PORT" &
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

# Step 3: Train against the live server (hidden states generated on-the-fly)
echo "=== Step 3: Training on NPU $TRAIN_NPUS ==="
mkdir -p "$OUTPUT_DIR/logs"
PYTHONPATH="$PWD/src:$PWD/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}" \
    ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node "$NUM_TRAIN_NPUS" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type dflash2 \
    --num-layers "$NUM_LAYERS" \
    --block-size "$BLOCK_SIZE" \
    --no-sample-from-anchor \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --total-seq-len "$SEQ_LENGTH" \
    --max-anchors "$MAX_ANCHORS" \
    --loss-fn '{"ce":0.1,"tv":0.9}' \
    --loss-implementation eager \
    --per-position-loss-weight fixed-exp-decay \
    --dflash-decay-gamma 4 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay 0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --epochs "$EPOCHS" \
    --seed 42 \
    --draft-attn-impl sdpa \
    --sliding-window-non-causal \
    --hidden-states-dtype bfloat16 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --on-missing generate \
    --on-generate delete \
    --no-resume-from-checkpoint \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --selector-loss-alpha 0.1 \
    2>&1 | tee "$OUTPUT_DIR/logs/dflash2_npu.log"

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
