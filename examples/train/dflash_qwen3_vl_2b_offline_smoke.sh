#!/usr/bin/env bash
# Offline multimodal (image+text) DFlash smoke run for Qwen3-VL-2B-Instruct.
#
# Usage:
#   bash examples/train/dflash_qwen3_vl_2b_offline_smoke.sh
#
# This launcher is training-only. Before running it:
#   1. Serve the VL verifier with scripts/launch_vllm.py and TARGET_LAYER_IDS,
#      passing --allowed-local-media-path <image_root> after "--" so vLLM can
#      read local image files. Hidden-state extraction rides the normal Chat
#      Completions path for image rows, so no extra switch is needed there.
#   2. Prepare and tokenize the multimodal conversations with
#      scripts/prepare_data.py --render-endpoint, writing DATA_PATH. The
#      built-in sharegpt4v_coco preset (COCO_DIR env var) produces the
#      expected `messages` column with {"type": "image", "path": ...} parts;
#      custom ShareGPT-style JSONL with the same shape works as well.
#   3. Extract the matching hidden states with
#      scripts/data_generation_offline.py, writing HIDDEN_STATES_PATH.
#   4. Stop the target server so the GPUs are free.
#
# The draft config is built plain-rope for DFlash speculators: mrope_section
# from the verifier's text_config is stripped because the draft only consumes
# the captured verifier hidden states (the verifier's own forward already
# encodes vision into them) and vLLM's DFlash serving path rejects MRoPE
# draft configs.
#
# The default is a short 10-step smoke. For a 100-step signal run:
#   STEPS=100 EPOCHS=5 CHECKPOINT_FREQ=5 \
#     bash examples/train/dflash_qwen3_vl_2b_offline_smoke.sh

set -euo pipefail

# ============ Configuration ============
VERIFIER="${VERIFIER:-Qwen/Qwen3-VL-2B-Instruct}"
RUN_ROOT="${RUN_ROOT:-./output/qwen3_vl_2b_dflash_vl_smoke}"
DATA_PATH="${DATA_PATH:-$RUN_ROOT/data}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-$RUN_ROOT/hidden_states}"

GPUS="${GPUS:-0,1}"
MASTER_PORT="${MASTER_PORT:-29503}"

STEPS="${STEPS:-10}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-6e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
SEQ_LENGTH=1024
MAX_ANCHORS=128
NUM_LAYERS=5
BLOCK_SIZE=8
DRAFT_VOCAB_SIZE=151936
# Qwen3-VL-2B has 26 text layers: [2, L//2, L-3, L].
TARGET_LAYER_IDS=(2 13 23 25)
# =======================================

die() {
    echo "error: $*" >&2
    exit 1
}

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NUM_GPUS="${#GPU_IDS[@]}"

for gpu in "${GPU_IDS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id in GPUS: $gpu"
done

if command -v nvidia-smi > /dev/null 2>&1; then
    for gpu in "${GPU_IDS[@]}"; do
        nvidia-smi -i "$gpu" > /dev/null 2>&1 ||
            die "GPU $gpu is not visible"
    done
else
    echo "warning: nvidia-smi not found; skipping GPU availability preflight" >&2
fi

[[ -d "$DATA_PATH" ]] ||
    die "DATA_PATH does not exist: $DATA_PATH"
compgen -G "$DATA_PATH/*.arrow" > /dev/null ||
    die "DATA_PATH has no prepared Arrow shards: $DATA_PATH"
[[ -f "$DATA_PATH/token_freq.pt" ]] ||
    die "DATA_PATH is missing token_freq.pt: $DATA_PATH"
[[ -d "$HIDDEN_STATES_PATH" ]] ||
    die "HIDDEN_STATES_PATH does not exist: $HIDDEN_STATES_PATH"
compgen -G "$HIDDEN_STATES_PATH/hs_*.safetensors" > /dev/null ||
    die "HIDDEN_STATES_PATH has no hs_*.safetensors cache files"

mkdir -p "$RUN_ROOT/logs"

PYTHONPATH="$PWD/src:$PWD/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    scripts/train.py \
    --verifier-name-or-path "$VERIFIER" \
    --data-path "$DATA_PATH" \
    --hidden-states-backend file \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --save-path "$RUN_ROOT/dflash/checkpoints" \
    --speculator-type dflash \
    --num-layers "$NUM_LAYERS" \
    --block-size "$BLOCK_SIZE" \
    --no-sample-from-anchor \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --total-seq-len "$SEQ_LENGTH" \
    --train-data-ratio 0.9 \
    --max-anchors "$MAX_ANCHORS" \
    --loss-fn '{"ce":0.1,"tv":0.9}' \
    --per-position-loss-weight fixed-exp-decay \
    --dflash-decay-gamma 4 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay 0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --epochs "$EPOCHS" \
    --max-steps "$STEPS" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --seed 42 \
    --draft-attn-impl simple_flex_attention \
    --sliding-window-non-causal \
    --hidden-states-dtype bfloat16 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --on-missing raise \
    --no-resume-from-checkpoint \
    2>&1 | tee "$RUN_ROOT/logs/dflash_vl.log"

echo "Multimodal DFlash smoke run complete: $RUN_ROOT/dflash/checkpoints"
