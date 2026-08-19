#!/usr/bin/env bash
# Paired offline DFlash2 vs DSpark smoke run for Qwen3-4B.
#
# Usage:
#   bash examples/train/dflash2_dspark_qwen3_4b_offline_smoke.sh
#
# This launcher is training-only. Before running it:
#   1. Serve VERIFIER with scripts/launch_vllm.py and TARGET_LAYER_IDS.
#   2. Prepare and tokenize the conversations with scripts/prepare_data.py
#      --render-endpoint, writing DATA_PATH.
#   3. Extract the matching hidden states with
#      scripts/data_generation_offline.py, writing HIDDEN_STATES_PATH.
#   4. Stop the target server so all eight GPUs are free.
#
# Inspect the node with nvidia-smi first. The preflight below also refuses to
# launch when a requested GPU has an active compute process. The default is the
# short 10-step smoke used for the initial comparison. For the 100-step signal run:
#   STEPS=100 EPOCHS=5 CHECKPOINT_FREQ=5 \
#     bash examples/train/dflash2_dspark_qwen3_4b_offline_smoke.sh

set -euo pipefail

# ============ Configuration ============
VERIFIER="${VERIFIER:-Qwen/Qwen3-4B}"
RUN_ROOT="${RUN_ROOT:-./output/qwen3_4b_dflash2_dspark_smoke}"
DATA_PATH="${DATA_PATH:-$RUN_ROOT/data}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-$RUN_ROOT/hidden_states}"

DFLASH2_GPUS="${DFLASH2_GPUS:-4,5,6,7}"
DSPARK_GPUS="${DSPARK_GPUS:-0,1,2,3}"
NUM_GPUS_PER_RUN="${NUM_GPUS_PER_RUN:-4}"
DFLASH2_MASTER_PORT="${DFLASH2_MASTER_PORT:-29501}"
DSPARK_MASTER_PORT="${DSPARK_MASTER_PORT:-29502}"

STEPS="${STEPS:-10}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-6e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
SEQ_LENGTH=1024
MAX_ANCHORS=128
NUM_LAYERS=5
BLOCK_SIZE=8
DRAFT_VOCAB_SIZE=151936
TARGET_LAYER_IDS=(1 9 17 25 33)

CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16
SELECTOR_RANK=256
SELECTOR_TOP_K=16
# =======================================

die() {
    echo "error: $*" >&2
    exit 1
}

[[ "$NUM_GPUS_PER_RUN" =~ ^[1-9][0-9]*$ ]] ||
    die "NUM_GPUS_PER_RUN must be a positive integer"
[[ "$DFLASH2_MASTER_PORT" =~ ^[0-9]+$ ]] ||
    die "DFLASH2_MASTER_PORT must be an integer"
[[ "$DSPARK_MASTER_PORT" =~ ^[0-9]+$ ]] ||
    die "DSPARK_MASTER_PORT must be an integer"
[[ "$DFLASH2_MASTER_PORT" != "$DSPARK_MASTER_PORT" ]] ||
    die "the paired jobs need distinct master ports"

IFS=',' read -r -a DFLASH2_GPU_IDS <<< "$DFLASH2_GPUS"
IFS=',' read -r -a DSPARK_GPU_IDS <<< "$DSPARK_GPUS"

(( ${#DFLASH2_GPU_IDS[@]} == NUM_GPUS_PER_RUN )) ||
    die "DFLASH2_GPUS must contain $NUM_GPUS_PER_RUN GPU ids"
(( ${#DSPARK_GPU_IDS[@]} == NUM_GPUS_PER_RUN )) ||
    die "DSPARK_GPUS must contain $NUM_GPUS_PER_RUN GPU ids"

validate_gpu_set() {
    local name="$1"
    shift
    local gpu
    local seen_gpu
    local -a seen_ids=()

    for gpu in "$@"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id in $name: $gpu"
        for seen_gpu in "${seen_ids[@]}"; do
            [[ "$gpu" != "$seen_gpu" ]] ||
                die "$name contains duplicate GPU id: $gpu"
        done
        seen_ids+=("$gpu")
    done
}

validate_gpu_set "DFLASH2_GPUS" "${DFLASH2_GPU_IDS[@]}"
validate_gpu_set "DSPARK_GPUS" "${DSPARK_GPU_IDS[@]}"

for dflash2_gpu in "${DFLASH2_GPU_IDS[@]}"; do
    for dspark_gpu in "${DSPARK_GPU_IDS[@]}"; do
        [[ "$dflash2_gpu" != "$dspark_gpu" ]] ||
            die "GPU $dflash2_gpu appears in both GPU sets"
    done
done

command -v nvidia-smi > /dev/null 2>&1 ||
    die "nvidia-smi is required for the GPU availability preflight"

check_gpu_free() {
    local gpu="$1"
    local compute_pids

    nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader \
        > /dev/null 2>&1 ||
        die "GPU $gpu is not visible"
    if ! compute_pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
        --format=csv,noheader 2>/dev/null)"; then
        die "could not query compute processes on GPU $gpu"
    fi
    [[ -z "$compute_pids" ]] ||
        die "GPU $gpu is busy with compute PID(s): $compute_pids"
}

for gpu in "${DFLASH2_GPU_IDS[@]}" "${DSPARK_GPU_IDS[@]}"; do
    check_gpu_free "$gpu"
done

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

mkdir -p "$RUN_ROOT/dflash2" "$RUN_ROOT/dspark" "$RUN_ROOT/logs"

run_training() {
    local algorithm="$1"
    local gpu_set="$2"
    local master_port="$3"
    local save_path="$4"
    shift 4

    PYTHONPATH="$PWD/src:$PWD/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}" \
        CUDA_VISIBLE_DEVICES="$gpu_set" .venv/bin/torchrun \
        --nnodes 1 \
        --node_rank 0 \
        --nproc_per_node "$NUM_GPUS_PER_RUN" \
        --master_addr 127.0.0.1 \
        --master_port "$master_port" \
        scripts/train.py \
        --verifier-name-or-path "$VERIFIER" \
        --data-path "$DATA_PATH" \
        --hidden-states-backend file \
        --hidden-states-path "$HIDDEN_STATES_PATH" \
        --save-path "$save_path" \
        --speculator-type "$algorithm" \
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
        "$@"
}

DFLASH2_LOG="$RUN_ROOT/logs/dflash2.log"
DSPARK_LOG="$RUN_ROOT/logs/dspark.log"
DFLASH2_PID=""
DSPARK_PID=""

cleanup() {
    local pid
    for pid in "$DFLASH2_PID" "$DSPARK_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Launching DFlash2 on GPUs $DFLASH2_GPUS (port $DFLASH2_MASTER_PORT)"
run_training \
    dflash2 \
    "$DFLASH2_GPUS" \
    "$DFLASH2_MASTER_PORT" \
    "$RUN_ROOT/dflash2/checkpoints" \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --selector-loss-alpha 0.1 \
    > "$DFLASH2_LOG" 2>&1 &
DFLASH2_PID=$!

echo "Launching DSpark on GPUs $DSPARK_GPUS (port $DSPARK_MASTER_PORT)"
run_training \
    dspark \
    "$DSPARK_GPUS" \
    "$DSPARK_MASTER_PORT" \
    "$RUN_ROOT/dspark/checkpoints" \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha 1.0 \
    > "$DSPARK_LOG" 2>&1 &
DSPARK_PID=$!

set +e
wait "$DFLASH2_PID"
DFLASH2_STATUS=$?
DFLASH2_PID=""
wait "$DSPARK_PID"
DSPARK_STATUS=$?
DSPARK_PID=""
set -e

trap - EXIT INT TERM

if (( DFLASH2_STATUS != 0 || DSPARK_STATUS != 0 )); then
    echo "DFlash2 exit status: $DFLASH2_STATUS ($DFLASH2_LOG)" >&2
    echo "DSpark exit status: $DSPARK_STATUS ($DSPARK_LOG)" >&2
    exit 1
fi

echo "Paired smoke run complete."
echo "DFlash2: $RUN_ROOT/dflash2/checkpoints"
echo "DSpark:  $RUN_ROOT/dspark/checkpoints"
