#!/usr/bin/env bash
# Qwen3-4B D-PARD training on cached native Speculators data.
# Run from the repository root:
#   DATA_PATH=/path/to/native_dataset bash examples/train/dspark_qwen3_4b_dpard_offline.sh
# Cache target layers [1, 17, 33] and the verifier's final hidden states.
# The 512 anchors are shared by each 8192-token packed sequence.

set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the native Speculators dataset directory}"
OUTPUT_DIR=${OUTPUT_DIR:-./output/dspark_qwen3_4b_dpard_b16}
NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-2}

# This split reserves one validation row from 36,624; adjust for other datasets.
torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS" scripts/train.py \
    --verifier-name-or-path Qwen/Qwen3-4B \
    --data-path "$DATA_PATH" \
    --save-path "$OUTPUT_DIR" \
    --speculator-type dspark \
    --draft-vocab-size 151936 \
    --num-layers 3 \
    --target-layer-ids 1 17 33 \
    --block-size 16 \
    --sample-from-anchor \
    --max-anchors 512 \
    --sliding-window 2048 \
    --no-sliding-window-non-causal \
    --total-seq-len 8192 \
    --train-data-ratio 0.9999726955002184 \
    --noise-std 0.05 \
    --hidden-states-dtype bfloat16 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --loss-implementation fused \
    --loss-fn renyi_half \
    --per-position-loss-weight dpard \
    --dpard-alpha 0.5 \
    --dflash-decay-gamma 7.0 \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha 1.0 \
    --optimizer adamw \
    --lr 6e-4 \
    --weight-decay 0.01 \
    --scheduler-type linear \
    --scheduler-warmup-ratio 0.04 \
    --seed 42 \
    --epochs 6 \
    --checkpoint-freq 1 \
    --no-resume-from-checkpoint \
    --log-freq 25 \
    --on-missing raise \
    --on-generate delete
