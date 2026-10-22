#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?producer-hybrid|producer-lilicorr|hspec|lilicorr|combined}
MODEL=Qwen/Qwen3-4B; DATA=${DATA:?Set DATA to the prepared 100k dataset}; OUT=${OUT:-"$PWD/output/100k"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
TRANSFER="$OUT/transfer-hybrid"; if [[ "$MODE" == *lilicorr ]]; then TRANSFER="$OUT/transfer-lilicorr"; fi
mkdir -p "$TRANSFER" "$OUT/$MODE"
COMMON=(--verifier-name-or-path "$MODEL" --data-path "$DATA" --train-data-ratio 0.99 --seed 42 --optimizer adamw --optimizer-implementation fused --lr 6e-4 --hidden-states-dtype bfloat16 --mask-token-id 151669 --sliding-window 2048 --num-workers 2 --prefetch-factor 2 --checkpoint-freq 1 --log-freq 10)
COMMON+=(--hidden-states-backend file --hidden-states-path "$TRANSFER" --on-missing generate --vllm-endpoint http://127.0.0.1:8000/v1 --save-path "$OUT/$MODE/checkpoints" --log-dir "$OUT/$MODE/logs" --run-name "$MODE")
HYBRID=(--speculator-type mamba_attn_hybrid --block-pattern mamba mlp mamba attention mlp mamba attention mlp mamba attention mlp --attn-kv-layer-ids 17 25 34 --latent-fusion-layer-ids 1 9 17 25 34)
HYBRID+=(--mamba-seed-mode shared --mamba-num-heads 44 --mamba-head-dim 64 --mamba-d-state 4 --mamba-n-groups 4 --mamba-conv-kernel 4 --trainable-q-proj --trainable-kv-proj --use-qk-norm --sliding-window-indices 0 1 2 --no-sliding-window-non-causal --parallel-drafting-loss-weight dflash)
LILI=(--epochs 6 --total-seq-len 3120 --max-anchors 3072 --block-size 16 --scheduler-type linear --scheduler-warmup-ratio 0.04)
SERVER=(--tensor-parallel-size 2 --dtype bfloat16 --host 127.0.0.1 --port 8000 --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 8 --gpu-memory-utilization 0.80 --no-enable-prefix-caching --no-enable-chunked-prefill)
train() { CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 -m speculators.train "${COMMON[@]}" "$@"; }
case "$MODE" in
producer-hybrid) CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm_nkv.py "$MODEL" --hidden-states-path "$TRANSFER" --attn-kv-layer-ids 17 25 34 --latent-fusion-layer-ids 1 9 17 25 34 -- "${SERVER[@]}" ;;
producer-lilicorr) CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py train "$MODEL" --hidden-states-path "$TRANSFER" --target-layer-ids 2 10 18 26 34 --provenance-dir "$OUT/producer-lilicorr" -- "${SERVER[@]}" ;;
hspec) train "${HYBRID[@]}" --epochs 5 --total-seq-len 8192 --max-anchors 3072 --block-size 8 --dflash-decay-gamma 4 --scheduler-type cosine --scheduler-num-cosine-cycles 0.5 --scheduler-warmup-ratio 0.01 --weight-decay 0.01 --loss-fn '{"ce":0.1,"tv":0.9}' --markov-rank 256 --markov-head-type vanilla --draft-vocab-size 32000 --token-freq-path "$DATA/token_freq.pt" --enable-confidence-head --confidence-head-with-markov --confidence-head-alpha 1.0 ;;
lilicorr) train "${LILI[@]}" --dflash-decay-gamma 7 --speculator-type lilicorr --num-layers 5 --target-layer-ids 2 10 18 26 34 --no-sample-from-anchor --sliding-window-non-causal --per-position-loss-weight fixed-exp-decay --weight-decay 0 --loss-fn ce ;;
combined) train "${HYBRID[@]}" "${LILI[@]}" --dflash-decay-gamma 4 --enable-lilicorr --markov-rank 0 --no-enable-confidence-head --no-confidence-head-with-markov --weight-decay 0.01 --loss-fn '{"ce":0.1,"tv":0.9}' ;;
*) exit 2 ;; esac
