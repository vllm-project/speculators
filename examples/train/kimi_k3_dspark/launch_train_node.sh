#!/usr/bin/env bash
#
# DSpark draft training: 4 DDP ranks on one node, consuming hidden states from
# the extractor through Mooncake.
#
# Required: MOONCAKE_MASTER, VLLM_ENDPOINT, FABRIC_SUBNET

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
model_path="${MODEL_PATH:-/mnt/shared/weights/kimi-k3}"
data_dir="${DATA_DIR:-$repo_root/runs/kimi_k3_dspark/data}"
run_name="${RUN_NAME:-kimi-k3-dspark}"
run_dir="${RUN_DIR:-$repo_root/runs/kimi_k3_dspark/$run_name}"

mooncake_master="${MOONCAKE_MASTER:?MOONCAKE_MASTER is required}"
vllm_endpoint="${VLLM_ENDPOINT:?VLLM_ENDPOINT is required}"

fabric_subnet="${FABRIC_SUBNET:?FABRIC_SUBNET is required}"
fabric_iface="$(ip -o -4 addr show | awk -v s="^$fabric_subnet" '$4 ~ s {print $2; exit}')"
fabric_addr="$(ip -o -4 addr show dev "$fabric_iface" | awk '{sub(/\/.*/, "", $4); print $4; exit}')"

export PYTHONPATH="$repo_root/hs_connectors/src:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_SOCKET_IFNAME="$fabric_iface"
export GLOO_SOCKET_IFNAME="$fabric_iface"
export MOONCAKE_LOCAL_HOSTNAME="$fabric_addr"
export NCCL_MNNVL_ENABLE=1
export TORCH_DISTRIBUTED_USE_LIBUV=0
export OMP_NUM_THREADS=8

mkdir -p "$run_dir/logs" "$run_dir/checkpoints"

extra_args=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  extra_args+=(--max-steps "$MAX_STEPS")
fi

echo "[$(hostname -s)] train endpoint=$vllm_endpoint iface=$fabric_iface"
cd "$repo_root"
exec torchrun --standalone --nproc-per-node 4 \
  scripts/train.py \
  --verifier-name-or-path "$model_path" \
  --trust-remote-code \
  --draft-config "$script_dir/k3_draft_layer_config.json" \
  --data-path "$data_dir" \
  --save-path "$run_dir/checkpoints" \
  --draft-vocab-size 163840 \
  --mask-token-id 163837 \
  --epochs 1 \
  --checkpoint-freq 0.1 \
  --total-seq-len 8192 \
  --train-data-ratio 0.999 \
  --speculator-type dspark \
  --target-layer-ids 24 48 72 88 92 \
  --block-size 8 \
  --max-anchors 1024 \
  --dflash-decay-gamma 4.0 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --confidence-head-alpha 1.0 \
  --loss-fn '{"ce":0.1,"tv":0.9}' \
  --optimizer muon \
  --lr 1e-4 \
  --scheduler-type cosine \
  --scheduler-warmup-ratio 0.03 \
  --hidden-states-backend mooncake \
  --mooncake-master "$mooncake_master" \
  --mooncake-metadata-server P2PHANDSHAKE \
  --mooncake-protocol tcp \
  --mooncake-global-segment-gib 0 \
  --mooncake-local-buffer-gib 4 \
  --mooncake-writer-threads 4 \
  --vllm-endpoint "$vllm_endpoint" \
  --on-missing generate \
  --on-generate delete \
  --request-timeout 900 \
  --max-retries 5 \
  --generation-validation-retries 2 \
  --max-consecutive-generation-failures 20 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --log-freq 20 \
  --log-dir "$run_dir/logs" \
  --run-name "$run_name" \
  "${extra_args[@]}"
