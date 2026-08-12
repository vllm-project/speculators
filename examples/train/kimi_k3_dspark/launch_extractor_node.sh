#!/usr/bin/env bash
#
# One Kimi-K3 extraction node. Run this on both nodes of the TP8 pair, with
# NODE_RANK=0 on the head and NODE_RANK=1 on the other.
#
# Required: NODE_RANK, EXTRACT_ADDR (head node's fabric IP), MOONCAKE_MASTER,
#           FABRIC_SUBNET (prefix of the fast-fabric IPv4 subnet)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
model_path="${MODEL_PATH:-moonshotai/Kimi-K3}"
vllm_python="${VLLM_PYTHON:-python}"

node_rank="${NODE_RANK:-${SLURM_PROCID:?NODE_RANK is required (0 on the head node)}}"
extract_addr="${EXTRACT_ADDR:?EXTRACT_ADDR is required (head node fabric IP)}"
mooncake_master="${MOONCAKE_MASTER:?MOONCAKE_MASTER is required}"

# Every rank must bind to the fast fabric, not the management NIC. Different
# ranks picking different interfaces hang at rendezvous with no error.
fabric_subnet="${FABRIC_SUBNET:?FABRIC_SUBNET is required}"
fabric_iface="$(ip -o -4 addr show | awk -v s="^$fabric_subnet" '$4 ~ s {print $2; exit}')"
fabric_addr="$(ip -o -4 addr show dev "$fabric_iface" | awk '{sub(/\/.*/, "", $4); print $4; exit}')"

export PYTHONPATH="$repo_root/hs_connectors/src:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_SOCKET_IFNAME="$fabric_iface"
export GLOO_SOCKET_IFNAME="$fabric_iface"
export MOONCAKE_LOCAL_HOSTNAME="$fabric_addr"
export NCCL_MNNVL_ENABLE=1
export TORCH_DISTRIBUTED_USE_LIBUV=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_RUST_FRONTEND=1

vllm_args=(
  --served-model-name "$model_path"
  --trust-remote-code
  --load-format fastsafetensors
  --moe-backend auto
  --all2all-backend flashinfer_nvlink_one_sided
  --enable-expert-parallel
  --gpu-memory-utilization 0.95
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
  --tensor-parallel-size 8
  --nnodes 2
  --node-rank "$node_rank"
  --master-addr "$extract_addr"
  --port 8000
  --max-model-len 8193
  --max-num-seqs 64
  --max-num-batched-tokens 32768
  --kv-cache-dtype auto
  --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":false}'
  --no-enable-prefix-caching
  --language-model-only
)
if [[ "$node_rank" != "0" ]]; then
  vllm_args+=(--headless)
fi

echo "[$(hostname -s)] extractor rank=$node_rank/2 tp=8 iface=$fabric_iface"
cd "$repo_root"
exec "$vllm_python" scripts/launch_vllm.py "$model_path" \
  --hidden-states-backend mooncake \
  --mooncake-master "$mooncake_master" \
  --mooncake-metadata-server P2PHANDSHAKE \
  --mooncake-protocol tcp \
  --mooncake-global-segment-gib 32 \
  --mooncake-local-buffer-gib 4 \
  --mooncake-writer-threads 4 \
  --target-layer-ids 24 48 72 88 92 \
  --trust-remote-code \
  -- "${vllm_args[@]}"
