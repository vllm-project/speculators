#!/usr/bin/env bash
#
# Wires the three components together on a 3-node allocation:
#   nodes 0-1  Kimi-K3 TP8 extractor (node 0 also runs the Mooncake master)
#   node 2     DSpark draft training, 4 DDP ranks
#
# Submit with:  sbatch examples/train/kimi_k3_dspark/run.sbatch

#SBATCH --job-name=k3-dspark
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
data_dir="${DATA_DIR:-$repo_root/runs/kimi_k3_dspark/data}"
run_name="${RUN_NAME:-kimi-k3-dspark-$SLURM_JOB_ID}"
run_dir="$repo_root/runs/kimi_k3_dspark/$run_name"

if ! compgen -G "$data_dir/*.arrow" >/dev/null || [[ ! -f "$data_dir/d2t.npy" ]]; then
  echo "No prepared data in $data_dir; run prepare_data.sbatch first." >&2
  exit 1
fi
mkdir -p "$run_dir/logs"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
extract_nodes="${nodes[0]},${nodes[1]}"
train_node="${nodes[2]}"

# The extractor head's address on the fast fabric: vLLM's rendezvous point and
# the host for both the Mooncake master and the HTTP endpoint.
fabric_subnet="${FABRIC_SUBNET:?FABRIC_SUBNET is required}"
extract_addr="$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" \
  bash -c "ip -o -4 addr show | awk -v s='^$fabric_subnet' '\$4 ~ s {sub(/\/.*/, \"\", \$4); print \$4; exit}'")"
export EXTRACT_ADDR="$extract_addr"
export FABRIC_SUBNET="$fabric_subnet"
export MOONCAKE_MASTER="$extract_addr:50051"
export VLLM_ENDPOINT="http://$extract_addr:8000/v1"
export DATA_DIR="$data_dir"
export RUN_NAME="$run_name"
export RUN_DIR="$run_dir"

echo "Run:        $run_name"
echo "Extraction: $extract_nodes (TP8, $VLLM_ENDPOINT)"
echo "Training:   $train_node (4 DDP ranks)"
echo "Mooncake:   $MOONCAKE_MASTER"

background_pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${background_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" \
  --output="$run_dir/logs/mooncake-master.log" \
  mooncake_master --rpc_port 50051 --metrics_port 9003 \
  --rpc_thread_num 8 --enable_disk_eviction=false --logtostderr=true &
background_pids+=($!)

srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --nodelist="$extract_nodes" --gpus-per-node=4 \
  --output="$run_dir/logs/extractor-%t.log" \
  bash "$script_dir/launch_extractor_node.sh" &
extractor_pid=$!
background_pids+=("$extractor_pid")

deadline=$((SECONDS + 3600))
until curl --silent --fail "http://$extract_addr:8000/health" >/dev/null 2>&1; do
  if ! kill -0 "$extractor_pid" 2>/dev/null; then
    echo "Extractor exited before becoming healthy; see $run_dir/logs/extractor-*.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for the extractor to report healthy." >&2
    exit 1
  fi
  sleep 5
done
echo "Extractor is healthy. Starting training."

srun --overlap --nodes=1 --ntasks=1 --nodelist="$train_node" --gpus-per-node=4 \
  --output="$run_dir/logs/trainer.log" \
  bash "$script_dir/launch_train_node.sh"

echo "Training completed: $run_dir/checkpoints"
