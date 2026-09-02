#!/usr/bin/env bash
#
# Tokenizes the JSONL into an Arrow dataset and builds the draft vocabulary
# mapping. Run this once, before any distributed job.
#
#   SOURCE_JSONL=/path/to/data.jsonl sbatch --export=ALL,SOURCE_JSONL \
#     examples/train/kimi_k3_dspark/prepare_data.sbatch

#SBATCH --job-name=k3-dspark-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
model_path="${MODEL_PATH:-/mnt/shared/weights/kimi-k3}"
source_jsonl="${SOURCE_JSONL:?SOURCE_JSONL is required}"
data_dir="${DATA_DIR:-$repo_root/runs/kimi_k3_dspark/data}"

export PYTHONPATH="$repo_root/hs_connectors/src:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

cd "$repo_root"
python scripts/prepare_data.py \
  --model "$model_path" \
  --trust-remote-code \
  --data "$source_jsonl" \
  --output "$data_dir" \
  --seq-length 8192 \
  --num-preprocessing-workers 48 \
  --minimum-valid-tokens 16
