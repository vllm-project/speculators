#!/usr/bin/env bash
# Build a DSpine checkpoint without starting training or a vLLM server.
# Requires the verifier weights and tokenizer locally or through HF access.
# Run from the Speculators repository root after activating its environment:
#   SAVE_PATH=./output/dspine-dry-run bash examples/train/dspine_qwen3_8b_dry_run.sh
# Inspect config.json before using the checkpoint for training or serving.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SAVE_PATH="${SAVE_PATH:-./output/dspine-dry-run}"

python -m speculators.train \
    --dry-run \
    --speculator-type dspine \
    --verifier-name-or-path "$MODEL" \
    --save-path "$SAVE_PATH" \
    --target-layer-ids 1 9 17 25 33 \
    --num-layers 5 \
    --block-size 16 \
    --transfer-dim 1024 \
    --message-dim 512 \
    --transition-top-k 16 \
    --known-anchor-injection

python -c 'import json, sys; from pathlib import Path; c=json.loads((Path(sys.argv[1]) / "config.json").read_text()); assert c["known_anchor_injection"] is True; assert c["block_size"] == 16; assert c["sample_from_anchor"] is False; print("DSpine checkpoint: known anchor, 15 draft proposals")' "$SAVE_PATH"
