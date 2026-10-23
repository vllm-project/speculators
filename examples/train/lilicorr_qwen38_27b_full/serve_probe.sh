#!/bin/bash
# Serve the trained 40-step LiLiCorr smoke draft for Qwen3.8-27B and confirm
# it serves cleanly — validates the 8-tap train->save->serve contract (the
# vLLM translator + LiLiCorrDraftModel have only been exercised at 5 taps)
# before committing to the full run. The draft is barely trained, so AL will
# be ~1; what matters is clean load + generation + spec metrics.
set -euo pipefail

CKPT=${1:?usage: serve_probe_38.sh <checkpoint_dir> [port] [gpu]}
PORT=${2:-8002}
GPU=${3:-0}
LOG=/tmp/lilicorr_check/qwen38-perfectblend/vllm_probe_${PORT}.log
export PYTHONPATH=/workspace/vllm/.claude/worktrees/lilicorr-speculators-config

MODEL=/tmp/models/Qwen3.8-27B

cd /tmp/lilicorr_check/qwen38-perfectblend
# KV check from the extraction server applies: 27B hybrid mamba needs
# 13.48 GiB for one 8192-token request. At util 0.95 one A100 has
# 77.4 - 55.6 (weights) - ~4 (draft) ~= 18 GiB for KV, so cap the model
# len at 2048 (3.4 GiB/seq) and keep max-num-seqs small.
CUDA_VISIBLE_DEVICES=$GPU /workspace/vllm/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --dtype bfloat16 --max-model-len 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.95 --port "$PORT" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$CKPT\",\"num_speculative_tokens\":15,\"draft_sample_method\":\"greedy\"}" \
  > "$LOG" 2>&1 &
PID=$!

cleanup() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap cleanup EXIT

until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "SERVER DIED"; tail -60 "$LOG"; exit 1
  fi
  sleep 2
done
echo "SERVER READY (pid $PID)"

for p in "Explain the theory of relativity:" "The process of photosynthesis converts" "Write a Python function that sorts a list:" "The French Revolution began in" "Water boils at a temperature of" "In quantum mechanics, the uncertainty principle states that" "The largest planet in the solar system is" "A binary search algorithm works by"; do
  curl -s "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$p\",\"max_tokens\":80,\"temperature\":0}" > /dev/null
done
sleep 20
echo "=== acceptance metrics ==="
grep "SpecDecoding metrics" "$LOG" | tail -3
echo "=== errors/warnings in log ==="
grep -E "ERROR|Traceback" "$LOG" | head -10 || true
