#!/bin/bash
# Loop: wait for each new training checkpoint, pause training, run a
# HumanEval throughput eval (default sampling temperature) against it via
# evaluate.py, then resume training. Runs forever until killed.
#
# Usage: eval_each_checkpoint.sh <last_seen_mtime> <config_file> <run_dir>
set -uo pipefail

cd /home/MeganEFlynn/speculators/examples/train/lilicorr_qwen38_27b_full
export PYTHONPATH=/home/MeganEFlynn/speculators/src

last_seen="$1"
CONFIG_FILE="${2:-full.yaml}"
RUN_DIR="${3:-/tmp/lilicorr_check/qwen38-perfectblend/runfull}"
CKPT_DIR="$RUN_DIR/checkpoints/0"
EVAL_BASE="$RUN_DIR/humaneval_each_checkpoint"
mkdir -p "$EVAL_BASE"

MM="/home/MeganEFlynn/.local/bin/micromamba run -n speculators"

echo "=== eval loop starting, config=$CONFIG_FILE run_dir=$RUN_DIR last_seen=$last_seen ==="

# Kill every GPU compute process we own (vLLM/torch workers get renamed via
# setproctitle to things like "VLLM::Worker", which plain `pkill -f` against
# the original launch command does NOT match -- this caused a real failure
# on 2026-10-27 where orphaned workers held ~74 GiB on GPU 0, so the next
# eval server's engine init failed with "Free memory ... less than desired
# GPU memory utilization"), then poll until the given GPU indices actually
# report near-zero memory used, instead of a blind fixed sleep.
wait_for_gpus_clear() {
  local gpu_indices="$1"
  for attempt in $(seq 1 20); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | tr -d ' ' | while read -r pid; do
        [ -z "$pid" ] && continue
        owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        [ "$owner" = "$(whoami)" ] && kill -9 "$pid" 2>/dev/null
      done
    sleep 3
    total_used=0
    for idx in $gpu_indices; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
      total_used=$((total_used + ${used:-0}))
    done
    if [ "$total_used" -lt 1000 ]; then
      echo "--- GPUs $gpu_indices clear (attempt $attempt) ---"
      return 0
    fi
    echo "--- GPUs $gpu_indices still at ${total_used} MiB, retrying teardown (attempt $attempt) ---"
  done
  echo "!!! GPUs $gpu_indices never cleared after 20 attempts !!!"
  return 1
}

while true; do
  # training_state.json is written LAST in the checkpoint-save sequence (it's
  # the resume "commit" marker) -- watching it instead of model.safetensors
  # avoids a race where the loop detects a new model.safetensors mtime while
  # training_state.json/scheduler_state_dict.pt are still from the PREVIOUS
  # checkpoint, which happened on 2026-10-27 at step ~16347 when pkill landed
  # mid-save: the loop reported "step=10898" (stale) even though model
  # weights were already from the new step, risking resume with a mismatched
  # step counter / dataloader skip position.
  current_mtime=$(stat -c %Y "$CKPT_DIR/training_state.json" 2>/dev/null)
  if [ -z "$current_mtime" ] || [ "$current_mtime" == "$last_seen" ]; then
    sleep 20
    continue
  fi

  step=$(python3 -c "import json;print(json.load(open('$CKPT_DIR/training_state.json'))['global_step'])" 2>/dev/null)
  echo "=== $(date): new checkpoint detected, step=$step mtime=$current_mtime ==="
  last_seen="$current_mtime"

  echo "--- pausing training ---"
  pkill -9 -f "speculators.train" 2>/dev/null
  pkill -9 -f "launch_vllm.py" 2>/dev/null
  pkill -9 -f "vllm.entrypoints.cli.main serve" 2>/dev/null

  EVAL_OUT="$EVAL_BASE/step_${step}"
  mkdir -p "$EVAL_OUT"
  healthy=0

  if wait_for_gpus_clear "0 1 2 3 4 5 6 7"; then
    echo "--- launching eval server for step $step ---"
    # --max-model-len must comfortably exceed evaluate.py's hardcoded
    # max_tokens=4096 (throughput mode) + prompt length. 2048 made every
    # HumanEval request 400 instantly (found 2026-10-27); --gen-kwargs does
    # NOT fix this since evaluate.py's internal max_tokens goes into the
    # backend's top-level field, not the extras.body it merges from
    # --gen-kwargs -- confirmed by direct reproduction. 8192 is the real fix.
    CUDA_VISIBLE_DEVICES=0 $MM python -m vllm.entrypoints.openai.api_server \
        --model /tmp/models/Qwen3.8-27B --dtype bfloat16 \
        --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.90 \
        --port 8002 --per-request-spec-decode-metrics summary \
        --attention-config '{"flash_attn_version": 2}' --enforce-eager \
        --speculative-config "{\"method\":\"dflash\",\"model\":\"$CKPT_DIR\",\"num_speculative_tokens\":15,\"draft_sample_method\":\"greedy\"}" \
        > "$EVAL_OUT/serve_eval.log" 2>&1 &

    for _ in $(seq 1 120); do
      if curl -sf http://localhost:8002/health >/dev/null 2>&1; then
        healthy=1
        break
      fi
      sleep 5
    done
  else
    echo "!!! skipping eval for step $step -- GPUs never cleared !!!"
  fi

  if [ "$healthy" -eq 1 ]; then
    echo "--- running evaluate.py (HumanEval, default temperature) ---"
    $MM python ../../../scripts/evaluate/evaluate.py \
        --target http://localhost:8002/v1 throughput \
        --subsets HumanEval \
        --output-dir "$EVAL_OUT" \
        > "$EVAL_OUT/evaluate.log" 2>&1
    echo "--- eval done for step $step: $EVAL_OUT ---"
  else
    echo "!!! eval server never became healthy for step $step, see $EVAL_OUT/serve_eval.log !!!"
  fi

  echo "--- stopping eval server ---"
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  wait_for_gpus_clear "0 1"

  echo "--- relaunching extraction server ---"
  CUDA_VISIBLE_DEVICES=0,1 $MM python ../../../scripts/launch_vllm.py /tmp/models/Qwen3.8-27B \
      --target-layer-ids 4 12 20 28 36 44 52 60 \
      --provenance-dir "$RUN_DIR" \
      -- --data-parallel-size 2 --port 8000 \
      --max-model-len 8448 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
      --attention-config '{"flash_attn_version": 2}' \
      --enforce-eager \
      > "$RUN_DIR/vllm.log" 2>&1 &

  until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 3; done
  echo "--- extraction server healthy, resuming trainer ---"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 $MM python -m torch.distributed.run \
      --standalone --nproc_per_node 6 \
      -m speculators.train --config "$CONFIG_FILE" \
      >> "$RUN_DIR/train.log" 2>&1 &

  echo "=== resumed training after step $step eval ==="
done
