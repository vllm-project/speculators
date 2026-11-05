#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=/data/fast/qwen38_synthetic_sweep_20261021_h100x2_no_linear_backend
PROJECT_DIR=/workspace/speculators
EVAL_VENV="$RUN_DIR/eval-venv"
EVAL_PY="$EVAL_VENV/bin/python"
GUIDELLM_BIN="$EVAL_VENV/bin/guidellm"
VLLM_BIN=/workspace/vllm/.venv/bin/vllm
BENCH="$RUN_DIR/throughput_interactivity_timeout.py"
TARGET_MODEL=Qwen/Qwen3.8-27B
DATASET=RedHatAI/speculator_benchmarks
PORT=${VLLM_PORT:-8110}
SERVER_URL="http://127.0.0.1:$PORT"
SHORT_STREAMS=1,2,4,8,16
LONG_STREAMS=32,64,128
REPEATS=3
SHORT_WARMUP=30
SHORT_SECONDS=90
LONG_WARMUP=60
LONG_SECONDS=120
MAX_TOKENS=1024
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=16384
PYTHON_HOOKS="$RUN_DIR/eval-python-hooks"

eval_run() {
  env \
    PYTHONPATH="$PYTHON_HOOKS${PYTHONPATH:+:$PYTHONPATH}" \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=1 \
    "$@"
}

export CUDA_VISIBLE_DEVICES=0,1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export HF_HOME="$RUN_DIR/cache/hf"
export HF_HUB_CACHE="$RUN_DIR/cache/hf/hub"
export HF_DATASETS_CACHE="$RUN_DIR/cache/hf/datasets"
export HF_XET_CACHE="$RUN_DIR/cache/hf/xet"
export TRANSFORMERS_CACHE="$RUN_DIR/cache/hf/hub"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
export VLLM_CACHE_ROOT="$RUN_DIR/cache/vllm"
export TORCH_HOME="$RUN_DIR/cache/torch_home"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/torch"
export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export TMPDIR=/data/fast/q38nbtmp
export PATH="/workspace/vllm/.venv/bin:$EVAL_VENV/bin:$PATH"

mkdir -p "$RUN_DIR"/{logs,provenance,reports,plots,results,smoke,cache/hf/hub,cache/hf/datasets,cache/hf/xet,cache/xdg,cache/vllm,cache/torch,cache/torch_home,cache/torch_extensions,cache/triton,cache/cuda,tmp}
mkdir -p "$TMPDIR"
cd "$PROJECT_DIR" || exit 2

STATUS_FILE="$RUN_DIR/model_status.tsv"
printf 'model_alias\tdrafter\tstartup\tsmoke\tmath_reasoning_short_rc\tmath_reasoning_long_rc\trag_short_rc\trag_long_rc\tvalidation_plots\tfinal_status\n' > "$STATUS_FILE"

if [[ ! -s "$RUN_DIR/provenance/run_config.txt" ]]; then
cat > "$RUN_DIR/provenance/run_config.txt" <<CONFIG
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
target_model=$TARGET_MODEL
dataset=$DATASET
subsets=math_reasoning,rag
repeats=$REPEATS
short_streams=$SHORT_STREAMS
short_warmup_seconds=$SHORT_WARMUP
short_measured_seconds=$SHORT_SECONDS
long_streams=$LONG_STREAMS
long_warmup_seconds=$LONG_WARMUP
long_measured_seconds=$LONG_SECONDS
max_tokens=$MAX_TOKENS
tensor_parallel_size=2
max_model_len=$MAX_MODEL_LEN
max_num_seqs=$MAX_NUM_SEQS
max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS
prefix_caching=disabled
synthetic_acceptance_rates=0.815,0.646,0.511,0.404,0.322,0.259,0.208,0.17
vllm_cli=$VLLM_BIN
vllm_version=$($VLLM_BIN --version 2>&1)
eval_python=$(eval_run "$EVAL_PY" --version 2>&1)
guidellm_version=$(eval_run "$GUIDELLM_BIN" --version 2>&1)
speculators_version=$(/workspace/speculators/.venv/bin/python -c 'import importlib.metadata as m; print(m.version("speculators"))' 2>&1)
speculators_git_sha=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>&1)
vllm_git_sha=$(git -C /workspace/vllm rev-parse HEAD 2>&1)
vllm_git_describe=$(git -C /workspace/vllm describe --tags --always --dirty 2>&1)
CONFIG
else
  printf 'resumed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/provenance/run_config.txt"
fi

VLLM_PID=
stop_server() {
  if [[ -n "$VLLM_PID" ]]; then
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      kill -KILL -- "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=
  fi
}
trap stop_server EXIT
trap 'stop_server; exit 143' TERM
trap 'stop_server; exit 130' INT
trap 'stop_server; exit 129' HUP

start_server() {
  local alias=$1 drafter=$2
  local command_file="$RUN_DIR/provenance/vllm_command_${alias}.txt"
  local server_log="$RUN_DIR/logs/serve_${alias}.log"
  local pid_file="$RUN_DIR/logs/serve_${alias}.pid"
  local waited=0
  local -a cmd=("$VLLM_BIN" serve "$TARGET_MODEL"
    --port "$PORT"
    --spec-model "$drafter" --spec-tokens 8 --spec-method dspark
    --tensor-parallel-size 2
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --seed 1234 --no-enable-prefix-caching
    --speculative-config '{"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.815,0.646,0.511,0.404,0.322,0.259,0.208,0.17]}')

  if curl -fsS --max-time 2 "$SERVER_URL/health" >/dev/null 2>&1; then
    printf 'Refusing to launch %s: a server already answers on %s\n' "$alias" "$SERVER_URL" | tee -a "$RUN_DIR/logs/orchestrator.log"
    return 1
  fi
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'drafter=%s\n' "$drafter"
    printf 'cwd=%s\n' "$PROJECT_DIR"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    printf 'vllm_version=%s\n' "$($VLLM_BIN --version 2>&1)"
    printf 'vllm_git_sha=%s\n' "$(git -C /workspace/vllm rev-parse HEAD)"
  } > "$command_file"
  setsid "${cmd[@]}" > "$server_log" 2>&1 &
  VLLM_PID=$!
  printf '%s\n' "$VLLM_PID" > "$pid_file"
  printf 'launching %s (%s), pid=%s\n' "$alias" "$drafter" "$VLLM_PID" | tee -a "$RUN_DIR/logs/orchestrator.log"

  while (( waited < 1800 )); do
    if curl -fsS --max-time 5 "$SERVER_URL/health" >/dev/null 2>&1; then
      printf 'healthy %s after %ss\n' "$alias" "$waited" | tee -a "$RUN_DIR/logs/orchestrator.log"
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      printf 'server exited for %s; see %s\n' "$alias" "$server_log" | tee -a "$RUN_DIR/logs/orchestrator.log"
      wait "$VLLM_PID" 2>/dev/null || true
      VLLM_PID=
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  printf 'health timeout for %s after %ss; see %s\n' "$alias" "$waited" "$server_log" | tee -a "$RUN_DIR/logs/orchestrator.log"
  return 1
}

smoke_test() {
  local alias=$1
  local smoke_dir="$RUN_DIR/smoke/$alias"
  mkdir -p "$smoke_dir"
  curl -fsS --max-time 20 "$SERVER_URL/v1/models" > "$smoke_dir/models.json" || return 1
  curl -fsS --max-time 120 "$SERVER_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Reply with only: ready"}],"max_tokens":8,"temperature":0}' \
    > "$smoke_dir/completion_1.json" || return 1
  curl -fsS --max-time 120 "$SERVER_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"What is 2 plus 2? Give only the number."}],"max_tokens":8,"temperature":0}' \
    > "$smoke_dir/completion_2.json" || return 1
  eval_run "$EVAL_PY" - "$smoke_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
models = json.loads((root / "models.json").read_text())
assert any(item.get("id") == "Qwen/Qwen3.8-27B" for item in models.get("data", [])), models
for name in ("completion_1.json", "completion_2.json"):
    payload = json.loads((root / name).read_text())
    assert payload.get("choices"), payload
    assert payload["choices"][0].get("message", {}).get("content") is not None, payload
print("/v1/models and both chat completion smoke responses are valid")
PY
}

collect_subset() {
  local alias=$1 subset=$2 streams=$3 warmup=$4 seconds=$5
  local subset_dir="$RUN_DIR/results/${alias}_${subset}"
  local csv_path="$RUN_DIR/results/${alias}_${subset}.csv"
  mkdir -p "$subset_dir"
  eval_run "$EVAL_PY" "$BENCH" collect \
    --target "$SERVER_URL" --model "$TARGET_MODEL" \
    --dataset "$DATASET" --subset "$subset" --max-tokens "$MAX_TOKENS" \
    --repeats "$REPEATS" --out-dir "$subset_dir" --label "$alias" \
    --csv "$csv_path" --streams "$streams" \
    --warmup-seconds "$warmup" --max-seconds "$seconds" \
    --keep-going --guidellm-bin "$GUIDELLM_BIN"
}

models=(
  'bf16_5L|RelaxingSnorlax/Qwen3.8-27B-DSpark-BF16-5L|#b52513'
  'fp8_block_5L|RelaxingSnorlax/Qwen3.8-27B-DSpark-FP8-BLOCK-5L|#8a5a9e'
  'fp8_dynamic_5L|RelaxingSnorlax/Qwen3.8-27B-DSpark-FP8-DYNAMIC-5L|#c64570'
  'bf16_15L|RelaxingSnorlax/Qwen3.8-27B-DSpark-BF16-15L|#2c6e9b'
  'fp8_block_15L|RelaxingSnorlax/Qwen3.8-27B-DSpark-FP8-BLOCK-15L|#d1773c'
  'fp8_dynamic_15L|RelaxingSnorlax/Qwen3.8-27B-DSpark-FP8-DYNAMIC-15L|#577590'
  'bf16_30L|RelaxingSnorlax/Qwen3.8-27B-DSpark-BF16-30L|#7a8c3f'
  'fp8_block_30L|RelaxingSnorlax/Qwen3.8-27B-DSpark-FP8-BLOCK-30L|#3a9d8f'
  'fp8_dynamic_30L|RelaxingSnorlax/Qwen3.8-27B-DSpark-FP8-DYNAMIC-30L|#9c755f'
)

for row in "${models[@]}"; do
  IFS='|' read -r alias drafter color <<< "$row"
  startup=PASS smoke=SKIPPED math_short=SKIPPED math_long=SKIPPED rag_short=SKIPPED rag_long=SKIPPED validation=SKIPPED final_status=PASS

  if ! start_server "$alias" "$drafter"; then
    startup=FAIL final_status=FAIL
    stop_server
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$alias" "$drafter" "$startup" "$smoke" "$math_short" "$math_long" "$rag_short" "$rag_long" "$validation" "$final_status" >> "$STATUS_FILE"
    continue
  fi

  if [[ "$alias" == bf16_5L ]]; then
    if smoke_test "$alias" > "$RUN_DIR/logs/smoke_${alias}.log" 2>&1; then
      smoke=PASS
      printf 'smoke checks passed for %s\n' "$alias" | tee -a "$RUN_DIR/logs/orchestrator.log"
    else
      smoke=FAIL final_status=FAIL
      printf 'smoke checks failed for %s; see %s\n' "$alias" "$RUN_DIR/logs/smoke_${alias}.log" | tee -a "$RUN_DIR/logs/orchestrator.log"
      stop_server
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$alias" "$drafter" "$startup" "$smoke" "$math_short" "$math_long" "$rag_short" "$rag_long" "$validation" "$final_status" >> "$STATUS_FILE"
      continue
    fi
  fi

  for subset in math_reasoning rag; do
    if [[ "$subset" == math_reasoning ]]; then
      if collect_subset "$alias" "$subset" "$SHORT_STREAMS" "$SHORT_WARMUP" "$SHORT_SECONDS" > "$RUN_DIR/logs/${alias}_${subset}_short.log" 2>&1; then math_short=PASS; else math_short=FAIL; final_status=FAIL; fi
      if collect_subset "$alias" "$subset" "$LONG_STREAMS" "$LONG_WARMUP" "$LONG_SECONDS" > "$RUN_DIR/logs/${alias}_${subset}_long.log" 2>&1; then math_long=PASS; else math_long=FAIL; final_status=FAIL; fi
    else
      if collect_subset "$alias" "$subset" "$SHORT_STREAMS" "$SHORT_WARMUP" "$SHORT_SECONDS" > "$RUN_DIR/logs/${alias}_${subset}_short.log" 2>&1; then rag_short=PASS; else rag_short=FAIL; final_status=FAIL; fi
      if collect_subset "$alias" "$subset" "$LONG_STREAMS" "$LONG_WARMUP" "$LONG_SECONDS" > "$RUN_DIR/logs/${alias}_${subset}_long.log" 2>&1; then rag_long=PASS; else rag_long=FAIL; final_status=FAIL; fi
    fi
  done

  stop_server

  model_validation=PASS
  for subset in math_reasoning rag; do
    csv_path="$RUN_DIR/results/${alias}_${subset}.csv"
    validation_path="$RUN_DIR/reports/${alias}_${subset}_validation.txt"
    plot_path="$RUN_DIR/plots/${alias}_${subset}.png"
    if [[ -s "$csv_path" ]]; then
      if eval_run "$EVAL_PY" "$BENCH" validate "$csv_path" > "$validation_path" 2>&1; then
        :
      else
        model_validation=FAIL
        final_status=FAIL
      fi
      if eval_run "$EVAL_PY" "$BENCH" plot \
        --series "$csv_path:${alias}:${color}" \
        --title "Throughput vs. Interactivity: ${alias} / ${subset}" \
        --subtitle "${TARGET_MODEL} · ${DATASET} ${subset} · synthetic acceptance rates · ${REPEATS} repeats" \
        --out "$plot_path" > "$RUN_DIR/logs/${alias}_${subset}_plot.log" 2>&1; then
        :
      else
        model_validation=FAIL
        final_status=FAIL
      fi
    else
      printf 'No non-empty CSV was produced for %s / %s\n' "$alias" "$subset" > "$validation_path"
      model_validation=FAIL
      final_status=FAIL
    fi
  done
  validation=$model_validation
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$alias" "$drafter" "$startup" "$smoke" "$math_short" "$math_long" "$rag_short" "$rag_long" "$validation" "$final_status" >> "$STATUS_FILE"
  printf 'completed %s with status %s\n' "$alias" "$final_status" | tee -a "$RUN_DIR/logs/orchestrator.log"
done

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/provenance/run_config.txt"
printf 'all model evaluations completed\n' | tee -a "$RUN_DIR/logs/orchestrator.log"
