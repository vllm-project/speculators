#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${1:?usage: run_matrix.sh RUN_DIR}"
PORT=8110
VLLM=/workspace/vllm/.venv/bin/vllm
PY=/workspace/speculators/.venv/bin/python
BENCH=/workspace/speculators/scripts/evaluate/throughput_interactivity.py
GUIDELLM="$RUN/eval-venv/bin/guidellm"
TARGET=zai-org/GLM-5.3-Flash
REV=46a050bec13002ad29ee795c16e61cf4fa3c4a9c
ACC='[0.801,0.615,0.463,0.335,0.237,0.164,0.112,0.074]'
MODEL=zai-org/GLM-5.3-Flash
SUBSET=rag

export HF_HUB_CACHE=/root/.cache/huggingface
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export OMP_NUM_THREADS=1

ARMS=(
  bf16_5L no_speculator fp8_dynamic_5L fp8_block_5L
  bf16_10L fp8_dynamic_10L fp8_block_10L
  bf16_15L fp8_dynamic_15L fp8_block_15L
  bf16_30L fp8_dynamic_30L fp8_block_30L
)

drafter_for() {
  case "$1" in
    bf16_5L)        echo "RedHatAI/GLM-5.3-Flash-speculator.dspark-preview" ;;
    fp8_dynamic_5L) echo "/workspace/quantized/GLM-5.3-Flash-speculator.dspark-preview-FP8-dynamic" ;;
    fp8_block_5L)   echo "/workspace/quantized/GLM-5.3-Flash-speculator.dspark-preview-FP8-block" ;;
    bf16_10L)       echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-BF16-10L" ;;
    fp8_dynamic_10L) echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-FP8-dynamic-10L" ;;
    fp8_block_10L)  echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-FP8-block-10L" ;;
    bf16_15L)       echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-BF16-15L" ;;
    fp8_dynamic_15L) echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-FP8-dynamic-15L" ;;
    fp8_block_15L)  echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-FP8-block-15L" ;;
    bf16_30L)       echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-BF16-30L" ;;
    fp8_dynamic_30L) echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-FP8-dynamic-30L" ;;
    fp8_block_30L)  echo "/root/glm53-layer-sweep/GLM-5.3-Flash-speculator.dspark-preview-FP8-block-30L" ;;
    *) echo "" ;;
  esac
}

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
now() { date -u +%s; }

gpu_free() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{bad=1} END{exit bad?1:0}'
}

stop_server() {
  local label="$1" pid="$2"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -KILL -- -"$pid" 2>/dev/null || true
  fi
  pkill -TERM -f "vllm serve.*--port $PORT" 2>/dev/null || true
  for _ in $(seq 1 90); do gpu_free && return 0; sleep 2; done
  log "WARN: GPU memory not released after $label"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  return 1
}

current_pid=""
current_arm=""
trap 'if [ -n "$current_pid" ]; then stop_server trap "$current_pid" || true; fi' EXIT

write_serve_script() {
  local arm="$1" drafter="$2"
  local specfile="$RUN/commands/$arm.spec.json"
  if [ "$arm" = "no_speculator" ]; then
    cat > "$RUN/commands/serve_$arm.sh" <<EOF
#!/usr/bin/env bash
export HF_HUB_CACHE=/root/.cache/huggingface
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export OMP_NUM_THREADS=1
exec $VLLM serve $TARGET \\
  --revision $REV \\
  --served-model-name $TARGET \\
  --host 127.0.0.1 --port $PORT \\
  --tensor-parallel-size 8 \\
  --max-model-len 16384 \\
  --kv-cache-dtype fp8 \\
  --gpu-memory-utilization 0.9 \\
  --max-num-seqs 128 \\
  --max-num-batched-tokens 8192 \\
  --no-enable-prefix-caching \\
  --no-enable-flashinfer-autotune
EOF
  else
    python3 - "$specfile" "$drafter" "$ACC" <<'PY'
import json, sys
path, drafter, acc = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
spec = {"method": "dspark", "model": drafter, "num_speculative_tokens": 8,
        "draft_tensor_parallel_size": 1, "draft_sample_method": "greedy",
        "rejection_sample_method": "synthetic", "synthetic_acceptance_rates": acc}
open(path, "w").write(json.dumps(spec, indent=1) + "\n")
PY
    local spec
    spec=$(cat "$specfile")
    cat > "$RUN/commands/serve_$arm.sh" <<EOF
#!/usr/bin/env bash
export HF_HUB_CACHE=/root/.cache/huggingface
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export OMP_NUM_THREADS=1
exec $VLLM serve $TARGET \\
  --revision $REV \\
  --served-model-name $TARGET \\
  --host 127.0.0.1 --port $PORT \\
  --tensor-parallel-size 8 \\
  --max-model-len 16384 \\
  --kv-cache-dtype fp8 \\
  --gpu-memory-utilization 0.9 \\
  --max-num-seqs 128 \\
  --max-num-batched-tokens 8192 \\
  --no-enable-prefix-caching \\
  --no-enable-flashinfer-autotune \\
  --speculative-config '$spec'
EOF
  fi
  chmod +x "$RUN/commands/serve_$arm.sh"
}

set_status() {
  local arm="$1" phase="$2"
  mkdir -p "$RUN/results/$arm"
  local points
  points=$(find "$RUN/results/$arm/$SUBSET" -maxdepth 1 -name 'conc*_r*.json' ! -name '*.metrics.json' 2>/dev/null | wc -l || true)
  printf '%s | %s | %s/24 | %s\n' "$arm" "$phase" "$points" "$(date -u +%FT%TZ)" > "$RUN/results/$arm/status.txt"
  {
    echo "# Live progress"
    echo
    echo "Updated: $(date -u +%FT%TZ)"
    echo
    echo "| arm | phase | points | updated |"
    echo "|---|---|---|---|"
    for a in "${ARMS[@]}"; do
      if [ -f "$RUN/results/$a/status.txt" ]; then
        local s; s=$(cat "$RUN/results/$a/status.txt")
        echo "| $a | $(echo "$s" | cut -d'|' -f2) | $(echo "$s" | cut -d'|' -f3) | $(echo "$s" | cut -d'|' -f4) |"
      else
        echo "| $a | pending | 0/24 | |"
      fi
    done
    echo
    echo "Run dir: \`$RUN\`"
  } > "$RUN/live-progress.md"
}

start_server() {
  local arm="$1"
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "FAIL: port $PORT busy before $arm"; return 1; fi
  if ! gpu_free; then log "FAIL: GPU not free before $arm"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; return 1; fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$RUN/provenance/$arm.gpu.before.txt"
  setsid bash "$RUN/commands/serve_$arm.sh" > "$RUN/logs/$arm.server.log" 2>&1 < /dev/null &
  local pid=$!
  current_pid=$pid
  echo "$pid" > "$RUN/logs/$arm.server.pid"
  log "server pid=$pid"
  local ready=0
  for i in $(seq 1 180); do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; log "healthy after $((i*10))s"; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "FAIL: server exited during startup for $arm"; tail -80 "$RUN/logs/$arm.server.log"; return 1
    fi
    sleep 10
  done
  if [ "$ready" != 1 ]; then log "FAIL: $arm not healthy after 1800s"; tail -80 "$RUN/logs/$arm.server.log"; return 1; fi
  curl -sS -m 30 "http://127.0.0.1:$PORT/version"   > "$RUN/provenance/$arm.version.json"
  curl -sS -m 30 "http://127.0.0.1:$PORT/v1/models" > "$RUN/provenance/$arm.models.json"
  curl -sS -m 30 "http://127.0.0.1:$PORT/metrics"   > "$RUN/provenance/$arm.metrics.startup.prom"
}

smoke_chat() {
  local arm="$1"
  cat > "$RUN/requests/$arm.smoke.json" <<'EOF'
{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"Reply with one sentence about speculative decoding."}],"max_tokens":64,"temperature":0}
EOF
  local status
  status=$(curl -sS -m 300 -o "$RUN/responses/$arm.smoke.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d @"$RUN/requests/$arm.smoke.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" || echo 000)
  echo "$status" > "$RUN/responses/$arm.status.txt"
  log "smoke chat status=$status"
  [ "$status" = "200" ]
}

collect_group() {
  local arm="$1" group="$2" streams="$3" warmup="$4" maxsecs="$5"
  local out="$RUN/results/$arm/$SUBSET"
  local csv="$RUN/results/$arm/$SUBSET.csv"
  local args=(
    "$PY" "$BENCH" collect
    --target "http://127.0.0.1:$PORT" --model "$MODEL"
    --dataset RedHatAI/speculator_benchmarks --subset "$SUBSET"
    --max-tokens 1024 --repeats 3
    --streams "$streams" --warmup-seconds "$warmup" --max-seconds "$maxsecs"
    --out-dir "$out" --label "$arm" --csv "$csv"
    --guidellm-bin "$GUIDELLM" --keep-going
    --extra-body temperature=0
  )
  log "dry-run $arm/$group"
  if ! "${args[@]}" --dry-run > "$RUN/provenance/$arm.collect.$group.dryrun.log" 2>&1; then
    log "FAIL: dry-run $arm/$group"; tail -40 "$RUN/provenance/$arm.collect.$group.dryrun.log"; return 1
  fi
  if ! grep -q 'temperature' "$RUN/provenance/$arm.collect.$group.dryrun.log"; then
    log "FAIL: dry-run for $arm/$group does not show temperature=0"; return 1
  fi
  log "collect $arm/$group"
  local t0; t0=$(now)
  set +e
  "${args[@]}" > "$RUN/provenance/$arm.collect.$group.log" 2>&1
  local rc=$?
  set -e
  local t1; t1=$(now)
  echo "$group rc=$rc seconds=$((t1-t0))" >> "$RUN/provenance/$arm.timings.txt"
  log "collect $arm/$group rc=$rc seconds=$((t1-t0))"
  return 0
}

run_arm() {
  local arm="$1"
  local drafter; drafter=$(drafter_for "$arm")
  log "================ ARM $arm drafter=${drafter:-none}"
  mkdir -p "$RUN/results/$arm"
  if [ -f "$RUN/results/$arm/COMPLETE" ]; then log "skip $arm (COMPLETE)"; return 0; fi

  set_status "$arm" "serving"
  write_serve_script "$arm" "$drafter"
  if ! start_server "$arm"; then set_status "$arm" "serve_failed"; echo "serve_failed" > "$RUN/results/$arm/STATUS"; return 2; fi
  set_status "$arm" "smoke"
  if ! smoke_chat "$arm"; then
    log "FAIL: smoke chat $arm"
    stop_server "$arm" "$current_pid"; current_pid=""
    set_status "$arm" "smoke_failed"; return 1
  fi

  set_status "$arm" "short"
  collect_group "$arm" short "1,2,4,8,16" 30 90 || true
  if ! kill -0 "$current_pid" 2>/dev/null; then log "FAIL: server died during short $arm"; set_status "$arm" "server_died_short"; return 1; fi

  set_status "$arm" "long"
  collect_group "$arm" long "32,64,128" 60 120 || true
  if ! kill -0 "$current_pid" 2>/dev/null; then log "FAIL: server died during long $arm"; set_status "$arm" "server_died_long"; return 1; fi

  set_status "$arm" "parsing"
  "$PY" "$BENCH" parse "$RUN/results/$arm/$SUBSET" --label "$arm" --csv "$RUN/results/$arm/$SUBSET.csv" > "$RUN/provenance/$arm.parse.log" 2>&1 || true
  "$PY" "$BENCH" validate "$RUN/results/$arm/$SUBSET.csv" > "$RUN/provenance/$arm.validate.log" 2>&1 || true

  local points
  points=$(find "$RUN/results/$arm/$SUBSET" -maxdepth 1 -name 'conc*_r*.json' ! -name '*.metrics.json' 2>/dev/null | wc -l || true)
  local sidecars
  sidecars=$(find "$RUN/results/$arm/$SUBSET" -maxdepth 1 -name 'conc*_r*.metrics.json' 2>/dev/null | wc -l || true)
  echo "$points $sidecars" > "$RUN/results/$arm/points.txt"

  stop_server "$arm" "$current_pid"; current_pid=""
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$RUN/provenance/$arm.gpu.after.txt"
  if ! gpu_free; then log "FAIL: GPU not free after $arm"; set_status "$arm" "gpu_not_released"; return 1; fi

  if [ "$points" = "24" ] && [ "$sidecars" = "24" ]; then
    touch "$RUN/results/$arm/COMPLETE"
    set_status "$arm" "complete"
    log "ARM $arm COMPLETE points=$points sidecars=$sidecars"
  else
    set_status "$arm" "incomplete_${points}pts"
    log "ARM $arm INCOMPLETE points=$points sidecars=$sidecars"
  fi
}

main() {
  if [ ! -x "$GUIDELLM" ]; then log "FAIL: $GUIDELLM missing"; exit 1; fi
  local t_start; t_start=$(now)
  local n=0
  for arm in "${ARMS[@]}"; do
    n=$((n+1))
    set_status "$arm" "queued"
    local rc=0
    run_arm "$arm" || rc=$?
    if [ "$rc" = "2" ]; then
      log "STOP: serve failure on $arm; not continuing"
      echo "$arm" > "$RUN/STOP_SERVE_FAILURE"
      exit 2
    fi
    local elapsed=$(( $(now) - t_start ))
    local done_count
    done_count=$(find "$RUN/results" -maxdepth 2 -name COMPLETE | wc -l)
    log "progress $done_count/${#ARMS[@]} arms, elapsed=$((elapsed/60))m"
    if [ "$done_count" -ge 1 ] && [ "$done_count" -lt "${#ARMS[@]}" ]; then
      local per=$((elapsed / done_count))
      log "ETA rough: $(( (per * (${#ARMS[@]} - done_count)) / 60 ))m remaining at ${per}s/arm"
    fi
  done
  log "MATRIX FINISHED"
}

main
