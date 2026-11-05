#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=/data/fast/nm35_rag_10arm_20261023T142227Z
PROV="$RUN_DIR/provenance"
RESULTS="$RUN_DIR/results"
LOGS="$RUN_DIR/logs"
PLOTS="$RUN_DIR/plots"
SPEC_REPO=/workspace/speculators
VLLM=/workspace/vllm/.venv/bin/vllm
VLLM_PYTHON=/workspace/vllm/.venv/bin/python
EVAL_PY="$RUN_DIR/eval-venv/bin/python"
GUIDELLM="$RUN_DIR/eval-venv/bin/guidellm"
BENCH="$SPEC_REPO/scripts/evaluate/throughput_interactivity.py"
PORT=8110
SERVER_URL="http://127.0.0.1:$PORT"
TARGET_ID=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16
DATASET_DIR="$RUN_DIR/data"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export HF_HOME=/tmp/nm35_hf_home
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR=/data/fast/nm35tmp
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR" "$RESULTS" "$LOGS" "$PLOTS"
printf '%s\n' "$$" > "$PROV/runner.pid"

arms=(
    bf16_5l
    bf16_15l
    bf16_30l
    fp8_block_5l
    fp8_block_15l
    fp8_block_30l
    fp8_dynamic_5l
    fp8_dynamic_15l
    fp8_dynamic_30l
)
plot_arms=(
    bf16_5l
    target_only
    bf16_15l
    bf16_30l
    fp8_block_5l
    fp8_block_15l
    fp8_block_30l
    fp8_dynamic_5l
    fp8_dynamic_15l
    fp8_dynamic_30l
)
plot_colors=(
    '#b52513' '#7a8c3f' '#d66c3d' '#a94442' '#2c6e9b'
    '#5c8eaf' '#173b5f' '#7a4c99' '#a47db7' '#49305c'
)
server_pid=""
current_arm="setup"
completed_arms=1
failed_arms=()
RUN_STATUS=running
printf '%s\n' "$RUN_STATUS" > "$PROV/runner.status"

model_row() {
    awk -F '\t' -v alias="$1" '$1 == alias { value=$2 "\t" $3 } END { print value }' "$PROV/models.tsv"
}

model_path() {
    awk -F '\t' -v alias="$1" '$1 == alias { value=$4 } END { print value }' "$PROV/model_paths.tsv"
}

ensure_model() {
    local alias=$1 row repo revision path
    row=$(model_row "$alias")
    IFS=$'\t' read -r repo revision <<< "$row"
    path=$(model_path "$alias")
    if [[ -n "$path" && -d "$path" ]]; then
        printf '%s\n' "$path"
        return 0
    fi
    "$VLLM_PYTHON" "$PROV/download_model.py" "$alias" "$repo" "$revision" >&2
    path=$(model_path "$alias")
    if [[ -z "$path" || ! -d "$path" ]]; then
        echo "Failed to download snapshot for $alias" >&2
        return 1
    fi
    printf '%s\n' "$path"
}

write_progress() {
    local measured_json measured_sidecars now
    now=$(date -u +%FT%TZ)
    measured_json=$(find "$RESULTS" -type f -name '*.json' ! -name '*.metrics.json' | wc -l)
    measured_sidecars=$(find "$RESULTS" -type f -name '*.metrics.json' | wc -l)
    {
        echo '# Live progress'
        echo
        echo "Updated: $now"
        echo "Run ID: nm35_rag_10arm_20261023T142227Z"
        echo "Runner PID: $(cat "$PROV/runner.pid" 2>/dev/null || echo pending)"
        echo "Runner status: $RUN_STATUS"
        echo "Current arm: $current_arm"
        echo "Completed arms: $completed_arms/10"
        echo "Primary JSON results: $measured_json/240"
        echo "Metrics sidecars: $measured_sidecars/240"
        echo "Failures recorded: ${failed_arms[*]:-none}"
        echo "Smoke status: $(cat "$PROV/quantized-smoke.status" 2>/dev/null || echo pending)"
        echo
        echo 'Target-only is complete. All nine drafter arms use the model-card RAG T=0 synthetic acceptance vector; the original real BF16 5L run is archived.'
    } > "$RUN_DIR/live-progress.md.tmp"
    mv "$RUN_DIR/live-progress.md.tmp" "$RUN_DIR/live-progress.md"
}

stop_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -- "-$server_pid" 2>/dev/null || kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}

cleanup() {
    stop_server
    if [[ "$RUN_STATUS" == running ]]; then
        RUN_STATUS=interrupted
        write_progress
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

start_server() {
    local alias=$1 target_path=$2 drafter_path=${3:-} mode=${4:-real}
    local spec_config=""
    local -a cmd

    if [[ "$alias" != target_only ]]; then
        if [[ "$mode" == real ]]; then
            spec_config="{\"method\":\"dspark\",\"model\":\"${drafter_path}\",\"num_speculative_tokens\":8,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"greedy\"}"
        else
            spec_config=$("$VLLM_PYTHON" - "$drafter_path" "$PROV/bf16_5l_acceptance_profile.json" <<'PY'
import json
import sys

draft = sys.argv[1]
payload = json.load(open(sys.argv[2]))
rates = payload["profiles"][0]["acceptance_at_position"]
if len(rates) != 8 or any(not 0 <= rate <= 1 for rate in rates):
    raise SystemExit(f"Expected eight valid acceptance rates; got {rates}")
print(json.dumps({
    "method": "dspark",
    "model": draft,
    "num_speculative_tokens": 8,
    "draft_tensor_parallel_size": 1,
    "draft_sample_method": "greedy",
    "rejection_sample_method": "synthetic",
    "synthetic_acceptance_rates": rates,
}, separators=(",", ":")))
PY
)
        fi
    fi

    cmd=("$VLLM" serve "$target_path"
        --served-model-name "$TARGET_ID"
        --port "$PORT"
        --trust-remote-code
        --limit-mm-per-prompt '{"video":1,"image":1}'
        --video-pruning-rate 0.5
        --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'
        --tensor-parallel-size 4
        --gpu-memory-utilization 0.89
        --prefix-match-unit 64
        --block-size 64
        --compilation-config '{"max_cudagraph_capture_size":192}'
        --enable-expert-parallel
        --max-num-seqs 128
        --max-model-len 114688
        --max-num-batched-tokens 16384
        --no-enable-prefix-caching
        --async-scheduling
        --mamba-backend flashinfer
        --mamba-ssm-cache-dtype float16
        --enable-mamba-cache-stochastic-rounding
        --mamba-cache-philox-rounds 5
        --chat-template-content-format string
        --mamba-cache-mode align
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
        --reasoning-parser nemotron_v3)
    if [[ "$alias" != target_only ]]; then
        cmd+=(--speculative-config "$spec_config")
    fi

    {
        printf 'export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=%q\n' "$VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
        printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$PROV/serve_${alias}.command.txt"

    if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
        echo "Port $PORT already has a healthy server; refusing duplicate launch" >&2
        return 1
    fi
    setsid "${cmd[@]}" > "$LOGS/serve_${alias}.log" 2>&1 &
    server_pid=$!
    local waited=0
    while ! curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -100 "$LOGS/serve_${alias}.log" >&2
            return 1
        fi
        if (( waited >= 3600 )); then
            echo "vLLM for $alias did not become healthy within 3600 seconds" >&2
            tail -100 "$LOGS/serve_${alias}.log" >&2
            return 1
        fi
        sleep 10
        waited=$((waited + 10))
        if (( waited > 0 && waited % 60 == 0 )); then
            write_progress
        fi
    done
    curl -fsS "$SERVER_URL/v1/models" > "$PROV/models_${alias}.json"
    echo "vLLM healthy for $alias after ${waited}s"
}

run_collect() {
    local alias=$1 out_dir="$RESULTS/${1}_rag" csv="$RESULTS/${1}_rag.csv"
    local streams warmup seconds name rc=0 collect_pid
    local -a common
    common=("$EVAL_PY" "$BENCH" collect
        --target "$SERVER_URL"
        --model "$TARGET_ID"
        --tokenizer "$(model_path target)"
        --dataset "$DATASET_DIR"
        --subset rag
        --max-tokens 1024
        --repeats 3
        --out-dir "$out_dir"
        --label "$alias"
        --csv "$csv"
        --keep-going
        --guidellm-bin "$GUIDELLM")

    for profile in short long; do
        if [[ "$profile" == short ]]; then
            streams=1,2,4,8,16
            warmup=30
            seconds=90
        else
            streams=32,64,128
            warmup=60
            seconds=120
        fi
        name="$LOGS/collect_${alias}_${profile}.log"
        "${common[@]}" --streams "$streams" --warmup-seconds "$warmup" --max-seconds "$seconds" > "$name" 2>&1 &
        collect_pid=$!
        while kill -0 "$collect_pid" 2>/dev/null; do
            sleep 30
            if kill -0 "$collect_pid" 2>/dev/null; then
                write_progress
            fi
        done
        if wait "$collect_pid"; then
            echo "Collected $alias $profile profile"
        else
            echo "Collection failed for $alias $profile; see $name" >&2
            rc=1
        fi
    done
    return "$rc"
}

main() {
    if [[ ! -s "$DATASET_DIR/rag.jsonl" ]]; then
        echo "Pinned local rag.jsonl is missing from $DATASET_DIR" >&2
        exit 1
    fi
    if [[ ! -f "$PROV/quantized-smoke.status" ]]; then
        echo "Quantized drafter smoke has not passed; refusing to begin evaluations" >&2
        exit 1
    fi
    if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
        echo "Port $PORT is occupied before benchmark start; refusing duplicate launch" >&2
        exit 1
    fi

    local target_path alias drafter_path collect_rc index=0
    local target_json_count target_sidecar_count
    target_json_count=$(find "$RESULTS/target_only_rag" -maxdepth 1 -type f -name '*.json' ! -name '*.metrics.json' | wc -l)
    target_sidecar_count=$(find "$RESULTS/target_only_rag" -maxdepth 1 -type f -name '*.metrics.json' | wc -l)
    if [[ "$target_json_count" -ne 24 || "$target_sidecar_count" -ne 24 || ! -s "$RESULTS/target_only_rag.csv" ]]; then
        echo "Validated target-only baseline is missing or incomplete" >&2
        exit 1
    fi
    if ! "$EVAL_PY" "$BENCH" validate "$RESULTS/target_only_rag.csv" > "$LOGS/validate_target_only_final.log" 2>&1; then
        echo "Target-only baseline validation failed" >&2
        exit 1
    fi

    local target_path alias drafter_path collect_rc
    if ! target_path=$(ensure_model target); then
        RUN_STATUS=failed_target_download
        write_progress
        exit 1
    fi
    write_progress
    for alias in "${arms[@]}"; do
        current_arm=$alias
        write_progress
        echo "=== Starting arm $alias ==="
        if ! drafter_path=$(ensure_model "$alias"); then
            failed_arms+=("$alias:model_download")
            write_progress
            continue
        fi
        if ! start_server "$alias" "$target_path" "$drafter_path" synthetic; then
            failed_arms+=("$alias:server_start")
            RUN_STATUS=failed
            write_progress
            if [[ "$alias" == bf16_5l ]]; then
                exit 1
            fi
            continue
        fi

        collect_rc=0
        run_collect "$alias" || collect_rc=$?
        stop_server
        if (( collect_rc != 0 )); then
            failed_arms+=("$alias:collect")
        fi

        if [[ -s "$RESULTS/${alias}_rag.csv" ]]; then
            if ! "$EVAL_PY" "$BENCH" validate "$RESULTS/${alias}_rag.csv" > "$LOGS/validate_${alias}.log" 2>&1; then
                failed_arms+=("$alias:validate")
            fi
        else
            failed_arms+=("$alias:missing_csv")
        fi
        completed_arms=$((completed_arms + 1))
        index=$((index + 1))
        write_progress
        echo "=== Completed arm $alias ($completed_arms/10) ==="
    done

    local -a series
    for index in "${!plot_arms[@]}"; do
        alias=${plot_arms[$index]}
        series+=(--series "$RESULTS/${alias}_rag.csv:$alias:${plot_colors[$index]}")
    done
    if ! "$EVAL_PY" "$BENCH" plot "${series[@]}" \
        --title 'Nemotron 3.5 DSpark throughput vs interactivity' \
        --subtitle 'RedHatAI/speculator_benchmarks rag · 3 repetitions · prefix caching disabled' \
        --out "$PLOTS/rag_10arm.png" > "$LOGS/plot_model_card_synthetic.log" 2>&1; then
        failed_arms+=("plot")
    fi

    local total_json total_sidecars
    total_json=$(find "$RESULTS" -type f -name '*.json' ! -name '*.metrics.json' | wc -l)
    total_sidecars=$(find "$RESULTS" -type f -name '*.metrics.json' | wc -l)
    if (( ${#failed_arms[@]} == 0 )) && [[ "$total_json" -eq 240 && "$total_sidecars" -eq 240 ]]; then
        RUN_STATUS=complete
    else
        RUN_STATUS=complete_with_failures
        if [[ "$total_json" -ne 240 || "$total_sidecars" -ne 240 ]]; then
            failed_arms+=("count:${total_json}json_${total_sidecars}sidecars")
        fi
    fi
    current_arm=finished
    write_progress
    printf '%s\n' "$RUN_STATUS" > "$PROV/runner.status"
    printf '%s\n' "${failed_arms[@]}" > "$PROV/failed-items.txt"
    "$VLLM_PYTHON" - "$RUN_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
commands = sorted((root / "provenance").glob("serve_*.command.txt"))
bad = [p.name for p in commands if "--linear-backend" in p.read_text()]
if bad:
    raise SystemExit(f"Unexpected --linear-backend in: {bad}")
print(f"Verified no --linear-backend in {len(commands)} saved vLLM launch records")
PY
    echo "Sweep finished: $RUN_STATUS; $total_json JSON results, $total_sidecars metrics sidecars"
}

main "$@"
