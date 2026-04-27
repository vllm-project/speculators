#!/usr/bin/env bash
# =============================================================================
# deepseek_benchmark.sh — fault-tolerant DeepSeek-V4-Flash perf benchmark
#
# Does everything in one invocation:
#   1. Starts vLLM server in background (skips if already running)
#   2. Waits until the health endpoint responds
#   3. Runs gen-len estimation per subset   (resumes if partially done)
#   4. Computes per-subset max_tokens caps
#   5. Runs guidellm sweep per subset       (fresh server per subset for clean
#      Prometheus counters; retries on failure)
#   6. Computes per-subset acceptance rates from the after-snapshots
#   7. Writes a consolidated CSV
#
# Usage:
#   bash deepseek_benchmark.sh [OPTIONS]
#
# Options (all also accept env-var overrides of the same name in UPPER_SNAKE):
#   --output-dir DIR        default: perf_results_TIMESTAMP
#   --subsets  A,B,...      default: all 9 benchmark subsets
#   --max-requests N        default: 50  (per sweep strategy point)
#   --max-concurrency N     default: 128
#   --gen-len-rate N        default: 128
#   --max-retries N         default: 3   (per subset)
#   --gen-kwargs JSON       e.g. '{"temperature":0.6}'
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_START=$(date +%s)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_ts()      { date '+%Y-%m-%d %H:%M:%S'; }
info()     { echo "[$(_ts)] [INFO]  $*"; }
warn()     { echo "[$(_ts)] [WARN]  $*"; }
error()    { echo "[$(_ts)] [ERROR] $*" >&2; }

elapsed() {
    local secs=$(( $(date +%s) - SCRIPT_START ))
    printf "%dh %02dm %02ds" $(( secs / 3600 )) $(( (secs % 3600) / 60 )) $(( secs % 60 ))
}

elapsed_since() {
    local start="$1"
    local secs=$(( $(date +%s) - start ))
    printf "%dh %02dm %02ds" $(( secs / 3600 )) $(( (secs % 3600) / 60 )) $(( secs % 60 ))
}

banner() {
    local title="$1"
    local width=70
    local line
    line=$(printf '═%.0s' $(seq 1 $width))
    echo ""
    echo "[$(_ts)] ${line}"
    echo "[$(_ts)]   ${title}"
    echo "[$(_ts)] ${line}"
}

sub_banner() {
    local title="$1"
    local width=70
    local line
    line=$(printf '─%.0s' $(seq 1 $width))
    echo ""
    echo "[$(_ts)] ${line}"
    echo "[$(_ts)]   ${title}"
    echo "[$(_ts)] ${line}"
}

# ---------------------------------------------------------------------------
# Defaults (overridable via env or CLI)
# ---------------------------------------------------------------------------
HARDWARE="${HARDWARE:-b200}"   # b200 (4×B200) | h100 (8×H100)
DATASET="${DATASET:-RedHatAI/speculator_benchmarks}"
SUBSETS="${SUBSETS:-HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing}"
OUTPUT_DIR="${OUTPUT_DIR:-perf_results_$(date +%Y%m%d_%H%M%S)}"
MAX_REQUESTS="${MAX_REQUESTS:-50}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
GEN_LEN_RATE="${GEN_LEN_RATE:-128}"
MAX_RETRIES="${MAX_RETRIES:-3}"
GEN_KWARGS="${GEN_KWARGS:-}"
SERVER_LOG="${SCRIPT_DIR}/server-logs.log"
SERVER_STARTUP_TIMEOUT=1800  # 30 min — CUDA graph compilation can take ~20 min

# PORT is discovered dynamically in start_server(); TARGET/METRICS_BASE follow.
PORT=8000
TARGET="http://localhost:${PORT}/v1"
METRICS_BASE="http://localhost:${PORT}"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --hardware)        HARDWARE="$2";        shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --subsets)         SUBSETS="$2";         shift 2 ;;
        --max-requests)    MAX_REQUESTS="$2";    shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
        --gen-len-rate)    GEN_LEN_RATE="$2";    shift 2 ;;
        --max-retries)     MAX_RETRIES="$2";     shift 2 ;;
        --gen-kwargs)      GEN_KWARGS="$2";      shift 2 ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------
case "${HARDWARE}" in
    b200)
        CHG_GPUS=4
        DOCKER_IMAGE="vllm/vllm-openai:deepseekv4-cu130"
        SPEC_TOKENS=10
        HW_EXTRA_ARGS=(--attention_config.use_fp4_indexer_cache=True)
        ;;
    h100)
        CHG_GPUS=8
        DOCKER_IMAGE="vllm/vllm-openai:deepseekv4-cu129"
        SPEC_TOKENS=1
        HW_EXTRA_ARGS=()
        ;;
    *)
        error "Unknown --hardware '${HARDWARE}'. Valid values: b200, h100"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Directories & subset list
# ---------------------------------------------------------------------------
GEN_LEN_DIR="${OUTPUT_DIR}/gen_len"
SWEEP_DIR="${OUTPUT_DIR}/sweeps"
ACCEPTANCE_DIR="${OUTPUT_DIR}/acceptance"
MAX_TOKENS_FILE="${OUTPUT_DIR}/max_tokens.json"
ACCEPTANCE_RATES_FILE="${OUTPUT_DIR}/acceptance_rates.json"
CSV_FILE="${OUTPUT_DIR}/perf_results.csv"

mkdir -p "${GEN_LEN_DIR}" "${SWEEP_DIR}" "${ACCEPTANCE_DIR}"

IFS=',' read -ra SUBSET_ARRAY <<< "${SUBSETS}"
N_SUBSETS=${#SUBSET_ARRAY[@]}

banner "DeepSeek-V4-Flash benchmark  |  $(date '+%Y-%m-%d %H:%M:%S')"
info "Hardware    : ${HARDWARE}  (${CHG_GPUS} GPUs, image: ${DOCKER_IMAGE}, spec_tokens: ${SPEC_TOKENS})"
info "Output dir  : ${OUTPUT_DIR}"
info "Subsets     : ${SUBSETS}"
info "Max requests: ${MAX_REQUESTS}  |  Max concurrency: ${MAX_CONCURRENCY}"
info "Gen-len rate: ${GEN_LEN_RATE}  |  Max retries: ${MAX_RETRIES}"
[[ -n "${GEN_KWARGS}" ]] && info "Gen kwargs  : ${GEN_KWARGS}"

# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------
port_in_use() {
    local p="$1"
    # ss is preferred; fall back to lsof on systems where ss is absent
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${p} "
    else
        lsof -i ":${p}" &>/dev/null
    fi
}

find_free_port() {
    local p="${1:-8000}"
    while port_in_use "${p}"; do
        info "Port ${p} is in use — trying $(( p + 1 ))..."
        p=$(( p + 1 ))
    done
    echo "${p}"
}

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
SERVER_BGP_PID=""
SCRIPT_STARTED_SERVER=false

wait_for_gpus() {
    local needed="${1:-${CHG_GPUS}}"
    info "Waiting for ${needed} AVAILABLE GPU(s) (polling every 30s)..."
    while true; do
        local available
        available=$(chg status 2>&1 \
            | sed 's/\x1b\[[0-9;]*[mGKHF]//g' \
            | grep -E '^\s+[0-9]+\s+\|' \
            | grep 'AVAILABLE' \
            | wc -l)
        if [[ $available -ge $needed ]]; then
            info "GPU check passed: ${available} AVAILABLE GPU(s) found."
            return 0
        fi
        info "Only ${available}/${needed} GPU(s) AVAILABLE — next check in 30s..."
        sleep 30
    done
}

start_server() {
    wait_for_gpus "${CHG_GPUS}"

    # Discover a free host port for this server instance
    PORT=$(find_free_port 8000)
    TARGET="http://localhost:${PORT}/v1"
    METRICS_BASE="http://localhost:${PORT}"

    info "Starting vLLM server [${HARDWARE} / ${CHG_GPUS} GPUs] on port ${PORT}  (logs → ${SERVER_LOG})"
    chg run --gpus "${CHG_GPUS}" -- docker run --gpus all \
        --privileged --ipc=host -p "${PORT}:8000" \
        -v "${HF_HUB_CACHE}:/root/.cache/huggingface" \
        -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
        "${DOCKER_IMAGE}" deepseek-ai/DeepSeek-V4-Flash \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --enable-expert-parallel \
        --data-parallel-size 4 \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
        "${HW_EXTRA_ARGS[@]+"${HW_EXTRA_ARGS[@]}"}" \
        --speculative_config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}" \
        >> "${SERVER_LOG}" 2>&1 &
    SERVER_BGP_PID=$!
    SCRIPT_STARTED_SERVER=true
    info "Server process started  (PID: ${SERVER_BGP_PID}, port: ${PORT})"
}

stop_server() {
    info "Stopping vLLM server (port: ${PORT})..."
    if [[ -n "${SERVER_BGP_PID}" ]] && kill -0 "${SERVER_BGP_PID}" 2>/dev/null; then
        kill "${SERVER_BGP_PID}" 2>/dev/null || true
    fi
    # Also stop the Docker container in case chg run already exited on its own
    local cid
    cid=$(docker ps --filter "publish=${PORT}" --format "{{.ID}}" 2>/dev/null | head -1) || true
    if [[ -n "${cid}" ]]; then
        info "Stopping container ${cid}..."
        docker stop "${cid}" > /dev/null 2>&1 || true
    fi
    SERVER_BGP_PID=""
    # Let the port and GPU memory fully release before the next start
    sleep 10
    info "Server stopped."
}

server_alive() {
    curl -sf "${METRICS_BASE}/health" > /dev/null 2>&1
}

wait_for_server() {
    local start_ts
    start_ts=$(date +%s)
    local deadline=$(( start_ts + SERVER_STARTUP_TIMEOUT ))
    info "Waiting for server to be ready on ${METRICS_BASE} (timeout: ${SERVER_STARTUP_TIMEOUT}s, polling every 120s)..."
    while ! server_alive; do
        if [[ $(date +%s) -ge $deadline ]]; then
            error "Server did not become ready within ${SERVER_STARTUP_TIMEOUT}s."
            exit 1
        fi
        local waited
        waited=$(elapsed_since "${start_ts}")
        info "  Still starting up... (waited ${waited}, next check in 120s)"
        sleep 120
    done
    info "Server is ready!  (startup took $(elapsed_since "${start_ts}"))"
}

ensure_server() {
    if ! server_alive; then
        warn "Server on ${METRICS_BASE} is unreachable — restarting..."
        stop_server
        start_server
        wait_for_server
    fi
}

snapshot() {
    local output="$1"
    python "${SCRIPT_DIR}/scripts/get_acceptance_rate.py" \
        --endpoint "${METRICS_BASE}" \
        --output "${output}"
}

# On exit: stop the server if this script started it
cleanup() {
    echo ""
    info "Script exiting (total elapsed: $(elapsed))"
    if [[ "${SCRIPT_STARTED_SERVER}" == "true" ]]; then
        stop_server
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 0: Start server (or verify existing one is up)
# ---------------------------------------------------------------------------
banner "Step 0 · Server startup"
if server_alive; then
    info "Server already running on ${METRICS_BASE} — skipping start."
else
    start_server
    wait_for_server
fi

# ---------------------------------------------------------------------------
# Step 1: Gen-len estimation
# ---------------------------------------------------------------------------
banner "Step 1 of 5 · Gen-len estimation  [${N_SUBSETS} subsets]"
STEP1_START=$(date +%s)
idx=0
for subset in "${SUBSET_ARRAY[@]}"; do
    idx=$(( idx + 1 ))
    out="${GEN_LEN_DIR}/gen_len_${subset}.json"

    if [[ -f "${out}" ]]; then
        info "[${idx}/${N_SUBSETS}] ${subset} — already done, skipping."
        continue
    fi

    sub_banner "Gen-len ${idx}/${N_SUBSETS}: ${subset}"
    GEN_LEN_ARGS=(
        --target "${TARGET}"
        --dataset "${DATASET}"
        --subset "${subset}"
        --output-file "${out}"
        --rate "${GEN_LEN_RATE}"
        --max-concurrency "${MAX_CONCURRENCY}"
    )
    [[ -n "${GEN_KWARGS}" ]] && GEN_LEN_ARGS+=(--gen-kwargs "${GEN_KWARGS}")

    attempt=0
    while [[ $attempt -lt $MAX_RETRIES ]]; do
        attempt=$(( attempt + 1 ))
        info "  Attempt ${attempt}/${MAX_RETRIES}..."
        ensure_server
        if bash "${SCRIPT_DIR}/scripts/run_gen_len_estimation.sh" "${GEN_LEN_ARGS[@]}"; then
            info "  Gen-len complete: ${subset}"
            break
        fi
        warn "  Gen-len attempt ${attempt}/${MAX_RETRIES} failed for ${subset}."
        rm -f "${out}"
        if [[ $attempt -eq $MAX_RETRIES ]]; then
            error "Gen-len exhausted all retries for: ${subset}"
            exit 1
        fi
    done
done
info "Step 1 complete  (elapsed: $(elapsed_since "${STEP1_START}"))"

# ---------------------------------------------------------------------------
# Step 2: Compute max_tokens per subset
# ---------------------------------------------------------------------------
banner "Step 2 of 5 · Computing max_tokens per subset"
STEP2_START=$(date +%s)

GEN_LEN_FILES=()
for subset in "${SUBSET_ARRAY[@]}"; do
    GEN_LEN_FILES+=("${GEN_LEN_DIR}/gen_len_${subset}.json")
done

python "${SCRIPT_DIR}/scripts/parse_gen_len.py" \
    --output "${MAX_TOKENS_FILE}" \
    "${GEN_LEN_FILES[@]}"

info "max_tokens mapping saved to: ${MAX_TOKENS_FILE}"
info "Step 2 complete  (elapsed: $(elapsed_since "${STEP2_START}"))"

# ---------------------------------------------------------------------------
# Step 3: Sweeps — fresh server restart per subset for clean metrics
# ---------------------------------------------------------------------------
# Each subset gets its own server instance so Prometheus counters start at
# zero.  The "after" snapshot is therefore the exact per-subset rate.
# ---------------------------------------------------------------------------
banner "Step 3 of 5 · Performance sweeps  [${N_SUBSETS} subsets, fresh server each]"
STEP3_START=$(date +%s)
idx=0

for subset in "${SUBSET_ARRAY[@]}"; do
    idx=$(( idx + 1 ))
    sweep_out="${SWEEP_DIR}/sweep_${subset}.json"
    after_snap="${ACCEPTANCE_DIR}/after_${subset}.json"
    max_tokens=$(python -c "import json; print(json.load(open('${MAX_TOKENS_FILE}'))['${subset}'])")

    if [[ -f "${sweep_out}" && -f "${after_snap}" ]]; then
        info "[${idx}/${N_SUBSETS}] ${subset} — already done, skipping."
        continue
    fi

    sub_banner "Sweep ${idx}/${N_SUBSETS}: ${subset}  [max_tokens=${max_tokens}]"

    SWEEP_ARGS=(
        --target "${TARGET}"
        --dataset "${DATASET}"
        --subset "${subset}"
        --output-file "${sweep_out}"
        --max-tokens "${max_tokens}"
        --max-requests "${MAX_REQUESTS}"
        --max-concurrency "${MAX_CONCURRENCY}"
    )
    [[ -n "${GEN_KWARGS}" ]] && SWEEP_ARGS+=(--gen-kwargs "${GEN_KWARGS}")

    success=false
    attempt=0
    while [[ $attempt -lt $MAX_RETRIES ]]; do
        attempt=$(( attempt + 1 ))
        SUBSET_START=$(date +%s)

        sub_banner "  Attempt ${attempt}/${MAX_RETRIES} for ${subset}  |  total elapsed: $(elapsed)"
        info "  Restarting server for clean Prometheus counters..."
        stop_server
        start_server
        wait_for_server

        # SWEEP_ARGS may reference the old TARGET; rebuild with the new port
        SWEEP_ARGS[0]="--target"
        SWEEP_ARGS[1]="${TARGET}"

        rm -f "${sweep_out}"
        info "  Running sweep..."
        if bash "${SCRIPT_DIR}/scripts/run_sweep.sh" "${SWEEP_ARGS[@]}"; then
            info "  Sweep finished — capturing Prometheus snapshot..."
            snapshot "${after_snap}"
            info "  ✓ Subset ${idx}/${N_SUBSETS} (${subset}) complete  (attempt ${attempt}, took $(elapsed_since "${SUBSET_START}"))"
            success=true
            break
        else
            warn "  Sweep failed (attempt ${attempt}/${MAX_RETRIES})."
            rm -f "${sweep_out}"
        fi
    done

    if [[ "${success}" != "true" ]]; then
        error "All ${MAX_RETRIES} retries exhausted for subset: ${subset}"
        exit 1
    fi
done

info "Step 3 complete  (elapsed: $(elapsed_since "${STEP3_START}"))"

# ---------------------------------------------------------------------------
# Step 4: Compute per-subset acceptance rates
# ---------------------------------------------------------------------------
banner "Step 4 of 5 · Computing per-subset acceptance rates"
STEP4_START=$(date +%s)

python "${SCRIPT_DIR}/scripts/compute_acceptance_rates.py" \
    --acceptance-dir "${ACCEPTANCE_DIR}" \
    --subsets "${SUBSETS}" \
    --output "${ACCEPTANCE_RATES_FILE}"

info "Step 4 complete  (elapsed: $(elapsed_since "${STEP4_START}"))"

# ---------------------------------------------------------------------------
# Step 5: Parse sweep results → CSV
# ---------------------------------------------------------------------------
banner "Step 5 of 5 · Extracting sweep metrics to CSV"
STEP5_START=$(date +%s)

SWEEP_FILES=()
for subset in "${SUBSET_ARRAY[@]}"; do
    SWEEP_FILES+=("${SWEEP_DIR}/sweep_${subset}.json")
done

python "${SCRIPT_DIR}/scripts/parse_sweep_results.py" \
    --output "${CSV_FILE}" \
    --acceptance-rates "${ACCEPTANCE_RATES_FILE}" \
    "${SWEEP_FILES[@]}"

info "Step 5 complete  (elapsed: $(elapsed_since "${STEP5_START}"))"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "Benchmark complete!  |  Total time: $(elapsed)"
info "Output directory : ${OUTPUT_DIR}"
info "  gen-len        : ${GEN_LEN_DIR}/"
info "  max_tokens     : ${MAX_TOKENS_FILE}"
info "  sweeps         : ${SWEEP_DIR}/"
info "  acceptance     : ${ACCEPTANCE_DIR}/"
info "  accept rates   : ${ACCEPTANCE_RATES_FILE}"
info "  CSV results    : ${CSV_FILE}"
