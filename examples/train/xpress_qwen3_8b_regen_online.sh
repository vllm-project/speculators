#!/usr/bin/env bash
# XPress on Qwen3-8B, one node / 8 GPUs: vLLM verifier on GPU 0, 7 trainer ranks on
# GPUs 1-7. Trains on a REGENERATED corpus (on-policy responses from the target).
#
# The validated XPress b16 recipe. Every training flag below was audited rather than
# guessed -- see XPRESS-TRAINING.md before changing any of them.
#
#   bash examples/train/xpress_qwen3_8b_regen_online.sh
#
# Known deviations from the published recipe (see XPRESS-TRAINING.md):
#   * hidden states are produced online by vLLM; data order differs.
#   * no fp32 master weights: the bf16 params are clipped (at the same 1.0) and
#     stepped directly, rather than fp32 copies of them.
#
# EVAL protocol: the SEPARATE eval corpus (--val-data-path), single-conversation batches
# (--no-packing reaches the val loader too), accept length averaged uniformly over
# conversations, block_size-1 Jacobi passes.
set -Eeuo pipefail

PY=${PY:-python}
cd "$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -z "${VLLM_PY:-}" ]]; then
    if $PY -c 'import vllm' 2>/dev/null; then
        VLLM_PY=$PY
    elif [[ -x ./.venv_vllm/bin/python ]] && ./.venv_vllm/bin/python -c 'import vllm' 2>/dev/null; then
        VLLM_PY=$(cd .venv_vllm/bin && pwd)/python
        echo "note: \$PY has no vllm; using ./.venv_vllm/bin/python for the verifier"
    else
        echo "FATAL: no interpreter with vllm found." >&2
        echo "  \$PY ($PY) cannot import vllm, and ./.venv_vllm/bin/python is absent or lacks it." >&2
        echo "  Fix: uv venv .venv_vllm --python 3.12 && VIRTUAL_ENV=.venv_vllm uv pip install 'vllm>=0.22.0'" >&2
        echo "  Or set VLLM_PY=/path/to/python-with-vllm" >&2
        exit 1
    fi
fi

$VLLM_PY -c 'import vllm' 2>/dev/null || {
    echo "FATAL: VLLM_PY ($VLLM_PY) cannot import vllm." >&2
    exit 1
}

$VLLM_PY -c '
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory as F
import sys
if "ExampleHiddenStatesConnector" not in getattr(F, "_registry", {}):
    sys.exit("this vllm build has no ExampleHiddenStatesConnector "
             "(needed for hidden-state extraction); install a newer vllm")
' || { echo "FATAL: see above ($VLLM_PY)" >&2; exit 1; }
$PY -c 'import speculators, hs_connectors' 2>/dev/null || {
    echo "FATAL: \$PY ($PY) cannot import speculators/hs_connectors." >&2
    echo "  Fix: uv pip install -e ./hs_connectors -e .   (hs_connectors FIRST)" >&2
    exit 1
}

export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}

_want_tmp=${TMPDIR:-/dev/shm/spec-tmp}
mkdir -p "$_want_tmp" 2>/dev/null || true
_exec_ok=0
if [[ -d "$_want_tmp" ]]; then
    _probe="$_want_tmp/.execprobe.$$"
    printf '#!/bin/sh\nexit 0\n' > "$_probe" 2>/dev/null && chmod +x "$_probe" 2>/dev/null \
        && "$_probe" 2>/dev/null && _exec_ok=1
    rm -f "$_probe"
fi
if [[ "$_exec_ok" != 1 ]]; then
    _fallback=${XDG_CACHE_HOME:-$HOME/.cache}/spec-tmp
    echo "note: $_want_tmp is not exec-capable (noexec mount?); using $_fallback instead" >&2
    echo "      -- torch.compile dlopens .so files it builds under TMPDIR" >&2
    _want_tmp=$_fallback
    mkdir -p "$_want_tmp"
fi
export TMPDIR=$_want_tmp
# Pin the compile caches explicitly so they follow TMPDIR even if something else
# repoints TMPDIR later; these are the dirs that actually need exec permission.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TMPDIR/triton}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# wandb: export WANDB_API_KEY yourself (never hardcode it) and set the project.
export WANDB_PROJECT=${WANDB_PROJECT:-ripple-dspark}

# Self-diagnosis for a multi-week run. A wedged rank previously showed up only as a
# frozen progress bar with no way to see WHERE it was wedged (py-spy needs CAP_SYS_PTRACE,
# which containers usually withhold). With these, a stuck collective aborts with every
# rank's stack plus an NCCL flight-recorder dump instead of hanging silently forever.
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}

MODEL="Qwen/Qwen3-8B"
OUT_ROOT=${OUT_ROOT:-/root/speculators_out}
BACKBONE=$OUT_ROOT/dflash_b16_zlab_converted
DATA_DIR=${DATA_DIR:-/root/DeepSpec/data}
TRAIN_JSONL=${TRAIN_JSONL:-$DATA_DIR/qwen3_8B_refiner_train_nothink.jsonl}
EVAL_JSONL=${EVAL_JSONL:-$DATA_DIR/qwen3_8B_refiner_eval_nothink.jsonl}
OUTPUT_DIR=$OUT_ROOT/xpress_b16_zlab_8gpu
TRAIN_DATA=$OUTPUT_DIR/train
EVAL_DATA=$OUTPUT_DIR/eval
RUN_NAME=${RUN_NAME:-xpress-b16-speculators-8gpu}
VLLM_PORT=${VLLM_PORT:-8300}
RDZV_PORT=${RDZV_PORT:-29610}
SEQ_LENGTH=4096
TARGET_LAYER_IDS="1 9 17 25 33"
MAX_SAMPLES=${MAX_SAMPLES:-1311126}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-}
LOG_FREQ=${LOG_FREQ:-50}
mkdir -p "$OUT_ROOT" "$OUTPUT_DIR"

if [[ -t 1 ]]; then
    echo "WARNING: stdout is a TTY. rich/tqdm and vLLM per-request logs share one" >&2
    echo "  console lock; if the terminal (ssh/tmux) cannot drain them fast enough" >&2
    echo "  the writer blocks while holding it and rank 0 stalls -- the other ranks" >&2
    echo "  then spin in NCCL forever. A TTY run also dies with your ssh session." >&2
    echo "  Prefer:  nohup bash $0 > run.log 2>&1 &" >&2
    sleep 5
fi

echo "=== Step 0: convert z-lab b16 -> speculators format (idempotent) ==="
if [ ! -f "$BACKBONE/config.json" ]; then
    $PY - <<PYEOF
from speculators.convert.entrypoints import convert_model
convert_model(
    model="z-lab/Qwen3-8B-DFlash-b16",
    verifier="$MODEL",
    algorithm="dflash",
    output_path="$BACKBONE",
    aux_hidden_state_layer_ids=[1, 9, 17, 25, 33],
)
PYEOF
fi

$PY examples/train/xpress_morph_config.py "$BACKBONE" --rank 256 --mlp-ratio 2

echo "=== Step 1: prepare data (one-time; marker records WHAT was prepared) ==="
prepare_split () {
    local stamp="$2/.data_ready" want
    want="$1|${3:-all}|$SEQ_LENGTH|$(stat -c %s "$1")"
    if [[ -f "$stamp" && "$(cat "$stamp")" == "$want" ]]; then
        echo "  [skip] $2 already prepared for $want"
        return
    fi
    echo "  [prep] $2  <-  $1  (max-samples=${3:-all})"
    rm -f "$stamp"
    $PY scripts/prepare_data.py \
        --model "$MODEL" --data "$1" --output "$2" \
        --seq-length "$SEQ_LENGTH" --overwrite \
        ${3:+--max-samples "$3"}
    printf '%s' "$want" > "$stamp"
}
prepare_split "$TRAIN_JSONL" "$TRAIN_DATA" "$MAX_SAMPLES"
prepare_split "$EVAL_JSONL"  "$EVAL_DATA"  ""

rm -rf /tmp/hidden_states

echo "=== Step 2: vLLM verifier on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 $VLLM_PY scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --port "$VLLM_PORT" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
    curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup"; exit 1; }
    sleep 5
done
curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null || { echo "FATAL: vLLM not ready"; exit 1; }
echo "vLLM ready"

echo "=== Step 3: 7 trainer ranks on GPUs 1-7 ==="
CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -m torch.distributed.run \
    --nnodes 1 --node_rank 0 \
    --rdzv_id xpressb16 --rdzv_backend c10d --rdzv_endpoint "127.0.0.1:${RDZV_PORT}" \
    --nproc_per_node 7 \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --from-pretrained "$BACKBONE" \
    --data-path "$TRAIN_DATA" \
    --val-data-path "$EVAL_DATA" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs 10 \
    --lr 6e-4 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --optimizer adamw \
    --weight-decay 0.0 \
    --checkpoint-freq 0.02 \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type xpress \
    --max-anchors 400 \
    --target-layer-ids $TARGET_LAYER_IDS \
    --xpress-rank 256 \
    --consistency-weight 0.3 \
    --consistency-passes 3 \
    --base-anchor-weight 0.6 \
    --base-anchor-floor 0.2 \
    --decayed-loss-norm \
    --no-packing \
    --eval-interval "$EVAL_INTERVAL" \
    ${EVAL_MAX_BATCHES:+--eval-max-batches "$EVAL_MAX_BATCHES"} \
    --log-freq "$LOG_FREQ" \
    --ce-from-data \
    --loss-fn '{"ce": 0.1, "tv": 1.8}' \
    --on-missing generate \
    --on-generate delete \
    --logger wandb \
    --run-name "$RUN_NAME"
