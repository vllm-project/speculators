#!/bin/bash
# Online Eagle3 Training Script for Qwen3-VL-4B on hao05/llava-cot-5k-reannotated
#
# Runs the full online training pipeline:
#   1. Download the multimodal Parquet dataset snapshot from Hugging Face
#   2. Materialize local image files and an absolute-path JSONL
#   3. Prepare arrow data with multimodal preprocessing
#   4. Launch a hidden-state extraction vLLM server
#   5. Train Eagle3 with on-the-fly hidden-state generation
#
# Usage:
#   MODEL=<local-HF-snapshot-path-ending-in-MODEL_REVISION> \
#   MODEL_REVISION=<full-40-character-model-commit-SHA> \
#   SOURCE_REVISION=<full-40-character-source-commit-SHA> \
#   DATASET_REVISION=<full-40-character-dataset-commit-SHA> \
#     bash examples/train/eagle3_qwen3_vl_4b_llava_cot_5k_online.sh
#
# Notes:
# - `prepare_data.py` currently accepts local json/jsonl files or built-in dataset
#   aliases. This example snapshots the public HF Parquet dataset locally first.
# - The uploaded dataset stores image bytes in Parquet and preserves original
#   relative paths in `image_path`. This script materializes image files and a
#   JSONL with absolute image paths so vLLM can load images reliably during
#   online training.
# - This script intentionally accepts only the named 5k dataset. Use a separate,
#   explicitly reviewed pipeline for a different dataset or scale.
#
# ### Example E2E run for Qwen3-VL-4B on 5k samples from LLaVA-CoT ###
#
# Note: This 5k setup is primarily a pipeline sanity check. It is enough to
# verify that multimodal online training, hidden-state generation, and
# checkpointing all work end-to-end, but it is not intended to represent final
# model quality.
#
# Timing from an observed run on 3x NVIDIA GeForce RTX 5090 32GB GPUs
# (vLLM on GPUs 0,1 and training on GPU 2):
# Data preprocessing: 990 seconds (16 mins 30 secs)
# vLLM server startup: 378 seconds (6 mins 18 secs)
# Training (5 epochs): 2686 seconds (44 mins 46 secs)
# Total (prepare_data start to checkpoint save): 4054 seconds (67 mins 34 secs)
#
# Final validation metrics from that run:
# val/loss_epoch: 7.803
# val/full_acc_0_epoch: 64.2%
# val/full_acc_1_epoch: 39.2%
# val/full_acc_2_epoch: 23.9%

set -euo pipefail

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
DATASET_REPO="${DATASET_REPO:-hao05/llava-cot-5k-reannotated}"
# A branch, tag, or abbreviated SHA is rejected before any artifact is created.
DATASET_REVISION="${DATASET_REVISION:-}"
# Explicit *_DIR values are exact artifact paths. The *_ROOT values are used
# only to derive defaults when their corresponding exact path is absent.
DATASET_ROOT="${DATASET_ROOT:-./data/llava-cot-5k-reannotated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_qwen3_vl_4b_llava_cot_online}"
ATTEMPT_LABEL="${ATTEMPT_LABEL:-default}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-5120}"
VLLM_TP="${VLLM_TP:-2}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
if [[ -z "${VLLM_EXTRA_ARGS+x}" ]]; then
    VLLM_EXTRA_ARGS="--enforce-eager"
fi

# GPU assignments
VLLM_GPUS="${VLLM_GPUS:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
if [[ -z "${NUM_TRAIN_GPUS:-}" ]]; then
    IFS=',' read -r -a TRAIN_GPU_ARR <<< "$TRAIN_GPUS"
    NUM_TRAIN_GPUS="${#TRAIN_GPU_ARR[@]}"
fi
# =======================================

# Optional mirror for environments without direct access to huggingface.co
# export HF_ENDPOINT=https://hf-mirror.com

PROVENANCE_HELPER="scripts/pipeline_provenance.py"
if [[ "$DATASET_REPO" != "hao05/llava-cot-5k-reannotated" ]]; then
    echo "This 5k example only permits hao05/llava-cot-5k-reannotated." >&2
    exit 2
fi
DATASET_REVISION="$(python "$PROVENANCE_HELPER" validate-revision "$DATASET_REVISION")"
SOURCE_REVISION="$(
    python "$PROVENANCE_HELPER" validate-source-checkout \
        --source-dir . \
        --revision "$SOURCE_REVISION"
)"
MODEL_REVISION="$(
    python "$PROVENANCE_HELPER" validate-revision "$MODEL_REVISION" \
        --field-name MODEL_REVISION
)"
MODEL="$(
    python "$PROVENANCE_HELPER" validate-model-snapshot \
        --model "$MODEL" \
        --revision "$MODEL_REVISION" \
        --expected-profile qwen3-vl-4b-instruct
)"
if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    VLLM_EXTRA_ARGS_PROVENANCE="$VLLM_EXTRA_ARGS"
else
    VLLM_EXTRA_ARGS_PROVENANCE="none"
fi
CONFIG_FIELDS=(
    --field "attempt_label=$ATTEMPT_LABEL"
    --field "dataset_repo=$DATASET_REPO"
    --field "dataset_revision=$DATASET_REVISION"
    --field "epochs=$EPOCHS"
    --field "lr=$LR"
    --field "max_samples=$MAX_SAMPLES"
    --field "model=$MODEL"
    --field "model_revision=$MODEL_REVISION"
    --field "num_train_gpus=$NUM_TRAIN_GPUS"
    --field "seq_length=$SEQ_LENGTH"
    --field "source_revision=$SOURCE_REVISION"
    --field "train_gpus=$TRAIN_GPUS"
    --field "vllm_extra_args=$VLLM_EXTRA_ARGS_PROVENANCE"
    --field "vllm_gpus=$VLLM_GPUS"
    --field "vllm_max_model_len=$VLLM_MAX_MODEL_LEN"
    --field "vllm_tp=$VLLM_TP"
)
ATTEMPT_ID="$(
    python "$PROVENANCE_HELPER" fingerprint "${CONFIG_FIELDS[@]}"
)"

# Every reusable artifact is both revision-scoped and attempt-scoped. Changing
# any provenance field selects a fresh directory without manual cleanup.
DEFAULT_ATTEMPT_DIR="$OUTPUT_ROOT/attempts/$ATTEMPT_ID"
DATASET_DIR="${DATASET_DIR:-$DATASET_ROOT/revisions/$DATASET_REVISION/attempts/$ATTEMPT_ID}"
DATASET_JSONL="$DATASET_DIR/train.absolute_paths.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_ATTEMPT_DIR/prepared}"
PREPARED_DATA_DIR="$OUTPUT_DIR"
DEFAULT_RUNTIME_DIR="${OUTPUT_DIR}.runtime"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$DEFAULT_RUNTIME_DIR/hidden_states_online}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_RUNTIME_DIR/checkpoints}"
DATASET_MANIFEST="${DATASET_DIR}.provenance.json"
PREPARED_MANIFEST="${PREPARED_DATA_DIR}.provenance.json"
RUNTIME_MANIFEST="${CHECKPOINT_DIR}.runtime.provenance.json"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$DEFAULT_RUNTIME_DIR/vllm_server.log}"
VLLM_LOG_DIR="${VLLM_LOG_FILE%/*}"
if [[ -z "$VLLM_LOG_DIR" ]]; then
    VLLM_LOG_DIR="/"
elif [[ "$VLLM_LOG_DIR" == "$VLLM_LOG_FILE" ]]; then
    VLLM_LOG_DIR="."
fi

PROVENANCE_FIELDS=(
    "${CONFIG_FIELDS[@]}"
    --field "checkpoint_dir=$CHECKPOINT_DIR"
    --field "dataset_dir=$DATASET_DIR"
    --field "hidden_states_dir=$HIDDEN_STATES_DIR"
    --field "output_dir=$OUTPUT_DIR"
    --field "vllm_log_file=$VLLM_LOG_FILE"
)

echo "Pipeline attempt: $ATTEMPT_LABEL ($ATTEMPT_ID)"
echo "Dataset artifact: $DATASET_DIR"
echo "Output artifact: $OUTPUT_DIR"

# Serializes the deterministic repair/reuse path. A stale lock fails closed and
# requires explicit operator inspection instead of risking concurrent mutation.
PIPELINE_LOCK_DIR="${OUTPUT_DIR}.pipeline.lock"
PIPELINE_LOCK_PARENT="${PIPELINE_LOCK_DIR%/*}"
if [[ -z "$PIPELINE_LOCK_PARENT" ]]; then
    PIPELINE_LOCK_PARENT="/"
elif [[ "$PIPELINE_LOCK_PARENT" == "$PIPELINE_LOCK_DIR" ]]; then
    PIPELINE_LOCK_PARENT="."
fi
mkdir -p "$PIPELINE_LOCK_PARENT"
if ! mkdir "$PIPELINE_LOCK_DIR" 2>/dev/null; then
    echo "Pipeline attempt is already active or has a stale lock: $PIPELINE_LOCK_DIR" >&2
    exit 1
fi
VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM server..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    rmdir "$PIPELINE_LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Step 1: Downloading dataset snapshot ==="
echo "Dataset snapshot: ${DATASET_REPO}@${DATASET_REVISION}"
DATASET_ACTION="$(
    python "$PROVENANCE_HELPER" claim \
        --manifest "$DATASET_MANIFEST" \
        --artifact-dir "$DATASET_DIR" \
        "${PROVENANCE_FIELDS[@]}"
)"
if [[ "$DATASET_ACTION" == "run" ]]; then
    mkdir -p "$DATASET_DIR"
    hf download "$DATASET_REPO" \
        --repo-type dataset \
        --revision "$DATASET_REVISION" \
        --local-dir "$DATASET_DIR" \
        --include "README.md" \
        --include "data/*.parquet"
else
    echo "Reusing exact completed dataset artifact."
fi
python "$PROVENANCE_HELPER" validate-shards --dataset-dir "$DATASET_DIR"

echo "=== Step 2: Materializing Parquet dataset to absolute-path JSONL ==="
# Re-run on reuse to validate and atomically repair the materialized outputs.
python scripts/materialize_multimodal_parquet.py \
    --dataset-dir "$DATASET_DIR" \
    --max-samples "$MAX_SAMPLES"

if [[ ! -s "$DATASET_JSONL" ]]; then
    echo "Materializer did not produce a non-empty $DATASET_JSONL" >&2
    exit 1
fi
if [[ "$DATASET_ACTION" == "run" ]]; then
    python "$PROVENANCE_HELPER" complete \
        --manifest "$DATASET_MANIFEST" \
        --artifact-dir "$DATASET_DIR" \
        "${PROVENANCE_FIELDS[@]}"
fi

echo "=== Step 3: Preparing multimodal data ==="
PREPARED_ACTION="$(
    python "$PROVENANCE_HELPER" claim \
        --manifest "$PREPARED_MANIFEST" \
        --artifact-dir "$PREPARED_DATA_DIR" \
        "${PROVENANCE_FIELDS[@]}"
)"
if [[ "$PREPARED_ACTION" == "run" ]]; then
    python scripts/prepare_data.py \
        --model "$MODEL" \
        --data "$DATASET_JSONL" \
        --output "$PREPARED_DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --seq-length "$SEQ_LENGTH" \
        --multimodal

else
    echo "Reusing exact completed prepared-data artifact."
fi

shopt -s nullglob
PREPARED_ARROW_FILES=("$PREPARED_DATA_DIR"/*.arrow)
shopt -u nullglob
if (( ${#PREPARED_ARROW_FILES[@]} == 0 )); then
    echo "Prepared artifact contains no Arrow shards: $PREPARED_DATA_DIR" >&2
    exit 1
fi
python "$PROVENANCE_HELPER" validate-prepared \
    --prepared-dir "$PREPARED_DATA_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"
if [[ "$PREPARED_ACTION" == "run" ]]; then
    python "$PROVENANCE_HELPER" complete \
        --manifest "$PREPARED_MANIFEST" \
        --artifact-dir "$PREPARED_DATA_DIR" \
        "${PROVENANCE_FIELDS[@]}"
fi

validate_runtime_outputs() {
    python "$PROVENANCE_HELPER" validate-runtime \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --hidden-states-dir "$HIDDEN_STATES_DIR" \
        --prepared-dir "$PREPARED_DATA_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --epochs "$EPOCHS"
}

remove_checkpoint_convenience_symlinks() {
    local checkpoint_link link_name
    local checkpoint_links=()
    shopt -s nullglob
    checkpoint_links=(
        "$CHECKPOINT_DIR"/checkpoint_best
        "$CHECKPOINT_DIR"/epoch*_end
        "$CHECKPOINT_DIR"/epoch*_step*
    )
    shopt -u nullglob
    for checkpoint_link in "${checkpoint_links[@]}"; do
        link_name="${checkpoint_link##*/}"
        if [[ ! -L "$checkpoint_link" ]]; then
            echo "Expected checkpoint convenience link is not a symlink: $link_name" >&2
            return 1
        fi
        unlink "$checkpoint_link"
    done
}

if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "Refusing to launch: port $VLLM_PORT already has a healthy server." >&2
    exit 1
fi

RUNTIME_ACTION="$(
    python "$PROVENANCE_HELPER" claim \
        --manifest "$RUNTIME_MANIFEST" \
        --artifact-dir "$CHECKPOINT_DIR" \
        --guard-dir "$HIDDEN_STATES_DIR" \
        "${PROVENANCE_FIELDS[@]}"
)"
if [[ "$RUNTIME_ACTION" == "reuse" ]]; then
    validate_runtime_outputs
    echo "This exact runtime attempt is already complete."
    echo "Use a new ATTEMPT_LABEL to run the same configuration again."
    echo "Checkpoints: $CHECKPOINT_DIR"
    exit 0
fi

mkdir -p "$CHECKPOINT_DIR" "$HIDDEN_STATES_DIR" "$VLLM_LOG_DIR"
if [[ -z "${VLLM_ALLOWED_LOCAL_MEDIA_PATH:-}" ]]; then
    VLLM_ALLOWED_LOCAL_MEDIA_PATH="$(cd "$DATASET_DIR" && pwd)"
fi
read -r -a VLLM_EXTRA_ARR <<< "$VLLM_EXTRA_ARGS"

echo "=== Step 4: Launching vLLM server ==="
echo "vLLM logs will be written to: $VLLM_LOG_FILE"

CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image":1}' \
    --allowed-local-media-path "$VLLM_ALLOWED_LOCAL_MEDIA_PATH" \
    "${VLLM_EXTRA_ARR[@]}" \
    > "$VLLM_LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
VLLM_START_TIME="$(date +%s)"
while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server exited before becoming healthy. See $VLLM_LOG_FILE" >&2
        wait "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi

    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        break
    fi

    VLLM_ELAPSED="$(( $(date +%s) - VLLM_START_TIME ))"
    if (( VLLM_ELAPSED >= VLLM_STARTUP_TIMEOUT )); then
        echo "Timed out after ${VLLM_STARTUP_TIMEOUT}s waiting for vLLM health." >&2
        echo "See $VLLM_LOG_FILE for startup logs." >&2
        exit 1
    fi

    sleep 2
done
echo "vLLM server ready."

echo "=== Step 5: Online training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$PREPARED_DATA_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --num-layers 1 \
    --ttt-steps 3 \
    --ttt-step-loss-decay 1.0 \
    --on-missing generate \
    --on-generate cache \
    --run-name eagle3_qwen3_vl_4b_llava_cot_5k_online

remove_checkpoint_convenience_symlinks
validate_runtime_outputs
python "$PROVENANCE_HELPER" complete \
    --manifest "$RUNTIME_MANIFEST" \
    --artifact-dir "$CHECKPOINT_DIR" \
    --guard-dir "$HIDDEN_STATES_DIR" \
    "${PROVENANCE_FIELDS[@]}"

echo "Done. Checkpoints saved to $CHECKPOINT_DIR"
