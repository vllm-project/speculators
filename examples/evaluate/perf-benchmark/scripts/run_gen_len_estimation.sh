#!/usr/bin/env bash
# Run guidellm in throughput mode to estimate output token length distributions.

set -euo pipefail

# ==============================================================================
# Configuration Variables
# ==============================================================================

TARGET=""
DATASET=""
SUBSET=""
OUTPUT_FILE=""
RATE=""
MAX_CONCURRENCY=""
GEN_KWARGS=""
DATA_COLUMN_MAPPER=""

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 --target URL --dataset DATASET --subset SUBSET [OPTIONS]

Required:
  --target URL               vLLM server endpoint
  --dataset DATASET          HF dataset ID or local path
  --subset SUBSET            Dataset subset name (e.g. HumanEval)

Optional:
  --output-file FILE         Output JSON path (default: gen_len_SUBSET.json)
  --rate N                   Request rate (default: 128)
  --max-concurrency N        Max concurrent requests (default: 128)
  --gen-kwargs JSON          Flat JSON with generation kwargs, e.g.
                             '{"temperature":0.6, "top_p":0.95}'
  --data-column-mapper JSON  Column mapping for guidellm
                             (default: '{"text_column":"prompt"}')
  -h, --help                 Show this help message
EOF
}

build_backend_args() {
    local gen_kwargs="${1:-}"

    if [[ -z "${gen_kwargs}" ]]; then
        echo ""
        return
    fi

    python -c "
import json, sys
body = json.loads(sys.argv[1])
print(json.dumps({'extras': {'body': body}}))
" "${gen_kwargs}"
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --rate)
            RATE="$2"
            shift 2
            ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --gen-kwargs)
            GEN_KWARGS="$2"
            shift 2
            ;;
        --data-column-mapper)
            DATA_COLUMN_MAPPER="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

# ==============================================================================
# Apply Defaults & Validate
# ==============================================================================

RATE="${RATE:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
OUTPUT_FILE="${OUTPUT_FILE:-gen_len_${SUBSET}.json}"
DATA_COLUMN_MAPPER="${DATA_COLUMN_MAPPER:-{\"text_column\":\"prompt\"}}"

if [[ -z "${TARGET}" ]]; then
    echo "[ERROR] --target is required" >&2
    exit 1
fi
if [[ -z "${DATASET}" ]]; then
    echo "[ERROR] --dataset is required" >&2
    exit 1
fi
if [[ -z "${SUBSET}" ]]; then
    echo "[ERROR] --subset is required" >&2
    exit 1
fi

# ==============================================================================
# Build backend-args from gen-kwargs
# ==============================================================================

BACKEND_ARGS=$(build_backend_args "${GEN_KWARGS}")

# ==============================================================================
# Run GuideLLM
# ==============================================================================

echo "[INFO] Gen-len estimation: subset=${SUBSET}, rate=${RATE}"
echo "[INFO]   Target: ${TARGET}"
echo "[INFO]   Output: ${OUTPUT_FILE}"
[[ -n "${GEN_KWARGS}" ]] && echo "[INFO]   Gen kwargs: ${GEN_KWARGS}"

GUIDELLM_CMD=(
    guidellm benchmark
    --target "${TARGET}"
    --data "${DATASET}"
    --data-args "{\"data_files\": \"${SUBSET}.jsonl\"}"
    --data-column-mapper "${DATA_COLUMN_MAPPER}"
    --profile throughput
    --rate "${RATE}"
    --output-path "${OUTPUT_FILE}"
)

[[ -n "${BACKEND_ARGS}" ]] && GUIDELLM_CMD+=(--backend-args "${BACKEND_ARGS}")

GUIDELLM__MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
    "${GUIDELLM_CMD[@]}"

echo "[INFO] Gen-len estimation complete: ${OUTPUT_FILE}"
