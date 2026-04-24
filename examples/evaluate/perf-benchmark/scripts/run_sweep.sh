#!/usr/bin/env bash
# Run guidellm sweep benchmark with a max_tokens cap.

set -euo pipefail

# ==============================================================================
# Configuration Variables
# ==============================================================================

TARGET=""
DATASET=""
SUBSET=""
OUTPUT_FILE=""
MAX_TOKENS=""
MAX_REQUESTS=""
MAX_CONCURRENCY=""
GEN_KWARGS=""
DATA_COLUMN_MAPPER=""

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 --target URL --dataset DATASET --subset SUBSET --max-tokens N [OPTIONS]

Required:
  --target URL               vLLM server endpoint
  --dataset DATASET          HF dataset ID or local path
  --subset SUBSET            Dataset subset name (e.g. HumanEval)
  --max-tokens N             Max output tokens (from gen-len estimation)

Optional:
  --output-file FILE         Output JSON path (default: sweep_SUBSET.json)
  --max-requests N           Max requests per sweep point (default: 200)
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
    local max_tokens="$2"

    python -c "
import json, sys

gen_kwargs = sys.argv[1]
max_tokens = int(sys.argv[2])

body = json.loads(gen_kwargs) if gen_kwargs else {}
body['max_tokens'] = max_tokens
print(json.dumps({'extras': {'body': body}}))
" "${gen_kwargs}" "${max_tokens}"
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
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS="$2"
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

MAX_REQUESTS="${MAX_REQUESTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
OUTPUT_FILE="${OUTPUT_FILE:-sweep_${SUBSET}.json}"
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
if [[ -z "${MAX_TOKENS}" ]]; then
    echo "[ERROR] --max-tokens is required" >&2
    exit 1
fi

# ==============================================================================
# Build backend-args from gen-kwargs + max_tokens
# ==============================================================================

BACKEND_ARGS=$(build_backend_args "${GEN_KWARGS}" "${MAX_TOKENS}")

# ==============================================================================
# Run GuideLLM Sweep
# ==============================================================================

echo "[INFO] Sweep: subset=${SUBSET}, max_tokens=${MAX_TOKENS}, max_requests=${MAX_REQUESTS}"
echo "[INFO]   Target: ${TARGET}"
echo "[INFO]   Output: ${OUTPUT_FILE}"
[[ -n "${GEN_KWARGS}" ]] && echo "[INFO]   Gen kwargs: ${GEN_KWARGS}"

GUIDELLM__MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
guidellm benchmark \
    --target "${TARGET}" \
    --data "${DATASET}" \
    --data-args "{\"data_files\": \"${SUBSET}.jsonl\"}" \
    --data-column-mapper "${DATA_COLUMN_MAPPER}" \
    --profile sweep \
    --max-requests "${MAX_REQUESTS}" \
    --output-path "${OUTPUT_FILE}" \
    --backend-args "${BACKEND_ARGS}"

echo "[INFO] Sweep complete: ${OUTPUT_FILE}"
