#!/usr/bin/env bash
# Run acceptance rate evaluation for a single dataset subset.
#
# Sends requests via guidellm throughput mode and then snapshots the vLLM
# Prometheus metrics.  Assumes the server was freshly started (counters at
# zero) so the snapshot is the per-subset acceptance rate.
#
# Usage:
#   bash eval_acceptance.sh --endpoint http://localhost:8000 \
#       --subset HumanEval --output acceptance/after_HumanEval.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_SCRIPTS="${SCRIPT_DIR}/../scripts"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ENDPOINT=""
DATASET="RedHatAI/speculator_benchmarks"
SUBSET=""
OUTPUT=""
MAX_REQUESTS=200
MAX_CONCURRENCY=128
RATE=128
GEN_KWARGS=""

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint)        ENDPOINT="$2";        shift 2 ;;
        --dataset)         DATASET="$2";         shift 2 ;;
        --subset)          SUBSET="$2";          shift 2 ;;
        --output)          OUTPUT="$2";          shift 2 ;;
        --max-requests)    MAX_REQUESTS="$2";    shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
        --rate)            RATE="$2";            shift 2 ;;
        --gen-kwargs)      GEN_KWARGS="$2";      shift 2 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${ENDPOINT:?--endpoint is required}"
: "${SUBSET:?--subset is required}"
: "${OUTPUT:?--output is required}"

TARGET="${ENDPOINT}/v1"
METRICS_ENDPOINT="${ENDPOINT}"

# ---------------------------------------------------------------------------
# Build backend-args from gen-kwargs (if any)
# ---------------------------------------------------------------------------
BACKEND_ARGS=""
if [[ -n "${GEN_KWARGS}" ]]; then
    BACKEND_ARGS=$(python -c "
import json, sys
body = json.loads(sys.argv[1])
print(json.dumps({'extras': {'body': body}}))
" "${GEN_KWARGS}")
fi

# ---------------------------------------------------------------------------
# Step 1: Send requests via guidellm throughput mode
# ---------------------------------------------------------------------------
echo "[INFO] Sending ${MAX_REQUESTS} requests to ${TARGET} for subset: ${SUBSET}"

GUIDELLM_CMD=(
    guidellm benchmark
    --target "${TARGET}"
    --data "${DATASET}"
    --data-args "{\"data_files\": \"${SUBSET}.jsonl\"}"
    --data-column-mapper '{"text_column":"prompt"}'
    --profile throughput
    --rate "${RATE}"
    --max-requests "${MAX_REQUESTS}"
)
[[ -n "${BACKEND_ARGS}" ]] && GUIDELLM_CMD+=(--backend-args "${BACKEND_ARGS}")

GUIDELLM__MAX_CONCURRENCY="${MAX_CONCURRENCY}" "${GUIDELLM_CMD[@]}"

echo "[INFO] Requests complete."

# ---------------------------------------------------------------------------
# Step 2: Snapshot Prometheus acceptance metrics
# ---------------------------------------------------------------------------
echo "[INFO] Capturing acceptance rate from ${METRICS_ENDPOINT}..."

mkdir -p "$(dirname "${OUTPUT}")"
python "${PARENT_SCRIPTS}/get_acceptance_rate.py" \
    --endpoint "${METRICS_ENDPOINT}" \
    --output "${OUTPUT}"

echo "[INFO] Acceptance rate saved to: ${OUTPUT}"
