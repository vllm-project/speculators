#!/bin/bash
#
# Sweep concurrency levels against a running vLLM server to find the
# optimal --concurrency for response regeneration.
#
# Prerequisites: a vLLM server must already be running.
#
# Usage:
#   # Sweep only — prints results table and recommended concurrency
#   ./sweep_concurrency.sh --model google/gemma-4-31B-it --dataset ultrachat
#
#   # Sweep, then run full regeneration with the best concurrency
#   ./sweep_concurrency.sh --model google/gemma-4-31B-it --dataset ultrachat --run --resume
#
#   # Custom endpoint, duration, and concurrency set
#   ./sweep_concurrency.sh --endpoint http://host:8000/v1/chat/completions \
#       --model MODEL --dataset tulu3 --duration 90 --concurrencies "64 256 1024 4096"
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Sweep defaults ──────────────────────────────────────────────────
ENDPOINT="http://127.0.0.1:8000/v1/chat/completions"
CONCURRENCIES=(64 128 256 512 1024 2048)
DURATION=60
WARMUP_SAMPLES=20
RUN_AFTER=false

# ── Argument parsing ───────────────────────────────────────────────
# Everything not consumed below passes through to script.py.
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --concurrencies)
            IFS=' ,' read -ra CONCURRENCIES <<< "$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --warmup-samples)
            WARMUP_SAMPLES="$2"
            shift 2
            ;;
        --run)
            RUN_AFTER=true
            shift
            ;;
        --concurrency)
            echo "Note: --concurrency ignored during sweep (use --concurrencies)"
            shift 2
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Health check ────────────────────────────────────────────────────
MODELS_ENDPOINT="${ENDPOINT/\/v1\/chat\/completions/\/v1\/models}"
printf "Checking server at %s... " "$MODELS_ENDPOINT"
if ! curl -s --connect-timeout 5 --max-time 10 "$MODELS_ENDPOINT" > /dev/null 2>&1; then
    echo "FAILED"
    echo ""
    echo "Error: vLLM server not reachable at $MODELS_ENDPOINT"
    echo "Start a vLLM server first, then re-run this script."
    exit 1
fi
echo "OK"
echo ""

# ── Temp directory (cleaned up on exit) ─────────────────────────────
SWEEP_TMPDIR=$(mktemp -d)
trap 'rm -rf "$SWEEP_TMPDIR"' EXIT

# ── Helper: count conversations and total completion tokens ─────────
# Prints two space-separated values: conversation_count  token_count
count_output() {
    python3 -c "
import json, sys
seen = set()
tokens = 0
with open(sys.argv[1]) as f:
    for line in f:
        try:
            d = json.loads(line)
            seen.add(d.get('primary_id') or d.get('id'))
            tokens += d.get('metadata', {}).get('usage', {}).get('completion_tokens', 0)
        except (json.JSONDecodeError, KeyError):
            pass
print(len(seen), tokens)
" "$1" 2>/dev/null || echo "0 0"
}

# ── Warmup ──────────────────────────────────────────────────────────
echo "Warming up ($WARMUP_SAMPLES samples at concurrency 64)..."
echo ""
WARMUP_START=$(date +%s%N)
python "$SCRIPT_DIR/script.py" \
    "${PYTHON_ARGS[@]}" \
    --endpoint "$ENDPOINT" \
    --concurrency 64 \
    --limit "$WARMUP_SAMPLES" \
    --outfile "$SWEEP_TMPDIR/warmup.jsonl"
WARMUP_END=$(date +%s%N)
WARMUP_S=$(awk "BEGIN {printf \"%.1f\", ($WARMUP_END - $WARMUP_START) / 1000000000}")
echo ""
echo "Warmup done (${WARMUP_S}s)"
echo ""

# ── Randomised sweep ───────────────────────────────────────────────
SHUFFLED=($(printf '%s\n' "${CONCURRENCIES[@]}" | shuf))

declare -A RESULTS_CONVS
declare -A RESULTS_TOKENS
declare -A RESULTS_RATE

echo "Sweeping concurrencies: ${CONCURRENCIES[*]}"
echo "Duration per trial: ${DURATION}s"
echo ""

for C in "${SHUFFLED[@]}"; do
    OUTFILE="$SWEEP_TMPDIR/sweep_${C}.jsonl"
    touch "$OUTFILE"
    echo "── Trial: concurrency=$C (${DURATION}s) ──"

    # Run for a fixed duration, then SIGINT to stop gracefully.
    # script.py flushes after each completed conversation, so the output
    # file contains all work that finished before the signal.
    timeout --signal=INT "${DURATION}s" \
        python "$SCRIPT_DIR/script.py" \
            "${PYTHON_ARGS[@]}" \
            --endpoint "$ENDPOINT" \
            --concurrency "$C" \
            --limit 999999 \
            --outfile "$OUTFILE" \
        || true

    read -r CONV_COUNT TOKEN_COUNT <<< "$(count_output "$OUTFILE")"
    TOKEN_RATE=$(awk "BEGIN {printf \"%.0f\", $TOKEN_COUNT / $DURATION}")

    RESULTS_CONVS[$C]=$CONV_COUNT
    RESULTS_TOKENS[$C]=$TOKEN_COUNT
    RESULTS_RATE[$C]=$TOKEN_RATE

    echo ""
    echo "=> $CONV_COUNT conversations, $TOKEN_COUNT tokens in ${DURATION}s (${TOKEN_RATE} tok/s)"
    echo ""
done

echo ""

# ── Find best ───────────────────────────────────────────────────────
BEST_C=""
BEST_RATE="0"
for C in "${CONCURRENCIES[@]}"; do
    RATE=${RESULTS_RATE[$C]}
    if awk "BEGIN {exit !($RATE > $BEST_RATE)}"; then
        BEST_RATE=$RATE
        BEST_C=$C
    fi
done

# ── Results table ───────────────────────────────────────────────────
echo "==========================================="
echo "Concurrency Sweep Results (${DURATION}s per trial)"
echo "==========================================="
printf "%-15s %-15s %-12s %-10s\n" "Concurrency" "Conversations" "Tokens" "Tok/sec"
printf "%-15s %-15s %-12s %-10s\n" "-----------" "-------------" "------" "-------"
for C in "${CONCURRENCIES[@]}"; do
    MARKER=""
    [ "$C" = "$BEST_C" ] && MARKER="  <-- best"
    printf "%-15s %-15s %-12s %-10s%s\n" "$C" "${RESULTS_CONVS[$C]}" "${RESULTS_TOKENS[$C]}" "${RESULTS_RATE[$C]}" "$MARKER"
done
echo ""
echo "Optimal concurrency: $BEST_C (${BEST_RATE} tok/s)"
echo ""

# ── Optional: chain into full regeneration ──────────────────────────
if [ "$RUN_AFTER" = true ]; then
    echo "==========================================="
    echo "Running full regeneration (concurrency=$BEST_C)"
    echo "==========================================="
    echo ""
    python "$SCRIPT_DIR/script.py" \
        "${PYTHON_ARGS[@]}" \
        --endpoint "$ENDPOINT" \
        --concurrency "$BEST_C"
    exit $?
fi

echo "Run full regeneration with:"
echo "  python $SCRIPT_DIR/script.py ${PYTHON_ARGS[*]} --endpoint $ENDPOINT --concurrency $BEST_C"
