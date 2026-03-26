#!/usr/bin/env bash
# Run all 4 MTP acceptance-rate evaluations sequentially (base + 3 finetuned epochs)
# then generate a side-by-side comparison report.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/fast_mtp/run_mtp_evals.sh
#
# Output: output/mtp_eval_<date>/comparison_report.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${REPO_ROOT}/examples/evaluate/eval-guidellm"
OUTPUT_BASE="${REPO_ROOT}/output/mtp_eval_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_BASE}"

echo "=================================================="
echo "MTP Acceptance Rate Evaluation"
echo "Output: ${OUTPUT_BASE}"
echo "=================================================="

run_eval() {
    local config="$1"
    local tag="$2"
    echo ""
    echo "--- Running eval: ${tag} ---"
    (
        cd "${EVAL_DIR}"
        bash run_evaluation.sh \
            -c "configs/${config}" \
            -o "${OUTPUT_BASE}/${tag}"
    )
    echo "--- Done: ${tag} ---"
}

run_eval "qwen3-next-80b-mtp-base.env"   "base"
run_eval "qwen3-next-80b-mtp-epoch0.env" "epoch0"
run_eval "qwen3-next-80b-mtp-epoch1.env" "epoch1"
run_eval "qwen3-next-80b-mtp-epoch2.env" "epoch2"

echo ""
echo "=================================================="
echo "Generating comparison report..."
echo "=================================================="

python "${EVAL_DIR}/scripts/compare_acceptance.py" \
    --results \
        "Base:${OUTPUT_BASE}/base/acceptance_analysis.txt" \
        "Epoch0:${OUTPUT_BASE}/epoch0/acceptance_analysis.txt" \
        "Epoch1:${OUTPUT_BASE}/epoch1/acceptance_analysis.txt" \
        "Epoch2:${OUTPUT_BASE}/epoch2/acceptance_analysis.txt" \
    --output "${OUTPUT_BASE}/comparison_report.md"

echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "Comparison report: ${OUTPUT_BASE}/comparison_report.md"
echo "=================================================="
