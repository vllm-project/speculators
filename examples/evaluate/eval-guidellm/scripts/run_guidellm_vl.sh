#!/usr/bin/env bash
# Run VL chat benchmark against vLLM server.

set -euo pipefail

# ==============================================================================
# Configuration Variables
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET=""
DATASET=""
GUIDELLM_RESULTS=""
GUIDELLM_LOG=""
TEMPERATURE=""
TOP_P=""
TOP_K=""
IMAGE_ROOT="${IMAGE_ROOT:-}"

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 -d DATASET [OPTIONS]

Required:
  -d DATASET        Multimodal dataset input. Can be:
                    - HuggingFace dataset (e.g., "org/dataset")
                    - HuggingFace dataset with specific file (e.g., "org/dataset:file.jsonl")
                    - Local .json/.jsonl file path
                    - Local directory (runs eval on all .json/.jsonl files)

Optional:
  --target URL              Target server URL (default: http://localhost:8000/v1)
  --output-file FILE        Output base file path (default: guidellm_results.json)
  --log-file FILE           Output log file (default: guidellm_output.log)
  --image-root DIR          Optional image root for relative image paths
  --temperature TEMP        Sampling temperature (default: 0)
  --top-p TOP_P             Top-p sampling (default: 0.95)
  --top-k TOP_K             Top-k sampling (default: 20)
  -h, --help                Show this help message
EOF
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -d)
            DATASET="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --output-file)
            GUIDELLM_RESULTS="$2"
            shift 2
            ;;
        --log-file)
            GUIDELLM_LOG="$2"
            shift 2
            ;;
        --image-root)
            IMAGE_ROOT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
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
# Apply Defaults
# ==============================================================================

TARGET="${TARGET:-http://localhost:8000/v1}"
GUIDELLM_RESULTS="${GUIDELLM_RESULTS:-guidellm_results.json}"
GUIDELLM_LOG="${GUIDELLM_LOG:-guidellm_output.log}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

# ==============================================================================
# Validate Arguments
# ==============================================================================

if [[ -z "${DATASET}" ]]; then
    echo "[ERROR] Missing required argument: -d DATASET" >&2
    show_usage
    exit 1
fi

# ==============================================================================
# Process Dataset Input
# ==============================================================================

DATASET_DIR=""
DATASET_FILES=()
SPECIFIC_FILE=""

# Check for colon syntax: "HF_dataset:specific_file.jsonl"
if [[ "${DATASET}" == *:* ]]; then
    HF_DATASET="${DATASET%%:*}"
    SPECIFIC_FILE="${DATASET##*:}"
    echo "[INFO] Detected HuggingFace dataset with specific file"
    echo "[INFO]   Dataset: ${HF_DATASET}"
    echo "[INFO]   File: ${SPECIFIC_FILE}"
    DATASET="${HF_DATASET}"
fi

# Case 1: HuggingFace dataset stub (contains "/" and doesn't exist locally)
if [[ "${DATASET}" == */* ]] && [[ ! -e "${DATASET}" ]]; then
    echo "[INFO] Detected HuggingFace dataset: ${DATASET}"

    dataset_dir=$(hf download "${DATASET}" --repo-type dataset 2>&1 | tail -1)
    if [[ $? -ne 0 ]] || [[ -z "${dataset_dir}" ]]; then
        echo "[ERROR] Failed to download dataset: ${DATASET}" >&2
        exit 1
    fi

    echo "[INFO] Dataset downloaded to: ${dataset_dir}"
    DATASET_DIR="${dataset_dir}"

# Case 2: Local directory
elif [[ -d "${DATASET}" ]]; then
    echo "[INFO] Detected local directory: ${DATASET}"
    DATASET_DIR="${DATASET}"

# Case 3: Local file
elif [[ -f "${DATASET}" ]]; then
    echo "[INFO] Using local file: ${DATASET}"
    DATASET_FILES=("${DATASET}")

# Case 4: Unsupported input in VL mode
else
    echo "[ERROR] VL mode requires a local file/directory or HuggingFace dataset path." >&2
    echo "[ERROR] Unsupported DATASET value: ${DATASET}" >&2
    exit 1
fi

# If we have a directory, find .json/.jsonl files
if [[ -n "${DATASET_DIR}" ]]; then
    if [[ -n "${SPECIFIC_FILE}" ]]; then
        echo "[INFO] Searching for specific file: ${SPECIFIC_FILE}"

        specific_path=$(find -L "${DATASET_DIR}" -type f -name "${SPECIFIC_FILE}" | head -1)
        if [[ -z "${specific_path}" ]]; then
            echo "[ERROR] Specific file not found: ${SPECIFIC_FILE}" >&2
            echo "[ERROR] Available files in dataset:" >&2
            find -L "${DATASET_DIR}" -type f \( -name "*.jsonl" -o -name "*.json" \) -exec basename {} \; | sort >&2
            exit 1
        fi

        DATASET_FILES=("${specific_path}")
        echo "[INFO] Using specific file: ${specific_path}"
    else
        echo "[INFO] Searching for .json/.jsonl files in: ${DATASET_DIR}"

        while IFS= read -r -d '' file; do
            DATASET_FILES+=("$file")
        done < <(find -L "${DATASET_DIR}" -type f \( -name "*.jsonl" -o -name "*.json" \) -print0 | sort -z)

        if [[ ${#DATASET_FILES[@]} -eq 0 ]]; then
            echo "[ERROR] No .json/.jsonl files found in directory: ${DATASET_DIR}" >&2
            exit 1
        fi

        echo "[INFO] Found ${#DATASET_FILES[@]} dataset file(s)"
    fi
fi

# ==============================================================================
# Run VL Evaluation(s)
# ==============================================================================

for dataset_file in "${DATASET_FILES[@]}"; do
    if [[ ${#DATASET_FILES[@]} -gt 1 ]]; then
        dataset_basename="$(basename "${dataset_file}")"
        dataset_basename="${dataset_basename%.*}"
        vl_results_file="${GUIDELLM_RESULTS%.json}_vl_eval_results_${dataset_basename}.jsonl"
        vl_summary_file="${GUIDELLM_RESULTS%.json}_vl_eval_summary_${dataset_basename}.json"
        log_file="${GUIDELLM_LOG%.log}_${dataset_basename}.log"
    else
        output_dir="$(dirname "${GUIDELLM_RESULTS}")"
        vl_results_file="${output_dir}/vl_eval_results.jsonl"
        vl_summary_file="${output_dir}/vl_eval_summary.json"
        log_file="${GUIDELLM_LOG}"
    fi

    echo "[INFO] Running VL chat eval..."
    echo "[INFO]   Target: ${TARGET}"
    echo "[INFO]   Dataset: ${dataset_file}"
    echo "[INFO]   Sampling params - temperature: ${TEMPERATURE}, top_p: ${TOP_P}, top_k: ${TOP_K}"
    echo "[INFO]   Output records: ${vl_results_file}"
    echo "[INFO]   Output summary: ${vl_summary_file}"
    if [[ -n "${IMAGE_ROOT}" ]]; then
        echo "[INFO]   Image root: ${IMAGE_ROOT}"
    fi

    vl_cmd=(
        python "${SCRIPT_DIR}/run_vl_chat_eval.py"
        --target "${TARGET}"
        --dataset-file "${dataset_file}"
        --output-file "${vl_results_file}"
        --summary-file "${vl_summary_file}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --top-k "${TOP_K}"
    )
    if [[ -n "${IMAGE_ROOT}" ]]; then
        vl_cmd+=(--image-root "${IMAGE_ROOT}")
    fi

    set +e
    "${vl_cmd[@]}" | tee "${log_file}"
    vl_status=${PIPESTATUS[0]}
    set -e

    if [[ ${vl_status} -eq 10 ]]; then
        echo "[ERROR] Dataset has no multimodal records: ${dataset_file}" >&2
        exit 1
    fi
    if [[ ${vl_status} -ne 0 ]]; then
        echo "[ERROR] VL eval failed for dataset: ${dataset_file}" >&2
        exit "${vl_status}"
    fi

    echo "[INFO] VL eval complete for: ${dataset_file}"
done

echo "[INFO] All VL evaluations complete"
