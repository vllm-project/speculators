#!/usr/bin/env bash
# Run GuideLLM benchmark against vLLM server

set -euo pipefail

# ==============================================================================
# Default Configuration
# ==============================================================================

TARGET="http://localhost:8000/v1"
DATASET=""
GUIDELLM_RESULTS="guidellm_results.json"
GUIDELLM_LOG="guidellm_output.log"

# Sampling parameters
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 -d DATASET [OPTIONS]

Required:
  -d DATASET        Dataset for benchmarking. Can be:
                    - Built-in dataset name (e.g., "emulated")
                    - HuggingFace dataset (e.g., "org/dataset")
                    - HuggingFace dataset with specific file (e.g., "org/dataset:file.jsonl")
                    - Local .jsonl file path
                    - Local directory (runs benchmark on all .jsonl files)

Optional:
  --target URL              Target server URL (default: http://localhost:8000/v1)
  --output-file FILE        Results JSON file (default: guidellm_results.json)
  --log-file FILE          Output log file (default: guidellm_output.log)
  --temperature TEMP        Sampling temperature (default: 0.6)
  --top-p TOP_P            Top-p sampling (default: 0.95)
  --top-k TOP_K            Top-k sampling (default: 20)
  -h, --help               Show this help message

Examples:
  $0 -d emulated                                            # Built-in dataset
  $0 -d "RedHatAI/speculator_benchmarks"                    # HuggingFace (all files)
  $0 -d "RedHatAI/speculator_benchmarks:math_reasoning.jsonl"  # HuggingFace (specific file)
  $0 -d "./my_dataset.jsonl"                                # Single local file
  $0 -d "./my_datasets/"                                    # Directory (all .jsonl files)
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

    # Download the dataset using hf download and capture the download path
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

# Case 3: Local file or built-in dataset name
else
    if [[ -f "${DATASET}" ]]; then
        echo "[INFO] Using local file: ${DATASET}"
    else
        echo "[INFO] Using built-in dataset: ${DATASET}"
    fi
    DATASET_FILES=("${DATASET}")
fi

# If we have a directory, find .jsonl files
if [[ -n "${DATASET_DIR}" ]]; then
    # If a specific file was requested, find only that file
    if [[ -n "${SPECIFIC_FILE}" ]]; then
        echo "[INFO] Searching for specific file: ${SPECIFIC_FILE}"

        specific_path=$(find -L "${DATASET_DIR}" -type f -name "${SPECIFIC_FILE}" | head -1)

        if [[ -z "${specific_path}" ]]; then
            echo "[ERROR] Specific file not found: ${SPECIFIC_FILE}" >&2
            echo "[ERROR] Available files in dataset:" >&2
            find -L "${DATASET_DIR}" -type f -name "*.jsonl" -exec basename {} \; | sort >&2
            exit 1
        fi

        DATASET_FILES=("${specific_path}")
        echo "[INFO] Using specific file: ${specific_path}"
    else
        # No specific file requested, find all .jsonl files
        echo "[INFO] Searching for .jsonl files in: ${DATASET_DIR}"

        while IFS= read -r -d '' file; do
            DATASET_FILES+=("$file")
        done < <(find -L "${DATASET_DIR}" -type f -name "*.jsonl" -print0 | sort -z)

        if [[ ${#DATASET_FILES[@]} -eq 0 ]]; then
            echo "[ERROR] No .jsonl files found in directory: ${DATASET_DIR}" >&2
            exit 1
        fi

        echo "[INFO] Found ${#DATASET_FILES[@]} dataset file(s)"
    fi
fi

# ==============================================================================
# Run Benchmark(s)
# ==============================================================================

for dataset_file in "${DATASET_FILES[@]}"; do
    # Generate output filenames
    if [[ ${#DATASET_FILES[@]} -gt 1 ]]; then
        # Multiple files: append dataset basename to output names
        dataset_basename=$(basename "${dataset_file}" .jsonl)
        output_file="${GUIDELLM_RESULTS%.json}_${dataset_basename}.json"
        log_file="${GUIDELLM_LOG%.log}_${dataset_basename}.log"
    else
        # Single file: use provided filenames
        output_file="${GUIDELLM_RESULTS}"
        log_file="${GUIDELLM_LOG}"
    fi

    echo "[INFO] Running guidellm benchmark..."
    echo "[INFO]   Target: ${TARGET}"
    echo "[INFO]   Dataset: ${dataset_file}"
    echo "[INFO]   Sampling params - temperature: ${TEMPERATURE}, top_p: ${TOP_P}, top_k: ${TOP_K}"
    echo "[INFO]   Output: ${output_file}"

    GUIDELLM__PREFERRED_ROUTE="chat_completions" \
    guidellm benchmark \
      --target "${TARGET}" \
      --data "${dataset_file}" \
      --rate-type throughput \
      --output-path "${output_file}" \
      --backend-args "{\"extra_body\": {\"chat_completions\": {\"temperature\":${TEMPERATURE}, \"top_p\":${TOP_P}, \"top_k\":${TOP_K}}}}" \
      | tee "${log_file}"

    echo "[INFO] Benchmark complete for: ${dataset_file}"
done

echo "[INFO] All benchmarks complete"
