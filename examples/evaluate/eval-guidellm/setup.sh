#!/usr/bin/env bash
# Install dependencies for speculator model evaluation

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

USE_UV=false

# ==============================================================================
# Helper Functions
# ==============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Optional:
  --use-uv       Use uv for faster package installation
  -h, --help     Show this help message

Example:
  $0              # Install with pip
  $0 --use-uv     # Install with uv (faster)
EOF
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-uv)
            USE_UV=true
            shift
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
# Validate Environment
# ==============================================================================

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "[ERROR] Requirements file not found: ${REQUIREMENTS_FILE}" >&2
    exit 1
fi

if [[ "${USE_UV}" == "true" ]]; then
    if ! command -v uv &> /dev/null; then
        echo "[ERROR] uv not found but --use-uv was specified" >&2
        echo "[ERROR] Install uv: https://github.com/astral-sh/uv" >&2
        echo "[ERROR] Or run without --use-uv to use pip" >&2
        exit 1
    fi
fi

# ==============================================================================
# Install Dependencies
# ==============================================================================

echo "[INFO] Installing dependencies for speculator model evaluation..."

if [[ "${USE_UV}" == "true" ]]; then
    echo "[INFO] Using uv for installation (faster)"
    uv pip install -r "${REQUIREMENTS_FILE}"
else
    echo "[INFO] Using pip for installation"
    pip install -r "${REQUIREMENTS_FILE}"
fi

echo "[INFO] Setup complete!"
echo "[INFO] Dependencies installed:"
echo "[INFO]   - vllm"
echo "[INFO]   - guidellm"
echo "[INFO]   - huggingface-hub (provides 'hf' command)"
