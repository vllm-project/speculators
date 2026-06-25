#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <unit|integration>}"

echo "~~~ System info"
cat /etc/issue

echo "--- Installing system packages"
git fetch --tags --unshallow 2>/dev/null || git fetch --tags
apt-get update -qq && apt-get install -y -qq curl g++ gcc make python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:$PATH"

echo "~~~ GPU info"
nvidia-smi

echo "--- Setting up Python environment"
find /model-cache/.builds -maxdepth 1 -type d -mmin +240 -exec rm -rf {} + 2>/dev/null || true
BUILD_DIR="/model-cache/.builds/${BUILDKITE_JOB_ID}"
mkdir -p "$BUILD_DIR"
trap 'rm -rf "$BUILD_DIR" || true' EXIT

export UV_NO_PROGRESS=1
export UV_CACHE_DIR="$BUILD_DIR/.uv-cache"
uv venv "$BUILD_DIR/testvenv" --python "${PYTHON_VERSION}"
source "$BUILD_DIR/testvenv/bin/activate"

export UV_TORCH_BACKEND=cu130
export HF_HOME=/model-cache
uv pip install .[dev]

if [ -n "${TRANSFORMERS_VERSION:-}" ] && [ "${TRANSFORMERS_VERSION}" != "latest" ]; then
  uv pip install "transformers${TRANSFORMERS_VERSION}"
fi

echo "+++ Running tests"
python -m pytest -ra "tests/${TEST_TYPE}"
