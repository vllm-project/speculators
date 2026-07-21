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
export UV_NO_PROGRESS=1
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv testvenv --python "${PYTHON_VERSION}"
source testvenv/bin/activate

export UV_TORCH_BACKEND=cu130
export HF_HOME=/model-cache
uv pip install .[dev]

if [ -n "${TRANSFORMERS_VERSION:-}" ] && [ "${TRANSFORMERS_VERSION}" != "latest" ]; then
  uv pip install "transformers${TRANSFORMERS_VERSION}"
fi

# Install FA4 on Hopper+ GPUs (compute capability >= 9.0) so FA4 tests run.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ "${CC:-0}" -ge 90 ]; then
  echo "--- Installing flash-attn (FA4) for Hopper+ GPU"
  uv pip install flash-attn --prerelease allow || echo "WARNING: flash-attn install failed, FA4 tests will be skipped"
fi

echo "+++ Running tests"
python -m pytest -ra "tests/${TEST_TYPE}"
