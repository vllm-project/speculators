#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <unit|integration|smoke>}"

case "${TEST_TYPE}" in
  unit) TEST_PATH="tests/unit" ;;
  integration) TEST_PATH="tests/integration" ;;
  smoke) TEST_PATH="tests/e2e/smoke" ;;
  *) echo "Unknown test type: ${TEST_TYPE}" >&2; exit 1 ;;
esac

echo "~~~ System info"
cat /etc/issue

export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "--- Installing system packages"
git fetch --tags --unshallow 2>/dev/null || git fetch --tags
apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq curl g++ gcc make python3-dev
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

if [ "${TEST_TYPE}" = "smoke" ]; then
  echo "--- Setting up vLLM environment"
  uv venv vllm_venv --python "${PYTHON_VERSION}"
  VLLM_VENV_PYTHON="$PWD/vllm_venv/bin/python"
  UV_TORCH_BACKEND=cu130 uv pip install --python "${VLLM_VENV_PYTHON}" vllm
  export VLLM_PYTHON="${VLLM_VENV_PYTHON}"

  # This image has no CUDA toolkit (nvcc), so FlashInfer can't JIT-compile its
  # sampling kernel at startup. Fall back to vLLM's native sampler instead.
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

echo "+++ Running tests"
python -m pytest -ra "${TEST_PATH}"
