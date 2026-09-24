#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <unit|integration|smoke|e2e|regression|multi-gpu>}"

PYTEST_EXTRA_ARGS=()
case "${TEST_TYPE}" in
  unit)       TEST_PATH="tests/unit" ;;
  integration) TEST_PATH="tests/integration" ;;
  smoke)      TEST_PATH="tests/e2e/smoke" ;;
  e2e)        TEST_PATH="tests/e2e"; PYTEST_EXTRA_ARGS+=(--ignore=tests/e2e/hs_connectors -m "not regression") ;;
  regression) TEST_PATH="tests/e2e"; PYTEST_EXTRA_ARGS+=(--ignore=tests/e2e/hs_connectors -m "regression") ;;
  multi-gpu)  TEST_PATH="tests/e2e"; PYTEST_EXTRA_ARGS+=(-m "multi_gpu") ;;
  *) echo "Unknown test type: ${TEST_TYPE}" >&2; exit 1 ;;
esac

echo "~~~ System info"
cat /etc/issue
df -h

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
  if [ "${TRANSFORMERS_VERSION}" = "main" ]; then
    uv pip install --reinstall "transformers @ git+https://github.com/huggingface/transformers.git@main"
  else
    uv pip install --reinstall "transformers${TRANSFORMERS_VERSION}"
  fi
fi

if [[ "${TEST_TYPE}" =~ ^(smoke|e2e|regression|multi-gpu)$ ]]; then
  echo "--- Setting up vLLM environment"
  uv venv vllm_venv --python "${PYTHON_VERSION}"
  VLLM_VENV_PYTHON="$PWD/vllm_venv/bin/python"

  # vllm/torch require setuptools but don't explicitly depend on it for Python <= 3.11
  if [[ "${PYTHON_VERSION}" =~ 3.1[01] ]]; then
    uv pip install --python "${VLLM_VENV_PYTHON}" setuptools
  fi

  if [ "${VLLM_VERSION:-}" = "nightly" ]; then
    UV_TORCH_BACKEND=cu130 uv pip install --python "${VLLM_VENV_PYTHON}" vllm \
      --extra-index-url https://wheels.vllm.ai/nightly/cu130
  else
    UV_TORCH_BACKEND=cu130 uv pip install --python "${VLLM_VENV_PYTHON}" vllm
  fi

  export VLLM_PYTHON="${VLLM_VENV_PYTHON}"
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

echo "+++ Running tests"
if [ -n "${VLLM_PYTHON:-}" ]; then
  VLLM_BIN_DIR="$(dirname "${VLLM_PYTHON}")"
  export PATH="$PATH:${VLLM_BIN_DIR}"
fi
python -m pytest -ra "${TEST_PATH}" "${PYTEST_EXTRA_ARGS[@]}"
