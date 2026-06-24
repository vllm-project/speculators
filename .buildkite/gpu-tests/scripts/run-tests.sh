#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <unit|integration>}"

cat /etc/issue

git fetch --tags --unshallow 2>/dev/null || git fetch --tags

apt-get update -qq && apt-get install -y -qq curl g++ gcc make python3-dev

curl -LsSf https://astral.sh/uv/install.sh | sh

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:$PATH"
nvidia-smi

export UV_NO_PROGRESS=1
uv venv testvenv --python "${PYTHON_VERSION}"
source testvenv/bin/activate

export UV_TORCH_BACKEND=cu130
export HF_HOME=/model-cache
uv pip install .[dev]

if [ -n "${TRANSFORMERS_VERSION:-}" ] && [ "${TRANSFORMERS_VERSION}" != "latest" ]; then
  uv pip install "transformers${TRANSFORMERS_VERSION}"
fi

python -m pytest -ra "tests/${TEST_TYPE}"
