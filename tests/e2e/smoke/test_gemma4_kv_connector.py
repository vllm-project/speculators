"""E2E test for the out-of-tree Gemma4KVConnector.

Gemma4KVConnector extends vLLM's ExampleHiddenStatesConnector to also export the
verifier's last sliding-window (local) and last full-attention (global) layer KV
caches alongside the hidden states, into one .safetensors per request. This is
the training-data producer for finetuning Gemma4 MTP draft models, whose
query-only attention borrows exactly those two verifier KV caches.
"""

from pathlib import Path

import pytest

from tests.conftest import requires_cuda, requires_multi_gpu
from tests.e2e.utils import run_gemma4_kv_extraction

MODEL = "google/gemma-4-E4B-it"


@pytest.mark.e2e
@pytest.mark.slow
@requires_cuda
def test_gemma4_kv_connector_smoke(tmp_path: Path):
    """Single-GPU (tp=1) extraction: shapes, alignment, both verifier KV caches."""
    run_gemma4_kv_extraction(MODEL, tmp_path, tensor_parallel_size=1)


@pytest.mark.e2e
@pytest.mark.slow
@requires_multi_gpu
def test_gemma4_kv_connector_tp2(tmp_path: Path):
    """tp=2: verifier KV heads are sharded per rank; the connector all-gathers
    (and dedups GQA replicas) so saved tensors carry the full head count."""
    run_gemma4_kv_extraction(MODEL, tmp_path, tensor_parallel_size=2)
