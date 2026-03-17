"""Stub vllm so data_generation unit tests can import without a vllm install."""

import sys
from unittest.mock import MagicMock

_VLLM_STUBS = [
    "vllm",
    "vllm.config",
    "vllm.sampling_params",
    "vllm.utils",
    "vllm.utils.hashing",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.executor",
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.request",
    "vllm.v1.structured_output",
]

for _mod in _VLLM_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
