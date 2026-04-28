"""KV connector that quantizes hidden states to FP8 before writing to disk.

Drop-in replacement for vLLM's ExampleHiddenStatesConnector. Loaded via
the ``kv_connector_module_path`` factory mechanism — no vLLM modifications
required.

Launch example::

    python scripts/launch_vllm.py MODEL --fp8-quantize -- ...
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import safetensors.torch
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    ExampleHiddenStatesConnectorMetadata,
    extract_from_kv_cache,
)

from speculators.data_generation.fp8_utils import (
    SCALES_KEY,
    quantize_tensor_to_fp8,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheConfig


class FP8HiddenStatesConnector(ExampleHiddenStatesConnector):
    """Quantizes hidden states to float8_e4m3fn with per-token scaling
    before saving to safetensors.

    The output file contains three tensors::

        hidden_states        – fp8 [seq_len, num_layers, hidden_size]
        hidden_states_scales – fp32 [seq_len, 1]
        token_ids            – int64 [seq_len]
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._pending: list[Future] = []

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata)

        os.makedirs(self._storage_path, exist_ok=True)
        for request in connector_metadata.requests:
            hidden_states = extract_from_kv_cache(
                kv_layer, request.slot_mapping, request.token_ids.shape[0]
            )
            cpu_hs = hidden_states.detach().cpu()

            # Flatten to [seq_len, num_layers * hidden_size] for quantization,
            # then reshape back so the file layout stays consistent.
            original_shape = cpu_hs.shape  # [seq_len, num_layers, hidden_size]
            flat = cpu_hs.reshape(cpu_hs.shape[0], -1)
            fp8_flat, scales = quantize_tensor_to_fp8(flat)
            fp8_hs = fp8_flat.reshape(original_shape)

            tensors: dict[str, torch.Tensor] = {
                "hidden_states": fp8_hs,
                SCALES_KEY: scales,
                "token_ids": request.token_ids.detach().cpu(),
            }
            self._pending.append(
                self._executor.submit(
                    safetensors.torch.save_file, tensors, request.filename
                )
            )

    def wait_for_save(self):
        for f in self._pending:
            f.result()
        self._pending.clear()
