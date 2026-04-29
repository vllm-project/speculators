"""KV connector that quantizes hidden states to FP8 before writing to disk.

Drop-in replacement for vLLM's ExampleHiddenStatesConnector. Loaded via
the ``kv_connector_module_path`` factory mechanism - no vLLM modifications
required.

Launch example::

    python scripts/launch_vllm.py MODEL --fp8-quantize -- ...
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as _eh_mod,
)
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionMetadata

from speculators.data_generation.fp8_utils import (
    SCALES_KEY,
    quantize_tensor_to_fp8,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata


class FP8HiddenStatesConnector(_eh_mod.ExampleHiddenStatesConnector):
    """Quantizes hidden states to float8_e4m3fn with per-token scaling
    before saving to safetensors.

    The output file contains three tensors::

        hidden_states        - fp8 [seq_len, num_layers, hidden_size]
        hidden_states_scales - fp32 [seq_len, 1]
        token_ids            - int64 [seq_len]
    """

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **_kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        assert isinstance(
            connector_metadata, _eh_mod.ExampleHiddenStatesConnectorMetadata
        )

        os.makedirs(self._storage_path, exist_ok=True)
        for request in connector_metadata.requests:
            hidden_states = _eh_mod.extract_from_kv_cache(
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
            safetensors.torch.save_file(tensors, request.filename)
