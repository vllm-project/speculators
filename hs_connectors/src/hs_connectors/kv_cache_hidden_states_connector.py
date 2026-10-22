from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as base,
)
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class KVCachePendingSave(base.PendingSave):
    kv_block_ids: list[int] = field(default_factory=list)


class KVCacheHiddenStatesConnector(base.ExampleHiddenStatesConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError(
                "KVCacheHiddenStatesConnector requires tensor_parallel_size=1"
            )
        layer_ids = sorted(
            self._kv_transfer_config.get_from_extra_config("kv_cache_layer_ids", [])
        )
        if not layer_ids:
            raise ValueError(
                "kv_cache_layer_ids must be set in kv_connector_extra_config"
            )
        layer_groups = {
            name: group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }
        self._kv_layer_names = []
        for layer_id in layer_ids:
            names = [
                name
                for name in layer_groups
                if f".layers.{layer_id}." in name and "self_attn" in name
            ]
            if len(names) != 1:
                raise ValueError(f"No unique attention layer found for id {layer_id}")
            self._kv_layer_names.append(names[0])
        group_ids = {layer_groups[name] for name in self._kv_layer_names}
        if len(group_ids) != 1:
            raise ValueError("kv_cache_layer_ids must share one KV cache group")
        self._kv_group_id = group_ids.pop()
        self._kv_caches: list[torch.Tensor] = []
        self._pending_kv: dict[str, torch.Tensor] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self._kv_caches = [kv_caches[name] for name in self._kv_layer_names]

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ):
        result = super().request_finished_all_groups(request, block_ids)
        pending = self._pending_saves.get(request.request_id)
        if pending is not None:
            self._pending_saves[request.request_id] = KVCachePendingSave(
                **vars(pending), kv_block_ids=list(block_ids[self._kv_group_id])
            )
        return result

    def _submit_async_write(self, pending: base.PendingSave) -> None:
        if self._is_tp_rank_zero and isinstance(pending, KVCachePendingSave):
            block_size = self._kv_caches[0].shape[2]
            block_ids = torch.tensor(pending.kv_block_ids, dtype=torch.long)
            slot_mapping = (
                block_ids[:, None] * block_size + torch.arange(block_size)
            ).flatten()
            num_tokens = pending.token_ids.shape[0]

            copy_stream = self._get_copy_stream()
            ready_event = torch.cuda.Event()
            ready_event.record()
            copy_stream.wait_event(ready_event)
            with torch.cuda.stream(copy_stream):
                with gpu_sync_allowed():
                    slot_mapping = slot_mapping.to(
                        self._kv_caches[0].device, non_blocking=True
                    )
                kv = torch.stack(
                    [
                        base.extract_from_kv_cache(kv_cache, slot_mapping, num_tokens)
                        for kv_cache in self._kv_caches
                    ],
                    dim=1,
                )
                pinned_kv = torch.empty_like(kv, device="cpu", pin_memory=True)
                pinned_kv.copy_(kv, non_blocking=True)
            self._pending_kv[pending.filename] = pinned_kv
        super()._submit_async_write(pending)

    def _write_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
    ) -> None:
        tensors["kv_cache"] = self._pending_kv.pop(filename)
        super()._write_tensors(tensors, event, filename, lock_fd)
