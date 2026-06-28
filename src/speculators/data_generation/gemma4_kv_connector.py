"""Out-of-tree KV connector for Gemma4 MTP training-data extraction.

Load it out-of-tree via --kv-transfer-config:

    {
      "kv_connector": "Gemma4KVConnector",
      "kv_connector_module_path":
        "speculators.data_generation.gemma4_kv_connector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {"shared_storage_path": "<dir>"}
    }
"""

import fcntl
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.communication_op import tensor_model_parallel_gather
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
    PendingSave,
    extract_from_kv_cache,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_KV_CACHE_NDIM = 5
_KV_CACHE_KV_AXIS_SIZE = 2

GEMMA4_KV_KEYS = (
    "kv_last_local_k",
    "kv_last_local_v",
    "kv_last_global_k",
    "kv_last_global_v",
)


def extract_real_kv_from_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-token K and V from a real attention layer's paged cache.

    Args:
        kv_cache: The layer's paged KV buffer, 5-D as described above.
        slot_mapping: Per-token physical cache slots (1-D, length >= num_tokens).
        num_tokens: Number of leading tokens to extract.

    Returns:
        A (K, V) tuple, each of shape
        (num_tokens, num_kv_heads, head_size).
    """
    if kv_cache.dim() != _KV_CACHE_NDIM or kv_cache.shape[1] != _KV_CACHE_KV_AXIS_SIZE:
        raise ValueError(
            "Gemma4KVConnector expects the 5-D KV layout "
            "(num_blocks, 2, block_size, num_kv_heads, head_size) used by the "
            f"FlashAttention/Triton backends; got shape {tuple(kv_cache.shape)}."
        )
    block_size = kv_cache.shape[2]
    block_idx = slot_mapping // block_size
    block_off = slot_mapping % block_size
    k = kv_cache[:, 0][block_idx, block_off][:num_tokens]
    v = kv_cache[:, 1][block_idx, block_off][:num_tokens]
    return k, v


@dataclass
class Gemma4PendingSave(PendingSave):
    """PendingSave plus the per-group block_ids of the two verifier KV
    layers, carried scheduler -> worker so their KV can be extracted."""

    local_block_ids: list[int] | None = None
    global_block_ids: list[int] | None = None


class Gemma4KVConnector(ExampleHiddenStatesConnector):
    """Hidden-states + verifier-KV extraction connector for Gemma4 MTP training."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        pp = vllm_config.parallel_config.pipeline_parallel_size
        if pp != 1:
            raise ValueError(
                f"Gemma4KVConnector does not support pipeline_parallel_size>1 "
                f"(got pp={pp})."
            )

        tcfg = vllm_config.model_config.hf_config.get_text_config()
        self._local_total_kv_heads = tcfg.num_key_value_heads

        global_kv = getattr(tcfg, "num_global_key_value_heads", None)
        self._global_total_kv_heads = global_kv or tcfg.num_key_value_heads

        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype != "auto":
            raise ValueError(
                "Gemma4KVConnector requires an unquantized KV cache "
                f"(cache_dtype='auto'); got {cache_dtype!r}. Quantized-cache "
                "handling (per-token head scales) is not yet supported."
            )

        self.verifier_local_layer, self._local_group_idx = self._resolve_layer(
            "sliding_attention"
        )
        self.verifier_global_layer, self._global_group_idx = self._resolve_layer(
            "full_attention"
        )

        self._local_kv_cache: torch.Tensor | None = None
        self._global_kv_cache: torch.Tensor | None = None

        logger.info(
            "Gemma4KVConnector: local=%s (group %d) global=%s (group %d)",
            self.verifier_local_layer,
            self._local_group_idx,
            self.verifier_global_layer,
            self._global_group_idx,
        )

    def _resolve_layer(self, attn_type: str) -> tuple[str, int]:
        """Resolve the last non-KV-shared verifier layer of attn_type.

        Args:
            attn_type: "sliding_attention" or "full_attention".

        Returns:
            A (layer_name, kv_cache_group_idx) tuple.
        """
        tcfg = self._vllm_config.model_config.hf_config.get_text_config()
        layer_types = list(getattr(tcfg, "layer_types", []))
        if not layer_types:
            raise ValueError(
                "Gemma4KVConnector requires the verifier config to expose "
                "'layer_types'; none found."
            )
        n_shared = getattr(tcfg, "num_kv_shared_layers", 0)
        num_non_shared = len(layer_types) - n_shared

        type_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lt in enumerate(layer_types[:num_non_shared]):
            type_to_indices[lt].append(idx)
        indices = type_to_indices.get(attn_type)
        if not indices:
            counts = {k: len(v) for k, v in type_to_indices.items()}
            raise ValueError(
                f"Gemma4KVConnector found no non-KV-shared '{attn_type}' "
                f"layer; layer-type counts: {counts}"
            )

        prefix = self._resolve_attn_layer_prefix()
        layer_name = f"{prefix}.{indices[-1]}.self_attn.attn"
        return layer_name, self._find_group_idx(layer_name)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        for name in (self.verifier_local_layer, self.verifier_global_layer):
            if name not in kv_caches:
                raise ValueError(
                    f"Resolved verifier KV layer {name!r} is not among the "
                    f"registered kv_caches keys: {sorted(kv_caches)[:8]}..."
                )
        self._local_kv_cache = kv_caches[self.verifier_local_layer]
        self._global_kv_cache = kv_caches[self.verifier_global_layer]

    def _resolve_attn_layer_prefix(self) -> str:
        assert self._kv_cache_config is not None
        for group in self._kv_cache_config.kv_cache_groups:
            for name in group.layer_names:
                if ".layers." in name and name.endswith(".self_attn.attn"):
                    return name.split(".layers.")[0] + ".layers"
        raise ValueError(
            "Gemma4KVConnector could not resolve the verifier attention-layer "
            "name prefix (no '*.layers.N.self_attn.attn' layer found)."
        )

    def _find_group_idx(self, layer_name: str) -> int:
        assert self._kv_cache_config is not None
        for i, group in enumerate(self._kv_cache_config.kv_cache_groups):
            if layer_name in group.layer_names:
                return i
        raise ValueError(
            f"Gemma4KVConnector: layer {layer_name!r} not found in any kv_cache_group."
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        result = super().request_finished_all_groups(request, block_ids)

        req_id = request.request_id
        base = self._pending_saves.get(req_id)
        if base is not None:
            self._pending_saves[req_id] = Gemma4PendingSave(
                req_id=base.req_id,
                filename=base.filename,
                token_ids=base.token_ids,
                block_ids=base.block_ids,
                local_block_ids=list(block_ids[self._local_group_idx]),
                global_block_ids=list(block_ids[self._global_group_idx]),
            )
        return result

    @staticmethod
    def _slot_mapping_from_blocks(
        block_ids: list[int], block_size: int
    ) -> torch.Tensor:
        block_ids_t = torch.tensor(block_ids, dtype=torch.long)
        num_blocks = block_ids_t.shape[0]
        offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            offsets.reshape((1, block_size))
            + block_ids_t.reshape((num_blocks, 1)) * block_size
        )
        return slot_mapping.flatten()

    def _gather_kv_heads(
        self, x: torch.Tensor, total_kv_heads: int
    ) -> torch.Tensor | None:
        """Gather a per-rank KV-head shard onto rank 0 of the TP group..

        Args:
            x: This rank's shard, shape (num_tokens, num_kv_heads_local, head_size).
            total_kv_heads: The unsharded KV-head count for this layer, used to
                dedup under GQA replication.

        Returns:
            (num_tokens, total_kv_heads, head_size) on rank 0; None on other ranks.
        """
        if self._tp_size == 1:
            return x

        gathered = tensor_model_parallel_gather(x, dst=0, dim=1)
        if gathered is None:
            return None

        stride = gathered.shape[1] // total_kv_heads
        if stride > 1:
            gathered = gathered[:, ::stride, :]

        return gathered.contiguous()

    def _submit_async_write(self, pending: PendingSave) -> None:
        num_tokens = pending.token_ids.shape[0]

        gathered_kv: dict[str, torch.Tensor] = {}
        if isinstance(pending, Gemma4PendingSave) and pending.local_block_ids:
            for role, cache, blocks, total_heads in (
                (
                    "local",
                    self._local_kv_cache,
                    pending.local_block_ids,
                    self._local_total_kv_heads,
                ),
                (
                    "global",
                    self._global_kv_cache,
                    pending.global_block_ids,
                    self._global_total_kv_heads,
                ),
            ):
                assert cache is not None
                assert blocks is not None
                slots = self._slot_mapping_from_blocks(blocks, cache.shape[2]).to(
                    device=cache.device, non_blocking=True
                )
                k, v = extract_real_kv_from_cache(cache, slots, num_tokens)
                # Returns None on non-rank-0; only rank 0 keeps the result.
                gk = self._gather_kv_heads(k, total_heads)
                gv = self._gather_kv_heads(v, total_heads)
                if gk is not None and gv is not None:
                    gathered_kv[f"kv_last_{role}_k"] = gk
                    gathered_kv[f"kv_last_{role}_v"] = gv

        # Only rank 0 writes to disk.
        if not self._is_tp_rank_zero:
            return
        assert self._kv_cache is not None
        if not gathered_kv:
            logger.warning(
                "Gemma4KVConnector: req %s has no verifier block_ids; "
                "writing hidden states only.",
                pending.req_id,
            )

        copy_stream = self._get_copy_stream()
        ready_event = torch.cuda.Event()
        ready_event.record()
        copy_stream.wait_event(ready_event)

        tensors: dict[str, torch.Tensor] = {}
        with torch.cuda.stream(copy_stream):
            hs_slots = self._slot_mapping_from_blocks(
                pending.block_ids, self._kv_cache.shape[1]
            ).to(device=self._kv_cache.device, non_blocking=True)
            tensors["hidden_states"] = self._pin(
                extract_from_kv_cache(self._kv_cache, hs_slots, num_tokens)
            )
            # Pin the already-gathered (full-head) verifier KV tensors.
            for key, full in gathered_kv.items():
                tensors[key] = self._pin(full)

        copy_done = torch.cuda.Event()
        copy_done.record(copy_stream)

        assert not pending.token_ids.is_cuda
        tensors["token_ids"] = pending.token_ids.clone()

        prior = self._req_futures.get(pending.req_id)
        assert prior is None, "Found another KV transfer request with same req_id!"

        os.makedirs(os.path.dirname(pending.filename), exist_ok=True)
        lock_fd = self._lock_fds.pop(pending.req_id, None)
        if lock_fd is None and self.use_lock:
            lock_path = pending.filename + ".lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

        future = self._executor.submit(
            self._write_tensors, tensors, copy_done, pending.filename, lock_fd
        )
        self._req_copy_events[pending.req_id] = copy_done
        self._req_futures[pending.req_id] = future
        future.add_done_callback(partial(self._on_write_done, pending.req_id))

    @staticmethod
    def _pin(gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Async DtoH copy into pinned host memory. Call inside the copy
        stream so the copy is ordered with the other per-request copies."""
        pinned = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)
        pinned.copy_(gpu_tensor, non_blocking=True)
        return pinned
