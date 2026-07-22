"""Mooncake-backed sibling of vLLM's ``ExampleHiddenStatesConnector``.

Stores the same ``{"hidden_states", "token_ids"}`` payload in a Mooncake store
keyed by request id instead of safetensors files, so the vLLM target and the
trainer don't need a shared filesystem. Loaded out-of-tree via
``kv_connector_module_path``; must be used with the ``extract_hidden_states``
speculative method.
"""

from __future__ import annotations

import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def extract_from_kv_cache(
    kv_cache: torch.Tensor, slot_mapping: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    block_size = kv_cache.shape[1]
    return kv_cache[slot_mapping // block_size, slot_mapping % block_size][:num_tokens]


def sanitize_key(key: str) -> str:
    """Make a request id safe to use as a Mooncake key."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    return ("k" + safe) if safe[:1].isdigit() else safe


@dataclass
class PendingSave:
    req_id: str
    mooncake_key: str
    token_ids: torch.Tensor
    block_ids: list[int]


@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[PendingSave] = field(default_factory=list)


class MooncakeHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """Stores extracted hidden states to a Mooncake distributed store."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Must be False so the drafter KV cache isn't merged with the verifier's.
        return False

    @classmethod
    def _find_cache_kv_group_id(cls, kv_cache_config: KVCacheConfig | None) -> int:
        """Index of the KV cache group holding the extracted hidden states.

        Located by spec type so it resolves on both scheduler and worker side.
        """
        if kv_cache_config is None:
            return 0

        from vllm.v1.kv_cache_interface import HiddenStateCacheSpec  # noqa: PLC0415

        groups = kv_cache_config.kv_cache_groups
        group_ids = [
            gid
            for gid, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        ]
        if len(group_ids) == 1:
            return group_ids[0]
        if not group_ids and len(groups) == 1:
            return 0
        raise ValueError(
            "Could not uniquely identify the extract-hidden-states KV cache "
            f"group among {len(groups)} groups; the hidden-states layer must be "
            "isolated in its own group (MLA verifiers are unsupported)."
        )

    @staticmethod
    def _get_cache_block_size(
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        cache_kv_group_id: int,
    ) -> int:
        """Block size of the hidden-states group, read from its own spec.

        cache_config.block_size is bumped to a common multiple for hybrid
        verifiers; the page-aligned hidden-states group keeps a smaller one.
        """
        if kv_cache_config is None:
            return vllm_config.cache_config.block_size
        cache_group = kv_cache_config.kv_cache_groups[cache_kv_group_id]
        return cache_group.kv_cache_spec.block_size

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        # Read the hidden-states group and its block size from the group spec;
        # cache_config.block_size is bumped (wrong) for hybrid verifiers.
        self._hs_group_idx = self._find_cache_kv_group_id(kv_cache_config)
        self._block_size = self._get_cache_block_size(
            vllm_config, kv_cache_config, self._hs_group_idx
        )

        if (
            self._vllm_config.speculative_config is None
            or self._vllm_config.speculative_config.method != "extract_hidden_states"
        ):
            raise ValueError(
                "MooncakeHiddenStatesConnector requires the "
                "'extract_hidden_states' speculative method"
            )

        mooncake_cfg = MooncakeStoreConfig.from_dict(
            self._kv_transfer_config.get_from_extra_config("mooncake", {})
        )
        self._store = MooncakeHiddenStatesStore(mooncake_cfg)

        # Scheduler-side state.
        self._request_keys: dict[str, str] = {}
        self._pending_saves: dict[str, PendingSave] = {}

        # Worker-side state (set in register_kv_caches).
        self._kv_cache: torch.Tensor | None = None
        self._is_tp_rank_zero: bool = True
        self._store_ready: bool = False
        # Dedicated CUDA stream for DtoH copies so they don't block
        # the default stream (model forward).
        self._copy_stream: torch.cuda.Stream | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=mooncake_cfg.num_writer_threads,
            thread_name_prefix="vllm-mooncake-hs",
        )
        self._req_futures: dict[str, Future] = {}
        self._accumulated_finished_req_ids: set[str] = set()

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        pass  # store-only

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # store-only

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        pass  # extraction happens in get_finished, once all tokens are done

    def wait_for_save(self) -> None:
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0

        from vllm.model_executor.models.extract_hidden_states import (  # noqa: PLC0415
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        cache_layers = list(layers.keys())
        assert len(cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(cache_layers)}"
        )
        self._kv_cache = kv_caches[cache_layers[0]]
        self._copy_stream = torch.cuda.Stream()

    def _get_copy_stream(self) -> torch.cuda.Stream:
        """Return the dedicated copy stream, lazily creating it if needed."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    def _ensure_store(self) -> None:
        if not self._store_ready:
            self._store.setup()
            self._store_ready = True

    def _write_sample(
        self, pending: PendingSave, ready_event: torch.cuda.Event
    ) -> None:
        assert self._kv_cache is not None

        copy_stream = self._get_copy_stream()
        # Make the copy stream wait until the forward pass has finished
        # writing to the KV cache (the event was recorded on the default
        # stream in get_finished).
        copy_stream.wait_event(ready_event)

        block_ids_t = torch.tensor(pending.block_ids, dtype=torch.long)
        num_blocks = block_ids_t.shape[0]
        block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_t.reshape((num_blocks, 1)) * self._block_size
        ).flatten()

        num_tokens = pending.token_ids.shape[0]

        with torch.cuda.stream(copy_stream):
            slot_mapping = slot_mapping.to(
                self._kv_cache.device, non_blocking=True
            )
            hidden_states = extract_from_kv_cache(
                self._kv_cache, slot_mapping, num_tokens
            )
            # Async DtoH copy into pinned host memory.
            pinned_hs = torch.empty_like(
                hidden_states, device="cpu", pin_memory=True
            )
            pinned_hs.copy_(hidden_states, non_blocking=True)

        # Wait for the DtoH copy to complete before handing data to the store.
        copy_stream.synchronize()

        self._store.put_sample(
            pending.mooncake_key,
            {"hidden_states": pinned_hs, "token_ids": pending.token_ids},
        )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # A request is reported done only after its put_sample completes, so a
        # consumer reading the returned key is guaranteed a hit.
        if self._is_tp_rank_zero and self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, MooncakeConnectorMetadata):
                for pending in metadata.pending_saves:
                    if pending.req_id in self._req_futures:
                        continue
                    self._ensure_store()
                    # Record an event on the current (default) stream so
                    # the worker thread can wait for the forward pass to
                    # finish writing to the KV cache before reading it.
                    ready_event = torch.cuda.Event()
                    ready_event.record()
                    self._req_futures[pending.req_id] = self._executor.submit(
                        self._write_sample, pending, ready_event
                    )

        self._accumulated_finished_req_ids.update(finished_req_ids)
        done_sending: set[str] = set()
        for req_id in list(self._accumulated_finished_req_ids):
            future = self._req_futures.get(req_id)
            # No future on non-rank-0 workers; treat as done.
            if future is None or future.done():
                if future is not None:
                    exc = future.exception()
                    if exc is not None:
                        logger.error("Mooncake write failed for %s: %r", req_id, exc)
                    self._req_futures.pop(req_id, None)
                done_sending.add(req_id)
                self._accumulated_finished_req_ids.discard(req_id)

        return done_sending or None, None

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: Request,  # noqa: ARG002 (KVConnector interface)
        num_computed_tokens: int,  # noqa: ARG002 (KVConnector interface)
    ) -> tuple[int | None, bool]:
        return 0, False  # store-only

    def update_state_after_alloc(
        self,
        request: Request,  # noqa: ARG002 (KVConnector interface)
        blocks: KVCacheBlocks,  # noqa: ARG002 (KVConnector interface)
        num_external_tokens: int,
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        meta.pending_saves = list(self._pending_saves.values())
        self._pending_saves.clear()

        for new_req in scheduler_output.scheduled_new_reqs:
            self._request_keys[new_req.req_id] = sanitize_key(new_req.req_id)
        return meta

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        mooncake_key = self._request_keys.pop(req_id, sanitize_key(req_id))

        kv_params = request.kv_transfer_params or {}
        if kv_params.get("include_output_tokens", False):
            # Drop the final token: it was output, never an input to a forward
            # pass, so its hidden state is not in the cache.
            token_ids = torch.tensor(list(request.all_token_ids)[:-1])
        elif request.prompt_token_ids is not None:
            token_ids = torch.tensor(request.prompt_token_ids)
        else:
            token_ids = torch.tensor([], dtype=torch.long)

        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            mooncake_key=mooncake_key,
            token_ids=token_ids,
            block_ids=list(block_ids),
        )
        # Returning True delays block freeing until get_finished extracts.
        return True, {"handle": mooncake_key}

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[self._hs_group_idx])

    @classmethod
    def get_required_kvcache_layout(
        cls,
        vllm_config: VllmConfig,  # noqa: ARG003 (KVConnector interface)
    ) -> str | None:
        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called on the base class"
            )
        # NHD keeps each token's hidden states contiguous in memory.
        return "NHD"
