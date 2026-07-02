"""Mooncake-backed sibling of vLLM's ``ExampleHiddenStatesConnector``.

Stores the same ``{"hidden_states", "token_ids"}`` payload in a Mooncake store
keyed by request id instead of safetensors files, so the vLLM target and the
trainer don't need a shared filesystem. Loaded out-of-tree via
``kv_connector_module_path``; must be used with the ``extract_hidden_states``
speculative method.
"""

from __future__ import annotations

import math
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

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._block_size = vllm_config.cache_config.block_size

        assert self._vllm_config.speculative_config is not None, (
            "MooncakeHiddenStatesConnector requires the 'extract_hidden_states' "
            "speculative method"
        )

        raw_mooncake_cfg = self._kv_transfer_config.get_from_extra_config(
            "mooncake", {}
        )
        mooncake_cfg = MooncakeStoreConfig.from_dict(raw_mooncake_cfg)
        self._transfer_buffer_user_set = "transfer_buffer_size" in raw_mooncake_cfg
        self._store = MooncakeHiddenStatesStore(mooncake_cfg)

        # Scheduler-side state.
        self._request_keys: dict[str, str] = {}
        self._pending_saves: dict[str, PendingSave] = {}

        # Worker-side state (set in register_kv_caches).
        self._kv_cache: torch.Tensor | None = None
        self._hs_group_idx: int = 0
        self._is_tp_rank_zero: bool = True
        self._store_ready: bool = False
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

        if not self._transfer_buffer_user_set:
            # Size transfer buffers for a max_model_len sample (hidden states
            # + token ids + header): undersized buffers make long requests
            # fail to write, leaving consumers a dead key. Costs
            # transfer_pool_size x this size in pinned host memory.
            per_token_bytes = (
                math.prod(self._kv_cache.shape[2:]) * self._kv_cache.element_size()
                + 8  # int64 token id
            )
            max_len = self._vllm_config.model_config.max_model_len
            self._store.config.transfer_buffer_size = max_len * per_token_bytes + 4096
            logger.info(
                "Auto-sized mooncake transfer buffers to %.1f MB "
                "(max_model_len=%d x %d B/token; pool of %d)",
                self._store.config.transfer_buffer_size / 1e6,
                max_len,
                per_token_bytes,
                self._store.config.transfer_pool_size,
            )

        if self._kv_cache_config is not None:
            for i, group in enumerate(self._kv_cache_config.kv_cache_groups):
                if cache_layers[0] in group.layer_names:
                    self._hs_group_idx = i
                    break

    def _ensure_store(self) -> None:
        if not self._store_ready:
            self._store.setup()
            self._store_ready = True

    def _write_sample(self, pending: PendingSave) -> None:
        assert self._kv_cache is not None
        block_ids_t = torch.tensor(pending.block_ids, dtype=torch.long)
        num_blocks = block_ids_t.shape[0]
        block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_t.reshape((num_blocks, 1)) * self._block_size
        ).flatten()

        num_tokens = pending.token_ids.shape[0]
        slot_mapping = slot_mapping.to(self._kv_cache.device)
        # Keep the gathered hidden states on the GPU: put_sample stages them
        # straight into its pinned transfer buffer on a dedicated copy stream,
        # avoiding a blocking default-stream DtoH plus a pageable-memory hop.
        hidden_states = extract_from_kv_cache(self._kv_cache, slot_mapping, num_tokens)

        self._store.put_sample(
            pending.mooncake_key,
            {"hidden_states": hidden_states, "token_ids": pending.token_ids},
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
                    self._req_futures[pending.req_id] = self._executor.submit(
                        self._write_sample, pending
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
        return True, {"mooncake_key": mooncake_key}

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
