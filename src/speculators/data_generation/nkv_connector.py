import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def parse_attn_layer_id(layer_name: str) -> int | None:
    if "model.layers." not in layer_name or ".self_attn" not in layer_name:
        return None
    part = layer_name.split("model.layers.")[1].split(".", maxsplit=1)[0]
    if not part.isnumeric():
        return None
    return int(part)


def _get_tp_info() -> tuple[int, int]:
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
    except (ImportError, AssertionError, RuntimeError):
        tp_size, tp_rank = 1, 0
    return tp_size, tp_rank


def extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    block_size = kv_cache.shape[-3]
    if kv_cache.dim() == 4:
        return kv_cache[slot_mapping // block_size, slot_mapping % block_size][
            :num_tokens
        ]
    if kv_cache.shape[0] == 2:
        return kv_cache[:, slot_mapping // block_size, slot_mapping % block_size][
            :, :num_tokens
        ]
    return kv_cache[slot_mapping // block_size, :, slot_mapping % block_size].transpose(
        0, 1
    )[:, :num_tokens]


def resolve_layer_slot_mapping(
    layer_name: str, attn_metadata: "AttentionMetadata"
) -> torch.Tensor:
    try:
        sm = get_forward_context().slot_mapping
        if isinstance(sm, dict):
            per_layer = sm.get(layer_name)
            if per_layer is not None:
                return per_layer
    except Exception:
        pass
    return attn_metadata.slot_mapping


@dataclass
class ReqMeta:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    new_req: bool

    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        new_req: bool,
    ) -> "ReqMeta":
        return ReqMeta(
            req_id=req_id,
            filename=filename,
            token_ids=torch.tensor(token_ids),
            new_req=new_req,
        )


@dataclass
class NKVConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        new_req: bool = True,
    ) -> None:
        self.requests.append(ReqMeta.make_meta(req_id, filename, token_ids, new_req))


class NKVDataConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        assert self._vllm_config.speculative_config is not None, (
            "NKVDataConnector only works when using "
            "'extract_hidden_states' speculative method"
        )
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        self._request_filenames: dict[str, str] = {}
        self._active_requests: dict[str, NewRequestData] = {}
        self._req_blocks: dict[str, list[int]] = {}

        self._layer_mapping: list[list[int]] | None = (
            self._kv_transfer_config.get_from_extra_config("layer_mapping", None)
        )

        if self._layer_mapping is not None:
            self._target_layer_ids = sorted(
                {lid for group in self._layer_mapping for lid in group}
            )
        else:
            self._target_layer_ids: list[int] = (
                self._kv_transfer_config.get_from_extra_config("target_layer_ids", [])
            )

        self._target_attn_layers: dict[int, str] = {}
        self._kv_buffers: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)

        logger.info("NKV target layer IDs: %s", self._target_layer_ids)
        if self._layer_mapping is not None:
            logger.info("NKV layer mapping: %s", self._layer_mapping)

        _tp = getattr(self._vllm_config.parallel_config, "tensor_parallel_size", 1)
        if _tp > 1:
            _mc = self._vllm_config.model_config
            _hf = getattr(_mc, "hf_text_config", None) or _mc.hf_config
            _kv_counts = [
                h
                for h in (
                    getattr(_hf, "num_key_value_heads", None),
                    getattr(_hf, "num_global_key_value_heads", None),
                )
                if h is not None
            ]
            if _kv_counts and min(_kv_counts) < _tp:
                raise NotImplementedError(
                    f"NKV TP head-gather does not support the KV-head-replication regime "
                    f"(min num_kv_heads {min(_kv_counts)} < tensor_parallel_size {_tp}): vLLM "
                    f"replicates KV heads there, so the gather would duplicate them and save a "
                    f"wrong head count. Serve data-gen with tensor_parallel_size <= num_kv_heads."
                )

    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )

        for layer_name in kv_caches:
            layer_id = parse_attn_layer_id(layer_name)
            if layer_id is not None and layer_id in self._target_layer_ids:
                self._target_attn_layers[layer_id] = layer_name

        logger.info(
            "NKV registered %d target attention layers: %s",
            len(self._target_attn_layers),
            self._target_attn_layers,
        )
        missing = set(self._target_layer_ids) - set(self._target_attn_layers.keys())
        if missing:
            logger.warning("NKV: requested layer IDs %s not found in model", missing)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name in self._target_attn_layers.values():
            self._buffer_kv_layer(layer_name, kv_layer, attn_metadata)
            return

        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata), (
            "NKVDataConnector only supports CacheOnlyAttentionBackend"
        )

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, NKVConnectorMetadata)

        os.makedirs(self._storage_path, exist_ok=True)

        _, tp_rank = _get_tp_info()

        slot_mapping = resolve_layer_slot_mapping(layer_name, attn_metadata)
        offset = 0
        for request in connector_metadata.requests:
            num_tokens = request.token_ids.shape[0]

            kv_buffer = self._kv_buffers.pop(request.req_id, {})

            if tp_rank != 0:
                offset += num_tokens
                continue

            req_slot_mapping = slot_mapping[offset : offset + num_tokens]
            offset += num_tokens

            hidden_states = extract_from_kv_cache(
                kv_layer, req_slot_mapping, num_tokens
            )
            tensors: dict[str, torch.Tensor] = {
                "hidden_states": hidden_states.detach().cpu(),
                "token_ids": request.token_ids.detach().cpu(),
            }

            if kv_buffer:
                tensors["kv_caches"] = torch.stack(
                    [kv_buffer[lid] for lid in self._target_layer_ids]
                )

            safetensors.torch.save_file(tensors, request.filename)

    def _buffer_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None:
        layer_id = None
        for lid, lname in self._target_attn_layers.items():
            if lname == layer_name:
                layer_id = lid
                break
        assert layer_id is not None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, NKVConnectorMetadata)

        tp_size, _ = _get_tp_info()

        slot_mapping = resolve_layer_slot_mapping(layer_name, attn_metadata)
        offset = 0
        for request in connector_metadata.requests:
            num_tokens = request.token_ids.shape[0]
            req_slot_mapping = slot_mapping[offset : offset + num_tokens]
            offset += num_tokens

            kv_data = extract_from_kv_cache(kv_layer, req_slot_mapping, num_tokens)
            if tp_size > 1:
                from vllm.distributed.parallel_state import get_tp_group

                kv_data = get_tp_group().all_gather(kv_data.contiguous(), dim=2)
            self._kv_buffers[request.req_id][layer_id] = kv_data.detach().cpu()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NKVConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            n_sched = getattr(scheduler_output, "num_scheduled_tokens", {}).get(
                new_req.req_id
            )
            if n_sched is not None and n_sched < len(token_ids):
                raise RuntimeError(
                    f"NKVDataConnector: request {new_req.req_id} scheduled {n_sched} < "
                    f"prompt {len(token_ids)} tokens (prefix-cache hit or chunked "
                    f"prefill) — saved rows would be position-shifted. Launch data-gen "
                    f"with --no-enable-prefix-caching and --no-enable-chunked-prefill."
                )
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
            )
            self._request_filenames[new_req.req_id] = filename
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]

            cached_req = self._active_requests[req_id]
            req_block_ids = self._req_blocks[req_id]

            if new_block_ids is None:
                continue

            block_ids = new_block_ids[0]

            req_block_ids.extend(block_ids)
            filename = os.path.join(self._storage_path, f"{req_id}.safetensors")

            meta.add_request(
                req_id=req_id,
                filename=filename,
                token_ids=cached_req.prompt_token_ids or [],
                new_req=False,
            )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)
        _ = self._kv_buffers.pop(req_id, None)

        return False, {"hidden_states_path": req_filename}

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[0])

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        return "NHD"
