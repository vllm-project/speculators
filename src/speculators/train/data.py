import hashlib
import json
import math
import os
import random
import threading
import uuid
import warnings
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import load_from_disk
from safetensors.torch import save_file
from torch.utils.data import Dataset

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.artifact_cache import (
    ArtifactCacheError,
    HiddenStateArtifactCache,
    canonical_hidden_state_request_id,
)
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
)
from speculators.data_generation.windowed_artifacts import (
    ArtifactReadLease,
    GenerationClaim,
    StreamSampleIndex,
    WindowedArtifactCoordinator,
    canonical_stream_id,
)
from speculators.train.noise_transforms import TransformTensors

BatchType = dict[str, Any]
WINDOWED_LEASE_KEY = "_windowed_artifact_lease"
WINDOWED_BATCH_LEASES_KEY = "_windowed_artifact_leases"


def _validate_integer_config(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _validate_non_negative_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            if not file.endswith("pt"):
                continue
            file_path = Path(root) / file
            datapath.append(file_path)

    return datapath


def slice_and_pad_to_length(tensor, length):
    sliced_tensor = tensor[:length]
    padding = [0, 0] * sliced_tensor.dim()
    padding[-1] = length - sliced_tensor.shape[0]
    return F.pad(sliced_tensor, padding)


def split_files(datapath: str, ratio: float = 0.9, seed: int = 0):
    """Given a datapath, split the files into a training and validation set
    ratio is the proportion of files to put in the training set
    1 - ratio is the proportion of files to put in the validation set
    """
    random.seed(seed)
    file_list = list_files(datapath)
    random.shuffle(file_list)
    num_files = len(file_list)
    num_train_files = int(num_files * ratio)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]
    return train_files, val_files


# Data standardization functions
StandardizeFnSig = Callable[[dict[str, Any]], dict[str, Any]]


def create_empty_sample(
    hidden_size: int, num_target_layers: int = 3, dtype: torch.dtype = torch.bfloat16
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def standardize_data_v1(data: dict[str, Any]) -> dict[str, Any]:
    # v1 data format:
    # {
    #  "input_ids": [seq_len],
    #  "loss_mask": [seq_len],
    #  "hidden_states": [
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    ...
    #  ],
    # }

    return {
        "hidden_states": torch.cat(data["hidden_states"][:-1], dim=-1),
        "input_ids": data["input_ids"],
        "verifier_last_hidden_states": data["hidden_states"][-1],
        "loss_mask": data["loss_mask"],
    }


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


def build_client_item(dataset_item: dict) -> ClientItem:
    """Build a request payload for vLLM hidden-state extraction.

    When ``messages`` is included, ``generate_hidden_states`` uses the Chat
    Completions API and vLLM **re-tokenizes from the raw messages**, ignoring
    ``input_ids``.  This is required for multimodal inputs (the Completions
    API cannot carry image/video/audio references), but harmful for text-only
    data: preprocessing truncates ``input_ids`` to ``seq_length``, yet the
    ``messages`` column stores the original un-truncated conversation.
    Re-tokenizing those messages produces a longer sequence that can exceed
    ``max_model_len``.

    We therefore only forward ``messages`` when the conversation actually
    contains multimodal content.  Text-only conversations always go through
    the Completions API with the pre-truncated ``input_ids``.

    This matters for models like Qwen3.5-0.8B whose ``AutoProcessor`` returns
    a ``ProcessorMixin`` (``Qwen3VLProcessor``), causing preprocessing to
    populate the ``messages`` column even for purely text-only datasets.
    Text-only EAGLE-3 models (e.g. Llama) use a plain tokenizer, so
    ``messages`` is never created and this guard is a no-op.
    """
    out_dict: dict = {"input_ids": dataset_item["input_ids"].tolist()}

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]

    return cast("ClientItem", out_dict)


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
    ):
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def _compute_approx_lengths(self):
        raise NotImplementedError

    def _get_raw_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index) -> BatchType | None:
        data = self._get_raw_data(index)

        if data is None:
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Convert hidden states to the correct dtype
        data = {
            k: v.to(self.hidden_states_dtype) if "hidden_states" in k else v
            for k, v in data.items()
        }

        # Add lengths tensor
        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        # shape: [1]

        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        # shape: [seq_len]

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        #     "position_ids": [seq_len],
        # }

        # Apply transform
        if self.transform:
            data = self.transform(data)

        return data


def _atomic_save_hs_file(data: dict[str, torch.Tensor], file_path: Path) -> None:
    temporary = file_path.parent / (
        f".{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        save_file(data, temporary)
        temporary.replace(file_path)
    finally:
        temporary.unlink(missing_ok=True)


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        transfer: HiddenStatesTransfer | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        split_ratio: float = 1.0,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        shared_artifacts_path: str | PathLike | None = None,
        shared_artifacts_namespace: str | None = None,
        shared_artifacts_ttl_seconds: float | None = 3600.0,
        shared_artifacts_lock_timeout_seconds: float = 300.0,
        shared_artifacts_consumer_id: str | None = None,
        shared_artifacts_lookbehind: int = 2,
        shared_artifacts_lookahead: int = 40,
        shared_artifacts_max_prefetch_per_consumer: int = 8,
        shared_artifacts_capture_batch_size: int = 8,
        shared_artifacts_capture_batch_wait_seconds: float = 0.002,
        shared_artifacts_max_inflight: int = 32,
        shared_artifacts_consumer_timeout_seconds: float = 120.0,
        shared_artifacts_claim_timeout_seconds: float = 300.0,
        shared_artifacts_generation_attempts: int = 3,
    ):
        self.data = load_from_disk(datapath)
        self.start_file_idx = 0
        if split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            self.start_file_idx = 0
            split_idx = int(len(self.data) * split_ratio)
            self.data = self.data.select(range(split_idx))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.start_file_idx = split_idx
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0.")

        self.transfer = transfer or FileTransfer(Path(datapath) / "hidden_states")
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        self.client: openai.OpenAI | None = None
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.shared_artifacts_namespace = shared_artifacts_namespace
        if shared_artifacts_path is not None and not shared_artifacts_namespace:
            raise ValueError(
                "shared_artifacts_namespace is required when shared artifacts "
                "are enabled"
            )
        self.artifact_cache = (
            HiddenStateArtifactCache(
                shared_artifacts_path,
                artifact_ttl_seconds=(
                    None
                    if shared_artifacts_consumer_id is not None
                    else shared_artifacts_ttl_seconds
                ),
                lock_timeout_seconds=shared_artifacts_lock_timeout_seconds,
            )
            if shared_artifacts_path is not None
            else None
        )
        self.shared_artifacts_path = (
            Path(shared_artifacts_path).expanduser().resolve()
            if shared_artifacts_path is not None
            else None
        )
        self.shared_artifacts_consumer_id = shared_artifacts_consumer_id
        self.shared_artifacts_lock_timeout_seconds = (
            shared_artifacts_lock_timeout_seconds
        )
        self.shared_artifacts_lookbehind = shared_artifacts_lookbehind
        self.shared_artifacts_lookahead = shared_artifacts_lookahead
        _validate_integer_config(
            "shared_artifacts_max_prefetch_per_consumer",
            shared_artifacts_max_prefetch_per_consumer,
            minimum=0,
        )
        if shared_artifacts_max_prefetch_per_consumer > shared_artifacts_lookahead + 1:
            raise ValueError(
                "shared_artifacts_max_prefetch_per_consumer must not exceed "
                "shared_artifacts_lookahead + 1"
            )
        _validate_integer_config(
            "shared_artifacts_capture_batch_size",
            shared_artifacts_capture_batch_size,
            minimum=1,
        )
        _validate_non_negative_number(
            "shared_artifacts_capture_batch_wait_seconds",
            shared_artifacts_capture_batch_wait_seconds,
        )
        self.shared_artifacts_max_prefetch_per_consumer = (
            shared_artifacts_max_prefetch_per_consumer
        )
        self.shared_artifacts_capture_batch_size = shared_artifacts_capture_batch_size
        self.shared_artifacts_capture_batch_wait_seconds = float(
            shared_artifacts_capture_batch_wait_seconds
        )
        self.shared_artifacts_max_inflight = shared_artifacts_max_inflight
        self.shared_artifacts_consumer_timeout_seconds = (
            shared_artifacts_consumer_timeout_seconds
        )
        self.shared_artifacts_claim_timeout_seconds = (
            shared_artifacts_claim_timeout_seconds
        )
        self.shared_artifacts_generation_attempts = shared_artifacts_generation_attempts
        self.windowed_artifacts_enabled = shared_artifacts_consumer_id is not None
        if self.windowed_artifacts_enabled and self.artifact_cache is None:
            raise ValueError(
                "shared_artifacts_consumer_id requires shared_artifacts_path"
            )
        if self.windowed_artifacts_enabled and self.on_missing != "generate":
            raise ValueError("windowed artifacts require on_missing='generate'")
        self._windowed_stream_id: str | None = None
        self._windowed_coordinator: WindowedArtifactCoordinator | None = None
        self._windowed_coordinator_pid: int | None = None
        self._windowed_producer_thread: threading.Thread | None = None
        self._windowed_producer_stop: threading.Event | None = None
        self._windowed_producer_error: Exception | None = None
        if self.artifact_cache is not None and not self.windowed_artifacts_enabled:
            self.artifact_cache.cleanup_stale()

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def _map_to_file_idx(self, index: int):
        return index + self.start_file_idx

    def _setup_client(self):
        self.client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = self.client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id
        self.transfer.setup()

    def __len__(self):
        return len(self.data)

    def _new_windowed_coordinator(self) -> WindowedArtifactCoordinator:
        if self.shared_artifacts_path is None:
            raise RuntimeError("windowed artifacts are not configured")
        return WindowedArtifactCoordinator(
            self.shared_artifacts_path,
            consumer_timeout_seconds=self.shared_artifacts_consumer_timeout_seconds,
            claim_timeout_seconds=self.shared_artifacts_claim_timeout_seconds,
            max_generation_attempts=self.shared_artifacts_generation_attempts,
        )

    def _coordinator_for_process(self) -> WindowedArtifactCoordinator:
        pid = os.getpid()
        if self._windowed_coordinator_pid != pid:
            if self._windowed_coordinator is not None:
                self._windowed_coordinator.close()
            self._windowed_coordinator = self._new_windowed_coordinator()
            self._windowed_coordinator_pid = pid
        if self._windowed_coordinator is None:
            raise RuntimeError("windowed artifact coordinator is unavailable")
        return self._windowed_coordinator

    def configure_windowed_stream(self, sampler: Any) -> str:
        """Bind this dataset split to a deterministic sampler-order contract."""
        if not self.windowed_artifacts_enabled:
            raise RuntimeError("windowed artifacts are not enabled")
        lengths_digest = hashlib.sha256()
        for length in sampler.lengths:
            lengths_digest.update(f"{int(length)}\n".encode())
        contract = {
            "batch_max_length": int(sampler.batch_max_length),
            "dataset_fingerprint": str(
                getattr(self.data, "_fingerprint", "unavailable")
            ),
            "dataset_length": len(self.data),
            "dp_rank": int(sampler.rank),
            "dp_size": int(sampler.num_replicas),
            "lengths_digest": lengths_digest.hexdigest(),
            "namespace": self.shared_artifacts_namespace,
            "order": "MultipackDistributedBatchSamplerV2",
            "sampler_seed": int(sampler.seed),
            "schema_version": 1,
            "verifier_model": self.model,
        }
        stream_id = canonical_stream_id(contract)
        with self._new_windowed_coordinator() as coordinator:
            observed = coordinator.register_stream(contract)
        if observed != stream_id:
            raise RuntimeError("coordinator returned an inconsistent stream identity")
        self._windowed_stream_id = stream_id
        return stream_id

    def windowed_request_id(self, dataset_index: int) -> str:
        if not self.model:
            raise RuntimeError("windowed artifacts require an explicit verifier model")
        dataset_item = self.data[dataset_index]
        return canonical_hidden_state_request_id(
            self.model,
            build_client_item(dataset_item),
            namespace=self.shared_artifacts_namespace,
        )

    def prepare_windowed_epoch(
        self,
        samples: tuple[StreamSampleIndex, ...],
        *,
        cursor: int,
        reset: bool,
    ) -> None:
        if not self.windowed_artifacts_enabled:
            return
        if (
            self._windowed_stream_id is None
            or self.shared_artifacts_consumer_id is None
        ):
            raise RuntimeError("windowed stream was not configured by the DataLoader")
        if samples and any(
            sample.stream_id != self._windowed_stream_id for sample in samples
        ):
            raise RuntimeError("sampler positions belong to another stream")
        with self._new_windowed_coordinator() as coordinator:
            coordinator.register_positions(samples)
            coordinator.register_consumer(
                self.shared_artifacts_consumer_id,
                stream_id=self._windowed_stream_id,
                lookbehind=self.shared_artifacts_lookbehind,
                lookahead=self.shared_artifacts_lookahead,
                max_prefetch=self.shared_artifacts_max_prefetch_per_consumer,
                max_inflight=self.shared_artifacts_max_inflight,
                cursor=cursor,
                reset=reset,
            )

    def _acquire_windowed_hs(
        self, sample: StreamSampleIndex
    ) -> tuple[dict[str, torch.Tensor], ArtifactReadLease]:
        if self.shared_artifacts_consumer_id is None or self.artifact_cache is None:
            raise RuntimeError("windowed artifacts are not configured")
        coordinator = self._coordinator_for_process()
        lease = coordinator.acquire(
            self.shared_artifacts_consumer_id,
            sample,
            timeout_seconds=self.request_timeout,
        )
        dataset_item = self.data[sample.dataset_index]
        try:
            loaded = self.artifact_cache.load(
                sample.request_id,
                lambda data: check_hidden_states(
                    data, dataset_item["input_ids"].tolist()
                ),
            )
            if lease.cache_hit:
                self.artifact_cache.record_reuse()
            return loaded, lease
        except BaseException:
            coordinator.abandon(
                self.shared_artifacts_consumer_id, [lease.as_batch_metadata()]
            )
            raise

    def ack_windowed_batch(self, leases: list[dict[str, Any]]) -> int | None:
        if not self.windowed_artifacts_enabled or not leases:
            return None
        if self.shared_artifacts_consumer_id is None:
            raise RuntimeError("windowed consumer identity is missing")
        return self._coordinator_for_process().ack(
            self.shared_artifacts_consumer_id, leases
        )

    def abandon_windowed_batch(self, leases: list[dict[str, Any]]) -> None:
        if not self.windowed_artifacts_enabled or not leases:
            return
        if self.shared_artifacts_consumer_id is None:
            raise RuntimeError("windowed consumer identity is missing")
        self._coordinator_for_process().abandon(
            self.shared_artifacts_consumer_id, leases
        )

    def start_windowed_producer(self) -> None:
        """Start a trainer-main-process dispatcher after DataLoader workers fork."""
        if not self.windowed_artifacts_enabled:
            return
        if self._windowed_producer_thread is not None:
            if not self._windowed_producer_thread.is_alive():
                error = self._windowed_producer_error
                raise RuntimeError("windowed artifact producer stopped") from error
            return
        if self._windowed_stream_id is None:
            raise RuntimeError("windowed stream is not prepared")
        self._windowed_producer_stop = threading.Event()
        self._windowed_producer_error = None
        self._windowed_producer_thread = threading.Thread(
            target=self._run_windowed_producer,
            name=f"artifact-producer-{self.shared_artifacts_consumer_id}",
            daemon=True,
        )
        self._windowed_producer_thread.start()

    def _run_windowed_producer(self) -> None:
        if (
            self.shared_artifacts_path is None
            or self.shared_artifacts_consumer_id is None
            or self._windowed_stream_id is None
            or self._windowed_producer_stop is None
        ):
            self._windowed_producer_error = RuntimeError(
                "windowed producer started without a complete configuration"
            )
            return
        owner = f"{self.shared_artifacts_consumer_id}:{os.getpid()}:{uuid.uuid4().hex}"
        cache = HiddenStateArtifactCache(
            self.shared_artifacts_path,
            artifact_ttl_seconds=None,
            lock_timeout_seconds=self.shared_artifacts_lock_timeout_seconds,
        )
        try:
            if self.client is None:
                self._setup_client()
            with (
                self._new_windowed_coordinator() as coordinator,
                ThreadPoolExecutor(
                    max_workers=self.shared_artifacts_capture_batch_size,
                    thread_name_prefix="artifact-capture",
                ) as executor,
            ):
                while not self._windowed_producer_stop.is_set():
                    coordinator.heartbeat(self.shared_artifacts_consumer_id)
                    coordinator.recover_expired()
                    self._evict_windowed_artifacts(coordinator, cache)
                    if self._windowed_producer_stop.wait(
                        self.shared_artifacts_capture_batch_wait_seconds
                    ):
                        break
                    claims = coordinator.claim_generation(
                        owner,
                        stream_id=self._windowed_stream_id,
                        max_claims=self.shared_artifacts_capture_batch_size,
                        max_active_claims=self.shared_artifacts_capture_batch_size,
                    )
                    if claims:
                        futures = {
                            executor.submit(
                                self._produce_windowed_claim,
                                coordinator,
                                cache,
                                owner,
                                claim,
                            ): claim
                            for claim in claims
                        }
                        pending = set(futures)
                        while pending:
                            done, pending = wait(
                                pending,
                                timeout=1.0,
                                return_when=FIRST_COMPLETED,
                            )
                            coordinator.heartbeat(self.shared_artifacts_consumer_id)
                            for future in done:
                                claim = futures[future]
                                try:
                                    future.result()
                                except Exception as error:  # noqa: BLE001
                                    coordinator.fail_generation(owner, claim, error)
                        continue
                    self._windowed_producer_stop.wait(0.02)
        except Exception as error:  # noqa: BLE001 - background thread boundary
            self._windowed_producer_error = error

    def _produce_windowed_claim(
        self,
        coordinator: WindowedArtifactCoordinator,
        cache: HiddenStateArtifactCache,
        owner: str,
        claim: GenerationClaim,
    ) -> None:
        expected_request_id = self.windowed_request_id(claim.dataset_index)
        if expected_request_id != claim.request_id:
            raise RuntimeError("generation claim no longer matches dataset")
        dataset_item = self.data[claim.dataset_index]
        client_item = build_client_item(dataset_item)
        result = cache.get_or_create(
            claim.request_id,
            lambda: self._materialize_shared_hs(
                claim.dataset_index,
                dataset_item,
                client_item,
            ),
            lambda data: check_hidden_states(data, dataset_item["input_ids"].tolist()),
        )
        coordinator.complete_generation(
            owner,
            claim,
            path=result.path,
            size_bytes=result.path.stat().st_size,
        )

    @staticmethod
    def _evict_windowed_artifacts(
        coordinator: WindowedArtifactCoordinator,
        cache: HiddenStateArtifactCache,
    ) -> None:
        for eviction in coordinator.begin_evictions(limit=16):
            try:
                removed = cache.remove(eviction.request_id, expected_path=eviction.path)
            except (ArtifactCacheError, OSError):
                coordinator.finish_eviction(eviction, removed=False)
            else:
                coordinator.finish_eviction(
                    eviction, removed=removed or not eviction.path.exists()
                )

    def _materialize_shared_hs(
        self,
        index: int,
        dataset_item: dict,
        client_item: ClientItem,
    ) -> dict[str, torch.Tensor]:
        file_idx = self._map_to_file_idx(index)
        cached = self.transfer.get_cached(file_idx)
        if cached is not None:
            check_hidden_states(cached, dataset_item["input_ids"].tolist())
            return cached
        if not self.client:
            self._setup_client()
        return self._generate_shared_hs(dataset_item, client_item)

    def stop_windowed_producer(self, *, completed: bool = False) -> None:
        stop = self._windowed_producer_stop
        thread = self._windowed_producer_thread
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive():
                raise RuntimeError("windowed artifact producer did not stop")
        self._windowed_producer_thread = None
        self._windowed_producer_stop = None
        if completed and self.shared_artifacts_consumer_id is not None:
            coordinator = self._coordinator_for_process()
            coordinator.complete_consumer(self.shared_artifacts_consumer_id)
            if self.artifact_cache is not None:
                self._evict_windowed_artifacts(coordinator, self.artifact_cache)
        if self._windowed_producer_error is not None:
            error = self._windowed_producer_error
            self._windowed_producer_error = None
            raise RuntimeError("windowed artifact producer failed") from error

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor] | None:
        if not self.client:
            self._setup_client()

        dataset_item = self.data[index]
        client_item = build_client_item(dataset_item)

        try:
            if self.artifact_cache is not None:
                request_id = canonical_hidden_state_request_id(
                    self.model,  # type:ignore[arg-type]
                    client_item,
                    namespace=self.shared_artifacts_namespace,
                )
                result = self.artifact_cache.get_or_create(
                    request_id,
                    lambda: self._generate_shared_hs(dataset_item, client_item),
                    lambda data: check_hidden_states(
                        data, dataset_item["input_ids"].tolist()
                    ),
                )
                loaded_hs = result.data
                if self.on_generate == "cache" and isinstance(
                    self.transfer, FileTransfer
                ):
                    file_idx = self._map_to_file_idx(index)
                    target_path = (
                        self.transfer.hidden_states_path / f"hs_{file_idx}.safetensors"
                    )
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_save_hs_file(loaded_hs, target_path)
                return loaded_hs

            handle = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(handle)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            check_hidden_states(loaded_hs, dataset_item["input_ids"].tolist())

            file_idx = self._map_to_file_idx(index)
            match self.on_generate:
                case "cache":
                    self.transfer.cache(handle, file_idx)
                case "delete":
                    self.transfer.delete(handle)
        except Exception as e:
            if isinstance(e, ValueError) and "NaN" in str(e):
                raise
            warnings.warn(
                f"Failed to load/cache hidden states for sample {index}: {e}",
                stacklevel=1,
            )
            return None

        return loaded_hs

    def _generate_shared_hs(
        self, dataset_item: dict, client_item: ClientItem
    ) -> dict[str, torch.Tensor]:
        handle: str | None = None
        cleanup_generated = False
        try:
            handle = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
            cleanup_generated = True
            try:
                loaded_hs = self.transfer.get_generated(handle)
            except TimeoutError:
                cleanup_generated = False
                raise
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")
            check_hidden_states(loaded_hs, dataset_item["input_ids"].tolist())
            return loaded_hs
        finally:
            # A failed retrieval may still have an in-flight backend writer.
            if handle is not None and cleanup_generated:
                self.transfer.delete(handle)

    def _load_requested_hidden_states(
        self,
        dataset_index: int,
        windowed_sample: StreamSampleIndex | None,
    ) -> tuple[dict[str, torch.Tensor] | None, ArtifactReadLease | None]:
        lease: ArtifactReadLease | None = None
        file_idx = self._map_to_file_idx(dataset_index)
        if windowed_sample is not None:
            loaded_hs, lease = self._acquire_windowed_hs(windowed_sample)
        else:
            loaded_hs = self.transfer.get_cached(file_idx)

        if loaded_hs is None:
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(dataset_index)
                case "skip":
                    return None, None
                case "warn":
                    warnings.warn(
                        "Failed to load hidden states for sample "
                        f"{dataset_index}. Skipping...",
                        stacklevel=1,
                    )
                    return None, None
                case "raise":
                    raise RuntimeError(
                        f"Failed to load hidden states for sample {dataset_index}."
                    )
        return loaded_hs, lease

    def _get_raw_data(self, index):
        windowed_sample = index if isinstance(index, StreamSampleIndex) else None
        dataset_index = (
            windowed_sample.dataset_index if windowed_sample is not None else int(index)
        )
        loaded_hs, lease = self._load_requested_hidden_states(
            dataset_index, windowed_sample
        )

        if loaded_hs is None:
            return None

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        if not torch.equal(
            loaded_hs["token_ids"], self.data[dataset_index]["input_ids"]
        ):
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index "
                f"{dataset_index} don't match input ids "
                f"{self.data[dataset_index]['input_ids']}",
                stacklevel=1,
            )
            if lease is not None and self.shared_artifacts_consumer_id is not None:
                self._coordinator_for_process().abandon(
                    self.shared_artifacts_consumer_id,
                    [lease.as_batch_metadata()],
                )
            return None

        result = {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": self.data[dataset_index]["loss_mask"],  # [seq_len]
        }
        if lease is not None:
            result[WINDOWED_LEASE_KEY] = lease.as_batch_metadata()
        return result


class SampleFileDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform: TransformTensors | None = None,
        hidden_states_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the SampleFileDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory. All `.pt` files in this directory
            or its subdirectories will be loaded and used as training data. MUTUALLY
            EXCLUSIVE with `file_list`.
            file_list: The list of explict file paths to load data from. These files
            must be in the format produced by the Speculators generation scripts.
            MUTUALLY EXCLUSIVE with `datapath`.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
            standardize_fn: The function to standardize the data.

            Note: datapath or file_list must be provided, but not both.

        """

        if datapath is not None and file_list is not None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        if datapath is not None:
            file_list = list_files(datapath)

        if file_list is None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        self.data: list[str] = file_list

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples.

        First tries to load exact lengths from sample_lengths.json if available.
        Falls back to approximation based on file sizes.
        """
        # Look for the sample_lengths.json file
        sample_lengths_path = Path(self.data[0]).parent / "sample_lengths.json"
        if sample_lengths_path.exists():
            try:
                with sample_lengths_path.open() as f:
                    sample_lengths = json.load(f)
                # Extract file index from filename (e.g., data_42.pt -> 42)
                lengths = []
                for fname in self.data:
                    file_stem = Path(fname).stem
                    file_idx = file_stem.split("_")[-1]
                    lengths.append(sample_lengths[file_idx])
                return lengths
            except (KeyError, ValueError):
                pass

        # Fallback: approximate lengths from file sizes
        item_0 = self.__getitem__(0)
        if item_0 is None:
            raise ValueError(
                "Failed to load first element of datasets for length approximation"
            )
        lengths_0 = item_0["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def _get_raw_data(self, index):
        return standardize_data_v1(
            torch.load(
                self.data[index], mmap=True, weights_only=True, map_location="cpu"
            )
        )


def create_collate_fn(
    max_len: int,
    hidden_size: int,
    num_target_layers: int = 3,
    dtype: torch.dtype = torch.bfloat16,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        # Lease metadata stays on CPU and is never passed through model preprocessing.
        valid_batch = [sample for sample in batch if sample is not None]
        leases = [
            sample.pop(WINDOWED_LEASE_KEY)
            for sample in valid_batch
            if WINDOWED_LEASE_KEY in sample
        ]
        # Apply per-sample preprocessing and filter failed samples
        batch = [preprocess(b) if preprocess else b for b in valid_batch]

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(hidden_size, num_target_layers, dtype=dtype)
            if preprocess:
                empty = preprocess(empty)
            batch = [empty]

        collated_data = {}
        for key in batch[0]:  # type: ignore[union-attr]
            # Concatenate the tensors along the seq (0th) dimension
            collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
            # shape: [total_seq_len, ...]

            if key != "lengths":
                # Slice and pad on seq (0th) dimension to max_len
                collated_data[key] = slice_and_pad_to_length(
                    collated_data[key], max_len
                ).unsqueeze(0)
                # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data.pop("lengths")
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        lengths = torch.tensor(new_lengths, dtype=torch.long)

        # Create document_ids: maps each position to its document index, -1 for padding
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long), lengths
        )
        document_ids = torch.cat(
            [
                document_ids,
                -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
            ]
        ).unsqueeze(0)
        # shape: [1, max_len]
        collated_data["document_ids"] = document_ids
        if leases:
            collated_data[WINDOWED_BATCH_LEASES_KEY] = leases

        return collated_data

    return collate_fn
