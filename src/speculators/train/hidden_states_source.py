"""Hidden-state acquisition for training data."""

import logging
import warnings
from typing import Any, Literal, Protocol, cast

import openai
import torch

from hs_connectors import HiddenStatesTransfer
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
)
from speculators.train.data_types import DatasetRecord
from speculators.train.recovery import GenerationRecoveryGuard, SampleUnavailable

logger = logging.getLogger("speculators")

HiddenStateSample = dict[str, torch.Tensor]
OfflineMissingPolicy = Literal["skip", "warn", "raise"]


class HiddenStatesSource(Protocol):
    """Source of hidden states for one training record."""

    def get_one(self, record: DatasetRecord) -> HiddenStateSample | SampleUnavailable:
        """Return hidden states or a structured unavailable result."""

    def close(self) -> None:
        """Release source resources."""


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio)."""
    return any(isinstance(m.get("content"), list) for m in messages)


def build_client_item(record: DatasetRecord) -> ClientItem:
    """Build a vLLM request payload from a dataset record.

    Text-only records use their preprocessed token IDs. Multimodal records
    also forward their messages so vLLM can resolve image, video, or audio
    inputs through the Chat Completions API.
    """
    out_dict: dict[str, Any] = {"input_ids": record.input_ids.tolist()}

    if record.messages is not None and _has_multimodal_content(record.messages):
        out_dict["messages"] = record.messages

    return cast("ClientItem", out_dict)


class OfflineHiddenStatesSource:
    """Load hidden states that were already written to the transfer backend."""

    def __init__(
        self,
        *,
        transfer: HiddenStatesTransfer,
        on_missing: OfflineMissingPolicy,
    ):
        self.transfer = transfer
        self.on_missing = on_missing

    def get_one(self, record: DatasetRecord) -> HiddenStateSample | SampleUnavailable:
        """Load one cached sample by its storage index."""
        cached = self.transfer.get_cached(record.storage_index)
        if cached is not None:
            return cached

        if self.on_missing == "skip":
            return SampleUnavailable()
        if self.on_missing == "warn":
            warnings.warn(
                "Failed to load hidden states for sample "
                f"{record.storage_index}. Skipping...",
                stacklevel=1,
            )
            return SampleUnavailable(
                reason=f"Hidden states unavailable for sample {record.storage_index}"
            )
        raise RuntimeError(
            f"Failed to load hidden states for sample {record.storage_index}."
        )

    def close(self) -> None:
        """Offline transfers do not currently own closeable resources."""


class VllmHiddenStatesSource:
    """Generate and retrieve hidden states from a vLLM endpoint.

    This source always generates a sample. Cache-miss handling belongs to
    :class:`OfflineHiddenStatesSource`; this object owns the vLLM round trip,
    validation, cleanup, and generation recovery.
    """

    def __init__(
        self,
        *,
        transfer: HiddenStatesTransfer,
        vllm_endpoint: str = "http://localhost:8000/v1",
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        generation_validation_retries: int = 2,
        max_consecutive_generation_failures: int = 20,
    ):
        self.transfer = transfer
        self.vllm_endpoint = vllm_endpoint
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.client: openai.OpenAI | None = None
        self.generation_recovery = GenerationRecoveryGuard(
            retries=generation_validation_retries,
            max_consecutive_failures=max_consecutive_generation_failures,
        )

    def _setup_client(self) -> None:
        client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id
        self.transfer.setup()
        # Do not retain a half-initialized client if model discovery or Mooncake
        # setup failed; the outer full-round-trip retry should redo initialization.
        self.client = client

    def _generate_once(
        self,
        record: DatasetRecord,
    ) -> dict[str, torch.Tensor]:
        handle: str | None = None
        try:
            if not self.client:
                self._setup_client()
            handle, hs_transfer_params = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                build_client_item(record),
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(hs_transfer_params)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            # Covers token/shape mismatches and non-finite values. The Mooncake
            # transfer performs manifest validation first.
            check_hidden_states(loaded_hs, record.input_ids.tolist())

            try:
                self.transfer.delete(handle)
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning(
                    "Loaded a valid hidden-state sample but failed to delete "
                    "handle %s: %s",
                    handle,
                    cleanup_error,
                )
            return loaded_hs
        except Exception:
            if handle is not None:
                try:
                    self.transfer.delete(handle)
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Failed to clean generated hidden-state handle %s: %s",
                        handle,
                        cleanup_error,
                    )
            raise

    def get_one(
        self,
        record: DatasetRecord,
    ) -> HiddenStateSample | SampleUnavailable:
        """Generate one sample, including full-round-trip recovery."""
        return self.generation_recovery.run(
            lambda: self._generate_once(record),
            description=(
                "Hidden-state round trip failed for storage index "
                f"{record.storage_index}"
            ),
        )

    def close(self) -> None:
        """Close the lazily-created vLLM client, if any."""
        if self.client is None:
            return
        self.client.close()
        self.client = None
