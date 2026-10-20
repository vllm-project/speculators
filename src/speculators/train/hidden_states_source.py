"""Hidden-state acquisition for training data."""

import logging
from typing import Any

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
from speculators.train.recovery import GenerationRecoveryGuard, SampleUnavailable

BatchType = dict[str, Any]
logger = logging.getLogger("speculators")


class VllmHiddenStatesSource:
    """Generate and retrieve hidden states from a vLLM endpoint.

    This is intentionally a small extraction of the existing ArrowDataset
    behavior. Cache lookup and the ``on_missing`` policy remain in
    :class:`ArrowDataset` for now; this object owns only the vLLM round trip,
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
        dataset_item: BatchType,
        client_item: ClientItem,
    ) -> dict[str, torch.Tensor]:
        handle: str | None = None
        try:
            if not self.client:
                self._setup_client()
            handle, hs_transfer_params = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(hs_transfer_params)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            # Covers token/shape mismatches and non-finite values. The Mooncake
            # transfer performs manifest validation first.
            check_hidden_states(loaded_hs, dataset_item["input_ids"].tolist())

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
        dataset_item: BatchType,
        client_item: ClientItem,
        *,
        dataset_index: int,
        file_index: int,
    ) -> dict[str, torch.Tensor] | SampleUnavailable:
        """Generate one sample, including full-round-trip recovery."""
        return self.generation_recovery.run(
            lambda: self._generate_once(dataset_item, client_item),
            description=(
                f"Hidden-state round trip failed for dataset index {dataset_index}, "
                f"file index {file_index}"
            ),
        )

    def close(self) -> None:
        """Close the lazily-created vLLM client, if any."""
        if self.client is None:
            return
        self.client.close()
        self.client = None
