"""Hidden-state acquisition for training data."""

import asyncio
import logging
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol, cast

import openai
import torch

from hs_connectors import HiddenStatesTransfer
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states_async,
    has_multimodal_content,
)
from speculators.train.data_types import DatasetRecord, GenerationReceipt
from speculators.train.recovery import GenerationRecoveryGuard, SampleUnavailable

logger = logging.getLogger("speculators")

HiddenStateSample = dict[str, torch.Tensor]
HiddenStateResult = HiddenStateSample | SampleUnavailable
OfflineMissingPolicy = Literal["skip", "warn", "raise"]


class HiddenStatesSource(Protocol):
    """Source of hidden states for one training record."""

    def get_one(self, record: DatasetRecord) -> HiddenStateResult:
        """Return hidden states or a structured unavailable result."""

    def get_batch(self, records: Sequence[DatasetRecord]) -> list[HiddenStateResult]:
        """Return one ordered result for each input record."""

    def close(self) -> None:
        """Release source resources."""


def build_client_item(record: DatasetRecord) -> ClientItem:
    """Build a vLLM request payload from a dataset record.

    Text-only records use their preprocessed token IDs. Multimodal records
    also forward their messages so vLLM can resolve image, video, or audio
    inputs through the Chat Completions API.
    """
    out_dict: dict[str, Any] = {"input_ids": record.input_ids.tolist()}

    if record.messages is not None and has_multimodal_content(record.messages):
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

    def get_one(self, record: DatasetRecord) -> HiddenStateResult:
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

    def get_batch(self, records: Sequence[DatasetRecord]) -> list[HiddenStateResult]:
        """Load cached samples in input order."""
        return [self.get_one(record) for record in records]

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
        model: str,
        vllm_endpoint: str = "http://localhost:8000/v1",
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        generation_validation_retries: int = 2,
        max_consecutive_generation_failures: int = 20,
        max_transfer_workers: int = 4,
    ):
        if max_transfer_workers <= 0:
            raise ValueError("max_transfer_workers must be > 0")

        self.transfer = transfer
        self.vllm_endpoint = vllm_endpoint
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.max_transfer_workers = max_transfer_workers
        self.client: openai.AsyncOpenAI | None = None
        self._runner: asyncio.Runner | None = None
        self._transfer_executor: ThreadPoolExecutor | None = None
        self._is_setup = False
        self.generation_recovery = GenerationRecoveryGuard(
            retries=generation_validation_retries,
            max_consecutive_failures=max_consecutive_generation_failures,
        )

    def _setup(self) -> None:
        if self._is_setup:
            return

        if self.client is None:
            self.transfer.setup()
            self.client = openai.AsyncOpenAI(
                base_url=self.vllm_endpoint,
                api_key="EMPTY",
                max_retries=0,
            )
        if self._runner is None:
            self._runner = asyncio.Runner()
        if self._transfer_executor is None:
            self._transfer_executor = ThreadPoolExecutor(
                max_workers=self.max_transfer_workers,
                thread_name_prefix="speculators-hidden-state-transfer",
            )
        self._is_setup = True

    def _load_generated(
        self,
        record: DatasetRecord,
        receipt: GenerationReceipt,
    ) -> dict[str, torch.Tensor]:
        try:
            loaded_hs = self.transfer.get_generated(receipt.transfer_params)
            if loaded_hs is None:
                raise ValueError(
                    f"Failed to load hidden states for handle {receipt.handle}"
                )

            # Covers token/shape mismatches and non-finite values. The Mooncake
            # transfer performs manifest validation first.
            check_hidden_states(loaded_hs, record.input_ids.tolist())

            try:
                self.transfer.delete(receipt.handle)
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup is best effort
                logger.warning(
                    "Loaded a valid hidden-state sample but failed to delete "
                    "handle %s: %s",
                    receipt.handle,
                    cleanup_error,
                )
            return loaded_hs
        except Exception:
            try:
                self.transfer.delete(receipt.handle)
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup is best effort
                logger.warning(
                    "Failed to clean generated hidden-state handle %s: %s",
                    receipt.handle,
                    cleanup_error,
                )
            raise

    def get_one(
        self,
        record: DatasetRecord,
    ) -> HiddenStateResult:
        """Generate one sample through the batch path."""
        return self.get_batch([record])[0]

    def get_batch(self, records: Sequence[DatasetRecord]) -> list[HiddenStateResult]:
        """Generate samples concurrently and return results in input order."""
        if not records:
            return []
        if not self._is_setup:
            self._setup()
        runner = cast("asyncio.Runner", self._runner)
        return runner.run(self._get_batch_async(records))

    async def _get_batch_async(
        self,
        records: Sequence[DatasetRecord],
    ) -> list[HiddenStateResult]:
        """Run one independent worker for each record in the batch."""
        if self.client is None or self._transfer_executor is None:
            raise RuntimeError("Hidden-state source was not set up")
        tasks = [
            asyncio.create_task(
                self._worker(
                    self.client,
                    record,
                    self._transfer_executor,
                )
            )
            for record in records
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for result in results:
            if isinstance(result, BaseException):
                raise result
        return cast("list[HiddenStateResult]", results)

    async def _worker(
        self,
        client: openai.AsyncOpenAI,
        record: DatasetRecord,
        transfer_executor: ThreadPoolExecutor,
    ) -> HiddenStateResult:
        """Generate, transfer, and validate one sample with recovery."""
        description = (
            f"Hidden-state round trip failed for storage index {record.storage_index}"
        )

        async def generate_and_transfer() -> dict[str, torch.Tensor]:
            handle, transfer_params = await generate_hidden_states_async(
                client,
                self.model,
                build_client_item(record),
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
            receipt = GenerationReceipt(handle, transfer_params)
            future = transfer_executor.submit(self._load_generated, record, receipt)
            while not future.done():  # noqa: ASYNC110 - keep request tasks running
                await asyncio.sleep(0.001)
            return future.result()

        return await self.generation_recovery.run_async(
            generate_and_transfer,
            description=description,
        )

    def close(self) -> None:
        """Close the lazily-created vLLM client, if any."""
        if (
            self.client is None
            and self._runner is None
            and self._transfer_executor is None
        ):
            return

        if self._runner is None:
            self._runner = asyncio.Runner()
        if self.client is not None:
            self._runner.run(self.client.close())
        if self._transfer_executor is not None:
            self._transfer_executor.shutdown(wait=True, cancel_futures=True)
        self.client = None
        self._transfer_executor = None
        self._is_setup = False
        self._runner.close()
        self._runner = None
