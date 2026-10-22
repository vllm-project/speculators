from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.distributed as dist

from speculators.train.distributed import get_local_rank, get_rank, is_distributed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("speculators")

RECOVERY_METADATA_KEY = "__recovery__"

T = TypeVar("T")


@dataclass(frozen=True)
class SampleUnavailable:
    """Picklable result for a sample the dataset could not provide."""

    reason: str = ""
    counts_as_failure: bool = False
    consecutive_failures: int = 0
    fatal: bool = False


@dataclass(frozen=True)
class RecoveryMetadata:
    """Recovery state carried from the collator to the trainer."""

    failure_count: int = 0
    locally_empty: bool = False
    fatal: bool = False
    error: str = ""

    @classmethod
    def from_unavailable(
        cls,
        unavailable: Sequence[SampleUnavailable],
        *,
        locally_empty: bool,
    ) -> RecoveryMetadata:
        """Summarize unavailable samples for one collated batch."""
        failures = [sample for sample in unavailable if sample.counts_as_failure]
        fatal = next((sample for sample in failures if sample.fatal), None)
        return cls(
            failure_count=len(failures),
            locally_empty=locally_empty,
            fatal=fatal is not None,
            error=fatal.reason if fatal is not None else "",
        )


class GenerationRecoveryGuard:
    """Retry generation operations and bound consecutive exhausted samples.

    DataLoader workers each receive their own dataset copy, so a guard stored on the
    dataset naturally tracks failures independently per persistent worker.
    """

    def __init__(self, retries: int, max_consecutive_failures: int):
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be > 0")
        self.retries = retries
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive samples which exhausted all attempts."""
        return self._consecutive_failures

    def run(
        self,
        operation: Callable[[], T],
        *,
        description: str,
    ) -> T | SampleUnavailable:
        """Run ``operation`` until it succeeds or all attempts are exhausted."""
        total_attempts = self.retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                result = operation()
            except Exception as error:  # noqa: BLE001 - worker recovery boundary
                if attempt == total_attempts:
                    return self._record_exhausted(
                        description,
                        attempt=attempt,
                        total_attempts=total_attempts,
                        error=error,
                    )
                logger.warning(
                    "%s (attempt %d/%d): %s: %s",
                    description,
                    attempt,
                    total_attempts,
                    type(error).__name__,
                    error,
                )
                continue

            self._record_success()
            return result

        raise RuntimeError("Generation retry loop completed without a result")

    def _record_exhausted(
        self,
        description: str,
        *,
        attempt: int,
        total_attempts: int,
        error: Exception,
    ) -> SampleUnavailable:
        self._consecutive_failures += 1
        message = (
            f"{description} (attempt {attempt}/{total_attempts}, "
            f"consecutive failures={self._consecutive_failures}/"
            f"{self.max_consecutive_failures}): {type(error).__name__}: {error}"
        )
        logger.warning(message)
        return SampleUnavailable(
            reason=message,
            counts_as_failure=True,
            consecutive_failures=self._consecutive_failures,
            fatal=self._consecutive_failures >= self.max_consecutive_failures,
        )

    def _record_success(self) -> None:
        if self._consecutive_failures:
            logger.warning(
                "Hidden-state generation recovered after %d consecutive failure(s).",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0


class BatchRecoveryCoordinator:
    """Consume batch metadata and periodically coordinate recovery across ranks."""

    def __init__(
        self,
        phase: str,
        *,
        device: torch.device | int | None = None,
    ) -> None:
        self.phase = phase
        self.device = device if device is not None else self._default_device()
        self._pending = RecoveryMetadata()

    def consume(self, batch: dict[str, Any], *, synchronize: bool = False) -> None:
        """Remove recovery metadata and optionally synchronize pending state."""
        metadata = batch.pop(RECOVERY_METADATA_KEY, RecoveryMetadata())
        if not isinstance(metadata, RecoveryMetadata):
            raise TypeError(
                f"{RECOVERY_METADATA_KEY} must contain RecoveryMetadata, "
                f"got {type(metadata).__name__}"
            )
        if leftover := sorted(key for key in batch if key.startswith("__")):
            raise RuntimeError(f"Unconsumed data recovery metadata: {leftover}")

        if metadata.failure_count or metadata.locally_empty:
            logger.warning(
                "%s data recovery on rank %d: failed_samples=%d, "
                "locally_empty_batch=%s",
                self.phase,
                get_rank(),
                metadata.failure_count,
                metadata.locally_empty,
            )

        self._pending = RecoveryMetadata(
            failure_count=(self._pending.failure_count + metadata.failure_count),
            locally_empty=self._pending.locally_empty or metadata.locally_empty,
            fatal=self._pending.fatal or metadata.fatal,
            error=self._pending.error or metadata.error,
        )

        if synchronize or not is_distributed():
            self.synchronize()

    def synchronize(self) -> None:
        """Aggregate pending failures and raise a circuit breaker on every rank."""
        pending = self._pending
        self._pending = RecoveryMetadata()

        if is_distributed():
            status = torch.tensor(
                [int(pending.fatal), pending.failure_count],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(status, op=dist.ReduceOp.SUM)
            fatal_ranks, total_failures = (int(value) for value in status.tolist())
        else:
            fatal_ranks = int(pending.fatal)
            total_failures = pending.failure_count

        rank = get_rank()
        if total_failures and rank == 0:
            logger.warning(
                "%s hidden-state generation dropped %d sample(s) across ranks; "
                "affected ranks continue with remaining or locally empty data.",
                self.phase,
                total_failures,
            )
        if fatal_ranks:
            detail = f" Local error: {pending.error}" if pending.error else ""
            raise RuntimeError(
                "Hidden-state generation circuit breaker tripped on "
                f"{fatal_ranks} rank(s) during {self.phase}.{detail}"
            )

    @staticmethod
    def _default_device() -> torch.device:
        if not torch.accelerator.is_available():
            return torch.device("cpu")
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            return torch.device("cpu")
        return torch.device(accelerator.type, get_local_rank())
