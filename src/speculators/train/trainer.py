import json
import logging
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import (
    BaseCheckpointer,
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.distributed import (
    apply_fully_sharded,
    get_local_rank,
    get_rank,
    is_distributed,
)
from speculators.train.graceful_shutdown import with_graceful_shutdown
from speculators.train.optimizers import build_optimizers
from speculators.train.recovery import BatchRecoveryCoordinator
from speculators.train.utils import normalize_counted_metrics

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")


def _all_reduce_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sum *metrics* across ranks with a single collective.

    Used by both the training and validation metric reductions. Values are
    returned as float tensors in the same key order.
    """
    if not metrics:
        return {}
    keys = list(metrics)
    stacked = torch.stack([metrics[k].float().reshape(()) for k in keys])
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
    return dict(zip(keys, stacked, strict=True))


class _StepTimer:
    # A synchronizing mark()/now() forces an accelerator.synchronize to capture
    # true GPU time. This serialises the CUDA pipeline, so the training loop
    # synchronizes only on the step's closing mark and uses event marks for
    # the phases in between.
    # Caveat: with log_freq > 1, "start" falls back to an unsynchronized
    # timestamp from the end of the previous *unlogged* step, so start-relative
    # fields (fetch_ms, step_ms, queue_ms) also absorb any GPU backlog still
    # draining from that step. Phase-relative fields (fwd/bwd/opt/h2d/clip_ms)
    # are unaffected, as every mark resolves against a synchronized one.
    #
    # mark(name, sync=False) records a device event instead: the phase boundary
    # is resolved after the step's closing synchronized mark, by subtracting the
    # event-to-event device time from that mark's host timestamp. An event
    # completes when the device reaches it (or when the host records it, if the
    # device is idle), which is the same instant a synchronize-then-timestamp
    # observes -- without draining the queue mid-step.
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._marks: dict[str, float] = {}
        self._events: dict[str, torch.Event] = {}
        self._memory: dict[str, float] = {}

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._marks.clear()
        self._events.clear()
        self._memory.clear()

    def mark(self, name: str, sync: bool = True) -> None:
        if self.enabled:
            if not torch.accelerator.is_available():
                self._marks[name] = time.perf_counter()
                return
            if sync:
                anchor = torch.Event(enable_timing=True)
                anchor.record()
                torch.accelerator.synchronize()
                self._marks[name] = time.perf_counter()
                self._resolve_events(anchor, self._marks[name])
            else:
                event = torch.Event(enable_timing=True)
                event.record()
                self._events[name] = event
            self._memory[name] = torch.accelerator.memory_allocated() / (1024**2)

    def _resolve_events(self, anchor: torch.Event, anchor_time: float) -> None:
        for name, event in self._events.items():
            self._marks[name] = anchor_time - event.elapsed_time(anchor) / 1000
        self._events.clear()

    def mark_value(self, name: str, value: float) -> None:
        if self.enabled:
            self._marks[name] = value

    def last(self, name: str) -> float | None:
        # Host timestamp of a synchronized mark, reusable as the next "start".
        if not self.enabled:
            return None
        return self._marks[name]

    def now(self) -> float | None:
        if not self.enabled:
            return None
        torch.accelerator.synchronize()
        return time.perf_counter()

    def profile(self, num_tokens: int) -> dict | None:
        if not self.enabled:
            return None
        m = self._marks
        fetch_ms = (m["fetch"] - m["start"]) * 1000
        step_ms = (m["opt"] - m["start"]) * 1000
        result: dict = {
            "fetch_ms": fetch_ms,
            "fwd_ms": (m["fwd"] - m["fetch"]) * 1000,
            "bwd_ms": (m["bwd"] - m["fwd"]) * 1000,
            "opt_ms": (m["opt"] - m["bwd"]) * 1000,
            "step_ms": step_ms,
            "tokens_per_s": num_tokens / (step_ms / 1000) if step_ms > 0 else 0.0,
            "fetch_frac": fetch_ms / step_ms if step_ms > 0 else 0.0,
            "queue_ms": (m["queue"] - m["start"]) * 1000,
            "h2d_ms": (m["fetch"] - m["pre_h2d"]) * 1000,
            "clip_ms": (m["bwd"] - m["pre_clip"]) * 1000,
        }
        if self._memory:
            result["memory_mb"] = dict(self._memory)
        return result


def _metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    # One device-to-host transfer for the usual all-float32 scalar metrics,
    # instead of one per metric; values are identical to ``.item()``.
    values = list(metrics.values())
    if values and all(v.dtype == torch.float32 and v.dim() == 0 for v in values):
        return dict(zip(metrics, torch.stack(values).tolist(), strict=True))
    return {k: v.item() for k, v in metrics.items()}


class _BatchPrefetcher:
    """Iterate a loader, optionally starting the next batch's H2D copy early.

    ``prefetch()`` fetches the upcoming batch and copies its tensors to the
    device on a side stream, so the copy overlaps the current step's device
    work instead of running at the start of the next step. Each item is
    ``(batch, device_tensors)``; ``device_tensors`` is None when the batch was
    not prefetched. A loader error raised while prefetching is re-raised when
    iteration reaches that batch.
    """

    def __init__(self, loader, device: int, enabled: bool):
        self._loader = loader
        self._device = device
        self._stream = torch.cuda.Stream(device) if enabled else None
        self._batches = None
        self._next: tuple[dict, dict[str, torch.Tensor]] | None = None
        self._error: Exception | None = None
        self._exhausted = False

    def __iter__(self):
        self._batches = iter(self._loader)
        while True:
            if self._next is not None:
                batch, staged = self._next
                self._next = None
                current = torch.cuda.current_stream(self._device)
                current.wait_stream(self._stream)  # type: ignore[arg-type]
                for tensor in staged.values():
                    tensor.record_stream(current)
                yield batch, staged
                continue
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            if self._exhausted:
                return
            try:
                batch = next(self._batches)
            except StopIteration:
                return
            yield batch, None

    def prefetch(self) -> None:
        if self._stream is None or self._batches is None or self._next is not None:
            return
        try:
            batch = next(self._batches)
        except StopIteration:
            self._exhausted = True
            return
        except Exception as e:  # noqa: BLE001 -- re-raised by __iter__
            self._error = e
            return
        with torch.cuda.stream(self._stream):
            staged = {
                key: value.to(self._device, non_blocking=True)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
        self._next = (batch, staged)


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
MIN_STEP_PCT = 0.25

# Bound circuit-breaker detection latency without synchronizing every batch.
_RECOVERY_SYNC_INTERVAL = 50

# Bound rank skew before the validation metrics reduction.
_VAL_SYNC_INTERVAL = 50


def _should_sync_recovery(
    step: int,
    total_steps: int,
    *,
    will_stop: bool = False,
) -> bool:
    return (
        will_stop
        or step == total_steps
        or (_RECOVERY_SYNC_INTERVAL > 0 and step % _RECOVERY_SYNC_INTERVAL == 0)
    )


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    train_call_kwargs: dict | None = None
    val_call_kwargs: dict | None = None
    optimizer: Literal["adamw", "muon"] = "adamw"
    weight_decay: float = 0.01
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    scheduler_type: Literal["linear", "cosine", "none"] = "linear"
    scheduler_warmup_steps: int | None = None
    scheduler_warmup_ratio: float | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5
    checkpoint_freq: float = 1
    save_best: bool = False
    hidden_states_dtype: torch.dtype = torch.bfloat16
    log_freq: int = 1
    fsdp_shard: bool = False
    gradient_checkpointing: bool = False
    max_steps: int | None = None


def _resolve_scheduler_steps(
    config: TrainerConfig,
    train_loader_len: int,
) -> tuple[int, int]:
    """Resolve ``(warmup_steps, total_steps)`` for the LR scheduler.

    Explicit ``scheduler_warmup_steps`` wins; otherwise ``scheduler_warmup_ratio``
    (a fraction of total steps, validated to ``[0, 1]``) is used; otherwise the
    default of 1% of the resolved total steps. ``scheduler_total_steps`` defaults
    to ``max_steps`` when set, else ``num_epochs * train_loader_len``.
    """
    default_total_steps = config.num_epochs * train_loader_len
    # max_steps bounds the training loop, so the LR schedule must decay over the
    # same horizon; otherwise the LR endpoint is never reached.
    if config.max_steps is not None:
        default_total_steps = config.max_steps
    scheduler_total_steps = (
        config.scheduler_total_steps
        if config.scheduler_total_steps is not None
        else default_total_steps
    )

    if config.scheduler_warmup_steps is not None:
        scheduler_warmup_steps = config.scheduler_warmup_steps
        if config.scheduler_warmup_ratio is not None:
            warnings.warn(
                "Both scheduler_warmup_steps and scheduler_warmup_ratio are set; "
                "using scheduler_warmup_steps.",
                stacklevel=2,
            )
    elif config.scheduler_warmup_ratio is not None:
        if not 0 <= config.scheduler_warmup_ratio <= 1:
            raise ValueError("scheduler_warmup_ratio must be between 0 and 1.")
        scheduler_warmup_steps = int(
            scheduler_total_steps * config.scheduler_warmup_ratio
        )
    else:
        scheduler_warmup_steps = scheduler_total_steps // 100

    return scheduler_warmup_steps, scheduler_total_steps


class Trainer:
    def __init__(
        self,
        model: SpeculatorModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.config = config
        self.local_rank = get_local_rank()
        self.rank = get_rank()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_distributed = is_distributed()
        self.resume_from_checkpoint = config.resume_from_checkpoint
        acc = torch.accelerator.current_accelerator()
        self.device_type = acc.type if acc is not None else "cuda"
        checkpointer_class: type[BaseCheckpointer] = (
            DistributedCheckpointer
            if self.is_distributed and config.fsdp_shard
            else SingleGPUCheckpointer
        )
        self.checkpointer: BaseCheckpointer = checkpointer_class(self.config.save_path)

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()

    def _training_state_path(self, epoch: int) -> Path:
        return self.checkpointer.path / str(epoch) / "training_state.json"

    def _save_training_state(self, epoch: int, local_step: int) -> None:
        if not self.is_distributed or dist.get_rank() == 0:
            state = {
                "epoch": epoch,
                "local_step": local_step,
                "global_step": self.global_step,
            }
            p = self._training_state_path(epoch)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state))

    def _load_training_state(self) -> dict:
        epoch = self.checkpointer.previous_epoch
        p = self._training_state_path(epoch)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError as e:
                root_logger.warning(f"Failed to decode training state {p}: {e}")
            except (FileNotFoundError, PermissionError, OSError) as e:
                root_logger.warning(f"Failed to read training state {p}: {e}")
        return {}

    def setup_trainer(self):
        if self.checkpointer.previous_epoch != -1:
            root_logger.info(f"Found checkpoint at {self.checkpointer.prev_path}.")
            self.current_epoch = self.checkpointer.previous_epoch + 1
            if self.resume_from_checkpoint:
                # Check if this was a mid-epoch checkpoint — if so, resume
                # from within that epoch rather than jumping to the next one.
                state = self._load_training_state()
                is_mid_epoch = (
                    state
                    and state.get("epoch") == self.checkpointer.previous_epoch
                    and state.get("local_step", 0) > 0  # 0 means end-of-epoch
                )
                if is_mid_epoch:
                    # Resume within the same epoch from the exact step.
                    self.current_epoch = state["epoch"]
                    self._resume_local_step = state["local_step"]
                    self._resume_global_step = state.get("global_step", 0)
                    root_logger.info(
                        f"Resuming mid-epoch from epoch={self.current_epoch} "
                        f"local_step={self._resume_local_step} "
                        f"global_step={self._resume_global_step}."
                    )
                else:
                    # End-of-epoch or no state — advance to next epoch.
                    self._resume_local_step = 0
                    resume_global = state.get("global_step", 0) if state else 0
                    self._resume_global_step = resume_global
                    root_logger.info(
                        f"Resuming training on epoch {self.current_epoch}."
                    )
            else:
                root_logger.warning(
                    "`resume_from_checkpoint` is False, starting "
                    "training from scratch. This will overwrite the "
                    f"existing checkpoints in {self.checkpointer.path}."
                )
                self.current_epoch = 0
                self._resume_local_step = 0
                self._resume_global_step = 0
        else:
            root_logger.info(
                "No previous training checkpoint found in "
                f"'{self.checkpointer.path}'. Starting fresh training run."
            )
            self.current_epoch = 0
            self._resume_local_step = 0
            self._resume_global_step = 0
        self.global_step = self._resume_global_step
        self.best_val_loss = float("inf")

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            saved = self.checkpointer.load_best_val_loss()
            if saved is not None:
                self.best_val_loss = saved
                root_logger.info(
                    f"Restored best_val_loss={self.best_val_loss:.6f} from checkpoint"
                )

    def setup_model(self):
        # Verify model is compatible with training infrastructure
        SpeculatorModel.verify_training_compatible(self.model)

        # Enable gradient checkpointing BEFORE FSDP/DDP wrapping to save
        # activation memory at the cost of recomputation during backward.
        # Each decoder layer's forward is checkpointed: only the layer input
        # is saved for backward; intermediate activations (MLP, attention)
        # are recomputed. Saves ~10 GB for 5-layer DSpark with 32K seq.
        if self.config.gradient_checkpointing:
            if not self.model.supports_gradient_checkpointing:
                raise ValueError(
                    f"{type(self.model).__name__} does not support "
                    "gradient checkpointing"
                )
            self.model.gradient_checkpointing_enable()
            root_logger.info("Gradient checkpointing enabled")

        load_checkpoint = (
            self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1
        )

        if not self.is_distributed:
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if load_checkpoint:
                self.checkpointer.load_model_state_dict(self.model)
            return

        if self.config.fsdp_shard:
            self._setup_model_fsdp(load_checkpoint)
        else:
            self._setup_model_ddp(load_checkpoint)

    def _setup_model_fsdp(self, load_checkpoint: bool):
        # Capture full state dict on rank 0 before FSDP sharding
        full_state_dict = {}
        if not load_checkpoint and dist.get_rank() == 0:
            full_state_dict = self.model.state_dict()

        apply_fully_sharded(self.model, param_dtype=self.config.hidden_states_dtype)

        if load_checkpoint:
            self.checkpointer.load_model_state_dict(self.model)
        else:
            # Broadcast full state dict from rank 0 to all ranks
            set_model_state_dict(
                self.model,
                full_state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )
            del full_state_dict
            dist.barrier()

    def _setup_model_ddp(self, load_checkpoint: bool):
        self.model.to(self.local_rank)  # type: ignore[arg-type]

        if load_checkpoint:
            if dist.get_rank() == 0:
                self.checkpointer.load_model_state_dict(self.model)
        else:
            # Fresh init: broadcast rank 0's random initialization to all ranks
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

        # DDP constructor broadcasts rank 0's params to all ranks
        self.model = DistributedDataParallel(self.model)  # type: ignore[assignment]

    def setup_optimizer(self):
        # Setup optimizer(s). The "muon" option returns two optimizers (Muon for the
        # 2D weight matrices, AdamW for everything else); "adamw" returns a single one.
        self.optimizers = build_optimizers(self.model, self.config)
        last_epoch = -1
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.optimizers)
            last_epoch = self.checkpointer.previous_epoch

        # Setup scheduler(s) — one per optimizer so each optimizer's base LR (e.g.
        # Muon's higher LR vs AdamW's) is warmed up / decayed independently.
        if self.config.scheduler_type == "none":
            self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
            return

        scheduler_warmup_steps, scheduler_total_steps = _resolve_scheduler_steps(
            self.config, len(self.train_loader)
        )

        def make_scheduler(opt: torch.optim.Optimizer):
            if self.config.scheduler_type == "linear":
                return get_linear_schedule_with_warmup(
                    opt,
                    num_warmup_steps=scheduler_warmup_steps,
                    num_training_steps=scheduler_total_steps,
                    last_epoch=last_epoch,
                )
            return get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                num_cycles=self.config.scheduler_num_cosine_cycles,
                last_epoch=last_epoch,
            )

        self.schedulers = [make_scheduler(opt) for opt in self.optimizers]

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_scheduler_state_dict(self.schedulers)

    def _optimizers_zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def _optimizers_step(self):
        for opt in self.optimizers:
            opt.step()

    def _schedulers_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def _prepare_resume_skip(self, epoch: int) -> int:
        """Prepare fast-skip state for mid-epoch resume and return skipped steps."""
        skip_steps = 0
        if epoch == getattr(self, "current_epoch", epoch):
            skip_steps = getattr(self, "_resume_local_step", 0)
            # Only skip once — clear after use.
            self._resume_local_step = 0

        # Fast-skip: slice the sampler's pre-generated batch list so we never
        # call __getitem__ (and thus never call vLLM) for skipped batches.
        sampler = self.train_loader.batch_sampler
        has_fast_skip_api = hasattr(sampler, "_generate_batches") and hasattr(
            sampler, "_cached_generated_batches"
        )
        if skip_steps > 0 and has_fast_skip_api:
            all_batches = sampler._generate_batches(epoch)  # type: ignore[union-attr]  # noqa: SLF001
            remaining = all_batches[skip_steps:]
            # Temporarily override the sampler cache with the sliced list.
            sampler._cached_generated_batches = (  # type: ignore[union-attr]  # noqa: SLF001
                epoch,
                remaining,
            )
            root_logger.info(
                f"Fast-skipping {skip_steps} batches via sampler slice "
                f"(no vLLM calls for skipped batches). "
                f"epoch={epoch}, global_step={self.global_step}."
            )
        elif skip_steps > 0:
            root_logger.warning(
                "Sampler lacks fast-skip API; resume will replay "
                f"{skip_steps} batches from the start of the epoch."
            )
        return skip_steps

    def train_epoch(
        self,
        epoch: int,
        *,
        step_callback: Callable[[], None] | None = None,
    ):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        # Capture full-epoch step count before any resume fast-skip mutation.
        num_steps = len(self.train_loader)

        # Determine how many batches to skip for mid-epoch resume.
        skip_steps = self._prepare_resume_skip(epoch)

        train_loader = self.train_loader
        if self.rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        step_interval = (
            max(1, round(num_steps * self.config.checkpoint_freq))
            if self.config.checkpoint_freq < 1
            else None
        )
        t_before_fetch = time.perf_counter()
        timer = _StepTimer()
        recovery = BatchRecoveryCoordinator("training")
        remaining_steps = len(self.train_loader)
        # A logged step's record is emitted once the next step's backward has
        # been enqueued, so formatting and handlers run while the device is
        # busy rather than between steps. Records stay in step order.
        pending_log: tuple[dict, dict] | None = None

        def flush_log() -> None:
            nonlocal pending_log
            if pending_log is not None:
                record, extra = pending_log
                pending_log = None
                metric_logger.info(record, extra=extra)

        batches = _BatchPrefetcher(
            train_loader, self.local_rank, enabled=self.device_type == "cuda"
        )
        try:
            for local_step_rel, (batch, staged) in enumerate(batches, 1):
                # local_step is 1-based index into the *full* epoch (not the slice).
                local_step = local_step_rel + skip_steps
                timer.reset(self.global_step % self.config.log_freq == 0)

                timer.mark_value("start", t_before_fetch)
                will_stop = (
                    self.config.max_steps is not None
                    and self.global_step + 1 >= self.config.max_steps
                )
                timer.mark("queue", sync=False)
                recovery.consume(
                    batch,
                    synchronize=_should_sync_recovery(
                        local_step_rel,
                        remaining_steps,
                        will_stop=will_stop,
                    ),
                )
                timer.mark("pre_h2d", sync=False)
                gpu_batch = {
                    k: staged[k]
                    if staged is not None and k in staged
                    else v.to(self.local_rank, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in batch.items()
                }

                with torch.autocast(
                    self.device_type, dtype=self.config.hidden_states_dtype
                ):
                    timer.mark("fetch", sync=False)
                    _draft_tokens, loss, metrics = self.model(
                        **gpu_batch, **(self.config.train_call_kwargs or {})
                    )

                timer.mark("fwd", sync=False)
                self._optimizers_zero_grad()
                loss.backward()
                flush_log()
                timer.mark("pre_clip", sync=False)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # torch.full fills on the device; torch.tensor would copy the
                # value from host memory and wait for the queued backward.
                metrics["error_records_sum"] = torch.full(
                    (), batch["error_records"], dtype=torch.float32, device=loss.device
                )
                metrics["error_records_total"] = torch.full(
                    (),
                    1.0 if self.rank == 0 else 0,
                    dtype=torch.float32,
                    device=loss.device,
                )

                timer.mark("bwd", sync=False)
                self._optimizers_step()

                current_lrs = {
                    type(opt).__name__: opt.param_groups[0]["lr"]
                    for opt in self.optimizers
                }
                self._schedulers_step()
                if not will_stop:
                    batches.prefetch()
                timer.mark("opt")
                t_before_fetch = timer.last("opt") or time.perf_counter()

                profile = None
                if timer.enabled:
                    num_tokens = int((gpu_batch["document_ids"] != -1).sum().item())
                    profile = timer.profile(num_tokens)
                    if self.is_distributed:
                        metrics = _all_reduce_metrics(metrics)

                    metrics = _metrics_to_floats(metrics)
                    world_size = dist.get_world_size() if self.is_distributed else 1
                    metrics = normalize_counted_metrics(metrics, world_size)
                    lr_info = (
                        current_lrs
                        if len(current_lrs) > 1
                        else next(iter(current_lrs.values()))
                    )
                    pending_log = (
                        {
                            "train": metrics,
                            "profile": profile,
                            "epoch": epoch,
                            "lr": lr_info,
                            "global_step": self.global_step,
                        },
                        {"step": self.global_step},
                    )
                self.global_step += 1
                if step_callback is not None:
                    step_callback()

                if (
                    self.config.max_steps is not None
                    and self.global_step >= self.config.max_steps
                ):
                    break

                if (
                    step_interval is not None
                    and not self.config.save_best
                    and local_step % step_interval == 0
                    and num_steps - local_step >= step_interval * MIN_STEP_PCT
                    # Avoid saving back to back ay the end of each epoch
                ):
                    flush_log()
                    self.maybe_save_checkpoint(epoch, local_step=local_step)
        finally:
            flush_log()

    def _maybe_val_sync(self, batch_index: int) -> None:
        if not self.is_distributed or _VAL_SYNC_INTERVAL <= 0:
            return
        if batch_index > 0 and batch_index % _VAL_SYNC_INTERVAL == 0:
            dist.barrier()

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        accumulated: dict[str, torch.Tensor] = {}
        num_batches = len(val_loader)
        recovery = BatchRecoveryCoordinator("validation")
        for i, batch in enumerate(val_loader):
            self._maybe_val_sync(i)
            recovery.consume(
                batch,
                synchronize=_should_sync_recovery(i, num_batches - 1),
            )
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            with torch.autocast(
                self.device_type, dtype=self.config.hidden_states_dtype
            ):
                _draft_tokens, _loss, metrics = self.model(
                    **gpu_batch, **(self.config.val_call_kwargs or {})
                )

            for k, v in metrics.items():
                acc = accumulated.get(k)
                accumulated[k] = v.float() if acc is None else acc + v.float()

        val_metrics: dict[str, float] = {}
        if accumulated:
            if self.is_distributed:
                accumulated = _all_reduce_metrics(accumulated)
            val_metrics = {k: v.item() for k, v in accumulated.items()}

        world_size = dist.get_world_size() if self.is_distributed else 1
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics = normalize_counted_metrics(val_metrics, world_size)
        val_metrics = {f"{k}_epoch": v for k, v in val_metrics.items()}

        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )

        return val_metrics

    def maybe_save_checkpoint(self, epoch: int | str, local_step: int = 0):
        if epoch != "interrupted" and (
            self.config.save_best
            or (
                self.config.checkpoint_freq >= 1
                and isinstance(epoch, int)
                and epoch != 0
                and (epoch + 1) % self.config.checkpoint_freq != 0
            )
        ):
            return

        root_logger.info(f"Saving checkpoint to {self.checkpointer.path / str(epoch)}")
        self.checkpointer.save_checkpoint(self.model, self.optimizers, epoch)
        if self.schedulers:
            self.checkpointer.save_scheduler_state_dict(self.schedulers, epoch)
        if isinstance(epoch, int):
            self._save_training_state(epoch, local_step)
            # Create a human-readable symlink for checkpoint readability.
            # e.g. epoch0_step16626 -> 0/ (mid) or epoch0_end -> 0/ (end)
            if not self.is_distributed or dist.get_rank() == 0:
                ckpt_dir = self.checkpointer.path
                suffix = f"step{local_step}" if local_step > 0 else "end"
                link_name = ckpt_dir / f"epoch{epoch}_{suffix}"
                target = Path(str(epoch))  # relative symlink
                # Remove any previous link for this epoch
                for old in ckpt_dir.glob(f"epoch{epoch}_*"):
                    if old.is_symlink():
                        old.unlink()
                link_name.symlink_to(target)
        root_logger.info(f"Checkpoint saved to {self.checkpointer.path / str(epoch)}")

    def maybe_update_best(self, epoch: int, val_metrics: dict | None):
        if val_metrics is None or "loss_epoch" not in val_metrics:
            return
        if val_metrics["loss_epoch"] >= self.best_val_loss:
            return

        if self.config.save_best:
            self.checkpointer.save_checkpoint(self.model, self.optimizers, epoch)
            if self.schedulers:
                self.checkpointer.save_scheduler_state_dict(self.schedulers, epoch)
        elif self.config.checkpoint_freq >= 1 and not (
            epoch == 0 or (epoch + 1) % int(self.config.checkpoint_freq) == 0
        ):
            return

        self.best_val_loss = val_metrics["loss_epoch"]
        self.checkpointer.save_val_metrics(epoch, val_metrics)
        self.checkpointer.update_best_symlink(epoch)
        root_logger.info(
            f"Updated checkpoint_best -> {epoch} (loss_epoch={self.best_val_loss:.6f})"
        )
        if self.config.save_best:
            self.checkpointer.cleanup_keep_only_best(best_epoch=epoch)

    @with_graceful_shutdown()
    def run_training(self):
        n_epochs = self.config.num_epochs
        for epoch in range(self.current_epoch, n_epochs):
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
            self.train_epoch(epoch)
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_save_checkpoint(epoch)

            if self.is_distributed:
                dist.barrier()

            val_metrics = None

            if self.val_loader is None:
                root_logger.warning("No val loader, skipping validation epoch")
            else:
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                val_metrics = self.val_epoch(epoch)
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_update_best(epoch, val_metrics)

            if self.is_distributed:
                dist.barrier()
