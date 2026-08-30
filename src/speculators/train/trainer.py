import json
import logging
import time
import warnings
from pathlib import Path
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from rich.errors import LiveError
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
from speculators.train.utils import normalize_counted_metrics

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")


class _StepTimer:
    # Each mark()/now() forces an accelerator.synchronize to capture true GPU time.
    # This serialises the CUDA pipeline, so profiled steps are slower; keep
    # log_freq > 1 in perf-sensitive runs.
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._marks: dict[str, float] = {}

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._marks.clear()

    def mark(self, name: str) -> None:
        if self.enabled:
            torch.accelerator.synchronize()
            self._marks[name] = time.perf_counter()

    def mark_value(self, name: str, value: float) -> None:
        if self.enabled:
            self._marks[name] = value

    def now(self) -> float | None:
        if not self.enabled:
            return None
        torch.accelerator.synchronize()
        return time.perf_counter()

    def profile(self, num_tokens: int) -> dict[str, float] | None:
        if not self.enabled:
            return None
        m = self._marks
        has_start = "start" in m
        fwd_ms = (m["fwd"] - m["fetch"]) * 1000
        bwd_ms = (m["bwd"] - m["fwd"]) * 1000
        opt_ms = (m["opt"] - m["bwd"]) * 1000
        fetch_ms = (m["fetch"] - m["start"]) * 1000 if has_start else 0.0
        step_ms = (m["opt"] - m["start"]) * 1000 if has_start else 0.0
        tokens_per_s = num_tokens / (step_ms / 1000) if step_ms > 0 else 0.0
        fetch_frac = fetch_ms / step_ms if step_ms > 0 else 0.0
        return {
            "fetch_ms": fetch_ms,
            "fwd_ms": fwd_ms,
            "bwd_ms": bwd_ms,
            "opt_ms": opt_ms,
            "step_ms": step_ms,
            "tokens_per_s": tokens_per_s,
            "fetch_frac": fetch_frac,
        }


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
MIN_STEP_PCT = 0.25

# Re-synchronise ranks every N validation batches to bound cross-rank skew, which would
# otherwise blow the NCCL watchdog at the end-of-epoch metrics all-reduce. 0 disables.
_VAL_SYNC_INTERVAL = 50


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    train_call_kwargs: dict | None = None
    val_call_kwargs: dict | None = None
    optimizer: Literal["adamw", "muon"] = "adamw"
    weight_decay: float = 0.01
    # Keep an fp32 master copy of every low-precision parameter and step THAT,
    # copying the result back. Costs roughly +6 bytes per parameter (fp32 weights
    # plus fp32 Adam moments) over stepping bf16 directly, so it is opt-in.
    fp32_master_weights: bool = False
    # Run the accept-length eval every N optimizer steps instead of only at epoch
    # end. None = epoch end only.
    eval_interval: int | None = None
    # Per-rank batch cap for those evals; None sweeps the whole validation set.
    eval_max_batches: int | None = None
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
    max_steps: int | None = None


def _resolve_scheduler_steps(
    config: TrainerConfig,
    train_loader_len: int,
) -> tuple[int, int]:
    """Resolve ``(warmup_steps, total_steps)`` for the LR scheduler.

    Explicit ``scheduler_warmup_steps`` wins; otherwise ``scheduler_warmup_ratio``
    (a fraction of total steps, validated to ``[0, 1]``) is used; otherwise the
    default of 1% of the resolved total steps. ``scheduler_total_steps`` defaults
    to ``num_epochs * train_loader_len``.
    """
    default_total_steps = config.num_epochs * train_loader_len
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
        self._fp32_masters: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._anchor_start: float | None = None
        self._anchor_total: int = 1
        self._anchor_now: float = 0.0
        self._anchor_buf: torch.Tensor | None = None
        self._metric_keys: list[str] | None = None
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
        self.optimizers, self._fp32_masters = build_optimizers(self.model, self.config)
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
        # The optimizer owns the masters, so the call above cleared THEIR grads;
        # the parameters autograd actually writes into are a separate set.
        for param, _ in self._fp32_masters:
            param.grad = None

    def _optimizers_step(self):
        for opt in self.optimizers:
            opt.step()

    def _clip_grads(self):
        """Clip gradients, in fp32 when master weights are in use."""
        if not self._fp32_masters:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            return
        # DDP has already all-reduced into param.grad. Move each gradient into its
        # fp32 master so the clip and the step both happen in fp32, where a small
        # update is not rounded away at bf16's ~8-bit mantissa.
        for param, master in self._fp32_masters:
            master.grad = None if param.grad is None else param.grad.float()
        torch.nn.utils.clip_grad_norm_(
            [master for _, master in self._fp32_masters], 1.0
        )

    def _sync_from_masters(self):
        """Copy the stepped fp32 masters back into the model's parameters."""
        if not self._fp32_masters:
            return
        with torch.no_grad():
            for param, master in self._fp32_masters:
                param.data.copy_(master.data)

    def _schedulers_step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def _anneal_base_anchor(self) -> None:
        """Decay ``base_anchor_weight`` toward ``base_anchor_floor`` over the run.

        A no-op unless the algorithm passes a floor. ``train_call_kwargs`` is the
        live dict handed to ``forward``, so mutating it takes effect on the next
        call.
        """
        call_kwargs = self.config.train_call_kwargs or {}
        floor = call_kwargs.get("base_anchor_floor")
        if floor is None:
            return
        if self._anchor_start is None:
            self._anchor_start = float(call_kwargs.get("base_anchor_weight", 0.6))
            self._anchor_total = max(1, self.config.num_epochs * len(self.train_loader))
            # Hand the model a 0-dim TENSOR, never a Python float. Dynamo guards a
            # float by its exact value, so an annealed scalar invalidates the guard
            # every step: the frame recompiles until it hits recompile_limit and then
            # falls back to eager for the REST of the run -- torch.compile silently
            # stops applying, after one warning in the first minutes. A tensor carries
            # the same number with no value guard, and reusing one buffer keeps the
            # object identity (and its shape/dtype/device guards) stable.
            self._anchor_buf = torch.zeros(
                (), device=self.local_rank, dtype=torch.float32
            )
        progress = min(1.0, self.global_step / self._anchor_total)
        self._anchor_now = self._anchor_start * (1 - progress) + float(floor) * progress
        self._anchor_buf.fill_(self._anchor_now)  # type: ignore[union-attr]
        call_kwargs["base_anchor_weight"] = self._anchor_buf

    def _reduce_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        """Sum each metric across ranks, in a canonical key order.

        Iterating dict values relies on every rank having built this dict
        identically; if one rank takes a different branch the reduces pair up
        mismatched tensors and the logged metrics are silently wrong. Sorting
        removes the ordering half of that risk; a differing key SET would still
        misalign, so warn when the key list shifts rather than failing quietly.
        """
        keys = sorted(metrics)
        if self._metric_keys is None:
            self._metric_keys = keys
        elif keys != self._metric_keys:
            root_logger.warning(
                "metric keys changed between logged steps (%d -> %d); "
                "cross-rank reduction may be misaligned",
                len(self._metric_keys),
                len(keys),
            )
            self._metric_keys = keys
        for key in keys:
            dist.reduce(metrics[key], dst=0, op=dist.ReduceOp.SUM)

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

    def train_epoch(self, epoch: int):
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
        for local_step_rel, batch in enumerate(train_loader, 1):
            # local_step is 1-based index into the *full* epoch (not the slice).
            local_step = local_step_rel + skip_steps
            timer.reset(self.global_step % self.config.log_freq == 0)

            timer.mark_value("start", t_before_fetch)
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            self._anneal_base_anchor()

            with torch.autocast(
                self.device_type, dtype=self.config.hidden_states_dtype
            ):
                timer.mark("fetch")
                _draft_tokens, loss, metrics = self.model(
                    **gpu_batch, **(self.config.train_call_kwargs or {})
                )

            timer.mark("fwd")
            self._optimizers_zero_grad()
            loss.backward()
            self._clip_grads()

            timer.mark("bwd")
            self._optimizers_step()
            self._sync_from_masters()

            current_lrs = {
                type(opt).__name__: opt.param_groups[0]["lr"] for opt in self.optimizers
            }
            self._schedulers_step()
            timer.mark("opt")
            t_before_fetch = timer.now() or time.perf_counter()

            profile = None
            if timer.enabled:
                num_tokens = int((gpu_batch["document_ids"] != -1).sum().item())
                profile = timer.profile(num_tokens)
                if self._anchor_buf is not None:
                    metrics["base_anchor_weight_sum"] = self._anchor_buf.clone()
                    metrics["base_anchor_weight_total"] = torch.ones_like(
                        self._anchor_buf
                    )
                if self.is_distributed:
                    self._reduce_metrics(metrics)

                metrics = {k: v.item() for k, v in metrics.items()}
                world_size = dist.get_world_size() if self.is_distributed else 1
                metrics = normalize_counted_metrics(metrics, world_size)
                lr_info = (
                    current_lrs
                    if len(current_lrs) > 1
                    else next(iter(current_lrs.values()))
                )
                metric_logger.info(
                    {
                        "train": metrics,
                        "profile": profile,
                        "epoch": epoch,
                        "lr": lr_info,
                        "global_step": self.global_step,
                    },
                    extra={"step": self.global_step},
                )
            self.global_step += 1

            # Mid-epoch accept-length eval. Runs right after an optimizer step, so
            # the gradients are already applied and zeroed. global_step is identical
            # on every rank, so all ranks enter val_epoch's collectives together.
            if (
                self.config.eval_interval
                and self.val_loader is not None
                and self.global_step % self.config.eval_interval == 0
            ):
                self.val_epoch(epoch, max_batches=self.config.eval_max_batches)
                self.model.train()  # val_epoch left it in eval mode

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
                self.maybe_save_checkpoint(epoch, local_step=local_step)

    def _maybe_val_sync(self, batch_index: int) -> None:
        if not self.is_distributed or _VAL_SYNC_INTERVAL <= 0:
            return
        if batch_index > 0 and batch_index % _VAL_SYNC_INTERVAL == 0:
            dist.barrier()

    @torch.no_grad()
    def val_epoch(  # noqa: C901
        self, epoch: int, max_batches: int | None = None
    ) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.rank == 0:
            try:
                val_loader = tqdm(val_loader, desc=f"Epoch {epoch} [val]")  # type: ignore[assignment]
            except LiveError:
                # A mid-epoch eval runs inside train_epoch, whose progress bar is
                # still live. Older rich allows only one live display and raises
                # here; newer rich nests them. Detecting which by attribute is
                # version-fragile, so fall back to no bar rather than ending a
                # multi-day run over a cosmetic.
                val_loader = self.val_loader

        accumulated: dict[str, torch.Tensor] = {}
        num_batches = len(val_loader)
        if max_batches is not None:
            # Every rank must stop after the SAME number of batches or the metric
            # all-reduce below deadlocks. max_batches is a constant, so they do.
            num_batches = min(num_batches, max_batches)
        seen = 0
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            seen = i + 1
            self._maybe_val_sync(i)
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
            stacked = torch.stack(list(accumulated.values()))
            if self.is_distributed:
                dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
            val_metrics = dict(zip(accumulated, stacked.tolist(), strict=True))

        world_size = dist.get_world_size() if self.is_distributed else 1
        divisor = max(1, seen or num_batches)
        val_metrics = {k: v / divisor for k, v in val_metrics.items()}
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
