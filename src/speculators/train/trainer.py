import logging
import time
import warnings
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
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
from speculators.train.profiling import GpuUtilSampler, StepTimer
from speculators.train.utils import apply_fully_sharded

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    is_distributed: bool = False
    local_rank: int = 0
    train_call_kwargs: dict = {}
    val_call_kwargs: dict = {}
    scheduler_type: Literal["linear", "cosine", "none"] = "linear"
    scheduler_warmup_steps: int | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5
    checkpoint_freq: int = 1
    save_best: bool = False
    hidden_states_dtype: torch.dtype = torch.bfloat16
    log_interval: int = 10
    profile_warmup_steps: int = 10


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
        self.local_rank = config.local_rank
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_distributed = config.is_distributed
        self.resume_from_checkpoint = config.resume_from_checkpoint
        checkpointer_class = (
            DistributedCheckpointer if self.is_distributed else SingleGPUCheckpointer
        )
        self.checkpointer: BaseCheckpointer = checkpointer_class(self.config.save_path)

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()

        self.step_timer = StepTimer(warmup_steps=config.profile_warmup_steps)
        self.gpu_sampler = GpuUtilSampler(device=self.local_rank)

    def setup_trainer(self):
        if self.checkpointer.previous_epoch != -1:
            root_logger.info(f"Found checkpoint at {self.checkpointer.prev_path}.")
            self.current_epoch = self.checkpointer.previous_epoch + 1
            if self.resume_from_checkpoint:
                root_logger.info(f"Resuming training on {self.current_epoch} epoch.")
            else:
                root_logger.warning(
                    "`resume_from_checkpoint` is False, starting "
                    "training from scratch. This will overwrite the "
                    f"existing checkpoints in {self.checkpointer.path}."
                )
                self.current_epoch = 0
        else:
            root_logger.info("No previous checkpoint found. Starting from scratch.")
            self.current_epoch = 0
        self.global_step = 0
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

        self.model.to(self.config.hidden_states_dtype)  # type: ignore[arg-type]
        load_checkpoint = (
            self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1
        )

        if not self.is_distributed:
            # Single device case
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if load_checkpoint:
                self.checkpointer.load_model_state_dict(self.model)
            return

        # Distributed case
        # Capture full state dict on rank 0 before FSDP sharding
        full_state_dict = {}
        if not load_checkpoint and dist.get_rank() == 0:
            full_state_dict = self.model.state_dict()

        apply_fully_sharded(self.model)

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

    def setup_optimizer(self):
        # Setup optimizer
        self.opt = torch.optim.AdamW(self.model.named_parameters(), lr=self.config.lr)
        last_epoch = -1
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.opt)
            last_epoch = self.checkpointer.previous_epoch

        # Setup scheduler
        if self.config.scheduler_type == "none":
            self.scheduler = None
            return

        # Compute defaults if None
        scheduler_warmup_steps = (
            self.config.scheduler_warmup_steps
            or (self.config.num_epochs * len(self.train_loader)) // 100
        )
        scheduler_total_steps = self.config.scheduler_total_steps or (
            self.config.num_epochs * len(self.train_loader)
        )

        if self.config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                last_epoch=last_epoch,
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                num_cycles=self.config.scheduler_num_cosine_cycles,
                last_epoch=last_epoch,
            )

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_scheduler_state_dict(self.scheduler)

    def train_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        train_loader = self.train_loader
        if self.local_rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        world_size = dist.get_world_size() if self.is_distributed else 1

        torch.cuda.synchronize(self.local_rank)
        t_prev = time.perf_counter()

        for batch in train_loader:
            t_fetch = time.perf_counter()

            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }
            torch.cuda.synchronize(self.local_rank)
            t_h2d = time.perf_counter()

            _draft_tokens, loss, metrics = self.model(
                **gpu_batch, **self.config.train_call_kwargs
            )
            torch.cuda.synchronize(self.local_rank)
            t_fwd = time.perf_counter()

            self.opt.zero_grad()
            loss.backward()
            torch.cuda.synchronize(self.local_rank)
            t_bwd = time.perf_counter()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            current_lr = self.opt.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()
            torch.cuda.synchronize(self.local_rank)
            t_opt = time.perf_counter()

            tokens = int(batch["input_ids"].numel()) * world_size
            self.step_timer.record(
                {
                    "fetch": t_fetch - t_prev,
                    "h2d": t_h2d - t_fetch,
                    "fwd": t_fwd - t_h2d,
                    "bwd": t_bwd - t_fwd,
                    "opt": t_opt - t_bwd,
                },
                tokens=tokens,
            )
            t_prev = t_opt

            if self.is_distributed:
                for v in metrics.values():
                    dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)

            metrics = {k: v.item() for k, v in metrics.items()}
            log_payload: dict = {
                "train": metrics,
                "epoch": epoch,
                "lr": current_lr,
            }
            if (self.global_step + 1) % self.config.log_interval == 0:
                profile_snap = self.step_timer.window_snapshot()
                gpu_util = self.gpu_sampler.pop_mean()
                if gpu_util is not None:
                    profile_snap["gpu_util_pct"] = gpu_util
                log_payload["profile"] = profile_snap
            metric_logger.info(
                log_payload, extra={"step": self.global_step}
            )
            self.global_step += 1

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.local_rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        val_metrics: dict[str, float] = {}
        num_batches = len(val_loader)
        for batch in val_loader:
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            _draft_tokens, _loss, metrics = self.model(
                **gpu_batch, **self.config.val_call_kwargs
            )

            if self.is_distributed:
                for m in metrics.values():
                    dist.all_reduce(m, op=dist.ReduceOp.AVG)

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        val_metrics = {f"{k}_epoch": v / num_batches for k, v in val_metrics.items()}
        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )
        return val_metrics

    def maybe_save_checkpoint(self, epoch: int, val_metrics: dict | None):
        if (
            self.config.save_best
            and val_metrics is not None
            and "loss_epoch" in val_metrics
        ):
            if val_metrics["loss_epoch"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss_epoch"]
                root_logger.info(
                    f"Saving new best checkpoint at epoch {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )
                self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
                if self.scheduler is not None:
                    self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)
                self.checkpointer.save_val_metrics(epoch, val_metrics)
                self.checkpointer.update_best_symlink(epoch)
                root_logger.info(
                    f"Updated checkpoint_best -> {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )
                # Keep ONLY the best checkpoint folder + best_checkpoint symlink
                self.checkpointer.cleanup_keep_only_best(best_epoch=epoch)

        elif epoch == 0 or (epoch + 1) % self.config.checkpoint_freq == 0:
            root_logger.info(
                f"Saving checkpoint to {self.checkpointer.path / str(epoch)}"
            )
            self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
            if self.scheduler is not None:
                self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)
            if val_metrics is not None:
                self.checkpointer.save_val_metrics(epoch, val_metrics)
            root_logger.info(
                f"Checkpoint saved to {self.checkpointer.path / str(epoch)}"
            )
            if (
                val_metrics is not None
                and "loss_epoch" in val_metrics
                and val_metrics["loss_epoch"] < self.best_val_loss
            ):
                self.best_val_loss = val_metrics["loss_epoch"]
                root_logger.info(
                    f"Updating new best checkpoint symlink at epoch {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )
                self.checkpointer.update_best_symlink(epoch)
                root_logger.info(
                    f"Updated checkpoint_best -> {epoch} "
                    f"(loss_epoch={self.best_val_loss:.6f})"
                )

    def run_training(self):
        n_epochs = self.config.num_epochs
        self.gpu_sampler.start()
        try:
            for epoch in range(self.current_epoch, n_epochs):
                root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
                self.train_epoch(epoch)
                root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

                if self.is_distributed:
                    dist.barrier()

                val_metrics = None

                if self.val_loader is None:
                    root_logger.warning("No val loader, skipping validation epoch")
                else:
                    root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                    val_metrics = self.val_epoch(epoch)
                    root_logger.info(
                        f"Validation epoch {epoch + 1}/{n_epochs} completed"
                    )

                if self.is_distributed:
                    dist.barrier()

                self.maybe_save_checkpoint(epoch, val_metrics)

                if self.is_distributed:
                    dist.barrier()
        finally:
            self.gpu_sampler.stop()
            sustained = self.step_timer.sustained_snapshot()
            if sustained:
                root_logger.info(f"Sustained throughput: {sustained}")
                metric_logger.info(
                    {"profile": sustained}, extra={"step": self.global_step}
                )
