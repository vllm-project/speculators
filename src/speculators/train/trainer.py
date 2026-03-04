import logging
import warnings
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
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
    save_best: bool = False
    save_optimizer_state: bool = True
    max_checkpoints: int | None = None


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

    def setup_model(self):
        # Verify model is compatible with training infrastructure
        SpeculatorModel.verify_training_compatible(self.model)

        if self.is_distributed:
            apply_fully_sharded(self.model)

            if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
                self.checkpointer.load_model_state_dict(self.model)
            else:
                for m in self.model.layers.children():  # type: ignore[union-attr]
                    if not isinstance(m, FSDPModule):
                        continue
                    acc = torch.accelerator.current_accelerator()
                    if acc is None:
                        m.to_empty(device="cuda")  # type: ignore[attr-defined]
                    else:
                        acc_type = acc.type
                        m.to_empty(device=acc_type)  # type: ignore[attr-defined]
                    for sub_module in m.modules():  # type: ignore[attr-defined]
                        if hasattr(sub_module, "reset_parameters"):
                            sub_module.reset_parameters()  # type: ignore[operator]
                # todo: Ensure lm_head and embed_tokens are loaded after reset
        else:
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
                self.checkpointer.load_model_state_dict(self.model)

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

        for batch in train_loader:
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            _draft_tokens, loss, metrics = self.model(
                **gpu_batch, **self.config.train_call_kwargs
            )

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            current_lr = self.opt.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_distributed:
                for v in metrics.values():
                    dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)

            metrics = {k: v.item() for k, v in metrics.items()}
            metric_logger.info(
                {"train": metrics, "epoch": epoch, "lr": current_lr},
                extra={"step": self.global_step},
            )
            self.global_step += 1

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> float | None:
        """Run validation epoch and return average validation loss.

        Returns:
            Average validation loss for the epoch, or None if no val loader.
        """
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.local_rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        val_metrics: dict[str, float] = {}
        total_loss = 0.0
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
                _loss_reduced = _loss.clone()
                dist.reduce(_loss_reduced, dst=0, op=dist.ReduceOp.AVG)
                total_loss += _loss_reduced.item()
                for v in metrics.values():
                    dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)
            else:
                total_loss += _loss.item()

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_metrics = {f"{k}_epoch": v / num_batches for k, v in val_metrics.items()}
        val_metrics["loss_epoch"] = avg_val_loss
        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )
        return avg_val_loss

    def save_checkpoint(
        self,
        epoch: int,
        save_optimizer_state: bool = True,
    ):
        self.checkpointer.save_checkpoint(
            self.model,
            self.opt,
            epoch,
            save_optimizer_state=save_optimizer_state,
        )
        if save_optimizer_state and self.scheduler is not None:
            self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)

    def _should_save_checkpoint(
        self, val_loss: float | None, best_val_loss: float | None
    ) -> bool:
        """Determine whether to save a checkpoint for the current epoch.

        If ``save_best`` is enabled, saves only when validation loss improves.
        Otherwise, always saves.
        """
        if not self.config.save_best:
            return True
        if val_loss is None:
            # No validation loss available; always save
            return True
        if best_val_loss is None:
            return True
        return val_loss < best_val_loss

    def run_training(self):
        n_epochs = self.config.num_epochs
        best_val_loss: float | None = None

        for epoch in range(self.current_epoch, n_epochs):
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
            self.train_epoch(epoch)
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            val_loss: float | None = None
            if self.val_loader is None:
                root_logger.warning("No val loader, skipping validation epoch")
            else:
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                val_loss = self.val_epoch(epoch)
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            if self._should_save_checkpoint(val_loss, best_val_loss):
                root_logger.info(
                    f"Started saving checkpoint to "
                    f"{self.checkpointer.path / str(epoch)}"
                )
                self.save_checkpoint(
                    epoch,
                    save_optimizer_state=self.config.save_optimizer_state,
                )
                root_logger.info(
                    f"Finished saving checkpoint to "
                    f"{self.checkpointer.path / str(epoch)}"
                )

                # Cleanup old checkpoints if max_checkpoints is set
                if self.config.max_checkpoints is not None:
                    self.checkpointer.cleanup_old_checkpoints(
                        self.config.max_checkpoints
                    )

                if val_loss is not None:
                    best_val_loss = val_loss
                    root_logger.info(
                        f"Best validation loss updated to {best_val_loss:.6f}"
                    )
            else:
                root_logger.info(
                    f"Skipping checkpoint save (val_loss={val_loss:.6f} "
                    f">= best={best_val_loss:.6f})"
                )
