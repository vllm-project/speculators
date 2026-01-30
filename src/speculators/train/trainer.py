import logging
import warnings
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    # Fallback for newer PyTorch versions where FSDPModule was removed
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDPModule
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from transformers import (
    PreTrainedModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from speculators.models.eagle3 import Eagle3DraftModel
from speculators.models.dflash import DFlashDraftModel
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
    scheduler_type: Literal["linear", "cosine", "none"] = "none"
    scheduler_warmup_steps: int | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        print("weights shape at line 60 trainer", self.model.lm_head.weight.shape, flush=True)
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
        print("weights shape at line 74 trainer", self.model.lm_head.weight.shape, flush=True)
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
        if self.is_distributed:
            apply_fully_sharded(self.model)

            if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
                self.checkpointer.load_model_state_dict(self.model)
            else:
                # Skip verifier-shared layers during reset to preserve pretrained weights
                skip_modules = {self.model.lm_head, self.model.embed_tokens, self.model.verifier_lm_head}

                for m in self.model.layers.children():  # type: ignore[union-attr]
                    if not isinstance(m, FSDPModule):
                        continue
                    m.to_empty(device="cuda")  # type: ignore[attr-defined]
                    for sub_module in m.modules():  # type: ignore[attr-defined]
                        if sub_module in skip_modules:
                            continue
                        if hasattr(sub_module, "reset_parameters"):
                            sub_module.reset_parameters()  # type: ignore[operator]
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
    def val_epoch(self, epoch: int):
        if self.val_loader is None:
            return
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
                for v in metrics.values():
                    dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        val_metrics = {f"{k}_epoch": v / num_batches for k, v in val_metrics.items()}
        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )

    def save_checkpoint(self, epoch: int):
        self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
        if self.scheduler is not None:
            self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)

    def run_training(self):
        n_epochs = self.config.num_epochs
        for epoch in range(self.current_epoch, n_epochs):
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
            self.train_epoch(epoch)
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            if self.val_loader is None:
                root_logger.warning("No val loader, skipping validation epoch")
            else:
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                self.val_epoch(epoch)
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            root_logger.info(
                f"Started saving checkpoint to {self.checkpointer.path / str(epoch)}"
            )
            self.save_checkpoint(epoch)
            root_logger.info(
                f"Finished saving checkpoint to {self.checkpointer.path / str(epoch)}"
            )
