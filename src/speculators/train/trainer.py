import logging
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from transformers import PreTrainedModel

from speculators.train.checkpointer import (
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.utils import apply_fully_sharded

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    is_distributed: bool = False
    local_rank: int = 0
    train_call_kwargs: dict = {}
    val_call_kwargs: dict = {}


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
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
        self.checkpointer = checkpointer_class(
            self.config.save_path,
            try_load_last_checkpoint=self.config.resume_from_checkpoint,
        )

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()

    def setup_trainer(self):
        if self.resume_from_checkpoint:
            self.current_epoch = self.checkpointer.previous_epoch + 1
        else:
            self.current_epoch = 0
        self.global_step = 0

    def setup_model(self):
        if self.is_distributed:
            apply_fully_sharded(self.model)

            if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
                self.checkpointer.load_model_state_dict(self.model)
            else:
                # Currently we make assumptions based on the Eagle3DraftModel
                # architecture, including the existence of a layers attribute.
                # todo: generalize to non-Eagle3DraftModel
                for m in self.model.layers.children():  # type: ignore[union-attr]
                    if not isinstance(m, FSDPModule):
                        continue
                    m.to_empty(device="cuda")  # type: ignore[attr-defined]
                    for sub_module in m.modules():  # type: ignore[attr-defined]
                        if hasattr(sub_module, "reset_parameters"):
                            sub_module.reset_parameters()
                # todo: Ensure lm_head and embed_tokens are loaded after reset
        else:
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
                self.checkpointer.load_model_state_dict(self.model)

    def setup_optimizer(self):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.opt)

    def train_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        train_loader = self.train_loader
        if self.local_rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]
        root_logger.info(f"Training Epoch {epoch} started")

        for batch in train_loader:
            gpu_batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            _draft_tokens, loss, draft_accuracies = self.model(
                **gpu_batch, **self.config.train_call_kwargs
            )

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            loss = loss.detach().clone()
            if self.is_distributed:
                # Note: this is not needed for training, just for logging
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
                dist.reduce(draft_accuracies, dst=0, op=dist.ReduceOp.AVG)

            acc_values = {
                f"acc_{i}": acc.item() for i, acc in enumerate(draft_accuracies)
            }
            metric_logger.info(
                {"train": {"loss": loss.item(), **acc_values}, "epoch": epoch},
                extra={"step": self.global_step},
            )
            self.global_step += 1

        root_logger.info(f"Training Epoch {epoch} completed")

    @torch.no_grad()
    def val_epoch(self, epoch: int):
        if self.val_loader is None:
            root_logger.warning("No val loader, skipping validation")
            return
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        root_logger.info(f"Validation Epoch {epoch} started")
        val_loader = self.val_loader
        if self.local_rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]
        val_loss = torch.zeros(1, device=self.local_rank)
        val_accuracies = torch.zeros(
            (), device=self.local_rank
        )  # initialize to tensor of shape ()
        for batch in val_loader:
            gpu_batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            _draft_tokens, loss, draft_accuracies = self.model(
                **gpu_batch, **self.config.val_call_kwargs
            )

            if self.is_distributed:
                dist.reduce(val_loss, dst=0, op=dist.ReduceOp.AVG)
                dist.reduce(draft_accuracies, dst=0, op=dist.ReduceOp.AVG)

            val_loss += loss.detach().clone()
            # Can't use += here because val_accuracies has shape () on first iteration
            val_accuracies = val_accuracies + draft_accuracies.detach()

        val_loss /= len(val_loader)
        val_accuracies /= len(val_loader)
        acc_values = {
            f"acc_{i}_epoch": acc.item() for i, acc in enumerate(val_accuracies)
        }
        metric_logger.info(
            {"val": {"loss_epoch": val_loss.item(), **acc_values}, "epoch": epoch},
            extra={"step": self.global_step},
        )
        root_logger.info(f"Validation Epoch {epoch} completed")

    def save_checkpoint(self, epoch: int):
        self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
        root_logger.info(f"Checkpoint saved to {self.checkpointer.path / str(epoch)}")

    def run_training(self):
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.train_epoch(epoch)
            if self.is_distributed:
                dist.barrier()
            self.val_epoch(epoch)
            self.save_checkpoint(epoch)
