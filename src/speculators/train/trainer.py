import torch
from torch.distributed.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.utils.data import DataLoader
from tqdm.rich import tqdm  # todo: requries tqdm and rich


import torch.distributed as dist
import logging

from speculators.train.checkpointer import (
    SingleGPUCheckpointer,
    DistributedCheckpointer,
)

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")


def apply_fully_sharded(model: torch.nn.Module):
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }

    for layer in model.layers:  # todo: this is hardcoded to the Eagle3DraftModel definition, should be made more general
        # we apply fully_shard to each DecoderLayer
        layer.to_empty(device="meta")
        fully_shard(layer, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        verifier_lm_head: torch.nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        is_distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.verifier_lm_head = verifier_lm_head
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_distributed = is_distributed
        self.local_rank = local_rank
        self.world_size = world_size
        checkpointer_class = (
            DistributedCheckpointer if is_distributed else SingleGPUCheckpointer
        )
        self.checkpointer = checkpointer_class(
            config["save_path"],
            try_load_last_checkpoint=config.get("resume_from_checkpoint", False),
        )

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()

    def setup_trainer(self):
        self.current_epoch = self.checkpointer.previous_epoch + 1
        self.global_step = 0

    def setup_model(self):
        if self.is_distributed:
            apply_fully_sharded(self.model)

            if self.checkpointer.previous_epoch != -1:
                self.checkpointer.load_model_state_dict(self.model)
            else:
                for m in self.model.layers.children():  # todo: generalize
                    if not isinstance(m, FSDPModule):
                        continue
                    m.to_empty(device="cuda")  # todo: generalize
                    for sub_module in m.modules():
                        if hasattr(sub_module, "reset_parameters"):
                            sub_module.reset_parameters()
                # todo: We need to make sure we're loading lm_head and embed_tokens after this reset
        else:
            self.model.to(self.local_rank)
            if self.checkpointer.previous_epoch != -1:
                self.checkpointer.load_model_state_dict(self.model)
        self.verifier_lm_head = self.verifier_lm_head.to(self.local_rank)

    def setup_optimizer(self):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        if self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.opt)

    def train_epoch(self, epoch: int):
        self.model.train()
        self.train_loader.batch_sampler.set_epoch(
            epoch
        )  # todo: check if this is safe to call

        if self.local_rank == 0:
            train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            train_loader = self.train_loader
        root_logger.info(f"Training Epoch {epoch} started")

        for batch in train_loader:
            batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_logits = self.verifier_lm_head(batch["verifier_last_hidden_states"])
            del batch["verifier_last_hidden_states"]

            _draft_tokens, loss, draft_accuracies = self.model(
                **batch, target_logits=target_logits, use_off_policy_tokens=False
            )  # set this in a better way

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
        self.val_loader.batch_sampler.set_epoch(epoch)
        root_logger.info(f"Validation Epoch {epoch} started")
        if self.local_rank == 0:
            val_loader = tqdm(self.val_loader, desc=f"Epoch {epoch}")
        else:
            val_loader = self.val_loader
        val_loss = torch.zeros(1, device=self.local_rank)
        val_accuracies = torch.zeros(
            (), device=self.local_rank
        )  # initialize to tensor of shape ()
        for batch in val_loader:
            batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_logits = self.verifier_lm_head(batch["verifier_last_hidden_states"])
            del batch["verifier_last_hidden_states"]

            _draft_tokens, loss, draft_accuracies = self.model(
                **batch, target_logits=target_logits, use_off_policy_tokens=False
            )  # set this in a better way

            if self.is_distributed:
                dist.reduce(val_loss, dst=0, op=dist.ReduceOp.AVG)
                dist.reduce(draft_accuracies, dst=0, op=dist.ReduceOp.AVG)

            val_loss += loss.detach().clone()
            # Can't use += here because val_accuracies is a tensor of shape () on first iteration
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
        for epoch in range(self.current_epoch, self.config["num_epochs"]):
            self.train_epoch(epoch)
            if self.is_distributed:
                dist.barrier()
            self.val_epoch(epoch)
            self.save_checkpoint(epoch)
