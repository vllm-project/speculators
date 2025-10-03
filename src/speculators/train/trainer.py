import os
import torch
from torch.utils.data import DataLoader
from tqdm.rich import tqdm  # todo: requries tqdm and rich

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from speculators.train.utils import log_rank0


@torch.no_grad()
def compute_draft_accuracy(
    target_logits: torch.Tensor,
    draft_tokens: list[torch.Tensor],
    loss_mask: torch.Tensor | None = None,
):
    # target_logits.shape: [batch_size, total_seq_len, draft_vocab_size]
    # draft_tokens[i].shape: [batch_size, total_seq_len]
    # loss_mask.shape: [batch_size, total_seq_len]

    accuracies = []
    target_tokens = torch.argmax(target_logits, dim=-1)
    # shape: [batch_size, total_seq_len]

    for step, drafts in enumerate(draft_tokens):
        correct = target_tokens[:, (step + 1) :] == drafts[:, : -(step + 1)]
        if loss_mask is not None:
            correct = correct[:, loss_mask[:, (step + 1) :]]
        accuracies.append(correct.float().mean())

    return torch.tensor(accuracies, device=target_logits.device)


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

        self.setup_model()
        self.setup_optimizer()

    def setup_model(self):
        self.model = self.model.to(self.local_rank)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.verifier_lm_head = self.verifier_lm_head.to(self.local_rank)

    def setup_optimizer(self):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def train_epoch(self, epoch: int):
        self.model.train()
        self.train_loader.batch_sampler.set_epoch(
            epoch
        )  # todo: check if this is safe to call

        if self.local_rank == 0:
            train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            train_loader = self.train_loader

        log_rank0(f"Training Epoch {epoch} started")

        for batch in train_loader:
            batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_logits = self.verifier_lm_head(batch["verifier_last_hidden_states"])
            del batch["verifier_last_hidden_states"]

            draft_tokens, loss = self.model(
                **batch, target_logits=target_logits, use_off_policy_tokens=True
            )  # set this in a better way

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            draft_accuracies = compute_draft_accuracy(
                target_logits, draft_tokens, batch["loss_mask"]
            )

            loss = loss.detach().clone()
            if self.is_distributed:
                # Note: this is not needed for training, just for logging
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
                dist.reduce(draft_accuracies, dst=0, op=dist.ReduceOp.AVG)

            log_rank0(loss.item())
            log_rank0(draft_accuracies.tolist())
            
        log_rank0(f"Training Epoch {epoch} completed")

    def val_epoch(self, epoch: int):
        if self.val_loader is None:
            log_rank0("No val loader, skipping validation")
            return
        self.model.eval()
        self.val_loader.batch_sampler.set_epoch(epoch)

        if self.local_rank == 0:
            val_loader = tqdm(self.val_loader, desc=f"Epoch {epoch}")
        else:
            val_loader = self.val_loader

        log_rank0(f"Validation Epoch {epoch} started")

        for batch in val_loader:
            batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_logits = self.verifier_lm_head(batch["verifier_last_hidden_states"])
            del batch["verifier_last_hidden_states"]

            draft_tokens, val_loss = self.model(
                **batch, target_logits=target_logits, use_off_policy_tokens=True
            )  # set this in a better way

            draft_accuracies = compute_draft_accuracy(
                target_logits, draft_tokens, batch["loss_mask"]
            )

            if self.is_distributed:
                dist.reduce(val_loss, dst=0, op=dist.ReduceOp.AVG)
                dist.reduce(draft_accuracies, dst=0, op=dist.ReduceOp.AVG)

            log_rank0(val_loss.item())
            log_rank0(draft_accuracies.tolist())
        
        log_rank0(f"Validation Epoch {epoch} completed")

    def save_checkpoint(self, epoch: int):
        os.makedirs(self.config["save_path"], exist_ok=True)
        save_path = f"{self.config['save_path']}/checkpoint_epoch_{epoch}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "epoch": epoch,
            },
            save_path,
        )
        log_rank0(f"Checkpoint saved to {save_path}")

    def run_training(self):
        for epoch in range(self.config["num_epochs"]):
            self.train_epoch(epoch)
            if self.is_distributed:
                dist.barrier()
            self.val_epoch(epoch)
            if self.local_rank == 0:
                self.save_checkpoint(epoch)
