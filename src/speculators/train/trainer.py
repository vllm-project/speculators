import os
import torch
from torch.utils.data import DataLoader
from tqdm.rich import tqdm  # todo: requries tqdm and rich

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from speculators.train.utils import log_rank0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        is_distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
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

        for batch in train_loader:
            batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            _, loss = self.model(
                **batch, use_off_policy_tokens=True
            )  # set this in a better way

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss = loss.detach().clone()
            if self.is_distributed:
                # Note: this is not needed for training, just for logging
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)

            log_rank0(loss.item())

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

    def train(self):
        for epoch in range(self.config["num_epochs"]):
            self.train_epoch(epoch)
            if self.is_distributed:
                dist.barrier()
            log_rank0(f"Epoch {epoch} completed")
            if self.local_rank == 0:
                self.save_checkpoint(epoch)
