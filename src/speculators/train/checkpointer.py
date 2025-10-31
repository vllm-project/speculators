from abc import abstractmethod
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from transformers.modeling_utils import PreTrainedModel


class BaseCheckpointer:
    """Helper class to save and load checkpoints.

    Checkpoint file structure:
    ../path/
        0/ # epoch number
            model.safetensors
            optimizer_state_dict.pt
        1/
            model.safetensors
            optimizer_state_dict.pt
        ...
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.previous_epoch = self._get_previous_epoch()

        if self.previous_epoch != -1:
            self.prev_path: Path | None = self.path / str(self.previous_epoch)
        else:
            self.prev_path = None

    @abstractmethod
    def load_model_state_dict(self, model: PreTrainedModel):
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_state_dict(
        self, model: PreTrainedModel, optimizer: torch.optim.Optimizer
    ):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
        self, model: PreTrainedModel, optimizer: torch.optim.Optimizer, epoch: int
    ):
        raise NotImplementedError

    def _get_previous_epoch(self) -> int:
        if not self.path.exists():
            return -1
        last_checkpoint_num = -1
        for d in self.path.iterdir():
            if d.is_dir():
                try:
                    last_checkpoint_num = max(last_checkpoint_num, int(d.name))
                except ValueError:
                    continue
        return last_checkpoint_num

    def model_path(self, epoch: int):
        model_fname = "model.safetensors"
        return self.path / str(epoch) / model_fname

    def optimizer_path(self, epoch: int):
        optimizer_fname = "optimizer_state_dict.pt"
        return self.path / str(epoch) / optimizer_fname


def load_safetensors_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    full_state_dict = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():  # noqa: SIM118
            full_state_dict[key] = f.get_tensor(key)
    return full_state_dict


class SingleGPUCheckpointer(BaseCheckpointer):
    def load_model_state_dict(self, model: PreTrainedModel):
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch), "cuda:0"
        )
        # Note: `strict=False` because we don't load the verifier weights
        model.load_state_dict(full_state_dict, strict=False)

    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,  # noqa: ARG002
        optimizer: torch.optim.Optimizer,
    ):
        full_state_dict = torch.load(
            self.optimizer_path(self.previous_epoch),
            weights_only=True,
            map_location="cuda:0",  # todo: make this configurable
        )
        optimizer.load_state_dict(full_state_dict)

    def save_checkpoint(
        self, model: PreTrainedModel, optimizer: torch.optim.Optimizer, epoch: int
    ):
        model.save_pretrained(self.path / str(epoch))
        torch.save(optimizer.state_dict(), self.optimizer_path(epoch))


class DistributedCheckpointer(BaseCheckpointer):
    def load_model_state_dict(self, model: PreTrainedModel):
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch), "cpu"
        )
        # Note: `strict=False` because we don't load the verifier weights
        set_model_state_dict(
            model,
            full_state_dict,  # type: ignore[arg-type]
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )
        dist.barrier()

    def load_optimizer_state_dict(self, model, optimizer: torch.optim.Optimizer):
        full_state_dict = torch.load(
            self.optimizer_path(self.previous_epoch),
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        set_optimizer_state_dict(
            model,
            optimizer,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        dist.barrier()

    def save_checkpoint(
        self, model: PreTrainedModel, optimizer: torch.optim.Optimizer, epoch: int
    ):
        model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        optimizer_state_dict = get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        if dist.get_rank() == 0:
            # Only rank 0 saves the checkpoint
            model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
            torch.save(optimizer_state_dict, self.optimizer_path(epoch))

        dist.barrier()
