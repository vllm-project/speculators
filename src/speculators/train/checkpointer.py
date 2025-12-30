from abc import abstractmethod
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
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
            scheduler_state_dict.pt (optional)
        1/
            model.safetensors
            optimizer_state_dict.pt
            scheduler_state_dict.pt (optional)
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
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        float_dtype: torch.dtype | None = None,
    ):
        raise NotImplementedError

    def load_scheduler_state_dict(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler
    ):
        scheduler_path = self.scheduler_path(self.previous_epoch)
        if not scheduler_path.exists():
            return
        full_state_dict = torch.load(scheduler_path, weights_only=True)
        scheduler.load_state_dict(full_state_dict)

    def save_scheduler_state_dict(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int
    ):
        scheduler_path = self.scheduler_path(epoch)
        torch.save(scheduler.state_dict(), scheduler_path)

    @abstractmethod
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        float_dtype: torch.dtype = torch.bfloat16,
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

    def scheduler_path(self, epoch: int):
        scheduler_fname = "scheduler_state_dict.pt"
        return self.path / str(epoch) / scheduler_fname


def convert_float_dtype(sd: pytree.PyTree, dtype: torch.dtype) -> pytree.PyTree:
    def convert_fn(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.to(dtype)
        return x

    return pytree.tree_map(convert_fn, sd)


def load_safetensors_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    full_state_dict = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():  # noqa: SIM118
            full_state_dict[key] = f.get_tensor(key)
    return full_state_dict


class SingleGPUCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        acc = torch.accelerator.current_accelerator()
        if acc is None:
            device = "cuda:0"
        else:
            acc_type = acc.type
            device_idx = torch.accelerator.current_device_index()
            device = f"{acc_type}:{device_idx}"
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch),
            device,  # todo: generalize
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )
        # Note: `strict=False` because we don't load the verifier weights
        model.load_state_dict(full_state_dict, strict=False)

    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,  # noqa: ARG002
        optimizer: torch.optim.Optimizer,
        float_dtype: torch.dtype | None = None,
    ):
        acc = torch.accelerator.current_accelerator()
        if acc is None:
            device = "cuda:0"
        else:
            acc_type = acc.type
            device_idx = torch.accelerator.current_device_index()
            device = f"{acc_type}:{device_idx}"
        full_state_dict = torch.load(
            self.optimizer_path(self.previous_epoch),
            weights_only=True,
            map_location=device,  # todo: generalize
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )
        optimizer.load_state_dict(full_state_dict)

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        model_state_dict = convert_float_dtype(model.state_dict(), float_dtype)
        model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
        optimizer_state_dict = convert_float_dtype(optimizer.state_dict(), float_dtype)
        torch.save(optimizer_state_dict, self.optimizer_path(epoch))


class DistributedCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch), "cpu"
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
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

    def load_optimizer_state_dict(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        float_dtype: torch.dtype | None = None,
    ):
        full_state_dict = torch.load(
            self.optimizer_path(self.previous_epoch),
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )

        set_optimizer_state_dict(
            model,
            optimizer,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        dist.barrier()

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        model_state_dict = convert_float_dtype(model_state_dict, float_dtype)

        optimizer_state_dict = get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        optimizer_state_dict = convert_float_dtype(optimizer_state_dict, float_dtype)

        if dist.get_rank() == 0:
            # Only rank 0 saves the checkpoint
            model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
            torch.save(optimizer_state_dict, self.optimizer_path(epoch))

        dist.barrier()
