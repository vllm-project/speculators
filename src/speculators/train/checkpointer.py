import json
import logging
import shutil
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

from speculators.utils.util import get_current_device

logger = logging.getLogger("speculators")


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
        self, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int | str
    ):
        scheduler_path = self.scheduler_path(epoch)
        torch.save(scheduler.state_dict(), scheduler_path)

    @abstractmethod
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raise NotImplementedError

    def _get_previous_epoch(self) -> int:
        if not self.path.exists():
            return -1
        last_checkpoint_num = -1
        for d in self.path.iterdir():
            if d.is_dir():
                if d.name == "interrupted":
                    logger.warning(
                        f"Found interrupted checkpoint at {d}. "
                        "To resume from it, rename it to an epoch number "
                        "(e.g., 'mv interrupted 5' to resume as epoch 5)."
                    )
                    continue
                try:
                    last_checkpoint_num = max(last_checkpoint_num, int(d.name))
                except ValueError:
                    continue
        return last_checkpoint_num

    def model_path(self, epoch: int | str):
        model_fname = "model.safetensors"
        return self.path / str(epoch) / model_fname

    def optimizer_path(self, epoch: int | str):
        optimizer_fname = "optimizer_state_dict.pt"
        return self.path / str(epoch) / optimizer_fname

    def scheduler_path(self, epoch: int | str):
        scheduler_fname = "scheduler_state_dict.pt"
        return self.path / str(epoch) / scheduler_fname

    def best_path(self) -> Path:
        return self.path / "checkpoint_best"

    def val_metrics_path(self, epoch: int) -> Path:
        return self.path / str(epoch) / "val_metrics.json"

    def save_val_metrics(self, epoch: int, val_metrics: dict[str, float]):
        path = self.val_metrics_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(val_metrics))

    def load_best_val_loss(self) -> float | None:
        best_epoch = self.read_best_epoch()
        if best_epoch is None:
            return None
        p = self.val_metrics_path(best_epoch)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return float(data["loss_epoch"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def read_best_epoch(self) -> int | None:
        """Return the epoch that `checkpoint_best` points to."""
        best_path = self.best_path()
        if not best_path.exists() or not best_path.is_symlink():
            return None
        try:
            target = best_path.readlink()
        except OSError:
            return None
        try:
            return int(Path(target).name)
        except ValueError:
            return None

    def load_model_state_dict_for_epoch(
        self, model: PreTrainedModel, epoch: int, float_dtype: torch.dtype | None = None
    ):
        """Temporarily load weights for a specific epoch."""
        old_epoch = self.previous_epoch
        try:
            self.previous_epoch = epoch
            self.load_model_state_dict(model, float_dtype=float_dtype)
        finally:
            self.previous_epoch = old_epoch

    def update_best_symlink(self, epoch: int):
        best_path = self.best_path()
        target = Path(str(epoch))  # relative symlink inside checkpoint root

        if best_path.is_symlink() or best_path.exists():
            if best_path.is_dir() and not best_path.is_symlink():
                shutil.rmtree(best_path)
            else:
                best_path.unlink()

        best_path.symlink_to(target, target_is_directory=True)

    def cleanup_keep_only_best(self, best_epoch: int) -> None:
        """
        Delete all epoch dir. except best_epoch, and keep best_checkpoint symlink.
        """
        keep_dir = self.path / str(best_epoch)
        best_link = self.best_path()

        # Safety checks
        if not keep_dir.exists() or not keep_dir.is_dir():
            raise FileNotFoundError(f"Best epoch dir does not exist: {keep_dir}")

        for child in self.path.iterdir():
            # Keep the symlink itself
            if child == best_link:
                continue

            # Keep the best epoch directory
            if child == keep_dir:
                continue

            # Delete numbered epoch directories and any other stray dirs/files
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except (FileNotFoundError, PermissionError, OSError) as exc:
                raise RuntimeError(f"Failed to delete {child}") from exc


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
        device = get_current_device()
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch),
            device,
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )
        # Note: `strict=False` because we don't load the verifier weights
        model.load_state_dict(full_state_dict, strict=False)

    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        float_dtype: torch.dtype | None = None,
    ):
        device = get_current_device()
        full_state_dict = torch.load(
            self.optimizer_path(self.previous_epoch),
            weights_only=True,
            map_location=device,
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )
        optimizer.load_state_dict(full_state_dict)

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        epoch: int | str,
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

        # Cast step counters back to float32
        for state in optimizer.state.values():
            if "step" in state and isinstance(state["step"], torch.Tensor):
                state["step"] = state["step"].float()

        dist.barrier()

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        epoch: int | str,
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

    def update_best_symlink(self, epoch: int):
        if dist.get_rank() == 0:
            super().update_best_symlink(epoch)

        dist.barrier()

    def cleanup_keep_only_best(self, best_epoch: int) -> None:
        if dist.get_rank() == 0:
            super().cleanup_keep_only_best(best_epoch)

        dist.barrier()

    def save_val_metrics(self, epoch: int, val_metrics: dict[str, float]):
        if dist.get_rank() == 0:
            super().save_val_metrics(epoch, val_metrics)
        dist.barrier()

    def save_scheduler_state_dict(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int | str
    ):
        if dist.get_rank() == 0:
            super().save_scheduler_state_dict(scheduler, epoch)

        dist.barrier()
