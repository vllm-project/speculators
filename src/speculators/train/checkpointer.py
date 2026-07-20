import functools
import json
import logging
import os
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
from torch.nn.parallel import DistributedDataParallel
from transformers.modeling_utils import PreTrainedModel

from speculators.train.distributed import get_rank, is_distributed
from speculators.utils.util import get_current_device

logger = logging.getLogger("speculators")

# Optimizers/schedulers may be a single object (legacy) or a list (e.g. Muon + AdamW).
OptimizerOrList = torch.optim.Optimizer | list[torch.optim.Optimizer]
SchedulerOrList = (
    torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler]
)


def _as_list(value):
    """Normalize a single object or a list/tuple of objects into a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _rank0_only(fn):
    """Execute *fn* only on rank 0, then barrier (no-op when not distributed)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = None
        if get_rank() == 0:
            result = fn(*args, **kwargs)
        if is_distributed():
            dist.barrier()
        return result

    return wrapper


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
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        raise NotImplementedError

    def load_scheduler_state_dict(self, scheduler: SchedulerOrList):
        scheduler_path = self.scheduler_path(self.previous_epoch)
        if not scheduler_path.exists():
            return
        loaded = torch.load(scheduler_path, weights_only=True)
        schedulers = _as_list(scheduler)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        for sched, state_dict in zip(schedulers, loaded_list, strict=True):
            sched.load_state_dict(state_dict)

    @_rank0_only
    def save_scheduler_state_dict(self, scheduler: SchedulerOrList, epoch: int | str):
        schedulers = _as_list(scheduler)
        state_dicts = [sched.state_dict() for sched in schedulers]
        # Preserve the legacy single-scheduler format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.scheduler_path(epoch))

    @abstractmethod
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raise NotImplementedError

    def _get_previous_epoch(self) -> int:
        if not self.path.exists():
            return -1
        last_checkpoint_num = -1
        for d in self.path.iterdir():
            if d.is_symlink():
                continue  # skip descriptive symlinks like epoch0_step16626
            if d.is_dir():
                if d.name == "interrupted":
                    logger.warning(
                        f"Found interrupted checkpoint at {d}. "
                        "To resume from it, rename it to an epoch number "
                        "(e.g., 'mv interrupted 5' to resume as epoch 5)."
                    )
                    continue
                try:
                    epoch_num = int(d.name)
                except ValueError:
                    continue
                if not (d / self.COMPLETE_MARKER_FILENAME).exists():
                    logger.warning(
                        f"Skipping checkpoint at {d}: missing "
                        f"'{self.COMPLETE_MARKER_FILENAME}' marker (save was "
                        "interrupted, or the checkpoint predates this version). "
                        "To resume from it anyway, create the marker: "
                        f"'touch {d / self.COMPLETE_MARKER_FILENAME}'."
                    )
                    continue
                last_checkpoint_num = max(last_checkpoint_num, epoch_num)
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

    TRAIN_COMMAND_FILENAME = "train_command.txt"
    COMPLETE_MARKER_FILENAME = "checkpoint_complete"

    def complete_marker_path(self, epoch: int | str) -> Path:
        return self.path / str(epoch) / self.COMPLETE_MARKER_FILENAME

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        dir_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def clear_checkpoint_complete(self, epoch: int | str) -> None:
        """Durably remove a stale marker before rewriting an epoch dir."""
        epoch_dir = self.path / str(epoch)
        marker = self.complete_marker_path(epoch)
        if marker.exists():
            marker.unlink()
            self._fsync_directory(epoch_dir)

    def mark_checkpoint_complete(self, epoch: int | str) -> None:
        """Sync checkpoint files, then atomically publish the marker."""
        epoch_dir = self.path / str(epoch)
        for f in epoch_dir.rglob("*"):
            if f.is_file() and not f.is_symlink():
                with f.open("rb") as fh:
                    os.fsync(fh.fileno())

        marker = self.complete_marker_path(epoch)
        temporary_marker = marker.with_name(f".{marker.name}.tmp")
        try:
            with temporary_marker.open("wb") as fh:
                os.fsync(fh.fileno())
            os.replace(temporary_marker, marker)
            self._fsync_directory(epoch_dir)
        except Exception:
            temporary_marker.unlink(missing_ok=True)
            raise

    def _copy_train_command(self, epoch: int | str) -> None:
        src = self.path / self.TRAIN_COMMAND_FILENAME
        if src.exists():
            shutil.copy2(src, self.path / str(epoch) / self.TRAIN_COMMAND_FILENAME)

    @_rank0_only
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

    @_rank0_only
    def update_best_symlink(self, epoch: int):
        best_path = self.best_path()
        target = Path(str(epoch))  # relative symlink inside checkpoint root

        if best_path.is_symlink() or best_path.exists():
            if best_path.is_dir() and not best_path.is_symlink():
                shutil.rmtree(best_path)
            else:
                best_path.unlink()

        best_path.symlink_to(target, target_is_directory=True)

    @_rank0_only
    def cleanup_keep_only_best(self, best_epoch: int) -> None:
        """
        Delete all epoch dir. except best_epoch, and keep best_checkpoint symlink.
        """
        keep_dir = self.path / str(best_epoch)
        best_link = self.best_path()

        # Safety checks
        if not keep_dir.exists() or not keep_dir.is_dir():
            raise FileNotFoundError(f"Best epoch dir does not exist: {keep_dir}")

        train_cmd_file = self.path / self.TRAIN_COMMAND_FILENAME

        for child in self.path.iterdir():
            # Keep the symlink itself
            if child == best_link:
                continue

            # Keep the best epoch directory
            if child == keep_dir:
                continue

            if child == train_cmd_file:
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


def patch_config_dtype(config_path: Path, float_dtype: torch.dtype) -> None:
    """Patch config.json to match the actual on-disk tensor dtype.

    When models are kept in FP32 but saved as BF16, save_pretrained writes
    the in-memory dtype to config.json. This patches it to match the saved dtype.
    """
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    # Convert torch.bfloat16 -> "bfloat16"
    dtype_str = str(float_dtype).split(".")[-1]
    # Support both dtype (transformers 5.x) and torch_dtype (older versions)
    if "dtype" in config:
        config["dtype"] = dtype_str
    if "torch_dtype" in config:
        config["torch_dtype"] = dtype_str
    config_path.write_text(json.dumps(config, indent=2) + "\n")


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
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        device = get_current_device()
        loaded = torch.load(
            self.optimizer_path(self.previous_epoch),
            weights_only=True,
            map_location=device,
        )
        optimizers = _as_list(optimizer)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        dtype = float_dtype or raw_model.dtype
        for opt, state_dict in zip(optimizers, loaded_list, strict=True):
            opt.load_state_dict(convert_float_dtype(state_dict, dtype))

    @_rank0_only
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raw_model: PreTrainedModel = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )  # type: ignore[assignment]
        model_state_dict = convert_float_dtype(raw_model.state_dict(), float_dtype)
        raw_model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
        patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)

        optimizers = _as_list(optimizer)
        state_dicts = [
            convert_float_dtype(opt.state_dict(), float_dtype) for opt in optimizers
        ]
        # Preserve the legacy single-optimizer format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.optimizer_path(epoch))
        self._copy_train_command(epoch)


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
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        optimizers = _as_list(optimizer)
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
            optimizers,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

        # Cast step counters back to float32
        for opt in optimizers:
            for state in opt.state.values():
                if "step" in state and isinstance(state["step"], torch.Tensor):
                    state["step"] = state["step"].float()

        dist.barrier()

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        model_state_dict = convert_float_dtype(model_state_dict, float_dtype)

        optimizer_state_dict = get_optimizer_state_dict(
            model,
            _as_list(optimizer),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        optimizer_state_dict = convert_float_dtype(optimizer_state_dict, float_dtype)

        if get_rank() == 0:
            # Only rank 0 saves the checkpoint
            model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
            patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)
            torch.save(optimizer_state_dict, self.optimizer_path(epoch))
            self._copy_train_command(epoch)

        dist.barrier()
