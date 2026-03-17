"""Convenience factory for building a :class:`~speculators.train.trainer.Trainer`
pre-configured for FastMTP fine-tuning.
"""

from pathlib import Path
from typing import Literal

import torch

from speculators.models.fast_mtp import FastMTPSpeculator
from speculators.train.fast_mtp_data import make_fast_mtp_dataloader
from speculators.train.trainer import Trainer, TrainerConfig

__all__ = ["build_fast_mtp_trainer"]


def build_fast_mtp_trainer(
    model: FastMTPSpeculator,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    max_len: int = 4096,
    lr: float = 5e-5,
    num_epochs: int = 3,
    batch_size: int = 1,
    train_ratio: float = 0.9,
    scheduler_type: Literal["linear", "cosine", "none"] = "linear",
    hidden_states_dtype: torch.dtype = torch.bfloat16,
) -> Trainer:
    """Build a :class:`~speculators.train.trainer.Trainer` for FastMTP fine-tuning.

    Assembles a :class:`~speculators.train.fast_mtp_data.FastMTPSampleFileDataset`
    backed by ``.pt`` files in *data_dir*, wires the per-step loss weights from
    :meth:`~speculators.models.fast_mtp.FastMTPSpeculator.get_trainer_kwargs`,
    and returns a ready-to-use :class:`~speculators.train.trainer.Trainer`.

    :param model: The :class:`~speculators.models.fast_mtp.FastMTPSpeculator`
        to fine-tune.
    :param data_dir: Directory of ``.pt`` files produced by
        :func:`~speculators.data_generation.fast_mtp_generator.generate_and_save_fast_mtp`.
    :param output_dir: Where to write checkpoints.
    :param max_len: Collation target sequence length.
    :param lr: AdamW learning rate.
    :param num_epochs: Number of training epochs.
    :param batch_size: Sequences per DataLoader batch.
    :param train_ratio: Fraction of files used for training (rest = validation).
    :param scheduler_type: LR scheduler: ``"linear"``, ``"cosine"``, or ``"none"``.
    :param hidden_states_dtype: dtype for hidden state tensors in the dataset.
    :return: Configured :class:`~speculators.train.trainer.Trainer`.
    """
    train_loader, val_loader = make_fast_mtp_dataloader(
        data_dir=data_dir,
        max_len=max_len,
        batch_size=batch_size,
        train_ratio=train_ratio,
        hidden_states_dtype=hidden_states_dtype,
    )

    train_kwargs, val_kwargs = FastMTPSpeculator.get_trainer_kwargs()

    config = TrainerConfig(
        lr=lr,
        num_epochs=num_epochs,
        save_path=str(output_dir),
        train_call_kwargs=train_kwargs,
        val_call_kwargs=val_kwargs,
        scheduler_type=scheduler_type,
    )

    return Trainer(model, config, train_loader, val_loader)
