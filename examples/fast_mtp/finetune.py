"""Finetune a FastMTP speculator on pre-generated hidden state data.

Loads an existing FastMTP checkpoint (e.g., converted from the Qwen3-Next MTP
head), trains it on .pt files produced by generate_dataset.py, and saves
checkpoints after each epoch.

Usage (single GPU):
    python examples/fast_mtp/finetune.py \\
        --speculator-path Qwen3-Next-80B-A3B-Instruct_mtp_speculator \\
        --data-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \\
        --output-dir output/qwen3next_gsm8k_finetuned \\
        --max-len 4096 --lr 5e-5 --num-epochs 3 --batch-size 64

Usage (multi-GPU with torchrun):
    torchrun --standalone --nproc_per_node=4 \\
        examples/fast_mtp/finetune.py \\
        --speculator-path Qwen3-Next-80B-A3B-Instruct_mtp_speculator \\
        --data-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \\
        --output-dir output/qwen3next_gsm8k_finetuned \\
        --max-len 4096 --lr 5e-5 --num-epochs 3 --batch-size 16 \\
        --logger wandb --run-name fastmtp_gsm8k_{time}
"""

import argparse
import random

import numpy as np
import torch

from speculators.models.fast_mtp import FastMTPSpeculator
from speculators.train.fast_mtp_data import make_fast_mtp_dataloader
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune FastMTP speculator")
    p.add_argument(
        "--speculator-path",
        required=True,
        help="Path to FastMTP checkpoint directory (output of convert_checkpoint.py)",
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory of .pt training files from generate_dataset.py",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save training checkpoints",
    )
    p.add_argument("--max-len", type=int, default=4096, help="Max sequence length")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size",
    )
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument(
        "--scheduler-type",
        choices=["cosine", "linear", "none"],
        default="cosine",
    )
    p.add_argument("--scheduler-warmup-steps", type=int, default=None)
    p.add_argument("--scheduler-total-steps", type=int, default=None)
    p.add_argument("--scheduler-num-cosine-cycles", type=float, default=0.5)
    p.add_argument(
        "--step-weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("W0", "W1", "W2"),
        help="Per-step MTP loss weights for steps 0, 1, 2 "
        "(default: 0.51 0.31 0.18, β=0.6 exponential decay, normalized). "
        "Must have exactly 3 values matching num_speculative_steps.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic-cuda",
        action="store_true",
        default=False,
        help="Sets cuda to deterministic mode. May impact performance.",
    )
    p.add_argument("--save-best", action="store_true", help="Keep only best checkpoint")
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    p.add_argument("--no-resume-from-checkpoint", action="store_true")
    p.add_argument(
        "--logger",
        type=str,
        default="",
        help="One of 'wandb', 'tensorboard', 'trackio' or comma-separated list",
    )
    p.add_argument("--log-dir", type=str, default="./logs")
    p.add_argument(
        "--run-name",
        type=str,
        default="fastmtp_{time}",
        help="Run name for logging. ``{time}`` is expanded to a timestamp by "
        "the logger setup (default: %(default)s).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic_cuda)

    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    local_rank, _world_size, _rank, is_distributed = maybe_setup_distributed()

    model = FastMTPSpeculator.from_pretrained(args.speculator_path)

    train_call_kwargs, val_call_kwargs = FastMTPSpeculator.get_trainer_kwargs(
        step_weights=args.step_weights,
    )

    train_loader, val_loader = make_fast_mtp_dataloader(
        data_dir=args.data_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        hidden_size=model.config.hidden_size,
        train_ratio=args.train_ratio,
    )

    config = TrainerConfig(
        lr=args.lr,
        num_epochs=args.num_epochs,
        save_path=args.output_dir,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        scheduler_type=args.scheduler_type,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_total_steps=args.scheduler_total_steps,
        scheduler_num_cosine_cycles=args.scheduler_num_cosine_cycles,
        checkpoint_freq=args.checkpoint_freq,
        save_best=args.save_best,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.run_training()

    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
