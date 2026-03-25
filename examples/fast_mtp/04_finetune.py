"""Finetune a FastMTP speculator on pre-generated hidden state data.

Loads an existing FastMTP checkpoint (e.g., converted from the Qwen3-Next MTP
head), trains it on .pt files produced by generate_dataset.py, and saves
checkpoints after each epoch.

Usage (single GPU):
    python examples/fast_mtp/04_finetune.py \\
        --speculator-path Qwen3-Next-80B-A3B-Instruct_mtp_speculator \\
        --data-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \\
        --output-dir output/qwen3next_gsm8k_finetuned \\
        --max-len 4096 \\
        --lr 5e-5 \\
        --num-epochs 3 \\
        --batch-size 64

Usage (multi-GPU with torchrun):
    torchrun --standalone --nproc_per_node=8 \\
        examples/fast_mtp/04_finetune.py \\
        --speculator-path Qwen3-Next-80B-A3B-Instruct_mtp_speculator \\
        --data-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \\
        --output-dir output/qwen3next_gsm8k_finetuned \\
        --max-len 4096 --lr 5e-5 --num-epochs 3 --batch-size 8
"""

import argparse
import os

import torch
import torch.distributed as dist

from speculators.models.fast_mtp import FastMTPSpeculator
from speculators.train.fast_mtp_data import make_fast_mtp_dataloader
from speculators.train.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune FastMTP speculator")
    p.add_argument(
        "--speculator-path",
        required=True,
        help="Path to FastMTP checkpoint directory (output of convert_checkpoints.py)",
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
        help="Per-GPU batch size (effective = batch_size * num_gpus * grad_accum)",
    )
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument(
        "--scheduler-type",
        choices=["cosine", "linear", "none"],
        default="cosine",
    )
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument(
        "--step-weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("W0", "W1", "W2"),
        help="Per-step loss weights (default: 0.51 0.31 0.18)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best", action="store_true", help="Keep only best checkpoint")
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    return p.parse_args()


def setup_distributed() -> tuple[bool, int]:
    """Initialize distributed training if launched with torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    is_distributed, local_rank = setup_distributed()
    is_main = local_rank == 0

    if is_main:
        print(f"Loading FastMTP speculator from {args.speculator_path}")

    model = FastMTPSpeculator.from_pretrained(args.speculator_path)

    train_call_kwargs, val_call_kwargs = FastMTPSpeculator.get_trainer_kwargs(
        step_weights=args.step_weights,
    )

    if is_main:
        print(f"Building dataloaders from {args.data_dir}")
    train_loader, val_loader = make_fast_mtp_dataloader(
        data_dir=args.data_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
    )
    if is_main:
        print(
            f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}"
        )

    config = TrainerConfig(
        lr=args.lr,
        num_epochs=args.num_epochs,
        save_path=args.output_dir,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        scheduler_type=args.scheduler_type,
        scheduler_warmup_steps=args.warmup_steps,
        checkpoint_freq=args.checkpoint_freq,
        save_best=args.save_best,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    if is_main:
        print("Starting training ...")
    trainer.train()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
