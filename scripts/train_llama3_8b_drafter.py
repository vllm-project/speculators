import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.data import (
    Eagle3SampleFileDataset,
    create_collate_fn,
    split_files,
    standardize_data_v0,
    standardize_data_v1,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.eagle3.core import Eagle3DraftModel
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed

# Model
DRAFT_VOCAB_SIZE = 32000  # Must match t2d and d2t tensors
TOTAL_SEQ_LEN = 8192
VERIFIER_MODEL_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"
HIDDEN_SIZE = 4096  # Must match the verifier model's hidden size
VERIFIER_VOCAB_SIZE = 128256  # Must match the verifier model's vocab size
NORM_BEFORE_RESIDUAL = True

# Dataloader
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
NOISE_STD = 0.2


def setup_dataloader(
    file_list: list[str],
    world_size: int,
    local_rank: int,
    add_noise: bool = True,
    data_format_version: int = 1,
):
    if add_noise:
        noise_transform = AddUniformNoise(
            std=NOISE_STD, tensors=("hidden_states", "verifier_last_hidden_states")
        )
    else:
        noise_transform = None

    standardize_fn = (
        standardize_data_v1 if data_format_version == 1 else standardize_data_v0
    )

    dataset = Eagle3SampleFileDataset(
        file_list=file_list,
        max_len=TOTAL_SEQ_LEN,
        transform=noise_transform,
        standardize_fn=standardize_fn,
    )
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=TOTAL_SEQ_LEN,
        lengths=dataset.approx_lengths,
        num_replicas=world_size,
        rank=local_rank,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
        collate_fn=create_collate_fn(TOTAL_SEQ_LEN),
        persistent_workers=True,
    )


def main(args: argparse.Namespace):
    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    local_rank, world_size, rank, is_distributed = maybe_setup_distributed()
    device = torch.device(local_rank)

    # Setup speculator config
    llama_config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        vocab_size=VERIFIER_VOCAB_SIZE,
        num_hidden_layers=args.num_layers,
    )
    llama_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
    speculator_config = Eagle3SpeculatorConfig(
        transformer_layer_config=llama_config,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        norm_before_residual=NORM_BEFORE_RESIDUAL,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    proposal_type="greedy",
                    speculative_tokens=args.ttt_steps,
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=VERIFIER_MODEL_NAME_OR_PATH,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )

    # Load t2d and d2t tensors
    d2t = torch.from_numpy(np.load(args.d2t_path)).to(device)
    t2d = torch.from_numpy(np.load(args.t2d_path)).to(device)

    # Setup draft model
    draft_model = Eagle3DraftModel(
        config=speculator_config, t2d=t2d, d2t=d2t, ttt_steps=args.ttt_steps
    )

    # Setup dataloaders
    train_files, val_files = split_files(args.data_path, ratio=0.9)
    train_loader = setup_dataloader(
        train_files,
        world_size,
        local_rank,
        add_noise=True,
        data_format_version=args.data_format_version,
    )
    val_loader = setup_dataloader(
        val_files,
        world_size,
        local_rank,
        add_noise=False,
        data_format_version=args.data_format_version,
    )

    # Setup trainer
    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        save_path=args.save_path,
        lr=args.lr,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs={"use_off_policy_tokens": False},
        val_call_kwargs={"use_off_policy_tokens": False},
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    maybe_destroy_distributed()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-resume-from-checkpoint", action="store_true")
    parser.add_argument(
        "--logger",
        type=str,
        default="",
        help="One of 'trackio', 'wandb', 'tensorboard' or comma separated list of them",
    )
    parser.add_argument("--data-format-version", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--d2t-path", type=str, default="d2t.npy")
    parser.add_argument("--t2d-path", type=str, default="t2d.npy")
    parser.add_argument("--ttt-steps", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# RUN WITH:
# torchrun --nnodes=1 --nproc_per_node=<num_gpus>  scripts/train.py
# for FSDP training
# OR
# python scripts/train.py
# for single GPU training
