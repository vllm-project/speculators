import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaConfig
from transformers.models.auto.configuration_auto import AutoConfig

from speculators.data_generation.preprocessing import load_and_preprocess_dataset
from speculators.model import SpeculatorModel
from speculators.train.data import (
    Eagle3OnlineVLLMDataset,
    create_collate_fn,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed
from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_if_needed(
    preprocess_dir: str,
    verifier_name_or_path: str,
    dataset_name: str,
    total_seq_len: int,
    preprocess_num_workers: int,
    overwrite: bool,
    rank: int,
    is_distributed: bool,
) -> str:
    """Run preprocessing to produce sample_lengths.json and token_freq.pt.

    Only rank 0 performs the work; other ranks wait at a barrier.
    Returns the path to sample_lengths.json.
    """
    preprocess_path = Path(preprocess_dir)
    lengths_path = preprocess_path / "sample_lengths.json"
    token_freq_path = preprocess_path / "token_freq.pt"

    needs_preprocessing = (
        overwrite or not lengths_path.exists() or not token_freq_path.exists()
    )

    if rank == 0:
        if needs_preprocessing:
            log.info("Running preprocessing (tokenization + sample lengths)...")
            preprocess_path.mkdir(parents=True, exist_ok=True)

            dataset, _tokenizer = load_and_preprocess_dataset(
                target_model_path=verifier_name_or_path,
                train_data_path=dataset_name,
                seq_length=total_seq_len,
                build_dataset_num_proc=preprocess_num_workers,
                seed=0,
                token_freq_path=str(token_freq_path),
            )

            sample_lengths = [
                len(dataset[idx]["input_ids"])
                for idx in tqdm(range(len(dataset)), desc="Computing sample lengths")
            ]
            with open(lengths_path, "w") as f:
                json.dump(sample_lengths, f)

            log.info(
                f"Preprocessing complete: {len(sample_lengths)} samples saved to "
                f"{preprocess_dir}"
            )
        else:
            log.info(
                f"Preprocessing outputs already exist in {preprocess_dir}, skipping."
            )

    if is_distributed:
        dist.barrier()

    return str(lengths_path)


def build_vocab_mappings_if_needed(
    preprocess_dir: str,
    draft_vocab_size: int | None,
    target_vocab_size: int,
    overwrite: bool,
    rank: int,
    is_distributed: bool,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Build d2t/t2d vocab mappings if --draft-vocab-size is set.

    Returns (d2t, t2d, draft_vocab_size). If draft_vocab_size is None,
    returns (None, None, target_vocab_size) for full-vocab mode.
    """
    if draft_vocab_size is None:
        return None, None, target_vocab_size

    preprocess_path = Path(preprocess_dir)
    d2t_path = preprocess_path / "d2t.npy"
    t2d_path = preprocess_path / "t2d.npy"
    token_freq_path = preprocess_path / "token_freq.pt"

    needs_build = overwrite or not d2t_path.exists() or not t2d_path.exists()

    if rank == 0 and needs_build:
        log.info(
            f"Building vocab mappings (draft_vocab_size={draft_vocab_size}, "
            f"target_vocab_size={target_vocab_size})..."
        )
        token_freq_dict = torch.load(token_freq_path, weights_only=True)
        d2t, t2d = build_vocab_mappings_from_distribution(
            token_freq_dict=token_freq_dict,
            draft_vocab_size=draft_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        np.save(d2t_path, d2t.cpu().numpy())
        np.save(t2d_path, t2d.cpu().numpy())
        log.info(f"Saved vocab mappings to {preprocess_dir}")
    elif rank == 0:
        log.info(f"Vocab mappings already exist in {preprocess_dir}, skipping.")

    if is_distributed:
        dist.barrier()

    d2t = torch.from_numpy(np.load(d2t_path)).to(device)
    t2d = torch.from_numpy(np.load(t2d_path)).to(device)
    return d2t, t2d, draft_vocab_size


def setup_dataloader(
    dataset_name: str,
    model: str,
    vllm_url: str,
    lengths_file: str,
    total_seq_len: int,
    world_size: int,
    local_rank: int,
    add_noise: bool = True,
    noise_std: float = 0.05,
    num_workers: int = 12,
    prefetch_factor: int = 4,
) -> DataLoader:
    """Setup dataloader for training.
    Args:
        dataset_name: Name of the dataset or path to JSON/JSONL file.
        model: HuggingFace model ID or local path.
        vllm_url: URL of the VLLM server.
        lengths_file: Path to sample_lengths.json from preprocess_data.py.
        total_seq_len: Maximum total sequence length for batch packing.
        world_size: Number of processes in the distributed training.
        local_rank: Rank of the current process.
        add_noise: Whether to add noise to the data.
        noise_std: Standard deviation for noise augmentation.
        num_workers: Number of dataloader workers.
        prefetch_factor: Dataloader prefetch factor.
    Returns:
        DataLoader: Dataloader for training.
    """
    if add_noise:
        noise_transform = AddUniformNoise(
            std=noise_std, tensors=("hidden_states", "verifier_last_hidden_states")
        )
    else:
        noise_transform = None

    dataset = Eagle3OnlineVLLMDataset(
        max_len=total_seq_len,
        dataset=dataset_name,
        vllm_url=vllm_url,
        model=model,
        lengths_file=lengths_file,
        transform=noise_transform,
    )
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=world_size,
        rank=local_rank,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        collate_fn=create_collate_fn(total_seq_len),
        persistent_workers=True,
    )


def create_transformer_layer_config(
    verifier_name_or_path: str, num_layers: int
) -> LlamaConfig:
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)

    # For multimodal models (Qwen3VL, etc.), extract text_config
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    transformer_layer_config = LlamaConfig(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=verifier_config.intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=verifier_config.num_attention_heads,
        num_key_value_heads=verifier_config.num_key_value_heads,
        hidden_act=verifier_config.hidden_act,
        max_position_embeddings=verifier_config.max_position_embeddings,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=getattr(verifier_config, "head_dim", None),
    )
    transformer_layer_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
    return transformer_layer_config


def main(args: argparse.Namespace):
    # Set random seed for reproducibility
    set_seed(args.seed, args.deterministic_cuda)

    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    local_rank, world_size, rank, is_distributed = maybe_setup_distributed()
    device = torch.device(local_rank)

    # Preprocessing: tokenize dataset, compute sample lengths and token frequencies
    lengths_file = preprocess_if_needed(
        preprocess_dir=args.preprocess_dir,
        verifier_name_or_path=args.verifier_name_or_path,
        dataset_name=args.dataset_name,
        total_seq_len=args.total_seq_len,
        preprocess_num_workers=args.preprocess_num_workers,
        overwrite=args.overwrite_preprocess,
        rank=rank,
        is_distributed=is_distributed,
    )

    # Load verifier config (needed for target_vocab_size and transformer layer config)
    verifier_config = AutoConfig.from_pretrained(args.verifier_name_or_path)
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    target_vocab_size = verifier_config.vocab_size

    # Build vocab mappings if draft-vocab-size is specified
    d2t, t2d, draft_vocab_size = build_vocab_mappings_if_needed(
        preprocess_dir=args.preprocess_dir,
        draft_vocab_size=args.draft_vocab_size,
        target_vocab_size=target_vocab_size,
        overwrite=args.overwrite_preprocess,
        rank=rank,
        is_distributed=is_distributed,
        device=device,
    )

    # Setup speculator config
    transformer_layer_config = create_transformer_layer_config(
        args.verifier_name_or_path, args.num_layers
    )

    # Get model class from registry and create model using its factory method
    if SpeculatorModel.registry_auto_discovery:
        SpeculatorModel.auto_populate_registry()

    if args.speculator_type not in SpeculatorModel.registry:
        raise ValueError(
            f"Unknown speculator type: {args.speculator_type}. "
            f"Available: {list(SpeculatorModel.registry.keys())}"
        )

    model_class = SpeculatorModel.registry[args.speculator_type]
    draft_model = model_class.from_training_args(
        verifier_config=transformer_layer_config,
        t2d=t2d,
        d2t=d2t,
        draft_vocab_size=draft_vocab_size,
        **vars(args),
    )

    # Setup dataloaders
    train_loader = setup_dataloader(
        args.dataset_name,
        args.verifier_name_or_path,
        args.vllm_url,
        lengths_file,
        args.total_seq_len,
        world_size,
        local_rank,
        add_noise=True,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = setup_dataloader(
        args.dataset_name,
        args.verifier_name_or_path,
        args.vllm_url,
        lengths_file,
        args.total_seq_len,
        world_size,
        local_rank,
        add_noise=False,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # Get trainer kwargs from model class
    train_call_kwargs, val_call_kwargs = model_class.get_trainer_kwargs(**vars(args))

    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        save_path=args.save_path,
        lr=args.lr,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        scheduler_type=args.scheduler_type,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_total_steps=args.scheduler_total_steps,
        scheduler_num_cosine_cycles=args.scheduler_num_cosine_cycles,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    maybe_destroy_distributed()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument(
        "--speculator-type",
        type=str,
        default="eagle3",
        help="Type of speculator model to train (e.g., eagle3)",
    )
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset or path to JSON/JSONL file (e.g., 'sharegpt')",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the VLLM server",
    )
    # Preprocessing arguments
    parser.add_argument(
        "--preprocess-dir",
        type=str,
        default="./preprocessed_data",
        help="Cache directory for preprocessing outputs (sample_lengths.json, "
        "token_freq.pt, d2t.npy, t2d.npy)",
    )
    parser.add_argument(
        "--preprocess-num-workers",
        type=int,
        default=8,
        help="Number of CPU workers for tokenization during preprocessing",
    )
    parser.add_argument(
        "--overwrite-preprocess",
        action="store_true",
        help="Force re-run of preprocessing even if cached outputs exist",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help="If set, builds vocab mappings (d2t/t2d) with this draft vocabulary size. "
        "If not set, uses the full verifier vocabulary (no mapping).",
    )
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
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument("--data-format-version", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic-cuda",
        action="store_true",
        default=False,
        help="Sets cuda to deterministic mode. This may impact performance.",
    )
    parser.add_argument(
        "--use-off-policy-tokens",
        action="store_true",
        default=False,
        help="Use off-policy tokens during training (required for regenerated data)",
    )
    # Model hyperparameters
    parser.add_argument(
        "--norm-before-residual",
        action="store_true",
        default=True,
        help="Whether to normalize before residual connection",
    )
    # Dataloader parameters
    parser.add_argument(
        "--num-workers", type=int, default=12, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=4, help="Dataloader prefetch factor"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Standard deviation for noise augmentation",
    )
    # lr scheduler
    parser.add_argument("--scheduler-type", type=str, default="linear")
    parser.add_argument("--scheduler-warmup-steps", type=int, default=None)
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--scheduler-num-cosine-cycles", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# RUN WITH:
# torchrun --nnodes=1 --nproc_per_node=<num_gpus>  scripts/train_online.py
# for FSDP training
# OR
# python scripts/train_online.py --verifier-name-or-path <model> --dataset-name sharegpt
# for single GPU training
