import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig
from transformers.models.auto.configuration_auto import AutoConfig

from speculators.config import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
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
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed
from speculators.utils.loading import extract_vocab_mappings, load_full_state_dict

logger = logging.getLogger(__name__)

# DRAFTER MODEL HYPARAMETERS
NORM_BEFORE_RESIDUAL = True

# Dataloader
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
NOISE_STD = 0.05


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


def load_pretrained_model(
    pretrained_path: str, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """
    Load pretrained EAGLE3 model and extract components.

    Returns:
        Tuple of (state_dict, d2t, t2d, draft_vocab_size)
    """
    logger.info(f"Loading pretrained model from {pretrained_path}")

    # Load full state dict
    state_dict = load_full_state_dict(pretrained_path)

    # Extract vocab mappings
    d2t, t2d = extract_vocab_mappings(state_dict, device)

    # Derive draft_vocab_size
    draft_vocab_size = d2t.shape[0]
    logger.info(f"Derived draft_vocab_size={draft_vocab_size}")

    return state_dict, d2t, t2d, draft_vocab_size


def load_vocab_mappings(
    d2t_path: str, t2d_path: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load vocabulary mappings from numpy files."""
    if not (d2t_path and t2d_path):
        raise ValueError(
            "Both d2t and t2d paths must be provided together. "
            f"Got d2t={'provided' if d2t_path else 'missing'}, "
            f"t2d={'provided' if t2d_path else 'missing'}"
        )

    d2t = torch.from_numpy(np.load(d2t_path)).to(device)
    t2d = torch.from_numpy(np.load(t2d_path)).to(device)
    draft_vocab_size = d2t.shape[0]

    return d2t, t2d, draft_vocab_size


def initialize_vocab_config(
    args: argparse.Namespace, device: torch.device
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    dict[str, torch.Tensor] | None,
]:
    """
    Initialize vocabulary configuration from args.

    Returns:
        Tuple of (d2t, t2d, draft_vocab_size, pretrained_state_dict)
    """
    # Check for conflicting args
    if args.pretrained_model_path and (args.d2t_path or args.t2d_path):
        raise ValueError(
            "--pretrained-model-path overrides --d2t-path and "
            "--t2d-path. Please remove --d2t-path and --t2d-path."
        )

    # Load from pretrained model
    if args.pretrained_model_path:
        state_dict, d2t, t2d, vocab_size = load_pretrained_model(
            args.pretrained_model_path,
            device,
        )
        return d2t, t2d, vocab_size, state_dict

    # Load from numpy files
    if args.d2t_path or args.t2d_path:
        d2t, t2d, vocab_size = load_vocab_mappings(args.d2t_path, args.t2d_path, device)
        return d2t, t2d, vocab_size, None

    # No vocab mapping provided
    verifier_config = AutoConfig.from_pretrained(args.verifier_name_or_path)
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    return None, None, verifier_config.vocab_size, None


def load_model_weights(
    model: Eagle3DraftModel,
    state_dict: dict[str, torch.Tensor],
    model_path: str,
):
    """Load pretrained weights into model with validation."""
    logger.info(f"Loading pretrained weights from {model_path}")
    logger.info(f"Parameters to load: {len(state_dict)}")

    # Load with strict=False (d2t/t2d passed to constructor)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Filter expected missing keys
    expected_missing = {"t2d", "d2t"}
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]

    # Report issues
    if unexpected_missing:
        logger.warning(f"Unexpected missing keys: {unexpected_missing}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    # Summary
    if unexpected_missing or unexpected_keys:
        logger.warning(
            "Weight loading completed with warnings. "
            "May indicate architecture mismatch."
        )
    else:
        logger.info("✓ Successfully loaded all weights")

    logger.info("Fine-tuning from pretrained weights.")
    logger.info("Note: Optimizer state starts fresh.")


def setup_dataloader(
    file_list: list[str],
    world_size: int,
    local_rank: int,
    add_noise: bool = True,
    data_format_version: int = 1,
) -> DataLoader:
    """Setup dataloader for training.
    Args:
        file_list: List of file paths to load data from.
        world_size: Number of processes in the distributed training.
        local_rank: Rank of the current process.
        add_noise: Whether to add noise to the data.
        data_format_version: Version of the data format. Default is 1.
    Returns:
        DataLoader: Dataloader for training.
    """
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
        max_len=args.total_seq_len,
        transform=noise_transform,
        standardize_fn=standardize_fn,
    )
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=args.total_seq_len,
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
        collate_fn=create_collate_fn(args.total_seq_len),
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
        rope_theta=getattr(verifier_config, "rope_theta", 10000.0),
        rope_scaling=getattr(verifier_config, "rope_scaling", None),
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

    # Initialize vocabulary configuration
    d2t, t2d, draft_vocab_size, pretrained_state_dict = initialize_vocab_config(
        args, device
    )

    # Setup speculator config
    # If finetuning, preserve the transformer_layer_config from pretrained model
    if args.pretrained_model_path:
        pretrained_config = SpeculatorModelConfig.from_pretrained(
            args.pretrained_model_path
        )
        transformer_layer_config = pretrained_config.transformer_layer_config
        logger.info(
            "Using transformer_layer_config from pretrained model "
            f"(rope_theta={transformer_layer_config.rope_theta})"
        )
    else:
        transformer_layer_config = create_transformer_layer_config(
            args.verifier_name_or_path, args.num_layers
        )

    speculator_config = Eagle3SpeculatorConfig(
        transformer_layer_config=transformer_layer_config,
        draft_vocab_size=draft_vocab_size,
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
                name_or_path=args.verifier_name_or_path,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )

    # Setup draft model
    draft_model = Eagle3DraftModel(config=speculator_config, t2d=t2d, d2t=d2t)

    # Load pretrained weights if provided
    if pretrained_state_dict is not None:
        load_model_weights(
            draft_model, pretrained_state_dict, args.pretrained_model_path
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
        train_call_kwargs={
            "use_off_policy_tokens": args.use_off_policy_tokens,
            "ttt_steps": args.ttt_steps,
            "ttt_step_loss_decay": args.ttt_step_loss_decay,
        },
        val_call_kwargs={
            "use_off_policy_tokens": False,
            "ttt_steps": args.ttt_steps,
            "ttt_step_loss_decay": args.ttt_step_loss_decay,
        },
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
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-resume-from-checkpoint", action="store_true")
    parser.add_argument(
        "--logger",
        type=str,
        default="",
        help=(
            "One of 'trackio', 'wandb', 'tensorboard' or comma separated list of them"
        ),
    )
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument("--data-format-version", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--d2t-path", type=str, default=None)
    parser.add_argument("--t2d-path", type=str, default=None)
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help=(
            "Path to pretrained EAGLE3 model directory "
            "(HuggingFace format with safetensors). "
            "When provided, d2t/t2d mappings and model weights "
            "will be loaded from this model, enabling "
            "warm-start/fine-tuning. Overrides --d2t-path "
            "and --t2d-path."
        ),
    )

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
        help=("Use off-policy tokens during training (required for regenerated data)"),
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
# torchrun --nnodes=1 --nproc_per_node=<num_gpus>  scripts/train.py
# for FSDP training
# OR
# python scripts/train.py
# for single GPU training
