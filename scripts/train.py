import argparse
import logging
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
)
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    SampleFileDataset,
    create_collate_fn,
    split_files,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import (
    maybe_destroy_distributed,
    maybe_setup_distributed,
    resolve_mask_token_id,
)
from speculators.train.vocab_mapping import (
    build_vocab_mappings_from_distribution,
    get_target_vocab_size,
)

logger = logging.getLogger(__name__)

DRAFT_ARCH_CONFIGS: dict[str, type] = {
    "llama": LlamaConfig,
    "qwen3": Qwen3Config,
}


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


def setup_dataloader(
    dataset: BaseDataset,
    world_size: int,
    local_rank: int,
    hidden_size: int,
    num_workers: int = 12,
    prefetch_factor: int = 4,
    preprocess=None,
) -> DataLoader:
    """Setup dataloader for training.
    Args:
        file_list: List of file paths to load data from.
        world_size: Number of processes in the distributed training.
        local_rank: Rank of the current process.
        add_noise: Whether to add noise to the data.
        noise_std: Standard deviation for noise augmentation.
        num_workers: Number of dataloader workers.
        prefetch_factor: Dataloader prefetch factor.
        preprocess: Optional per-sample preprocessing function applied
            before collation (e.g. shift_batch for Eagle3).
    Returns:
        DataLoader: Dataloader for training.
    """
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=args.total_seq_len,
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
        collate_fn=create_collate_fn(args.total_seq_len, hidden_size, preprocess),
        persistent_workers=True,
    )


def create_transformer_layer_config(
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str = "llama",
    hidden_act: str | None = None,
) -> PretrainedConfig:
    if draft_arch not in DRAFT_ARCH_CONFIGS:
        raise ValueError(
            f"Unknown draft architecture: {draft_arch}. "
            f"Available: {list(DRAFT_ARCH_CONFIGS.keys())}"
        )

    if draft_arch != "llama":
        warnings.warn(
            f"Draft architecture '{draft_arch}' is not yet supported in vLLM. "
            "The trained model may not be usable for inference in vLLM. "
            "Consider using 'llama' (the default) for full vLLM compatibility.",
            stacklevel=2,
        )

    config_class = DRAFT_ARCH_CONFIGS[draft_arch]
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)

    # For multimodal models (Qwen3VL, etc.), extract text_config
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    hidden_act = hidden_act or getattr(verifier_config, "hidden_act", None) or getattr(
        verifier_config, "hidden_activation", None
    )
    if hidden_act is None:
        raise AttributeError(
            f"{type(verifier_config).__name__} has neither 'hidden_act' nor 'hidden_activation'"
        )

    return config_class(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=verifier_config.intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=verifier_config.num_attention_heads,
        num_key_value_heads=verifier_config.num_key_value_heads,
        hidden_act=hidden_act,
        max_position_embeddings=verifier_config.max_position_embeddings,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=getattr(verifier_config, "head_dim", None),
        tie_word_embeddings=False,
    )


def _load_mappings(d2t_path, t2d_path, expected_draft_vocab_size: int | None):
    logger.info(f"Loading vocab mappings from '{d2t_path}' and '{t2d_path}'")
    # Load d2t and t2d tensors if provided
    d2t = torch.from_numpy(np.load(d2t_path))
    t2d = torch.from_numpy(np.load(t2d_path))
    draft_vocab_size = d2t.shape[0]
    if expected_draft_vocab_size and expected_draft_vocab_size != draft_vocab_size:
        raise ValueError(
            f"Explicit vocab mapping (t2d & d2t) files were provided, but don't"
            f"match the provided --draft-vocab-size {draft_vocab_size}."
            f"d2t.shape={d2t.shape}, dim 0 should match provided value."
        )
    return d2t, t2d, draft_vocab_size


def parse_vocab_mappings(args: argparse.Namespace):
    if args.d2t_path or args.t2d_path:
        if not (args.d2t_path and args.t2d_path):
            raise ValueError(
                "Both t2d and d2t must be provided together, or both must be omitted. "
                f"Got t2d={'provided' if args.t2d_path is not None else 'not provided'}"
                f"d2t={'provided' if args.d2t_path is not None else 'not provided'}"
            )

        return _load_mappings(args.d2t_path, args.t2d_path, args.draft_vocab_size)

    data_path = Path(args.data_path)
    default_t2d_path = data_path / "t2d.npy"
    default_d2t_path = data_path / "d2t.npy"

    if default_t2d_path.exists() and default_d2t_path.exists():
        return _load_mappings(default_d2t_path, default_t2d_path, args.draft_vocab_size)

    token_freq_path = args.token_freq_path or data_path / "token_freq.pt"
    token_freq_path = Path(token_freq_path)
    if token_freq_path.exists() and args.draft_vocab_size is not None:
        logger.info("No vocab mappings provided. Regenerating from token frequencies")
        token_freq_dict = torch.load(token_freq_path, weights_only=True)

        target_vocab_size = get_target_vocab_size(None, args.verifier_name_or_path)

        d2t, t2d = build_vocab_mappings_from_distribution(
            token_freq_dict=token_freq_dict,
            draft_vocab_size=args.draft_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        draft_vocab_size = d2t.shape[0]
        if args.draft_vocab_size and args.draft_vocab_size != draft_vocab_size:
            raise ValueError(
                f"Explicit vocab mapping (t2d & d2t) files were provided, but don't"
                f"match the provided --draft-vocab-size {draft_vocab_size}."
                f"d2t.shape={d2t.shape}, dim 0 should match provided value."
            )

        logger.info(f"Caching vocab mapping files to '{data_path}'")
        np.save(data_path / "d2t.npy", d2t.cpu().numpy())
        np.save(data_path / "t2d.npy", t2d.cpu().numpy())

        return d2t, t2d, draft_vocab_size

    logger.warning(
        "No vocab mappings found, and can't generate new ones because either "
        f"token_freq_path='{token_freq_path}' doesn't exist or --draft-vocab-size is "
        "None. Using full verifier vocab"
    )
    # When vocab mapping is not provided, use the full verifier vocab
    verifier_config = AutoConfig.from_pretrained(args.verifier_name_or_path)
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return None, None, verifier_config.vocab_size


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
    if not hasattr(torch, args.hidden_states_dtype):
        raise ValueError(
            "--hidden-states-dtype must be a dtype attribute of torch. e.g. `bfloat16`"
        )
    hidden_states_dtype = getattr(torch, args.hidden_states_dtype)

    d2t, t2d, draft_vocab_size = parse_vocab_mappings(args)

    # Setup speculator config
    transformer_layer_config = create_transformer_layer_config(
        args.verifier_name_or_path,
        args.num_layers,
        draft_arch=args.draft_arch,
        hidden_act=args.draft_hidden_act,
    )

    args.mask_token_id = resolve_mask_token_id(
        args.verifier_name_or_path,
        transformer_layer_config.vocab_size,
        args.mask_token_id,
    )

    registry = SpeculatorModel.registry
    if registry is None or args.speculator_type not in registry:
        available = list(registry.keys()) if registry else []
        raise ValueError(
            f"Unknown speculator type: {args.speculator_type}. Available: {available}"
        )

    model_class = registry[args.speculator_type]
    if args.from_pretrained:
        draft_model = model_class.from_pretrained(
            args.from_pretrained, t2d=t2d, d2t=d2t
        )
    else:
        args_dict = vars(args)
        args_dict["draft_vocab_size"] = draft_vocab_size
        draft_model = model_class.from_training_args(
            verifier_config=transformer_layer_config,
            t2d=t2d,
            d2t=d2t,
            **args_dict,
        )

    # Setup dataloaders
    preprocess = shift_batch if args.speculator_type == "eagle3" else None

    noise_transform = AddUniformNoise(std=args.noise_std)
    if args.legacy_data:
        warnings.warn(
            "Using '--legacy-data' is deprecated and will be removed soon.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        train_files, val_files = split_files(args.data_path, ratio=0.9)
        train_dataset: BaseDataset = SampleFileDataset(
            file_list=train_files,
            max_len=args.total_seq_len,
            transform=noise_transform,
            hidden_states_dtype=hidden_states_dtype,
        )
        val_dataset: BaseDataset = SampleFileDataset(
            file_list=val_files,
            max_len=args.total_seq_len,
            hidden_states_dtype=hidden_states_dtype,
        )
    else:
        train_dataset = ArrowDataset(
            datapath=args.data_path,
            max_len=args.total_seq_len,
            hidden_states_path=args.hidden_states_path,
            vllm_endpoint=args.vllm_endpoint,
            on_missing=args.on_missing,
            on_generate=args.on_generate,
            transform=noise_transform,
            split_ratio=0.9,
            model=args.verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        val_dataset = ArrowDataset(
            datapath=args.data_path,
            max_len=args.total_seq_len,
            hidden_states_path=args.hidden_states_path,
            vllm_endpoint=args.vllm_endpoint,
            on_missing=args.on_missing,
            on_generate=args.on_generate,
            split_ratio=-0.1,
            model=args.verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
        )

    train_loader = setup_dataloader(
        train_dataset,
        world_size,
        local_rank,
        transformer_layer_config.hidden_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=preprocess,
    )
    val_loader = setup_dataloader(
        val_dataset,
        world_size,
        local_rank,
        transformer_layer_config.hidden_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=preprocess,
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
        checkpoint_freq=args.checkpoint_freq,
        save_best=args.save_best,
        hidden_states_dtype=hidden_states_dtype,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    maybe_destroy_distributed()


def _checkpoint_freq(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("--checkpoint-freq must be >= 1")
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument(
        "--speculator-type",
        type=str,
        default="eagle3",
        help="Type of speculator model to train (e.g., eagle3)",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default="",
        help="The pretrained draft model to finetune",
    )
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument(
        "--hidden-states-path",
        type=str,
        default=None,
        help=(
            "The path where cached hidden states files are stored. (Default: "
            "args.data_path / 'hidden_states')"
        ),
    )
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help=(
            "vLLM endpoint address to use if generating hidden states on-demand."
            " Only required if `--on-missing=generate` and samples are missing."
            " Note: the vLLM instance must be configured to cache hidden states"
            " to a location that is accessible from the training instance. i.e."
            " on the same node, or a shared network drive. (Default: 'http://localhost:8000/v1')"
        ),
    )
    parser.add_argument(
        "--on-missing",
        choices=["generate", "skip", "warn", "raise"],
        default="generate",
        help=(
            "Dataloader behaviour when there are no cached hidden states for a sample."
            "Default: 'generate', which attempts to generate the hidden states on-"
            "demand using the provided vLLM endpoint. The other options skip the sample"
            ", skip and warn, or raise an error respectively."
        ),
    )
    parser.add_argument(
        "--on-generate",
        choices=["cache", "delete"],
        default="delete",
        help=(
            "Dataloader behaviour when a new hidden state has been generated"
            " (only applies if args.on_missing=='generate'). Default: 'delete', "
            "deletes hidden states once they are loaded. 'cache' will instead store"
            "the hidden states in the args.hidden_states_path. This can be used to "
            "enable hybrid online/offline training, with hidden states generated on the"
            "first epoch, and reused on subsequent epochs."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Timeout in seconds for each individual vLLM request "
            f"(default: {DEFAULT_REQUEST_TIMEOUT}). "
            "Only applies if --on-missing=generate."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum number of retry attempts per vLLM request on failure "
            f"(default: {DEFAULT_MAX_RETRIES}). "
            "Only applies if --on-missing=generate."
        ),
    )
    parser.add_argument(
        "--legacy-data",
        action="store_true",
        help=(
            "DEPRECATED. Use the old data format which stores hidden states alongside "
            "token_ids and assistant_masks, in data_i.pt files. This option will be "
            "removed soon."
        ),
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
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--draft-arch",
        type=str,
        default="llama",
        choices=list(DRAFT_ARCH_CONFIGS.keys()),
        help="Architecture for draft decoder layers. Defaults to 'llama'. "
        "Note: only 'llama' is currently supported in vLLM for inference.",
    )
    parser.add_argument(
        "--draft-hidden-act",
        type=str,
        default=None,
        help="Activation function for draft decoder layers. Defaults to the verifier's activation. "
        "Useful for dflash which uses Qwen3 layers that expects 'silu' for vLLM deployment.",
    )
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) A (space separated) list of integer layer ids. Defaults to"
            "[2, num_hidden_layers // 2, num_hidden_layers - 3, num_hidden_layers]. "
            "Note: must be set explicitly if custom values were used to launch vllm"
        ),
    )
    parser.add_argument(
        "--token-freq-path",
        type=str,
        default=None,
        help="Path to token frequency distribution file (.pt)",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help="Vocabulary size for the draft model",
    )
    parser.add_argument("--d2t-path", type=str, default=None)
    parser.add_argument("--t2d-path", type=str, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--hidden-states-dtype",
        type=str,
        default="bfloat16",
        help="The dtype to initialize model weights and dataloader hidden states to",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle normalization before residual connections (default: True)",
    )
    parser.add_argument(
        "--embed-requires-grad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train embedding layer weights (default: False)",
    )
    parser.add_argument(
        "--norm-before-fc",
        action="store_true",
        help="Use RMSNorm before fc in Eagle3 draft path "
        "(e.g. for gpt-oss). Omit for other models.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Block size for DFlash model (default: 8)",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=256,
        help="Maximum anchor positions for DFlash training (default: 256)",
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
    # Checkpoint Parameters
    parser.add_argument(
        "--checkpoint-freq",
        type=_checkpoint_freq,
        default=1,
        help="Save a checkpoint every N epochs.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        default=False,
        help="Pointing to checkpoint with lowest validation loss.",
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
# torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py
# for FSDP training
# OR
# python scripts/train.py
# for single GPU training
