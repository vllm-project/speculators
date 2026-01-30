import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig
from transformers.models.auto.configuration_auto import AutoConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
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

logger = logging.getLogger(__name__)

# DRAFTER MODEL HYPARAMETERS
NORM_BEFORE_RESIDUAL = True

# Dataloader
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
NOISE_STD = 0.05


def load_safetensors_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors format (single or sharded).
    
    Args:
        model_dir: Path to directory containing model.safetensors or sharded safetensors
        
    Returns:
        Dictionary mapping parameter names to tensors (on CPU)
        
    Raises:
        FileNotFoundError: If no safetensors files found
        RuntimeError: If safetensors library not available
    """
    model_path = Path(model_dir)
    
    # Check for safetensors library
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise RuntimeError(
            "safetensors library is required for loading pretrained models. "
            "Install it with: pip install safetensors"
        ) from e
    
    # Case 1: Single safetensors file
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        logger.info(f"Loading single safetensors file from {single_file}")
        state_dict = {}
        with safe_open(single_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    
    # Case 2: Sharded safetensors with index file
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        logger.info(f"Loading sharded safetensors from {model_path}")
        with open(index_file, "r") as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ValueError(
                f"model.safetensors.index.json exists but contains no weight_map: {index_file}"
            )
        
        # Collect all shard files
        shard_files = set(weight_map.values())
        
        # Load tensors from all shards
        state_dict = {}
        for shard_file in sorted(shard_files):
            shard_path = model_path / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Shard file not found: {shard_path} (referenced in index)"
                )
            
            logger.info(f"  Loading shard: {shard_file}")
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if weight_map.get(key) == shard_file:
                        state_dict[key] = f.get_tensor(key)
        
        return state_dict
    
    # Neither single file nor sharded format found
    raise FileNotFoundError(
        f"No safetensors files found in {model_dir}. "
        f"Expected either 'model.safetensors' or 'model.safetensors.index.json'"
    )


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
    )
    transformer_layer_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
    return transformer_layer_config


def main(args: argparse.Namespace):
    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    local_rank, world_size, rank, is_distributed = maybe_setup_distributed()
    device = torch.device(local_rank)

    # Initialize variables for pretrained model loading
    pretrained_state_dict = None
    
    # Load pretrained model if pretrained_model_path is provided
    if args.pretrained_model_path:
        logger.info(f"Loading pretrained model from {args.pretrained_model_path}")
        
        # Check for conflicting arguments
        if args.d2t_path or args.t2d_path:
            raise ValueError(
                "--pretrained-model-path overrides --d2t-path and --t2d-path. "
                "Please remove --d2t-path and --t2d-path when using --pretrained-model-path."
            )
        
        # Load the full state dict from safetensors
        pretrained_state_dict = load_safetensors_state_dict(args.pretrained_model_path)
        
        # Debug: print keys containing d2t or t2d if requested
        if args.debug_init_keys:
            logger.info("Keys containing 'd2t' or 't2d' in pretrained state dict:")
            for key in sorted(pretrained_state_dict.keys()):
                if "d2t" in key.lower() or "t2d" in key.lower():
                    tensor = pretrained_state_dict[key]
                    logger.info(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Extract d2t and t2d from state dict
        # Strategy: find keys containing 'd2t' and 't2d'
        d2t_candidates = [k for k in pretrained_state_dict.keys() if "d2t" in k.lower()]
        t2d_candidates = [k for k in pretrained_state_dict.keys() if "t2d" in k.lower()]
        
        # Select the best match (prefer exact matches like "d2t", "t2d")
        def select_key(candidates: list[str], target: str) -> str:
            if not candidates:
                raise ValueError(
                    f"No '{target}' key found in pretrained model state dict. "
                    f"Available keys: {list(pretrained_state_dict.keys())[:20]}..."
                )
            # Prefer exact match
            exact_matches = [k for k in candidates if k.lower() == target.lower()]
            if exact_matches:
                return exact_matches[0]
            # Prefer simple matches like "model.d2t" over complex ones
            if len(candidates) == 1:
                return candidates[0]
            # Multiple candidates without exact match - ask user to clarify
            raise ValueError(
                f"Multiple '{target}' candidates found in state dict: {candidates}. "
                f"Please verify the model structure."
            )
        
        d2t_key = select_key(d2t_candidates, "d2t")
        t2d_key = select_key(t2d_candidates, "t2d")
        
        logger.info(f"Extracting d2t from key: {d2t_key}")
        logger.info(f"Extracting t2d from key: {t2d_key}")
        
        # Extract and remove from state dict (will load later after model creation)
        d2t = pretrained_state_dict.pop(d2t_key).to(device)
        t2d = pretrained_state_dict.pop(t2d_key).to(device)
        
        # Validate shapes
        if d2t.dim() not in [1, 2]:
            raise ValueError(
                f"Unexpected d2t shape: {d2t.shape}. Expected 1D or 2D tensor."
            )
        if t2d.dim() not in [1, 2]:
            raise ValueError(
                f"Unexpected t2d shape: {t2d.shape}. Expected 1D or 2D tensor."
            )
        
        # Derive draft_vocab_size from d2t
        draft_vocab_size = d2t.shape[0]
        logger.info(f"Derived draft_vocab_size={draft_vocab_size} from d2t.shape[0]")
        logger.info(f"d2t shape: {d2t.shape}, t2d shape: {t2d.shape}")
        
    # Load t2d and d2t tensors from numpy files if no pretrained model
    elif args.d2t_path or args.t2d_path:
        if not (args.d2t_path and args.t2d_path):
            raise ValueError(
                "Both t2d and d2t must be provided together, or both must be omitted. "
                f"Got t2d={'provided' if args.t2d_path is not None else 'not provided'}"
                f"d2t={'provided' if args.d2t_path is not None else 'not provided'}"
            )
        d2t = torch.from_numpy(np.load(args.d2t_path)).to(device)
        t2d = torch.from_numpy(np.load(args.t2d_path)).to(device)
        draft_vocab_size = d2t.shape[0]
    else:
        d2t = None
        t2d = None
        # When vocab mapping is not provided, use the full verifier vocab
        verifier_config = AutoConfig.from_pretrained(args.verifier_name_or_path)
        if hasattr(verifier_config, "text_config"):
            verifier_config = verifier_config.text_config
        draft_vocab_size = verifier_config.vocab_size

    # Setup speculator config
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
        logger.info(f"Loading pretrained weights from {args.pretrained_model_path}")
        logger.info(f"Number of parameters to load: {len(pretrained_state_dict)}")
        
        # Load state dict with strict=True to ensure all keys match
        missing_keys, unexpected_keys = draft_model.load_state_dict(
            pretrained_state_dict, strict=False
        )
        
        if missing_keys:
            logger.warning(f"Missing keys in pretrained model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in pretrained model: {unexpected_keys}")
        
        # If we want strict loading (recommended), check for issues
        if missing_keys or unexpected_keys:
            logger.warning(
                "Weight loading completed with warnings. "
                "This may be expected if the model architecture changed slightly."
            )
        else:
            logger.info("Successfully loaded all pretrained weights (strict=True)")
        
        logger.info(f"Pretrained model loaded. Fine-tuning will start from these weights.")
        logger.info("Note: Optimizer and scheduler states are NOT loaded (fresh training state).")

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
        help="One of 'trackio', 'wandb', 'tensorboard' or comma separated list of them",
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
        help="Path to pretrained EAGLE3 model directory (HuggingFace format with safetensors). "
        "When provided, d2t/t2d mappings and model weights will be loaded from this model, "
        "enabling warm-start/fine-tuning. Overrides --d2t-path and --t2d-path.",
    )
    parser.add_argument(
        "--debug-init-keys",
        action="store_true",
        default=False,
        help="Print all state dict keys containing 'd2t' or 't2d' when loading pretrained model",
    )
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument(
        "--use-off-policy-tokens",
        action="store_true",
        default=False,
        help="Use off-policy tokens during training (required for regenerated data)",
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
