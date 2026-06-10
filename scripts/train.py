import argparse
import json
import logging
import random
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import transformers
from packaging import version
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
from speculators.models.metrics import resolve_loss_fn
from speculators.models.mtp.data import shift_batch_mtp
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


def unwrap_verifier_text_config(verifier_config: PretrainedConfig) -> PretrainedConfig:
    """Unwrap multimodal verifier configs to their text backbone config."""
    if hasattr(verifier_config, "thinker_config"):
        verifier_config = verifier_config.thinker_config
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config


def get_default_draft_intermediate_size(verifier_config: PretrainedConfig) -> int:
    """Infer a dense drafter FFN width from dense or MoE verifier configs."""
    moe_ffn = getattr(verifier_config, "moe_intermediate_size", None)
    if moe_ffn is not None:
        experts_per_tok = int(getattr(verifier_config, "num_experts_per_tok", 1))
        active = int(moe_ffn) * max(experts_per_tok, 1)
        shared_ffn = getattr(verifier_config, "shared_expert_intermediate_size", None)
        if shared_ffn is not None:
            active += int(shared_ffn)
        return active

    dense_ffn = getattr(verifier_config, "intermediate_size", None)
    if dense_ffn is not None:
        return int(dense_ffn)

    raise AttributeError(
        f"Cannot infer draft intermediate_size from {type(verifier_config).__name__}; "
        "pass --draft-intermediate-size explicitly."
    )


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
        collate_fn=create_collate_fn(
            args.total_seq_len, hidden_size, dataset.hidden_states_dtype, preprocess
        ),
        persistent_workers=True,
    )


def create_transformer_layer_config(  # noqa: C901
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str,
    hidden_act: str | None,
    # --- DFlash Hybrid Attention sliding window support ---
    sliding_window: int,
    sliding_window_indices: list[int],
    # --- Qwen3.6 Hybrid Attention support ---
    draft_intermediate_size: int | None = None,
    draft_num_attention_heads: int | None = None,
    draft_num_key_value_heads: int | None = None,
    draft_head_dim: int | None = None,
    draft_rope_scaling: dict | None = None,
    draft_rope_theta: float | None = None,
    draft_max_position_embeddings: int | None = None,
    mrope_full_head_hack: bool = True,
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
    verifier_config = unwrap_verifier_text_config(
        AutoConfig.from_pretrained(verifier_name_or_path, trust_remote_code=True)
    )

    hidden_act = (
        hidden_act
        or getattr(verifier_config, "hidden_act", None)
        or getattr(verifier_config, "hidden_activation", None)
    )
    if hidden_act is None:
        raise AttributeError(
            f"{type(verifier_config).__name__} has neither 'hidden_act' "
            "nor 'hidden_activation'"
        )

    if draft_intermediate_size is None:
        draft_intermediate_size = get_default_draft_intermediate_size(verifier_config)

    n_heads = draft_num_attention_heads or verifier_config.num_attention_heads
    n_kv = draft_num_key_value_heads or verifier_config.num_key_value_heads
    hd = draft_head_dim or getattr(verifier_config, "head_dim", None)
    hidden_size = verifier_config.hidden_size
    resolved_head_dim = hd or hidden_size // n_heads
    if n_heads % n_kv != 0:
        raise ValueError(
            f"Invalid GQA ratio: num_attention_heads({n_heads}) must be divisible "
            f"by num_key_value_heads({n_kv})."
        )
    if resolved_head_dim <= 0:
        raise ValueError(f"Invalid head_dim({resolved_head_dim}); must be positive.")

    rope_kwargs: dict = {}
    verifier_rope_theta = getattr(verifier_config, "rope_theta", None)
    if verifier_rope_theta is None:
        for src_name in ("rope_parameters", "rope_scaling"):
            src = getattr(verifier_config, src_name, None)
            if isinstance(src, dict) and src.get("rope_theta") is not None:
                verifier_rope_theta = src["rope_theta"]
                logger.info(
                    "Drafter rope_theta recovered from verifier "
                    f"{src_name}.rope_theta = {verifier_rope_theta}."
                )
                break
    if verifier_rope_theta is not None:
        rope_kwargs["rope_theta"] = verifier_rope_theta

    verifier_rope_scaling = getattr(verifier_config, "rope_scaling", None)
    verifier_rope_parameters = getattr(verifier_config, "rope_parameters", None)
    if verifier_rope_scaling is not None:
        rope_kwargs["rope_scaling"] = dict(verifier_rope_scaling)
    elif isinstance(verifier_rope_parameters, dict):
        rope_kwargs["rope_scaling"] = dict(verifier_rope_parameters)

    if draft_rope_scaling is not None:
        cli_rope_scaling = dict(draft_rope_scaling)
        cli_rope_theta = cli_rope_scaling.pop("rope_theta", None)
        if cli_rope_theta is not None:
            rope_kwargs["rope_theta"] = cli_rope_theta
        rope_kwargs["rope_scaling"] = cli_rope_scaling
        logger.info(f"Drafter rope_scaling overridden via CLI: {cli_rope_scaling}")

    if draft_rope_theta is not None:
        rope_kwargs["rope_theta"] = float(draft_rope_theta)
        logger.info(
            "Drafter rope_theta overridden via --draft-rope-theta: "
            f"{rope_kwargs['rope_theta']}"
        )

    rope_scaling_dict = rope_kwargs.get("rope_scaling")
    if isinstance(rope_scaling_dict, dict) and "mrope_section" in rope_scaling_dict:
        if draft_rope_theta is not None:
            rope_scaling_dict["rope_theta"] = float(draft_rope_theta)
        elif "rope_theta" in rope_kwargs and "rope_theta" not in rope_scaling_dict:
            rope_scaling_dict["rope_theta"] = rope_kwargs["rope_theta"]

        inherited_partial = float(rope_scaling_dict.get("partial_rotary_factor", 1.0))
        if mrope_full_head_hack and inherited_partial < 1.0:
            old_section = list(rope_scaling_dict["mrope_section"])
            inv = 1.0 / inherited_partial
            if abs(inv - round(inv)) > 1e-6:
                raise ValueError(
                    "mrope_full_head_hack cannot rescale mrope_section because "
                    f"1/partial_rotary_factor={inv} is not an integer."
                )
            scale = int(round(inv))
            new_section = [int(x) * scale for x in old_section]
            if 2 * sum(new_section) != resolved_head_dim:
                raise ValueError(
                    "mrope_full_head_hack rescaling produced inconsistent "
                    f"mrope_section {new_section}: 2*sum={2 * sum(new_section)} "
                    f"but head_dim={resolved_head_dim}."
                )
            rope_scaling_dict["mrope_section"] = new_section
            rope_scaling_dict["partial_rotary_factor"] = 1.0
            logger.warning(
                "MRoPE full-head hack applied: partial_rotary_factor "
                f"{inherited_partial} -> 1.0, mrope_section {old_section} -> "
                f"{new_section}."
            )
        elif not mrope_full_head_hack and inherited_partial < 1.0:
            logger.warning(
                "mrope_full_head_hack=False with partial_rotary_factor="
                f"{inherited_partial} < 1.0 can cause HF trainer / vLLM "
                "partial-rotation mismatch."
            )

    max_pos = (
        int(draft_max_position_embeddings)
        if draft_max_position_embeddings is not None
        else verifier_config.max_position_embeddings
    )

    # --- DFlash sliding window layer_types ---
    if sliding_window_indices and (
        min(sliding_window_indices) < 0 or max(sliding_window_indices) >= num_layers
    ):
        raise ValueError(
            "Sliding window indices must be valid draft layer ids "
            "in range [0, num_layers)."
        )
    layer_types = [
        "sliding_attention" if i in (sliding_window_indices or []) else "full_attention"
        for i in range(num_layers)
    ]

    config = config_class(
        vocab_size=verifier_config.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=draft_intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        hidden_act=hidden_act,
        max_position_embeddings=max_pos,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=resolved_head_dim,
        tie_word_embeddings=False,
        sliding_window=sliding_window,
        layer_types=layer_types,
    )

    # RoPE: use transformers >= 5.0 rope_parameters path
    if hasattr(verifier_config, "rope_parameters"):
        config.rope_parameters = deepcopy(verifier_config.rope_parameters)
        # Apply CLI overrides into rope_parameters
        if rope_kwargs.get("rope_scaling"):
            config.rope_parameters = deepcopy(rope_kwargs["rope_scaling"])
        if rope_kwargs.get("rope_theta") is not None:
            if isinstance(config.rope_parameters, dict):
                config.rope_parameters["rope_theta"] = rope_kwargs["rope_theta"]

    return config


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
    verifier_config = unwrap_verifier_text_config(
        AutoConfig.from_pretrained(args.verifier_name_or_path, trust_remote_code=True)
    )
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

    if args.speculator_type == "mtp":
        verifier_config = AutoConfig.from_pretrained(args.verifier_name_or_path)
        if hasattr(verifier_config, "text_config"):
            verifier_config = verifier_config.text_config
        d2t, t2d, draft_vocab_size = None, None, verifier_config.vocab_size
        transformer_layer_config = verifier_config
        args.mask_token_id = None
    else:
        d2t, t2d, draft_vocab_size = parse_vocab_mappings(args)

        if args.sliding_window_indices and args.speculator_type != "dflash":
            raise ValueError(
                "Currently sliding window attention is only supported by dflash "
                "draft models. Please open an issue/pr if you would like to use "
                "sliding window attention with a different speculator type"
            )
        # Setup speculator config
        transformer_layer_config = create_transformer_layer_config(
            verifier_name_or_path=args.verifier_name_or_path,
            num_layers=args.num_layers,
            draft_arch=args.draft_arch,
            hidden_act=args.draft_hidden_act,
            draft_intermediate_size=args.draft_intermediate_size,
            draft_num_attention_heads=args.draft_num_attention_heads,
            draft_num_key_value_heads=args.draft_num_key_value_heads,
            draft_head_dim=args.draft_head_dim,
            draft_rope_scaling=args.draft_rope_scaling,
            draft_rope_theta=args.draft_rope_theta,
            draft_max_position_embeddings=args.draft_max_position_embeddings,
            mrope_full_head_hack=args.draft_mrope_full_head_hack,
            sliding_window=args.sliding_window,
            sliding_window_indices=args.sliding_window_indices,
        )

        args.mask_token_id = resolve_mask_token_id(
            args.verifier_name_or_path,
            transformer_layer_config.vocab_size,
            args.mask_token_id,
            trust_remote_code=args.trust_remote_code,
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
        args.draft_vocab_size = draft_vocab_size
        draft_model = model_class.from_training_args(
            verifier_config=transformer_layer_config,
            t2d=t2d,
            d2t=d2t,
            **vars(args),
        )

    if args.speculator_type == "mtp":
        args.num_speculative_steps = draft_model.config.num_speculative_steps

    # Setup dataloaders
    preprocess_fns = {
        "eagle3": shift_batch,
        "peagle": shift_batch,
        "mtp": shift_batch_mtp,
    }
    preprocess = preprocess_fns.get(args.speculator_type)

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
            verifier_name_or_path=args.verifier_name_or_path,
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
            verifier_name_or_path=args.verifier_name_or_path,
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
        log_freq=args.log_freq,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    maybe_destroy_distributed()


def _checkpoint_freq(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("--checkpoint-freq must be > 0")
    if fvalue > 1 and not fvalue.is_integer():
        raise argparse.ArgumentTypeError(
            f"--checkpoint-freq={fvalue} is not an integer. Values > 1 are treated "
            "as epoch counts and must be whole numbers."
        )
    return fvalue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing code from HF Hub when loading the verifier's tokenizer.",
    )
    parser.add_argument(
        "--speculator-type",
        type=str,
        default="eagle3",
        help="Type of speculator model to train (eagle3, dflash, peagle, mtp)",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default="",
        help="The pretrained draft model to finetune",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./output",
        help=(
            "Root data directory containing the preprocessed dataset, "
            "vocab mappings (d2t.npy, t2d.npy), token frequencies "
            "(token_freq.pt), and hidden states (default: ./output)"
        ),
    )
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
    parser.add_argument("--save-path", type=str, default="./output/checkpoints")
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
    parser.add_argument(
        "--log-freq",
        type=int,
        default=1,
        help="Log training metrics every N steps (default: 1)",
    )
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
        default="silu",
        help="Activation function for draft decoder layers. Defaults to 'silu' for "
        "sigmoid linear unit. Qwen3 layers of dflash expect 'silu' activation for "
        "vLLM deployment. If another function is desired, set as a string or leave "
        "as None to automatically fall back to the verifier's activation function.",
    )
    parser.add_argument(
        "--draft-intermediate-size",
        type=int,
        default=None,
        help="Override draft FFN intermediate size; useful for MoE verifiers.",
    )
    parser.add_argument(
        "--draft-num-attention-heads",
        type=int,
        default=None,
        help="Override the drafter's num_attention_heads.",
    )
    parser.add_argument(
        "--draft-num-key-value-heads",
        type=int,
        default=None,
        help="Override the drafter's num_key_value_heads.",
    )
    parser.add_argument(
        "--draft-head-dim",
        type=int,
        default=None,
        help="Override the drafter's per-head hidden dimension.",
    )
    parser.add_argument(
        "--draft-rope-scaling",
        type=lambda s: json.loads(s) if s else None,
        default=None,
        help="JSON RoPE scaling dict to apply to the drafter during training.",
    )
    parser.add_argument(
        "--draft-rope-theta",
        type=float,
        default=None,
        help="Override the drafter's RoPE frequency base.",
    )
    parser.add_argument(
        "--draft-max-position-embeddings",
        type=int,
        default=None,
        help="Override the drafter's max_position_embeddings.",
    )
    parser.add_argument(
        "--draft-mrope-full-head-hack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For MRoPE configs with partial_rotary_factor < 1, rescale "
            "mrope_section and set partial_rotary_factor=1.0 so HF training "
            "and vLLM inference use equivalent full-head rotary semantics."
        ),
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
        help=(
            "Path to token frequency distribution file (.pt). Used together with "
            "--draft-vocab-size to build vocab mappings at training time. Falls back "
            "to '<data-path>/token_freq.pt' if not provided. If neither that file "
            "exists nor --draft-vocab-size is set, vocab mapping is skipped and the "
            "full verifier vocab is used."
        ),
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help=(
            "Vocabulary size for the draft model. Must be provided together with a "
            "token frequency file (--token-freq-path or '<data-path>/token_freq.pt') "
            "to generate vocab mappings. If either is absent, vocab mapping is skipped "
            "and the full verifier vocab is used, making this argument a no-op."
        ),
    )
    parser.add_argument("--d2t-path", type=str, default=None)
    parser.add_argument("--t2d-path", type=str, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument(
        "--num-speculative-steps",
        type=int,
        default=3,
        help="Number of MTP prediction steps (default: 3). Only used with MTP.",
    )
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="kl_div",
        choices=["kl_div", "ce"],
        help=(
            "Loss function used during draft model training. "
            "'kl_div' = KL divergence (default). "
            "'ce' = cross-entropy."
        ),
    )
    parser.add_argument(
        "--step-weight-beta",
        type=float,
        default=0.6,
        help=(
            "Exponential decay factor for MTP step weights. "
            "Higher values weight earlier prediction steps more heavily. "
            "Only used with MTP algorithm."
        ),
    )
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
    # D-Flash specific parameters
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
    # P-EAGLE specific parameters
    parser.add_argument(
        "--num-depths",
        type=int,
        default=8,
        help="Number of parallel prediction depths for P-EAGLE (default: 8)",
    )
    parser.add_argument(
        "--down-sample-ratio",
        type=float,
        default=0.7,
        help="Geometric decay ratio for COD sampling in P-EAGLE (default: 0.7)",
    )
    parser.add_argument(
        "--down-sample-ratio-min",
        type=float,
        default=0.2,
        help="Minimum retention ratio for COD sampling in P-EAGLE (default: 0.2)",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=2048,
        help="Sliding window size for sliding window attention layers."
        "Must also set --sliding-window-indices.",
    )
    parser.add_argument(
        "--sliding-window-indices",
        type=int,
        nargs="+",
        default=[],
        help="(Optional) A (space separated) list of draft layer indices of sliding "
        " window layers. All other draft layers are assumed to be full attention "
        "layers. (e.g. 0 2 4 will make the first, third, and fifth layers use "
        "sliding window attention and the second and fourth will be full attention)."
        "Defaults to all layers using full attention.",
    )
    parser.add_argument(
        "--sliding-window-non-causal",
        action="store_true",
        default=False,
        help="Use non-causal (bidirectional) masking within draft blocks for sliding "
        "window attention layers. Full attention layers are always bidirectional, for"
        "DFlash. Note: vLLM currently doesn't support these models",
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
        default=1.0,
        help="Save a checkpoint every N epochs. Values < 1 enable sub-epoch "
        "checkpointing (e.g. 0.5 = every half epoch).",
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

    args = parser.parse_args()
    resolve_loss_fn(args.loss_fn)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


# RUN WITH:
# torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py
# for FSDP training
# OR
# python scripts/train.py
# for single GPU training
