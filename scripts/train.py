import argparse
import gc
import logging
import random
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import transformers
from packaging import version
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from hs_connectors import HiddenStatesBackend
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.models.eagle3.rotary_partial import install_partial_neox_rotary
from speculators.models.mtp.data import shift_batch_mtp
from speculators.models.utils import (
    get_verifier_config,
    resolve_draft_intermediate_size,
)
from speculators.train.config import TrainConfig
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.distributed import (
    get_rank,
    is_distributed,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import resolve_mask_token_id
from speculators.train.vocab_mapping import (
    build_vocab_mappings_from_distribution,
    get_target_vocab_size,
)
from speculators.utils.loading import is_config_only_dir

logger = logging.getLogger(__name__)

DRAFT_ARCH_CONFIGS: dict[str, type] = {
    "llama": LlamaConfig,
    "qwen3": Qwen3Config,
}
MROPE_INVERSE_TOLERANCE = 1e-6


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


def _maybe_apply_mrope_full_head_hack(
    rope_params: dict,
    resolved_head_dim: int,
    enabled: bool,
) -> None:
    """Optionally rescale partial MRoPE settings to full-head semantics."""
    if "mrope_section" not in rope_params:
        return

    inherited_partial = float(rope_params.get("partial_rotary_factor", 1.0))
    if enabled and inherited_partial < 1.0:
        old_section = list(rope_params["mrope_section"])
        inv = 1.0 / inherited_partial
        if abs(inv - round(inv)) > MROPE_INVERSE_TOLERANCE:
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
        rope_params["mrope_section"] = new_section
        rope_params["partial_rotary_factor"] = 1.0
        logger.warning(
            "MRoPE full-head hack applied: partial_rotary_factor "
            f"{inherited_partial} -> 1.0, mrope_section {old_section} -> "
            f"{new_section}."
        )
    elif not enabled and inherited_partial < 1.0:
        logger.warning(
            "mrope_full_head_hack=False with partial_rotary_factor="
            f"{inherited_partial} < 1.0 can cause HF trainer / vLLM "
            "partial-rotation mismatch."
        )


def create_transformer_layer_config(  # noqa: C901
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str,
    hidden_act: str | None,
    sliding_window: int,
    full_attention_indices: list[int],
    mrope_full_head_hack: bool = True,
) -> PretrainedConfig:
    if draft_arch not in DRAFT_ARCH_CONFIGS:
        raise ValueError(
            f"Unknown draft architecture: {draft_arch}. "
            f"Available: {list(DRAFT_ARCH_CONFIGS.keys())}"
        )

    if draft_arch not in ("llama", "qwen3"):
        warnings.warn(
            f"Draft architecture '{draft_arch}' is not yet supported in vLLM. "
            "The trained model may not be usable for inference in vLLM. "
            "Consider using 'llama' or 'qwen3' for full vLLM compatibility.",
            stacklevel=2,
        )

    config_class = DRAFT_ARCH_CONFIGS[draft_arch]
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)

    # For multimodal models (Qwen3VL, etc.), extract text_config
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

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

    head_dim = getattr(verifier_config, "head_dim", None)
    num_attention_heads = verifier_config.num_attention_heads
    num_key_value_heads = verifier_config.num_key_value_heads

    if (
        head_dim
        and verifier_config.hidden_size % num_attention_heads != 0
        and verifier_config.hidden_size % head_dim == 0
    ):
        num_attention_heads = verifier_config.hidden_size // head_dim
        if num_attention_heads % num_key_value_heads != 0:
            num_key_value_heads = num_attention_heads
    resolved_head_dim = head_dim or verifier_config.hidden_size // num_attention_heads

    if full_attention_indices and (
        min(full_attention_indices) < 0 or max(full_attention_indices) >= num_layers
    ):
        raise ValueError(
            "Full attention indices must be valid draft layer ids "
            "in range [0, num_layers)."
        )
    layer_types = [
        "full_attention" if i in full_attention_indices else "sliding_attention"
        for i in range(num_layers)
    ]

    config = config_class(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=resolve_draft_intermediate_size(verifier_config),
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_act=hidden_act,
        max_position_embeddings=verifier_config.max_position_embeddings,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=head_dim,
        tie_word_embeddings=False,
        sliding_window=sliding_window,
        use_sliding_window="sliding_attention" in layer_types,
        layer_types=layer_types,
    )

    # New rope parameters definition introduced in transformers 5.0
    if version.parse(transformers.__version__) >= version.parse("5.0.0"):
        if hasattr(verifier_config, "rope_parameters"):
            rope_params = deepcopy(verifier_config.rope_parameters)
            # Some verifiers (e.g. Laguna) use the nested per-layer-type rope format
            # {"full_attention": {...}, "sliding_attention": {...}} with no top-level
            # "rope_theta". The llama-style draft uses a single rope, so collapse it to
            # a flat default rope (preferring the sliding-attention theta).
            if isinstance(rope_params, dict) and "rope_theta" not in rope_params:
                sub = (
                    rope_params.get("sliding_attention")
                    or rope_params.get("full_attention")
                    or {}
                )
                if isinstance(sub, dict):
                    rope_params = dict(sub)
                    rope_params.setdefault("rope_type", "default")
                    rope_params.setdefault("rope_theta", 10000.0)
                else:
                    rope_params = {"rope_type": "default", "rope_theta": 10000.0}

            if isinstance(rope_params, dict):
                _maybe_apply_mrope_full_head_hack(
                    rope_params, resolved_head_dim, mrope_full_head_hack
                )
                # ``type`` is a legacy alias (only "mrope" on VL models) that
                # transformers strips during validation and that breaks vLLM's
                # config checks; drop it while keeping the real MRoPE fields.
                rope_params.pop("type", None)
                rope_params.pop("mrope_interleaved", None)
                # The verifier (e.g. Mistral) may use partial rotary embeddings,
                # but the draft model doesn't support partial_rotary_factor.
                # Only keep it for MRoPE configs that need it.
                if "mrope_section" not in rope_params:
                    rope_params.pop("partial_rotary_factor", None)
            config.rope_parameters = rope_params
    else:
        if hasattr(verifier_config, "rope_scaling"):
            rope_scaling = deepcopy(verifier_config.rope_scaling)
            if isinstance(rope_scaling, dict):
                _maybe_apply_mrope_full_head_hack(
                    rope_scaling, resolved_head_dim, mrope_full_head_hack
                )
                # Strip legacy fields for consistency with rope_parameters path
                rope_scaling.pop("type", None)
                rope_scaling.pop("mrope_interleaved", None)
                # Same partial_rotary_factor guard as the rope_parameters path.
                if "mrope_section" not in rope_scaling:
                    rope_scaling.pop("partial_rotary_factor", None)
            config.rope_scaling = rope_scaling
        config.rope_theta = getattr(verifier_config, "rope_theta", 10000.0)

    return config


def load_draft_transformer_layer_config(
    draft_config: str,
    verifier_name_or_path: str,
) -> PretrainedConfig:
    """Load the draft decoder ``transformer_layer_config`` from a config source.

    ``draft_config`` may be a HF hub id, a local directory containing a
    ``config.json``, or a path to a config JSON file. It is expected to hold a
    plain decoder config (``LlamaConfig`` for eagle3/peagle, ``Qwen3Config`` for
    dflash). If a full speculator config is given instead, its nested
    ``transformer_layer_config`` is extracted as a convenience.

    The decoder is reconciled against the verifier: ``hidden_size`` must match
    (draft/verifier hidden-size mismatch is not yet supported) and ``vocab_size``
    is aligned to the verifier's target vocabulary. The pruned draft vocabulary
    is controlled separately via ``--draft-vocab-size``.
    """
    config_dict, _ = PretrainedConfig.get_config_dict(draft_config)
    if "transformer_layer_config" in config_dict:
        # A full speculator config was passed; use only the decoder definition.
        config_dict = config_dict["transformer_layer_config"]

    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError(
            "--draft-config must define a 'model_type' (e.g. 'llama' for "
            "eagle3/peagle, 'qwen3' for dflash); none was found in the config "
            f"loaded from '{draft_config}'."
        )
    config_class: type[PretrainedConfig] = type(AutoConfig.for_model(model_type))
    draft_config_obj = config_class.from_dict(config_dict)

    verifier_config = get_verifier_config(verifier_name_or_path)
    if draft_config_obj.hidden_size != verifier_config.hidden_size:
        raise ValueError(
            f"--draft-config hidden_size ({draft_config_obj.hidden_size}) must match "
            f"the verifier hidden_size ({verifier_config.hidden_size}). Draft/verifier "
            "hidden-size mismatch is not yet supported."
        )
    if draft_config_obj.vocab_size != verifier_config.vocab_size:
        logger.warning(
            "Overriding --draft-config vocab_size (%s) with the verifier vocab_size "
            "(%s). Use --draft-vocab-size to control the pruned draft vocabulary.",
            draft_config_obj.vocab_size,
            verifier_config.vocab_size,
        )
        draft_config_obj.vocab_size = verifier_config.vocab_size
    return draft_config_obj


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


def _build_from_config_only(
    model_class: type[SpeculatorModel],
    path: str,
    t2d: torch.Tensor | None,
    d2t: torch.Tensor | None,
    verifier_name_or_path: str | None = None,
    draft_attn_impl: str | None = None,
) -> SpeculatorModel:
    """Initialize a fresh draft from a saved speculator *config* (no weights).

    Mirrors the tail of ``from_training_args``: build the model from the full
    speculator config, load vocab mappings, and pull verifier weights -- but with
    no trained draft weights to restore (decoder weights are randomly initialized).
    """
    config = model_class.config_class.from_pretrained(path)
    if draft_attn_impl is not None:
        config.transformer_layer_config._attn_implementation = draft_attn_impl
    speculators_config = getattr(config, "speculators_config", None)
    # Fall back to the CLI --verifier-name-or-path only when the saved config has
    # no verifier path -- either null or blanked to "". A real path in the config
    # takes precedence and the CLI value is ignored.
    if (
        verifier_name_or_path
        and speculators_config is not None
        and not getattr(speculators_config.verifier, "name_or_path", None)
    ):
        speculators_config.verifier.name_or_path = verifier_name_or_path
    model = model_class(config=config)
    if hasattr(model, "load_vocab_mappings"):
        model.load_vocab_mappings(t2d, d2t)  # type: ignore[attr-defined, operator]
    if hasattr(model, "load_verifier_weights"):
        model.load_verifier_weights()  # type: ignore[attr-defined, operator]
    return model


def build_draft_model(
    args: argparse.Namespace,
    model_class: type[SpeculatorModel],
    t2d: torch.Tensor | None,
    d2t: torch.Tensor | None,
    draft_vocab_size: int | None,
) -> SpeculatorModel:
    """Resolve the draft model from one of these sources:

    * ``--from-pretrained``: finetune existing weights, or -- when the path is a
      config-only directory -- initialize fresh weights from a full saved
      speculator config.
    * ``--draft-config``: take the decoder ``transformer_layer_config`` from a
      config file; build the rest of the speculator from the other CLI args.
    * neither: synthesize the decoder from the verifier config + CLI flags.

    MTP is special-cased: when not loading ``--from-pretrained``, it reuses the
    verifier's own decoder config as the draft ``transformer_layer_config`` and
    extracts the native MTP head weights from the verifier, so the decoder-shaping
    flags and ``--draft-config`` do not apply.
    """
    if args.from_pretrained:
        if is_config_only_dir(args.from_pretrained):
            logger.info(
                "--from-pretrained points to a config-only directory ('%s'); "
                "initializing fresh draft weights from the saved speculator config.",
                args.from_pretrained,
            )
            return _build_from_config_only(
                model_class,
                args.from_pretrained,
                t2d=t2d,
                d2t=d2t,
                verifier_name_or_path=args.verifier_name_or_path,
                draft_attn_impl=(
                    args.draft_attn_impl if args.speculator_type != "mtp" else None
                ),
            )
        if args.speculator_type != "mtp":
            # _attn_implementation is never serialized by HF configs, so re-apply
            # the CLI selection before construction -- mirroring from_training_args.
            # MTP is skipped: its from_training_args never sets the field and its
            # __init__ resolves its own default ("eager") when it is absent.
            config = model_class.config_class.from_pretrained(args.from_pretrained)
            config.transformer_layer_config._attn_implementation = args.draft_attn_impl
            return model_class.from_pretrained(
                args.from_pretrained,
                config=config,
                t2d=t2d,
                d2t=d2t,
                verifier=args.verifier_name_or_path,
            )
        return model_class.from_pretrained(
            args.from_pretrained,
            t2d=t2d,
            d2t=d2t,
            verifier=args.verifier_name_or_path,
        )

    if args.speculator_type == "mtp":
        # MTP uses the verifier's own decoder config as the draft
        # transformer_layer_config and extracts the native MTP head weights from
        # the verifier; the decoder-shaping flags and --draft-config do not apply,
        # and there is no draft mask token to resolve.
        transformer_layer_config = get_verifier_config(args.verifier_name_or_path)
    else:
        if args.draft_config:
            transformer_layer_config = load_draft_transformer_layer_config(
                args.draft_config, args.verifier_name_or_path
            )
        else:
            full_attention_indices = args.full_attention_indices
            if not full_attention_indices:
                logger.info(
                    "All %d draft layers using sliding window attention "
                    "(window=%d). To use full attention on specific layers, "
                    "pass '--full-attention-indices <layer_ids>'.",
                    args.num_layers,
                    args.sliding_window,
                )

            transformer_layer_config = create_transformer_layer_config(
                verifier_name_or_path=args.verifier_name_or_path,
                num_layers=args.num_layers,
                draft_arch=args.draft_arch,
                hidden_act=args.draft_hidden_act,
                sliding_window=args.sliding_window,
                full_attention_indices=full_attention_indices,
                mrope_full_head_hack=args.draft_mrope_full_head_hack,
            )

        args.mask_token_id = resolve_mask_token_id(
            args.verifier_name_or_path,
            transformer_layer_config.vocab_size,
            args.mask_token_id,
            trust_remote_code=args.trust_remote_code,
        )

    args.draft_vocab_size = draft_vocab_size
    return model_class.from_training_args(
        verifier_config=transformer_layer_config,
        t2d=t2d,
        d2t=d2t,
        **vars(args),
    )


def main(cfg: TrainConfig):  # noqa: C901
    # Phase-1 adapter: the model layer still consumes a flat vars(args)-shaped
    # dict via **kwargs, so flatten the typed config back into a namespace here.
    # New code should read cfg.<group>.<field> directly and must NOT add new
    # args.* accesses below this line.
    args = argparse.Namespace(**cfg.flatten())

    # Set random seed for reproducibility
    set_seed(args.seed, args.deterministic_cuda)

    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    maybe_setup_distributed()

    if args.fsdp_shard and not is_distributed():
        raise ValueError(
            "--fsdp-shard requires launching with torchrun/distributed training; "
            "otherwise parameters are not sharded."
        )

    # Install partial-neox rotary patch if not using full-head hack
    if not args.draft_mrope_full_head_hack:
        install_partial_neox_rotary()
        logger.info(
            "Installed partial-neox rotary patch for HF/vLLM RoPE alignment "
            "(draft_mrope_full_head_hack=False)"
        )
    # Write the reproducibility artifacts (run.yaml + train_command.txt) next to
    # the checkpoints at rank 0 only, so every checkpoint carries the resolved
    # config that produced it.
    if get_rank() == 0:
        cfg.save(args.save_path)

    hidden_states_dtype = getattr(torch, args.hidden_states_dtype)

    if args.speculator_type == "mtp":
        if args.draft_attn_impl != "simple_flex_attention":
            raise ValueError(
                "--draft-attn-impl is not configurable for MTP. "
                "Must be left with the default value ('simple_flex_attention')."
            )
        # MTP reuses the verifier's own decoder as the draft and extracts the
        # native MTP head weights from the verifier, so there are no vocab
        # mappings or draft mask token to resolve from the CLI. This works both
        # with --from-pretrained (a previously converted checkpoint) and without
        # it (weights are extracted from the verifier on the fly). The decoder
        # transformer_layer_config is resolved later in build_draft_model.
        d2t, t2d, draft_vocab_size = None, None, None
        args.mask_token_id = None
    else:
        d2t, t2d, draft_vocab_size = parse_vocab_mappings(args)

        if args.full_attention_indices and args.speculator_type == "mtp":
            raise ValueError(
                "--full-attention-indices is not supported for mtp draft models."
            )

    registry = SpeculatorModel.registry
    if registry is None or args.speculator_type not in registry:
        available = list(registry.keys()) if registry else []
        raise ValueError(
            f"Unknown speculator type: {args.speculator_type}. Available: {available}"
        )

    model_class = registry[args.speculator_type]

    draft_model = build_draft_model(args, model_class, t2d, d2t, draft_vocab_size)

    # Get target layer IDs from the model (resolved at model level)
    num_target_layers = len(draft_model.target_layer_ids)  # type: ignore[arg-type]

    if args.speculator_type == "mtp":
        args.num_speculative_steps = draft_model.config.num_speculative_steps

    # Dry-run: persist an initialized checkpoint and exit before training so the
    # config/weights can be validated (e.g. in vLLM). The saved checkpoint can be
    # fed straight back via --from-pretrained to start training.
    if args.dry_run:
        # Save in hidden_states_dtype (bf16) for compact checkpoints.
        draft_model.to(hidden_states_dtype)
        if get_rank() == 0:
            logger.info(
                "[dry-run] Saving initialized checkpoint (%s) to '%s'",
                hidden_states_dtype,
                args.save_path,
            )
            draft_model.save_pretrained(args.save_path)
            logger.info(
                "[dry-run] Done. Validate this checkpoint, then train with "
                "'--from-pretrained %s'.",
                args.save_path,
            )
        maybe_destroy_distributed()
        return

    hidden_size = draft_model.config.transformer_layer_config.hidden_size

    # Setup dataloaders
    preprocess_fns = {
        "eagle3": shift_batch,
        "peagle": shift_batch,
        "mtp": shift_batch_mtp,
    }
    preprocess = preprocess_fns.get(args.speculator_type)

    backend_registry = HiddenStatesBackend.registry
    backend_cls = backend_registry[args.hidden_states_backend]
    # from_train_args is the live runtime consumer of the backend's (mirrored)
    # train-args, read off the flattened namespace. hs_connectors stays
    # argparse-based and standalone so vLLM can use it without speculators; that
    # is why the backend's train-args are mirrored into the pydantic schema rather
    # than the plugin depending on pydantic. test_backend_reconciliation.py keeps
    # the mirror complete so nothing read here was dropped during resolution.
    transfer = backend_cls.from_train_args(args, args.data_path)

    train_loader, val_loader = create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=hidden_states_dtype,
        noise_std=args.noise_std,
        legacy_data=args.legacy_data,
        transfer=transfer,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        verifier_name_or_path=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=preprocess,
        train_data_ratio=args.train_data_ratio,
    )

    # Get trainer kwargs from model class
    train_call_kwargs, val_call_kwargs = model_class.get_trainer_kwargs(**vars(args))

    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        save_path=args.save_path,
        lr=args.lr,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        scheduler_type=args.scheduler_type,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_warmup_ratio=args.scheduler_warmup_ratio,
        scheduler_total_steps=args.scheduler_total_steps,
        scheduler_num_cosine_cycles=args.scheduler_num_cosine_cycles,
        checkpoint_freq=args.checkpoint_freq,
        save_best=args.save_best,
        hidden_states_dtype=hidden_states_dtype,
        log_freq=args.log_freq,
        fsdp_shard=args.fsdp_shard,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    del trainer, draft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    maybe_destroy_distributed()


if __name__ == "__main__":
    main(TrainConfig.resolve())


# RUN WITH:
# torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py
# for multi-GPU training (DDP by default)
# OR
# torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py --fsdp-shard
# for FSDP sharded training (when model doesn't fit in a single GPU)
# OR
# python scripts/train.py
# for single GPU training
