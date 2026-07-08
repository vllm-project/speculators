import argparse
import gc
import logging
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from packaging import version
from pydantic import ValidationError
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.models.mtp.data import shift_batch_mtp
from speculators.models.utils import (
    get_verifier_config,
    resolve_draft_intermediate_size,
)
from speculators.train.config import (
    CONFIG_DESTS,
    add_config_cli_arguments,
    build_train_config,
    config_from_flat,
    decoder_shaping_flags,
    dump_config_yaml,
    flatten_config,
    required_flags,
    save_resolved_config,
)
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.distributed import (
    get_rank,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import resolve_mask_token_id, save_train_command
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

# Speculator types that default every draft layer to sliding window attention;
# --full-attention-indices opts specific layers back into full attention. All
# other speculator types use full attention on every layer.
SLIDING_WINDOW_SPECULATOR_TYPES = ("dflash", "dspark")


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


def create_transformer_layer_config(  # noqa: C901
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str,
    hidden_act: str | None,
    sliding_window: int,
    full_attention_indices: list[int],
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
                rope_params = {
                    "rope_type": "default",
                    "rope_theta": sub.get("rope_theta", 10000.0),
                }
            config.rope_parameters = rope_params

            _MROPE_KEYS = ("mrope_section", "mrope_interleaved", "type")  # noqa: N806
            for key in _MROPE_KEYS:
                config.rope_parameters.pop(key, None)
    else:
        if hasattr(verifier_config, "rope_scaling"):
            config.rope_scaling = deepcopy(verifier_config.rope_scaling)
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
) -> SpeculatorModel:
    """Initialize a fresh draft from a saved speculator *config* (no weights).

    Mirrors the tail of ``from_training_args``: build the model from the full
    speculator config, load vocab mappings, and pull verifier weights -- but with
    no trained draft weights to restore (decoder weights are randomly initialized).
    """
    config = model_class.config_class.from_pretrained(path)
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
        model.load_vocab_mappings(t2d, d2t)  # type: ignore[attr-defined]
    if hasattr(model, "load_verifier_weights"):
        model.load_verifier_weights()  # type: ignore[attr-defined]
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
            if args.speculator_type in SLIDING_WINDOW_SPECULATOR_TYPES:
                full_attention_indices = args.full_attention_indices
                if not full_attention_indices:
                    logger.info(
                        "All %d draft layers using sliding window attention "
                        "(window=%d). To use full attention on specific layers, "
                        "pass '--full-attention-indices <layer_ids>'.",
                        args.num_layers,
                        args.sliding_window,
                    )
            else:
                # Other speculator types (eagle3, peagle) only support full
                # attention: mark every layer full-attention.
                full_attention_indices = list(range(args.num_layers))

            transformer_layer_config = create_transformer_layer_config(
                verifier_name_or_path=args.verifier_name_or_path,
                num_layers=args.num_layers,
                draft_arch=args.draft_arch,
                hidden_act=args.draft_hidden_act,
                sliding_window=args.sliding_window,
                full_attention_indices=full_attention_indices,
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


def main(args: argparse.Namespace):
    # Set random seed for reproducibility
    set_seed(args.seed, args.deterministic_cuda)

    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    maybe_setup_distributed()

    if get_rank() == 0:
        save_train_command(args.save_path)
        save_resolved_config(vars(args), args.save_path)

    # --hidden-states-dtype is validated at the CLI layer (DataArgs field
    # validator), so it is guaranteed to be a real torch dtype attribute here.
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

        if (
            args.full_attention_indices
            and args.speculator_type not in SLIDING_WINDOW_SPECULATOR_TYPES
        ):
            raise ValueError(
                "--full-attention-indices is only meaningful for dflash and dspark "
                "draft models (which use sliding window attention by default). "
                "Please open an issue/pr if you would like to use sliding window "
                "attention with a different speculator type."
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
    num_target_layers = len(draft_model.target_layer_ids)

    if args.speculator_type == "mtp":
        args.num_speculative_steps = draft_model.config.num_speculative_steps

    # Dry-run: persist an initialized checkpoint and exit before training so the
    # config/weights can be validated (e.g. in vLLM). The saved checkpoint can be
    # fed straight back via --from-pretrained to start training.
    if args.dry_run:
        # Match Trainer.setup_model: weights are (re)initialized in
        # hidden_states_dtype, so save the dry-run checkpoint in that dtype too
        # rather than the float32 the model is built in.
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

    train_loader, val_loader = create_train_val_loaders(
        data_path=args.data_path,
        train_data_ratio=args.train_data_ratio,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=hidden_states_dtype,
        noise_std=args.noise_std,
        legacy_data=args.legacy_data,
        hidden_states_path=args.hidden_states_path,
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
    )

    # Get trainer kwargs from model class
    train_call_kwargs, val_call_kwargs = model_class.get_trainer_kwargs(**vars(args))

    # The Trainer is configured from the typed groups rather than the flat
    # namespace (see the RFC's "Option 2": one visible typed consumer of the
    # schema). We recover that typed view from the resolved working-dict;
    # validators are idempotent on already-resolved values, so this is a cheap,
    # faithful no-op that reads only optimizer/scheduler/trainer fields -- none of
    # which main() mutates.
    cfg = config_from_flat(vars(args))
    trainer_config = TrainerConfig(
        num_epochs=cfg.trainer.epochs,
        save_path=cfg.trainer.save_path,
        lr=cfg.optimizer.lr,
        resume_from_checkpoint=not cfg.trainer.no_resume_from_checkpoint,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        optimizer=cfg.optimizer.optimizer,
        weight_decay=cfg.optimizer.weight_decay,
        muon_lr=cfg.optimizer.muon_lr,
        muon_momentum=cfg.optimizer.muon_momentum,
        muon_weight_decay=cfg.optimizer.muon_weight_decay,
        muon_ns_steps=cfg.optimizer.muon_ns_steps,
        muon_adjust_lr_fn=cfg.optimizer.muon_adjust_lr_fn,
        scheduler_type=cfg.scheduler.scheduler_type,
        scheduler_warmup_steps=cfg.scheduler.scheduler_warmup_steps,
        scheduler_warmup_ratio=cfg.scheduler.scheduler_warmup_ratio,
        scheduler_total_steps=cfg.scheduler.scheduler_total_steps,
        scheduler_num_cosine_cycles=cfg.scheduler.scheduler_num_cosine_cycles,
        checkpoint_freq=cfg.trainer.checkpoint_freq,
        save_best=cfg.trainer.save_best,
        hidden_states_dtype=hidden_states_dtype,
        log_freq=cfg.trainer.log_freq,
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


# CLI flags that synthesize the draft decoder shape. They conflict with both
# --from-pretrained and --draft-config, each of which fully defines the draft.
# Derived from the schema (fields tagged _DECODER_SHAPING) so the set stays in one
# place; adding a shaping knob is still a single edit in speculators.train.config.
DECODER_SHAPING_FLAGS: dict[str, str] = decoder_shaping_flags()


def validate_draft_init_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    provided: set[str],
) -> None:
    """Enforce the draft-init contract.

    The draft model may be defined in exactly one way:

    * ``--from-pretrained`` -- load a complete speculator checkpoint (or a
      config-only directory); or
    * ``--draft-config`` -- load just the decoder config and build the rest of
      the speculator from the other CLI args; or
    * the decoder-shaping flags (``--num-layers`` etc.) -- synthesize everything.

    ``--from-pretrained`` takes precedence over all other model-definition
    options: it is mutually exclusive with ``--draft-config`` and with the
    decoder-shaping flags, since those values come from the checkpoint.
    ``--draft-config`` is likewise incompatible with the decoder-shaping flags.
    MTP from scratch (``--speculator-type mtp`` without ``--from-pretrained``)
    reuses the verifier's own decoder config, so ``--draft-config`` and the
    decoder-shaping flags do not apply and are rejected.

    ``provided`` is the set of config dests supplied on the CLI or present in the
    ``--config`` YAML (as computed by
    :func:`speculators.train.config.build_train_config`). A flag passed at its
    default value still counts, so it is treated as a conflict; only the
    :data:`DECODER_SHAPING_FLAGS` subset is checked here.
    """
    shaping = [flag for dest, flag in DECODER_SHAPING_FLAGS.items() if dest in provided]
    if args.from_pretrained:
        conflicting = shaping + (["--draft-config"] if args.draft_config else [])
        if conflicting:
            parser.error(
                "--from-pretrained loads a complete draft model and takes precedence "
                "over all other model-definition options, so these conflict with it "
                f"(remove them): {', '.join(conflicting)}"
            )
        return
    if args.speculator_type == "mtp":
        # MTP-from-scratch reuses the verifier's own decoder config and extracts the
        # native MTP head weights; --draft-config and the decoder-shaping flags do not
        # apply, so reject them rather than silently ignoring them.
        conflicting = shaping + (["--draft-config"] if args.draft_config else [])
        if conflicting:
            parser.error(
                "--speculator-type mtp reuses the verifier's decoder config, so these "
                f"options do not apply (remove them): {', '.join(conflicting)}"
            )
        return
    if args.draft_config and shaping:
        parser.error(
            "--draft-config defines the draft decoder, so these flags conflict with "
            f"it (remove them): {', '.join(shaping)}"
        )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file supplying any of the arguments below. "
        "CLI flags override values in the file, which override the defaults. "
        "Optional: with no --config the behaviour is identical to a pure-CLI run.",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        default=False,
        help="Print the fully-resolved config as YAML to stdout and exit. The "
        "output is a valid --config file; use it to scaffold a run.yaml.",
    )
    # Every tunable flag is generated from the pydantic config schema
    # (speculators.train.config): one typed field there == one CLI flag here,
    # with the right type, choices, bool style, and help text.
    add_config_cli_arguments(parser)

    args = parser.parse_args(argv)

    # Every generated flag defaults to argparse.SUPPRESS, so the namespace holds
    # only the flags the user actually passed -- that set IS the CLI provenance.
    # Only these become overrides; everything else resolves from --config YAML
    # then the coded defaults.
    cli_values = {
        dest: value for dest, value in vars(args).items() if dest in CONFIG_DESTS
    }

    # Build + validate the typed config. Field/model validators (dtype, loss,
    # checkpoint_freq, per-type defaults) run here; surface any failure as a
    # clean CLI error rather than a pydantic traceback. A broken --config file
    # (missing, unreadable, malformed YAML, or not a top-level mapping) raises
    # OSError/yaml.YAMLError/ValueError from the YAML load -- route those through
    # parser.error too, so a --config typo (the most likely mistake) exits 2 with
    # a clean message instead of a traceback.
    try:
        config, provided = build_train_config(cli_values, args.config)
    except ValidationError as exc:
        parser.error(_format_config_error(exc))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        parser.error(f"--config '{args.config}': {exc}")

    # Flatten back to the vars(args)-shaped dict the rest of the script (and the
    # model layer's **kwargs contract) consumes. The typed config is the source
    # of truth; this namespace is the mutable working copy (derived values such
    # as mask_token_id are written onto it later, as before).
    resolved = argparse.Namespace(**flatten_config(config))
    resolved.config = args.config
    resolved.dump_config = args.dump_config

    # Draft-init conflict contract: reads the CLI-or-YAML provenance set so a
    # conflict expressed entirely in the YAML file is still caught. Run it BEFORE
    # --dump-config so the scaffold we emit is guaranteed to be a loadable config,
    # never one that would fail its own draft-init contract on re-read.
    validate_draft_init_args(parser, resolved, provided)

    if args.dump_config:
        sys.stdout.write(dump_config_yaml(config))
        parser.exit()

    return resolved


def _format_config_error(exc: ValidationError) -> str:
    """Render a pydantic ValidationError as a concise CLI message."""
    lines = []
    for err in exc.errors():
        loc = err["loc"]
        label = ".".join(str(p) for p in loc) or "<config>"
        # Drop pydantic's "Value error, " prefix for a cleaner CLI message.
        msg = err["msg"].removeprefix("Value error, ")
        # For a genuinely-missing field, point at the flag to set rather than an
        # opaque group path (e.g. "verifier: Field required" -> name
        # --verifier-name-or-path).
        flags = required_flags(loc) if err.get("type") == "missing" else []
        hint = f" (set {' or '.join(flags)})" if flags else ""
        lines.append(f"{label}: {msg}{hint}")
    return "invalid configuration:\n  " + "\n  ".join(lines)


if __name__ == "__main__":
    args = parse_args()
    main(args)


# RUN WITH:
# torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py
# for FSDP training
# OR
# python scripts/train.py
# for single GPU training
