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

from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
)
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.models.metrics import resolve_loss_config
from speculators.models.mtp.data import shift_batch_mtp
from speculators.models.utils import (
    get_verifier_config,
    resolve_draft_intermediate_size,
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
from speculators.utils.argparse_utils import explicitly_provided_dests
from speculators.utils.loading import is_config_only_dir

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


def create_transformer_layer_config(  # noqa: C901
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str,
    hidden_act: str | None,
    sliding_window: int,
    sliding_window_indices: list[int],
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

    if sliding_window_indices and (
        min(sliding_window_indices) < 0 or max(sliding_window_indices) >= num_layers
    ):
        raise ValueError(
            "Sliding window indices must be validate draft layer ids "
            "in range [0, num_layers)."
        )
    layer_types = [
        "sliding_attention" if i in sliding_window_indices else "full_attention"
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
            transformer_layer_config = create_transformer_layer_config(
                verifier_name_or_path=args.verifier_name_or_path,
                num_layers=args.num_layers,
                draft_arch=args.draft_arch,
                hidden_act=args.draft_hidden_act,
                sliding_window=args.sliding_window,
                sliding_window_indices=args.sliding_window_indices,
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


def main(args: argparse.Namespace):  # noqa: C901
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

    if not hasattr(torch, args.hidden_states_dtype):
        raise ValueError(
            "--hidden-states-dtype must be a dtype attribute of torch. e.g. `bfloat16`"
        )
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

        if args.sliding_window_indices and args.speculator_type not in (
            "dflash",
            "dspark",
        ):
            raise ValueError(
                "Currently sliding window attention is only supported by dflash "
                "and dspark draft models. Please open an issue/pr if you would like "
                "to use sliding window attention with a different speculator type"
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
    del trainer, draft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


# CLI flags that synthesize the draft decoder shape. They conflict with both
# --from-pretrained and --draft-config, each of which fully defines the draft.
DECODER_SHAPING_FLAGS: dict[str, str] = {
    "num_layers": "--num-layers",
    "draft_arch": "--draft-arch",
    "draft_hidden_act": "--draft-hidden-act",
    "sliding_window": "--sliding-window",
    "sliding_window_indices": "--sliding-window-indices",
}


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

    ``provided`` is the set of decoder-shaping dests the user explicitly passed
    (see :func:`speculators.utils.argparse_utils.explicitly_provided_dests`); a flag
    passed at its default value still counts as a conflict.
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
        help="Type of speculator model to train (eagle3, dflash, dspark, peagle, mtp)",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default="",
        help="Path or HF id of a pretrained draft. May also point to a "
        "local directory containing only a config.json, in which case a "
        "fresh draft is initialized from that full speculator config. Takes precedence "
        "over and is mutually exclusive with --draft-config and the decoder-shaping "
        "flags (--num-layers, --draft-arch, --draft-hidden-act, --sliding-window, "
        "--sliding-window-indices).",
    )
    parser.add_argument(
        "--draft-config",
        type=str,
        default="",
        help="HF id, directory, or JSON path of a decoder config (LlamaConfig for "
        "eagle3/peagle, Qwen3Config for dflash) to use as the draft "
        "transformer_layer_config; the rest of the speculator is built from the other "
        "CLI args. Mutually exclusive with --from-pretrained and with the "
        "decoder-shaping flags (--num-layers, --draft-arch, --draft-hidden-act, "
        "--sliding-window, --sliding-window-indices).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Build the speculator, initialize weights, save a checkpoint to "
        "--save-path, then exit before training. Useful to validate the config and "
        "weights (e.g. in vLLM) before launching a full run. Can be combined with "
        "--draft-config or --from-pretrained.",
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
        help=(
            "One of 'trackio', 'wandb', 'tensorboard', 'mlflow' or "
            "comma separated list."
        ),
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
        default="qwen3",
        choices=list(DRAFT_ARCH_CONFIGS.keys()),
        help="Architecture for draft decoder layers. Defaults to 'qwen3'.",
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
        help=(
            "Loss function specification. Pass a name for a single loss "
            "(kl_div, ce, tv, nla, lk_hybrid) or a JSON dict for a weighted "
            'combination, e.g. \'{"ce": 0.1, "tv": 0.9}\'.'
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
        default=False,
        help="Use RMSNorm before FC layer in draft path "
        "(e.g., for Eagle 3.1 / gpt-oss models).",
    )
    parser.add_argument(
        "--norm-output",
        action="store_true",
        default=False,
        help="Feed post-norm hidden states back across TTT steps to stabilize "
        "magnitude drift across speculation depths (Eagle 3.1).",
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
    parser.add_argument(
        "--dflash-decay-gamma",
        type=float,
        default=4.0,
        help="Decay gamma for DFlash/DSpark loss weighting (default: 4.0)",
    )
    # DSpark-specific arguments (sequential Markov head + confidence head).
    parser.add_argument(
        "--markov-rank",
        type=int,
        default=256,
        help="DSpark: low-rank dim of the Markov logit-bias head (0 disables it).",
    )
    parser.add_argument(
        "--markov-head-type",
        type=str,
        default="vanilla",
        choices=["vanilla", "gated", "rnn"],
        help="DSpark: sequential head variant (default: vanilla).",
    )
    parser.add_argument(
        "--enable-confidence-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DSpark: attach the per-position acceptance confidence head.",
    )
    parser.add_argument(
        "--confidence-head-with-markov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DSpark: feed the Markov previous-token embedding into the "
        "confidence head alongside the backbone hidden state.",
    )
    parser.add_argument(
        "--confidence-head-alpha",
        type=float,
        default=1.0,
        help="DSpark: weight of the confidence-head BCE term (default: 1.0).",
    )
    parser.add_argument(
        "--draft-attn-impl",
        type=str,
        default="simple_flex_attention",
        choices=["simple_flex_attention", "sdpa", "eager"],
        help="Attention implementation for draft layers. "
        "Use 'sdpa' or 'eager' for hardware that doesn't support flex attention."
        "Not supported for MTP.",
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

    # optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon"],
        help=(
            "Optimizer to use. 'muon' applies Muon to 2D weight matrices and AdamW to "
            "the remaining params (norms, biases, embeddings, lm_head)."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for the AdamW optimizer (and the AdamW group in muon mode).",
    )
    parser.add_argument(
        "--muon-lr",
        type=float,
        default=0.02,
        help="LR for the Muon (2D weights) group. Only used with --optimizer muon.",
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn",
        type=str,
        default="match_rms_adamw",
        choices=["original", "match_rms_adamw"],
        help="Muon LR adjustment. 'match_rms_adamw' matches AdamW's update RMS.",
    )

    args = parser.parse_args()
    provided = explicitly_provided_dests(parser, DECODER_SHAPING_FLAGS)
    validate_draft_init_args(parser, args, provided)
    resolve_loss_config(args.loss_fn)
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
