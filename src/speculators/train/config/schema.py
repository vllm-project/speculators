"""The trainer's tunable surface -- the "what", separated from the mechanism.

This module is the **single source of truth** for every knob ``scripts/train.py``
exposes. Each tunable is one typed field on a cohesive ``pydantic`` group; the
argparse CLI, the YAML schema, ``--help`` text, ``--dump-config`` output, and
``--set`` support are all *derived* from these fields. Adding a tunable is one
typed field here (see "Adding an argument" below) -- no ``add_argument`` call.

The *mechanism* that acts on this schema lives in sibling modules, so no single
module is a god-object (see ADR 0004):

* :mod:`.cli` -- schema -> argparse generation, ``--set`` parsing, required-flag
  mapping, the draft-init conflict check.
* :mod:`.sources` -- source layering, YAML load + ``train:`` unwrap, warnings.
* :mod:`.provenance` -- the typed :class:`~.provenance.Provenance` value + walk.
* :mod:`.adapter` -- the phase-1 flat <-> nested seam (:meth:`TrainConfig.flatten`).
* :mod:`.artifacts` -- the reproducibility writers (``run.yaml`` + sidecar).

:class:`TrainConfig` is the one public type (ADR 0004): it parses itself
(:meth:`resolve` / :meth:`from_sources`), reports its own
:attr:`~TrainConfig.provenance`, flattens to the model layer (:meth:`flatten`),
and serializes itself (:meth:`dump_yaml` / :meth:`save`). Each method delegates to
one of the sibling modules; the class is broad but every module is small.

Resolution model:

* **Precedence** -- ``flag > --set > YAML > default``. The four layers are fed to
  pydantic-settings in that priority order (see :mod:`.sources`); a higher layer
  wins per key while untouched siblings fall through to lower layers.
* **Lists replace, they do not merge.** partial-update applies only to nested
  group models, so a ``list`` leaf is overridden wholesale.
* **Every source is optional.** With no ``--config``, no ``--set``, and the same
  flags, behaviour is byte-identical to the previous pure-argparse script.

Every field name here is byte-identical to the corresponding argparse ``dest``;
the grouping is purely organisational. That invariant is what lets a single flat
CLI flag map onto exactly one nested field.

Adding an argument
------------------
Add one typed field to the relevant group (or a root scalar on ``TrainConfig``).
The CLI flag, YAML key, ``--help`` text, ``--dump-config`` output, ``--set``
support, and the flat working-dict are all derived from it. Two follow-ups are
enforced by tests, not by this module:

* update ``tests/unit/train/golden_flat_dict.json`` ``_baseline`` with the new
  dest and its default (``test_baseline_covers_every_config_dest`` guards this);
* if the field is a *decoder-shaping* knob, tag it ``_DECODER_SHAPING`` so the
  draft-init conflict check covers it.
"""

from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
)
from speculators.models.metrics import resolve_loss_config

if TYPE_CHECKING:
    from .provenance import Provenance

# Marker for a boolean field that should get an argparse ``--x/--no-x``
# (BooleanOptionalAction) flag even though it defaults to False. Bools that
# default to True or None get the optional form automatically; a plain
# False-default bool becomes a ``store_true`` flag unless tagged with this.
# The explicit ``dict[str, JsonValue]`` annotation is load-bearing: a bare
# ``{"cli_bool": "optional"}`` literal infers ``dict[str, str]``, which mypy
# rejects as an argument to ``Field(json_schema_extra=...)``. Do not remove it.
_CLI_BOOL_OPTIONAL: dict[str, JsonValue] = {"cli_bool": "optional"}

# Marker for a draft *decoder-shaping* field: these synthesize the draft decoder,
# so they conflict with --from-pretrained / --draft-config (which fully define the
# draft). Tagging them in the schema keeps the "which flags shape the decoder" set
# here -- the single source of truth -- instead of a hand-listed dict in train.py.
# See :func:`speculators.train.config.cli.decoder_shaping_flags`.
_DECODER_SHAPING: dict[str, JsonValue] = {"decoder_shaping": True}


class _Group(BaseModel):
    """Base for a config group: mutable (post-resolution) and lenient on extras."""

    model_config = ConfigDict(extra="ignore", validate_assignment=False)


class VerifierArgs(_Group):
    # Not argparse-``required`` any more: the verifier may be supplied via YAML.
    # Presence is enforced by pydantic (no default) after the sources merge.
    verifier_name_or_path: str = Field(
        description="Path or HF id of the verifier/target model."
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow executing code from HF Hub when loading the verifier's "
        "tokenizer.",
    )


class DraftArgs(_Group):
    """The draft model definition: init source, decoder shape, shared
    normalization hyperparameters, and vocabulary mapping."""

    # init source
    from_pretrained: str = Field(
        default="",
        description="Path or HF id of a pretrained draft. May also point to a local "
        "directory containing only a config.json, in which case a fresh draft is "
        "initialized from that full speculator config. Takes precedence over and is "
        "mutually exclusive with --draft-config and the decoder-shaping flags.",
    )
    draft_config: str = Field(
        default="",
        description="HF id, directory, or JSON path of a decoder config (LlamaConfig "
        "for eagle3/peagle, Qwen3Config for dflash) to use as the draft "
        "transformer_layer_config; the rest of the speculator is built from the other "
        "CLI args. Mutually exclusive with --from-pretrained and the decoder-shaping "
        "flags.",
    )
    # decoder shape -- these synthesize the draft decoder and so are tagged
    # _DECODER_SHAPING (they conflict with --from-pretrained / --draft-config).
    num_layers: int = Field(
        default=1,
        description="Number of draft decoder layers to synthesize.",
        json_schema_extra=_DECODER_SHAPING,
    )
    draft_arch: Literal["llama", "qwen3"] | None = Field(
        default=None,
        description="Architecture for draft decoder layers "
        "(default: 'llama' for eagle3, 'qwen3' otherwise).",
        json_schema_extra=_DECODER_SHAPING,
    )
    draft_hidden_act: str = Field(
        default="silu",
        description="Activation function for draft decoder layers. Defaults to 'silu'. "
        "Qwen3 layers of dflash expect 'silu' for vLLM deployment.",
        json_schema_extra=_DECODER_SHAPING,
    )
    target_layer_ids: list[int] | None = Field(
        default=None,
        description="(Optional) space-separated list of integer layer ids. Defaults to "
        "[2, n//2, n-3, n]. Must be set explicitly if custom values were used to "
        "launch vllm.",
    )
    sliding_window: int = Field(
        default=2048,
        description="Sliding window size for sliding window attention layers "
        "(default: 2048). For dflash and dspark, all layers use sliding window by "
        "default.",
        json_schema_extra=_DECODER_SHAPING,
    )
    full_attention_indices: list[int] = Field(
        default_factory=list,
        description="(Optional) space-separated draft layer indices that should use "
        "full attention instead of sliding window. For dflash and dspark, all layers "
        "use sliding window attention by default. (e.g. '--full-attention-indices 0 2' "
        "makes layers 0 and 2 use full attention; the rest use sliding window).",
        json_schema_extra=_DECODER_SHAPING,
    )
    sliding_window_non_causal: bool = Field(
        default=False,
        description="Use non-causal (bidirectional) masking within draft blocks for "
        "sliding window attention layers. Full attention layers are always "
        "bidirectional. Note: vLLM currently doesn't support these models.",
    )
    # shared normalization hyperparameters
    norm_before_residual: bool = Field(
        default=True,
        description="Toggle normalization before residual connections (default: True).",
    )
    norm_before_fc: bool | None = Field(
        default=None,
        description="Apply a single RMSNorm to the concatenated auxiliary hidden "
        "states before the FC projection (gpt-oss style). "
        "(default: True for eagle3, False otherwise).",
    )
    fc_norm: bool = Field(
        default=False,
        description="Apply per-layer RMSNorm to each auxiliary hidden state before "
        "concatenation and FC projection (Eagle 3.1 paper approach).",
    )
    norm_output: bool | None = Field(
        default=None,
        description="Feed post-norm hidden states back across TTT steps to stabilize "
        "magnitude drift (default: True for eagle3, False otherwise).",
    )
    embed_requires_grad: bool = Field(
        default=False,
        description="Whether to train embedding layer weights (default: False).",
        json_schema_extra=_CLI_BOOL_OPTIONAL,
    )
    # vocabulary mapping
    draft_vocab_size: int | None = Field(
        default=None,
        description="Vocabulary size for the draft model. Must be provided together "
        "with a token frequency file to generate vocab mappings; otherwise a no-op.",
    )
    token_freq_path: str | None = Field(
        default=None,
        description="Path to token frequency distribution file (.pt). Used with "
        "--draft-vocab-size to build vocab mappings. Falls back to "
        "'<data-path>/token_freq.pt'.",
    )
    d2t_path: str | None = Field(
        default=None,
        description="Draft-to-target vocab mapping file (.npy). Defaults to "
        "'<data-path>/d2t.npy'.",
    )
    t2d_path: str | None = Field(
        default=None,
        description="Target-to-draft vocab mapping file (.npy). Defaults to "
        "'<data-path>/t2d.npy'.",
    )
    mask_token_id: int | None = Field(
        default=None,
        description="Token id used to mask positions during training. Resolved from "
        "the verifier tokenizer when unset.",
    )


class DataArgs(_Group):
    data_path: str = Field(
        default="./output",
        description="Root data directory with the preprocessed dataset, vocab "
        "mappings (d2t.npy, t2d.npy), token frequencies, and hidden states.",
    )
    hidden_states_path: str | None = Field(
        default=None,
        description="Path where cached hidden states are stored "
        "(default: <data-path>/hidden_states).",
    )
    total_seq_len: int = Field(
        default=8192, description="Maximum training sequence length, in tokens."
    )
    train_data_ratio: float = Field(
        default=0.9,
        description="Fraction of the dataset used for training; the remainder is "
        "held out for validation.",
    )
    noise_std: float = Field(
        default=0.05, description="Standard deviation for noise augmentation."
    )
    legacy_data: bool = Field(
        default=False,
        description="DEPRECATED. Use the old data format storing hidden states "
        "alongside token_ids and assistant_masks in data_i.pt files.",
    )
    hidden_states_dtype: str = Field(
        default="bfloat16",
        description="The dtype to initialize model weights and dataloader hidden "
        "states to.",
    )
    use_off_policy_tokens: bool = Field(
        default=False,
        description="Use off-policy tokens during training (required for "
        "regenerated data).",
    )
    num_workers: int = Field(default=12, description="Number of dataloader workers.")
    prefetch_factor: int = Field(default=4, description="Dataloader prefetch factor.")
    # Shared across dflash/dspark/peagle (anchor-position subsampling), so it
    # lives in the shared data block rather than any single algo block.
    max_anchors: int = Field(
        default=3072,
        description="Maximum anchor positions for DFlash, DSpark, and P-EAGLE.",
    )

    @field_validator("hidden_states_dtype")
    @classmethod
    def _dtype_is_torch_attr(cls, v: str) -> str:
        if not hasattr(torch, v):
            raise ValueError(
                "hidden_states_dtype must be a dtype attribute of torch. "
                "e.g. `bfloat16`"
            )
        return v


class GenerationArgs(_Group):
    """On-demand hidden-state generation against a vLLM endpoint."""

    vllm_endpoint: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM endpoint used to generate hidden states on demand. Only "
        "needed if --on-missing=generate and samples are missing.",
    )
    on_missing: Literal["generate", "skip", "warn", "raise"] = Field(
        default="generate",
        description="Dataloader behaviour when a sample has no cached hidden states.",
    )
    on_generate: Literal["cache", "delete"] = Field(
        default="delete",
        description="Behaviour after generating a hidden state (only if "
        "--on-missing=generate). 'cache' enables hybrid online/offline training.",
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="Timeout in seconds for each vLLM request "
        "(only if --on-missing=generate).",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        description="Max retry attempts per vLLM request on failure "
        "(only if --on-missing=generate).",
    )


class LossArgs(_Group):
    loss_fn: str = Field(
        default="kl_div",
        description="Loss function spec. A name (kl_div, rkl, ce, tv, nla, lk_hybrid) "
        'or a JSON dict for a weighted combination, e.g. \'{"ce": 0.1, "tv": 0.9}\'.',
    )
    ttt_steps: int = Field(
        default=3,
        description="Number of test-time-training (TTT) steps the draft is unrolled "
        "over during training.",
    )
    ttt_step_loss_decay: float = Field(
        default=1.0,
        description="Multiplicative loss weight decay applied per TTT step "
        "(1.0 = weight every step equally).",
    )

    @field_validator("loss_fn")
    @classmethod
    def _loss_parseable(cls, v: str) -> str:
        # Raises if the spec is not a known loss name or valid weighted-JSON dict.
        resolve_loss_config(v)
        return v


class OptimizerArgs(_Group):
    optimizer: Literal["adamw", "muon"] = Field(
        default="muon",
        description="Optimizer. 'muon' applies Muon to 2D weight matrices and AdamW to "
        "the rest (norms, biases, embeddings, lm_head).",
    )
    lr: float = Field(default=1e-4, description="Learning rate (AdamW / base group).")
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for AdamW (and the AdamW group in muon mode).",
    )
    muon_lr: float | None = Field(
        default=None,
        description="LR for the Muon (2D weights) group. Only used with --optimizer "
        "muon. Defaults to 10*lr (and --lr defaults to 1e-4).",
    )
    muon_momentum: float = Field(
        default=0.95,
        description="Momentum for the Muon group (only with --optimizer muon).",
    )
    muon_weight_decay: float = Field(
        default=0.1, description="Weight decay for the Muon group."
    )
    muon_ns_steps: int = Field(
        default=5, description="Newton-Schulz iteration steps for Muon."
    )
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] = Field(
        default="match_rms_adamw",
        description="Muon LR adjustment. 'match_rms_adamw' matches AdamW's update RMS.",
    )


class SchedulerArgs(_Group):
    scheduler_type: Literal["linear", "cosine", "none"] = Field(
        default="linear", description="LR scheduler type."
    )
    scheduler_warmup_steps: int | None = Field(
        default=None, description="Warmup steps (default: scheduler-dependent)."
    )
    scheduler_warmup_ratio: float | None = Field(
        default=None,
        description="Warmup as a fraction of total scheduler steps, in [0, 1]. "
        "Ignored (with a warning) when --scheduler-warmup-steps is also set.",
    )
    scheduler_total_steps: int | None = Field(
        default=None, description="Total scheduler steps (default: inferred)."
    )
    scheduler_num_cosine_cycles: float = Field(
        default=0.5, description="Number of cosine cycles for the cosine scheduler."
    )


class TrainerArgs(_Group):
    epochs: int = Field(default=20, description="Number of training epochs.")
    checkpoint_freq: float = Field(
        default=1.0,
        description="Save a checkpoint every N epochs. Values < 1 enable sub-epoch "
        "checkpointing (e.g. 0.5 = every half epoch).",
    )
    save_best: bool = Field(
        default=False,
        description="Also point a 'best' checkpoint at the lowest validation loss.",
    )
    no_resume_from_checkpoint: bool = Field(
        default=False, description="Do not resume training from an existing checkpoint."
    )
    log_freq: int = Field(
        default=1, description="Log training metrics every N steps (default: 1)."
    )
    save_path: str = Field(
        default="./output/checkpoints",
        description="Directory to write checkpoints and the resolved run.yaml.",
    )

    @field_validator("checkpoint_freq")
    @classmethod
    def _validate_checkpoint_freq(cls, v: float) -> float:
        # Parity with the previous argparse ``type=_checkpoint_freq`` (which no
        # longer runs now the generated flag uses ``type=float``).
        if v <= 0:
            raise ValueError("--checkpoint-freq must be > 0")
        if v > 1 and not float(v).is_integer():
            raise ValueError(
                f"--checkpoint-freq={v} is not an integer. Values > 1 are treated "
                "as epoch counts and must be whole numbers."
            )
        return v


class LoggingArgs(_Group):
    logger: str = Field(
        default="",
        description="One of 'trackio', 'wandb', 'tensorboard', 'mlflow' or a comma "
        "separated list.",
    )
    log_dir: str = Field(
        default="./logs", description="Directory for training log output."
    )
    run_name: str | None = Field(
        default=None,
        description="Name for this run (used by metric loggers). Auto-generated "
        "when unset.",
    )


class DFlashArgs(_Group):
    """DFlash-family backbone knobs (also used by DSpark, which is-a DFlash)."""

    block_size: int = Field(
        default=8, description="Block size for DFlash model (default: 8)."
    )
    dflash_decay_gamma: float = Field(
        default=4.0, description="Decay gamma for DFlash/DSpark loss weighting."
    )
    per_position_loss_weight: Literal["fixed-exp-decay", "dpace"] = Field(
        default="fixed-exp-decay",
        description="Per-position loss weight scheme for DFlash/DSpark. 'dpace' "
        "enables D-PACE confidence-based weighting (requires --loss-fn ce).",
    )
    dpace_alpha: float = Field(
        default=0.5,
        description="Smoothing constant for the D-PACE loss (only used with "
        "--per-position-loss-weight dpace). Must be in (0, 1].",
    )
    draft_attn_impl: Literal["simple_flex_attention", "sdpa", "eager"] = Field(
        default="simple_flex_attention",
        description="Attention implementation for draft layers. Use 'sdpa' or 'eager' "
        "for hardware without flex attention. Not supported for MTP.",
    )


class DSparkArgs(_Group):
    """DSpark-exclusive heads (sequential Markov head + confidence head)."""

    markov_rank: int = Field(
        default=256,
        description="DSpark: low-rank dim of the Markov logit-bias head "
        "(0 disables it).",
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla", description="DSpark: sequential head variant."
    )
    enable_confidence_head: bool = Field(
        default=True,
        description="DSpark: attach the per-position acceptance confidence head.",
    )
    confidence_head_with_markov: bool = Field(
        default=True,
        description="DSpark: feed the Markov previous-token embedding into the "
        "confidence head alongside the backbone hidden state.",
    )
    confidence_head_alpha: float = Field(
        default=1.0, description="DSpark: weight of the confidence-head BCE term."
    )


class PEagleArgs(_Group):
    num_depths: int = Field(
        default=8,
        description="Number of parallel prediction depths for P-EAGLE (default: 8).",
    )
    down_sample_ratio: float = Field(
        default=0.7, description="Geometric decay ratio for COD sampling in P-EAGLE."
    )
    down_sample_ratio_min: float = Field(
        default=0.2, description="Minimum retention ratio for COD sampling in P-EAGLE."
    )


class MTPArgs(_Group):
    num_speculative_steps: int = Field(
        default=3, description="Number of MTP prediction steps. Only used with MTP."
    )
    step_weight_beta: float = Field(
        default=0.6,
        description="Exponential decay factor for MTP step weights; higher weights "
        "earlier steps more. Only used with MTP.",
    )


# Group attribute name -> group model. Order defines the layout of a dumped
# ``run.yaml`` and the order of the flattened working-dict.
_GROUPS: dict[str, type[_Group]] = {
    "verifier": VerifierArgs,
    "draft": DraftArgs,
    "data": DataArgs,
    "generation": GenerationArgs,
    "loss": LossArgs,
    "optimizer": OptimizerArgs,
    "scheduler": SchedulerArgs,
    "trainer": TrainerArgs,
    "logging": LoggingArgs,
    "dflash": DFlashArgs,
    "dspark": DSparkArgs,
    "peagle": PEagleArgs,
    "mtp": MTPArgs,
}


class TrainConfig(BaseSettings):
    """Top-level trainer configuration -- the config subsystem's one public type.

    Composed from cohesive typed groups and populated from four layered sources,
    highest precedence first: CLI flags, ``--set`` overrides, the ``--config``
    YAML file, then coded field defaults. Callers use the two entry points:

    * :meth:`resolve` -- the impure CLI boundary ``train.py`` calls (parses argv,
      handles ``--dump-config``, turns a config error into ``SystemExit(2)``).
    * :meth:`from_sources` -- the pure, argv-free, exception-raising core (the
      primary unit-test seam): layers the sources, validates, attaches
      :attr:`provenance`.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        nested_model_default_partial_update=True,
        validate_default=False,
    )

    # Which layer supplied each dest (flag | set | yaml | default) + the
    # contributor trail. Attached by ``from_sources`` in the same precedence walk
    # that resolves values, so precedence and provenance cannot drift. A private
    # attribute (not a field) so it never leaks into ``model_dump``/``dump_yaml``.
    _provenance: "Provenance | None" = PrivateAttr(default=None)
    # The raw argv this config was resolved from, recorded for the reproducibility
    # manifest. ``None`` when built off-argv (e.g. via ``from_flat``).
    _argv: "list[str] | None" = PrivateAttr(default=None)

    # root-level scalars
    speculator_type: str = Field(
        default="eagle3",
        description="Type of speculator model to train "
        "(eagle3, dflash, dspark, peagle, mtp).",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    deterministic_cuda: bool = Field(
        default=False,
        description="Set cuda to deterministic mode. This may impact performance.",
    )
    dry_run: bool = Field(
        default=False,
        description="Build the speculator, initialize weights, save a checkpoint to "
        "--save-path, then exit before training. Useful to validate the config/weights "
        "(e.g. in vLLM) before a full run.",
    )

    # groups
    verifier: VerifierArgs
    draft: DraftArgs = Field(default_factory=DraftArgs)
    data: DataArgs = Field(default_factory=DataArgs)
    generation: GenerationArgs = Field(default_factory=GenerationArgs)
    loss: LossArgs = Field(default_factory=LossArgs)
    optimizer: OptimizerArgs = Field(default_factory=OptimizerArgs)
    scheduler: SchedulerArgs = Field(default_factory=SchedulerArgs)
    trainer: TrainerArgs = Field(default_factory=TrainerArgs)
    logging: LoggingArgs = Field(default_factory=LoggingArgs)
    dflash: DFlashArgs = Field(default_factory=DFlashArgs)
    dspark: DSparkArgs = Field(default_factory=DSparkArgs)
    peagle: PEagleArgs = Field(default_factory=PEagleArgs)
    mtp: MTPArgs = Field(default_factory=MTPArgs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # pydantic-settings calls this hook BY KEYWORD, so all five parameter
        # names are part of the contract: renaming or dropping any (even the
        # unused env/dotenv/secret ones) raises TypeError. Keep the signature.
        #
        # The three explicit layers ride in via a single private ``_layers`` init
        # kwarg (popped here so it never becomes model data), each fed as its own
        # InitSettingsSource in precedence order: flag > --set > YAML. Field
        # defaults are pydantic's implicit lowest layer. Env/secret sources are
        # intentionally excluded -- the trainer is configured only via CLI + YAML.
        # This keeps source construction reentrant: no global model_config mutation.
        layers: dict[str, dict[str, Any]] = cast(
            "InitSettingsSource", init_settings
        ).init_kwargs.pop("_layers", {})
        return tuple(
            InitSettingsSource(settings_cls, init_kwargs=layers.get(name, {}))
            for name in ("flag", "set", "yaml")
        )

    @model_validator(mode="after")
    def _resolve_derived_defaults(self) -> "TrainConfig":
        """Fill defaults that derive from other fields.

        Mirrors the tail of the previous ``parse_args`` exactly: unset
        ``draft_arch`` -> ``llama`` for eagle3 else ``qwen3``; unset
        ``norm_before_fc`` / ``norm_output`` -> ``True`` for eagle3 else
        ``False``; unset ``muon_lr`` -> ``10 * lr``.
        """
        is_eagle3 = self.speculator_type == "eagle3"
        if self.draft.draft_arch is None:
            self.draft.draft_arch = "llama" if is_eagle3 else "qwen3"
        if self.draft.norm_before_fc is None:
            self.draft.norm_before_fc = is_eagle3
        if self.draft.norm_output is None:
            self.draft.norm_output = is_eagle3
        if self.optimizer.muon_lr is None:
            self.optimizer.muon_lr = 10 * self.optimizer.lr
        return self

    @model_validator(mode="after")
    def _validate_dpace(self) -> "TrainConfig":
        """D-PACE per-position loss weighting requires CE loss and a smoothing
        constant in ``(0, 1]``. Only enforced when it is actually selected, so it
        never fires on a default (``fixed-exp-decay``) run."""
        if self.dflash.per_position_loss_weight == "dpace":
            if self.loss.loss_fn != "ce":
                raise ValueError(
                    "--per-position-loss-weight=dpace requires --loss-fn=ce"
                )
            if not 0.0 < self.dflash.dpace_alpha <= 1.0:
                raise ValueError(
                    f"--dpace-alpha must be in (0, 1], got {self.dflash.dpace_alpha}"
                )
        return self

    # ------------------------------------------------------------------ #
    # Public interface (ADR 0004). Each method delegates to one mechanism
    # module so the class is broad but no single module is a god-object.
    # ------------------------------------------------------------------ #

    @classmethod
    def resolve(cls, argv: list[str] | None = None) -> "TrainConfig":
        """Impure CLI boundary: parse ``argv``, layer sources, validate.

        Handles ``--dump-config`` (print the resolved config as a valid config
        file, then exit) and turns any configuration error into a clean
        ``SystemExit(2)`` with a message that names the flag to fix -- never a
        traceback. This is what ``scripts/train.py`` calls.
        """
        from .cli import resolve as _resolve  # noqa: PLC0415

        return _resolve(cls, argv)

    @classmethod
    def from_sources(
        cls,
        *,
        cli: dict[str, Any] | None = None,
        overrides: list[str] | None = None,
        config_path: str | None = None,
        argv: list[str] | None = None,
    ) -> "TrainConfig":
        """Pure, argv-free, exception-raising core -- the primary test seam.

        Layers the four sources (``flag > --set > YAML > default``), runs the
        field/model validators and the draft-init conflict check, and attaches
        typed :attr:`provenance`. Raises :class:`~.errors.ConfigError` (or a
        pydantic ``ValidationError``) on bad input; never touches ``sys.argv`` and
        never calls ``sys.exit``.

        :param cli: flat ``{dest: value}`` for flags explicitly passed on the CLI.
        :param overrides: raw ``dotted.key=value`` strings from ``--set``.
        :param config_path: path to a ``--config`` YAML file, or ``None``.
        :param argv: the raw argv, recorded for the reproducibility manifest.
        """
        from .sources import build_from_sources  # noqa: PLC0415

        return build_from_sources(
            cls,
            cli=cli or {},
            overrides=overrides or [],
            config_path=config_path,
            argv=argv,
        )

    @property
    def provenance(self) -> "Provenance":
        """Which layer supplied each dest (``flag | set | yaml | default``).

        Attached by :meth:`from_sources`. A config built off-argv (e.g. via
        :meth:`from_flat`) reports every dest as ``default``.
        """
        from .provenance import Provenance  # noqa: PLC0415

        if self._provenance is None:
            return Provenance.all_default()
        return self._provenance

    def flatten(self) -> dict[str, Any]:
        """The phase-1 anti-corruption seam (ADR 0002): the ``vars(args)``-shaped
        flat dict the five ``SpeculatorModel`` classes consume via ``**kwargs``,
        with byte-identical ordering to the pre-refactor parser."""
        from .adapter import flatten  # noqa: PLC0415

        return flatten(self)

    @classmethod
    def from_flat(cls, flat: dict[str, Any]) -> "TrainConfig":
        """Inverse of :meth:`flatten`: recover the typed config from a flat
        working-dict (non-config keys dropped; validators are idempotent)."""
        from .adapter import from_flat  # noqa: PLC0415

        return from_flat(cls, flat)

    def dump_yaml(self) -> str:
        """Serialize to the stage-shaped, round-trippable YAML (nested under a
        top-level ``train:`` key); feed straight back via ``--config``."""
        from .artifacts import dump_yaml  # noqa: PLC0415

        return dump_yaml(self)

    def save(self, save_dir: str) -> None:
        """Write the reproducibility artifacts next to the checkpoints: the
        resolved ``run.yaml``, the ``train_command.txt`` manifest, and the
        audit-only ``run.provenance.yaml`` sidecar."""
        from .artifacts import save  # noqa: PLC0415

        save(self, save_dir)


# --------------------------------------------------------------------------- #
# Schema-derived metadata (single source of truth for dest <-> group mapping)
# --------------------------------------------------------------------------- #

# Root-level scalar fields (not inside any group), auto-discovered from the
# schema -- like the group fields -- so a new root scalar becomes a flag with no
# extra bookkeeping. Declaration order in TrainConfig is preserved (the scalars
# are declared before the group fields), which keeps the flat working-dict order
# byte-identical to the pre-refactor parser.
_ROOT_FIELDS: tuple[str, ...] = tuple(
    name for name in TrainConfig.model_fields if name not in _GROUPS
)


def _assert_all_groups_registered() -> None:
    """Fail loudly if a ``_Group``-typed field on TrainConfig is missing from
    ``_GROUPS``.

    ``_ROOT_FIELDS`` is derived as "declared on TrainConfig but not in _GROUPS",
    so an unregistered group field would be silently misclassified as a root
    scalar and generate a broken ``--<group>`` flag (``type=<GroupModel>``). This
    keeps the "add a field = add a flag" contract safe for new *groups* too.
    """
    for name in _ROOT_FIELDS:
        annotation = TrainConfig.model_fields[name].annotation
        if isinstance(annotation, type) and issubclass(annotation, _Group):
            raise RuntimeError(
                f"config field '{name}' is a group ({annotation.__name__}) but is "
                f"not registered in _GROUPS -- add it there."
            )


_assert_all_groups_registered()


def _build_dest_to_group() -> dict[str, str | None]:
    """Map each flat argparse dest to its owning group (``None`` for a root scalar)."""
    mapping: dict[str, str | None] = dict.fromkeys(_ROOT_FIELDS)
    for gname, gmodel in _GROUPS.items():
        for fname in gmodel.model_fields:
            if fname in mapping:
                raise RuntimeError(f"duplicate config field '{fname}' across groups")
            mapping[fname] = gname
    return mapping


# Flat argparse dest -> owning group attribute (or None for a root scalar).
_DEST_TO_GROUP: dict[str, str | None] = _build_dest_to_group()

# All flat dests owned by the config (excludes run-mode flags like --config).
CONFIG_DESTS: frozenset[str] = frozenset(_DEST_TO_GROUP)


def nest_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Turn a flat ``{dest: value}`` dict into the nested group structure.

    Lives here (next to :data:`_DEST_TO_GROUP`, the mapping it walks) rather than
    in :mod:`.adapter` because it is core layering plumbing -- :mod:`.sources` uses
    it to shape the flag/``--set`` layers -- and :mod:`.adapter` is the phase-2
    deletion target (ADR 0002). Keeping it here means deleting the adapter never
    strands the core path.
    """
    nested: dict[str, Any] = {}
    for dest, value in flat.items():
        group = _DEST_TO_GROUP.get(dest)
        if group is None:
            nested[dest] = value
        else:
            nested.setdefault(group, {})[dest] = value
    return nested
