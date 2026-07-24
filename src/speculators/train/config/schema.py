"""The trainer's tunable fields: the single source of truth for every flag.

Each tunable is one typed ``pydantic`` field; adding one is a one-field edit.
:class:`TrainConfig` is the sole public type (``resolution`` builds the CLI).
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, cast

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

# A bool that must render as an argparse ``--x/--no-x`` (``BooleanOptionalAction``)
# even though it defaults to False. Bools defaulting to True or None get that form
# automatically; a plain False-default bool renders as ``store_true`` unless tagged
# here. The explicit ``dict[str, JsonValue]`` annotation is load-bearing: a bare
# literal infers ``dict[str, str]``, which pydantic's ``json_schema_extra`` rejects.
_CLI_BOOL_OPTIONAL: dict[str, JsonValue] = {"cli_bool": "optional"}

# A field the CLI must require (argparse ``required=True``). The schema gives it a
# placeholder default so a config is still constructible from defaults; the
# resolution layer turns the marker back into a required flag / missing-value error.
_CLI_REQUIRED: dict[str, JsonValue] = {"cli_required": True}

# A str field whose valid values are a dynamic registry (argparse ``choices=``)
# resolved at CLI-generation time. The marker names the registry indirectly -- a
# string key the resolution layer maps to the live registry -- so the schema keeps no
# runtime dependency on the backends and stays purely declarative.
_CLI_CHOICES: dict[str, JsonValue] = {"cli_choices": "hidden_states_backends"}


class _Group(BaseModel):
    """Base for a config group: mutable (post-resolution) and lenient on extras."""

    model_config = ConfigDict(extra="ignore", validate_assignment=False)


class VerifierArgs(_Group):
    verifier_name_or_path: str = Field(
        default="",
        description="Path or HF id of the verifier/target model.",
        json_schema_extra=_CLI_REQUIRED,
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow executing code from HF Hub when loading the verifier's "
        "tokenizer.",
    )


class DraftArgs(_Group):
    """The draft model definition: init source, decoder shape, shared
    normalization hyperparameters, and vocabulary mapping."""

    from_pretrained: str = Field(
        default="",
        description="Path or HF id of a pretrained draft. May also point to a local "
        "directory containing only a config.json, in which case a fresh draft is "
        "initialized from that full speculator config. Takes precedence over and is "
        "mutually exclusive with --draft-config and the decoder-shaping flags "
        "(--num-layers, --draft-arch, --draft-hidden-act, --sliding-window, "
        "--full-attention-indices).",
    )
    draft_config: str = Field(
        default="",
        description="HF id, directory, or JSON path of a decoder config (LlamaConfig "
        "for eagle3/peagle, Qwen3Config for dflash) to use as the draft "
        "transformer_layer_config; the rest of the speculator is built from the other "
        "CLI args. Mutually exclusive with --from-pretrained and the decoder-shaping "
        "flags (--num-layers, --draft-arch, --draft-hidden-act, --sliding-window, "
        "--full-attention-indices).",
    )
    num_layers: int = Field(
        default=1, description="Number of draft decoder layers to synthesize."
    )
    draft_arch: Literal["llama", "qwen3"] | None = Field(
        default=None,
        description="Architecture for draft decoder layers "
        "(default: 'llama' for eagle3, 'qwen3' otherwise).",
    )
    draft_hidden_act: str = Field(
        default="silu",
        description="Activation function for draft decoder layers. Defaults to 'silu'. "
        "Qwen3 layers of dflash expect 'silu' for vLLM deployment. Leave as None to "
        "fall back to the verifier's activation function.",
    )
    draft_mrope_full_head_hack: bool = Field(
        default=True,
        description="For MRoPE configs with partial_rotary_factor < 1, rescale "
        "mrope_section and set partial_rotary_factor=1.0 so HF training and vLLM "
        "inference use equivalent full-head rotary semantics.",
    )
    target_layer_ids: list[int] | None = Field(
        default=None,
        description="(Optional) space-separated list of integer layer ids. Defaults to "
        "[2, n//2, n-3, n]. Must be set explicitly if custom values were used to "
        "launch vllm.",
    )
    token_freq_path: str | None = Field(
        default=None,
        description="Path to token frequency distribution file (.pt). Used with "
        "--draft-vocab-size to build vocab mappings at training time. Falls back to "
        "'<data-path>/token_freq.pt'. If neither exists nor --draft-vocab-size is set, "
        "vocab mapping is skipped and the full verifier vocab is used.",
    )
    draft_vocab_size: int | None = Field(
        default=None,
        description="Vocabulary size for the draft model. Must be provided together "
        "with a token frequency file to generate vocab mappings; otherwise a no-op.",
    )
    d2t_path: str | None = Field(
        default=None, description="Draft-to-target vocab mapping file (.npy)."
    )
    t2d_path: str | None = Field(
        default=None, description="Target-to-draft vocab mapping file (.npy)."
    )
    mask_token_id: int | None = Field(
        default=None,
        description="Token id used to mask positions during training. Resolved from "
        "the verifier tokenizer when unset.",
    )
    norm_before_residual: bool = Field(
        default=True,
        description="Toggle normalization before residual connections (default: True).",
    )
    embed_requires_grad: bool = Field(
        default=False,
        description="Whether to train embedding layer weights (default: False).",
        json_schema_extra=_CLI_BOOL_OPTIONAL,
    )
    norm_before_fc: bool | None = Field(
        default=None,
        description="Apply a single RMSNorm to the concatenated auxiliary hidden "
        "states before the FC projection (gpt-oss style). See --fc-norm for the "
        "per-layer alternative from the Eagle 3.1 paper. "
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
        "magnitude drift across speculation depths "
        "(default: True for eagle3, False otherwise).",
    )
    sliding_window: int = Field(
        default=2048,
        description="Sliding window size for sliding window attention layers "
        "(default: 2048). All draft layers use sliding window by default (except mtp).",
    )
    full_attention_indices: list[int] = Field(
        default_factory=list,
        description="(Optional) space-separated draft layer indices that should use "
        "full attention instead of sliding window. All draft layers use sliding window "
        "by default (except mtp). (e.g. '--full-attention-indices 0 2' makes layers 0 "
        "and 2 use full attention; the rest use sliding window).",
    )
    sliding_window_non_causal: bool = Field(
        default=False,
        description="Use non-causal (bidirectional) masking within draft blocks for "
        "sliding window attention layers. Full attention layers are always "
        "bidirectional. Note: vLLM currently doesn't support these models.",
    )
    draft_attn_impl: Literal["simple_flex_attention", "sdpa", "eager"] = Field(
        default="simple_flex_attention",
        description="Attention implementation for draft layers. Use 'sdpa' or 'eager' "
        "for hardware that doesn't support flex attention. Not supported for MTP.",
    )


class DataArgs(_Group):
    data_path: str = Field(
        default="./output",
        description="Root data directory with the preprocessed dataset, vocab "
        "mappings (d2t.npy, t2d.npy), token frequencies (token_freq.pt), and hidden "
        "states.",
    )
    hidden_states_backend: str = Field(
        default="file",
        description="Hidden states transfer backend. Each backend may add its own CLI "
        "arguments. Default: 'file'.",
        json_schema_extra=_CLI_CHOICES,
    )
    hidden_states_path: str | None = Field(
        default=None,
        description="Path where cached hidden states are stored "
        "(default: <data-path>/hidden_states). Contributed by the 'file' backend.",
    )
    total_seq_len: int = Field(
        default=8192, description="Maximum training sequence length, in tokens."
    )
    train_data_ratio: float = Field(
        default=0.9,
        description="Fraction of the dataset used for training; the remainder is held "
        "out for validation.",
    )
    noise_std: float = Field(
        default=0.05, description="Standard deviation for noise augmentation."
    )
    legacy_data: bool = Field(
        default=False,
        description="DEPRECATED. Use the old data format which stores hidden states "
        "alongside token_ids and assistant_masks in data_i.pt files. Will be removed "
        "soon.",
    )
    hidden_states_dtype: str = Field(
        default="bfloat16",
        description="Data type for dataloader hidden states and autocast compute. "
        "Model master weights are always kept in fp32. Options: float32, bfloat16 "
        "(recommended). float16 is not supported (requires gradient scaling).",
    )
    num_workers: int = Field(default=12, description="Number of dataloader workers.")
    prefetch_factor: int = Field(default=4, description="Dataloader prefetch factor.")
    max_anchors: int = Field(
        default=3072,
        description="Maximum anchor positions for DFlash, DSpark, and P-EAGLE training "
        "(default: 3072).",
    )

    @field_validator("hidden_states_dtype")
    @classmethod
    def _dtype_is_torch_attr(cls, v: str) -> str:
        # Local import keeps this schema module purely declarative and torch-free
        # at import time; torch is only needed to validate the dtype string.
        import torch  # noqa: PLC0415

        # hasattr alone is too weak: torch.nn / torch.cuda / torch.Tensor all
        # exist but are not dtypes and would only fail later as an opaque
        # autocast error. Require the resolved attribute to be a torch.dtype.
        resolved = getattr(torch, v, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(
                "hidden_states_dtype must name a torch dtype, e.g. `bfloat16`."
            )
        if resolved is torch.float16:
            raise ValueError(
                "hidden_states_dtype float16 is not supported (requires gradient "
                "scaling); use bfloat16."
            )
        return v


class GenerationArgs(_Group):
    """On-demand hidden-state generation against a vLLM endpoint."""

    vllm_endpoint: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM endpoint used to generate hidden states on demand. Only "
        "needed if --on-missing=generate and samples are missing. The vLLM instance "
        "must cache hidden states to a location reachable from the training instance.",
    )
    on_missing: Literal["generate", "skip", "warn", "raise"] = Field(
        default="generate",
        description="Dataloader behaviour when a sample has no cached hidden states. "
        "Default 'generate' generates them on demand via the vLLM endpoint; the others "
        "skip the sample, skip with a warning, or raise.",
    )
    on_generate: Literal["cache", "delete"] = Field(
        default="delete",
        description="Behaviour after generating a hidden state (only if "
        "--on-missing=generate). 'delete' discards it once loaded; 'cache' stores it "
        "in the hidden states path, enabling hybrid online/offline training.",
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="Timeout in seconds for each individual vLLM request. Only applies "
        "if --on-missing=generate.",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        description="Maximum retry attempts per vLLM request on failure. Only applies "
        "if --on-missing=generate.",
    )


class LossArgs(_Group):
    loss_fn: str = Field(
        default="kl_div",
        description="Loss function specification. A name (kl_div, rkl, jsd, ce, tv, "
        'nla, lk_hybrid) or a JSON dict for a weighted combination, e.g. \'{"ce": 0.1, '
        '"tv": 0.9}\'.',
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
        resolve_loss_config(v)
        return v


class OptimizerArgs(_Group):
    optimizer: Literal["adamw", "muon"] = Field(
        default="muon",
        description="Optimizer to use. 'muon' applies Muon to 2D weight matrices and "
        "AdamW to the remaining params (norms, biases, embeddings, lm_head).",
    )
    lr: float = Field(default=1e-4, description="Learning rate (AdamW / base group).")
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for the AdamW optimizer (and the AdamW group in muon "
        "mode).",
    )
    muon_lr: float | None = Field(
        default=None,
        description="LR for the Muon (2D weights) group. Only used with --optimizer "
        "muon. Defaults to 10*lr (and --lr defaults to 1e-4).",
    )
    muon_momentum: float = Field(
        default=0.95, description="Momentum for the Muon group."
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
        description="Warmup as a fraction of total scheduler steps, in [0, 1]. Ignored "
        "(with a warning) when --scheduler-warmup-steps is also set.",
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
        description="Also point a checkpoint at the lowest validation loss.",
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
    fsdp_shard: bool = Field(
        default=False,
        description="Shard model parameters across GPUs with FSDP. By default "
        "parameters are fully replicated (DDP-like). Enable when the model does not "
        "fit in a single GPU's memory.",
    )

    @field_validator("checkpoint_freq")
    @classmethod
    def _validate_checkpoint_freq(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("--checkpoint-freq must be > 0")
        if v > 1 and not float(v).is_integer():
            raise ValueError(
                f"--checkpoint-freq={v} is not an integer. Values > 1 are treated as "
                "epoch counts and must be whole numbers."
            )
        return v


class LoggingArgs(_Group):
    logger: str = Field(
        default="",
        description="One of 'trackio', 'wandb', 'tensorboard', 'mlflow' or a comma "
        "separated list.",
    )
    log_dir: str = Field(default="./logs", description="Directory for training logs.")
    run_name: str | None = Field(
        default=None,
        description="Name for this run (used by metric loggers). Auto-generated when "
        "unset.",
    )


class DFlashArgs(_Group):
    """DFlash-family backbone knobs (also used by DSpark, which is-a DFlash)."""

    block_size: int = Field(
        default=8, description="Block size for DFlash model (default: 8)."
    )
    sample_from_anchor: bool | None = Field(
        default=None,
        description="Sample from the anchor position (all positions predict). "
        "Default: False for dflash, True for dspark.",
    )
    dflash_decay_gamma: float = Field(
        default=4.0, description="Decay gamma for DFlash/DSpark loss weighting."
    )
    per_position_loss_weight: Literal["fixed-exp-decay", "dpace"] = Field(
        default="fixed-exp-decay",
        description="Per-position loss weight option for D-PACE support "
        "(default: fixed-exp-decay).",
    )
    dpace_alpha: float = Field(
        default=0.5,
        description="Smoothing constant for the D-PACE loss (default: 0.5). Must be in "
        "(0, 1] when --per-position-loss-weight=dpace.",
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
        default=3, description="Number of MTP prediction steps (default: 3)."
    )
    step_weight_beta: float = Field(
        default=0.6,
        description="Exponential decay factor for MTP step weights; higher weights "
        "earlier prediction steps more. Only used with MTP.",
    )


# Group attribute name -> group model. Order defines both the flatten() key order
# (after the root scalars) and a dumped run.yaml's group layout.
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
    """Top-level trainer configuration -- the package's one public type.

    Composed from the typed groups (:data:`_GROUPS`) plus the root-level scalars
    declared here. Constructible from defaults with no arguments: the verifier
    path is not required at the schema level (a config may supply it later via
    YAML), and the resolution layer restores the required-flag contract.

    Subclasses ``BaseSettings`` to reuse pydantic-settings' source ordering as
    the precedence engine: the resolution layer passes the ``flag`` and ``yaml``
    layers as a private ``_layers`` kwarg and lets pydantic deep-merge them over
    the field defaults, giving ``flag > yaml > default`` without a hand-rolled
    merge (see :meth:`settings_customise_sources`). Plain keyword construction
    (:meth:`from_flat`) still works and bypasses that machinery.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        validate_assignment=False,
        nested_model_default_partial_update=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Turn the precedence rule ``flag > yaml > default`` into source ordering.

        pydantic-settings calls this hook by keyword, so all five parameters are
        part of the contract even though env/dotenv/secret sources are excluded:
        the trainer is configured only via CLI + YAML. The resolution layer
        passes the two explicit layers as a private ``_layers`` kwarg (popped here
        so it never becomes model data); each becomes its own
        :class:`InitSettingsSource` in precedence order, above the field defaults.
        A plain keyword construction carries no ``_layers`` and behaves like a
        normal ``BaseModel``.
        """
        init_kwargs = dict(cast("InitSettingsSource", init_settings).init_kwargs)
        layers = init_kwargs.pop("_layers", None)
        if layers is None:
            return (init_settings,)
        return tuple(
            InitSettingsSource(settings_cls, init_kwargs=layers.get(name, {}))
            for name in ("flag", "yaml")
        )

    # Resolution bookkeeping: the per-dest winning layer
    # (``flag | yaml | default``) and the argv the run was resolved from. In
    # memory only -- consumed by the draft-init conflict check and the
    # reproducibility manifest, never persisted and not part of the config shape.
    # Exposed read-only via the ``provenance`` / ``argv`` properties below.
    _provenance: dict[str, str] = PrivateAttr(default_factory=dict)
    _argv: list[str] = PrivateAttr(default_factory=list)

    @property
    def provenance(self) -> Mapping[str, str]:
        """The per-field winning layer (``'flag'`` | ``'yaml'`` | ``'default'``)."""
        return MappingProxyType(self._provenance)

    @property
    def argv(self) -> Sequence[str]:
        """The argv the run was resolved from."""
        return tuple(self._argv)

    speculator_type: str = Field(
        default="eagle3",
        description="Type of speculator model to train "
        "(eagle3, dflash, dspark, peagle, mtp).",
    )
    dry_run: bool = Field(
        default=False,
        description="Build the speculator, initialize weights, save a checkpoint to "
        "--save-path, then exit before training. Useful to validate the config and "
        "weights (e.g. in vLLM) before a full run. Can be combined with --draft-config "
        "or --from-pretrained.",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    deterministic_cuda: bool = Field(
        default=False,
        description="Set cuda to deterministic mode. This may impact performance.",
    )

    verifier: VerifierArgs = Field(default_factory=VerifierArgs)
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

    @model_validator(mode="after")
    def _resolve_derived_defaults(self) -> "TrainConfig":
        """Fill defaults that derive from other fields, mirroring the tail of the
        pre-refactor ``parse_args``: unset ``draft_arch`` -> ``llama`` for eagle3 else
        ``qwen3``; unset ``norm_before_fc`` / ``norm_output`` -> ``True`` for eagle3
        else ``False``; unset ``muon_lr`` -> ``10 * lr``.

        Idempotent: a concrete value (as produced by :meth:`flatten`) is left
        untouched, so :meth:`from_flat` round-trips.
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
        """D-PACE per-position loss weighting requires CE loss and a smoothing constant
        in ``(0, 1]``. Only enforced when actually selected, so it never fires on a
        default (``fixed-exp-decay``) run."""
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

    def flatten(self) -> dict[str, Any]:
        """The flat ``vars(args)``-shaped dict the ``SpeculatorModel`` classes
        consume via ``**kwargs``.

        Keys are emitted in schema-declaration order (root scalars, then each
        group's fields); consumers bind by name, so only the key set and values
        matter, but the order is deterministic so the ``run.yaml`` dump is stable.
        """
        values = {name: getattr(self, name) for name in _ROOT_FIELDS}
        for gname in _GROUPS:
            group = getattr(self, gname)
            for fname in type(group).model_fields:
                values[fname] = getattr(group, fname)
        return values

    @classmethod
    def from_flat(cls, flat: dict[str, Any]) -> "TrainConfig":
        """Inverse of :meth:`flatten`: recover the typed config from a flat
        working-dict. Non-config keys are dropped; the value validators are
        idempotent on already-resolved values."""
        known = {dest: value for dest, value in flat.items() if dest in CONFIG_DESTS}
        return cls(**nest_flat(known))

    @classmethod
    def from_sources(
        cls,
        *,
        cli: dict[str, Any],
        config_path: str | None = None,
        argv: list[str],
    ) -> "TrainConfig":
        """The pure, argv-free core: layer the sources into a validated config and
        record each value's origin. Raises on bad input; never exits.

        ``cli`` is the flat ``{dest: value}`` map of flags the user passed;
        ``config_path`` names an optional stage-shaped YAML file layered beneath
        the flags; ``argv`` is the command recorded for the reproducibility
        manifest. This is the primary test seam -- it can be exercised without
        ``sys.argv``.
        """
        from speculators.train.config import resolution  # noqa: PLC0415

        return resolution.build_from_sources(
            cls, cli=cli, config_path=config_path, argv=argv
        )

    @classmethod
    def resolve(cls, argv: list[str] | None = None) -> "TrainConfig":
        """The impure CLI boundary: parse argv, layer, validate.

        Turns any configuration error into a clean ``SystemExit(2)``. This is what
        ``scripts/train.py`` calls.
        """
        from speculators.train.config import resolution  # noqa: PLC0415

        return resolution.resolve(cls, argv)

    def dump_yaml(self) -> str:
        """Serialize this config to the stage-shaped, round-trippable YAML.

        The ``train:``-wrapped form, clean of provenance annotations, so it
        re-loads identically via ``--config`` (see ``artifacts`` for the writer).
        """
        from speculators.train.config import artifacts  # noqa: PLC0415

        return artifacts.dump_yaml(self)

    def save(self, save_dir: str) -> None:
        """Write the reproducibility artifacts (``run.yaml`` + ``train_command.txt``)
        next to the checkpoints. Called at rank 0."""
        from speculators.train.config import artifacts  # noqa: PLC0415

        artifacts.save(self, save_dir)


# Root-level scalar fields (not inside any group), in declaration order.
_ROOT_FIELDS: tuple[str, ...] = tuple(
    name for name in TrainConfig.model_fields if name not in _GROUPS
)


def _build_dest_to_group() -> dict[str, str | None]:
    """Map each flat dest to its owning group (``None`` for a root scalar)."""
    mapping: dict[str, str | None] = dict.fromkeys(_ROOT_FIELDS)
    for gname, gmodel in _GROUPS.items():
        for fname in gmodel.model_fields:
            if fname in mapping:
                raise RuntimeError(f"duplicate config field '{fname}' across groups")
            mapping[fname] = gname
    return mapping


# Flat dest -> owning group attribute (or None for a root scalar).
_DEST_TO_GROUP: dict[str, str | None] = _build_dest_to_group()

# Every flat dest owned by the config.
CONFIG_DESTS: frozenset[str] = frozenset(_DEST_TO_GROUP)


def nest_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Turn a flat ``{dest: value}`` dict into the nested group structure that
    :class:`TrainConfig` is constructed from."""
    nested: dict[str, Any] = {}
    for dest, value in flat.items():
        group = _DEST_TO_GROUP.get(dest)
        if group is None:
            nested[dest] = value
        else:
            nested.setdefault(group, {})[dest] = value
    return nested
