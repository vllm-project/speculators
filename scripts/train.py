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

import json

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


def _patch_speculator_configs_for_transformers5() -> None:
    """Compat shim for transformers>=5.x.

    Starting with transformers 5.x, ``PreTrainedConfig`` became a
    ``@strict @dataclass`` whose auto-generated ``__init__`` performs
    ``setattr(self, f.name, f.default)`` for every field *before* any
    user/pydantic logic runs. Because speculators' ``SpeculatorModelConfig``
    multiple-inherits from both ``pydantic.BaseModel`` and
    ``PreTrainedConfig``, those ``setattr`` calls hit pydantic's
    ``__setattr__`` before pydantic's ``__init__`` has created
    ``__pydantic_fields_set__``, raising ``AttributeError``.

    We replace the init on every ``SpeculatorModelConfig`` subclass with a
    pydantic-first init that:
      1) pre-seeds pydantic's private slots so any incidental ``setattr``
         is safe,
      2) runs ``pydantic.BaseModel.__init__`` to validate & populate fields,
      3) forwards all validated field values back into kwargs,
      4) calls HF's ``__post_init__`` on the leftover kwargs to preserve
         ``_name_or_path``, ``_attn_implementation_internal``, ``num_labels``
         behavior, etc. — *without* the field-reset loop that clobbers
         pydantic state,
      5) stamps ``transformers_version``, matching the original
         ``SpeculatorModelConfig.__init__`` behavior.
    """
    from importlib.metadata import version as _pkg_version

    import pydantic
    from speculators import SpeculatorModelConfig

    def patched_init(self, **kwargs):
        # 1) Seed pydantic private slots BEFORE any setattr can fire.
        object.__setattr__(self, "__pydantic_fields_set__", set())
        object.__setattr__(self, "__pydantic_extra__", {})
        object.__setattr__(self, "__pydantic_private__", None)

        cls = type(self)
        model_fields = cls.model_fields

        # 2) Pydantic validation + field population (only recognized fields).
        pydantic.BaseModel.__init__(
            self, **{k: v for k, v in kwargs.items() if k in model_fields}
        )

        # 3) Mirror SpeculatorModelConfig.__init__: push validated values back
        #    so __post_init__ sees the canonical versions.
        for field in model_fields:
            kwargs[field] = getattr(self, field)

        # 4) Run HF PreTrainedConfig.__post_init__ on leftover kwargs only.
        leftover = {k: v for k, v in kwargs.items() if k not in model_fields}
        post_init = getattr(self, "__post_init__", None)
        if callable(post_init):
            post_init(**leftover)

        # 5) Match original behavior: always stamp transformers version.
        self.transformers_version = _pkg_version("transformers")

    # Apply to base + every already-registered subclass.
    targets = {SpeculatorModelConfig}

    def _collect_subclasses(klass):
        for sub in klass.__subclasses__():
            if sub in targets:
                continue
            targets.add(sub)
            _collect_subclasses(sub)

    _collect_subclasses(SpeculatorModelConfig)

    # Force pydantic schema rebuild with torch in the type namespace so that
    # field annotations referencing torch.Tensor (e.g. in Eagle3/DFlash
    # configs) resolve properly. Without this, pydantic keeps a mock
    # validator that raises "class-not-fully-defined" when we drive
    # BaseModel.__init__ directly inside the patched init.
    import torch as _torch  # noqa: WPS433

    for cls in targets:
        try:
            cls.model_rebuild(force=True, _types_namespace={"torch": _torch})
        except Exception as err:  # noqa: BLE001
            logger.debug("model_rebuild skipped for %s: %s", cls.__name__, err)

    for cls in targets:
        cls.__init__ = patched_init


_patch_speculator_configs_for_transformers5()

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
    """Map verifier FFN width to a dense draft FFN width.

    Handles three layouts:
      1) Dense configs (Llama / Qwen3 / Qwen3-text-only): use ``intermediate_size``.
      2) Classic MoE-text configs that still expose ``intermediate_size`` alongside
         ``moe_intermediate_size`` (e.g. some Qwen2/3 MoE variants): approximate
         the per-token active FFN as ``moe_intermediate_size * num_experts_per_tok``.
      3) Qwen3.5-MoE-style text configs that DELIBERATELY omit ``intermediate_size``
         and only expose ``moe_intermediate_size`` (+ ``shared_expert_intermediate_size``):
         approximate as ``moe_intermediate_size * num_experts_per_tok``
         plus the (always-on) shared-expert width when present.
    """
    # Detect MoE text layout via presence of moe_intermediate_size (more reliable
    # than model_type suffix matching, which misses Qwen3_5MoeTextConfig).
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
        f"Cannot infer draft intermediate_size from verifier config "
        f"{type(verifier_config).__name__}: neither 'intermediate_size' nor "
        f"'moe_intermediate_size' is present. Please pass "
        f"--draft-intermediate-size explicitly."
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
        collate_fn=create_collate_fn(args.total_seq_len, hidden_size, preprocess),
        persistent_workers=True,
    )


def create_transformer_layer_config(
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str = "llama",
    draft_intermediate_size: int | None = None,
    # --- NEW: optional drafter head-geometry overrides --------------------
    # Rationale: vLLM's FA3 kernel rejects some (n_heads, head_dim) combos
    # (e.g. Qwen3.6's native (16, 2, 256)).  Allowing the drafter to be
    # re-factored into an FA3-safe shape like (32, 4, 128) while keeping the
    # same total Q/K/V row count preserves hidden_size and GQA ratio, so the
    # training objective is unchanged and inference is unblocked.
    draft_num_attention_heads: int | None = None,
    draft_num_key_value_heads: int | None = None,
    draft_head_dim: int | None = None,
    # --- NEW: optional RoPE scaling override ------------------------------
    # Rationale: the drafter inherits `rope_scaling` from the verifier's
    # text config.  Qwen3.6-35B-A3B ships WITHOUT `rope_scaling`, so
    # training runs use unscaled RoPE and any downstream YaRN declared only
    # in the exported drafter config would be a train/inference mismatch.
    # Passing a scaling dict here propagates it into the drafter config at
    # training time, so Qwen3RotaryEmbedding actually builds YaRN-stretched
    # inv_freq during the forward pass and the trained weights learn the
    # stretched frequencies.
    draft_rope_scaling: dict | None = None,
    # --- NEW: direct RoPE base / max-position overrides -------------------
    # Rationale: users training drafters for short-context / non-YaRN
    # deployment (e.g. Eagle3 head with native 4k window and rope_theta=1e6)
    # need to decouple the drafter's RoPE base and position budget from the
    # verifier's long-context settings WITHOUT simultaneously injecting a
    # rope_scaling dict (which the existing --draft-rope-scaling backdoor
    # requires).  These two flags are the clean, orthogonal knobs.
    draft_rope_theta: float | None = None,
    draft_max_position_embeddings: int | None = None,
    # --- NEW: MRoPE full-head-rotation consolidation (Option 2) -----------
    # Rationale: Qwen3-Omni / Qwen3.6 verifiers ship with
    # ``rope_parameters.partial_rotary_factor < 1.0`` (e.g. 0.25 for Qwen3.6)
    # meaning only the first ``rotary_dim = head_dim * partial_rotary_factor``
    # channels of each attention head are rotated. At *inference* time vLLM
    # implements this natively via ``MRotaryEmbeddingInterleaved`` by slicing
    # ``q[..., :rotary_dim]`` and pairing neighbouring channels as
    # ``(i, i + rotary_dim/2)``. At *training* time the HuggingFace-based
    # trainer instead calls ``apply_rotary_pos_emb(q, k, cos, sin)`` which
    # applies rotation via ``rotate_half``, pairing channels as
    # ``(i, i + head_dim/2)``. These pairings are DIFFERENT whenever
    # ``rotary_dim < head_dim``, so no amount of cos/sin padding can make the
    # trainer's forward bit-match vLLM's. Empirically (Eagle3-v5 experiments)
    # a draft trained under partial=0.25 + the (broken) ``PartialMRoPE``
    # wrapper gets ~44% vLLM acceptance, whereas the SAME weights served with
    # a ``partial=1.0`` config (and a proportionally rescaled mrope_section)
    # get ~55% — because the two pairings coincide only when rotary_dim ==
    # head_dim. Since HF and vLLM agree bit-for-bit in that special case,
    # we force the draft into that regime: rescale ``mrope_section`` by
    # ``1/partial_rotary_factor`` and set ``partial_rotary_factor=1.0``. The
    # draft then rotates all head_dim channels; the trainer and vLLM see the
    # same rotation pairings; everything is self-consistent. The draft's
    # rotary semantic intentionally diverges from the verifier's at this
    # point — that is fine because the draft's attention operates on
    # ``target_hidden_3H`` (already-post-verifier-attention) and only needs
    # internal train/inference agreement. See
    # ``project_eagle3_v4_root_cause.md`` in the project memory for the full
    # derivation.
    #
    # Default: ``True`` so new runs automatically use the correct regime.
    # Set to ``False`` only when you have implemented a vLLM-compatible
    # partial-rotation in the trainer's attention (see "Option 1" in the
    # debugging notes) and want to verify the full-head hack is no longer
    # needed.
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
        AutoConfig.from_pretrained(verifier_name_or_path)
    )

    if draft_intermediate_size is None:
        draft_intermediate_size = get_default_draft_intermediate_size(verifier_config)

    # Resolve final head geometry: CLI override > verifier default.
    n_heads = (
        draft_num_attention_heads
        if draft_num_attention_heads is not None
        else verifier_config.num_attention_heads
    )
    n_kv = (
        draft_num_key_value_heads
        if draft_num_key_value_heads is not None
        else verifier_config.num_key_value_heads
    )
    hd = (
        draft_head_dim
        if draft_head_dim is not None
        else getattr(verifier_config, "head_dim", None)
    )

    # Invariants we actually need to enforce:
    #   1. GQA ratio must be integral (n_heads % n_kv == 0) so each KV head
    #      maps to a whole number of Q heads during attention broadcasting.
    #   2. head_dim must be positive.
    # NOTE: n_heads * head_dim does NOT need to equal hidden_size.  In Qwen3
    # and Llama-style attention the Q/K/V projections are free to project the
    # residual stream up to (n_heads * head_dim) and the O projection maps
    # back down to hidden_size.  Enforcing equality here would reject the
    # official DFlash release for Qwen3.6 (n_heads=32, head_dim=128,
    # hidden_size=2048 -> 4096 != 2048) and the verifier itself
    # (n_heads=16, head_dim=256 -> 4096 != 2048).
    hidden_size = verifier_config.hidden_size
    if n_heads % n_kv != 0:
        raise ValueError(
            f"Invalid GQA ratio: num_attention_heads({n_heads}) must be "
            f"divisible by num_key_value_heads({n_kv})."
        )
    if hd is not None and hd <= 0:
        raise ValueError(f"Invalid head_dim({hd}); must be positive.")
    if (
        draft_num_attention_heads is not None
        or draft_num_key_value_heads is not None
        or draft_head_dim is not None
    ):
        logger.info(
            f"Drafter head geometry overridden: num_attention_heads={n_heads}, "
            f"num_key_value_heads={n_kv}, head_dim={hd} "
            f"(verifier defaults were {verifier_config.num_attention_heads}, "
            f"{verifier_config.num_key_value_heads}, "
            f"{getattr(verifier_config, 'head_dim', None)})."
        )

    rope_kwargs: dict = {}

    # ------------------------------------------------------------------
    # rope_theta: inherit from verifier, honouring multiple shapes.
    # On vanilla Qwen3 the frequency base lives at ``rope_theta`` on the
    # text config. On Qwen3-Omni (and other recent transformers releases)
    # the same value is nested inside ``rope_parameters`` / ``rope_scaling``
    # alongside MRoPE metadata, and the top-level ``rope_theta`` attribute
    # is ``None``. If we only look at the top-level attribute the drafter
    # silently falls back to Qwen3Config's default of 1e4, which is
    # ~1000x off from the verifier's 1e7 and collapses draft acceptance.
    # CLI can still force a specific value via ``--draft-rope-scaling``
    # by embedding ``"rope_theta": ...`` in the JSON blob.
    # ------------------------------------------------------------------
    verifier_rope_theta = getattr(verifier_config, "rope_theta", None)
    if verifier_rope_theta is None:
        for src_name in ("rope_parameters", "rope_scaling"):
            src = getattr(verifier_config, src_name, None)
            if isinstance(src, dict) and src.get("rope_theta") is not None:
                verifier_rope_theta = src["rope_theta"]
                logger.info(
                    f"Drafter rope_theta recovered from verifier "
                    f"{src_name}.rope_theta = {verifier_rope_theta} "
                    f"(top-level rope_theta was None)."
                )
                break
    if verifier_rope_theta is not None:
        rope_kwargs["rope_theta"] = verifier_rope_theta

    if getattr(verifier_config, "rope_scaling", None) is not None:
        rope_kwargs["rope_scaling"] = dict(verifier_config.rope_scaling)
    # CLI override takes precedence over verifier-inherited rope_scaling.
    if draft_rope_scaling is not None:
        cli_rope_scaling = dict(draft_rope_scaling)
        # Allow callers to also override rope_theta via the same JSON blob
        # (kept separate from the rope_scaling dict so downstream configs
        # see a clean scaling payload).
        cli_rope_theta = cli_rope_scaling.pop("rope_theta", None)
        if cli_rope_theta is not None:
            rope_kwargs["rope_theta"] = cli_rope_theta
            logger.info(
                f"Drafter rope_theta overridden via CLI: {cli_rope_theta}"
            )
        rope_kwargs["rope_scaling"] = cli_rope_scaling
        logger.info(
            f"Drafter rope_scaling overridden via CLI: {cli_rope_scaling}"
        )

    # Direct --draft-rope-theta takes ultimate precedence (applied LAST so it
    # wins over both verifier inheritance and any rope_theta piggy-backed in
    # --draft-rope-scaling).  This is the intended path for users who want
    # rope_scaling=None but still need to retune the RoPE base (e.g. Eagle3
    # drafter with native 4k context and rope_theta=1e6 instead of the
    # verifier's 1e7).
    if draft_rope_theta is not None:
        rope_kwargs["rope_theta"] = float(draft_rope_theta)
        logger.info(
            f"Drafter rope_theta overridden via --draft-rope-theta: "
            f"{rope_kwargs['rope_theta']}"
        )

    # --------------------------------------------------------------
    # MRoPE (Qwen3-Omni / Qwen3-VL) bookkeeping.
    # Modern ``LlamaConfig`` is ``@strict`` and therefore does NOT accept a
    # top-level ``rope_theta`` kwarg; only ``rope_parameters`` survives. But
    # the ``rope_scaling`` attribute is defined as a property that proxies
    # to ``rope_parameters`` for backwards compatibility. When the verifier
    # carries MRoPE metadata (e.g. Qwen3.6's ``mrope_section``,
    # ``mrope_interleaved``), the ``rope_kwargs["rope_scaling"]`` branch
    # above already propagates the full dict -- including ``rope_theta``
    # and ``rope_type`` -- into the draft config's ``rope_parameters``.
    # That makes the subsequent ``_select_rotary_emb_class`` check inside
    # ``Eagle3DraftModel.__init__`` correctly pick
    # ``Qwen3OmniMoeThinkerTextRotaryEmbedding`` over the default
    # ``LlamaRotaryEmbedding``.
    #
    # To avoid a stale ``rope_theta`` when the user passes
    # ``--draft-rope-theta`` with an MRoPE verifier, fold the override into
    # the scaling dict too so the MRoPE rotary reads the intended base.
    # --------------------------------------------------------------
    rope_scaling_dict = rope_kwargs.get("rope_scaling")
    if (
        isinstance(rope_scaling_dict, dict)
        and "mrope_section" in rope_scaling_dict
    ):
        if draft_rope_theta is not None:
            rope_scaling_dict["rope_theta"] = float(draft_rope_theta)
        elif "rope_theta" in rope_kwargs and "rope_theta" not in rope_scaling_dict:
            # keep legacy path happy (Qwen3-Omni verifier already ships
            # rope_theta inside rope_parameters, so this branch is only a
            # safety net for non-standard configs).
            rope_scaling_dict["rope_theta"] = rope_kwargs["rope_theta"]
        logger.info(
            "MRoPE detected on draft config: "
            f"mrope_section={rope_scaling_dict.get('mrope_section')}, "
            f"mrope_interleaved={rope_scaling_dict.get('mrope_interleaved')}, "
            f"rope_theta={rope_scaling_dict.get('rope_theta')}. "
            "Eagle3DraftModel will use Qwen3OmniMoeThinkerTextRotaryEmbedding."
        )

        # -------------------------------------------------------------
        # Option 2 (see docstring above mrope_full_head_hack arg): force
        # the draft into the ``partial_rotary_factor=1.0`` regime so HF's
        # rotate_half-style rotation in the trainer matches vLLM's neox
        # partial rotation bit-for-bit at inference time.
        # -------------------------------------------------------------
        inherited_partial = rope_scaling_dict.get("partial_rotary_factor", 1.0)
        if mrope_full_head_hack and float(inherited_partial) < 1.0:
            old_partial = float(inherited_partial)
            old_section = list(rope_scaling_dict["mrope_section"])
            # 1/partial_rotary_factor must be an integer for the rescaling
            # to preserve the T/H/W ratio exactly; if not, raise loudly so
            # the user is forced to supply a compatible set manually via
            # --draft-rope-scaling.
            inv = 1.0 / old_partial
            if abs(inv - round(inv)) > 1e-6:
                raise ValueError(
                    "mrope_full_head_hack cannot rescale mrope_section: "
                    f"1/partial_rotary_factor = {inv} is not integer. "
                    "Supply a custom rope_scaling dict via "
                    "--draft-rope-scaling to pre-specify mrope_section and "
                    "partial_rotary_factor=1.0, or disable the hack with "
                    "--no-draft-mrope-full-head-hack."
                )
            scale = int(round(inv))
            new_section = [int(x) * scale for x in old_section]
            # Sanity: 2 * sum(new_section) must equal head_dim, otherwise
            # vLLM's MRotaryEmbedding.__init__ will assert-fail.
            expected = hd  # head_dim after overrides (see above)
            if 2 * sum(new_section) != expected:
                raise ValueError(
                    "mrope_full_head_hack rescaling produced inconsistent "
                    f"mrope_section {new_section}: 2*sum={2*sum(new_section)} "
                    f"but head_dim={expected}. This usually means the "
                    f"verifier's original mrope_section {old_section} does "
                    f"not evenly tile rotary_dim/2 = "
                    f"{int(expected * old_partial) // 2}. Inspect the "
                    "verifier's rope_parameters and pass a correct "
                    "--draft-rope-scaling manually."
                )
            rope_scaling_dict["mrope_section"] = new_section
            rope_scaling_dict["partial_rotary_factor"] = 1.0
            logger.warning(
                "=" * 70 + "\n"
                "MRoPE FULL-HEAD HACK APPLIED (--draft-mrope-full-head-hack):\n"
                f"  partial_rotary_factor: {old_partial} -> 1.0\n"
                f"  mrope_section:         {old_section} -> {new_section}\n"
                "Why: HF's rotate_half-style rotation (used by the trainer's "
                "LlamaAttention) pairs head-dim channels as (i, i+head_dim/2), "
                "while vLLM's neox-partial rotation (used at inference) pairs "
                "(i, i+rotary_dim/2). These pairings differ whenever "
                "rotary_dim < head_dim, so a partial=0.25 draft trained under "
                "HF will NOT be bit-equivalent to the same draft served under "
                "vLLM. Forcing partial=1.0 makes rotary_dim == head_dim, where "
                "both pairings coincide and train/inference agree exactly. "
                "The draft's rotation is self-consistent; it intentionally "
                "differs from the verifier's (which is fine because the draft "
                "operates on post-verifier-attention hidden states).\n"
                "To disable this hack (e.g. once the trainer's attention has "
                "been patched with a vLLM-compatible partial rotation per "
                "Option 1), pass --no-draft-mrope-full-head-hack.\n" + "=" * 70
            )
        elif not mrope_full_head_hack and float(inherited_partial) < 1.0:
            logger.warning(
                "mrope_full_head_hack=False with partial_rotary_factor="
                f"{inherited_partial} < 1.0. The trainer's HF-based attention "
                "will rotate different head-dim channels than vLLM's native "
                "partial rotation at inference, producing a silent "
                "train/inference mismatch. Only run with this combination if "
                "you have patched the trainer's attention to use vLLM-"
                "compatible neox-partial rotation (Option 1)."
            )
        # -------------------------------------------------------------

    # Resolve max_position_embeddings: CLI override > verifier default.
    # Kept orthogonal to RoPE theta/scaling so users can shrink the drafter's
    # position budget (e.g. 4096) without touching frequency settings.
    max_pos = (
        int(draft_max_position_embeddings)
        if draft_max_position_embeddings is not None
        else verifier_config.max_position_embeddings
    )
    if draft_max_position_embeddings is not None:
        logger.info(
            f"Drafter max_position_embeddings overridden via CLI: {max_pos} "
            f"(verifier default was {verifier_config.max_position_embeddings})."
        )

    logger.info(
        f"Drafter RoPE resolved: rope_theta="
        f"{rope_kwargs.get('rope_theta', '<default>')}, "
        f"rope_scaling={rope_kwargs.get('rope_scaling', None)}"
    )

    hidden_act = getattr(verifier_config, "hidden_act", None) or getattr(
        verifier_config, "hidden_activation", None
    )
    if hidden_act is None:
        raise AttributeError(
            f"{type(verifier_config).__name__} has neither 'hidden_act' "
            "nor 'hidden_activation'"
        )

    return config_class(
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
        head_dim=hd,
        tie_word_embeddings=False,
        **rope_kwargs,
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
    # `--draft-vocab-size 0` (or any non-positive value) is an explicit request
    # for the "full verifier vocab" path. Normalize it to None so every
    # downstream branch treats it uniformly and we never silently reload stale
    # cached mappings from a previous run.
    if args.draft_vocab_size is not None and args.draft_vocab_size <= 0:
        logger.info(
            "--draft-vocab-size<=0 received; using full verifier vocab and "
            "ignoring any cached d2t/t2d/token_freq artifacts in --data-path."
        )
        args.draft_vocab_size = None
        use_full_vocab = True
    else:
        use_full_vocab = False

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

    # Short-circuit to the full-vocab branch *before* looking at cached files
    # when the user explicitly opted in. This prevents the classic footgun where
    # a 32k-truncated d2t.npy cached by an earlier run silently overrides a new
    # full-vocab training job and later trips the t2d guard in
    # `DraftVocabMixin.load_verifier_weights`.
    if use_full_vocab:
        verifier_config = unwrap_verifier_text_config(
            AutoConfig.from_pretrained(args.verifier_name_or_path)
        )
        return None, None, verifier_config.vocab_size

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
        AutoConfig.from_pretrained(args.verifier_name_or_path)
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

    d2t, t2d, draft_vocab_size = parse_vocab_mappings(args)

    # Setup speculator config
    transformer_layer_config = create_transformer_layer_config(
        args.verifier_name_or_path,
        args.num_layers,
        draft_arch=args.draft_arch,
        draft_intermediate_size=args.draft_intermediate_size,
        draft_num_attention_heads=args.draft_num_attention_heads,
        draft_num_key_value_heads=args.draft_num_key_value_heads,
        draft_head_dim=args.draft_head_dim,
        draft_rope_scaling=args.draft_rope_scaling,
        draft_rope_theta=args.draft_rope_theta,
        draft_max_position_embeddings=args.draft_max_position_embeddings,
        mrope_full_head_hack=args.draft_mrope_full_head_hack,
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

    # Setup dataloaders
    preprocess = shift_batch if args.speculator_type in ("eagle3", "peagle") else None

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
        # Only enable verifier-aware RoPE (3D MRoPE) under --multimodal to
        # avoid unnecessary AutoConfig.from_pretrained IO on text-only runs,
        # and to keep train/val dataset RoPE behaviour symmetric.
        rope_verifier = args.verifier_name_or_path if args.multimodal else None

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
            verifier_name_or_path=rope_verifier,
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
            verifier_name_or_path=rope_verifier,
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
        help="Type of speculator model to train (e.g., eagle3)",
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
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help=(
            "Enable multimodal Arrow dataset handling, including sidecar loading, "
            "chat-based hidden-state generation, and 3D MRoPE position_ids."
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
        "--draft-intermediate-size",
        type=int,
        default=None,
        help=(
            "Override draft FFN intermediate size. For MoE verifiers (for example "
            "Qwen3-Omni-Thinking), the default is "
            "moe_intermediate_size * num_experts_per_tok."
        ),
    )
    parser.add_argument(
        "--draft-num-attention-heads",
        type=int,
        default=None,
        help=(
            "Override the drafter's num_attention_heads. The product "
            "num_attention_heads * head_dim must equal the verifier's "
            "hidden_size. Useful when the verifier's native head layout is "
            "incompatible with the inference engine's attention kernel "
            "(e.g. vLLM's FA3 rejects Qwen3.6's native (16, 2, 256); use "
            "(32, 4, 128) to match the official DFlash release."
        ),
    )
    parser.add_argument(
        "--draft-num-key-value-heads",
        type=int,
        default=None,
        help=(
            "Override the drafter's num_key_value_heads (GQA). Must divide "
            "--draft-num-attention-heads. When changing head geometry, keep "
            "n_heads/n_kv the same as the verifier to preserve the trained "
            "GQA ratio (e.g. Qwen3.6 verifier is 16/2=8, so use 32/4=8)."
        ),
    )
    parser.add_argument(
        "--draft-head-dim",
        type=int,
        default=None,
        help=(
            "Override the drafter's per-head hidden dim. Must satisfy "
            "num_attention_heads * head_dim == verifier hidden_size."
        ),
    )
    parser.add_argument(
        "--draft-rope-scaling",
        type=lambda s: json.loads(s) if s else None,
        default=None,
        help=(
            "JSON string declaring RoPE scaling (e.g. YaRN) to apply DURING "
            "TRAINING. Overrides any `rope_scaling` inherited from the "
            "verifier. Example: "
            "'{\"rope_type\":\"yarn\",\"factor\":64.0,\"original_max_position_embeddings\":4096,\"beta_fast\":32.0,\"beta_slow\":1.0}'. "
            "If your verifier lacks rope_scaling but you intend to serve the "
            "drafter with YaRN at inference, passing it here avoids the "
            "otherwise-silent train/inference RoPE-frequency mismatch."
        ),
    )
    parser.add_argument(
        "--draft-rope-theta",
        type=float,
        default=None,
        help=(
            "Override the drafter's RoPE frequency base (rope_theta) "
            "orthogonally to --draft-rope-scaling. Use when you want "
            "rope_scaling=None but a different theta than the verifier, "
            "e.g. an Eagle3 head trained with native 4k context and "
            "rope_theta=1e6 even though the verifier uses 1e7."
        ),
    )
    parser.add_argument(
        "--draft-max-position-embeddings",
        type=int,
        default=None,
        help=(
            "Override the drafter's max_position_embeddings. Defaults to the "
            "verifier's value, which is typically far larger than the "
            "drafter's true training/inference window. Set to e.g. 4096 for "
            "a short-context Eagle3 drafter."
        ),
    )
    parser.add_argument(
        "--draft-mrope-full-head-hack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the verifier carries MRoPE metadata with "
            "partial_rotary_factor < 1 (e.g. Qwen3.6's 0.25), rescale the "
            "draft's mrope_section by 1/partial_rotary_factor and pin "
            "partial_rotary_factor=1.0 before training. This keeps the "
            "trainer's HF-based rotate_half rotation bit-equivalent to "
            "vLLM's neox partial rotation at inference, eliminating a "
            "silent ~17pp acceptance regression observed with Qwen3.6 "
            "Eagle3 drafters. The draft's rotary intentionally diverges "
            "from the verifier's in this mode; see the long comment on "
            "`mrope_full_head_hack` in `create_transformer_layer_config` "
            "for the full rationale. "
            "Default: --draft-mrope-full-head-hack (on). Use "
            "--no-draft-mrope-full-head-hack only if you have implemented "
            "a vLLM-compatible partial rotation in the trainer's attention."
        ),
    )
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) A (space separated) list of verifier auxiliary hidden-state "
            "layer ids. Defaults to [2, num_hidden_layers // 2, num_hidden_layers - 3]. "
            "Do not include the final verifier layer, which is stored separately. "
            "Must match the layers used during hidden-states extraction."
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
