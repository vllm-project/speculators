"""H-Spec rotary embedding matching the target's shared-K rotation.

The Qwen3.8-style target rotates only a *prefix* of each attention head
(``partial_rotary_factor < 1``), so the K rows the drafter reads from the
target KV cache are prefix-rotated with pairing ``j <-> j + rotary_dim/2``
inside that prefix. The stock ``Qwen3RotaryEmbedding`` computes inverse
frequencies from the full ``head_dim`` and ignores the partial factor, so it
cannot reproduce that transform. This module provides:

* :class:`MambaAttnHybridRotaryEmbedding` — a small rotary module that
  generates ``rotary_dim = head_dim * partial_rotary_factor`` inverse
  frequencies (width-``rotary_dim`` cos/sin tables) while keeping full-head
  behavior when the factor is 1.0.
* helpers to derive the target's effective rope from the verifier config and
  reconcile it onto the draft ``transformer_layer_config`` (used by both the
  explicit ``--draft-config`` and the automatic decoder-config paths).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.utils.generic import maybe_autocast

logger = logging.getLogger(__name__)

__all__ = [
    "MambaAttnHybridRotaryEmbedding",
    "effective_rope",
    "reconcile_rope_with_verifier",
    "resolve_target_rope",
    "validate_checkpoint_rope_compat",
]

# Raw config keys that count as a deliberate rope override in a draft config
# JSON (checked *before* ``Qwen3Config`` inserts its defaults, so an omitted
# rope is distinguishable from an explicit one).
_EXPLICIT_ROPE_KEYS = (
    "rope_parameters",
    "rope_scaling",
    "rope_theta",
    "rope_dim",
    "partial_rotary_factor",
)


def _config_rope_parameters(config: PretrainedConfig) -> dict[str, Any]:
    """Return the effective flat rope parameter dict for a config."""
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is None:
        rope_params = getattr(config, "rope_scaling", None)
    if not isinstance(rope_params, dict):
        rope_params = {}
    rope_params = dict(rope_params)
    # Collapse nested per-layer-type rope (e.g. Laguna) to a flat rope,
    # preferring the full-attention variant: H-Spec taps full-attention
    # layers for KV sharing.
    if "rope_theta" not in rope_params:
        sub = rope_params.get("full_attention") or rope_params.get("sliding_attention")
        if isinstance(sub, dict):
            sub = dict(sub)
            sub.setdefault("rope_type", "default")
            sub.setdefault("rope_theta", 10000.0)
            rope_params = sub
        else:
            rope_params = {"rope_type": "default", "rope_theta": 10000.0}
    return rope_params


def _config_head_dim(config: PretrainedConfig) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    return int(config.hidden_size) // int(config.num_attention_heads)


def _rope_dim_from_params(head_dim: int, rope_params: dict[str, Any]) -> int:
    rope_dim = rope_params.get("rope_dim")
    if rope_dim is not None:
        return int(rope_dim)
    partial = float(rope_params.get("partial_rotary_factor", 1.0))
    if not 0.0 < partial <= 1.0:
        raise ValueError(f"partial_rotary_factor={partial} must be in (0.0, 1.0].")
    return int(head_dim * partial)


def effective_rope(config: PretrainedConfig) -> tuple[int, int, float]:
    """Return the effective ``(head_dim, rotary_dim, rope_theta)`` of a config."""
    head_dim = _config_head_dim(config)
    rope_params = _config_rope_parameters(config)
    theta = float(rope_params.get("rope_theta", 10000.0))
    return head_dim, _rope_dim_from_params(head_dim, rope_params), theta


def resolve_target_rope(
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Derive the target's effective *text/full-attention* rope settings.

    Returns ``(head_dim, rope_parameters)`` where ``rope_parameters`` is a
    normalized flat dict ``{"rope_type", "rope_theta", "partial_rotary_factor"}``.
    MRoPE metadata (``mrope_section`` / ``mrope_interleaved``) is dropped:
    for scalar text positions an MRotaryEmbedding applies the same ordinary
    partial neox rotation as a plain ``RotaryEmbedding``, so the normalized
    dict describes an equivalent transform.
    """
    verifier_config = AutoConfig.from_pretrained(
        verifier_name_or_path, trust_remote_code=trust_remote_code
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    head_dim = _config_head_dim(verifier_config)
    rope_params = _config_rope_parameters(verifier_config)
    rope_type = rope_params.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(
            "mamba_attn_hybrid supports only rope_type='default' targets; "
            f"verifier '{verifier_name_or_path}' uses rope_type='{rope_type}'."
        )
    normalized = {
        "rope_type": "default",
        "rope_theta": float(rope_params.get("rope_theta", 10000.0)),
        "partial_rotary_factor": float(rope_params.get("partial_rotary_factor", 1.0)),
    }
    return head_dim, normalized


def _load_raw_draft_config_dict(draft_config: str) -> dict[str, Any] | None:
    try:
        config_dict, _ = PretrainedConfig.get_config_dict(draft_config)
    except Exception:  # noqa: BLE001 -- best-effort provenance of overrides
        return None
    if "transformer_layer_config" in config_dict:
        config_dict = config_dict["transformer_layer_config"]
    return config_dict if isinstance(config_dict, dict) else None


def reconcile_rope_with_verifier(
    tl_config: PretrainedConfig,
    verifier_name_or_path: str,
    draft_config_path: str | None = None,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    """Reconcile the draft ``transformer_layer_config`` rope with the target.

    The drafter shares the target's rotated K prefix, so its Q/K rotation must
    be the same effective transform as the target's text/full-attention rope.
    On fresh model construction:

    * If the draft config JSON *omitted* rope fields (the common case), the
      target's rope is copied onto ``tl_config`` and the resolved settings are
      logged. This covers both the explicit ``--draft-config`` path and the
      automatic decoder-config path (whose generic builder may otherwise strip
      ``partial_rotary_factor`` or rescale partial MRoPE to full-head).
    * If the draft config JSON *explicitly* set rope fields, the effective
      transform (``rope_theta``, ``rotary_dim``) must match the target's;
      anything incompatible with the shared target K is rejected with a clear
      diagnostic rather than silently accepted.

    ``head_dim`` must always match the target's (KV sharing requires equal
    geometry).
    """
    target_head_dim, target_rope = resolve_target_rope(
        verifier_name_or_path, trust_remote_code=trust_remote_code
    )
    target_rotary_dim = _rope_dim_from_params(target_head_dim, target_rope)

    draft_head_dim = _config_head_dim(tl_config)
    if draft_head_dim != target_head_dim:
        raise ValueError(
            "mamba_attn_hybrid draft head_dim "
            f"({draft_head_dim}) must equal the verifier head_dim "
            f"({target_head_dim}) for KV sharing; verifier "
            f"'{verifier_name_or_path}'."
        )

    explicit = False
    if draft_config_path:
        raw = _load_raw_draft_config_dict(draft_config_path)
        if raw is not None and any(k in raw for k in _EXPLICIT_ROPE_KEYS):
            explicit = True

    if explicit:
        draft_params = _config_rope_parameters(tl_config)
        raw = _load_raw_draft_config_dict(draft_config_path) or {}
        raw_params = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
        draft_theta = float(
            raw_params.get("rope_theta", draft_params.get("rope_theta", 10000.0))
        )
        draft_rotary_dim = _rope_dim_from_params(draft_head_dim, raw_params)
        problems = []
        if draft_rotary_dim != target_rotary_dim:
            problems.append(
                f"rotary_dim {draft_rotary_dim} != target {target_rotary_dim}"
            )
        if draft_theta != target_rope["rope_theta"]:
            problems.append(
                f"rope_theta {draft_theta} != target {target_rope['rope_theta']}"
            )
        if problems:
            raise ValueError(
                "--draft-config sets an explicit rope that is incompatible "
                "with the shared target K (verifier "
                f"'{verifier_name_or_path}'): {', '.join(problems)}. Remove "
                "the rope overrides so the target rope is inherited, or make "
                "them match the target."
            )

    # Normalize the draft config to the target's effective transform. Setting
    # rope_parameters keeps rope_scaling in sync on transformers>=5 and
    # serializes into the saved config for serving-side reconstruction.
    tl_config.rope_parameters = dict(target_rope)
    logger.info(
        "mamba_attn_hybrid rope resolved from verifier '%s': head_dim=%d "
        "rotary_dim=%d rope_theta=%s partial_rotary_factor=%s "
        "(explicit draft override: %s)",
        verifier_name_or_path,
        target_head_dim,
        target_rotary_dim,
        target_rope["rope_theta"],
        target_rope["partial_rotary_factor"],
        explicit,
    )
    return tl_config


def validate_checkpoint_rope_compat(
    checkpoint_path: str,
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> None:
    """Raise a clear diagnostic when a saved checkpoint's rope is incompatible.

    An old wrong-rope checkpoint is not repaired by editing its JSON: its
    weights were trained against a different forward operator. Refuse to
    resume/finetune it against this verifier instead of silently changing the
    rotation the saved weights were trained with.
    """
    target_head_dim, target_rope = resolve_target_rope(
        verifier_name_or_path, trust_remote_code=trust_remote_code
    )
    target_rotary_dim = _rope_dim_from_params(target_head_dim, target_rope)
    target_theta = target_rope["rope_theta"]

    config_dict, _ = PretrainedConfig.get_config_dict(checkpoint_path)
    tl_dict = config_dict.get("transformer_layer_config", config_dict)
    head_dim = tl_dict.get("head_dim") or tl_dict.get("hidden_size", 0) // (
        tl_dict.get("num_attention_heads", 1) or 1
    )
    rope_params = tl_dict.get("rope_parameters") or tl_dict.get("rope_scaling") or {}
    if not isinstance(rope_params, dict):
        rope_params = {}
    theta = float(rope_params.get("rope_theta", 10000.0))
    rotary_dim = _rope_dim_from_params(int(head_dim), rope_params)

    problems = []
    if int(head_dim) != target_head_dim:
        problems.append(f"head_dim {head_dim} != target {target_head_dim}")
    if rotary_dim != target_rotary_dim:
        problems.append(f"rotary_dim {rotary_dim} != target {target_rotary_dim}")
    if theta != target_theta:
        problems.append(f"rope_theta {theta} != target {target_theta}")
    if problems:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has a rope configuration "
            f"incompatible with verifier '{verifier_name_or_path}': "
            f"{', '.join(problems)}. The checkpoint's weights were trained "
            "with a different rotation than the shared target K; resuming "
            "would silently change the forward operator. Retrain from "
            "scratch with the repaired rope instead."
        )


class MambaAttnHybridRotaryEmbedding(nn.Module):
    """Partial-prefix rotary embedding for the H-Spec drafter.

    Unlike the stock ``Qwen3RotaryEmbedding`` (which derives frequencies from
    the full ``head_dim`` and ignores ``partial_rotary_factor``), this module
    computes ``rotary_dim = head_dim * partial_rotary_factor`` inverse
    frequencies and emits width-``rotary_dim`` cos/sin tables, matching the
    target's rotation of the shared K prefix. With factor 1.0 it reproduces
    the stock full-head behavior. Only ``rope_type='default'`` is supported;
    scaled rope types are rejected rather than silently mistreated.
    """

    def __init__(self, config: PretrainedConfig, device=None) -> None:
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        head_dim = _config_head_dim(config)
        rope_params = _config_rope_parameters(config)
        rope_type = rope_params.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                "mamba_attn_hybrid supports only rope_type='default'; got "
                f"'{rope_type}'."
            )
        base = float(rope_params.get("rope_theta", 10000.0))
        rotary_dim = _rope_dim_from_params(head_dim, rope_params)
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim={rotary_dim} must be positive, even, and no "
                f"larger than head_dim={head_dim}."
            )
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = base
        self.rope_type = rope_type
        self.attention_scaling = 1.0

        # Frequencies over the rotary *prefix* only: both the arange range and
        # the denominator use rotary_dim (never truncate a full-head table,
        # whose denominator would still be head_dim).
        inv_freq, _ = self.compute_default_rope_parameters(config, device=device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        # Keep the retained update state consistent with inv_freq.
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: PretrainedConfig,
        device=None,
        **_kwargs,
    ) -> tuple[torch.Tensor, float]:
        """Partial-prefix inverse frequencies.

        Also serves as the HF ``_init_weights`` hook: transformers'
        ``PreTrainedModel._init_weights`` re-initializes any module whose
        class name contains "RotaryEmbedding" (and that holds an
        ``original_inv_freq`` buffer) by calling this method and copying the
        result into ``inv_freq``/``original_inv_freq`` — so it must compute
        the same partial-prefix frequencies as ``__init__``.
        """
        head_dim = _config_head_dim(config)
        rope_params = _config_rope_parameters(config)
        base = float(rope_params.get("rope_theta", 10000.0))
        rotary_dim = _rope_dim_from_params(head_dim, rope_params)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                / rotary_dim
            )
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
