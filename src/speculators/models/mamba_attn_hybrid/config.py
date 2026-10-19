from typing import Literal

from pydantic import Field, field_validator, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.mamba_attn_hybrid.base_config import MambaAttnHybridBaseConfig

__all__ = [
    "MambaAttnHybridSpeculatorConfig",
    "parse_block_pattern",
]

_VALID_BLOCKS = ("mamba", "attention", "mlp")


def parse_block_pattern(pattern: "str | list[str]") -> list[str]:
    raw = pattern if isinstance(pattern, str) else " ".join(str(t) for t in pattern)
    tokens = [t.lower() for t in raw.split()]

    if not tokens:
        raise ValueError(
            "block_pattern is empty; expected tokens from {mamba, attention, mlp}."
        )
    bad = [t for t in tokens if t not in _VALID_BLOCKS]
    if bad:
        raise ValueError(
            f"Invalid block_pattern token(s) {bad}; allowed tokens are {list(_VALID_BLOCKS)}."
        )
    if tokens[-1] != "mlp":
        raise ValueError(
            "block_pattern must end with 'mlp' (each mlp marks the end of one layer); "
            f"got last token {tokens[-1]!r}."
        )
    if "attention" not in tokens:
        raise ValueError(
            "block_pattern has no 'attention' sub-layer; mamba_attn_hybrid needs "
            "at least one."
        )
    if "mamba" not in tokens:
        raise ValueError(
            "block_pattern has no 'mamba' sub-layer; mamba_attn_hybrid needs "
            "at least one."
        )
    return tokens


@SpeculatorModelConfig.register("mamba_attn_hybrid")
class MambaAttnHybridSpeculatorConfig(MambaAttnHybridBaseConfig):
    speculators_model_type: Literal["mamba_attn_hybrid"] = "mamba_attn_hybrid"
    architectures: list[str] = Field(
        default_factory=lambda: ["MambaAttnHybridSpeculator"],
        description="Model architectures that can load these weights",
    )

    mamba_d_state: int = Field(default=16, description="SSM state size N.")
    mamba_num_heads: int = Field(default=64, description="Number of SSD heads.")
    mamba_head_dim: int = Field(default=64, description="SSD head dimension.")
    mamba_n_groups: int = Field(default=1, description="Number of B/C groups.")
    mamba_conv_kernel: int = Field(default=4, description="Depthwise conv width.")
    mamba_expand: int = Field(default=1, description="Inner-dim expansion factor.")

    latent_fusion_layer_ids: list[int] | None = Field(
        default=None,
        description="Verifier hidden-state layer IDs fused into the latent z "
        "(F = len(latent_fusion_layer_ids)). Must be set at construction.",
    )

    mamba_seed_mode: Literal["per_layer", "shared", "once"] = Field(
        default="per_layer",
        description="How the verifier latent z seeds the SSM initial state across mamba "
        "sub-layers: 'per_layer' = a distinct seed projection g_i per mamba (default, "
        "heaviest); 'shared' = one shared projection reused at every mamba; "
        "'once' = seed only the first mamba (later mambas start from a zero SSM state).",
    )

    disable_mamba_init_state_injection: bool = Field(
        default=False,
        description="Ablation: do not inject the projected verifier latent as the SSM "
        "initial state. Every mamba sub-layer starts from a ZERO recurrent state, and "
        "the fusion fc / hidden_norm / fc_norm / seed projections are not built (so "
        "they carry no weights). mamba_seed_mode is then irrelevant. The verifier "
        "hidden states are still loaded and used for the loss targets only.",
    )
    disable_target_kv_reuse: bool = Field(
        default=False,
        description="Ablation: the attention sub-layers never read the verifier KV "
        "cache. Each block position attends ONLY within its own block (anchor + mask "
        "slots), with the same intra-block causality rule as before (causal iff the "
        "sub-layer is windowed and not sliding_window_non_causal). attn_kv_layer_ids "
        "is still required: it drives the data plumbing and the q/k/v warm-start, but "
        "the KV is never read.",
    )

    block_pattern: list[str] | None = Field(
        default=None,
        description="Flat list of residual sub-layer tokens from {mamba, attention, "
        "mlp}; each mlp marks the end of one layer. Must be set at construction.",
    )

    attn_kv_layer_ids: list[int] | None = Field(
        default=None,
        description="Verifier KV layer ID each attention sub-layer reads, in "
        "attention-block order — the sole verifier-KV source for the hybrid. The data's "
        "KV set is sorted(set(...)) (== aux_hidden_state_layer_ids). Length must equal "
        "the number of 'attention' tokens in block_pattern.",
    )

    attn_rope: list[dict] | None = Field(
        default=None,
        description="Per-attention-sub-layer resolved verifier RoPE + sliding-window "
        "(rope_parameters, sliding_window, layer_type), in attention-block order. Resolved "
        "at save time from the verifier config; consumed unchanged at inference. Length must "
        "equal the number of 'attention' tokens in block_pattern (== len(attn_kv_layer_ids)).",
    )

    use_qk_norm: bool = Field(
        default=True,
        description="Apply per-head RMSNorm (q_norm/k_norm) in the draft attention "
        "sub-layers. An EXPLICIT training choice (--use-qk-norm / --no-use-qk-norm; "
        "default on) — no verifier auto-detection. The norms only ever act on "
        "draft-computed Q / noise-K, never on the injected verifier context KV. With "
        "a normless verifier (llama) and norms ON, they start at ones and are forced "
        "trainable (no verifier weights to copy).",
    )

    verifier_layer_prefix: str = Field(
        default="model.layers",
        description="Prefix of the verifier's per-layer attention weights used for warm-start "
        "(e.g. 'model.layers' for qwen3, 'model.language_model.layers' for multimodal gemma4).",
    )

    markov_rank: int = Field(
        default=0,
        ge=0,
        description="Low-rank dim of the DSpark Markov (correction) logit-bias head "
        "B = W1 @ W2. 0 disables the head (pure hybrid drafting).",
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla",
        description="DSpark correction-head variant: 'vanilla' (first-order Markov bias), "
        "'gated' (hidden-gated bias), or 'rnn' (recurrent state over the block). Only used "
        "when markov_rank > 0.",
    )

    enable_confidence_head: bool = Field(
        default=False,
        description="Attach the DSpark per-position acceptance-probability (confidence) "
        "head. Disabled by default → behavior unchanged when the flag is absent.",
    )
    confidence_head_with_markov: bool = Field(
        default=False,
        description="Concatenate the Markov previous-token embedding with the backbone "
        "hidden state as the confidence-head input. Requires markov_rank > 0.",
    )

    fc_norm: bool = Field(
        default=False,
        description="Apply a per-fusion-layer RMSNorm to each of the F verifier hidden "
        "states (chunked from the F*hidden concat) before the fc fusion projection, "
        "EAGLE-3 fc_norm style. Default off = unchanged.",
    )

    @model_validator(mode="after")
    def _validate_previous_token_heads(self):
        if self.confidence_head_with_markov and self.markov_rank == 0:
            raise ValueError(
                "confidence_head_with_markov=True requires markov_rank > 0."
            )
        return self

    @field_validator("block_pattern", mode="before")
    @classmethod
    def _normalize_block_pattern(cls, value: "str | list[str] | None"):
        if value is None:
            return None
        return parse_block_pattern(value)
