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

    enable_lilicorr: bool = Field(
        default=False,
        description="Attach the LiLiCorr candidate-lattice selector head "
        "(models/lilicorr/model_definitions.LiLiCorrHead) as ``self.lilicorr`` and "
        "add its objective to the loss. When False (default) the head is never "
        "instantiated and behavior is unchanged. When True requires: full verifier "
        "vocabulary (draft_vocab_size == verifier vocab), markov_rank=0, "
        "enable_confidence_head=False, and the latent-seed injection enabled "
        "(disable_mamba_init_state_injection=False) so the head's anchor context z "
        "exists. Head geometry comes from the lilicorr_* fields (defaults: K=8, "
        "head width 1024, factor width 1024, 2 layers, 8 heads, mlp_ratio 2, "
        "logit_scale 8, vector_eps 1e-4).",
    )
    lilicorr_candidate_topk: int = Field(
        default=8, description="LiLiCorr head: candidates kept per slot."
    )
    lilicorr_hidden_size: int | None = Field(
        default=1024,
        description="LiLiCorr head width; None uses the draft hidden size.",
    )
    lilicorr_factor_dim: int = Field(
        default=1024, description="LiLiCorr head: width of the out/in factor vectors."
    )
    lilicorr_num_layers: int = Field(
        default=2, description="LiLiCorr head: lattice layers."
    )
    lilicorr_num_heads: int = Field(
        default=8, description="LiLiCorr head: lattice attention heads."
    )
    lilicorr_mlp_ratio: float = Field(
        default=2.0, description="LiLiCorr head: lattice MLP ratio."
    )
    lilicorr_logit_scale: float = Field(
        default=8.0, description="LiLiCorr head: fixed scale on the cosine factors."
    )
    lilicorr_vector_eps: float = Field(
        default=1e-4, description="LiLiCorr head: epsilon for L2 normalization."
    )
    lilicorr_w_ce: float = Field(
        default=0.25,
        description="LiLiCorr head: weight of the lattice cross-entropy term in "
        "the combined loss (paper default 0.25).",
    )
    lilicorr_w_pen: float = Field(
        default=0.25,
        description="LiLiCorr head: weight of the distractor-penalty term in "
        "the combined loss (paper default 0.25).",
    )

    @model_validator(mode="after")
    def _validate_previous_token_heads(self):
        if self.confidence_head_with_markov and self.markov_rank == 0:
            raise ValueError(
                "confidence_head_with_markov=True requires markov_rank > 0."
            )
        return self

    @model_validator(mode="after")
    def _validate_lilicorr_combination(self):
        if not self.enable_lilicorr:
            return self
        verifier_vocab = getattr(self.transformer_layer_config, "vocab_size", None)
        if verifier_vocab is not None and self.draft_vocab_size != verifier_vocab:
            raise ValueError(
                "enable_lilicorr=True requires the full verifier vocabulary "
                f"(draft_vocab_size == {verifier_vocab}); got "
                f"draft_vocab_size={self.draft_vocab_size}. The LiLiCorr lattice "
                "gathers top-K candidates from the draft vocabulary logits and "
                "vLLM cannot serve a reduced draft vocab with the head."
            )
        if self.markov_rank != 0:
            raise ValueError(
                "enable_lilicorr=True requires markov_rank=0 (the LiLiCorr head "
                "replaces the Markov correction path); "
                f"got markov_rank={self.markov_rank}."
            )
        if self.enable_confidence_head:
            raise ValueError(
                "enable_lilicorr=True requires enable_confidence_head=False "
                "(the confidence head is a standalone-HSpec option)."
            )
        if self.disable_mamba_init_state_injection:
            raise ValueError(
                "enable_lilicorr=True requires the latent-seed injection enabled "
                "(disable_mamba_init_state_injection=False): the head's anchor "
                "context is HSpec's fused latent z."
            )
        return self

    @field_validator("block_pattern", mode="before")
    @classmethod
    def _normalize_block_pattern(cls, value: "str | list[str] | None"):
        if value is None:
            return None
        return parse_block_pattern(value)
