from typing import Literal

from pydantic import Field
from transformers import AutoConfig, PretrainedConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig

__all__ = [
    "GLM5_DSPARK_MODEL_TYPE",
    "DFlashSpeculatorConfig",
    "Glm5DSparkDecoderConfig",
]

GLM5_DSPARK_MODEL_TYPE = "glm5_dspark"

# DeepseekV3Config defaults describe a 256-expert MoE with an MTP head. The draft
# is a dense-MLP MLA stack, so these defaults are rewritten to "no MoE, no MTP"
# before they are serialized into ``transformer_layer_config``. Loaders that key
# on them (``n_routed_experts``, ``first_k_dense_replace``) then build dense layers.
_DENSE_DRAFT_DEFAULTS: dict[str, object] = {
    "n_routed_experts": 0,
    "n_shared_experts": 0,
    "num_experts_per_tok": 0,
    "moe_intermediate_size": 0,
    "n_group": None,
    "topk_group": None,
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "num_mtp_layers": 0,
}


class Glm5DSparkDecoderConfig(DeepseekV3Config):
    """DeepSeek-V3 MLA decoder config under the GLM-5 DSpark draft ``model_type``.

    GLM-5.x attention is DeepSeek-style dense MLA, so the upstream config already
    carries every field the draft needs (LoRA ranks, nope/rope/v head dims,
    ``rope_interleave``, YaRN parameters). This subclass gives the draft its own
    ``model_type`` (the verifier is ``glm_moe_dsa``; vLLM and vLLM-Ascend dispatch
    the draft independently of it) and replaces the upstream MoE/MTP defaults with
    dense-only values. Attention dimensions are copied from the verifier when the
    draft is built (``mla_kwargs_from_verifier``), not taken from defaults.
    """

    model_type = GLM5_DSPARK_MODEL_TYPE

    def __init__(self, **kwargs):
        for key, value in _DENSE_DRAFT_DEFAULTS.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)
        if "first_k_dense_replace" not in kwargs:
            # Every draft layer is dense: no layer index reaches the MoE range.
            self.first_k_dense_replace = self.num_hidden_layers


# Lets AutoConfig.for_model("glm5_dspark") (shared transformer_layer_config
# validator, --draft-config) construct this class.
AutoConfig.register(GLM5_DSPARK_MODEL_TYPE, Glm5DSparkDecoderConfig, exist_ok=True)


@SpeculatorModelConfig.register("dflash")
class DFlashSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for DFlash speculator with vocabulary mapping.

    DFlash features vocabulary mapping between draft (64K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    """

    speculators_model_type: Literal["dflash"] = "dflash"
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlashSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    block_size: int = Field(
        default=8,
        description=(
            "Default size of the draft block predicted with a forward pass of the model"
        ),
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the DFlash auxiliary hidden state layers",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for masking",
    )

    sliding_window_non_causal: bool = Field(
        default=False,
        description="Use non-causal (bidirectional) masking in sliding-window layers. "
        "Full-attention layers remain bidirectional by default. "
        "Overridden by transformer_layer_config.is_causal when set.",
    )

    sample_from_anchor: bool = Field(
        default=False,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
        ),
    )

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size
