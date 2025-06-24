"""
Configuration for LlamaEagle speculator models.

This module provides configuration classes for both EAGLE1 and HASS variants.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from speculators.config import SpeculatorModelConfig
from speculators.models.transformer import TransformerSpeculatorConfig

__all__ = [
    "LlamaDecoderParameters",
    "LlamaEagleSpeculatorConfig",
]


class LlamaDecoderParameters(BaseModel):
    """
    Parameters for configuring Llama decoder layers and components.

    This class encapsulates a subset of LlamaConfig parameters that are
    specifically needed for EAGLE/HASS speculator models. It includes only
    the parameters required for decoder layers, embeddings, and output
    projections, avoiding the full complexity of LlamaConfig.

    Default values are based on Llama 3.1 8B models commonly used with EAGLE,
    such as yuhuili/EAGLE-LLaMA3.1-Instruct-8B.
    """

    vocab_size: int = Field(
        default=128256,
        description=(
            "Vocabulary size of the model. "
            "Defines the number of different tokens. "
            "Default: 128256 (Llama 3/3.1 vocabulary)."
        ),
    )

    hidden_size: int = Field(
        default=4096,
        description=(
            "Dimension of the hidden representations. "
            "Default: 4096 (Llama 3.1 8B)."
        ),
    )

    intermediate_size: int = Field(
        default=14336,
        description=(
            "Dimension of the MLP representations in the decoder layers. "
            "Default: 14336 (Llama 3.1 8B)."
        ),
    )

    num_attention_heads: int = Field(
        default=32,
        description=(
            "Number of attention heads for each attention layer. "
            "Default: 32 (Llama 3.1 8B)."
        ),
    )

    num_key_value_heads: int = Field(
        default=8,
        description=(
            "Number of key_value heads for Grouped Query Attention (GQA). "
            "Default: 8 (Llama 3.1 8B uses 4:1 GQA ratio)."
        ),
    )

    hidden_act: str = Field(
        default="silu",
        description=(
            "The non-linear activation function in the decoder. "
            "Default: 'silu' (SwiGLU activation used in Llama)."
        ),
    )

    max_position_embeddings: int = Field(
        default=131072,
        description=(
            "The maximum sequence length the model can handle. "
            "Default: 131072 (Llama 3.1 extended context)."
        ),
    )

    rms_norm_eps: float = Field(
        default=1e-5,
        description=(
            "The epsilon used by the RMS normalization layers. "
            "Default: 1e-5 (Llama standard)."
        ),
    )

    rope_theta: float = Field(
        default=500000.0,
        description=(
            "The base period of the RoPE embeddings. "
            "Default: 500000.0 (Llama 3.1 RoPE base)."
        ),
    )

    attention_bias: bool = Field(
        default=False,
        description=(
            "Whether to use bias in the attention layers. "
            "Default: False (Llama models don't use attention bias)."
        ),
    )

    attention_dropout: float = Field(
        default=0.0,
        description=(
            "The dropout ratio for the attention probabilities. "
            "Default: 0.0 (no dropout in inference)."
        ),
    )

    mlp_bias: bool = Field(
        default=False,
        description=(
            "Whether to use bias in MLP layers. "
            "Default: False (Llama models don't use MLP bias)."
        ),
    )

    pad_token_id: Optional[int] = Field(
        default=None,
        description=(
            "Padding token id. "
            "Default: None (often set to 0 or same as eos_token_id)."
        ),
    )

    bos_token_id: int = Field(
        default=128000,
        description=(
            "Beginning of stream token id. "
            "Default: 128000 (Llama 3/3.1 BOS token)."
        ),
    )

    eos_token_id: int = Field(
        default=128001,
        description=(
            "End of stream token id. "
            "Default: 128001 (Llama 3/3.1 primary EOS token)."
        ),
    )


@SpeculatorModelConfig.register("llama_eagle")
class LlamaEagleSpeculatorConfig(TransformerSpeculatorConfig):
    """
    Configuration class for EAGLE1 and HASS speculator models.

    This unified configuration supports both EAGLE1 and HASS variants through
    configurable parameters, allowing a single model implementation to handle
    both architectures.
    """

    # Model identification
    speculators_model_type: Literal["llama_eagle"] = "llama_eagle"

    architectures: list[str] = Field(
        default_factory=lambda: ["LlamaEagleSpeculator"],
        description="The architectures this speculator uses."
    )


    # Fusion layer configuration
    fc_bias: bool = Field(
        default=False,
        description=(
            "Whether to use bias in the fusion (fc) layer. "
            "EAGLE1 typically uses False, HASS uses True."
        )
    )

    # Extra layernorm configuration
    use_extra_layernorms: bool = Field(
        default=False,
        description=(
            "Whether to use additional layernorm layers. "
            "EAGLE1 uses False (standard Llama layernorms only), "
            "HASS uses True (includes extra layernorms)."
        )
    )


    # Override inputs default for EAGLE/HASS
    inputs: list[str] = Field(
        default_factory=lambda: ["input_ids", "hidden_states[-1]"],
        description=(
            "The inputs from the verifier that this speculator uses. "
            "EAGLE uses input_ids and the last hidden states from the verifier."
        )
    )

    # Layer normalization handling
    replace_first_layer_norm: bool = Field(
        default=True,
        description=(
            "Whether to replace the first decoder layer's input_layernorm "
            "with Identity."
        )
    )

    # Additional layernorm parameters
    extra_layernorm_positions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Positions where extra layernorms should be added. "
            "Options: 'post_embedding' (after input embeddings), "
            "'pre_lm_head' (before LM head, acts as final layer norm). "
            "E.g., ['post_embedding', 'pre_lm_head']. "
            "Only used if use_extra_layernorms is True."
        )
    )

    # Number of decoder layers
    num_hidden_layers: int = Field(
        default=1,
        description="Number of hidden layers in the speculator.",
        ge=1,
    )

    # Llama decoder parameters
    llama_decoder_params: LlamaDecoderParameters = Field(
        default_factory=LlamaDecoderParameters,
        description="Parameters for configuring Llama decoder layers and components.",
    )

    @field_validator('llama_decoder_params', mode='before')
    @classmethod
    def validate_llama_decoder_params(cls, v):
        """Convert dict to LlamaDecoderParameters if necessary."""
        if isinstance(v, dict):
            return LlamaDecoderParameters(**v)
        return v

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Override from_dict to handle llama_decoder_params conversion."""
        # Convert llama_decoder_params dict to LlamaDecoderParameters
        if 'llama_decoder_params' in config_dict and isinstance(config_dict['llama_decoder_params'], dict):
            config_dict = config_dict.copy()
            config_dict['llama_decoder_params'] = LlamaDecoderParameters(**config_dict['llama_decoder_params'])
        return super().from_dict(config_dict, **kwargs)