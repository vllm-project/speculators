"""
LlamaEagle model implementation for EAGLE1 and HASS speculator models.

This module provides a unified implementation for both EAGLE1 and HASS variants
through configurable parameters. The key differences between the variants are:
- EAGLE1:  no bias in fusion layer, standard layernorms
- HASS: bias in fusion layer, extra layernorms

Classes:
    LlamaDecoderParameters: Parameters for Llama decoder components
    LlamaEagleSpeculatorConfig: Configuration for EAGLE/HASS models
    LlamaEagleSpeculator: Model implementation for EAGLE/HASS
"""

from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field, field_validator
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from speculators.config import SpeculatorModelConfig
from speculators.models.transformer import TransformerSpeculatorConfig

__all__ = [
    "LlamaDecoderParameters",
    "LlamaEagleSpeculator",
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

    def to_llama_config(self, num_hidden_layers: int = 1) -> LlamaConfig:
        """Convert to a LlamaConfig object for use with transformers components."""
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            mlp_bias=self.mlp_bias,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
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

    def _create_llama_config(self) -> LlamaConfig:
        """Create a LlamaConfig from the decoder parameters."""
        # Handle case where llama_decoder_params might be loaded as dict
        if isinstance(self.llama_decoder_params, dict):
            params = LlamaDecoderParameters(**self.llama_decoder_params)
            return params.to_llama_config(self.num_hidden_layers)
        return self.llama_decoder_params.to_llama_config(self.num_hidden_layers)




class LlamaEagleSpeculator(PreTrainedModel, GenerationMixin):
    """
    EAGLE/HASS speculator model implementation.

    This model implements both EAGLE1 and HASS variants through configuration.
    It uses a fusion mechanism that combines input embeddings with hidden states
    from the verifier model to generate draft tokens for speculative decoding.
    """

    config_class = LlamaEagleSpeculatorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config: LlamaEagleSpeculatorConfig):
        super().__init__(config)
        self.config = config


        # Create LlamaConfig from explicit parameters
        self.llama_decoder_layer_config = config._create_llama_config()  # noqa: SLF001

        # Handle case where llama_decoder_params might be loaded as dict
        if isinstance(config.llama_decoder_params, dict):
            decoder_params = LlamaDecoderParameters(**config.llama_decoder_params)
            config.llama_decoder_params = decoder_params
        
        self.padding_idx = config.llama_decoder_params.pad_token_id
        self.vocab_size = config.llama_decoder_params.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            config.llama_decoder_params.hidden_size,
            padding_idx=self.padding_idx
        )
        # Embeddings + hidden state
        fc_input_dim = config.llama_decoder_params.hidden_size * 2

        # Fusion layer
        self.fc = nn.Linear(
            fc_input_dim,
            config.llama_decoder_params.hidden_size,
            bias=config.fc_bias
        )

        # Extra layernorms
        if config.use_extra_layernorms and config.extra_layernorm_positions:
            self.extra_layernorms = nn.ModuleDict()
            for position in config.extra_layernorm_positions:
                self.extra_layernorms[position] = LlamaRMSNorm(
                    config.llama_decoder_params.hidden_size,
                    eps=config.llama_decoder_params.rms_norm_eps
                )

        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(self.llama_decoder_layer_config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Replace first layer's input_layernorm with Identity if configured
        if config.replace_first_layer_norm and len(self.layers) > 0:
            self.layers[0].input_layernorm = nn.Identity()

        # Final layer norm - not used in EAGLE models
        # To add a final layer norm, use extra_layernorms with "pre_lm_head" position
        # Example: extra_layernorm_positions=["pre_lm_head"]

        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=self.llama_decoder_layer_config)

        # LM Head
        self.lm_head = nn.Linear(
            config.llama_decoder_params.hidden_size,
            self.vocab_size,
            bias=False
        )

        # Initialize weights
        self.post_init()  # type: ignore[attr-defined]

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # noqa: ARG002
        return_dict: Optional[bool] = None,  # noqa: ARG002
    ):
        """
        Forward pass for EAGLE/HASS speculator.

        Args:
            input_ids: Input token IDs from the verifier
            hidden_states: Hidden states from the verifier model
            attention_mask: Attention mask for the input
            position_ids: Position IDs for rotary embeddings
            past_key_values: Past key values for caching
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object

        Returns:
            logits: Predicted token logits
        """
        batch_size, seq_length = input_ids.shape

        # Get input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Apply extra layernorm after embedding if configured (HASS)
        if (
            hasattr(self, "extra_layernorms")
            and "post_embedding" in self.extra_layernorms
        ):
            inputs_embeds = self.extra_layernorms["post_embedding"](inputs_embeds)

        # Fusion: concatenate embeddings and hidden states, then project
        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

        # Create position ids if not provided
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask for decoder layers
        # The LlamaDecoderLayer expects 4D attention masks in newer transformers
        # This utility handles the conversion from 2D to 4D and adds causal masking
        if attention_mask is not None and attention_mask.dim() == 2:  # noqa: PLR2004
            past_key_values_length = (
                past_key_values[0][0].shape[2] if past_key_values else 0
            )
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=getattr(self.config, "sliding_window", None)
            )

        # Get rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Pass through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=(cos, sin),
            )
            hidden_states = layer_outputs[0]

        # Final layer norm is handled via extra_layernorms["pre_lm_head"] below

        # Apply extra layernorm before LM head if configured (HASS)
        if (
            hasattr(self, "extra_layernorms")
            and "pre_lm_head" in self.extra_layernorms
        ):
            hidden_states = self.extra_layernorms["pre_lm_head"](hidden_states)

        # Get logits
        return self.lm_head(hidden_states)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,  # noqa: ARG002
        **kwargs
    ):
        """Prepare inputs for generation."""
        # May need to expand based on requirements
        return {
            "input_ids": input_ids,
            "hidden_states": kwargs.get("hidden_states"),
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }
