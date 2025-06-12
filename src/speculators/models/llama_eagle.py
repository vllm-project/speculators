"""
LlamaEagle model implementation for EAGLE1 and HASS speculator models.

This module provides a unified implementation for both EAGLE1 and HASS variants
through configurable parameters. The key differences between the variants are:
- EAGLE1:  no bias in fusion layer, standard layernorms
- HASS: bias in fusion layer, extra layernorms

Classes:
    LlamaEagleSpeculatorConfig: Configuration for EAGLE/HASS models
    LlamaEagleSpeculator: Model implementation for EAGLE/HASS
"""

from typing import Literal, Optional

import torch
from torch import nn
from pydantic import Field
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from speculators.config import SpeculatorModelConfig
from speculators.models.transformer import TransformerSpeculatorConfig

__all__ = ["LlamaEagleSpeculatorConfig", "LlamaEagleSpeculator"]


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

    # Variant selection
    eagle_variant: Literal["eagle", "eagle2", "hass"] = Field(
        default="eagle",
        description="The variant of EAGLE architecture to use. 'eagle_v1' for EAGLE1, 'hass' for HASS."
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

    # First layer normalization handling
    replace_first_layer_norm: bool = Field(
        default=True,
        description=(
            "Whether to replace the first decoder layer's input_layernorm with Identity. "
        )
    )

    # Additional HASS-specific parameters
    extra_layernorm_positions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Positions where extra layernorms should be added for HASS variant. "
            "E.g., ['post_embedding', 'pre_lm_head']. Only used if use_extra_layernorms is True."
        )
    )

    # Number of decoder layers
    num_hidden_layers: int = Field(
        default=1,
        description="Number of hidden layers in the speculator.",
        ge=1,
    )

    # Llama configuration for decoder layers
    llama_decoder_layer_config: Optional[LlamaConfig] = Field(
        default=None,
        description=(
            "LlamaConfig instance for creating decoder layers. "
            "If not provided, a default LlamaConfig will be created. "
            "This should include all necessary parameters like hidden_size, "
            "num_attention_heads, mlp_bias, etc."
        )
    )

    @property
    def vocab_size(self) -> int:
        """Get vocab_size from llama_decoder_layer_config."""
        if self.llama_decoder_layer_config:
            return self.llama_decoder_layer_config.vocab_size
        return 128256  # Default vocab size

    @property
    def hidden_size(self) -> int:
        """Get hidden_size from llama_decoder_layer_config."""
        if self.llama_decoder_layer_config:
            return self.llama_decoder_layer_config.hidden_size
        return 4096  # Default hidden size

    def __init__(self, **data):
        # Set defaults based on variant if not explicitly provided
        if "eagle_variant" in data:
            variant = data["eagle_variant"]

            # Set variant-specific defaults if not explicitly provided
            if "transformer_input_type" not in data:
                if variant in ["eagle", "eagle2"]:
                    data["transformer_input_type"] = "linear_no_bias"
                elif variant == "hass":
                    data["transformer_input_type"] = "linear_with_bias"

            if "fc_bias" not in data:
                data["fc_bias"] = (variant == "hass")

            if "use_extra_layernorms" not in data:
                data["use_extra_layernorms"] = (variant == "hass")

            if "extra_layernorm_positions" not in data and variant == "hass":
                # Default positions for HASS extra layernorms
                data["extra_layernorm_positions"] = ["post_embedding", "pre_lm_head"]

        super().__init__(**data)

    def to_dict(self) -> dict:
        """
        Override to properly serialize LlamaConfig to dict.
        """
        # Get the base dict from parent classes
        output = super().to_dict()

        # Convert LlamaConfig to dict if present
        if self.llama_decoder_layer_config is not None:
            output["llama_decoder_layer_config"] = self.llama_decoder_layer_config.to_dict()

        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Override to properly deserialize LlamaConfig from dict.
        """
        # Make a copy to avoid modifying the original
        config_dict = config_dict.copy()

        # Convert dict back to LlamaConfig if present
        if "llama_decoder_layer_config" in config_dict and isinstance(config_dict["llama_decoder_layer_config"], dict):
            config_dict["llama_decoder_layer_config"] = LlamaConfig(**config_dict["llama_decoder_layer_config"])

        # Remove None values for fields that have defaults
        # This allows the defaults to be applied properly
        fields_with_defaults = ["architectures", "torch_dtype"]
        for field in fields_with_defaults:
            if field in config_dict and config_dict[field] is None:
                del config_dict[field]

        # Call parent from_dict
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Override to use PretrainedConfig's from_pretrained and convert to our class.
        """
        # The parent from_pretrained may return a tuple (config, unused_kwargs)
        # when there are extra kwargs
        result = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if isinstance(result, tuple):
            config, unused_kwargs = result
            config_dict = config.to_dict()
            # Use our from_dict to properly handle LlamaConfig
            return cls.from_dict(config_dict, **unused_kwargs), unused_kwargs
        else:
            config_dict = result.to_dict()
            # Use our from_dict to properly handle LlamaConfig
            return cls.from_dict(config_dict, **kwargs)



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

        # Create or use provided LlamaConfig for decoder layers
        if config.llama_decoder_layer_config is None:
            # Create a default LlamaConfig with minimal required parameters
            self.llama_decoder_layer_config = LlamaConfig(
                hidden_size=4096,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=32,
                num_key_value_heads=8,
                intermediate_size=14336,
                vocab_size=128256,
                max_position_embeddings=131072,
                rms_norm_eps=1e-5,
                rope_theta=500000.0,
                attention_bias=False,
                attention_dropout=0.0,
                hidden_act="silu",
                mlp_bias=False,
            )
        else:
            self.llama_decoder_layer_config = config.llama_decoder_layer_config

        self.padding_idx = self.llama_decoder_layer_config.pad_token_id
        self.vocab_size = self.llama_decoder_layer_config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.llama_decoder_layer_config.hidden_size,
            padding_idx=self.padding_idx
        )
        # Embeddings + hidden state
        fc_input_dim = self.llama_decoder_layer_config.hidden_size * 2

        # Fusion layer
        self.fc = nn.Linear(
            fc_input_dim,
            self.llama_decoder_layer_config.hidden_size,
            bias=config.fc_bias
        )

        # Extra layernorms
        if config.use_extra_layernorms and config.extra_layernorm_positions:
            self.extra_layernorms = nn.ModuleDict()
            for position in config.extra_layernorm_positions:
                self.extra_layernorms[position] = LlamaRMSNorm(
                    self.llama_decoder_layer_config.hidden_size,
                    eps=self.llama_decoder_layer_config.rms_norm_eps
                )

        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(self.llama_decoder_layer_config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Replace first layer's input_layernorm with Identity if configured
        if config.replace_first_layer_norm and len(self.layers) > 0:
            self.layers[0].input_layernorm = nn.Identity()

        # Final layer norm
        # commented out because it is missing in
        # yuhuili/EAGLE-LLaMA3.1-Instruct-8B
        # self.norm = LlamaRMSNorm(self.llama_decoder_layer_config.hidden_size, eps=self.llama_decoder_layer_config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=self.llama_decoder_layer_config)

        # LM Head
        self.lm_head = nn.Linear(
            self.llama_decoder_layer_config.hidden_size,
            self.vocab_size,
            bias=False
        )

        # Initialize weights
        self.post_init()

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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        if hasattr(self, "extra_layernorms") and "post_embedding" in self.extra_layernorms:
            inputs_embeds = self.extra_layernorms["post_embedding"](inputs_embeds)

        # Fusion: concatenate embeddings and hidden states, then project
        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

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

        # Final layer norm
        # See comment above about missing norm in EAGLE
        # hidden_states = self.norm(hidden_states)

        # Apply extra layernorm before LM head if configured (HASS)
        if hasattr(self, "extra_layernorms") and "pre_lm_head" in self.extra_layernorms:
            hidden_states = self.extra_layernorms["pre_lm_head"](hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        return logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        # May need to expand based on requirements
        model_inputs = {
            "input_ids": input_ids,
            "hidden_states": kwargs.get("hidden_states"),
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }
        return model_inputs
