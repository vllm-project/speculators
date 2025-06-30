"""
Eagle model implementation for EAGLE1 and HASS speculator models.

This module provides a unified implementation for both EAGLE1 and HASS variants
through configurable parameters.

Classes:
    EagleSpeculatorConfig: Configuration for EAGLE/HASS models
    EagleSpeculator: Model implementation for EAGLE/HASS speculators
"""

from typing import Any, Literal, Optional, Union

import torch
from pydantic import Field, field_serializer, field_validator, model_validator
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from typing_extensions import Self

from speculators.config import (
    SpeculatorModelConfig,
)

__all__ = [
    "EagleSpeculator",
    "EagleSpeculatorConfig",
]


@SpeculatorModelConfig.register("eagle")
class EagleSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration class for EAGLE1 and HASS speculator models.

    This unified configuration supports both EAGLE1 and HASS variants through
    configurable parameters, allowing a single model implementation to handle
    both architectures.
    """

    speculators_model_type: Literal["eagle"] = "eagle"
    architectures: list[str] = Field(
        default_factory=lambda: ["EagleSpeculator"],
        description=(
            "List of model architectures that can be used with the "
            "model pretrained weights."
        ),
    )

    def __init__(self, **kwargs):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["EagleSpeculator"]
        super().__init__(**kwargs)

    transformer_layer_architecture: str = Field(
        default="LlamaDecoderLayer",
        description=(
            "The architecture of the transformer layer to use. "
            "Typically 'LlamaDecoderLayer' for Eagle 1, Eagle 2, and HASS."
        ),
    )
    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description=(
            "Configuration for the transformer layer to use in the "
            "Eagle model architecture. This must be a PretrainedConfig that matches "
            "the config required by the transformer_layer_architecture."
        ),
    )
    layernorms: bool = Field(
        default=False,
        description=(
            "Whether to use additional layernorms in the model architecture, "
            "specifically the layernorm after the verifier's hidden state, "
            "after the fusion layer, and before the LM head. "
            "For Eagle, Eagle 1, and HASS, this is False."
        ),
    )
    fusion_bias: bool = Field(
        default=False,
        description=(
            "Whether to add a bias to the fusion (fc) layer that is applied to the "
            "concat of the input embeddings and input hidden state. "
            "For Eagle and Eagle 2, this is False, while for HASS it is True."
        ),
    )

    @model_validator(mode="after")
    def check_add_architectures(self) -> Self:
        """
        Ensure that the transformer_layer_architecture is included in the
        architectures list.

        :return: Self
        """
        if self.transformer_layer_architecture not in self.architectures:
            self.architectures.append(self.transformer_layer_architecture)

        return self

    @field_serializer("transformer_layer_config")
    def serialize_transformer_layer_config(self, value: PretrainedConfig) -> dict:
        """
        Serialize the transformer_layer_config to a dictionary.

        :param value: The PretrainedConfig instance to serialize.
        :return: Serialized dictionary representation of the config.
        """
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_layer_config(cls, value: Any) -> PretrainedConfig:
        """
        Validate that the transformer_layer_config is a valid PretrainedConfig.

        :param value: The instance to validate to a PretrainedConfig.
        :return: The validated PretrainedConfig instance.
        """
        if isinstance(value, dict):
            return PretrainedConfig.from_dict(value)

        if isinstance(value, PretrainedConfig):
            return value

        raise ValueError(
            "transformer_layer_config must be a PretrainedConfig or a dict "
            "that can be converted to a PretrainedConfig."
        )


class EagleSpeculator(PreTrainedModel, GenerationMixin):
    """
    Eagle speculator model for speculative decoding.

    This model implements EAGLE1, EAGLE2, and HASS variants through configuration.
    The key differences between variants are:

    - EAGLE1/2: layernorms=False, fusion_bias=False (EAGLE2 same architecture)
    - HASS: layernorms=False, fusion_bias=False
    - TTT variant: layernorms=True, fusion_bias varies
    """

    config_class = EagleSpeculatorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config: EagleSpeculatorConfig):
        super().__init__(config)
        self.config = config

        self.vocab_size = config.transformer_layer_config.vocab_size
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.padding_idx = config.transformer_layer_config.pad_token_id

        self._init_embeddings()
        self._init_fusion_layer()
        self._init_decoder_layers()
        self._init_output_layer()

        if config.layernorms:
            self._init_extra_layernorms()

        # Tie weights between embed_tokens and lm_head
        self.lm_head.weight = self.embed_tokens.weight

        self.post_init()

    def _init_embeddings(self):
        """Initialize embedding layer and rotary embeddings."""
        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.hidden_size, self.padding_idx
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            config=self.config.transformer_layer_config
        )

    def _init_fusion_layer(self):
        """Initialize fusion layer that combines embeddings and hidden states."""
        self.fc = nn.Linear(
            2 * self.hidden_size,
            self.hidden_size,
            bias=self.config.fusion_bias,
        )

    def _init_decoder_layers(self):
        """Initialize single decoder layer."""
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.config.transformer_layer_config, layer_idx=0)]
        )

        # For models without layernorms, replace input_layernorm with Identity
        # to match the Eagle architecture
        if not self.config.layernorms:
            self.layers[0].input_layernorm = nn.Identity()

    def _init_output_layer(self):
        """Initialize output projection layer."""
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def _init_extra_layernorms(self):
        """Initialize extra layernorms for TTT variant."""
        eps = self.config.transformer_layer_config.rms_norm_eps

        self.post_embedding_layernorm = LlamaRMSNorm(self.hidden_size, eps=eps)
        self.pre_lm_head_layernorm = LlamaRMSNorm(self.hidden_size, eps=eps)

    @property
    def input_embeddings(self):
        """
        Get input embeddings layer.

        :return: Embedding layer
        """
        return self.embed_tokens

    @input_embeddings.setter
    def input_embeddings(self, value):
        """
        Set input embeddings layer.

        :param value: New embedding layer
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # noqa: ARG002
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithPast]:
        """
        Generate speculative token predictions using verifier's hidden states.

        :param input_ids: Input token IDs
        :param hidden_states: Hidden states from verifier model
        :param attention_mask: Attention mask
        :param position_ids: Position IDs
        :param past_key_values: Past key values for caching
        :param use_cache: Whether to use caching
        :param output_attentions: Whether to output attentions
        :param output_hidden_states: Whether to output hidden states
        :param return_dict: Whether to return a dict
        :return: Model outputs
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds = self.embed_tokens(input_ids)

        if hasattr(self, "post_embedding_layernorm"):
            inputs_embeds = self.post_embedding_layernorm(inputs_embeds)

        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

        hidden_states, attention_mask, position_ids = self._prepare_decoder_inputs(
            hidden_states, attention_mask, position_ids, past_key_values
        )

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        layer_outputs = self.layers[0](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[0] if past_key_values else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=(cos, sin),
        )

        hidden_states = layer_outputs[0]

        if hasattr(self, "pre_lm_head_layernorm"):
            hidden_states = self.pre_lm_head_layernorm(hidden_states)

        logits = self.lm_head(hidden_states)

        if not return_dict:
            return logits

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=layer_outputs[1] if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    def _prepare_decoder_inputs(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]],
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.LongTensor]:
        """
        Prepare inputs for the decoder layer.

        :param hidden_states: Hidden states
        :param attention_mask: Attention mask
        :param position_ids: Position IDs
        :param past_key_values: Past key values
        :return: Prepared hidden states, attention mask, and position IDs
        """
        batch_size, seq_length = hidden_states.shape[:2]

        if position_ids is None:
            device = hidden_states.device
            position_ids = (
                torch.arange(seq_length, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        if attention_mask is not None and attention_mask.dim() == 2:  # noqa: PLR2004
            past_key_values_length = (
                past_key_values[0][0].shape[2] if past_key_values else 0
            )
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=getattr(self.config, "sliding_window", None),
            )

        return hidden_states, attention_mask, position_ids

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # noqa: ARG002
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation step.

        :param input_ids: Input token IDs
        :param past_key_values: Past key values
        :param attention_mask: Attention mask
        :param inputs_embeds: Input embeddings (unused)
        :param kwargs: Additional keyword arguments
        :return: Model inputs dict
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError("Eagle speculator requires hidden_states from verifier")

        return {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load Eagle model from speculators-format checkpoint.

        For original checkpoints, use::

            speculators convert eagle <input_path> <output_path> <base_model>

        :param pretrained_model_name_or_path: Model name or path
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Model instance
        """
        config = kwargs.pop("config", None)
        if config is None:
            try:
                config = EagleSpeculatorConfig.from_pretrained(
                    pretrained_model_name_or_path
                )
                if (
                    not hasattr(config, "speculators_model_type")
                    or config.speculators_model_type != "eagle"
                ):
                    raise ValueError("Missing or incorrect speculators_model_type")
            except Exception as e:
                raise ValueError(
                    f"Failed to load EagleSpeculatorConfig from "
                    f"'{pretrained_model_name_or_path}'.\n"
                    "This checkpoint does not appear to be in speculators format.\n\n"
                    "To convert an original Eagle/HASS checkpoint, use:\n"
                    "  speculators convert eagle <input_path> <output_path> "
                    "--base-model <base_model>\n\n"
                    "For example:\n"
                    "  speculators convert eagle yuhuili/EAGLE-LLaMA3.1-Instruct-8B "
                    "./converted/eagle --base-model meta-llama/Llama-3.1-8B\n\n"
                    f"Original error: {str(e)}"
                ) from e

        if not isinstance(config, EagleSpeculatorConfig):
            raise ValueError(
                f"Expected EagleSpeculatorConfig but got {type(config).__name__}. "
                "Please ensure you're loading a properly converted Eagle checkpoint."
            )

        return super().from_pretrained(
            pretrained_model_name_or_path, *args, config=config, **kwargs
        )
