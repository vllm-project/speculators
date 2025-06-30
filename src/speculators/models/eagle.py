"""
Eagle model implementation for EAGLE1 and HASS speculator models.

This module provides a unified implementation for both EAGLE1 and HASS variants
through configurable parameters.

Classes:
    EagleSpeculatorConfig: Configuration for EAGLE/HASS models
    EagleSpeculator: Model implementation for EAGLE/HASS speculators
"""

from typing import Any, ClassVar, Literal, Optional, Union

import torch
from pydantic import Field, field_serializer, field_validator, model_validator
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
)
from typing_extensions import Self

from speculators import SpeculatorModel, SpeculatorModelConfig

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


@SpeculatorModel.register("eagle")
class EagleSpeculator(SpeculatorModel):
    """
    Eagle speculator model for speculative decoding.

    This model implements EAGLE1, EAGLE2, and HASS variants through configuration.
    The key differences between variants are:

    - EAGLE1/2: layernorms=False, fusion_bias=False (EAGLE2 same architecture)
    - HASS: layernorms=False, fusion_bias=False
    - TTT variant: layernorms=True, fusion_bias varies
    """

    # PreTrainedModel settings
    config_class: ClassVar[type[EagleSpeculatorConfig]] = EagleSpeculatorConfig

    def __init__(
        self, config: EagleSpeculatorConfig, verifier: Optional[PreTrainedModel] = None
    ):
        # Initialize model parameters from config
        self.vocab_size = config.transformer_layer_config.vocab_size
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.padding_idx = config.transformer_layer_config.pad_token_id

        # Set layers pulled from the verifier to None until attach is called
        self.embed_tokens: Optional[nn.Embedding] = None
        self.rotary_emb: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Linear] = None

        # Delayed initialization to ensure everything needed for attach_verifier is set
        super().__init__(config=config, verifier=verifier)

        # Initialize layers based on the configuration
        self.embedding_layernorm: Optional[nn.Module] = self._create_layernorm()
        self.fusion_fc: nn.Linear = nn.Linear(
            2 * self.hidden_size,
            self.hidden_size,
            bias=config.fusion_bias,
        )
        self.transformer: nn.Module = self._create_transformer_layer()
        self.pre_lm_head_layernorm: Optional[nn.Module] = self._create_layernorm()

        self.post_init()  # type: ignore[attr-defined]

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
        if self.embed_tokens is None or self.rotary_emb is None or self.lm_head is None:
            raise ValueError(
                "Verifier model layers not initialized. "
                "Call `attach_verifier` to set up the model before using forward."
            )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds = self.embed_tokens(input_ids)
        if self.embedding_layernorm is not None:
            inputs_embeds = self.embedding_layernorm(inputs_embeds)

        hidden_states = self.fusion_fc(
            torch.cat([inputs_embeds, hidden_states], dim=-1)
        )
        hidden_states, attention_mask, position_ids = self._prepare_decoder_inputs(
            hidden_states, attention_mask, position_ids, past_key_values
        )

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        layer_outputs = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[0] if past_key_values else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=(cos, sin),
        )
        hidden_states = layer_outputs[0]

        if self.pre_lm_head_layernorm is not None:
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
    ) -> tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[torch.LongTensor]]:
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
                torch.arange(seq_length, dtype=torch.long, device=device)  # type: ignore[assignment]
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

    def _create_layernorm(self) -> Optional[nn.Module]:
        if not self.config.layernorms:
            return None

        return self._layernorm_class()(
            self.hidden_size, eps=self.config.transformer_layer_config.rms_norm_eps
        )

    def _create_transformer_layer(self) -> nn.Module:
        layer_class = self._transformer_layer_class()
        layer = layer_class(
            self.config.transformer_layer_config,
            layer_idx=0,
        )

        if not self.config.layernorms:
            # Replace input_layernorm with Identity if layernorms are not used
            layer.input_layernorm = nn.Identity()

        return layer

    def _layernorm_class(self) -> type[nn.Module]:
        return LlamaRMSNorm

    def _transformer_layer_class(self) -> type[nn.Module]:
        return LlamaDecoderLayer
