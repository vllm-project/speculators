"""
Speculators implementations providing a unified implementation
for EAGLE v1, EAGLE v2, and HASS variants for spec decoding:
    - Eagle / Eagle v1: https://arxiv.org/abs/2401.15077
    - Eagle v2: https://arxiv.org/abs/2406.16858
    - HASS: https://arxiv.org/abs/2408.15766

Classes:
    EagleSpeculatorConfig: Configuration class for EAGLE/HASS model variants
    EagleSpeculator: Main model implementation for EAGLE/HASS speculators
"""

import os
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
    A SpeculatorModelConfig implementation to be used with the EagleSpeculator
    for EAGLE and HASS variants for spec decoding:
        - Eagle / Eagle v1: https://arxiv.org/abs/2401.15077
        - Eagle v2: https://arxiv.org/abs/2406.16858
        - HASS: https://arxiv.org/abs/2408.15766

    Model Configurations:
        - EAGLE1: layernorms=False, fusion_bias=False
        - EAGLE2: layernorms=False, fusion_bias=False
        - HASS: layernorms=False, fusion_bias=True

    Example:
        ```python
        from speculators import SpeculatorsConfig, VerifierConfig
        from speculators.models import EagleSpeculatorConfig
        from speculators.proposals import GreedyTokenProposalConfig
        from transformers import AutoConfig

        config = EagleSpeculatorConfig(
            transformer_layer_config=AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"),
            speculators_config=SpeculatorsConfig(
                algorithm="eagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(),
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                    architectures=["LlamaForCausalLM"],
                )
        )
        ```
    """

    speculators_model_type: Literal["eagle"] = "eagle"
    architectures: list[str] = Field(
        default_factory=lambda: ["EagleSpeculator"],
        description=(
            "List of model architectures that can be used with the model "
            "pretrained weights. Automatically includes the transformer layer "
            "architecture to ensure compatibility during model loading and "
            "validation."
        ),
    )

    transformer_layer_architecture: str = Field(
        default="LlamaDecoderLayer",
        description=(
            "The architecture class name of the transformer layer to use for "
            "the speculator's decoder layer. Must correspond to a valid "
            "transformer decoder layer class (e.g., 'LlamaDecoderLayer')."
        ),
    )
    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description=(
            "Configuration object for the transformer layer architecture. "
            "Must be a PretrainedConfig instance that matches the requirements "
            "of the transformer_layer_architecture. Contains parameters such as "
            "hidden_size, num_attention_heads, intermediate_size, vocab_size, "
            "and other architecture-specific settings."
        ),
    )
    layernorms: bool = Field(
        default=False,
        description=(
            "Whether to include additional layer normalization layers in the "
            "model architecture. When True, adds RMSNorm layers after the "
            "verifier's hidden state (embedding_layernorm), after the fusion "
            "layer output, and before the language model head (pre_lm_head_layernorm). "
            "When False, these layers are not included and the output layernorm "
            "within the transformer architecture is removed as well. "
            "Standard EAGLE1, EAGLE2, and HASS implementations use False."
        ),
    )
    fusion_bias: bool = Field(
        default=False,
        description=(
            "Whether to add a learnable bias term to the fusion (fully connected) "
            "layer that combines input embeddings with verifier hidden states. "
            "The fusion layer concatenates input embeddings and hidden states, "
            "then projects to hidden_size dimensions. Standard EAGLE1 and EAGLE2 "
            "use False, while HASS uses True."
        ),
    )

    @model_validator(mode="after")
    def check_add_architectures(self) -> Self:
        """
        Automatically adds the transformer layer architecture to the
        architectures list if it's not already present.

        :return: The validated configuration instance with updated architectures
        """
        if self.transformer_layer_architecture not in self.architectures:
            self.architectures.append(self.transformer_layer_architecture)

        return self

    @field_serializer("transformer_layer_config")
    def serialize_transformer_layer_config(self, value: PretrainedConfig) -> dict:
        """
        Serialize the transformer_layer_config to a dictionary for JSON storage.

        Converts the PretrainedConfig object to its dictionary representation
        using to_diff_dict() to only include non-default values.

        :param value: The PretrainedConfig instance to serialize
        :return: Dictionary representation of the transformer layer configuration
        """
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_layer_config(cls, value: Any) -> PretrainedConfig:
        """
        Validate and convert transformer_layer_config to a PretrainedConfig instance.

        Accepts either a dictionary that can be converted to a PretrainedConfig
        or an existing PretrainedConfig instance.

        :param value: The value to validate (dict or PretrainedConfig)
        :return: A validated PretrainedConfig instance
        :raises ValueError: If the value cannot be converted to a PretrainedConfig
        """
        if isinstance(value, dict):
            return PretrainedConfig.from_dict(value)

        if isinstance(value, PretrainedConfig):
            return value

        raise ValueError(
            "transformer_layer_config must be a PretrainedConfig instance or a "
            "dictionary that can be converted to a PretrainedConfig."
        )


@SpeculatorModel.register("eagle")
class EagleSpeculator(SpeculatorModel):
    """
    A SpeculatorModel implementation for EAGLE and HASS variants for spec decoding:
        - Eagle / Eagle v1: https://arxiv.org/abs/2401.15077
        - Eagle v2: https://arxiv.org/abs/2406.16858
        - HASS: https://arxiv.org/abs/2408.15766

    Architecture Overview:
        The EAGLE speculator consists of:
        1. Input embedding layer (shared with verifier)
        2. Optional embedding layer normalization
        3. Fusion layer: Concatenates and projects input embeddings + verifier hidden
           states to a latent space of hidden_size
        4. Single transformer decoder layer for candidate token generation
        5. Optional pre-LM head layer normalization
        6. Language model head (shared with verifier)

    Speculative Decoding Process:
        1. Verifier model processes input and generates hidden states
        2. EAGLE speculator uses these hidden states + input embeddings to predict
           next tokens
        3. Multiple candidate tokens generated in parallel using token proposal methods
        4. Verifier validates candidates and accepts/rejects based on probability
           thresholds
        5. Process continues iteratively for multi-token speculation

    Example:
        ```python
        from speculators import SpeculatorsConfig, VerifierConfig
        from speculators.models import EagleSpeculator, EagleSpeculatorConfig
        from speculators.proposals import GreedyTokenProposalConfig
        from transformers import AutoConfig, AutoTokenizer

        config = EagleSpeculatorConfig(
            transformer_layer_config=AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"),
            speculators_config=SpeculatorsConfig(
                algorithm="eagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(),
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                    architectures=["LlamaForCausalLM"],
                )
        )
        speculator = EagleSpeculator(
            config, verifier=verifier, verifier_attachment_mode="full"
        )

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        input_ids = tokenizer.encode("The future of AI is", return_tensors="pt")
        outputs = speculator.generate(
            input_ids, max_length=100, do_sample=True, temperature=0.7
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        ```
    """

    # PreTrainedModel settings
    config_class: ClassVar[type[EagleSpeculatorConfig]] = EagleSpeculatorConfig
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        "verifier*",
        "embed_tokens*",
        "lm_head*",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [
        "embed_tokens.weight",
        "lm_head.weight",
        "lm_head.bias",
    ]

    def __init__(
        self,
        config: EagleSpeculatorConfig,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[
            Literal["detached", "full", "train_only"]
        ] = None,
    ):
        """
        Initializes an EAGLE speculator architecture with configurable components based
        on the provided configuration. The model starts with verifier-dependent layers
        (embed_tokens, rotary_emb, lm_head) set to None until a verifier is attached.

        :param config: Configuration object specifying model architecture, layer
            settings, and speculative decoding parameters. Must be an instance of
            EagleSpeculatorConfig containing transformer layer configuration and
            EAGLE-specific settings.
        :param verifier: Optional verifier model to attach for speculative decoding.
            Can be a path to a model directory, Hugging Face model identifier, or
            PreTrainedModel instance. If None, must be attached later via
            attach_verifier() before using the model.
        :param verifier_attachment_mode: Mode for verifier attachment. "detached"
            prevents attachment even if verifier is provided. "full" enables
            complete integration for both training and generation. "train_only"
            attaches only components needed for training, optimizing memory usage.
        """
        if not isinstance(config, EagleSpeculatorConfig):
            raise ValueError(
                "config must be an instance of EagleSpeculatorConfig, "
                f"got {type(config)} instead."
            )

        # Initialize model parameters from config
        self.vocab_size = config.transformer_layer_config.vocab_size
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.padding_idx = config.transformer_layer_config.pad_token_id

        # Set layers pulled from the verifier to None until attach is called
        self.embed_tokens: Optional[nn.Embedding] = None
        self.rotary_emb: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Linear] = None

        # Delayed initialization to ensure everything needed for attach_verifier is set
        super().__init__(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
        )

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

    def attach_verifier(
        self,
        verifier: Union[str, os.PathLike, PreTrainedModel],
        mode: Optional[Literal["full", "train_only"]] = None,
    ) -> PreTrainedModel:
        """
        Attach a verifier model to the EagleSpeculator for speculative decoding.
        Utilizes the verifier's embed_tokens, rotary_emb, and lm_head layers
        for the speculator's forward pass and generation methods.
        Additionally, for `generate`, it uses the verifier's hidden states
        to generate speculative token predictions.

        If mode is "full", the verifier is fully integrated for use with
        both `generate` and `forward` methods.

        If mode is "train_only", only the verifier's layers required for a forward pass
        are attached, allowing for better resource utilization during training.
        `generate` will not be available until a full verifier is attached.

        Example:
            ```python
            # Load and attach a verifier
            verifier = EagleSpeculator(...)

            # For generation
            speculator.attach_verifier(verifier)
            outputs = speculator.generate(input_ids)
            speculator.detach_verifier()

            # For training
            speculator.attach_verifier(verifier, mode="train_only")
            outputs = speculator(input_ids, hidden_states)
            speculator.detach_verifier()
            ```

        :param verifier: The verifier model to attach. This can be a path to a local
            model directory, a Hugging Face model identifier, or an instance of
            PreTrainedModel. If a path or identifier is provided, the model will be
            loaded automatically. If an instance is provided, it will be used directly.
        :param mode: The mode for attaching the verifier. Can be "full" or "train_only".
            If None, defaults to "full". In "train_only" mode, only the layers
            required for a forward pass are attached, and the speculator cannot
            perform generation until a full verifier is attached.
        :return: The PreTrainedModel instance for the verifier that was attached.
        """
        verifier = super().attach_verifier(
            verifier=verifier,
            mode=mode,
        )

        # Extract layers from the verifier model
        self.embed_tokens = verifier.embed_tokens  # type: ignore[attr-defined]
        self.rotary_emb = verifier.rotary_emb  # type: ignore[attr-defined]
        self.lm_head = verifier.lm_head  # type: ignore[attr-defined]

        return verifier

    def detach_verifier(self):
        """
        Removes the reference to the attached verifier model and frees up the
        associated memory. After calling this method, the speculator will not
        be able to perform forward passes or generation until a new verifier
        is attached.
        """
        super().detach_verifier()

        del self.embed_tokens
        self.embed_tokens = None
        del self.rotary_emb
        self.rotary_emb = None
        del self.lm_head
        self.lm_head = None

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
        Execute the forward pass for speculative token generation.

        Processes input tokens and verifier hidden states through the EAGLE architecture
        to generate candidate tokens for speculative decoding. The method combines input
        embeddings with verifier hidden states via a fusion layer, processes them
        through a transformer decoder layer, and produces logits for next token
        prediction.

        :param input_ids: Token IDs for the current input sequence. Shape: (batch_size,
            sequence_length). These represent the tokens that will be converted to
            embeddings and combined with verifier hidden states.
        :param hidden_states: Hidden state representations from the verifier model
            corresponding to the input sequence. Shape: (batch_size, sequence_length,
            hidden_size). These capture the verifier's understanding of the context.
        :param attention_mask: Optional attention mask to avoid attending to padding
            tokens. Shape: (batch_size, sequence_length) for 2D or (batch_size, 1,
            sequence_length, sequence_length) for 4D causal mask.
        :param position_ids: Optional position indices for tokens in the sequence.
            Shape: (batch_size, sequence_length). If None, auto-generated based on
            sequence length and past key values.
        :param past_key_values: Optional cached key-value states from previous forward
            passes for efficient generation. Tuple of layer key-value pairs.
        :param use_cache: Whether to return key-value states for caching in subsequent
            forward passes. Useful for autoregressive generation efficiency.
        :param output_attentions: Whether to return attention weights from the
            transformer layer. Used for analysis and visualization.
        :param output_hidden_states: Whether to return hidden states from the
            transformer layer. Currently not implemented in this model.
        :param return_dict: Whether to return structured CausalLMOutputWithPast instead
            of raw logits. If None, uses config.use_return_dict default.
        :return: Either raw logits tensor (batch_size, sequence_length, vocab_size) if
            return_dict=False, or CausalLMOutputWithPast containing logits, past key
            values, and optional attention weights.
        :raises ValueError: If verifier components (embed_tokens, rotary_emb, lm_head)
            are not attached. Call attach_verifier() before using forward().
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
