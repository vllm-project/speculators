import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers.activations import ACT2FN


from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as HFLlamaDecoderLayer,
    LlamaRMSNorm,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from speculators.models.transformer import TransformerSpeculatorConfig
from pydantic import Field
from logging import getLogger

LOGGER = getLogger(__name__)


class LlamaEagleConfig(TransformerSpeculatorConfig):
    """
    Configuration class for a LlamaEagleSpeculator model.

    This class extends TransformerSpeculatorConfig to include parameters specific
    to a Llama-based Eagle architecture, such as additional LayerNorm controls,
    input fusion details, and behaviors like disabling the first decoder LayerNorm.
    It inherits standard parameters (vocab_size, hidden_size, etc.) from
    PretrainedConfig via its parent classes

    """

    architectures: List[str] = Field(
        default_factory=lambda: ["LlamaEagleSpeculator"],
        description="The architectures this LlamaEagle speculator model uses."
    )

    # Eagle-specific fields not already in TransformerSpeculatorConfig
    # These will be initialized by Pydantic if passed in kwargs or use their defaults.
    gradient_checkpointing: bool = Field(
        default=False,
        description="Whether to use gradient checkpointing for the speculator's transformer layers."
    )
    gradient_checkpointing_use_reentrant: bool = Field(
        default=False,
        description="Specifies `use_reentrant` for torch.utils.checkpoint.checkpoint if gradient_checkpointing is True."
    )
    freeze_embed_tokens: bool = Field(
        default=True,
        description="If True, the speculator's token embedding layer weights are frozen."
    )
    use_verifier_input_layernorm: bool = Field(
        default=True,
        description="If True, a LayerNorm is applied to the verifier's hidden_states input before fusion. "
                    "Should ideally be False if `inputs_hidden_states_normalized` is True."
    )
    use_speculator_embed_layernorm: bool = Field(
        default=True,
        description="If True, a LayerNorm is applied to the speculator's own token embeddings."
    )
    input_projector_bias: bool = Field(
        default=False,
        description="Controls whether the input projection linear layer has a bias term."
    )
    fused_hs_processor_output_dim: Optional[int] = Field(
        default=None,
        description="Output dimension for the internal linear processor of verifier's hidden states "
                    "when `transformer_input_type` is 'fused'. If None, defaults to `hidden_size // 2`."
    )
    disable_first_decoder_input_layernorm: bool = Field(
        default=True,
        description="If True, the input_layernorm of the first EagleLlamaDecoderLayer "
                    "in the speculator is replaced with nn.Identity()."
    )


class EagleLlamaDecoderLayer(HFLlamaDecoderLayer):
    """
    A LlamaDecoderLayer variant for the Eagle speculator, allowing the input
    LayerNorm to be disabled, typically for the first layer.
    """
    def __init__(self, config: LlamaConfig, layer_idx: int, disable_input_layernorm: bool = True):
        super().__init__(config, layer_idx)
        if disable_input_layernorm and hasattr(self, 'input_layernorm') and \
           isinstance(self.input_layernorm, (LlamaRMSNorm, nn.LayerNorm)):
            self.input_layernorm = nn.Identity()


# TODO: Should LlamaEagleSpeculator inherit from AutoModel?
class LlamaEagleSpeculator(nn.Module):
    """
    Llama-based Eagle speculator model.

    This model architecture is designed to generate speculative tokens based on
    inputs from a verifier model and its own previous predictions. It features
    configurable input fusion and optional LayerNorm stages.
    """
    def __init__(self, config: LlamaEagleConfig):
        super().__init__()
        self.config = config
        self.speculators_config = config.speculators_config
        self.verifier_config = self.speculators_config.verifier

        self._setup_core_properties()
        self._setup_embeddings()
        self._setup_optional_input_layernorms()

        # Speculator's own embedding dimension
        # Determine the hidden size for the main transformer layers, which might change
        # based on the input(s)
        speculator_embed_dim = self.config.hidden_size
        current_transformer_hidden_size = self.config.hidden_size

        self.input_projector, self.post_projection_act, speculator_transformer_hidden_size = \
            self._setup_input_fusion(speculator_embed_dim, current_transformer_hidden_size)

        self._setup_transformer_stack(speculator_transformer_hidden_size)
        self._setup_output_processing(speculator_transformer_hidden_size)
        self._apply_target_dtype()

    def _setup_core_properties(self):
        """Sets up core model properties like gradient checkpointing."""
        self.gradient_checkpointing = getattr(self.config, "gradient_checkpointing", False)
        self.padding_idx = getattr(self.config, "pad_token_id", None) # Typically the pad_token_id from verifier config???

    def _setup_embeddings(self):
        """Initializes and optionally freezes the token embedding layer."""
        speculator_embed_dim = self.config.hidden_size
        self.embed_tokens = nn.Embedding(
            self.verifier_config.vocab_size,
            speculator_embed_dim,
            self.padding_idx
        )
        if getattr(self.config, "freeze_embed_tokens", True):
            self.embed_tokens.requires_grad_(False)

    def _setup_optional_input_layernorms(self):
        """
        Initializes optional LayerNorms for inputs, based on config flags.
        These were present in the HASS/TTT model.
        """
        # LayerNorm for the verifier's hidden_states input
        if getattr(self.config, "use_verifier_input_layernorm", True):
            self.verifier_input_layernorm = LlamaRMSNorm(
                self.verifier_config.hidden_size,
                eps=getattr(self.config, "rms_norm_eps", 1e-6)
            )
        else:
            self.verifier_input_layernorm = nn.Identity()

        # LayerNorm for the speculator's own input_embeddings
        if getattr(self.config, "use_speculator_embed_layernorm", True):
            self.speculator_embed_layernorm = LlamaRMSNorm(
                self.config.hidden_size, # Speculator's embedding dim
                eps=getattr(self.config, "rms_norm_eps", 1e-6)
            )
        else:
            self.speculator_embed_layernorm = nn.Identity()

    def _setup_input_fusion(self, speculator_embed_dim: int, current_transformer_hs: int) -> Tuple[nn.Module, nn.Module, int]:
        """
        Configures the mechanism for fusing verifier inputs with speculator embeddings.
        Returns the projection layer, post-projection activation, and the
        effective hidden size for subsequent transformer layers.

        TODO: Refactor this to be more modular and handle different inputs
            maybe regex matching
        """

        verifier_blob_dim = 0
        # Accumulate dimensions of inputs coming from the verifier
        if "input_embeddings" in self.config.inputs:
            verifier_blob_dim += self.verifier_config.hidden_size

        # Add more verifier inputs here if specified in `self.config.inputs`
        if "hidden_states[-2]" in self.config.inputs:
            verifier_blob_dim += self.verifier_config.hidden_size

        input_type = self.config.transformer_input_type
        projector_bias = getattr(self.config, "input_projector_bias", False)
        hidden_act_fn = getattr(self.config, "hidden_act", "silu")

        if input_type == "concat":
            # Inputs are concatenated; this becomes the hidden size for transformer layers.
            final_transformer_hs = verifier_blob_dim + speculator_embed_dim
            if current_transformer_hs != final_transformer_hs:
                LOGGER.info(
                    f"For 'concat' input, speculator transformer hidden_size is {final_transformer_hs} "
                    f"(overriding initial {current_transformer_hs})."
                    )
            return nn.Identity(), nn.Identity(), final_transformer_hs

        elif input_type == "fused":
            # TODO: What should be done here?
            raise NotImplementedError(
                "The 'fused' transformer_input_type is not yet implemented in LlamaEagleSpeculator."
            )

        elif input_type in ["linear_no_bias", "linear_with_bias"]:
            projection_input_dim = verifier_blob_dim + speculator_embed_dim
            # TODO: consolidate to have just one argument
            use_bias = (input_type == "linear_with_bias") or projector_bias
            projector = nn.Linear(projection_input_dim, current_transformer_hs, bias=use_bias)
            activation = ACT2FN[hidden_act_fn]
            return projector, activation, current_transformer_hs
        else:
            raise ValueError(f"Unsupported transformer_input_type: {input_type}")

    def _get_decoder_llama_config(self, transformer_hidden_size: int) -> LlamaConfig:
        """Creates a LlamaConfig for the speculator's decoder layers."""
        # These attributes should be part of TransformerSpeculatorConfig, inherited from PretrainedConfig
        # or explicitly defined.
        params = {
            "vocab_size": self.verifier_config.vocab_size, # Match verifier
            "hidden_size": transformer_hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_hidden_layers": self.config.num_hidden_layers, # This is for the speculator
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
            "hidden_act": getattr(self.config, "hidden_act", "silu"),
            "max_position_embeddings": getattr(self.config, "max_position_embeddings", 2048),
            "rms_norm_eps": getattr(self.config, "rms_norm_eps", 1e-6),
            "use_cache": getattr(self.config, "use_cache", True),
            "pad_token_id": self.padding_idx,
            "bos_token_id": self.verifier_config.bos_token_id,
            "eos_token_id": self.verifier_config.eos_token_id,
            "tie_word_embeddings": getattr(self.config, "tie_word_embeddings", False),
            "rope_theta": getattr(self.config, "rope_theta", 10000.0),
            "attention_bias": getattr(self.config, "attention_bias", False),
        }
        active_params = {k: v for k, v in params.items() if v is not None}
        return LlamaConfig(**active_params)

    def _setup_transformer_stack(self, transformer_hidden_size: int):
        """Initializes the stack of LlamaDecoderLayers for the speculator."""
        decoder_config = self._get_decoder_llama_config(transformer_hidden_size)
        self.decoder_config = decoder_config

        self.layers = nn.ModuleList()
        for i in range(decoder_config.num_hidden_layers):
            # Disable input_layernorm for the first layer by default for Eagle-like behavior.
            is_first_layer = (i == 0)
            disable_ln = is_first_layer and getattr(self.config, "disable_first_decoder_input_layernorm", True)
            self.layers.append(
                EagleLlamaDecoderLayer(
                    config=decoder_config,
                    layer_idx=i,
                    disable_input_layernorm=disable_ln
                )
            )

    def _setup_output_processing(self, transformer_hidden_size: int):
        """Initializes the final LayerNorm and optional LM head."""
        if not self.config.transformer_remove_last_layer_norm:
            self.norm = LlamaRMSNorm(transformer_hidden_size, eps=getattr(self.config, "rms_norm_eps", 1e-6))
        else:
            self.norm = nn.Identity()

        if not self.config.use_verifier_lm_head:
            self.lm_head = nn.Linear(transformer_hidden_size, self.verifier_config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def _apply_target_dtype(self):
        """Converts the model to the specified torch_dtype from the config."""
        target_dtype_str = self.config.torch_dtype
        target_dtype = getattr(torch, target_dtype_str)
        try:
            self.to(target_dtype)
        except Exception as exception:
            LOGGER.warning(f"Warning: Failed to convert model to {target_dtype_str}")
            raise exception


    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The forward method for LlamaEagleSpeculator has not been implemented."
        )
