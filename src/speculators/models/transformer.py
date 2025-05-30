from typing import Literal

from pydantic import Field

from speculators.config import SpeculatorModelConfig

__all__ = ["TransformerSpeculatorConfig"]


class TransformerSpeculatorConfig(SpeculatorModelConfig):
    """
    TODO
    """

    architectures: list[str] = Field(
        default_factory=lambda: ["TransformerSpeculator"],
        description=("The architectures this speculator uses."),
    )
    torch_dtype: str = Field(
        default="bfloat16",
        description=(
            "The torch dtype this speculator uses. "
            "This is used to set the dtype of the model."
        ),
    )
    inputs: list[str] = Field(  # input_embeddings layer norm
        default_factory=lambda: ["input_embeddings", "hidden_states[-2]"],
        description=(
            "The inputs from the verifier that this speculator uses to generate "
            "proposal tokens for verification."
        ),
    )
    inputs_hidden_states_normalized: bool = Field(
        default=False,
        description=(
            "Whether to use the hidden states of the verifier after the layer norm is "
            "applied. If False, the hidden states are used before the "
            "layer norm is applied."
        ),
    )
    transformer_layer_type: str = Field(
        default="LlamaDecoderLayer",
        description=(
            "The type of transformer layer this speculator is created with. "
            "It must be a layer type supported in Transformers. "
            "Default is LlamaDecoderLayer."
        ),
    )
    transformer_input_type: Literal[
        "linear_no_bias", "linear_with_bias", "concat", "fused"
    ] = Field(
        default="linear_no_bias",
        description=(
            "How the inputs from the verifier are combined to create the input to "
            "the transformer layer. Must be one of the following: "
            "linear_no_bias: inputs are concatenated and passed through a linear "
            "layer with no bias. "
            "linear_with_bias: inputs are concatenated and passed through a linear "
            "layer with a bias. "
            "concat: inputs are only concatenated. "
            "fused: hidden states are concatenated and passed through a linear layer "
            "and then concatenated with the input_embeddings."
        ),
    )
    transformer_remove_last_layer_norm: bool = Field(
        default=False,
        description=(
            "Whether to remove the last layer norm from the transformer layer. "
        ),
    )
    use_verifier_lm_head: bool = Field(
        default=False,
        description=(
            "Whether to use the verifier's LM head to generate the proposal tokens. "
            "If False, the transformer layer uses its own LM head to generate the "
            "proposal tokens."
        ),
    )
