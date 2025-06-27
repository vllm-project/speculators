from pydantic import Field

from speculators.config import SpeculatorModelConfig

__all__ = ["MLPSpeculatorConfig"]


@SpeculatorModelConfig.register("mlp")
class MLPSpeculatorConfig(SpeculatorModelConfig):
    """
    TODO
    """

    architectures: list[str] = Field(
        default_factory=lambda: ["MLPSpeculator"],
        description=("The architectures this speculator uses."),
    )
    torch_dtype: str = Field(
        default="bfloat16",
        description=(
            "The torch dtype this speculator uses. "
            "This is used to set the dtype of the model."
        ),
    )
    inputs: list[str] = Field(
        default_factory=lambda: ["input_embeddings", "hidden_states[-1]"],
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
    hidden_size: int = Field(
        default=4096,
        description=("The hidden size from the verifier that this speculator targets."),
    )
    intermediate_size: int = Field(
        default=4096,
        description=(
            "The intermediate size the MLP speculator uses for predicting tokens from."
        ),
    )
    vocab_size: int = Field(
        default=128256,
        description=(
            "The size of the vocabulary the MLP speculator supports for "
            "predicting tokens from."
        ),
    )
    num_layers: int = Field(
        default=5,
        description=(
            "The number of layers in the MLP speculator which ties directly to the "
            "maximum number of tokens the speculator can predict."
        ),
    )
    tie_weights: bool = Field(
        default=True,
        description=(
            "Whether to tie the weights across all of the MLP layers together so they "
            "are shared. Reduces the overall number of parameters in the model. "
            "If False, each layer will have its own set of embeddings, linear weights, "
            "and head weights."
        ),
    )
