import math
import os
from typing import ClassVar, Optional, Union, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from pydantic import Field

from speculators.config import SpeculatorModelConfig
from speculators.model import SpeculatorModel

__all__ = ["MLPSpeculatorConfig", "MLPSpeculator"]


@SpeculatorModelConfig.register("mlp")
class MLPSpeculatorConfig(SpeculatorModelConfig):
    """
    A SpeculatorModelConfig implementation to be used with the MLPSpeculator

    Example:
        ```python
        from speculators import SpeculatorsConfig, VerifierConfig
        from speculators.models import MLPSpeculatorConfig
        from transformers import AutoConfig

        config = MLPSpeculatorConfig(
            speculators_config=SpeculatorsConfig(
                algorithm="mlp",
                verifier=VerifierConfig(
                    name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                    architectures=["LlamaForCausalLM"],
                )
            )
        )

    ```
    """

    speculators_model_type: Literal["mlp"] = "mlp"
    architectures: list[str] = Field(
        default_factory=lambda: ["MLPSpeculator"],
        description=("The architectures this speculator uses."),
    )
    dtype: Union[str, torch.dtype] = Field(
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


@SpeculatorModel.register("mlp")
class MLPSpeculator(SpeculatorModel):
    """
    A SpeculatorModel implementation that improves upon the Medusa architecture by conditioning on both
    context vectors AND sampled tokens.

    Architecture Overview:
        The MLP speculator consists of:
        1. Input embedding layer (shared with verifier)
        2. Multi-stage MLP prediction heads with token conditioning
        3. Weighted combination of context vectors and token embeddings
        4. Layer normalization and GELU activation

    Speculative Decoding Process:
        1. Verifier model processes input and generates hidden states (context vectors)
        2. MLP speculator uses both context vectors and token embeddings to predict next tokens
        3. Each prediction head conditions on the previous predicted token
        4. Weighted combination of state and embedding vectors
        5. Layer normalization and activation applied
        6. Linear projection to vocabulary space for token prediction

    Reference: https://arxiv.org/html/2404.19124v1
    """

    # PretrainedModel settings
    config_class: ClassVar[type[MLPSpeculatorConfig]] = MLPSpeculatorConfig  # type: ignore[misc]

    def __init__(
        self,
        config: MLPSpeculatorConfig,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[
            Literal["detached", "full"] # TODO: Add train_only mode
        ] = None,
    ):
        """
        Initializes a MLP speculator architecture with configurable components based
        on the provided configuration.
        """
        if not isinstance(config, MLPSpeculatorConfig):
            raise ValueError(
                "config must be an instance of MLPSpeculatorConfig, "
                f"got {type(config)} instead."
            )

        # Initialize model parameters from config
        self.n_predict = config.num_layers
        self.emb_dim = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size != 0 else config.hidden_size
        self.inner_dim = inner_dim
        self.vocab_size = config.vocab_size
        self.inputs_hidden_states_normalized = config.inputs_hidden_states_normalized
        self._tie_weights = config.tie_weights

        # Set layers pulled from the verifier to None until attach is called
        self.embed_tokens: Optional[nn.Embedding] = None
        self.lm_head: Optional[nn.Linear] = None

        # Intialize model inputs from config
        self.last_hidden_state = config.inputs[1]
        self.input_embeddings = config.inputs[0]

        # Delayed initialization to ensure everything needed for attach_verifier is set
        super().__init__(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
        )

        # Initialize MLP layers
        self._initialize_mlp_layers()

        self.post_init()  # type: ignore[attr-defined]

    def attach_verifier(
        self,
        verifier: Union[str, os.PathLike, PreTrainedModel],
        mode: Optional[Literal["full", "train_only"]] = None,
    ) -> PreTrainedModel:
        """
        Attach a verifier model to the MLPSpeculator.

        :param verifier: The verifier model to attach
        :param mode: The attachment mode
        :return: The attached verifier model
        """
        verifier = super().attach_verifier(verifier=verifier, mode=mode)

        # Extract layers from the verifier model
        if hasattr(verifier, "model"):
            self.embed_tokens = verifier.model.embed_tokens  # type: ignore[assignment,union-attr]
        else:
            self.embed_tokens = verifier.embed_tokens  # type: ignore[assignment,attr-defined]

        # lm_head is always at the top level of the verifier
        self.lm_head = verifier.lm_head  # type: ignore[assignment,attr-defined]

        return verifier

    def _initialize_mlp_layers(self):
        self.emb_layers = nn.ModuleList(
            [nn.Embedding(self.vocab_size, self.inner_dim) for _ in range(self.n_predict)]
        )
        self.proj_layers = nn.ModuleList(
            [
                nn.Linear((self.emb_dim if i == 0 else self.inner_dim), self.inner_dim, bias=False)
                for i in range(self.n_predict)
            ]
        )
        self.head = nn.ModuleList(
            [nn.Linear(self.inner_dim, self.vocab_size, bias=False) for _ in range(self.n_predict)]
        )
        self.layernorms = nn.ModuleList(
            [
                nn.LayerNorm(self.inner_dim, elementwise_affine=True)
                for _ in range(self.n_predict)
            ]
        )
        self.activation = nn.GELU()

        # Weight scaling parameters for proper state management
        self.state_weight = 0.5 ** (0.5 / self.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.inner_dim / 2))

        if self._tie_weights:
            for emb in self.emb_layers:
                emb.weight = self.emb_layers[0].weight

            for head in self.head:
                head.weight = self.head[0].weight

            for ln in self.layernorms:
                ln.weight = self.layernorms[0].weight
                ln.bias = self.layernorms[0].bias

            # First projection layer has a different size so allow different initial proj from base into model
            for i in range(2, self.n_predict):
                self.proj_layers[i].weight = self.proj_layers[1].weight

    def detach_verifier(self):
        """
        Removes the reference to the attached verifier model and frees up
        the associated memory. After calling this method, the speculator will not
        be able to perform forward passes or generation until a new verifier
        is attached.
        """
        super().detach_verifier()

        if self.embed_tokens is not None:
            del self.embed_tokens
            self.embed_tokens = None

        if self.lm_head is not None:
            del self.lm_head
            self.lm_head = None

    def tie_weights(self):
        """
        Tie weights across all MLP layers if tie_weights is enabled.

        This method implements weight tying for the MLP speculator layers,
        sharing weights across all prediction heads to reduce parameter count.
        """
        if self._tie_weights:
            # Tie embedding weights
            for i in range(1, len(self.emb_layers)):
                self.emb_layers[i].weight = self.emb_layers[0].weight

            # Tie head weights
            for i in range(1, len(self.head)):
                self.head[i].weight = self.head[0].weight

            # Tie layer norm weights and biases
            for i in range(1, len(self.layernorms)):
                self.layernorms[i].weight = self.layernorms[0].weight
                self.layernorms[i].bias = self.layernorms[0].bias

            # Tie projection layer weights (except first layer which has different input size)
            for i in range(2, len(self.proj_layers)):
                self.proj_layers[i].weight = self.proj_layers[1].weight


    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithPast]:
        """
        Forward pass for MLP speculation.

        :param input_ids: Token IDs for the current input sequence. Shape: (batch_size, sequence_length)
        :param hidden_states: Hidden state representations from the verifier model. Shape: (batch_size, sequence_length, hidden_size)
        :param attention_mask: Optional attention mask
        :param position_ids: Optional position IDs
        :param past_key_values: Optional cached key-values
        :param use_cache: Whether to cache key-values
        :param output_attentions: Return attention weights
        :param output_hidden_states: Return hidden states
        :param return_dict: Return dict output
        :return: Model outputs with logits for each prediction layer
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.embed_tokens is None or self.lm_head is None:
            raise ValueError(
                "Verifier model layers not initialized. "
                "Call `attach_verifier` to set up the model before using forward."
            )

        # Get batch size for shape verification
        batch_size = input_ids.shape[0]

        # Initialize logits list
        logits = []


        # Start with the last token from input_ids
        current_token = input_ids[:, -1:]  # b 1

        # Iterate through each head
        for i in range(self.n_predict):
            # Get embeddings for current token
            z = self.emb_layers[i](current_token)  # b 1 d

            # Project hidden states (context vector from base model)
            if i == 0:
                # Use the last hidden state from the verifier
                last_hidden = hidden_states[:, -1, :]  # b hidden_size
                state = self.proj_layers[i](last_hidden)  # b inner_dim
                state = state.unsqueeze(1)  # b 1 inner_dim
            else:
                # Subsequent layers: use the output from previous head
                state = self.proj_layers[i](state)  # b 1 inner_dim

            # Weighted sum of state_weight*state and emb_weight*z
            state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
            state = self.activation(self.layernorms[i](state))  # b 1 d

            # Get logits for this prediction layer
            logits = self.head[i](state)  # b 1 vocab_size

        stacked_logits = torch.stack(logits, dim=1)  # b n_predict vocab_size

        if not return_dict:
            return stacked_logits

        return CausalLMOutputWithPast(
            logits=stacked_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


    def get_position_embeddings(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Get position embeddings for the given position IDs.
        This is required by PreTrainedModel but not used in MLP speculator.
        """
        # MLP speculator doesn't use position embeddings
        pass

    def resize_position_embeddings(self, new_num_position_embeddings: int) -> None:
        """
        Resize position embeddings to new size.
        This is required by PreTrainedModel but not used in MLP speculator.
        """
        # MLP speculator doesn't use position embeddings
        pass
