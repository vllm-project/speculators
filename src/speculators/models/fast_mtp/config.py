"""Configuration for the FastMTP speculator model.

:class:`FastMTPConfig` stores two things:

* **Architecture geometry** — via ``transformer_layer_config`` (a
  ``PretrainedConfig`` subclass), from which ``hidden_size``, ``vocab_size``,
  and all other dimension fields are derived.  Fields are never duplicated at
  the top level.
* **vLLM compatibility** — ``num_nextn_predict_layers`` is kept at 1 because
  vLLM reads this field by name to instantiate the correct number of MTP head
  instances.

Derived properties (``hidden_size``, ``vocab_size``, ``num_speculative_steps``)
are computed on demand and are not stored in the config JSON.
"""

from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig

from speculators import SpeculatorModelConfig

__all__ = ["FastMTPConfig"]


@SpeculatorModelConfig.register("mtp")
class FastMTPConfig(SpeculatorModelConfig):
    """Configuration for FastMTP (Multi-Token Prediction) speculator.

    Targets Qwen3-Next checkpoints: sparse MoE, ``model_type="qwen3_next"``,
    hidden=2048, 16 attention heads, 2 KV heads, vocab=151936.

    Architecture: a single MTP layer with attention and MLP, combining verifier
    hidden states with token embeddings via an explicit input projection.
    ``embed_tokens`` and ``lm_head`` share the verifier's full vocabulary.

    The forward pass is teacher-forced: at step k the model receives
    ``input_ids[:, k:k+valid_len]`` as token embeddings and
    ``hidden_states[:, :valid_len]`` from the verifier.  Hidden states are
    passed recursively: each step's MTP output feeds the next step.

    **Stored fields:**

    :param transformer_layer_config: Configuration for the underlying
        transformer architecture (e.g., ``Qwen3NextConfig``). All architecture
        dimensions (``hidden_size``, ``vocab_size``, attention heads, MLP dims)
        are derived from this config.  Serialised as a ``to_diff_dict()``
        snapshot and reconstructed via ``AutoConfig`` on load.
    :param num_nextn_predict_layers: Fixed at 1.  Named to match vLLM's
        ``num_nextn_predict_layers`` field, which vLLM reads by name to
        configure the number of MTP head instances.

    **Derived properties (not stored):**

    :property hidden_size: Hidden dimension, from
        ``transformer_layer_config.hidden_size``.
    :property vocab_size: Vocabulary size, from
        ``transformer_layer_config.vocab_size``.
    :property num_speculative_steps: Number of teacher-forced prediction steps,
        derived from ``speculators_config.proposal_methods[0].speculative_tokens``.
        Reads from ``self.speculators_config``, which the SpeculatorModel
        framework sets after config construction.  Do not access before
        ``speculators_config`` is attached.
    """

    speculators_model_type: Literal["mtp"] = "mtp"
    architectures: list[str] = Field(
        default=["FastMTPSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=lambda: AutoConfig.for_model("qwen3_next"),
        description="Underlying transformer architecture config (e.g., Qwen3NextConfig)",  # noqa: E501
    )

    num_nextn_predict_layers: int = Field(
        default=1,
        description=(
            "Fixed at 1. Named to match vLLM's num_nextn_predict_layers field, "
            "which vLLM reads by name to configure MTP head instances."
        ),
    )

    @property
    def hidden_size(self) -> int:
        """Hidden dimension size, derived from transformer_layer_config."""
        return self.transformer_layer_config.hidden_size  # type: ignore[return-value]  # PretrainedConfig attrs are dynamically typed

    @property
    def vocab_size(self) -> int:
        """Vocabulary size, derived from transformer_layer_config."""
        return self.transformer_layer_config.vocab_size  # type: ignore[return-value]  # PretrainedConfig attrs are dynamically typed

    @property
    def num_speculative_steps(self) -> int:
        """Number of teacher-forced prediction steps, from the proposal config."""
        if not self.speculators_config or not self.speculators_config.proposal_methods:
            raise ValueError(
                "speculators_config must have at least one proposal method. "
                "num_speculative_steps cannot be accessed before speculators_config "
                "is attached to this config."
            )
        steps: int = self.speculators_config.proposal_methods[0].speculative_tokens  # type: ignore[union-attr,attr-defined]
        if steps <= 0:
            raise ValueError(
                f"speculative_tokens must be > 0, got {steps}. "
                "Check the proposal_methods configuration."
            )
        return steps

    @field_validator("num_nextn_predict_layers")
    @classmethod
    def validate_num_nextn_predict_layers(cls, value: int) -> int:
        """Reject configs that request more than one FastMTP layer."""
        if value != 1:
            raise ValueError(
                f"FastMTP currently only supports 1 layer, got {value}. "
                "Multi-layer support may be added in future versions."
            )
        return value

    @field_serializer("transformer_layer_config")
    def serialize_transformer_layer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer_layer_config to dict for JSON storage."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_layer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config from dict or PretrainedConfig."""
        if isinstance(value, dict):
            if "model_type" not in value:
                raise ValueError(
                    "transformer_layer_config dict must include a 'model_type' key "
                    "so the correct PretrainedConfig subclass can be selected. "
                    f"Keys present: {list(value.keys())}"
                )
            config_class = AutoConfig.for_model(  # type: ignore[attr-defined]
                model_type=value["model_type"]
            ).__class__
            return config_class(**value)
        return value
