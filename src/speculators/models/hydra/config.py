from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig

__all__ = [
    "HydraSpeculatorConfig",
]


@SpeculatorModelConfig.register("hydra")
class HydraSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for Hydra speculative decoding with sequentially-dependent
    draft heads.

    Hydra uses multiple MLP-based draft heads where each head conditions its
    prediction on the tokens predicted by earlier heads. Hydra++ extends this
    with a prefix attention layer and deeper MLPs.

    :param transformer_layer_config: Config for the optional prefix attention layer
    :param draft_vocab_size: Size of draft vocabulary for speculation
    :param num_hydra_heads: Number of draft heads (speculation depth)
    :param num_hydra_layers: Number of ResBlock layers per head
    :param use_prefix_mlp: Use prefix attention layer (Hydra++ mode)
    :param dropout_rate: Dropout rate between ResBlock layers
    """

    speculators_model_type: Literal["hydra"] = "hydra"
    architectures: list[str] = Field(
        default_factory=lambda: ["HydraSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Configuration for the prefix attention layer (Hydra++)",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    num_hydra_heads: int = Field(
        default=4,
        description="Number of draft heads for multi-token prediction",
    )

    num_hydra_layers: int = Field(
        default=4,
        description="Number of ResBlock layers per draft head",
    )

    use_prefix_mlp: bool = Field(
        default=True,
        description="Use prefix attention layer before MLP heads (Hydra++ mode)",
    )

    dropout_rate: float = Field(
        default=0.0,
        description=(
            "Dropout rate between ResBlock layers (0.2 recommended for Hydra++)"
        ),
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = Qwen3Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value

    @property
    def target_vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size
