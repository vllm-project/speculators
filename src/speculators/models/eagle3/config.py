from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorModelConfig

__all__ = [
    "Eagle3SpeculatorConfig",
]


@SpeculatorModelConfig.register("eagle3")
class Eagle3SpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for EAGLE-3 speculator with vocabulary mapping.

    EAGLE-3 features vocabulary mapping between draft (32K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    :param norm_before_residual: Apply hidden_norm before storing residual
    """

    speculators_model_type: Literal["eagle3"] = "eagle3"
    architectures: list[str] = Field(
        default_factory=lambda: ["Eagle3Speculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    norm_before_residual: bool = Field(
        default=False,
        description="Apply hidden_norm before storing residual",
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    eagle_aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the Eagle auxiliary hidden state layers",
    )

    embed_requires_grad: bool = Field(
        default=False,
        description="Whether embedding layer weights require gradients during training",
    )

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = LlamaConfig
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value
# ──────────────────────────────────────────────────────────────────────────────
# VwnConfig: VWN 参数容器 + 约束校验
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VwnConfig:
    """VWN 宽度扩展参数。

    约束（来自设计文档）：
        D' = r * D
        r  = n / m   →  r * m 必须为整数
        m  能整除 D

    """

    m: int = 2
    r: float = 1.5
    expand: bool = True

    def __post_init__(self):
        n_float = self.r * self.m
        self.n: int = round(n_float)
        if self.r == 1:
            self.expand = False

    def expanded_hidden_size(self, hidden_size: int) -> int:
        """返回 D' = r * D，同时校验整除约束。"""
        if hidden_size % self.m != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) 必须能被 m ({self.m}) 整除。"
            )
        d_prime = round(self.r * hidden_size)
        if d_prime % self.n != 0:
            raise ValueError(
                f"扩维后 D'={d_prime} 必须能被 n={self.n} 整除。"
            )
        return d_prime


# ──────────────────────────────────────────────────────────────────────────────
# VwnEagle3SpeculatorConfig
# ──────────────────────────────────────────────────────────────────────────────
@SpeculatorModelConfig.register("vwn_eagle3")  
class VwnEagle3SpeculatorConfig(Eagle3SpeculatorConfig):
    """
    VWN+Eagle3 草稿模型配置。
    """

    # 覆盖父类的 speculators_model_type
    speculators_model_type: Literal["vwn_eagle3"] = "vwn_eagle3"  # type: ignore[assignment]

    # ── 新增字段 ──────────────────────────────────────────────────────────────
    vwn_m: int = Field(
        default=2,
        description="VWN 原始 hidden vector 的等分份数，必须能整除 hidden_size。",
    )
    vwn_r: float = Field(
        default=1.5,
        description="VWN 扩维系数：D' = r * D，要求 r * m 为整数。",
    )
    vwn_expand: bool = Field(
        default=True,
        description="VWN 扩维开关。",
    )
    pre_vwn_version: int = Field(
        default=1,
        description="pre_vwn_layer 版本：0=完整融合(v0)，1=简化消融(v1)。",
    )
    vwn_draft_arch: str = Field(
        default="vwn_llama",
        description="VWN model_classes 的查表 key，值为 vwn_llama 或 vwn_qwen3。",
    )
    
    @property
    def vwn(self) -> VwnConfig:
        """返回经过校验的 VwnConfig 实例。"""
        return VwnConfig(m=self.vwn_m, r=self.vwn_r, expand=self.vwn_expand)
    
    @property
    def pre_vwn_layer_class(self) -> type:
        from .model_definitions import VwnEagle3PreVwnLayer, VwnEagle3PreVwnLayerV1
        """根据 pre_vwn_version 返回对应的 pre_vwn_layer 类。"""
        if self.pre_vwn_version == 0:
            return VwnEagle3PreVwnLayer
        elif self.pre_vwn_version == 1:
            return VwnEagle3PreVwnLayerV1
        else:
            raise ValueError(
                f"未知的 pre_vwn_version: {self.pre_vwn_version}，"
                "目前支持 0（完整融合）和 1（简化消融）。"
            )
