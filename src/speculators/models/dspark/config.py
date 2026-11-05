from typing import Any, Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.glm5 import GLM5_DSPARK_MODEL_TYPE

__all__ = [
    "EXPORT_ARCHITECTURES",
    "DSparkSpeculatorConfig",
]

# Serving-side class names keyed by draft decoder ``model_type``. HF
# ``save_pretrained`` stamps ``architectures`` with the Python class name; these
# entries replace it in the written ``config.json`` so the runtime can dispatch
# the draft (vLLM-Ascend registers ``Glm5DSparkForCausalLM``).
EXPORT_ARCHITECTURES: dict[str, str] = {
    GLM5_DSPARK_MODEL_TYPE: "Glm5DSparkForCausalLM",
}


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus a Markov logit-bias head and a confidence head.

    The Markov head lets each draft position condition on previously sampled
    tokens within the block; the confidence head predicts the per-position
    acceptance probability. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["dspark"] = "dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
            "Default True matches DeepSeek/DeepSpec convention."
        ),
    )

    # Sequential (Markov) head.
    markov_rank: int = Field(
        default=256,
        description=(
            "Low-rank dimension of the Markov logit-bias factorization B = W1 @ W2. "
            "Set to 0 to disable the sequential head (pure DFlash drafting)."
        ),
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla",
        description=(
            "Sequential head variant: 'vanilla' (first-order Markov bias), 'gated' "
            "(hidden-gated bias), or 'rnn' (recurrent state over the block)."
        ),
    )

    # Confidence head.
    enable_confidence_head: bool = Field(
        default=True,
        description="Whether to attach the per-position acceptance-probability head.",
    )
    confidence_head_with_markov: bool = Field(
        default=True,
        description=(
            "Concatenate the Markov previous-token embedding with the backbone "
            "hidden state as the confidence-head input."
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize with the serving architecture name when one is registered.

        Applied here (rather than on the ``architectures`` field) because HF
        ``save_pretrained`` overwrites that field right before writing the
        config; ``to_dict`` runs after the overwrite and feeds ``config.json``.
        """
        config_dict = super().to_dict()
        export_arch = EXPORT_ARCHITECTURES.get(
            self.transformer_layer_config.model_type
        )
        if export_arch is not None:
            config_dict["architectures"] = [export_arch]
        return config_dict
