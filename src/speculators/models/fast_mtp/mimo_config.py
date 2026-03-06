"""MiMoConfig shim for TencentBAC's FastMTP checkpoint compatibility.

MiMo uses model_type="mimo" in its transformer config, which is not registered
in standard transformers. This shim registers it so AutoConfig.for_model("mimo")
resolves correctly without requiring TencentBAC's patched transformers fork.
"""

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

__all__ = ["MiMoConfig"]


class MiMoConfig(Qwen2Config):
    """Transformer config for MiMo/FastMTP models.

    MiMo is a Qwen2-based architecture that adds multi-token prediction (MTP)
    fields. This shim preserves those fields when loading/saving configs.
    """

    model_type = "mimo"

    def __init__(
        self,
        num_nextn_predict_layers: int = 1,
        num_speculative_steps: int = 1,
        mtp_loss_step_weights: list = (1.0,),  # type: ignore[assignment]
        ntp_loss_weight: float = 0.0,
        mtp_loss_weight: float = 1.0,
        **kwargs,
    ):
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_speculative_steps = num_speculative_steps
        self.mtp_loss_step_weights = list(mtp_loss_step_weights)
        self.ntp_loss_weight = ntp_loss_weight
        self.mtp_loss_weight = mtp_loss_weight
        super().__init__(**kwargs)


# Register so AutoConfig.for_model("mimo") resolves correctly
CONFIG_MAPPING.register("mimo", MiMoConfig, exist_ok=True)
