from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.xpress.config import XPressSpeculatorConfig
from speculators.models.xpress.model_definitions import XPressRefinerHead
from speculators.models.xpress.objective import xpress_objective

__all__ = [
    "XPressDraftModel",
]


@SpeculatorModel.register("xpress")
class XPressDraftModel(DFlashDraftModel):
    """DFlash backbone plus the XPress causal-refiner head.

    Training runs one teacher-forced refine pass plus ``consistency_passes``
    free-running rounds (each conditioned on the previous round's argmax, the
    exact recurrence of inference-time Jacobi decoding, stop-grad between
    rounds), and anchors the co-trained backbone with a base-logits term.
    Everything else is inherited from DFlash.
    """

    objective = staticmethod(xpress_objective)
    legacy_correction_name = "refiner_head"

    config_class: ClassVar[type[XPressSpeculatorConfig]] = XPressSpeculatorConfig  # type: ignore[misc]

    def __init__(self, config: XPressSpeculatorConfig) -> None:
        super().__init__(config=config)
        self.correction = XPressRefinerHead(
            verifier_vocab_size=self.verifier_vocab_size,
            draft_vocab_size=self.draft_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            block_size=self.block_size,
            rank=config.xpress_rank,
            mlp_ratio=config.xpress_mlp_ratio,
        )
        self.post_init()

    @property
    def refiner_head(self) -> XPressRefinerHead:
        """Compatibility accessor; the module is registered only as correction."""
        assert isinstance(self.correction, XPressRefinerHead)  # noqa: S101
        return self.correction

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "XPressDraftModel":
        """Create an XPress model from training arguments (mirrors DSpark)."""
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        config = XPressSpeculatorConfig(
            **cls._build_base_config_kwargs("xpress", verifier_config, **kwargs),
            xpress_rank=kwargs.get("xpress_rank", 256),
            xpress_mlp_ratio=kwargs.get("xpress_mlp_ratio", 2),
            num_jacobi_passes=kwargs.get("num_jacobi_passes", 6),
        )
        if sample_from_anchor_arg is not None:
            config.sample_from_anchor = sample_from_anchor_arg

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve XPress's compound loss and consistency knobs."""
        implementation = kwargs.get("loss_implementation", "fused")
        shared = {
            "loss_config": resolve_loss_config(kwargs["loss_fn"], implementation),
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 3072),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "consistency_weight": kwargs.get("consistency_weight", 0.3),
            "consistency_passes": kwargs.get("consistency_passes", 3),
            "base_anchor_weight": kwargs.get("base_anchor_weight", 0.6),
            "base_anchor_full_weight": kwargs.get("base_anchor_full_weight", False),
            "ce_from_data": kwargs.get("ce_from_data", False),
        }
        return dict(shared), dict(shared)
