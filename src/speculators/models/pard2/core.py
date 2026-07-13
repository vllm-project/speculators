from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.pard2.config import Pard2SpeculatorConfig
from speculators.models.pard2.metrics import compute_metrics
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "Pard2DraftModel",
]


@SpeculatorModel.register("pard2")
class Pard2DraftModel(DFlashDraftModel):
    """DFlash backbone with PARD-2 confidence-adaptive token (CAT) optimization.

    Replaces fixed position decay with CAT weights (cumulative product of
    target confidence), adds combined CE + KD loss, and supports dual-mode
    (target-dependent / target-independent) via stochastic Bernoulli gating
    on injected target features during training.
    """

    config_class: ClassVar[type[Pard2SpeculatorConfig]] = Pard2SpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: Pard2SpeculatorConfig) -> None:
        super().__init__(config=config)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "Pard2DraftModel":
        """Create a PARD-2 model from training arguments."""
        config = Pard2SpeculatorConfig(
            **cls._build_base_config_kwargs("pard2", verifier_config, **kwargs),
            ce_alpha=kwargs.get("ce_alpha", 0.1),
            kd_alpha=kwargs.get("kd_alpha", 1.0),
            kd_temperature=kwargs.get("kd_temperature", 1.0),
            target_feat_dropout=kwargs.get("target_feat_dropout", 0.1),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve PARD-2 training kwargs."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        max_anchors = kwargs.get("max_anchors", 3072)
        ce_alpha = kwargs.get("ce_alpha", 0.1)
        kd_alpha = kwargs.get("kd_alpha", 1.0)
        kd_temperature = kwargs.get("kd_temperature", 1.0)
        target_feat_dropout = kwargs.get("target_feat_dropout", 0.1)
        shared = {
            "loss_config": loss_config,
            "max_anchors": max_anchors,
            "ce_alpha": ce_alpha,
            "kd_alpha": kd_alpha,
            "kd_temperature": kd_temperature,
            "target_feat_dropout": target_feat_dropout,
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        max_anchors: int = 3072,
        ce_alpha: float = 0.1,
        kd_alpha: float = 1.0,
        kd_temperature: float = 1.0,
        target_feat_dropout: float = 0.1,
        **kwargs,
    ):
        # Bernoulli gating for dual-mode: during training, randomly drop
        # target features to learn target-independent drafting.
        if self.training and target_feat_dropout > 0.0:
            keep = (
                torch.rand(1, device=hidden_states.device) > target_feat_dropout
            ).to(hidden_states.dtype)
            hidden_states = hidden_states * keep

        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )

        # Compute full verifier logits at aligned positions for CAT weights
        with torch.no_grad():
            full_verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            full_verifier_logits = torch.roll(full_verifier_logits, 1, dims=1)
            aligned_verifier_logits = full_verifier_logits[:, anchored_block_indices]
            aligned_token_ids = input_ids[:, anchored_block_indices]

        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            verifier_logits=aligned_verifier_logits,
            target_token_ids=aligned_token_ids,
            ce_alpha=ce_alpha,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
        )
        draft_tokens = torch.argmax(logits, dim=-1)
        return draft_tokens, loss, metrics
