from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig, kl_div_loss, resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.components import BlockLayout
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.backbone import DFlash2Backbone
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import (
    SelectorTrainingOutput,
    compute_metrics,
    selector_training_candidates,
)
from speculators.models.dflash2.model_definitions import CandidateSelector

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DFlash2DraftModel",
]


@SpeculatorModel.register("dflash2")
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash with local convolution and bilinear candidate reranking."""

    config_class: ClassVar[type[DFlash2SpeculatorConfig]] = DFlash2SpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    def _make_backbone(self, config: DFlashSpeculatorConfig) -> DFlash2Backbone:
        assert isinstance(config, DFlash2SpeculatorConfig)  # noqa: S101
        return DFlash2Backbone(
            config.transformer_layer_config,
            len(self.target_layer_ids),
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
        )

    def __init__(self, config: DFlash2SpeculatorConfig) -> None:
        target_vocab_size = config.transformer_layer_config.vocab_size
        if config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "DFlash2 currently requires the full verifier vocabulary: "
                f"draft_vocab_size={config.draft_vocab_size}, "
                f"verifier_vocab_size={target_vocab_size}."
            )
        if config.selector_top_k > target_vocab_size:
            raise ValueError(
                f"selector_top_k ({config.selector_top_k}) cannot exceed "
                f"verifier_vocab_size ({target_vocab_size})."
            )
        super().__init__(config=config)

        initializer_range = getattr(
            config.transformer_layer_config, "initializer_range", 0.02
        )
        self.candidate_selector = CandidateSelector(
            vocab_size=target_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
            initializer_range=initializer_range,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlash2DraftModel":
        """Create a DFlash2 model from the shared training arguments."""
        config = DFlash2SpeculatorConfig(
            **cls._build_base_config_kwargs("dflash2", verifier_config, **kwargs),
            conv_kernel_size=kwargs.get("conv_kernel_size", 2),
            conv_group_size=kwargs.get("conv_group_size", 16),
            selector_rank=kwargs.get("selector_rank", 256),
            selector_top_k=kwargs.get("selector_top_k", 16),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve the unary and selector objectives used during training."""
        implementation = kwargs.get("loss_implementation", "fused")
        loss_config = resolve_loss_config(kwargs["loss_fn"], implementation)
        shared = {
            "loss_config": loss_config,
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "selector_loss_alpha": kwargs.get("selector_loss_alpha", 1.0),
            "selector_miss_policy": kwargs.get("selector_miss_policy", "replace"),
        }
        return dict(shared), dict(shared)

    def forward(
        self,
        # [1, total_seq_len, num_target_layers * hidden_size]
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        selector_loss_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        selector_miss_policy: str = "replace",
        **kwargs,
    ):
        return super().forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            loss_config=loss_config,
            gamma=gamma,
            max_anchors=max_anchors,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            correction_kwargs={"selector_miss_policy": selector_miss_policy},
            loss_kwargs={
                "selector_loss_alpha": selector_loss_alpha,
                "selector_miss_policy": selector_miss_policy,
            },
            **kwargs,
        )

    def apply_training_correction(
        self,
        hidden,
        logits,
        targets,
        layout,
        input_ids,
        _document_ids,
        selector_miss_policy="replace",
        **_kwargs,
    ):
        unary_logits = logits
        predecessor_ids = layout.predecessor_ids(input_ids)

        target_ids = targets.argmax(dim=-1)
        # shape: [1, num_anchors*block_size]
        candidate_ids = unary_logits.topk(self.candidate_selector.top_k, dim=-1).indices
        # shape: [1, num_anchors*block_size, top_k]
        training_candidate_ids, target_positions, contains_target = (
            selector_training_candidates(
                candidate_ids, target_ids, miss_policy=selector_miss_policy
            )
        )
        candidate_logits = self.candidate_selector.score_candidates(
            unary_logits,
            hidden,
            predecessor_ids.reshape(1, -1),
            training_candidate_ids,
        )
        selected_token_ids = training_candidate_ids.gather(
            -1, candidate_logits.detach().argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        return SelectorTrainingOutput(
            candidate_ids,  # without injection
            candidate_logits,
            target_positions,
            contains_target,
            selected_token_ids,
            target_ids,
        )

    def compute_loss_and_metrics(
        self,
        logits,
        targets,
        layout: BlockLayout,
        correction,
        *,
        loss_config: LossConfig | None,
        gamma: float,
        per_position_loss_weight: str,
        dpace_alpha: float,
        selector_loss_alpha=1.0,
        selector_miss_policy="replace",
        **_kwargs,
    ):
        return compute_metrics(
            unary_logits=logits,
            targets=targets,
            correction=correction,
            loss_mask=layout.loss_mask,
            block_size=self.block_size,
            top_k=self.candidate_selector.top_k,
            sample_from_anchor=self.config.sample_from_anchor,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            selector_loss_alpha=selector_loss_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            selector_miss_policy=selector_miss_policy,
        )
