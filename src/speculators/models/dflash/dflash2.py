from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import CandidateSelector
from speculators.models.utils import conditional_torch_compile


@SpeculatorModel.register("dflash2")
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash backbone with grouped dynamic causal convolutions and a candidate
    selector head.

    Convolutions are initialized from config on each decoder layer (handled by
    the base class when ``conv_kernel_size`` / ``conv_group_size`` are set).
    The candidate selector adds learned bilinear edge scores to the draft
    logits before the loss, training the selector jointly with the backbone.
    """

    def __init__(self, config: DFlashSpeculatorConfig) -> None:
        super().__init__(config=config)

        self.candidate_selector: CandidateSelector | None = None
        if config.selector_rank is not None and config.selector_top_k is not None:
            self.candidate_selector = CandidateSelector(
                vocab_size=self.draft_vocab_size,
                hidden_size=config.transformer_layer_config.hidden_size,
                rank=config.selector_rank,
                top_k=config.selector_top_k,
            )

    def _predecessor_ids(
        self,
        input_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Build teacher-forced predecessor token IDs in draft vocab space."""
        block_tokens = input_ids[0, anchored_block_indices]
        num_blocks = block_tokens.shape[0] // self.block_size
        block_tokens = block_tokens.view(num_blocks, self.block_size)

        if self.config.sample_from_anchor:
            predecessor_ids = block_tokens
        else:
            predecessor_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )

        predecessor_ids = predecessor_ids.view(1, -1)

        if self.use_draft_vocab and self.t2d is not None:
            v2d = self.t2d.long().cumsum(0) - 1
            predecessor_ids = v2d[predecessor_ids]

        return predecessor_ids

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlash2DraftModel":
        base_kwargs = cls._build_base_config_kwargs("dflash2", verifier_config, **kwargs)
        base_kwargs.update(
            conv_kernel_size=kwargs.get("conv_kernel_size"),
            conv_group_size=kwargs.get("conv_group_size"),
            selector_rank=kwargs.get("selector_rank"),
            selector_top_k=kwargs.get("selector_top_k"),
        )
        config = DFlashSpeculatorConfig(**base_kwargs)
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], kwargs.get("loss_implementation", "fused")
        )
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 512)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
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
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
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

        if self.candidate_selector is not None:
            predecessor_ids = self._predecessor_ids(input_ids, anchored_block_indices)
            context = (
                self.candidate_selector.predecessor_codebook(predecessor_ids)
                * self.candidate_selector.hidden_projection(hidden)
            )
            edge_scores = context @ self.candidate_selector.successor_codebook.weight.T
            logits = logits + edge_scores

        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        return None, loss, metrics
