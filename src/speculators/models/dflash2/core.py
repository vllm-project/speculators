from functools import partial
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash.backbone import DFlashBackbone
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.dflash2.objective import dflash2_objective

__all__ = [
    "DFlash2DraftModel",
]


@SpeculatorModel.register("dflash2")
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash with local convolution and bilinear candidate reranking."""

    objective = staticmethod(dflash2_objective)
    legacy_correction_name = "candidate_selector"

    config_class: ClassVar[type[DFlash2SpeculatorConfig]] = DFlash2SpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

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

        for layer_ in self.get_decoder_layers():
            # Reset after the wrapper has initialized the backbone parameters.
            reset_convolutions = getattr(layer_, "reset_convolutions", None)
            if not callable(reset_convolutions):
                raise TypeError(
                    f"{type(layer_).__name__} has no reset_convolutions(); a DFlash2 "
                    "decoder layer must wrap its sublayers in resettable convolutions."
                )
            reset_convolutions()

        initializer_range = getattr(
            config.transformer_layer_config, "initializer_range", 0.02
        )
        self.correction = CandidateSelector(
            vocab_size=target_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
            initializer_range=initializer_range,
        )

    def _create_backbone(self, config: DFlashSpeculatorConfig) -> DFlashBackbone:
        assert isinstance(config, DFlash2SpeculatorConfig)  # noqa: S101
        return DFlashBackbone(
            config.transformer_layer_config,
            len(self.target_layer_ids),
            decoder_layer_factory=partial(
                Qwen3DFlash2DecoderLayer,
                block_size=config.block_size,
                conv_kernel_size=config.conv_kernel_size,
                conv_group_size=config.conv_group_size,
            ),
        )

    @property
    def candidate_selector(self) -> CandidateSelector:
        """Compatibility accessor; the module is registered only as correction."""
        assert isinstance(self.correction, CandidateSelector)  # noqa: S101
        return self.correction

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

    def _predecessor_ids(
        self,
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        anchored_block_indices: torch.Tensor,  # shape: [num_anchors*block_size]
    ) -> torch.Tensor:
        block_tokens = input_ids[0, anchored_block_indices].view(
            -1, self.block_size
        )  # shape: [num_anchors, block_size]
        if self.config.sample_from_anchor:
            return block_tokens
        # For token sequence 0 1 2 3 return previous token ids 0 0 1 2
        return torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
