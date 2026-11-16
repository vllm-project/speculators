from collections.abc import Callable
from typing import Any, ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig, kl_div_loss, resolve_loss_config, tv_loss
from speculators.model import SpeculatorModel
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.chunked_selector import (
    chunked_gather_values,
    chunked_topk_ids,
)
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import (
    compute_metrics,
    compute_metrics_from_hidden,
    selector_training_candidates,
)
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "DFlash2DraftModel",
]

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


@SpeculatorModel.register("dflash2")
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash with local convolution and bilinear candidate reranking."""

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

        for layer_ in self.layers:
            # By capability, so a _make_decoder_layer override still works.
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
        self.candidate_selector = CandidateSelector(
            vocab_size=target_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
            initializer_range=initializer_range,
        )

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> Qwen3DFlash2DecoderLayer:
        assert isinstance(config, DFlash2SpeculatorConfig)  # noqa: S101
        return Qwen3DFlash2DecoderLayer(
            config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
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
        # fused_linear computes the same unary objective from hidden states;
        # the config is resolved as "fused" and forward takes the hidden-state
        # path (see DFlashDraftModel.get_trainer_kwargs, same convention).
        fused_linear = implementation == "fused_linear"
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], "fused" if fused_linear else implementation
        )
        tv_loss_fn = resolve_loss_config(
            "tv", "fused" if fused_linear else implementation
        )["tv"][0]
        shared = {
            "loss_config": loss_config,
            "tv_loss_fn": tv_loss_fn,
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "selector_loss_alpha": kwargs.get("selector_loss_alpha", 1.0),
            "selector_miss_policy": kwargs.get("selector_miss_policy", "replace"),
            "fused_linear": fused_linear,
            "fused_linear_targets": kwargs.get("fused_linear_targets", "recompute"),
            "fused_linear_logits": kwargs.get("fused_linear_logits", "input"),
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

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 4.0,
        max_anchors: int = 512,
        selector_loss_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        selector_miss_policy: str = "replace",
        fused_linear: bool = False,
        fused_linear_targets: str = "recompute",
        fused_linear_logits: str = "input",
        **kwargs,
    ) -> tuple[None, torch.Tensor, dict[str, Any]]:
        recompute_targets = fused_linear and fused_linear_targets == "recompute"
        hidden, unary_logits, targets, aligned_loss_mask, block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                materialize_logits=not fused_linear,
                materialize_targets=not recompute_targets,
                max_anchors=max_anchors,
                **kwargs,
            )
        )
        predecessor_ids = self._predecessor_ids(input_ids, block_indices)
        top_k = self.candidate_selector.top_k

        if fused_linear:
            # loss_implementation="fused_linear": no [1, T, V] unary logits,
            # and with recomputed targets no [1, T, V] target logits either.
            # See chunked_selector for why this needs more than the scalar
            # losses/argmax fused_linear_compound already returns: the
            # candidate selector gathers real logit VALUES at a given
            # (post-injection) id set, not just the top-1 argmax.
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            targets_flat = targets.reshape(-1, targets.shape[-1])
            target_weight = self.verifier_lm_head.weight if recompute_targets else None

            # Cheap top-1 pass ahead of the real (full) unary loss below,
            # which recomputes target_ids as a harmless side effect -- see
            # the RFC notes on this tradeoff (duplicates the target
            # projection, not the full loss math).
            target_ids = (
                chunked_topk_ids(targets_flat, target_weight, top_k=1).squeeze(-1)
                if target_weight is not None
                else targets_flat.argmax(dim=-1)
            ).unsqueeze(0)

            candidate_ids = chunked_topk_ids(
                hidden_flat, self.lm_head.weight, top_k
            ).unsqueeze(0)
            training_candidate_ids, target_positions, contains_target = (
                selector_training_candidates(
                    candidate_ids, target_ids, miss_policy=selector_miss_policy
                )
            )

            unary_scores = chunked_gather_values(
                hidden_flat,
                self.lm_head.weight,
                training_candidate_ids.reshape(-1, top_k),
            ).reshape(1, -1, top_k)
            candidate_logits = self.candidate_selector.score_candidates_from_values(
                unary_scores,
                hidden,
                predecessor_ids.reshape(1, -1),
                training_candidate_ids,
            )

            loss, metrics = compute_metrics_from_hidden(
                hidden=hidden,
                lm_head_weight=self.lm_head.weight,
                targets=targets,
                training_candidate_ids=training_candidate_ids,
                candidate_logits=candidate_logits,
                target_positions=target_positions,
                contains_target=contains_target,
                loss_mask=aligned_loss_mask,
                block_size=self.block_size,
                top_k=top_k,
                sample_from_anchor=self.config.sample_from_anchor,
                loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
                gamma=gamma,
                selector_loss_alpha=selector_loss_alpha,
                per_position_loss_weight=per_position_loss_weight,
                dpace_alpha=dpace_alpha,
                selector_miss_policy=selector_miss_policy,
                target_weight=target_weight,
                logits_dtype=fused_linear_logits,
            )
            return None, loss, metrics

        target_ids = targets.argmax(dim=-1)
        # shape: [1, num_anchors*block_size]
        candidate_ids = unary_logits.topk(top_k, dim=-1).indices
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

        loss, metrics = compute_metrics(
            unary_logits=unary_logits,
            targets=targets,
            training_candidate_ids=training_candidate_ids,
            candidate_logits=candidate_logits,
            target_positions=target_positions,
            contains_target=contains_target,
            loss_mask=aligned_loss_mask,
            block_size=self.block_size,
            top_k=top_k,
            sample_from_anchor=self.config.sample_from_anchor,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            selector_loss_alpha=selector_loss_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            selector_miss_policy=selector_miss_policy,
        )
        return None, loss, metrics
