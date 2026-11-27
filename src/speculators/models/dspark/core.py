from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig, kl_div_loss, resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.components import BlockLayout
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import (
    compute_metrics,
    compute_metrics_from_hidden,
)
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        implementation = kwargs.get("loss_implementation", "fused")
        # fused_linear computes the fused losses from hidden states.
        fused_linear = implementation == "fused_linear"
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], "fused" if fused_linear else implementation
        )
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
            "fused_linear": fused_linear,
            "fused_linear_logits": kwargs.get("fused_linear_logits", "input"),
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
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        fused_linear: bool = False,
        **kwargs,
    ):
        if fused_linear and self.markov_head is None:
            raise ValueError(
                "DSpark fused_linear needs a Markov head (markov_rank > 0)"
            )
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
            fused_linear=fused_linear,
            correction_kwargs={"fused_linear": fused_linear},
            loss_kwargs={"confidence_head_alpha": confidence_head_alpha},
            **kwargs,
        )

    def apply_training_correction(
        self,
        hidden,
        logits,
        _targets,
        layout,
        input_ids,
        _document_ids,
        fused_linear=False,
        **_kwargs,
    ):
        # DSpark: add the Markov logit bias and predict per-position confidence.
        # With fused_linear there are no logits: return markov_w2's input u,
        # which the loss adds inside its GEMMs.
        num_blocks = layout.anchor_positions.numel()
        block = self.block_size
        mask_tokens_size = num_blocks * block
        prev_token_ids = layout.predecessor_ids(input_ids)
        hidden_blocks = hidden.view(num_blocks, block, -1)
        prev_emb = None
        markov_features = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            features = self.markov_head.block_features(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            if fused_linear:
                markov_features = features.reshape(1, mask_tokens_size, -1)
            else:
                bias = self.markov_head.markov_w2(features)
                logits = (logits.view(num_blocks, block, -1) + bias).reshape_as(logits)

        confidence_logits = None

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)],
                    dim=-1,
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )
        return logits, confidence_logits, markov_features

    def compute_loss_and_metrics(
        self,
        logits,
        targets,
        layout: BlockLayout,
        correction,
        *,
        fused_linear_logits: str = "input",
        loss_config: LossConfig | None,
        gamma: float,
        per_position_loss_weight: str,
        dpace_alpha: float,
        confidence_head_alpha=1.0,
        **_kwargs,
    ):
        logits, confidence_logits, markov_features = correction
        metric_kwargs = {
            "loss_config": loss_config or _DEFAULT_LOSS_CONFIG,
            "gamma": gamma,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
            "sample_from_anchor": self.config.sample_from_anchor,
        }
        if markov_features is not None and self.markov_head is not None:
            # fused_linear: no [1, T, V] draft logits, Markov bias or target
            # logits. With the full vocab lm_head is the verifier head, which
            # enables the shared-head delta form.
            head = self.lm_head if self.use_draft_vocab else self.verifier_lm_head
            return compute_metrics_from_hidden(
                logits,  # backbone hidden states in fused-linear mode
                head.weight,
                targets,  # post-norm verifier hidden states (project=False)
                self.verifier_lm_head.weight,
                markov_features,
                self.markov_head.markov_w2.weight,
                confidence_logits,
                layout.loss_mask,
                self.block_size,
                logits_dtype=fused_linear_logits,
                **metric_kwargs,
            )
        return compute_metrics(
            logits, targets, confidence_logits, layout.loss_mask, self.block_size,
            **metric_kwargs,
        )  # fmt: skip
