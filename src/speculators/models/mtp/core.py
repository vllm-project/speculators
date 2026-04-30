"""MTP speculator model implementation."""

from typing import Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp.config import MTPConfig
from speculators.models.mtp.model_definitions import (
    mtp_model_classes,
    resolve_model_type,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers

__all__ = ["MTPDraftModel", "compute_step_weights"]


def compute_step_weights(beta: float = 0.6, num_steps: int = 3) -> list[float]:
    """Compute normalized exponential-decay step weights.

    alpha_k = beta^(k-1) / sum(beta^(j-1) for j=1..K)

    See FastMTP (arXiv:2509.18362), Equation 2.
    """
    raw = [beta**k for k in range(num_steps)]
    total = sum(raw)
    return [w / total for w in raw]


@SpeculatorModel.register("mtp")
class MTPDraftModel(SpeculatorModel):
    """MTP speculator model for multi-token prediction.

    Predicts multiple future tokens (default: 3) per forward pass using
    a single layer with weighted multi-step loss for training.

    embed_tokens and lm_head are frozen copies of the verifier's weights,
    loaded at init time via _setup_embeddings_and_lm_head and saved with
    the checkpoint. They are not stitched back during weight merging.
    """

    config_class: ClassVar[type[MTPConfig]] = MTPConfig  # type: ignore[misc]

    def __init__(self, config: MTPConfig) -> None:
        super().__init__(config=config)
        tc = config.transformer_layer_config
        self._model_definitions = mtp_model_classes[resolve_model_type(tc.model_type)]
        self.mtp_layers = nn.ModuleList(
            [self._model_definitions.first_layer_class(tc, layer_idx=0)]
        )
        self.rotary_emb = self._model_definitions.rotary_emb_class(tc)
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self._setup_embeddings_and_lm_head()
        self.post_init()

    def _setup_embeddings_and_lm_head(self) -> None:
        if (
            self.config.speculators_config is None
            or self.config.speculators_config.verifier is None
            or self.config.speculators_config.verifier.name_or_path is None
        ):
            return

        path = self.config.speculators_config.verifier.name_or_path
        weights = load_model_layers(["embed_tokens.weight", "lm_head.weight"], path)

        embed_weight = weights["embed_tokens.weight"]
        lm_head_weight = weights.get("lm_head.weight", embed_weight)

        self.embed_tokens.weight = nn.Parameter(
            embed_weight.detach().clone(), requires_grad=False
        )
        self.lm_head.weight = nn.Parameter(
            lm_head_weight.detach().clone(), requires_grad=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        step_weights: list[float] | None = None,
        return_dict: bool = True,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple:
        """Forward pass for MTP multi-token prediction (teacher-forced).

        At step k, uses ground-truth input_ids[t+k+1] as the embedding input and
        the MTP output from step k-1 (or verifier hidden states for step 0) as the
        hidden state input. Hidden states are passed recursively: each step's MTP
        output feeds the next step.

        Targets are derived from input_ids via per-step offset slicing — no
        separate label tensor is needed. Use loss_mask to exclude positions
        (e.g. prompt tokens) from the loss.

        :param input_ids: Token IDs [batch, seq_len]. Serves as both the
            embedding source and the prediction target (offset by step+2).
        :param hidden_states: Hidden states from verifier [batch, seq_len, hidden_size]
        :param attention_mask: Optional attention mask [batch, seq_len]
        :param position_ids: Optional position IDs [batch, seq_len]
        :param loss_mask: Optional binary mask [batch, seq_len]; 1=compute loss,
            0=ignore.
        :param step_weights: Per-step loss weights (None = uniform). Training only.
        :param return_dict: Unused, kept for interface compatibility.
        :param kwargs: Absorbs unexpected batch keys
            (lengths, verifier_last_hidden_states)
        :return: Tuple of (logits_list, loss, metrics)
        """
        input_ids = input_ids.long()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        num_steps = self.config.num_speculative_steps

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        all_logits: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float] = {}

        current_hidden = hidden_states
        for step in range(num_steps):
            valid_len = seq_len - step - 2
            if valid_len <= 0:
                break
            step_hidden = current_hidden[:, :valid_len]
            step_embeds = self.embed_tokens(
                input_ids[:, step + 1 : step + 1 + valid_len]
            )
            step_pos_ids = position_ids[:, :valid_len]
            step_pos_emb = self.rotary_emb(step_hidden, step_pos_ids)
            step_attn_mask = (
                attention_mask[:, :valid_len] if attention_mask is not None else None
            )

            mtp_output = self.mtp_layers[0](
                hidden_states=step_hidden,
                token_embeddings=step_embeds,
                attention_mask=step_attn_mask,
                position_ids=step_pos_ids,
                position_embeddings=step_pos_emb,
            )

            logits = self.lm_head(mtp_output)
            all_logits.append(logits)

            step_targets = input_ids[:, step + 2 : step + 2 + valid_len]
            if loss_mask is not None:
                step_mask = loss_mask[:, step + 2 : step + 2 + valid_len]
                step_targets = step_targets.clone()
                step_targets[step_mask == 0] = -100
            weight = step_weights[step] if step_weights is not None else 1.0
            step_loss = weight * nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                step_targets.reshape(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + step_loss
            metrics[f"loss_step_{step}"] = step_loss.item()

            current_hidden = mtp_output

        return (all_logits, total_loss, metrics)

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: PretrainedConfig,
        *,
        num_speculative_steps: int = 3,
        verifier_name_or_path: str | None = None,
    ) -> "MTPDraftModel":
        config = MTPConfig(
            transformer_layer_config=verifier_config,
            speculators_config=SpeculatorsConfig(
                algorithm="mtp",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=num_speculative_steps,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config,
                    name_or_path=verifier_name_or_path,
                ),
            ),
        )

        return cls(config=config)

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for MTP.

        Step weights are computed from ``step_weight_beta`` and
        ``num_speculative_steps`` using the normalized exponential-decay
        formula from FastMTP (arXiv:2509.18362), Equation 2.

        Pass ``step_weights`` to override the computed weights.
        """
        step_weights = kwargs.get("step_weights")
        if step_weights is None:
            step_weights = compute_step_weights(
                beta=kwargs.get("step_weight_beta", 0.6),
                num_steps=kwargs.get("num_speculative_steps", 3),
            )
        train_kwargs: dict[str, Any] = {"step_weights": step_weights}
        val_kwargs = train_kwargs.copy()

        return train_kwargs, val_kwargs
