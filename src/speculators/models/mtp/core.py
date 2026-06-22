"""MTP speculator model implementation."""

import logging
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.masking_utils import create_causal_mask

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin
from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.models.mtp.model_definitions import (
    mtp_model_classes,
    resolve_model_type,
)
from speculators.models.utils import conditional_torch_compile
from speculators.proposals.greedy import GreedyTokenProposalConfig

logger = logging.getLogger(__name__)

__all__ = ["MTPDraftModel", "compute_step_weights"]

_IGNORE_INDEX = -100


def compute_step_weights(beta: float = 0.6, num_steps: int = 3) -> list[float]:
    """Compute normalized exponential-decay step weights.

    alpha_k = beta^(k-1) / sum(beta^(j-1) for j=1..K)

    See FastMTP (arXiv:2509.18362), Equation 2.
    """
    raw = [beta**k for k in range(num_steps)]
    total = sum(raw)
    return [w / total for w in raw]


@SpeculatorModel.register("mtp")
class MTPDraftModel(DraftVocabMixin, SpeculatorModel):
    """MTP speculator model for multi-token prediction.

    Predicts multiple future tokens (default: 3) per forward pass using
    a single layer with weighted multi-step loss for training.

    embed_tokens and lm_head are managed by DraftVocabMixin — initialized
    to NaN, populated via load_verifier_weights() (called automatically by
    from_pretrained), and excluded from saved checkpoints.
    verifier_lm_head is created by DraftVocabMixin but not used in
    the MTP forward pass.
    """

    config_class: ClassVar[type[MTPSpeculatorConfig]] = MTPSpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "embed_tokens.weight",
        "lm_head.weight",
    ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: MTPSpeculatorConfig) -> None:
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
        super().__init__(config=config)
        self._init_vocab(config)
        if self.use_draft_vocab:
            raise NotImplementedError(
                "Vocab reduction is not supported for MTP speculators"
            )

        tc = config.transformer_layer_config
        self._model_definitions = mtp_model_classes[resolve_model_type(tc.model_type)]
        self.mtp_layers = nn.ModuleList(
            [self._model_definitions.first_layer_class(tc, layer_idx=0)]
        )
        self.rotary_emb = self._model_definitions.rotary_emb_class(tc)

        self.post_init()

    @property
    def layers(self) -> nn.ModuleList:
        """Expose mtp_layers for FSDP wrapping compatibility."""
        return self.mtp_layers

    @property
    def target_layer_ids(self) -> list[int]:
        """MTP only uses the last hidden layer (verifier_last_hidden_states)."""
        return [self.config.transformer_layer_config.num_hidden_layers]

    def load_verifier_weights(self) -> None:
        """Re-set NaN sentinel before loading — meta-device init may clear
        it. Deletes verifier_lm_head after loading since MTP does not use it.
        """
        with torch.no_grad():
            self.embed_tokens.weight.fill_(torch.nan)
            self.lm_head.weight.fill_(torch.nan)
        super().load_verifier_weights()
        del self.verifier_lm_head

    @conditional_torch_compile
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

        Targets are derived from input_ids via per-step offset slicing -- no
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

        if step_weights is not None and len(step_weights) != num_steps:
            raise ValueError(
                f"step_weights has {len(step_weights)} entries but "
                f"num_speculative_steps={num_steps}; expected exactly "
                f"{num_steps} weights."
            )

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        all_logits: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float | torch.Tensor] = {}

        # Uniform valid_len keeps tensor shapes identical across loop
        # iterations, which torch.compile requires for stable codegen.
        # Cap steps so short sequences still produce partial results.
        effective_steps = min(num_steps, max(0, seq_len - 2))
        valid_len = seq_len - effective_steps - 1
        if valid_len <= 0 or effective_steps == 0:
            metrics["loss_sum"] = total_loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return (all_logits, total_loss, metrics)

        step_pos_ids = position_ids[:, :valid_len]
        causal_mask = create_causal_mask(
            config=self.config.transformer_layer_config,
            inputs_embeds=hidden_states[:, :valid_len],
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=step_pos_ids,
        )

        current_hidden = hidden_states
        for step in range(effective_steps):
            step_hidden = current_hidden[:, :valid_len]
            step_embeds = self.embed_tokens(
                input_ids[:, step + 1 : step + 1 + valid_len]
            )
            step_pos_emb = self.rotary_emb(step_hidden, step_pos_ids)

            mtp_output = self.mtp_layers[0](
                hidden_states=step_hidden,
                token_embeddings=step_embeds,
                attention_mask=causal_mask,
                position_ids=step_pos_ids,
                position_embeddings=step_pos_emb,
            )

            logits = self.lm_head(mtp_output)
            all_logits.append(logits)

            step_targets = input_ids[:, step + 2 : step + 2 + valid_len]
            if loss_mask is not None:
                step_mask = loss_mask[:, step + 2 : step + 2 + valid_len]
                step_targets = step_targets.clone()
                step_targets[step_mask == 0] = _IGNORE_INDEX
            weight = step_weights[step] if step_weights is not None else 1.0
            unreduced = nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                step_targets,
                ignore_index=_IGNORE_INDEX,
                reduction="none",
            )
            valid_count = (step_targets != _IGNORE_INDEX).sum()
            step_loss = weight * unreduced.sum() / valid_count.clamp(min=1)
            total_loss = total_loss + step_loss
            metrics[f"loss_step_{step}"] = step_loss.detach().clone()

            current_hidden = mtp_output

        metrics["loss_sum"] = total_loss.detach().clone()
        metrics["loss_total"] = torch.tensor(1.0, device=device)

        return (all_logits, total_loss, metrics)

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: PretrainedConfig,
        *,
        num_speculative_steps: int = 3,
        verifier_name_or_path: str | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> "MTPDraftModel":
        if verifier_name_or_path is None:
            raise ValueError(
                "verifier_name_or_path is required for MTP training. "
                "The verifier model must contain native MTP weights "
                "(mtp.* keys) to extract."
            )

        config = MTPSpeculatorConfig(
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

        model = cls(config=config)

        from speculators.convert.mtp.converter import MTPConverter  # noqa: PLC0415

        state_dict = MTPConverter().convert_to_state_dict(
            verifier_name_or_path  # type: ignore[arg-type]
        )
        model.load_state_dict(state_dict, strict=False)

        model.load_verifier_weights()
        return model

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
            if "num_speculative_steps" not in kwargs:
                raise ValueError(
                    "num_speculative_steps must be set from the model config "
                    "before calling get_trainer_kwargs"
                )
            step_weights = compute_step_weights(
                beta=kwargs.get("step_weight_beta", 0.6),
                num_steps=kwargs["num_speculative_steps"],
            )
        train_kwargs: dict[str, Any] = {"step_weights": step_weights}
        val_kwargs = train_kwargs.copy()

        return train_kwargs, val_kwargs
