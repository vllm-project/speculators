"""FastMTP speculator model implementation."""

from typing import Any, ClassVar  # noqa: F401 (Any used in forward **kwargs type)

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp.config import FastMTPConfig
from speculators.models.fast_mtp.model_definitions import fast_mtp_model_classes
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers

__all__ = ["FastMTPSpeculator"]


@SpeculatorModel.register("mtp")
class FastMTPSpeculator(SpeculatorModel):
    """FastMTP speculator model for multi-token prediction.

    FastMTP predicts multiple future tokens (default: 3) per forward pass using
    a single shared MTP layer applied recursively with weighted multi-step loss.

    The single MTP layer is applied K times with different token embeddings and
    the recursively updated hidden state. This matches the paper's design and keeps
    parameter count low while remaining compatible with vLLM (which reads
    ``num_nextn_predict_layers=1``).

    embed_tokens and lm_head are always initialized with random weights in __init__.
    When speculators_config.verifier.name_or_path is set, _setup_embeddings_and_lm_head
    overwrites them with weights from the verifier checkpoint. When loading a
    self-contained checkpoint (embed_tokens + lm_head weights present in the file),
    from_pretrained fills them directly.
    """

    config_class: ClassVar[type[FastMTPConfig]] = FastMTPConfig  # type: ignore[misc]

    def __init__(self, config: FastMTPConfig) -> None:
        super().__init__(config=config)
        tc = config.transformer_layer_config
        self._model_definitions = fast_mtp_model_classes[tc.model_type]
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

    def _setup_embeddings_and_lm_head(self) -> None:
        """Overwrite embed_tokens and lm_head from the verifier if configured."""
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
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        step_weights: list[float] | None = None,
        lengths: torch.Tensor | None = None,  # noqa: ARG002 — collation metadata
        **_kwargs: Any,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, dict[str, float]]:
        """Forward pass for FastMTP multi-token prediction (teacher-forced).

        Returns a 3-tuple ``(logits_list, loss, metrics)`` compatible with the
        speculators ``Trainer`` (which unpacks ``_draft_tokens, loss, metrics``).

        At step k, uses ground-truth ``input_ids[t+k+1]`` as the embedding input and
        the MTP output from step k-1 (or verifier hidden states for step 0) as the
        hidden state input. Hidden states are passed recursively: each step's MTP
        output feeds the next step, matching the paper's training procedure.

        When ``labels`` is ``None``, ``input_ids`` is used as labels — this is correct
        because :func:`~speculators.train.fast_mtp_data._shift_batch_fastmtp` has
        already aligned ``input_ids[i] = x_{i+1}``, so ``input_ids`` is the right
        supervision signal.

        :param input_ids: Token IDs [batch, seq_len]
        :param hidden_states: Hidden states from verifier [batch, seq_len, hidden_size]
        :param attention_mask: Optional attention mask [batch, seq_len]
        :param position_ids: Optional position IDs [batch, seq_len]
        :param labels: Ground truth labels [batch, seq_len]; defaults to input_ids
        :param loss_mask: Optional binary mask [batch, seq_len]; 1=compute loss,
            0=ignore. Positions with mask==0 have their label set to -100 so the
            cross-entropy ignores them. Aligned with labels using the same step+2
            offset. Training only.
        :param step_weights: Per-step loss weights (None = uniform). Training only.
        :param lengths: Sequence lengths from collation — unused in forward, accepted
            to absorb the batch field without error.
        :param kwargs: Additional batch fields forwarded by the Trainer (ignored).
        :return: ``(logits_list, loss, metrics)``
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        num_steps = self.config.num_speculative_steps

        # Default: use input_ids as labels (already shifted by _shift_batch_fastmtp)
        if labels is None:
            labels = input_ids

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        all_logits: list[torch.Tensor] = []
        total_loss: torch.Tensor = torch.tensor(0.0, device=device)
        metrics: dict[str, float] = {}

        current_hidden = hidden_states  # recursive: updated each step with MTP output
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

            step_labels = labels[:, step + 2 : step + 2 + valid_len]
            if loss_mask is not None:
                step_mask = loss_mask[:, step + 2 : step + 2 + valid_len]
                step_labels = step_labels.clone()
                step_labels[step_mask == 0] = -100
            weight = step_weights[step] if step_weights is not None else 1.0
            step_loss = weight * nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                step_labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + step_loss
            metrics[f"loss_step_{step}"] = step_loss.item()

            current_hidden = mtp_output  # feed MTP output as hidden for next step

        return (all_logits, total_loss, metrics)

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: PretrainedConfig,
        *,
        num_speculative_steps: int = 3,
        verifier_name_or_path: str | None = None,
    ) -> "FastMTPSpeculator":
        """Create FastMTP model from training arguments.

        :param verifier_config: Verifier model configuration
        :param num_speculative_steps: Number of future tokens to predict per step
        :param verifier_name_or_path: Path or repo ID for loading embed/lm_head weights
        :return: FastMTP model instance
        """
        config = FastMTPConfig(
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
        """Get training and validation kwargs for FastMTP.

        Pass ``step_weights`` to override the default exponential-decay weights
        ``[0.51, 0.31, 0.18]`` (β=0.6, normalized, matching Qwen3-Next defaults).

        :param kwargs: Training arguments
        :return: Tuple of (train_kwargs, val_kwargs)
        """
        train_kwargs = {
            "step_weights": kwargs.get("step_weights", [0.51, 0.31, 0.18]),
        }
        val_kwargs = train_kwargs.copy()

        return train_kwargs, val_kwargs
