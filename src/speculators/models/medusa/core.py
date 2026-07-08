"""Medusa speculator model implementation."""

import logging
from typing import Any, ClassVar

import torch
from torch import nn

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin
from speculators.models.medusa.config import MedusaSpeculatorConfig
from speculators.models.utils import conditional_torch_compile
from speculators.proposals.greedy import GreedyTokenProposalConfig

logger = logging.getLogger(__name__)

__all__ = ["MedusaDraftModel", "compute_head_weights"]

_IGNORE_INDEX = -100


def compute_head_weights(decay: float = 0.8, num_heads: int = 5) -> list[float]:
    """Compute exponential-decay head weights for Medusa.

    lambda_k = decay^k  (unnormalized, following the paper).
    """
    return [decay**k for k in range(num_heads)]


class ResidualBlock(nn.Module):
    """A single Medusa head block: Linear -> SiLU -> residual add.

    Matches the vLLM ResidualBlock architecture:
      x = x + SiLU(linear(x))  for each layer
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=bias) for _ in range(num_layers)]
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x


@SpeculatorModel.register("medusa")
class MedusaDraftModel(DraftVocabMixin, SpeculatorModel):
    """Medusa speculator model with multiple independent prediction heads.

    Each head is a ResidualBlock (MLP with SiLU + residual) that takes
    the verifier's last hidden state and predicts a future token at
    position t+k+1. All heads share the same input but predict different
    positions independently (no recursive hidden state passing).

    embed_tokens and lm_head are managed by DraftVocabMixin — initialized
    to NaN, populated via load_verifier_weights() (called automatically by
    from_pretrained), and excluded from saved checkpoints.
    """

    config_class: ClassVar[type[MedusaSpeculatorConfig]] = MedusaSpeculatorConfig  # type: ignore[misc]
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

    def _init_medusa_vocab(self, config: MedusaSpeculatorConfig) -> None:
        """Initialize vocab like DraftVocabMixin but without transformer_layer_config."""
        self.draft_vocab_size = config.draft_vocab_size
        self.verifier_vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.use_draft_vocab = self.draft_vocab_size != self.verifier_vocab_size

        t2d: torch.Tensor | None = None
        d2t: torch.Tensor | None = None
        if self.use_draft_vocab:
            t2d = torch.zeros((self.verifier_vocab_size,), dtype=torch.bool)
            d2t = torch.zeros((self.draft_vocab_size,), dtype=torch.long)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

        self.embed_tokens = nn.Embedding(self.verifier_vocab_size, self.hidden_size)
        self.embed_tokens.weight.requires_grad_(False)

        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)
        self.verifier_lm_head = nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        self.verifier_lm_head.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False

        torch.nn.init.constant_(self.lm_head.weight, torch.nan)
        torch.nn.init.constant_(self.embed_tokens.weight, torch.nan)
        torch.nn.init.constant_(self.verifier_lm_head.weight, torch.nan)
        self.lm_head._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001
        self.embed_tokens._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001
        self.verifier_lm_head._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001

    def __init__(self, config: MedusaSpeculatorConfig) -> None:
        super().__init__(config=config)
        self._init_medusa_vocab(config)

        hidden_size = config.hidden_size
        num_heads = config.num_heads
        num_hidden_layers = config.num_hidden_layers
        bias = config.medusa_fc_bias

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_size, num_hidden_layers, bias)
                for _ in range(num_heads)
            ]
        )

        if not config.original_lm_head:
            vocab_out = config.draft_vocab_size
            self.lm_heads = nn.ModuleList(
                [
                    nn.Linear(hidden_size, vocab_out, bias=False)
                    for _ in range(num_heads)
                ]
            )
        else:
            self.lm_heads = None

        self.post_init()

    @property
    def layers(self) -> nn.ModuleList:
        """Expose blocks for FSDP wrapping compatibility."""
        return self.blocks

    @property
    def target_layer_ids(self) -> list[int]:
        """Medusa only uses the last hidden layer."""
        return [0]

    def _get_lm_head(self, head_idx: int) -> nn.Module:
        if self.lm_heads is not None:
            return self.lm_heads[head_idx]
        return self.lm_head

    def load_verifier_weights(self) -> None:
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
        attention_mask: torch.Tensor | None = None,  # noqa: ARG002
        position_ids: torch.Tensor | None = None,  # noqa: ARG002
        loss_mask: torch.Tensor | None = None,
        head_weights: list[float] | None = None,
        return_dict: bool = True,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple:
        """Forward pass for Medusa multi-head prediction.

        Each head k independently predicts token at position t+k+1 from
        the verifier's hidden state h_t. Unlike MTP, there is no
        recursive hidden state passing between heads.

        :param input_ids: Token IDs [1, seq_len].
        :param hidden_states: Hidden states from verifier [1, seq_len, hidden_size]
        :param attention_mask: Unused (kept for interface compatibility).
        :param position_ids: Unused (kept for interface compatibility).
        :param loss_mask: Optional binary mask [1, seq_len]; 1=compute loss.
        :param head_weights: Per-head loss weights (None = uniform).
        :param return_dict: Unused.
        :param kwargs: Absorbs unexpected batch keys.
        :return: Tuple of (logits_list, loss, metrics)
        """
        input_ids = input_ids.long()
        device = input_ids.device
        seq_len = input_ids.shape[1]
        num_heads = self.config.num_heads

        if head_weights is not None and len(head_weights) != num_heads:
            raise ValueError(
                f"head_weights has {len(head_weights)} entries but "
                f"num_heads={num_heads}; expected exactly {num_heads} weights."
            )

        all_logits: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float | torch.Tensor] = {}

        effective_heads = min(num_heads, max(0, seq_len - 2))
        valid_len = seq_len - effective_heads - 1
        if valid_len <= 0 or effective_heads == 0:
            metrics["loss_sum"] = total_loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return (all_logits, total_loss, metrics)

        for head_idx in range(effective_heads):
            head_hidden = self.blocks[head_idx](hidden_states[:, :valid_len])
            lm_head = self._get_lm_head(head_idx)
            logits = lm_head(head_hidden)
            all_logits.append(logits)

            targets = input_ids[:, head_idx + 2 : head_idx + 2 + valid_len]
            if loss_mask is not None:
                step_mask = loss_mask[:, head_idx + 2 : head_idx + 2 + valid_len]
                targets = targets.clone()
                targets[step_mask == 0] = _IGNORE_INDEX
            weight = head_weights[head_idx] if head_weights is not None else 1.0
            unreduced = nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                targets,
                ignore_index=_IGNORE_INDEX,
                reduction="none",
            )
            valid_count = (targets != _IGNORE_INDEX).sum()
            head_loss = weight * unreduced.sum() / valid_count.clamp(min=1)
            total_loss = total_loss + head_loss
            metrics[f"loss_head_{head_idx}"] = head_loss.detach().clone()

        metrics["loss_sum"] = total_loss.detach().clone()
        metrics["loss_total"] = torch.tensor(1.0, device=device)

        return (all_logits, total_loss, metrics)

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: Any,
        *,
        verifier_name_or_path: str | None = None,
        num_heads: int = 5,
        num_layers: int = 1,
        head_weight_decay: float = 0.8,
        original_lm_head: bool = True,
        medusa_fc_bias: bool = False,
        **kwargs: Any,  # noqa: ARG003
    ) -> "MedusaDraftModel":
        if verifier_name_or_path is None:
            raise ValueError("verifier_name_or_path is required for Medusa training.")

        hidden_size = verifier_config.hidden_size
        vocab_size = verifier_config.vocab_size

        config = MedusaSpeculatorConfig(
            medusa_hidden_size=hidden_size,
            medusa_vocab_size=vocab_size,
            num_heads=num_heads,
            num_hidden_layers=num_layers,
            original_lm_head=original_lm_head,
            medusa_fc_bias=medusa_fc_bias,
            head_weight_decay=head_weight_decay,
            speculators_config=SpeculatorsConfig(
                algorithm="medusa",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=num_heads,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(verifier_name_or_path),
            ),
        )

        model = cls(config=config)
        model.load_verifier_weights()

        with torch.no_grad():
            for block in model.blocks:
                for layer in block.layers:
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for Medusa.

        Head weights use exponential decay: lambda_k = decay^k.
        """
        head_weights = kwargs.get("head_weights")
        if head_weights is None:
            num_heads = kwargs.get("num_heads", 5)
            decay = kwargs.get("head_weight_decay", 0.8)
            head_weights = compute_head_weights(decay=decay, num_heads=num_heads)
        train_kwargs: dict[str, Any] = {"head_weights": head_weights}
        val_kwargs = train_kwargs.copy()
        return train_kwargs, val_kwargs
