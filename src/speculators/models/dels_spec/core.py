from typing import Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dels_spec.config import DelsSpecSpeculatorConfig
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import (
    LossConfig,
    compute_accuracy_multi_step,
    dflash_loss_decay,
)


def _ce_loss_against_token_ids(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    gamma: float = 7.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Cross-entropy loss against actual token IDs with DFlash-style decay.

    Args:
        logits: Draft logits [1, T, V].
        target_ids: Ground-truth token IDs [1, T].
        loss_mask: Binary mask [1, T].
        pos_idx: Position indices within blocks [1, T].
        gamma: Decay rate for DFlash-style position weighting.

    Returns:
        (scalar_loss, metrics_dict)
    """
    batch_size, seq_len, vocab_size = logits.shape

    elementwise_loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)

    loss_mask_f = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask_f

    decay_mult = dflash_loss_decay(pos_idx.to(elementwise_loss.dtype), gamma=gamma)
    elementwise_loss = elementwise_loss * decay_mult

    denominator = loss_mask_f.sum(dim=1).clamp_min(1e-5)
    loss = (elementwise_loss.sum(dim=1) / denominator).mean()

    block_size = pos_idx.max().item() + 1
    pred_ids = torch.argmax(logits, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, int(block_size)
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    metrics["full_acc_sum"] = correct_per_pos[1:].sum()
    metrics["full_acc_total"] = total_per_pos[1:].sum()
    for pos in range(1, int(block_size)):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    return loss, metrics


@SpeculatorModel.register("dels_spec")
class DelsSpecDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DelsSpecSpeculatorConfig]] = DelsSpecSpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = []  # type: ignore[misc,assignment]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: DelsSpecSpeculatorConfig) -> None:
        super().__init__(config=config)
        self._init_vocab(config)

        embed_dim = config.transformer_layer_config.hidden_size
        self.block_size = config.block_size
        self.head_type = config.head_type

        self.embed_tokens.weight.requires_grad = False

        if config.head_type == "rnn":
            self.gru = nn.GRU(
                input_size=embed_dim,
                hidden_size=config.gru_hidden_size,
                num_layers=1,
                bias=False,
                batch_first=True,
            )
            proj_input_dim = config.gru_hidden_size
        else:
            self.markov_emb = nn.Embedding(
                config.transformer_layer_config.vocab_size, config.low_rank_dim
            )
            proj_input_dim = config.low_rank_dim

        self.w_rank = nn.Linear(proj_input_dim, config.low_rank_dim, bias=False)
        self.w_vocab = nn.Linear(
            config.low_rank_dim, config.draft_vocab_size, bias=False
        )

        self.layers = nn.ModuleList()

        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        """No auxiliary hidden states needed."""
        return []

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        document_ids: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        gamma: float = 7.0,
        max_anchors: int = 3072,
        **kwargs,
    ):
        del hidden_states, document_ids, verifier_last_hidden_states
        del position_ids, loss_config, kwargs
        device = input_ids.device

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size
        )

        block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        block_token_ids = input_ids[:, block_indices]

        target_token_ids = input_ids[:, block_indices + 1]

        with torch.no_grad():
            block_embeds = self.embed_tokens(block_token_ids)

        num_anchors = anchor_positions.shape[0]
        block_embeds_3d = block_embeds.view(
            1, num_anchors, self.block_size, -1
        ).squeeze(0)

        if self.head_type == "rnn":
            gru_out, _ = self.gru(block_embeds_3d)
            gru_out = gru_out.reshape(1, num_anchors * self.block_size, -1)
            proj = self.w_rank(gru_out)
        else:
            markov_input = self.markov_emb(block_token_ids)
            proj = markov_input

        logits = self.w_vocab(proj)

        aligned_loss_mask = loss_mask.clone()[:, block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        aligned_loss_mask[:, :: self.block_size] = 0

        seq_len = logits.shape[1]
        pos_idx = (torch.arange(seq_len, device=device) % self.block_size).unsqueeze(0)

        if self.t2d is not None:
            target_token_ids = target_token_ids + self.t2d[target_token_ids]

        loss, metrics = _ce_loss_against_token_ids(
            logits, target_token_ids, aligned_loss_mask, pos_idx, gamma=gamma
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DelsSpecDraftModel":
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        block_size = kwargs.get("block_size", 16)

        config = DelsSpecSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=block_size,
            gru_hidden_size=kwargs.get("gru_hidden_size", 1024),
            low_rank_dim=kwargs.get("low_rank_dim", 256),
            head_type=kwargs.get("dels_spec_head_type", "rnn"),
            fusion_alpha=kwargs.get("fusion_alpha", 0.3),
            fusion_beta=kwargs.get("fusion_beta", 0.3),
            mask_token_id=kwargs.get("mask_token_id"),
            speculators_config=SpeculatorsConfig(
                algorithm="dels_spec",
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    def load_verifier_weights(self):
        super().load_verifier_weights()
        self.embed_tokens.weight.requires_grad_(False)

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        gamma = kwargs.get("dflash_decay_gamma", 7.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        shared = {
            "gamma": gamma,
            "max_anchors": max_anchors,
        }
        return dict(shared), dict(shared)
