from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dels_spec.config import DeLSSpecSpeculatorConfig
from speculators.models.dels_spec.metrics import compute_metrics
from speculators.models.dels_spec.model_definitions import MarkovLocalHead, RNNLocalHead
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.utils import conditional_torch_compile
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = [
    "DeLSSpecDraftModel",
]


@SpeculatorModel.register("dels_spec")
class DeLSSpecDraftModel(DraftVocabMixin, SpeculatorModel):
    """DeLS-Spec local head: a lightweight short-context expert.

    The local head is a GRU (or Markov) model that captures intra-block
    causal dependencies.  It trains independently with next-token prediction
    and its logits are fused with DFlash at inference via a product-of-experts
    formula.
    """

    config_class: ClassVar[type[DeLSSpecSpeculatorConfig]] = DeLSSpecSpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: DeLSSpecSpeculatorConfig) -> None:
        super().__init__(config=config)
        self._init_vocab(config)

        embed_dim = config.transformer_layer_config.hidden_size
        self.block_size = config.block_size

        if config.head_type == "rnn":
            self.local_head: RNNLocalHead | MarkovLocalHead = RNNLocalHead(
                embed_dim=embed_dim,
                gru_hidden_size=config.gru_hidden_size,
                low_rank_dim=config.low_rank_dim,
                vocab_size=self.draft_vocab_size,
            )
        else:
            self.local_head = MarkovLocalHead(
                vocab_size=self.verifier_vocab_size,
                low_rank_dim=config.low_rank_dim,
            )

        # Empty ModuleList satisfies verify_training_compatible's `layers` check.
        # For multi-GPU FSDP, the full model (including local_head) is sharded as
        # one unit rather than layer-by-layer.
        self.layers = nn.ModuleList()

        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        """No auxiliary hidden states needed for the local head."""
        return []

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DeLSSpecDraftModel":
        block_size = kwargs.get("block_size", 16)
        config = DeLSSpecSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=block_size,
            gru_hidden_size=kwargs.get("gru_hidden_size", 1024),
            low_rank_dim=kwargs.get("low_rank_dim", 256),
            head_type=kwargs.get("dels_spec_head_type", "rnn"),
            fusion_alpha=kwargs.get("fusion_alpha", 0.3),
            fusion_beta=kwargs.get("fusion_beta", 0.3),
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

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        gamma = kwargs.get("dflash_decay_gamma", 7.0)
        max_anchors = kwargs.get("max_anchors", 512)
        shared: dict = {
            "gamma": gamma,
            "max_anchors": max_anchors,
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, T, num_hidden*H] — not used
        input_ids: torch.Tensor,  # [1, T]
        loss_mask: torch.Tensor,  # [1, T]
        verifier_last_hidden_states: torch.Tensor,  # [1, T, H] — not used
        document_ids: torch.Tensor,  # [1, T] — not used
        position_ids: torch.Tensor | None = None,
        gamma: float = 7.0,
        max_anchors: int = 512,
        **kwargs,
    ):
        del hidden_states, verifier_last_hidden_states, document_ids
        del position_ids, kwargs
        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size
        )

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # [num_anchors * block_size]

        num_blocks = max_anchors
        block = self.block_size
        block_tokens = input_ids[0, anchored_block_indices].view(
            num_blocks, block
        )  # [num_blocks, block_size]

        if isinstance(self.local_head, RNNLocalHead):
            block_embeddings = self.embed_tokens(block_tokens)
            all_logits = self.local_head(block_embeddings)
        else:
            prev_tokens = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
            all_logits = self.local_head(prev_tokens)
        # all_logits: [num_blocks, block_size, vocab]
        # GRU output at position j (having seen tokens 0..j) predicts token j+1.
        # aligned_logits[j%block==k] = all_logits[k-1], so position k in each
        # block holds the logits for predicting token at position k. Position 0
        # is a zero placeholder (anchor position, masked out of loss).
        # Build the logits tensor aligned with anchored blocks:
        # Position j in each block → logits predicting token at position j
        # = all_logits[:, j-1] for j >= 1, garbage for j == 0 (masked out)
        aligned_logits = torch.cat(
            [
                all_logits.new_zeros(num_blocks, 1, all_logits.shape[-1]),
                all_logits[:, :-1, :],
            ],
            dim=1,
        ).reshape(1, num_blocks * block, -1)
        # aligned_logits[0, j] for j%block==0 is zero (will be masked)
        # aligned_logits[0, j] for j%block==k (k>=1) = all_logits[block_idx, k-1]
        #   which predicts token at position k

        # Targets: actual token IDs at each position in the block
        target_ids = block_tokens.reshape(1, num_blocks * block)

        # Loss mask: zero out position 0 of each block (anchor, no prediction)
        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(block)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        aligned_loss_mask[:, ::block] = 0

        loss, metrics = compute_metrics(
            aligned_logits,
            target_ids,
            aligned_loss_mask,
            block_size=block,
            gamma=gamma,
        )

        draft_tokens = torch.argmax(aligned_logits, dim=-1)
        return draft_tokens, loss, metrics
