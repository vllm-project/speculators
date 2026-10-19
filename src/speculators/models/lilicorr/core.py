from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from transformers import PretrainedConfig

from speculators.losses import LossConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.lilicorr.config import LiLiCorrSpeculatorConfig
from speculators.models.lilicorr.model_definitions import LiLiCorrHead
from speculators.models.metrics import compute_accepted_length_counts
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "LiLiCorrDraftModel",
    "lilicorr_lattice_terms",
]


def lilicorr_lattice_terms(
    head: LiLiCorrHead,
    candidate_ids: torch.Tensor,  # [N, S, K], descending rank order
    candidate_log_probs: torch.Tensor,  # [N, S, K], descending rank order
    token_embeddings: torch.Tensor,  # [N, S, K, H_model], detached
    pass_hidden: torch.Tensor,  # [N, S, H_model]
    anchor_hidden: torch.Tensor,  # [N, H_model]
    anchor_valid: torch.Tensor,  # [N]
    targets_s: torch.Tensor,  # [N, S, V] aligned teacher logits for the slots
    valid_block: torch.Tensor,  # [N]
    slot_mask: torch.Tensor,  # [N, S] bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Score the candidate lattice and return its loss terms and metrics.

    Shared by the standalone LiLiCorr model and the mamba_attn_hybrid +
    LiLiCorr combination. Everything backbone-specific (candidate ids and their
    log-probs, pass hidden states, the anchor context row) arrives as explicit
    tensors; this helper only runs the head and the objective:

    - teacher-forced transitions (slot 0 against the anchor, slot s > 0 leaving
      the candidate that held the ground truth at slot s-1),
    - supervision only through the contiguous prefix whose teacher-argmax
      tokens occur in the lattice, on blocks whose slots are all supervised and
      within one document,
    - the distractor penalty, and the greedy walk / EAL metrics.

    ``candidate_log_probs`` and ``pass_hidden`` stay attached to the caller's
    graph so the lattice terms reach the backbone. Returns ``(ce_loss,
    pen_loss, metrics, selected_ids)``; a batch with no eligible lattice
    positions yields finite zero losses (``denominators`` clamp to 1).
    """
    topk = head.candidate_topk
    log_start, log_pair = head(
        token_embeddings,
        candidate_log_probs,
        pass_hidden,
        anchor_hidden,
        anchor_valid,
    )

    with torch.no_grad():
        gt_ids = targets_s.argmax(-1)  # [N, S]
        gt_hits = candidate_ids == gt_ids.unsqueeze(-1)
        gt_in_lattice = gt_hits.any(-1)
        gt_index = gt_hits.float().argmax(-1)  # [N, S]

    # Teacher-forced node potentials: slot 0 against the anchor, slot s > 0
    # leaving the candidate that held the ground truth at slot s-1.
    prev_gt = gt_index[:, :-1, None, None].expand(-1, -1, 1, topk)
    nodes = torch.cat(
        [log_start.unsqueeze(1), log_pair.gather(2, prev_gt).squeeze(2)], dim=1
    )  # [N, S, K]
    log_p = F.log_softmax(nodes, dim=-1)

    supervised = (
        gt_in_lattice.long().cumprod(-1).bool() & valid_block[:, None]
    ).float()
    denom = supervised.sum().clamp_min(1.0)
    ce = -log_p.gather(-1, gt_index.unsqueeze(-1)).squeeze(-1)
    ce_loss = (ce * supervised).sum() / denom

    # Distractor penalty: the head's mass on each candidate, weighted by how far
    # below the ground truth the target's logit puts it.
    cand_target_logits = targets_s.gather(-1, candidate_ids).float()
    gt_target_logit = cand_target_logits.gather(-1, gt_index.unsqueeze(-1))
    rejection = (gt_target_logit - cand_target_logits).clamp_min(0.0)
    pen_loss = ((log_p.exp() * rejection).sum(-1) * supervised).sum() / denom

    metrics: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        # Greedy left-to-right walk, as served.
        current = log_start.argmax(-1)
        picks = [current]
        for slot in range(head.num_slots - 1):
            row = log_pair[:, slot].gather(
                1, current[:, None, None].expand(-1, 1, topk)
            )
            current = row.squeeze(1).argmax(-1)
            picks.append(current)
        selected_ids = candidate_ids.gather(
            -1, torch.stack(picks, dim=1).unsqueeze(-1)
        ).squeeze(-1)
        eval_mask = slot_mask & valid_block[:, None]
        ones = torch.ones((), device=ce_loss.device)
        metrics["lilicorr_ce_sum"] = ce_loss.detach()
        metrics["lilicorr_ce_total"] = ones.clone()
        metrics["lilicorr_penalty_sum"] = pen_loss.detach()
        metrics["lilicorr_penalty_total"] = ones.clone()
        # Same blocks for all three, so lilicorr_eal - lilicorr_top1_eal is the
        # head's gain and lilicorr_oracle_eal bounds what any walk could reach.
        for name, correct in (
            ("lilicorr_eal", selected_ids == gt_ids),
            ("lilicorr_top1_eal", candidate_ids[..., 0] == gt_ids),
            ("lilicorr_oracle_eal", gt_in_lattice),
        ):
            eal_sum, eal_total = compute_accepted_length_counts(correct, eval_mask)
            metrics[f"{name}_sum"] = eal_sum
            metrics[f"{name}_total"] = eal_total

    return ce_loss, pen_loss, metrics, selected_ids


@SpeculatorModel.register("lilicorr")
class LiLiCorrDraftModel(DFlashDraftModel):
    """DFlash backbone plus the LiLiCorr candidate-lattice head.

    The head scores the backbone's per-slot top-k candidates and serving commits
    a path with a greedy left-to-right walk. Training adds the paper's ``base``
    objective to the DFlash loss::

        loss = dflash_loss + w_ce * CE + w_pen * penalty

    Both lattice terms are teacher-forced on the ground-truth predecessor and
    only cover the prefix of slots whose ground truth is in the top-k at that
    slot and every earlier one. They reach the backbone through the candidates'
    log-probs, the draft hidden states and the anchor row's ``fc``/``hidden_norm``.
    """

    config_class: ClassVar[type[LiLiCorrSpeculatorConfig]] = LiLiCorrSpeculatorConfig  # type: ignore[misc]

    def __init__(self, config: LiLiCorrSpeculatorConfig) -> None:
        super().__init__(config=config)
        if self.use_draft_vocab:
            raise ValueError(
                "LiLiCorr requires the full verifier vocabulary (vLLM cannot serve "
                "a reduced draft vocab); omit --draft-vocab-size."
            )
        if config.sample_from_anchor:
            raise ValueError("LiLiCorr requires sample_from_anchor=False.")
        topk = config.lilicorr_candidate_topk
        if topk < 1 or topk > 16 or topk & (topk - 1):  # noqa: PLR2004
            raise ValueError("lilicorr_candidate_topk must be a power of two <= 16.")
        tl_config = config.transformer_layer_config
        self.lilicorr = LiLiCorrHead(
            model_hidden_size=tl_config.hidden_size,
            block_size=self.block_size,
            candidate_topk=topk,
            hidden_size=config.lilicorr_hidden_size,
            factor_dim=config.lilicorr_factor_dim,
            num_layers=config.lilicorr_num_layers,
            num_heads=config.lilicorr_num_heads,
            mlp_ratio=config.lilicorr_mlp_ratio,
            logit_scale=config.lilicorr_logit_scale,
            vector_eps=config.lilicorr_vector_eps,
            rms_norm_eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "LiLiCorrDraftModel":
        config = LiLiCorrSpeculatorConfig(
            **cls._build_base_config_kwargs("lilicorr", verifier_config, **kwargs)
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        train_kwargs, val_kwargs = DFlashDraftModel.get_trainer_kwargs(**kwargs)
        # The paper's `base` objective: gamma = 0.25 on CE + lambda (1.0) * penalty.
        for call_kwargs in (train_kwargs, val_kwargs):
            call_kwargs["lilicorr_w_ce"] = kwargs.get("lilicorr_w_ce", 0.25)
            call_kwargs["lilicorr_w_pen"] = kwargs.get("lilicorr_w_pen", 0.25)
        return train_kwargs, val_kwargs

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_aux*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        lilicorr_w_ce: float = 0.25,
        lilicorr_w_pen: float = 0.25,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, block_indices = (
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
        dflash_loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=False,
        )

        num_blocks, block, topk = (
            max_anchors,
            self.block_size,
            self.lilicorr.candidate_topk,
        )
        # Block position 0 is the anchor token; the lattice covers positions 1..B-1.
        logits_s = logits.view(num_blocks, block, -1)[:, 1:]
        targets_s = targets.view(num_blocks, block, -1)[:, 1:]
        hidden_s = hidden.view(num_blocks, block, -1)[:, 1:]
        slot_mask = aligned_loss_mask.view(num_blocks, block)[:, 1:].bool()
        block_docs = document_ids[0, block_indices].view(num_blocks, block)
        # Score a block only if every slot is supervised and it stays in one document.
        valid_block = slot_mask.all(-1) & (block_docs == block_docs[:, :1]).all(-1)

        with torch.no_grad():
            candidate_ids = logits_s.topk(topk, dim=-1).indices  # [N, S, K]

        # Recomputed from the live logits so the lattice terms reach the backbone.
        candidate_log_probs = logits_s.gather(
            -1, candidate_ids
        ).float() - torch.logsumexp(logits_s.float(), dim=-1, keepdim=True)

        # The anchor row is the target context just before the block's anchor token,
        # projected exactly like the context the draft attends to.
        anchor_pos = block_indices.view(num_blocks, block)[:, 0]
        prev_pos = (anchor_pos - 1).clamp(min=0)
        anchor_valid = (anchor_pos > 0) & (
            document_ids[0, prev_pos] == document_ids[0, anchor_pos]
        )
        anchor_hidden = self.hidden_norm(self.fc(hidden_states[0, prev_pos]))

        ce_loss, pen_loss, lilicorr_metrics, _ = lilicorr_lattice_terms(
            self.lilicorr,
            candidate_ids,
            candidate_log_probs,
            self.embed_tokens(candidate_ids).detach(),
            hidden_s,
            anchor_hidden,
            anchor_valid,
            targets_s,
            valid_block,
            slot_mask,
        )

        loss = dflash_loss + lilicorr_w_ce * ce_loss + lilicorr_w_pen * pen_loss

        metrics["dflash_loss_sum"] = metrics.pop("loss_sum")
        metrics["dflash_loss_total"] = metrics.pop("loss_total")
        metrics["loss_sum"] = loss.detach().clone()
        metrics["loss_total"] = torch.ones((), device=loss.device)
        metrics.update(lilicorr_metrics)

        return None, loss, metrics
