from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from transformers import PretrainedConfig

from speculators.losses import LossConfig, dflash_loss_decay
from speculators.losses.utils import _LOSS_REDUCTION_EPS
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.lilicorr.config import LiLiCorrSpeculatorConfig
from speculators.models.lilicorr.model_definitions import LiLiCorrHead
from speculators.models.metrics import compute_accepted_length_counts
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "LiLiCorrDraftModel",
]


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
        # Argmax-supervision experiment (fresh 2-epoch run): default the
        # target source to the verifier teacher's own top-1 token instead of
        # the observed sampled token. The corpus is unchanged (still the
        # sampled sequences from the input-token run); only the supervision
        # signal changes -- backbone CE, lattice gold tokens, predecessor
        # selection, and the penalty's gold-token reference all follow
        # verifier argmax on those sampled histories. See plan.md
        # (qwen38_lilicorr_fresh_2ep_argmax) for the experiment rationale.
        target_source = kwargs.get("lilicorr_target_source", "argmax")
        if target_source not in ("argmax", "input_ids"):
            raise ValueError(
                "lilicorr_target_source must be 'argmax' or 'input_ids', got "
                f"{target_source!r}."
            )
        # The paper's `base` objective: gamma = 0.25 on CE + lambda (1.0) * penalty.
        for call_kwargs in (train_kwargs, val_kwargs):
            call_kwargs["lilicorr_w_ce"] = kwargs.get("lilicorr_w_ce", 0.25)
            call_kwargs["lilicorr_w_pen"] = kwargs.get("lilicorr_w_pen", 0.25)
            # "input_ids" (default) uses the token actually observed at each
            # position, for both the lattice head and the DFlash backbone's CE
            # term, matching NVIDIA ModelOpt's reference implementation;
            # "argmax" instead scores against the verifier teacher's own top-1
            # token (this codebase's original behavior).
            call_kwargs["lilicorr_target_source"] = target_source
            call_kwargs["loss_chunk_anchors"] = kwargs.get("loss_chunk_anchors")
        return train_kwargs, val_kwargs

    def _lattice_terms(
        self,
        logits: torch.Tensor,  # [1, n_blocks*block, V]
        targets: torch.Tensor,  # [1, n_blocks*block, V]
        hidden: torch.Tensor,  # [1, n_blocks*block, H]
        aligned_loss_mask: torch.Tensor,  # [1, n_blocks*block]
        block_indices: torch.Tensor,  # [n_blocks*block], absolute positions
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_aux*H]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        target_source: str = "argmax",
    ):
        """LiLiCorr head terms for one contiguous group of anchor blocks.

        The tensor arguments must cover exactly that group's rows (the
        monolithic path passes the full sequence of blocks; the chunked path
        passes one chunk). Returns *raw* (unnormalized) numerators and counts —
        ``(ce_num, pen_num, supervised_count, eal, top1_eal, oracle_eal)`` — so
        callers can pool several groups before dividing by the pooled
        denominators, which is what the unchunked computation does.

        ``target_source`` picks how the ground-truth token is defined:
        ``"argmax"`` (default) takes the verifier teacher distribution's top-1
        token, as this codebase has always done; ``"input_ids"`` instead uses
        the token actually observed at that position, matching NVIDIA
        ModelOpt's reference LiLiCorr implementation. The two agree whenever
        the training data was generated by greedy decoding from the verifier.
        """
        block = self.block_size
        topk = self.lilicorr.candidate_topk
        num_blocks = logits.shape[1] // block
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
            if target_source == "input_ids":
                gt_ids = input_ids[0, block_indices].view(num_blocks, block)[:, 1:]
            else:
                gt_ids = targets_s.argmax(-1)  # [N, S]
            gt_hits = candidate_ids == gt_ids.unsqueeze(-1)
            gt_in_lattice = gt_hits.any(-1)
            gt_index = gt_hits.float().argmax(-1)  # [N, S]

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

        log_start, log_pair = self.lilicorr(
            self.embed_tokens(candidate_ids).detach(),
            candidate_log_probs,
            hidden_s,
            anchor_hidden,
            anchor_valid,
        )

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
        ce = -log_p.gather(-1, gt_index.unsqueeze(-1)).squeeze(-1)
        ce_num = (ce * supervised).sum()

        # Distractor penalty: the head's mass on each candidate, weighted by how far
        # below the ground truth the target's logit puts it.
        cand_target_logits = targets_s.gather(-1, candidate_ids).float()
        gt_target_logit = cand_target_logits.gather(-1, gt_index.unsqueeze(-1))
        rejection = (gt_target_logit - cand_target_logits).clamp_min(0.0)
        pen_num = ((log_p.exp() * rejection).sum(-1) * supervised).sum()
        supervised_count = supervised.sum()

        with torch.no_grad():
            # Greedy left-to-right walk, as served.
            current = log_start.argmax(-1)
            picks = [current]
            for slot in range(block - 2):
                row = log_pair[:, slot].gather(
                    1, current[:, None, None].expand(-1, 1, topk)
                )
                current = row.squeeze(1).argmax(-1)
                picks.append(current)
            selected_ids = candidate_ids.gather(
                -1, torch.stack(picks, dim=1).unsqueeze(-1)
            ).squeeze(-1)
            eval_mask = slot_mask & valid_block[:, None]
            # Same blocks for all three, so lilicorr_eal - lilicorr_top1_eal is the
            # head's gain and lilicorr_oracle_eal bounds what any walk could reach.
            eal_counts = [
                compute_accepted_length_counts(correct, eval_mask)
                for correct in (
                    selected_ids == gt_ids,
                    candidate_ids[..., 0] == gt_ids,
                    gt_in_lattice,
                )
            ]
        return ce_num, pen_num, supervised_count, *eal_counts

    def _chunk_terms(
        self,
        hidden: torch.Tensor,  # [1, max_anchors*block, H], trunk output
        verifier_last_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        input_ids: torch.Tensor,
        aligned_loss_mask: torch.Tensor,
        block_indices: torch.Tensor,
        r0: int,
        r1: int,
        dflash_denominator: torch.Tensor,
        loss_config: LossConfig | None,
        gamma: float,
        per_position_loss_weight: str,
        dpace_alpha: float,
        target_source: str,
    ):
        """Head projections + dflash/lattice terms for mask-token rows [r0, r1).

        Wrapped in a checkpoint by the caller: the ``[chunk, vocab]`` logits and
        targets this materializes are dropped after the forward pass and
        recomputed chunk-by-chunk during backward, which is what bounds memory.
        """
        block = self.block_size
        seq_len = verifier_last_hidden_states.shape[1]
        rows = block_indices[r0:r1]
        target_indices = (rows - 1) % seq_len
        with torch.no_grad():
            targets_c = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states[:, target_indices])
            )
        hidden_c = hidden[:, r0:r1]
        logits_c = self.lm_head(hidden_c)

        ce_target_ids = input_ids[:, rows] if target_source == "input_ids" else None

        # Dividing by the *global* denominator makes the per-chunk losses and
        # term metrics directly additive across chunks.
        dflash_loss_c, metrics_c = compute_metrics(
            logits_c,
            targets_c,
            aligned_loss_mask[:, r0:r1],
            block,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=False,
            denominator=dflash_denominator,
            ce_target_ids=ce_target_ids,
        )
        ce_num_c, pen_num_c, sup_count_c, eal, top1_eal, oracle_eal = (
            self._lattice_terms(
                logits_c,
                targets_c,
                hidden_c,
                aligned_loss_mask[:, r0:r1],
                rows,
                hidden_states,
                document_ids,
                input_ids,
                target_source,
            )
        )

        # Keep only count-type "_total" keys; the scalar-loss markers are set
        # once by the chunk combiner after pooling.
        count_totals = {"full_acc_total", "eal_total"} | {
            f"position_{pos}_acc_total" for pos in range(1, block)
        }
        metrics_c = {
            key: val
            for key, val in metrics_c.items()
            if not key.endswith("_total") or key in count_totals
        }
        metrics_c["dflash_loss_sum"] = metrics_c.pop("loss_sum")
        metrics_c["lilicorr_eal_sum"], metrics_c["lilicorr_eal_total"] = eal
        metrics_c["lilicorr_top1_eal_sum"], metrics_c["lilicorr_top1_eal_total"] = (
            top1_eal
        )
        metrics_c["lilicorr_oracle_eal_sum"], metrics_c["lilicorr_oracle_eal_total"] = (
            oracle_eal
        )
        # Raw numerators; the combiner divides by the pooled lattice denominator.
        metrics_c["lilicorr_ce_sum"] = ce_num_c.detach()
        metrics_c["lilicorr_penalty_sum"] = pen_num_c.detach()
        return dflash_loss_c, ce_num_c, pen_num_c, sup_count_c.detach(), metrics_c

    # The chunk loop's recompute-inside-backward (custom autograd Functions,
    # Triton kernels, data-dependent Python) is not dynamo-traceable in any
    # useful sense; run it eagerly even when the entry forward is compiled.
    @torch.compiler.disable
    def _chunked_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        loss_config: LossConfig | None,
        gamma: float,
        max_anchors: int,
        per_position_loss_weight: str,
        dpace_alpha: float,
        lilicorr_w_ce: float,
        lilicorr_w_pen: float,
        lilicorr_target_source: str,
        loss_chunk_anchors: int,
        **kwargs,
    ):
        block = self.block_size
        hidden, _, _, aligned_loss_mask, block_indices = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            compute_heads=False,
            **kwargs,
        )
        # The denominator the unchunked reduction would use, so chunk losses
        # add up to exactly the monolithic value: the decay-weighted mask sum
        # (matching NVIDIA ModelOpt), not a raw position count.
        if per_position_loss_weight == "dpace":
            # D-PACE's weight depends on the per-position loss itself, which isn't
            # available before the chunked logits are computed; approximate with
            # the raw (unweighted) mask count, as before.
            weight_mask = aligned_loss_mask
        else:
            pos_idx = (
                torch.arange(
                    aligned_loss_mask.shape[1], device=aligned_loss_mask.device
                )
                .unsqueeze(0)
                .float()
                % block
            )
            decay_mult = dflash_loss_decay(pos_idx, gamma, sample_from_anchor=False)
            weight_mask = aligned_loss_mask.float() * decay_mult
        dflash_denominator = (weight_mask.sum(dim=1) + _LOSS_REDUCTION_EPS).detach()

        dflash_loss = None
        ce_num = pen_num = sup_count = None
        metrics: dict[str, torch.Tensor] = {}
        ones = torch.ones((), device=hidden.device)
        for b0 in range(0, max_anchors, loss_chunk_anchors):
            b1 = min(b0 + loss_chunk_anchors, max_anchors)
            dflash_loss_c, ce_num_c, pen_num_c, sup_count_c, metrics_c = (
                torch.utils.checkpoint.checkpoint(
                    self._chunk_terms,
                    hidden,
                    verifier_last_hidden_states,
                    hidden_states,
                    document_ids,
                    input_ids,
                    aligned_loss_mask,
                    block_indices,
                    b0 * block,
                    b1 * block,
                    dflash_denominator,
                    loss_config,
                    gamma,
                    per_position_loss_weight,
                    dpace_alpha,
                    lilicorr_target_source,
                    use_reentrant=False,
                )
            )
            dflash_loss = (
                dflash_loss_c if dflash_loss is None else dflash_loss + dflash_loss_c
            )
            ce_num = ce_num_c if ce_num is None else ce_num + ce_num_c
            pen_num = pen_num_c if pen_num is None else pen_num + pen_num_c
            sup_count = sup_count_c if sup_count is None else sup_count + sup_count_c
            for key, val in metrics_c.items():
                if key in metrics:
                    metrics[key] = metrics[key] + val.detach()
                else:
                    metrics[key] = val.detach()

        # The loop body always runs (max_anchors >= 1), so the accumulators
        # are always assigned.
        denom = sup_count.clamp_min(1.0)  # type: ignore[union-attr]
        ce_loss = ce_num / denom
        pen_loss = pen_num / denom
        loss = dflash_loss + lilicorr_w_ce * ce_loss + lilicorr_w_pen * pen_loss

        metrics["lilicorr_ce_sum"] = ce_loss.detach()
        metrics["lilicorr_penalty_sum"] = pen_loss.detach()
        metrics["loss_sum"] = loss.detach().clone()
        # Every accumulated "_sum" without a count-type "_total" is a pooled
        # scalar loss; give it the standard 1.0 marker denominator.
        for key in [k for k in metrics if k.endswith("_sum")]:
            if f"{key[:-4]}_total" not in metrics:
                metrics[f"{key[:-4]}_total"] = ones.clone()
        return None, loss, metrics

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
        lilicorr_target_source: str = "argmax",
        loss_chunk_anchors: int | None = None,
        **kwargs,
    ):
        if loss_chunk_anchors:
            return self._chunked_forward(
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
                lilicorr_w_ce=lilicorr_w_ce,
                lilicorr_w_pen=lilicorr_w_pen,
                lilicorr_target_source=lilicorr_target_source,
                loss_chunk_anchors=loss_chunk_anchors,
                **kwargs,
            )

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
        ce_target_ids = (
            input_ids[:, block_indices]
            if lilicorr_target_source == "input_ids"
            else None
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
            ce_target_ids=ce_target_ids,
        )

        ce_num, pen_num, sup_count, eal, top1_eal, oracle_eal = self._lattice_terms(
            logits,
            targets,
            hidden,
            aligned_loss_mask,
            block_indices,
            hidden_states,
            document_ids,
            input_ids,
            lilicorr_target_source,
        )
        denom = sup_count.clamp_min(1.0)
        ce_loss = ce_num / denom
        pen_loss = pen_num / denom

        loss = dflash_loss + lilicorr_w_ce * ce_loss + lilicorr_w_pen * pen_loss

        with torch.no_grad():
            ones = torch.ones((), device=loss.device)
            metrics["dflash_loss_sum"] = metrics.pop("loss_sum")
            metrics["dflash_loss_total"] = metrics.pop("loss_total")
            metrics["loss_sum"] = loss.detach().clone()
            metrics["loss_total"] = ones
            metrics["lilicorr_ce_sum"] = ce_loss.detach()
            metrics["lilicorr_ce_total"] = ones.clone()
            metrics["lilicorr_penalty_sum"] = pen_loss.detach()
            metrics["lilicorr_penalty_total"] = ones.clone()
            metrics["lilicorr_eal_sum"], metrics["lilicorr_eal_total"] = eal
            metrics["lilicorr_top1_eal_sum"], metrics["lilicorr_top1_eal_total"] = (
                top1_eal
            )
            metrics["lilicorr_oracle_eal_sum"], metrics["lilicorr_oracle_eal_total"] = (
                oracle_eal
            )

        return None, loss, metrics
