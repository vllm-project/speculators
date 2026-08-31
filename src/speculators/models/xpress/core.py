from typing import ClassVar, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.utils import conditional_torch_compile
from speculators.models.xpress.config import XPressSpeculatorConfig
from speculators.models.xpress.metrics import (
    compute_metrics,
    greedy_accept_length_metrics,
)
from speculators.models.xpress.model_definitions import XPressRefinerHead

_DEFAULT_LOSS_CONFIG: LossConfig | None = None

__all__ = [
    "XPressDraftModel",
]


def _buffer(module: nn.Module, name: str) -> torch.Tensor | None:
    """Fetch a named buffer, or None when the module has no such tensor.

    ``getattr`` on an ``nn.Module`` can return a submodule as easily as a buffer,
    so the isinstance check is what makes the duck-typed rotary lookup below
    safe as well as checkable.
    """
    value = getattr(module, name, None)
    return value if isinstance(value, torch.Tensor) else None


@SpeculatorModel.register("xpress")
class XPressDraftModel(DFlashDraftModel):
    """DFlash backbone plus the XPress causal-refiner head.

    Training runs one teacher-forced refine pass plus ``consistency_passes``
    free-running rounds (each conditioned on the previous round's argmax, the
    exact recurrence of inference-time Jacobi decoding, stop-grad between
    rounds), and anchors the co-trained backbone with a base-logits term.
    Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[XPressSpeculatorConfig]] = XPressSpeculatorConfig  # type: ignore[misc]

    def _init_weights(self, module) -> None:
        """Initialize weights HF's ``from_pretrained`` reports as missing.

        The speculators base classes define no ``_init_weights``, so on a
        partial checkpoint (e.g. a converted DFlash backbone warm-starting an
        XPress run) the missing ``refiner_head.*`` params would otherwise stay
        as uninitialized meta-materialized memory — NaNs included. Loaded
        weights are unaffected (HF only re-initializes missing modules).
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, XPressRefinerHead):
            # raw mixer starts at zero (validated recipe: mixer_init=zeros);
            # the causal tril + identity are applied functionally in forward.
            module.mix_l.data.zero_()
            # non-persistent buffer: not in any checkpoint, garbage after a
            # meta-device load — recompute.
            module.causal_tril.data.copy_(
                torch.tril(torch.ones_like(module.causal_tril))
            )

    def _reinit_nonpersistent_buffers(self) -> None:
        """Recompute non-persistent buffers after a meta-device load.

        Non-persistent buffers live in NO checkpoint and belong to modules that
        may have no missing *parameters*, so HF's ``from_pretrained`` neither
        loads nor re-initializes them — they surface as uninitialized memory
        (run-to-run nondeterministic NaNs, first observed in rotary_emb's
        ``inv_freq``).
        """
        for module in self.modules():
            inv_freq = _buffer(module, "inv_freq")
            if inv_freq is not None and hasattr(module, "config"):
                fresh = type(module)(module.config, device="cpu")
                fresh_inv_freq = _buffer(fresh, "inv_freq")
                if fresh_inv_freq is not None:
                    inv_freq.data.copy_(
                        fresh_inv_freq.to(inv_freq.device, inv_freq.dtype)
                    )
                if hasattr(fresh, "attention_scaling"):
                    module.attention_scaling = fresh.attention_scaling
                original = _buffer(module, "original_inv_freq")
                fresh_original = _buffer(fresh, "original_inv_freq")
                if original is not None and fresh_original is not None:
                    original.data.copy_(
                        fresh_original.to(original.device, original.dtype)
                    )
            elif isinstance(module, XPressRefinerHead):
                module.causal_tril.data.copy_(
                    torch.tril(torch.ones_like(module.causal_tril))
                )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # type: ignore[override]
        model = cast("XPressDraftModel", super().from_pretrained(*args, **kwargs))
        model._reinit_nonpersistent_buffers()  # noqa: SLF001
        return model

    def __init__(self, config: XPressSpeculatorConfig) -> None:
        super().__init__(config=config)
        self.refiner_head = XPressRefinerHead(
            verifier_vocab_size=self.verifier_vocab_size,
            draft_vocab_size=self.draft_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            block_size=self.block_size,
            rank=config.xpress_rank,
            mlp_ratio=config.xpress_mlp_ratio,
        )
        self.post_init()

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "XPressDraftModel":
        """Create an XPress model from training arguments (mirrors DSpark)."""
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        config = XPressSpeculatorConfig(
            **cls._build_base_config_kwargs("xpress", verifier_config, **kwargs),
            xpress_rank=kwargs.get("xpress_rank", 256),
            xpress_mlp_ratio=kwargs.get("xpress_mlp_ratio", 2),
            num_jacobi_passes=kwargs.get("num_jacobi_passes", 6),
        )
        if sample_from_anchor_arg is not None:
            config.sample_from_anchor = sample_from_anchor_arg

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve XPress's compound loss and consistency knobs."""
        shared = {
            "loss_config": resolve_loss_config(kwargs["loss_fn"]),
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 3072),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "consistency_weight": kwargs.get("consistency_weight", 0.3),
            "consistency_passes": kwargs.get("consistency_passes", 3),
            "base_anchor_weight": kwargs.get("base_anchor_weight", 0.6),
            "base_anchor_full_weight": kwargs.get("base_anchor_full_weight", False),
            "base_anchor_floor": kwargs.get("base_anchor_floor"),
            "decayed_loss_norm": kwargs.get("decayed_loss_norm", False),
            "ce_from_data": kwargs.get("ce_from_data", False),
        }
        return dict(shared), dict(shared)

    def _draft_to_verifier_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocab argmax ids back to verifier-vocab ids (d2t offsets)."""
        if self.d2t is not None:
            return draft_ids + self.d2t[draft_ids]
        return draft_ids

    @conditional_torch_compile
    def forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        consistency_weight: float = 0.3,
        consistency_passes: int = 3,
        base_anchor_weight: float | torch.Tensor = 0.6,
        base_anchor_full_weight: bool = False,
        decayed_loss_norm: bool = False,
        ce_from_data: bool = False,
        **kwargs,
    ):
        (
            hidden,
            logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            anchor_valid,
        ) = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )

        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        anchor_tokens = block_tokens[:, :1]
        anchor_pos = anchored_block_indices.view(num_blocks, block)[:, 0]
        am1_pos = (anchor_pos - 1).clamp(min=0)
        am1_ok = (anchor_pos > 0) & (
            document_ids[0, am1_pos] == document_ids[0, anchor_pos]
        )
        am1_tokens = torch.where(
            am1_ok, input_ids[0, am1_pos], anchor_tokens[:, 0]
        ).unsqueeze(1)

        data_labels = None
        if self.d2t is None:
            if self.config.sample_from_anchor:
                nxt = (anchored_block_indices + 1).clamp(max=input_ids.shape[1] - 1)
                data_labels = input_ids[0, nxt].view(1, mask_tokens_size)
            else:
                data_labels = block_tokens.reshape(1, mask_tokens_size)

        # ce_from_data: CE labels = the DATA tokens, not the teacher
        # argmax. Requires the full verifier vocab.
        ce_data_labels = None
        if ce_from_data:
            if data_labels is None:
                raise ValueError(
                    "ce_from_data requires draft_vocab == verifier_vocab "
                    "(XPress recipes are full-vocab)"
                )
            ce_data_labels = data_labels
        if self.config.sample_from_anchor:
            # Slot k predicts token p+k+1; teacher prev for slot k = token p+k.
            prev_tf = block_tokens
        else:
            # Fill-in: slot k predicts token p+k (slot 0 = anchor); teacher prev
            # for slot k = token p+k-1 (shifted).
            prev_tf = torch.cat([am1_tokens, block_tokens[:, :-1]], dim=1)

        hidden_blocks = hidden.view(num_blocks, block, -1)
        base_block_logits = logits.view(num_blocks, block, -1)
        hcache = self.refiner_head.hidden_cache(
            hidden_blocks.to(self.refiner_head.down_h.weight.dtype)
        )

        # Teacher-forced refine pass.
        bias_tf = self.refiner_head.block_bias(prev_tf, hcache=hcache)
        logits_tf = (base_block_logits + bias_tf).view(1, mask_tokens_size, -1)

        round_logits: list[torch.Tensor] = []
        if consistency_weight > 0.0 and consistency_passes > 0:
            pred = self._draft_to_verifier_ids(
                base_block_logits.detach().argmax(dim=-1)
            )
            for _ in range(int(consistency_passes)):
                if self.config.sample_from_anchor:
                    prev_j = torch.cat([anchor_tokens, pred[:, :-1]], dim=1)
                else:
                    pred[:, :1] = anchor_tokens  # slot 0 is the anchor, not predicted
                    prev_j = torch.cat([am1_tokens, pred[:, :-1]], dim=1)
                bias_j = self.refiner_head.block_bias(prev_j, hcache=hcache)
                refined_j = base_block_logits + bias_j
                round_logits.append(refined_j.view(1, mask_tokens_size, -1))
                pred = self._draft_to_verifier_ids(refined_j.detach().argmax(dim=-1))

        # Eval-only: free-running K-pass Jacobi rollout accept length — the
        # honest inference proxy (the analytical accept_rate above is
        # teacher-forced). Shares the exact recurrence of the consistency loop.
        if not self.training:
            with torch.no_grad():
                # The in-training accept-length rollout uses block_size - 1 passes;
                # num_jacobi_passes (default 6) is the OFFLINE package-eval setting
                # that converted checkpoints export, so it must not be reused here.
                _eval_passes = getattr(self.config, "eval_jacobi_passes", None)
                if not _eval_passes:
                    _eval_passes = self.block_size - 1
                refined_draft = base_block_logits.argmax(dim=-1)
                pred = self._draft_to_verifier_ids(refined_draft)
                for _ in range(int(_eval_passes)):
                    if self.config.sample_from_anchor:
                        prev_j = torch.cat([anchor_tokens, pred[:, :-1]], dim=1)
                    else:
                        pred[:, :1] = anchor_tokens
                        prev_j = torch.cat([am1_tokens, pred[:, :-1]], dim=1)
                    bias_j = self.refiner_head.block_bias(prev_j, hcache=hcache)
                    refined_draft = (base_block_logits + bias_j).argmax(dim=-1)
                    pred = self._draft_to_verifier_ids(refined_draft)
                # Ground truth / validity conventions (see
                # greedy_accept_length_metrics): DATA tokens at each slot's label
                # position, reachability instead of the assistant mask, and every
                # kept block in the denominator.
                _slot_pos = anchored_block_indices.view(num_blocks, block)
                _label_pos = (
                    _slot_pos + 1 if self.config.sample_from_anchor else _slot_pos
                )
                _seq_len = input_ids.shape[1]
                _in_bounds = _label_pos < _seq_len
                _label_pos = _label_pos.clamp(max=_seq_len - 1)
                _gt_tokens = input_ids[0, _label_pos]
                # packed corpora: a block may run past its document's end
                _same_doc = document_ids[0, _label_pos] == document_ids[
                    0, anchor_pos
                ].unsqueeze(1)
                rollout_metrics = greedy_accept_length_metrics(
                    self._draft_to_verifier_ids(refined_draft),
                    self._draft_to_verifier_ids(base_block_logits.argmax(dim=-1)),
                    _gt_tokens,
                    _in_bounds & _same_doc,
                    anchor_valid,
                    sample_from_anchor=self.config.sample_from_anchor,
                )
        else:
            rollout_metrics = {}

        # The trainer's annealing schedule passes a 0-dim tensor so its value stays
        # out of Dynamo's guards (a per-step float would recompile this frame until it
        # hit recompile_limit and fell back to eager). Comparing a tensor here would
        # make the branch data-dependent, so a tensor simply means "enabled" -- which
        # holds by construction, and if the schedule's floor were 0 the anchor term
        # would contribute exactly 0 anyway.
        base_anchor_on = (
            True
            if isinstance(base_anchor_weight, torch.Tensor)
            else base_anchor_weight > 0.0
        )

        loss, metrics = compute_metrics(
            logits_tf,
            round_logits,
            logits if base_anchor_on else None,
            targets,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or resolve_loss_config('{"ce": 0.1, "tv": 0.9}'),
            gamma=gamma,
            consistency_weight=consistency_weight,
            base_anchor_weight=base_anchor_weight,
            base_anchor_full_weight=base_anchor_full_weight,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            decayed_loss_norm=decayed_loss_norm,
            ce_data_labels=ce_data_labels,
            data_labels=data_labels,
        )
        metrics.update(rollout_metrics)
        return None, loss, metrics
