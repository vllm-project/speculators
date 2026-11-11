"""MTP speculator model implementation."""

import logging
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin
from speculators.models.attention import create_float_mask
from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
    extend_dense_mask_for_draft_tokens,
    extend_mask_for_draft_tokens,
)
from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.models.mtp.model_definitions import (
    mtp_model_classes,
    resolve_model_type,
)
from speculators.models.utils import (
    flatten_rope_parameters,
    resolve_verifier_norm_class,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig

logger = logging.getLogger(__name__)

__all__ = ["MTPDraftModel", "compute_step_weights"]

_QWEN3_5_MODEL_TYPES = {"qwen3_5_text", "qwen3_5_moe_text"}
_QWEN3_5_MROPE_POSITION_ID_RANK = 3
_QWEN3_5_MROPE_SECTIONS = 3


# Supervised positions per LM-head + CE chunk (bounds the fp32 logits buffer to
# ~2 GB at a 248k vocabulary).
_HEAD_CE_CHUNK = 2048


class _TTTCache:
    """Minimal cache for the HF attention's `past_key_values.update` call.

    Returns the frozen prefix (step-0 "context" KV plus earlier draft steps' KV)
    concatenated with this step's KV, and keeps this step's KV so the caller can
    append it for later steps. Stateless across calls, so a gradient-checkpoint
    recompute of the same step returns identical tensors.
    """

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor]) -> None:
        self.prefix_k, self.prefix_v = keys, values
        # set by update(), which the attention layer calls once per forward
        self.new_k: torch.Tensor
        self.new_v: torch.Tensor

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # noqa: ARG002
        self.new_k, self.new_v = key_states, value_states
        if not self.prefix_k:
            return key_states, value_states
        return (
            torch.cat([*self.prefix_k, key_states], dim=-2),
            torch.cat([*self.prefix_v, value_states], dim=-2),
        )


# Flex block masks are extended by whole blocks per draft step
# (`extend_mask_for_draft_tokens`), so the flex path pads rows to a multiple of it.
_FLEX_BLOCK_SIZE = 128


def _shift_left(x: torch.Tensor, k: int, fill: int | bool) -> torch.Tensor:
    """``x[..., t + k]`` at position t, ``fill`` past the end."""
    return torch.cat([x[..., k:], x.new_full((*x.shape[:-1], k), fill)], dim=-1)


def _prepare_mtp_position_ids(
    position_ids: torch.Tensor,
    valid_len: int,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare mask/layer and rotary position IDs for an MTP step.

    Released transformers versions expand Qwen3.5's 2-D position IDs inside
    the rotary embedding. Newer versions expect the caller to provide the
    three MRoPE sections explicitly. Keep the 2-D text positions for mask
    construction and the decoder layer, while providing the 3-D form to the
    rotary embedding when needed.

    Returns:
        A pair of ``(text_position_ids, rotary_position_ids)``.
    """
    if position_ids.ndim == _QWEN3_5_MROPE_POSITION_ID_RANK:
        if model_type not in _QWEN3_5_MODEL_TYPES:
            raise ValueError(
                "3-D position_ids are only supported for Qwen3.5 MTP models; "
                f"got model type {model_type!r}."
            )
        if position_ids.shape[0] != _QWEN3_5_MROPE_SECTIONS:
            raise ValueError(
                "Qwen3.5 MTP position_ids must have shape [3, batch, seq_len]; "
                f"got {tuple(position_ids.shape)}."
            )
        rotary_position_ids = position_ids[:, :, :valid_len]
        return rotary_position_ids[0], rotary_position_ids

    step_position_ids = position_ids[:, :valid_len]
    if model_type in _QWEN3_5_MODEL_TYPES:
        rotary_position_ids = step_position_ids.unsqueeze(0).expand(
            _QWEN3_5_MROPE_SECTIONS, -1, -1
        )
    else:
        rotary_position_ids = step_position_ids
    return step_position_ids, rotary_position_ids


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
    # The MTP layer mixes into the HF decoder layer, which is a
    # GradientCheckpointingLayer; HF defaults to use_reentrant=False, so the
    # keyword-only call in forward() still propagates gradients. Needed for
    # many speculative steps x long sequences, where each step's activations
    # are otherwise all kept for backward.
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_norm.weight",
    ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: MTPSpeculatorConfig) -> None:
        # Same attention backends as Eagle3: flex by default, sdpa / eager (dense
        # masks) for hardware without flex attention.
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
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
        self.rotary_emb = self._model_definitions.rotary_emb_class(
            flatten_rope_parameters(tc)
        )
        # vLLM feeds the MTP head the verifier's output hidden states (after its
        # final norm); training data holds the pre-norm last-layer hidden states.
        self.verifier_norm = resolve_verifier_norm_class(config)(
            tc.hidden_size,
            eps=tc.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        # The layer only reads the prefix KV (see _TTTCache), so it may keep the
        # cache under gradient checkpointing.
        self.mtp_layers[0]._can_checkpoint_with_cache = True  # type: ignore[assignment]  # noqa: SLF001

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

    def _head_ce(
        self, hidden: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LM head + unreduced cross entropy; also returns the greedy prediction.

        Not compiled: the number of supervised positions changes every batch,
        and a `dynamic=False` compile would recompile on each new size.
        """
        logits = self.lm_head(hidden)
        loss = nn.functional.cross_entropy(logits.float(), targets, reduction="none")
        return loss, logits.argmax(dim=-1)

    @torch.compiler.disable
    def _head_ce_chunked(
        self, hidden: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`_head_ce` over chunks of supervised positions, each checkpointed.

        A single checkpoint over all positions still materializes the full fp32
        logits in forward and again in backward (~10 GB at 10k positions x 248k
        vocab), which happens when almost every token is supervised (e.g.
        self-distilled chat data). Per-chunk checkpoints keep one chunk alive.
        """
        losses, preds = [], []
        for start in range(0, hidden.shape[0], _HEAD_CE_CHUNK):
            h = hidden[start : start + _HEAD_CE_CHUNK]
            t = targets[start : start + _HEAD_CE_CHUNK]
            if torch.is_grad_enabled():
                loss, pred = torch.utils.checkpoint.checkpoint(
                    self._head_ce, h, t, use_reentrant=False
                )
            else:
                loss, pred = self._head_ce(h, t)
            losses.append(loss)
            preds.append(pred)
        if not losses:
            # nothing supervised: an empty loss that is still part of the graph, so
            # backward() works (and DDP sees every parameter) on such batches
            return hidden.float().sum(dim=-1), targets.new_zeros(0)
        return torch.cat(losses), torch.cat(preds)

    @torch.compiler.disable
    def _extend_attn_mask(self, mask, seq_len: int):
        """Append a diagonal KV block (each anchor's own draft-step KV)."""
        if self._attn_impl == "simple_flex_attention":
            return extend_mask_for_draft_tokens(mask)
        return extend_dense_mask_for_draft_tokens(mask, seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        step_weights: list[float] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple:
        """Multi-step MTP training forward, matching vLLM's MTP proposer.

        - The head sees the verifier's output hidden states, i.e. after its final
          norm (the training data holds the pre-norm last-layer states).
        - Draft step k at anchor t runs at position t + k and attends to the step-0
          KV (computed from verifier hidden states) at positions <= t in the same
          document, plus only anchor t's own KV from earlier draft steps -- the
          Eagle3 TTT mask (``create_combined_mask_mod``), extended by one diagonal
          block per step.

        Step k is teacher-forced: its token input at anchor t is ``input_ids[t+k+1]``
        and its target ``input_ids[t+k+2]``, supervised where ``loss_mask`` is set
        and the target is in the anchor's document.

        :param input_ids: Token IDs [1, seq_len], packed documents.
        :param hidden_states: Verifier last-layer hidden states [1, seq_len, hidden].
        :param document_ids: Document index per position [1, seq_len], -1 for
            padding (as the training collate produces).
        :param loss_mask: Optional [1, seq_len]; 1 = supervise this target token.
        :param position_ids: Optional [1, seq_len]; Qwen3.5 also accepts the
            MRoPE layout [3, 1, seq_len].
        :param step_weights: Per-step loss weights (None = uniform).
        :return: ``(draft_tokens, loss, metrics)``. ``draft_tokens[k]`` holds the
            step-k greedy predictions at the supervised positions only.
        """
        input_ids = input_ids.long()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("MTP forward expects packed batches of size 1")
        num_steps = self.config.num_speculative_steps
        if step_weights is not None and len(step_weights) != num_steps:
            raise ValueError(
                f"step_weights has {len(step_weights)} entries but "
                f"num_speculative_steps={num_steps}"
            )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        draft_tokens: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float | torch.Tensor] = {}
        effective_steps = min(num_steps, max(0, seq_len - 2))
        if effective_steps == 0:
            metrics["loss_sum"] = total_loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return (draft_tokens, total_loss, metrics)

        doc = document_ids.reshape(-1).to(device=device, dtype=torch.long)
        supervised = (
            torch.ones(seq_len, dtype=torch.bool, device=device)
            if loss_mask is None
            else loss_mask.reshape(-1).bool()
        )

        # Pad to whole flex blocks; padded rows are document -1 (masked, unsupervised).
        pad = (
            -seq_len % _FLEX_BLOCK_SIZE
            if self._attn_impl == "simple_flex_attention"
            else 0
        )
        total_len = seq_len + pad
        if pad:
            input_ids = nn.functional.pad(input_ids, (0, pad))
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad))
            position_ids = nn.functional.pad(position_ids, (0, pad))
            doc = nn.functional.pad(doc, (0, pad), value=-1)
            supervised = nn.functional.pad(supervised, (0, pad), value=False)
        text_pos, rotary_pos = _prepare_mtp_position_ids(
            position_ids, total_len, self.config.transformer_layer_config.model_type
        )

        attn_mask = self._create_mask_fn(
            create_combined_mask_mod(doc, total_len),
            B=None,
            H=None,
            Q_LEN=total_len,
            KV_LEN=total_len,
            device=device,
        )
        layer = self.mtp_layers[0]
        kv_k: list[torch.Tensor] = []
        kv_v: list[torch.Tensor] = []
        anchors: torch.Tensor | None = None
        run_correct: torch.Tensor | None = None
        current_hidden = self.verifier_norm(hidden_states)
        for step in range(effective_steps):
            cache = _TTTCache(kv_k, kv_v)
            mtp_output = layer(
                hidden_states=current_hidden,
                token_embeddings=self.embed_tokens(_shift_left(input_ids, step + 1, 0)),
                attention_mask=attn_mask,
                position_ids=text_pos + step,
                position_embeddings=self.rotary_emb(current_hidden, rotary_pos + step),
                past_key_values=cache,
            )
            kv_k = [*kv_k, cache.new_k]
            kv_v = [*kv_v, cache.new_v]

            step_targets = _shift_left(input_ids, step + 2, 0)[0]
            step_mask = (
                _shift_left(supervised, step + 2, False)
                & (_shift_left(doc, step + 2, -1) == doc)
                & (doc != -1)
            )
            sel_targets = step_targets[step_mask]
            unreduced, sel_pred = self._head_ce_chunked(
                mtp_output[0][step_mask], sel_targets
            )
            draft_tokens.append(sel_pred)

            weight = step_weights[step] if step_weights is not None else 1.0
            step_loss = weight * unreduced.sum() / step_mask.sum().clamp(min=1)
            total_loss = total_loss + step_loss
            metrics[f"loss_step_{step}"] = step_loss.detach().clone()

            with torch.no_grad():
                correct = torch.zeros_like(step_mask)
                correct[step_mask] = sel_pred == sel_targets
                if anchors is None:
                    anchors = step_mask
                run_correct = correct if run_correct is None else run_correct & correct
                # Teacher-forced step inputs equal the drafted tokens whenever all
                # earlier steps are correct, so this cumulative rate matches vLLM's
                # spec_decode_num_accepted_tokens_per_pos / num_drafts.
                metrics[f"accept_pos_{step + 1}_sum"] = (
                    (run_correct & anchors).sum().float()
                )
                metrics[f"accept_pos_{step + 1}_total"] = anchors.sum().float()
                metrics[f"step_{step + 1}_acc_sum"] = correct.sum().float()
                metrics[f"step_{step + 1}_acc_total"] = step_mask.sum().float()

            current_hidden = mtp_output
            attn_mask = self._extend_attn_mask(attn_mask, total_len)

        metrics["loss_sum"] = total_loss.detach().clone()
        metrics["loss_total"] = torch.tensor(1.0, device=device)
        return (draft_tokens, total_loss, metrics)

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: PretrainedConfig,
        *,
        num_speculative_steps: int = 3,
        verifier_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> "MTPDraftModel":
        if verifier_name_or_path is None:
            raise ValueError(
                "verifier_name_or_path is required for MTP training. "
                "The verifier model must contain native MTP weights "
                "(mtp.* keys) to extract."
            )

        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
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
                # Read architectures from the verifier's published config.json (like
                # eagle3/dflash/peagle and the MTP converter). from_config would read
                # them off verifier_config, but that is the unwrapped text_config for
                # composite verifiers (e.g. Qwen3.5-MoE), which carries no
                # architectures -- leaving verifier.architectures empty.
                verifier=VerifierConfig.from_pretrained(verifier_name_or_path),
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
