"""MTP speculator model implementation."""

import logging
from typing import Any, ClassVar, Literal

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
from speculators.models.utils import (
    conditional_torch_compile,
    flatten_rope_parameters,
    resolve_verifier_norm_class,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig

logger = logging.getLogger(__name__)

__all__ = ["MTPDraftModel", "compute_step_weights"]

_IGNORE_INDEX = -100
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


# With head_dim 256 the default flex-attention tiles need more shared memory than
# Ada-class GPUs (e.g. L40S, ~100 KB per block) provide; smaller tiles fit.
_SMALL_SMEM_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_stages": 1,
    "BLOCK_M1": 16,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 16,
}
_SMALL_SMEM_BYTES = 128 * 1024


def _flex_kernel_options(device: torch.device) -> dict | None:
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    smem = getattr(
        props, "shared_memory_per_block_optin", props.shared_memory_per_block
    )
    return _SMALL_SMEM_FLEX_KERNEL_OPTIONS if smem < _SMALL_SMEM_BYTES else None


def _ttt_mask_mod(doc: torch.Tensor, n: int):
    """Mask for one TTT draft step over KV = [step 0 | 1 .. step], each ``n`` long.

    Mirrors vLLM's MTP proposer: the context KV in the cache was written by the
    first pass from verifier hidden states, and each later draft position adds
    its own KV. So the query for anchor t sees step-0 keys at positions <= t
    (same document) plus its own anchor's keys from later steps, and nothing
    from other anchors' draft chains.
    """

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        blk = kv_idx // n
        j = kv_idx % n
        context = (blk == 0) & (j <= q_idx) & (doc[j] == doc[q_idx])
        own_chain = (blk > 0) & (j == q_idx)
        return context | own_chain

    return mask_mod


def _ttt_block_mask(doc: torch.Tensor, n: int, step: int):
    """Flex block mask for draft step ``step`` (0-based); see `_ttt_mask_mod`."""
    from torch.nn.attention.flex_attention import create_block_mask  # noqa: PLC0415

    return create_block_mask(
        _ttt_mask_mod(doc, n),
        B=None,
        H=None,
        Q_LEN=n,
        KV_LEN=n * (step + 1),
        device=doc.device,
        # blockwise; eager materializes the dense Q x KV mask (~8 GB at 12k x 98k)
        _compile=True,
    )


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
        # SDPA, not eager — eager OOMs on long sequences (#886).
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = "sdpa"  # noqa: SLF001
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
            return hidden.new_zeros(0, dtype=torch.float32), targets.new_zeros(0)
        return torch.cat(losses), torch.cat(preds)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        step_weights: list[float] | None = None,
        return_dict: bool = True,
        train_forward: Literal["fastmtp", "vllm"] = "fastmtp",
        **kwargs: Any,
    ) -> tuple:
        """Multi-step MTP training forward.

        :param train_forward: ``"fastmtp"`` (default) is the FastMTP-style
            sequence-wide recursion on pre-norm hidden states. ``"vllm"`` trains the
            head the way vLLM's MTP proposer runs it: on the verifier's final-norm
            hidden states, with each draft step attending to the context plus only
            its own draft chain (see ``_forward_ttt``).
        :return: ``(draft_tokens, loss, metrics)``. ``draft_tokens[k]`` holds the
            step-k greedy predictions at the supervised positions only.
        """
        if train_forward not in ("fastmtp", "vllm"):
            raise ValueError(f"Unknown MTP train_forward {train_forward!r}")
        # The layer's attention reads its implementation at call time, including
        # in a gradient-checkpoint recompute during backward, so set it per forward
        # rather than restoring it afterwards. The FastMTP path builds its mask from
        # transformer_layer_config and must run with the matching implementation.
        attn_config: Any = self.mtp_layers[0].self_attn.config  # type: ignore[union-attr]
        attn_config._attn_implementation = (  # noqa: SLF001
            "flex_attention"
            if train_forward == "vllm"
            else self.config.transformer_layer_config._attn_implementation  # noqa: SLF001
        )
        if train_forward == "vllm":
            return self._forward_ttt(
                input_ids,
                self.verifier_norm(hidden_states),
                position_ids,
                loss_mask,
                step_weights,
            )
        return self._forward_recursive(
            input_ids,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            step_weights=step_weights,
            return_dict=return_dict,
            **kwargs,
        )

    # requires `dynamic=False`. See #876
    @conditional_torch_compile(dynamic=False)
    def _forward_recursive(
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

        :param input_ids: Token IDs [1, seq_len]. Serves as both the
            embedding source and the prediction target (offset by step+2).
        :param hidden_states: Hidden states from verifier [1, seq_len, hidden_size]
        :param attention_mask: Optional attention mask [1, seq_len]
        :param position_ids: Optional position IDs [batch, seq_len]. Qwen3.5
            also accepts the explicit MRoPE layout [3, batch, seq_len].
        :param loss_mask: Optional binary mask [1, seq_len]; 1=compute loss,
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

        step_pos_ids, rotary_step_pos_ids = _prepare_mtp_position_ids(
            position_ids,
            valid_len,
            self.config.transformer_layer_config.model_type,
        )
        causal_mask = create_causal_mask(
            config=self.config.transformer_layer_config,
            inputs_embeds=hidden_states[:, :valid_len],
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=step_pos_ids,
        )

        current_hidden = hidden_states
        anchors: torch.Tensor | None = (
            None  # positions whose step-0 target is supervised
        )
        run_correct: torch.Tensor | None = None  # steps 1..k all correct (vLLM-style)
        for step in range(effective_steps):
            step_hidden = current_hidden[:, :valid_len]
            step_embeds = self.embed_tokens(
                input_ids[:, step + 1 : step + 1 + valid_len]
            )
            step_pos_emb = self.rotary_emb(step_hidden, rotary_step_pos_ids)

            mtp_output = self.mtp_layers[0](
                hidden_states=step_hidden,
                token_embeddings=step_embeds,
                attention_mask=causal_mask,
                position_ids=step_pos_ids,
                position_embeddings=step_pos_emb,
            )

            step_targets = input_ids[:, step + 2 : step + 2 + valid_len]
            if loss_mask is not None:
                step_mask = loss_mask[:, step + 2 : step + 2 + valid_len].bool()
            else:
                step_mask = torch.ones_like(step_targets, dtype=torch.bool)

            # Only supervised positions need logits: the full LM head output is
            # seq_len x vocab per step (~8 GB at 16k x 248k in bf16). Even the
            # supervised slice (~1k x 248k, plus its fp32 copy inside the loss) adds
            # up across many steps, so LM head + CE are recomputed in backward
            # instead of keeping logits alive. The loss is identical either way.
            # Returns per-step greedy draft tokens [num_supervised] instead of logits.
            sel_targets = step_targets[step_mask]
            unreduced, sel_pred = self._head_ce_chunked(
                mtp_output[step_mask], sel_targets
            )
            all_logits.append(sel_pred)

            weight = step_weights[step] if step_weights is not None else 1.0
            valid_count = step_mask.sum()
            step_loss = weight * unreduced.sum() / valid_count.clamp(min=1)
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

        metrics["loss_sum"] = total_loss.detach().clone()
        metrics["loss_total"] = torch.tensor(1.0, device=device)

        return (all_logits, total_loss, metrics)

    def _forward_ttt(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None,
        loss_mask: torch.Tensor | None,
        step_weights: list[float] | None,
    ) -> tuple:
        """Multi-step MTP forward aligned with vLLM's MTP proposer (training-time test).

        Differences from `_forward_recursive` (FastMTP style), both matching vLLM:
        - Attention at step k >= 1 sees the step-0 KV (computed from verifier
          hidden states) for context positions <= anchor, plus only its own
          anchor's KV from earlier draft steps -- not a sequence-wide recursion.
        - Draft step k runs at position anchor + k (vLLM advances positions per
          draft step; `constant_draft_positions` is False for MTP).
        Targets, step weights, loss and metrics are identical to the recursive path.
        """
        input_ids = input_ids.long()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("TTT MTP forward expects packed batches of size 1")
        num_steps = self.config.num_speculative_steps
        if step_weights is not None and len(step_weights) != num_steps:
            raise ValueError(
                f"step_weights has {len(step_weights)} entries but "
                f"num_speculative_steps={num_steps}"
            )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        all_preds: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float | torch.Tensor] = {}
        effective_steps = min(num_steps, max(0, seq_len - 2))
        n = seq_len - effective_steps - 1
        if n <= 0 or effective_steps == 0:
            metrics["loss_sum"] = total_loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return (all_preds, total_loss, metrics)

        model_type = self.config.transformer_layer_config.model_type
        text_pos, rotary_pos = _prepare_mtp_position_ids(position_ids, n, model_type)
        doc = torch.cumsum(
            (text_pos[0] == 0).to(torch.int32), dim=0
        )  # packed docs restart at 0

        layer = self.mtp_layers[0]  # attention set to flex_attention by forward()
        # The cache is read-only per step (see _TTTCache), so it is safe to keep it
        # under gradient checkpointing.
        layer._can_checkpoint_with_cache = True  # type: ignore[assignment]  # noqa: SLF001

        kernel_options = _flex_kernel_options(device)
        kv_k: list[torch.Tensor] = []
        kv_v: list[torch.Tensor] = []
        anchors: torch.Tensor | None = None
        run_correct: torch.Tensor | None = None
        current_hidden = hidden_states[:, :n]
        for step in range(effective_steps):
            step_text_pos = text_pos + step
            step_rotary_pos = rotary_pos + step
            step_embeds = self.embed_tokens(input_ids[:, step + 1 : step + 1 + n])
            pos_emb = self.rotary_emb(current_hidden, step_rotary_pos)
            cache = _TTTCache(kv_k, kv_v)
            mtp_output = layer(
                hidden_states=current_hidden,
                token_embeddings=step_embeds,
                attention_mask=_ttt_block_mask(doc, n, step),
                position_ids=step_text_pos,
                position_embeddings=pos_emb,
                past_key_values=cache,
                kernel_options=kernel_options,
            )
            kv_k = [*kv_k, cache.new_k]
            kv_v = [*kv_v, cache.new_v]

            step_targets = input_ids[:, step + 2 : step + 2 + n]
            if loss_mask is not None:
                step_mask = loss_mask[:, step + 2 : step + 2 + n].bool()
            else:
                step_mask = torch.ones_like(step_targets, dtype=torch.bool)
            sel_targets = step_targets[step_mask]
            unreduced, sel_pred = self._head_ce_chunked(
                mtp_output[step_mask], sel_targets
            )
            all_preds.append(sel_pred)

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
                metrics[f"accept_pos_{step + 1}_sum"] = (
                    (run_correct & anchors).sum().float()
                )
                metrics[f"accept_pos_{step + 1}_total"] = anchors.sum().float()
                metrics[f"step_{step + 1}_acc_sum"] = correct.sum().float()
                metrics[f"step_{step + 1}_acc_total"] = step_mask.sum().float()

            current_hidden = mtp_output

        metrics["loss_sum"] = total_loss.detach().clone()
        metrics["loss_total"] = torch.tensor(1.0, device=device)
        return (all_preds, total_loss, metrics)

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
        train_kwargs: dict[str, Any] = {
            "step_weights": step_weights,
            "train_forward": kwargs.get("mtp_train_forward", "fastmtp"),
        }
        val_kwargs = train_kwargs.copy()

        return train_kwargs, val_kwargs
