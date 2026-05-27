# ruff: noqa: ERA001
import copy
import warnings
from typing import ClassVar

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoConfig, DynamicCache, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
    extend_mask_for_draft_tokens,
)
from speculators.models.eagle3.model_definitions import model_classes
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers


def _select_rotary_emb_class(
    tl_config: PretrainedConfig, default_cls: type,
) -> type:
    """Pick the rotary embedding class based on the draft config.

    When the draft ``transformer_layer_config`` carries ``rope_parameters``
    with ``mrope_section`` (inherited from a Qwen3-Omni / Qwen3-VL style
    multimodal verifier), we must use the MRoPE rotary so the returned
    cos/sin encode the 3D T/H/W position channels via the interleaved
    section split. Standard ``LlamaRotaryEmbedding`` / ``Qwen3RotaryEmbedding``
    would silently collapse 3D ``position_ids`` back to 1D and train the
    drafter with text-only rotary, producing a train/inference mismatch.

    We also honor ``partial_rotary_factor``: upstream
    ``Qwen3OmniMoeThinkerTextRotaryEmbedding`` hard-codes
    ``dim = head_dim`` in ``compute_default_rope_parameters`` (i.e. it rotates
    every head dim, ignoring ``partial_rotary_factor``). vLLM's
    ``MRotaryEmbeddingInterleaved`` respects ``partial_rotary_factor`` and
    rotates only the first ``rotary_dim = head_dim * partial_rotary_factor``
    dims of each head. Training under the former while serving under the
    latter is the exact mismatch that blew up Qwen3.6 Eagle3-v4 acceptance
    (cos-sim of draft logits dropped from >0.99 to ~0.93, ~17pp loss on
    pos0 acceptance). We subclass the Qwen3-Omni rotary so ``inv_freq``
    spans ``rotary_dim // 2`` entries and the returned cos/sin are identity-
    padded on the trailing ``head_dim - rotary_dim`` channels — making HF's
    ``apply_rotary_pos_emb(q, k, cos, sin)`` a no-op there, byte-matching
    vLLM's "rotate first ``rotary_dim``, pass through the rest" semantics.

    Falls back to ``default_cls`` (architecture default) for any config
    without ``mrope_section`` so text-only training is unaffected and no
    new imports are pulled in.
    """
    rope_params = getattr(tl_config, "rope_parameters", None)
    has_mrope = (
        isinstance(rope_params, dict) and rope_params.get("mrope_section") is not None
    )
    if not has_mrope:
        return default_cls

    try:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: PLC0415
            Qwen3OmniMoeThinkerTextRotaryEmbedding,
        )
    except ImportError:
        warnings.warn(
            "Draft config carries rope_parameters.mrope_section but the installed "
            "transformers does not expose Qwen3OmniMoeThinkerTextRotaryEmbedding. "
            "Falling back to the architecture's default rotary embedding, which "
            "will IGNORE the MRoPE section and produce a train/inference "
            "mismatch for multimodal inputs.",
            UserWarning,
            stacklevel=2,
        )
        return default_cls

    partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
    if partial_rotary_factor >= 1.0:
        # Whole head is rotated; upstream class already does the right thing
        # and — importantly — HF's rotate_half-based apply_rotary_pos_emb
        # pairs channels (i, i+head_dim/2) identically to vLLM's neox
        # partial rotation when rotary_dim == head_dim. This is the ONLY
        # regime where the trainer and vLLM forward are bit-equivalent for
        # MRoPE drafts; ``scripts/train.py::--draft-mrope-full-head-hack``
        # forces verifier configs with partial_rotary_factor<1 into this
        # regime at config-construction time.
        return Qwen3OmniMoeThinkerTextRotaryEmbedding
    # WARNING: PartialMRoPE below shrinks inv_freq to rotary_dim//2 and pads
    # cos/sin back to head_dim with [real | ones | real | ones], but HF's
    # rotate_half pairs channels (i, i+head_dim/2) while vLLM's neox-partial
    # rotation pairs (i, i+rotary_dim/2). These pairings DIFFER whenever
    # rotary_dim < head_dim, so PartialMRoPE is NOT bit-equivalent to vLLM
    # at inference time — and empirically (Eagle3-v5 experiments) drafts
    # trained with PartialMRoPE lose ~10pp acceptance vs the same weights
    # served under a rescaled partial=1.0 config hack. Prefer training with
    # --draft-mrope-full-head-hack (default) instead of relying on
    # PartialMRoPE. This branch is preserved only for the future Option 1
    # rewrite (vLLM-compatible partial rotation inside the trainer's
    # attention), at which point the pairing issue will be resolved upstream.
    warnings.warn(
        "Eagle3 draft config has partial_rotary_factor="
        f"{partial_rotary_factor} < 1.0. PartialMRoPE will be used, but it "
        "does NOT produce vLLM-bit-equivalent rotation (HF's rotate_half "
        "pairs (i,i+head_dim/2) whereas vLLM's neox-partial pairs "
        "(i,i+rotary_dim/2) — these differ when rotary_dim<head_dim). "
        "This causes a silent train/inference mismatch at serving time. "
        "Recommended fix: retrain with "
        "scripts/train.py --draft-mrope-full-head-hack (default ON), "
        "which rescales mrope_section by 1/partial_rotary_factor and pins "
        "partial_rotary_factor=1.0 so HF and vLLM pairings coincide.",
        UserWarning,
        stacklevel=2,
    )
    return _make_partial_mrope_rotary_cls(
        Qwen3OmniMoeThinkerTextRotaryEmbedding, partial_rotary_factor
    )


def _make_partial_mrope_rotary_cls(
    base_cls: type, partial_rotary_factor: float
) -> type:
    """Return a subclass of the Qwen3-Omni MRoPE rotary that respects
    ``partial_rotary_factor``.

    Differences from the base class:
      * ``compute_default_rope_parameters`` uses ``dim = int(head_dim *
        partial_rotary_factor)`` instead of ``head_dim``. This shrinks
        ``inv_freq`` to ``rotary_dim // 2`` entries.
      * ``forward`` pads the returned cos/sin back out to ``head_dim`` with
        ``cos=1`` and ``sin=0`` in the trailing channels, following HF's
        ``cat((freqs, freqs), -1)`` layout so that ``apply_rotary_pos_emb``
        leaves the trailing ``head_dim - rotary_dim`` channels untouched.

    The ``mrope_section`` assertion ``sum(mrope_section) == rotary_dim // 2``
    is implicit in ``apply_interleaved_mrope`` (it only writes freq indices
    up to ``mrope_section[i] * 3``); the vLLM side enforces it explicitly.
    Keep training and export configs consistent: ``sum(mrope_section)`` must
    equal ``int(head_dim * partial_rotary_factor) // 2``.
    """

    class PartialMRoPE(base_cls):  # type: ignore[misc,valid-type]
        _partial_rotary_factor: ClassVar[float] = partial_rotary_factor

        @staticmethod
        def compute_default_rope_parameters(
            config=None, device=None, seq_len=None,
        ):
            base = config.rope_parameters["rope_theta"]
            head_dim = (
                getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads
            )
            rotary_dim = int(head_dim * PartialMRoPE._partial_rotary_factor)
            # Ensure rotary_dim is even so cat((freqs, freqs), -1) yields an
            # even-length block suitable for rotate_half-style pairing.
            rotary_dim = (rotary_dim // 2) * 2
            attention_factor = 1.0
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float
                    )
                    / rotary_dim
                )
            )
            return inv_freq, attention_factor

        def __init__(self, config, device=None):
            super().__init__(config=config, device=device)
            head_dim = (
                getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads
            )
            rotary_dim = int(head_dim * self._partial_rotary_factor)
            rotary_dim = (rotary_dim // 2) * 2
            # Persisted so ``forward`` can size the identity-padding without
            # re-reading config.
            self._head_dim = head_dim
            self._rotary_dim = rotary_dim
            self._pad_half = (head_dim - rotary_dim) // 2

        def forward(self, x, position_ids):
            cos, sin = super().forward(x, position_ids)
            # Upstream returns cos/sin with last dim == rotary_dim (because
            # inv_freq was built on rotary_dim). Identity-pad so consumers
            # that expect head_dim-wide cos/sin still work.
            if cos.shape[-1] == self._head_dim:
                return cos, sin  # nothing to pad
            pad_shape = (*cos.shape[:-1], self._pad_half)
            pad_cos = torch.ones(pad_shape, dtype=cos.dtype, device=cos.device)
            pad_sin = torch.zeros(pad_shape, dtype=sin.dtype, device=sin.device)
            # HF MRoPE layout: emb = cat((freqs, freqs), -1) — two repeats of
            # the rotary block. We mirror that structure when padding so that
            # rotate_half's pairing (i, i + head_dim//2) keeps real<->real
            # inside the rotary region and pad<->pad outside it:
            #     [rot_first_half | pad | rot_second_half | pad]
            # where rot_first_half == rot_second_half == cos_rotary.
            half = cos.shape[-1] // 2
            cos_first, cos_second = cos[..., :half], cos[..., half:]
            sin_first, sin_second = sin[..., :half], sin[..., half:]
            cos_padded = torch.cat(
                [cos_first, pad_cos, cos_second, pad_cos], dim=-1
            )
            sin_padded = torch.cat(
                [sin_first, pad_sin, sin_second, pad_sin], dim=-1
            )
            return cos_padded, sin_padded

    PartialMRoPE.__name__ = (
        f"{base_cls.__name__}Partial{int(partial_rotary_factor * 100):03d}"
    )
    PartialMRoPE.__qualname__ = PartialMRoPE.__name__
    return PartialMRoPE


def align_for_step(
    logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, total_seq_len]
    ttt_step: int,
):
    """Align logits, targets, loss_mask, and prev_correct for a given ttt_step.

    There are no target values for the last ttt_step tokens, so we mask them out
    before computing the loss/accuracy. Likewise, there are no logits for the first
    ttt_step tokens, so we mask them out.
    This is equivalent to shifting the target values by ttt_step + 1 to the left
    which puts them in the correct position for the generated tokens.
    e.g.
        indices of targets = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices of logits for ttt_step_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9] # no shift
        indices of logits for ttt_step_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10] # shift by 1
        indices of logits for ttt_step_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11] # shift by 2
    The indices for the loss_mask need to be kept in line with the targets indices
    """
    logits = logits[:, :-ttt_step] if ttt_step > 0 else logits
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets = targets[:, ttt_step:]
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    if loss_mask is not None:
        loss_mask = loss_mask[:, ttt_step:]
        # shape: [1, total_seq_len - ttt_step]
    if prev_correct is not None:
        # Align with draft starts
        prev_correct = prev_correct[:, :-ttt_step] if ttt_step > 0 else prev_correct
        # shape: [1, total_seq_len - ttt_step]
    return logits, targets, loss_mask, prev_correct


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len - ttt_step]
    prev_correct: torch.Tensor | None,  # shape: [1, total_seq_len - ttt_step]
):
    # Note: logits, targets, and loss_mask are already aligned for the current ttt_step
    target_tokens = torch.argmax(targets, dim=-1)
    predicted_tokens = torch.argmax(logits, dim=-1)
    # shape: [1, total_seq_len - ttt_step]

    correct = predicted_tokens == target_tokens
    cond_denom: torch.Tensor | int = correct.numel()
    if prev_correct is not None:
        cond_denom = prev_correct.sum()
        # Update prev_correct in place
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))

    correct_sum = correct.float().sum()
    full_denom = correct.numel()

    return correct_sum / (full_denom + 1e-5), correct_sum / (cond_denom + 1e-5)


def loss_function(
    logits: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len - ttt_step]
):
    # Note: logits, targets, and loss_mask are already aligned for the current ttt_step
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    elementwise_loss = torch.nn.functional.kl_div(
        logits, target_p, reduction="none", log_target=False
    )

    if loss_mask is not None:
        elementwise_loss = elementwise_loss * loss_mask.unsqueeze(-1)
        denominator: torch.Tensor | int = loss_mask.sum(dim=1) + 1e-5
    else:
        denominator = logits.shape[1]  # total_seq_len - ttt_step
    batch_loss = torch.sum(elementwise_loss, dim=(1, 2)) / denominator
    # shape: [1]
    return batch_loss.mean()


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    prev_correct: torch.Tensor | None,
    ttt_step: int,
    ttt_step_loss_decay: float,
) -> tuple[torch.Tensor, dict]:
    """Compute metrics for a given ttt_step.

    Args:
        logits: The logits for the current ttt_step.
        targets: The targets for the current ttt_step.
        loss_mask: The loss mask for the current ttt_step.
        prev_correct: The previous correct predictions for the current ttt_step.
        ttt_step: The current ttt_step.
        ttt_step_loss_decay: The loss decay for the current ttt_step.

    Effects:
        Modifies prev_correct in place.

    Returns:
        Loss value and metrics dictionary.
    """

    s_metrics = {}
    s_logits, s_targets, s_loss_mask, s_prev_correct = align_for_step(
        logits, targets, loss_mask, prev_correct, ttt_step
    )
    loss_weight = ttt_step_loss_decay**ttt_step
    s_loss = loss_weight * loss_function(s_logits, s_targets, s_loss_mask)

    s_full_acc, s_cond_acc = compute_accuracy(
        s_logits, s_targets, s_loss_mask, s_prev_correct
    )
    s_metrics[f"loss_{ttt_step}"] = s_loss.detach().clone()
    s_metrics[f"full_acc_{ttt_step}"] = s_full_acc
    s_metrics[f"cond_acc_{ttt_step}"] = s_cond_acc

    return s_loss, s_metrics


def conditional_torch_compile(func):
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func)
    else:
        return func


@SpeculatorModel.register("eagle3")
class Eagle3DraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[Eagle3SpeculatorConfig]] = Eagle3SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "d2t",
        "t2d",
        "input_norm.weight",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: Eagle3SpeculatorConfig):
        # Forcibly override config settings
        impl = "simple_flex_attention"
        config.transformer_layer_config._attn_implementation = impl  # noqa: SLF001
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = self.config.transformer_layer_config
        self._model_definitions = model_classes[tl_config.model_type]

        # Eagle3-specific: embed_tokens grad depends on config
        self.embed_tokens.weight.requires_grad = self.config.embed_requires_grad

        # FC LAYER
        self.fc = torch.nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)

        # DECODER LAYERS
        num_layers = tl_config.num_hidden_layers
        fl_class = self._model_definitions.first_layer_class
        dl_class = self._model_definitions.decoder_layer_class
        layers = [
            fl_class(  # first layer
                tl_config,
                layer_idx=0,
                norm_before_residual=self.config.norm_before_residual,
            )
        ]
        layers.extend(  # remaining layers
            [dl_class(tl_config, layer_idx) for layer_idx in range(1, num_layers)]
        )
        self.layers = torch.nn.ModuleList(layers)

        # ROTARY EMBEDDINGS
        # Create a modified config for the rotary embedding to use 2x the hidden size
        modified_tl_config = copy.copy(config.transformer_layer_config)
        modified_tl_config.hidden_size *= 2
        rotary_cls = _select_rotary_emb_class(
            modified_tl_config, self._model_definitions.rotary_emb_class
        )
        self.rotary_emb = rotary_cls(modified_tl_config)

        # LAYER NORMS
        norm_class = self._model_definitions.norm_class
        self.norm = norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self.verifier_norm = norm_class(self.hidden_size, eps=tl_config.rms_norm_eps)
        self.verifier_norm.weight.requires_grad = False

        # Normalize draft path input (gpt-oss only)
        if config.norm_before_fc:
            self.input_norm = self._model_definitions.norm_class(
                3 * self.hidden_size,
                eps=config.transformer_layer_config.rms_norm_eps,
            )
        else:
            self.input_norm = None

        self.post_init()

    def load_verifier_weights(self):  # noqa: C901
        super().load_verifier_weights()

        verifier_config = self.config.speculators_config.verifier
        verifier_model_config = AutoConfig.from_pretrained(verifier_config.name_or_path)  # type: ignore[arg-type]

        # For multimodal models (Qwen3VL, etc.), extract text_config
        if hasattr(verifier_model_config, "text_config"):
            verifier_model_config = verifier_model_config.text_config

        if verifier_model_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Verifier hidden size {verifier_model_config.hidden_size} does not"
                f" match draft hidden size {self.hidden_size}."
            )

        # Load verifier norm weights
        verifier_weights = load_model_layers(
            ["model.norm.weight"],
            verifier_config.name_or_path,  # type: ignore[arg-type]
        )

        if "model.norm.weight" not in verifier_weights:
            warnings.warn(
                f"Could not find final norm weights in {verifier_config.name_or_path}. "
                "Using default initialization (weight=1.0).",
                UserWarning,
                stacklevel=2,
            )
        else:
            verifier_norm_sd = {"weight": verifier_weights["model.norm.weight"]}
            self.verifier_norm.load_state_dict(verifier_norm_sd)

    @conditional_torch_compile
    def forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, hidden_size]
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)
            # shape: [1, total_seq_len]

        past_key_values = DynamicCache(config=self.config.transformer_layer_config)

        combined_mask_mod = create_combined_mask_mod(lengths.to(device), total_seq_len)
        # Note: Attention mask is stored as a BlockMask object
        attention_mask = create_block_mask(
            combined_mask_mod,
            B=None,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
        )

        if self.input_norm is not None:
            hidden_states = self.input_norm(hidden_states)
        hidden_states = self.fc(hidden_states)
        # shape: [1, total_seq_len, hidden_size]

        original_input_ids = input_ids.detach().clone()
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                # shape: [1, total_seq_len, draft_vocab_size]
            loss = torch.tensor(0.0, device=device)

            # prev_correct is a boolean tensor that is True for tokens that have been
            # correctly predicted on all previous ttt_steps.
            # Initialized to True if the token is included in the loss_mask
            # or if there is no loss_mask
            prev_correct = (
                loss_mask.clone()
                if loss_mask is not None
                else torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
            )
            metrics = {}

        draft_tokens = []
        for ttt_step in range(ttt_steps):
            with torch.no_grad():
                input_embeds = self.embed_tokens(input_ids)
                # shape: [1, total_seq_len, hidden_size]
            cache_position = torch.arange(
                ttt_step * total_seq_len,
                (ttt_step + 1) * total_seq_len,
                dtype=torch.long,
                device=device,
            )
            # shape: [total_seq_len]

            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
            # shape: [1, total_seq_len, 2 * hidden_size]

            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            logits = self.lm_head(self.norm(hidden_states))
            # shape: [1, total_seq_len, draft_vocab_size]

            if return_loss:
                s_loss, s_metrics = compute_metrics(
                    logits,
                    targets,
                    loss_mask,
                    prev_correct,
                    ttt_step,
                    ttt_step_loss_decay,
                )
                loss += s_loss
                metrics.update(s_metrics)

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())
            # shape: [1, total_seq_len]
            # Use d2t to map draft tokens to verifier tokens.
            # Must be in verifier vocabulary space because we use the full verifier
            # vocabulary in the embedding.
            if self.d2t is not None:
                input_ids = input_ids + self.d2t[input_ids]  # type: ignore[index]

            if use_off_policy_tokens:
                # Overwrite input_ids with ground truth tokens
                # shift input_ids by 1 to the left and pad with 0
                # note: inputs_ids no longer line up with verifier_last_hidden_states
                # the draft logits generated from the padded tokens are ignored
                # and sliced out for loss calculation
                input_ids = torch.cat(
                    [
                        original_input_ids[:, 1 + ttt_step :],
                        original_input_ids.new_zeros(1, 1 + ttt_step),
                    ],
                    dim=-1,
                )
                # shape: [1, total_seq_len]

            attention_mask = extend_mask_for_draft_tokens(attention_mask)
            position_ids = position_ids + 1
            # shape: [1, total_seq_len]

        if return_loss:
            metrics["loss"] = loss.detach().clone()
            return draft_tokens, loss, metrics
        else:
            return draft_tokens

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "Eagle3DraftModel":
        """Create Eagle3 model from training arguments.

        Args:
            verifier_config: Verifier model configuration
            **kwargs: Training arguments with Eagle3-specific params
                - num_layers: Number of decoder layers
                - norm_before_residual: Whether to normalize before residual connection
                - t2d: Target-to-draft vocabulary mapping tensor
                - d2t: Draft-to-target vocabulary mapping tensor
                - ttt_steps: Number of TTT steps
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized Eagle3DraftModel
        """
        target_layer_ids = kwargs.get("target_layer_ids")
        if target_layer_ids is None:
            unmodified_verifier_config = AutoConfig.from_pretrained(
                kwargs["verifier_name_or_path"]
            )
            num_target_layers = unmodified_verifier_config.num_hidden_layers
            target_layer_ids = [2, num_target_layers // 2, num_target_layers - 3]
            warnings.warn(
                "--target-layer-ids is not explicitly set. Setting target "
                f"layers to {target_layer_ids}. If custom target layers were used "
                "when launching vllm datagen, please set them explicitly.",
                stacklevel=2,
            )

        config = Eagle3SpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs["norm_before_residual"],
            norm_before_fc=kwargs.get("norm_before_fc", False),
            embed_requires_grad=kwargs.get("embed_requires_grad", False),
            eagle_aux_hidden_state_layer_ids=target_layer_ids,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs["ttt_steps"],
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for Eagle3.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        train_kwargs = {
            "use_off_policy_tokens": kwargs["use_off_policy_tokens"],
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
        }
        val_kwargs = {
            "use_off_policy_tokens": False,
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
        }
        return train_kwargs, val_kwargs
