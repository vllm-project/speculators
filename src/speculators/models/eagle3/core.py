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
from speculators.models.eagle3.metrics import compute_metrics
from speculators.models.eagle3.model_definitions import model_classes
from speculators.models.eagle3.rotary_partial import install_partial_neox_rotary
from speculators.models.metrics import kl_div_loss, resolve_loss_fn
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig


def conditional_torch_compile(func):
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func)
    else:
        return func


def _wrap_qwen_omni_rotary_with_hf_layout(rotary_cls: type) -> type:
    """Adapt Qwen-Omni rotary to HF MRoPE layout.

    Qwen-Omni expects ``position_ids`` as ``[3, batch, seq_len]`` while our
    training path emits ``[batch, 3, seq_len]``. This wrapper only transposes
    the HF case to avoid MRoPE channel mis-broadcast and projection dim errors.
    """

    class HFLayoutMRoPE(rotary_cls):  # type: ignore[misc,valid-type]
        def forward(self, x, position_ids):  # type: ignore[override]
            # Only transpose clear HF layout ``[B, 3, T]``.
            # Other shapes pass through unchanged.
            if position_ids.dim() == 3 and position_ids.shape[1] == 3:
                position_ids = position_ids.transpose(0, 1).contiguous()
            return super().forward(x, position_ids)

    HFLayoutMRoPE.__name__ = f"{rotary_cls.__name__}HFLayout"
    HFLayoutMRoPE.__qualname__ = HFLayoutMRoPE.__name__
    return HFLayoutMRoPE


def _select_rotary_emb_class(
    tl_config: PretrainedConfig,
    default_cls: type,
) -> type:
    """Select an MRoPE-aware rotary class for multimodal Qwen draft configs."""
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
            "Falling back to the architecture default rotary embedding, which "
            "will ignore MRoPE and can cause a train/inference mismatch for "
            "multimodal inputs.",
            UserWarning,
            stacklevel=2,
        )
        return default_cls

    partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
    if partial_rotary_factor >= 1.0:
        return _wrap_qwen_omni_rotary_with_hf_layout(
            Qwen3OmniMoeThinkerTextRotaryEmbedding
        )

    # Align HF partial rotation with vLLM partial-neox behavior.
    # Keep native ``partial_rotary_factor``/``mrope_section`` unchanged.
    install_partial_neox_rotary()
    return _wrap_qwen_omni_rotary_with_hf_layout(
        _make_partial_mrope_rotary_cls(
            Qwen3OmniMoeThinkerTextRotaryEmbedding, partial_rotary_factor
        )
    )


def _make_partial_mrope_rotary_cls(
    base_cls: type, partial_rotary_factor: float
) -> type:
    """Return a partial-MRoPE rotary class for Qwen drafts.

    It emits unpadded ``[*, rotary_dim]`` cos/sin. The patched
    ``apply_rotary_pos_emb`` rotates only the leading ``rotary_dim`` channels
    and keeps the tail unchanged, matching vLLM partial-neox semantics.
    """

    class PartialMRoPE(base_cls):  # type: ignore[misc,valid-type]
        _partial_rotary_factor: ClassVar[float] = partial_rotary_factor

        @staticmethod
        def compute_default_rope_parameters(config=None, device=None, seq_len=None):
            base = config.rope_parameters["rope_theta"]
            head_dim = (
                getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads
            )
            rotary_dim = int(head_dim * PartialMRoPE._partial_rotary_factor)
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
            self._head_dim = head_dim
            self._rotary_dim = rotary_dim

        def forward(self, x, position_ids):
            # Keep cos/sin unpadded; patched apply_rotary_pos_emb handles
            # partial-neox rotation on the leading rotary channels.
            return super().forward(x, position_ids)

    PartialMRoPE.__name__ = (
        f"{base_cls.__name__}Partial{int(partial_rotary_factor * 100):03d}"
    )
    PartialMRoPE.__qualname__ = PartialMRoPE.__name__
    return PartialMRoPE


@SpeculatorModel.register("eagle3")
class Eagle3DraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[Eagle3SpeculatorConfig]] = Eagle3SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
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

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.eagle_aux_hidden_state_layer_ids

    def load_verifier_weights(self):
        super().load_verifier_weights()

        self.embed_tokens.weight.requires_grad_(self.config.embed_requires_grad)

        verifier_config = self.config.speculators_config.verifier
        verifier_model_config = AutoConfig.from_pretrained(verifier_config.name_or_path)  # type: ignore[arg-type]

        # For multimodal models (Qwen3VL/Omni/etc.), extract text_config
        if hasattr(verifier_model_config, "thinker_config"):
            verifier_model_config = verifier_model_config.thinker_config
        if hasattr(verifier_model_config, "text_config"):
            verifier_model_config = verifier_model_config.text_config

        if verifier_model_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Verifier hidden size {verifier_model_config.hidden_size} does not"
                f" match draft hidden size {self.hidden_size}."
            )

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
        loss_fn=kl_div_loss,
        **kwargs,
    ):
        loss_fn = loss_fn or kl_div_loss
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
                    loss_fn=loss_fn,
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
            metrics["loss_sum"] = loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
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
        # Resolve target layer IDs if not provided
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
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
        loss_fn = resolve_loss_fn(kwargs["loss_fn"])
        train_kwargs = {
            "use_off_policy_tokens": kwargs["use_off_policy_tokens"],
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_fn": loss_fn,
        }
        val_kwargs = {
            "use_off_policy_tokens": False,
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_fn": loss_fn,
        }
        return train_kwargs, val_kwargs
