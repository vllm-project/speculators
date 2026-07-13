from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.dflare.config import DFlareSpeculatorConfig
from speculators.models.dflare.model_definitions import (
    Qwen3DFlareDecoderLayer,
)
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import (
    Qwen3DFlashDecoderLayer,
)
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import (
    LossConfig,
    resolve_loss_config,
)
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids

__all__ = [
    "DFlareDraftModel",
]


@SpeculatorModel.register("dflare")
class DFlareDraftModel(DraftVocabMixin, SpeculatorModel):
    """DFlash with adaptive layer fusion and heterogeneous KV projections.

    Instead of a single FC-projected shared representation for all draft layers,
    each draft layer learns its own weighted combination of target hidden states
    via D x T learnable scalar fusion weights.
    """

    config_class: ClassVar[type[DFlareSpeculatorConfig]] = DFlareSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlareDecoderLayer", "Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(
        self,
        config: DFlareSpeculatorConfig,
    ) -> None:
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

        tl_config = config.transformer_layer_config

        num_draft_layers = tl_config.num_hidden_layers
        num_target_layers = len(self.target_layer_ids)

        layer_cls = (
            Qwen3DFlareDecoderLayer
            if config.use_heterogeneous_kv
            else Qwen3DFlashDecoderLayer
        )
        self.layers = nn.ModuleList(
            [
                layer_cls(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(num_draft_layers)
            ]
        )
        self.sliding_window = tl_config.sliding_window
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(tl_config.layer_types)
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(tl_config)  # type: ignore[arg-type]

        # Adaptive layer fusion: D x T learnable scalar weights
        self.fusion_weights = nn.Parameter(
            torch.zeros(num_draft_layers, num_target_layers)
        )
        self.fusion_norms = nn.ModuleList(
            [
                Qwen3RMSNorm(tl_config.hidden_size, eps=tl_config.rms_norm_eps)
                for _ in range(num_draft_layers)
            ]
        )

        self.verifier_norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size
        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        return self.config.aux_hidden_state_layer_ids

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlareDraftModel":
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        block_size = kwargs.get("block_size", 8)
        config = DFlareSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=block_size,
            aux_hidden_state_layer_ids=target_layer_ids,
            mask_token_id=kwargs.get("mask_token_id"),
            sliding_window_non_causal=kwargs.get("sliding_window_non_causal", False),
            use_heterogeneous_kv=kwargs.get("use_heterogeneous_kv", True),
            progressive_gamma=kwargs.get("progressive_gamma", True),
            gamma_start=kwargs.get("gamma_start", 4.5),
            gamma_max=kwargs.get("gamma_max", 10.5),
            speculators_config=SpeculatorsConfig(
                algorithm="dflare",
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
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.5)
        max_anchors = kwargs.get("max_anchors", 3072)
        progressive_gamma = kwargs.get("progressive_gamma", True)
        gamma_start = kwargs.get("gamma_start", 4.5)
        gamma_max = kwargs.get("gamma_max", 10.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "progressive_gamma": progressive_gamma,
            "gamma_start": gamma_start,
            "gamma_max": gamma_max,
        }
        return dict(shared), dict(shared)

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
            sliding_window_non_causal=sliding_window_non_causal,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size
        )

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    def _compute_layer_fused_hidden(
        self,
        hidden_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute per-layer fused target hidden states.

        Returns a list of D tensors, each [1, total_seq_len, hidden_size].
        """
        num_target_layers = len(self.target_layer_ids)
        hidden_size = hidden_states.shape[-1] // num_target_layers
        # Split into per-target-layer representations
        per_layer = hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            num_target_layers,
            hidden_size,
        )
        # per_layer: [1, seq, T, hidden_size]

        # Compute softmax fusion weights: [D, T]
        alpha = torch.softmax(self.fusion_weights, dim=-1)

        fused = []
        for i in range(len(self.layers)):
            # Weighted sum: [1, seq, hidden_size]
            w = alpha[i]  # [T]
            combined = torch.einsum("bstd,t->bsd", per_layer, w)
            combined = self.fusion_norms[i](combined)
            fused.append(combined)
        return fused

    def set_training_steps(self, total_steps: int) -> None:
        """Set total training steps for progressive gamma schedule."""
        self._total_training_steps = total_steps
        self._current_training_step = 0

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.5,
        max_anchors: int = 3072,
        progressive_gamma: bool = True,
        gamma_start: float = 4.5,
        gamma_max: float = 10.5,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, max_anchors, document_ids, device)
        )

        mask_tokens_size = max_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)

        # Adaptive layer fusion: per-layer fused representations
        layer_fused = self._compute_layer_fused_hidden(hidden_states)

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=layer_fused[layer_idx],
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        aligned_loss_mask[:, :: self.block_size] = 0

        # Progressive gamma: linearly interpolate from gamma_start to gamma_max
        effective_gamma = gamma
        if progressive_gamma and self.training:
            total = getattr(self, "_total_training_steps", 0)
            step = getattr(self, "_current_training_step", 0)
            if total > 0:
                progress = min(step / total, 1.0)
                effective_gamma = gamma_start + progress * (gamma_max - gamma_start)
                self._current_training_step = step + 1

        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=effective_gamma,
            loss_config=loss_config,
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics
