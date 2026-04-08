from typing import TYPE_CHECKING, Any, ClassVar
import warnings

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import (
    AutoTokenizer,  # noqa: PLC0415
    PretrainedConfig,
)
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)
from typing_extensions import Unpack

from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.utils import (
    build_target_layer_ids,
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.utils.loading import load_model_layers

if TYPE_CHECKING:
    from collections.abc import Callable


# Local copy of rotate_half to avoid dependency on internal transformers functions
def _rotate_half(x):
    """Rotates half the hidden dims of the input (local implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,  # noqa: ARG001
    unsqueeze_dim=1,
):
    """Apply rotary position embeddings (local implementation)."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Implements the custom attention which injects the target models
    # hidden states into the kv cache.
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads  # type: ignore[operator]
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            config.hidden_size,  # type: ignore[arg-type]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"  # type: ignore[index]
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Instead of computing the k and v matricies from the hidden states,
        # the target_hidden is injected into the kv cache, (shape is context
        # length + block size)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        # This is the main difference from the usual attention mechanism.
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        # note the length becomes context length + block size
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,  # type: ignore[arg-type]
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        # necessary, but kept here for BC
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # The main difference between this method and the qwen 3 layer it is
        # built from is that it
        # passes the extra hidden states to the self attention from the verifier model.
        # Note that target_hidden is not modified here.
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states  # type: ignore[operator]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states  # type: ignore[operator,return-value]


@SpeculatorModel.register("dflash")
class DFlashDraftModel(SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ["embed_tokens.weight"]  # type: ignore[misc]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
    ]

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
        t2d: torch.Tensor,
        d2t: torch.Tensor,
    ) -> None:
        super().__init__(
            config=config,
            verifier=None,
            verifier_attachment_mode="train_only",
        )
        self.config = config
        # Set attention implementation to simple_flex_attention
        # to support BlockMask
        if (
            self.config.transformer_layer_config._attn_implementation  # noqa: SLF001
            is None
        ):
            impl = "simple_flex_attention"
            self.config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                impl
            )
        self.register_buffer("t2d", t2d)  # shape: [verifier_vocab_size], bool
        self.register_buffer("d2t", d2t)  # shape: [draft_vocab_size], int offsets
        self.draft_vocab_size = config.draft_vocab_size

        # Load verifier embeddings and tokenizer (following Eagle3 pattern)
        self._setup_embeddings_and_mask_token(config.speculators_config.verifier, t2d)

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = config.transformer_layer_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(num_draft_layers)
            ]
        )

        # Load actual verifier config to get the real verifier layer count
        from transformers import AutoConfig  # noqa: PLC0415

        verifier_name_or_path = config.speculators_config.verifier.name_or_path
        if verifier_name_or_path is None:
            raise ValueError("Verifier name_or_path must be set in speculators_config")
        verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
        if hasattr(verifier_config, "text_config"):
            verifier_config = verifier_config.text_config
        num_verifier_layers = verifier_config.num_hidden_layers

        self.target_layer_ids = build_target_layer_ids(
            num_verifier_layers, num_draft_layers
        )
        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.block_size = config.block_size
        self.post_init()

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional, creates
                identity mapping if None)
            d2t: Draft-to-target vocabulary mapping tensor (optional, creates
                identity mapping if None)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - max_anchors: Max anchor positions during training (default: 256)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        config = DFlashSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=kwargs.get("block_size", 8),
            max_anchors=kwargs.get("max_anchors", 256),
            speculators_config=SpeculatorsConfig(
                algorithm="dflash",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("block_size", 8),
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        # Create identity mappings if t2d/d2t not provided (no vocab reduction)
        if t2d is None or d2t is None:
            vocab_size = kwargs["draft_vocab_size"]
            # t2d: all tokens in target vocab are in draft vocab
            t2d = torch.ones(vocab_size, dtype=torch.bool)
            # d2t: identity mapping (zero offset for all tokens)
            d2t = torch.zeros(vocab_size, dtype=torch.long)

        return cls(config=config, t2d=t2d, d2t=d2t)

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:  # noqa: ARG004
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        train_kwargs: dict[str, Any] = {}
        val_kwargs: dict[str, Any] = {}
        return train_kwargs, val_kwargs

    def _setup_embeddings_and_mask_token(self, verifier_config, t2d):
        """Setup embeddings and mask_token_id from verifier."""

        if verifier_config.name_or_path is None:
            raise ValueError("VerifierConfig `name_or_path` value is required.")

        # Load embedding weights
        verifier_weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight"],
            verifier_config.name_or_path,
        )

        # Create embedding layer (config already available in self.config)
        self.embed_tokens = nn.Embedding(
            self.config.transformer_layer_config.vocab_size,
            self.config.transformer_layer_config.hidden_size,
            padding_idx=getattr(
                self.config.transformer_layer_config, "pad_token_id", None
            ),
        )

        default_dtype = self.embed_tokens.weight.dtype
        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        self.embed_tokens.load_state_dict(
            {"weight": embed_tokens_weight.to(dtype=default_dtype)}
        )
        self.embed_tokens.weight.requires_grad_(False)
        # Use embed_tokens as fallback for lm_head if not found (tied weights)
        lm_head_weight = verifier_weights["lm_head.weight"]

        self.lm_head = torch.nn.Linear(
            self.config.transformer_layer_config.hidden_size,
            self.draft_vocab_size,
            bias=False,
        )
        self.verifier_lm_head = torch.nn.Linear(
            self.config.transformer_layer_config.hidden_size,
            self.draft_vocab_size,
            bias=False,
        )
        masked_lm_head_weight = lm_head_weight.to(
            device=t2d.device, dtype=default_dtype
        )[t2d.to(torch.bool), :]

        self.lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False
        # Load tokenizer to get mask_token_id with fallbacks
        tokenizer = AutoTokenizer.from_pretrained(verifier_config.name_or_path)
        if tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            # Try special tokens in order of preference
            token_options = [
                ("pad_token_id", tokenizer.pad_token_id),
                ("eos_token_id", tokenizer.eos_token_id),
                ("unk_token_id", tokenizer.unk_token_id),
            ]
            for i, (token_name, token_id) in enumerate(token_options):
                if token_id is not None:
                    self.mask_token_id = token_id
                    warnings.warn(
                        f"Tokenizer does not have mask_token. Using {token_name}={token_id} as fallback.",
                        stacklevel=2,
                    )
                    break
            else:
                raise ValueError("No suitable special token found in tokenizer")
        # Save to config so it persists when saved
        self.config.mask_token_id = self.mask_token_id

    @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: ARG002, E501
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = self.config.max_anchors

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, num_anchors, self.block_size
        )
        # shape: [num_anchors], [num_anchors]

        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            lengths=lengths.to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
        )

        attention_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size] # noqa: ERA001

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        # shape: [1, total_seq_len, hidden_size] # noqa: ERA001

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[:, anchor_positions], self.block_size, input_ids.numel()
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size] # noqa: ERA001

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size, input_ids.numel()
        )  # shape: [num_anchors*block_size]

        targets = input_ids.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size] # noqa: ERA001

        for layer in self.layers:
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        logits = self.lm_head(self.norm(noise_embedding))
        # shape: [1, num_anchors*block_size, vocab_size] # noqa: ERA001

        # Convert targets from verifier vocab to draft vocab
        # t2d is a boolean mask [verifier_vocab_size] - True where
        # verifier token exists in draft
        # cumsum gives us the draft index for each verifier token
        draft_indices = torch.cumsum(self.t2d.long(), dim=0) - 1  # type: ignore[operator]
        targets_draft = torch.where(
            self.t2d[targets],  # type: ignore[index]
            draft_indices[targets],  # type: ignore[index]
            torch.tensor(-100, dtype=torch.long, device=device),
        )

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size] # noqa: ERA001

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        aligned_loss_mask[:, :: self.block_size] = 0
        loss, metrics = compute_metrics(
            logits, targets_draft, aligned_loss_mask, self.block_size
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics
