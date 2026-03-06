# ruff: noqa: ERA001

from typing import ClassVar

import torch
from torch.nn.attention.flex_attention import create_block_mask

from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import (
    create_anchor_block_mask_mod
)

from speculators.utils.loading import load_model_layers
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)



@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len - ttt_step]
    block_size:int =1,
):
    # Note: logits, targets, and loss_mask are already aligned for the current ttt_step
    # target_tokens = torch.argmax(targets, dim=-1)
    target_tokens=targets
    predicted_tokens = torch.argmax(logits, dim=-1)
    correct = predicted_tokens == target_tokens

    if block_size!=1:
        accs=[]
        for i in range(block_size):
            pos_cor=torch.masked_select(correct[:, i::block_size], loss_mask.to(torch.bool)[:, i::block_size])
            accs.append(pos_cor.float().sum()/(pos_cor.numel()+1e-5))

    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    correct_sum = correct.float().sum()
    full_denom = correct.numel()
    if block_size==1:
        return correct_sum / (full_denom + 1e-5)
    else: 
        return correct_sum / (full_denom + 1e-5), accs
import torch

def build_kv_position_ids(
    base_position_ids: torch.Tensor,   # [B, total_seq_len]
    anchor_positions: torch.Tensor,    # [B, n] or [n] (indices into base seq)
    block_size: int,
) -> torch.Tensor:
    """
    Construct position_ids for KV = [base | anchor_blocks].

    Appended block for anchor a gets positions:
        base_position_ids[..., a] + [0..block_size-1]
    """
    B, T = base_position_ids.shape
    device = base_position_ids.device

    # Normalize anchor_positions to [B, n]
    if anchor_positions.ndim == 1:
        anchor_positions = anchor_positions.to(device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    elif anchor_positions.ndim == 2:
        anchor_positions = anchor_positions.to(device=device, dtype=torch.long)
        if anchor_positions.shape[0] != B:
            raise ValueError(f"anchor_positions batch {anchor_positions.shape[0]} != {B}")
    else:
        raise ValueError(f"anchor_positions must be [n] or [B, n], got {anchor_positions.shape}")

    n = anchor_positions.shape[1]

    # Position id at each anchor: [B, n]
    anchor_pos_ids = torch.gather(base_position_ids.to(torch.long), dim=1, index=anchor_positions)

    # Offsets within each block: [1, 1, block_size]
    offsets = torch.arange(block_size, device=device, dtype=torch.long).view(1, 1, block_size)

    # [B, n, block_size] -> [B, n*block_size]
    appended_pos_ids = (anchor_pos_ids.unsqueeze(-1) + offsets).reshape(B, n * block_size)

    return torch.cat([base_position_ids.to(torch.long), appended_pos_ids], dim=1)




def gather_anchor_spans(input_ids: torch.Tensor, anchor_positions: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    input_ids: [T]
    anchor_positions: [n] (positions into input_ids)
    returns: [n*block_size] = input_ids[anchor_i + 0 .. anchor_i + block_size-1] concatenated
    """
    input_ids = input_ids.view(-1)
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)

    offsets = torch.arange(block_size, device=input_ids.device, dtype=torch.long)  # [block_size]
    idx = anchor_positions[:, None] + offsets[None, :]                              # [n, block_size]

    if (idx < 0).any() or (idx >= input_ids.numel()).any():
        raise ValueError("Some anchor_positions + offsets are out of range for input_ids.")

    return input_ids[idx.reshape(-1)]

def loss_function(logits, target_ids, loss_mask, block_size=8, gamma=4.0):
    """
    logits:     [B, T, V]  (draft vocab)
    target_ids: [B, T]     (int64, in [0..V-1] or -100 for ignore)
    loss_mask:  [B, T]     (0/1)
    """
    B, T, V = logits.shape

    # per-token CE (no reduction yet)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        target_ids.reshape(B * T),
        reduction="none",
        ignore_index=-100,
    ).view(B, T)  # [B,T]

    # aligned t -> original p=t+1 ; k = p % b (0 means anchor)
    idx = torch.arange(T, device=logits.device)
    k = (idx + 1) % block_size
    w = torch.exp(-((k - 1).clamp(min=0)).to(logits.dtype) / gamma)
    w = (w * (k != 0).to(logits.dtype)).view(1, T)  # anchors weight 0

    m = loss_mask.to(logits.dtype).view(B, T)

    ce = ce * w * m

    denom = (m * w).sum(dim=1) + 1e-5
    return (ce.sum(dim=1) / denom).mean()

@torch.no_grad()
def compute_acceptance_rate(
    draft_logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    target_logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len]
    block_size: int = 1,
):
    """
    Compute acceptance rate for each position in the block according to EAGLE 3 criteria.

    EAGLE 3 acceptance formula: acceptance_prob = min(1, p(token) / p_draft(token))
    where p is the target model's probability and p_draft is the draft model's probability.

    Args:
        draft_logits: Logits from the draft model
        target_logits: Logits from the target/verifier model
        loss_mask: Mask indicating which positions to include in the calculation
        block_size: Size of each block for position-wise calculation

    Returns:
        If block_size == 1: overall acceptance rate
        Otherwise: (overall acceptance rate, list of per-position acceptance rates)
    """
    # Convert logits to probabilities
    draft_probs = torch.nn.functional.softmax(draft_logits, dim=-1)
    target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

    # Get the draft tokens (what the draft model predicted)
    draft_tokens = torch.argmax(draft_logits, dim=-1)

    # Gather the probabilities for the draft tokens from both distributions
    # Shape: [1, total_seq_len]
    draft_token_probs_from_draft = torch.gather(
        draft_probs, dim=-1, index=draft_tokens.unsqueeze(-1)
    ).squeeze(-1)

    draft_token_probs_from_target = torch.gather(
        target_probs, dim=-1, index=draft_tokens.unsqueeze(-1)
    ).squeeze(-1)

    # Compute acceptance probability: min(1, p_target / p_draft)
    # Add epsilon to avoid division by zero
    acceptance_prob = torch.clamp(
        draft_token_probs_from_target / (draft_token_probs_from_draft + 1e-10),
        max=1.0
    )
    accepted = acceptance_prob

    if block_size != 1:
        acc_rates = []
        for i in range(block_size):
            pos_accepted = torch.masked_select(
                accepted[:, i::block_size],
                loss_mask.to(torch.bool)[:, i::block_size]
            )
            acc_rates.append(pos_accepted.float().mean())

    if loss_mask is not None:
        accepted = torch.masked_select(accepted, loss_mask.to(torch.bool))

    overall_acceptance = accepted.float().mean()

    if block_size == 1:
        return overall_acceptance
    else:
        return overall_acceptance, acc_rates

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    block_size: int=1,
) -> tuple[torch.Tensor, dict]:
    """Compute metrics for a given draft.

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
    s_loss = loss_function(logits, targets, loss_mask)
    if block_size==1:

        s_full_acc=compute_accuracy(logits, targets, loss_mask, block_size)
        # s_accept_rate=compute_acceptance_rate(logits, targets, loss_mask, block_size)
        s_metrics=0
        s_metrics = {}
        s_metrics[f"loss"] = s_loss.detach().clone()
        s_metrics[f"full_acc"] = s_full_acc
        # s_metrics[f"accept_rate"] = s_accept_rate
    else:
        s_full_acc, per_position_acc=compute_accuracy(logits, targets, loss_mask, block_size)
        # s_accept_rate, per_position_accept=compute_acceptance_rate(logits, targets, loss_mask, block_size)
        s_metrics=0
        s_metrics = {}
        s_metrics[f"loss"] = s_loss.detach().clone()
        s_metrics[f"full_acc"] = s_full_acc
        # s_metrics[f"accept_rate"] = s_accept_rate
        for pos in range(len(per_position_acc)):
            s_metrics[f"position {pos} acc"]=per_position_acc[pos]
            # s_metrics[f"position {pos} accept"]=per_position_accept[pos]
    return s_loss, s_metrics



from typing import Optional
from typing_extensions import Unpack, Tuple
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
)
from transformers import DynamicCache
from transformers.cache_utils import Cache
from .utils import build_target_layer_ids


# Local copy of rotate_half to avoid dependency on internal transformers functions
def _rotate_half(x):
    """Rotates half the hidden dims of the input (local implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings (local implementation)."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types") and config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
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
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _select_anchors(loss_mask: torch.Tensor, n: int, block_size: int) -> torch.Tensor:
    if loss_mask.ndim != 2:
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    B, T = loss_mask.shape
    valid_mask = loss_mask.bool().clone()

    if block_size > 0:
        valid_mask[:, T - block_size:] = False

    out = []
    for b in range(B):
        valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(1)
        if valid_indices.numel() < n:
            # raise ValueError(f"Row {b} has only {valid_indices.numel()} valid positions, need {n}")
            n=valid_indices.numel()
        perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
        out.append(valid_indices[perm[:n]])

    return torch.stack(out, dim=0)
    
    


@SpeculatorModel.register("dflash")
class DFlashDraftModel(SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ["embed_tokens.weight"]  # type: ignore[misc]

    def __init__(self, config: DFlashSpeculatorConfig, t2d: torch.Tensor, d2t: torch.Tensor) -> None:
        super().__init__(
            config=config,
            verifier=None,
            verifier_attachment_mode="train_only",
        )
        self.config = config
        self.register_buffer("t2d", t2d)  # shape: [verifier_vocab_size], bool
        self.register_buffer("d2t", d2t)  # shape: [draft_vocab_size], int offsets
        self.draft_vocab_size = config.draft_vocab_size

        # Load verifier embeddings and tokenizer (following Eagle3 pattern)
        self._setup_embeddings_and_mask_token(config.speculators_config.verifier, t2d)

        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Use the actual number of layers from the verifier model config
        num_verifier_layers = config.transformer_layer_config.num_hidden_layers
        self.target_layer_ids = build_target_layer_ids(num_verifier_layers, config.num_hidden_layers)
        self.norm = Qwen3RMSNorm(config.transformer_layer_config.hidden_size, eps=config.transformer_layer_config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)

        self.fc = nn.Linear(len(self.target_layer_ids) * config.transformer_layer_config.hidden_size, config.transformer_layer_config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.transformer_layer_config.hidden_size, eps=config.transformer_layer_config.rms_norm_eps)
        self.block_size = config.block_size
        # self.post_init()

    def _setup_embeddings_and_mask_token(self, verifier_config, t2d):
        """Setup embeddings and mask_token_id from verifier."""
        from transformers import AutoTokenizer
    
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
            padding_idx=getattr(self.config.transformer_layer_config, 'pad_token_id', None),
        )

        default_dtype = self.embed_tokens.weight.dtype
        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        self.embed_tokens.load_state_dict({"weight": embed_tokens_weight.to(dtype=default_dtype)})
        self.embed_tokens.weight.requires_grad_(False)
        vocab_size=int(t2d.sum().item())
        # Use embed_tokens as fallback for lm_head if not found (tied weights)
        lm_head_weight = verifier_weights['lm_head.weight']
        

        self.lm_head = torch.nn.Linear(
            self.config.transformer_layer_config.hidden_size, self.draft_vocab_size, bias=False
        )
        self.verifier_lm_head = torch.nn.Linear(
            self.config.transformer_layer_config.hidden_size, self.draft_vocab_size, bias=False
        )
        masked_lm_head_weight = lm_head_weight.to(
            device=t2d.device, dtype=default_dtype
        )[t2d.to(torch.bool), :]

        self.lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.requires_grad = False
        print("lm head shape", self.lm_head.weight.shape, flush=True)
        self.lm_head.weight.requires_grad=False
        # Load tokenizer to get mask_token_id
        tokenizer = AutoTokenizer.from_pretrained(verifier_config.name_or_path)
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        self.mask_token_id = tokenizer.mask_token_id

        print("mask token id:", self.mask_token_id, flush=True)


    # @torch.compile  # Temporarily disabled - compilation hangs
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, 5 * hidden_size]  #These are the hidden states from the target model.  
        input_ids:torch.Tensor, 
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, hidden_size]
        **kwargs,
    ):

        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]


        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(  #MEGAN FLAG CHECK THAT THIS SHOULD BE +1
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)
        
        #past_key_values = DynamicCache(config=self.config.transformer_layer_config)
        past_key_values=None
        anchor_positions=_select_anchors(loss_mask, 512, self.block_size)


        with torch.no_grad():
            padding=torch.sum(lengths)
        # combined_mask_mod = create_combined_mask_mod(lengths.to(device), total_seq_len, block_size=8, padding=padding)

        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            lengths=lengths.to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions[0],
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

        mask_tokens_size=self.block_size*len(anchor_positions[0])

        mask_token_ids=torch.full((1,mask_tokens_size ), self.mask_token_id, dtype=torch.long, device=device)
        mask_token_ids[:, ::self.block_size] = input_ids[:, anchor_positions[0]]
        # print(mask_token_ids)
        noise_embedding=self.embed_tokens(mask_token_ids)

        fc_output = self.fc(hidden_states)

        fc_output = self.hidden_norm(fc_output)

        position_ids = build_kv_position_ids(position_ids, anchor_positions, block_size=self.block_size)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            # with torch.no_grad():
            #     targets = self.verifier_lm_head(verifier_last_hidden_states).detach()
            # targets=input_ids
            targets=gather_anchor_spans( input_ids.clone(),anchor_positions,self.block_size).unsqueeze(0)

            loss = torch.tensor(0.0, device=device)
            metrics = {}
        tar_tok=torch.argmax(targets, dim=-1)

        tar_tok=tar_tok+self.d2t[tar_tok]
        for i, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=attention_mask,
                position_ids=position_ids, 
                past_key_value=past_key_values,
                use_cache=False,  #FLAG MEGAN
                position_embeddings=position_embeddings,
                **kwargs,
            )
        noise_embedding=self.norm(noise_embedding)

        logits=self.lm_head(noise_embedding)

        if return_loss: 
            aligned_logits = logits#[:, 1:]                   
            aligned_targets = targets#[:, :-1]        

            aligned_loss_mask = gather_anchor_spans(loss_mask.clone(), anchor_positions[0],self.block_size).unsqueeze(0) #[:, 1:].clone()

            b = self.block_size
            anchor_aligned = (((torch.arange(aligned_logits.shape[1], device=device) ) % b) == 0)  # t=7,15,23,...

            aligned_loss_mask[:, anchor_aligned] = 0    
            s_loss, s_metrics = compute_metrics(
                aligned_logits,
                aligned_targets,
                aligned_loss_mask,
                self.block_size
            )
            loss += s_loss
            metrics.update(s_metrics)
        draft_tokens=torch.argmax(logits, dim=-1)
        if return_loss:
            metrics["loss"] = loss.detach().clone()
            return draft_tokens, loss, metrics
        else:
            return draft_tokens