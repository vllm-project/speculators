from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import LossConfig, tv_loss
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.mamba_attn_hybrid.config import (
    MambaAttnHybridSpeculatorConfig,
)
from speculators.models.mamba_attn_hybrid.model_definitions import (
    AttentionSubLayer,
    MambaSubLayer,
    MLPSubLayer,
)
from speculators.models.utils import (
    conditional_torch_compile,
    flatten_rope_parameters,
    resolve_target_layer_ids,
    resolve_verifier_norm_class,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers

__all__ = [
    "MambaAttnHybridDraftModel",
]

_compiled_create_block_mask = torch.compile(create_block_mask)
_UNTRAINED_ROW_NORM = 1e-4


@SpeculatorModel.register("mamba_attn_hybrid")
class MambaAttnHybridDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class = MambaAttnHybridSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["MambaSubLayer", "AttentionSubLayer", "MLPSubLayer"]
    _keys_to_ignore_on_load_missing = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save = [
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: MambaAttnHybridSpeculatorConfig) -> None:
        super().__init__(config=config)
        self._init_vocab(config)
        tl_config = config.transformer_layer_config

        num_attention = config.block_pattern.count("attention")
        if {len(config.attn_kv_layer_ids), tl_config.num_hidden_layers} != {
            num_attention
        }:
            raise ValueError(
                f"block_pattern has {num_attention} attention sub-layers, but "
                f"attn_kv_layer_ids has {len(config.attn_kv_layer_ids)} entries and "
                f"num_hidden_layers is {tl_config.num_hidden_layers}."
            )
        layers: list[nn.Module] = []
        for block in config.block_pattern:
            if block == "mamba":
                layers.append(MambaSubLayer(tl_config, config))
            elif block == "attention":
                num_built = sum(isinstance(m, AttentionSubLayer) for m in layers)
                layers.append(AttentionSubLayer(tl_config, num_built))
            else:
                layers.append(MLPSubLayer(tl_config))
        self.layers = nn.ModuleList(layers)
        kv_layer_ids = sorted(set(config.attn_kv_layer_ids))
        self._kv_index = [kv_layer_ids.index(i) for i in config.attn_kv_layer_ids]
        self._windows = {
            layer.self_attn.sliding_window for layer in self.attention_layers()
        }

        hidden_size = tl_config.hidden_size
        self.norm = Qwen3RMSNorm(hidden_size, eps=tl_config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(
            cast("Qwen3Config", flatten_rope_parameters(tl_config))
        )
        self.fc = nn.Linear(
            len(config.latent_fusion_layer_ids) * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = Qwen3RMSNorm(hidden_size, eps=tl_config.rms_norm_eps)
        self.seed_projs = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size,
                    config.mamba_num_heads
                    * config.mamba_head_dim
                    * config.mamba_d_state,
                    bias=False,
                )
            ]
        )
        self.verifier_norm = resolve_verifier_norm_class(config)(
            hidden_size, eps=tl_config.rms_norm_eps
        )
        self.verifier_norm.weight.requires_grad = False

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
            )
        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            self.confidence_head = ConfidenceHead(
                hidden_size
                + (config.markov_rank if config.confidence_head_with_markov else 0)
            )

        self.block_size = config.block_size
        self.post_init()

    def attention_layers(self) -> list[AttentionSubLayer]:
        return [layer for layer in self.layers if isinstance(layer, AttentionSubLayer)]

    @property
    def target_layer_ids(self) -> list[int]:
        return self.config.latent_fusion_layer_ids

    @property
    def kv_cache_shape(self) -> tuple[int, int, int]:
        attn = self.attention_layers()[0].self_attn
        num_kv_heads = self.config.transformer_layer_config.num_key_value_heads
        return (
            len(set(self.config.attn_kv_layer_ids)),
            num_kv_heads,
            2 * attn.head_dim,
        )

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError("mask_token_id is not set on the config.")
        return self.config.mask_token_id

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "MambaAttnHybridDraftModel":
        if kwargs.get("draft_attn_impl", "simple_flex_attention") != (
            "simple_flex_attention"
        ):
            raise ValueError("mamba_attn_hybrid supports only simple_flex_attention.")
        if kwargs.get("sample_from_anchor"):
            raise ValueError("mamba_attn_hybrid does not support sample_from_anchor.")
        verifier = kwargs["verifier_name_or_path"]
        block_size = kwargs["block_size"]
        config = MambaAttnHybridSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=block_size,
            mask_token_id=kwargs["mask_token_id"],
            sliding_window_non_causal=kwargs["sliding_window_non_causal"],
            latent_fusion_layer_ids=resolve_target_layer_ids(
                kwargs["target_layer_ids"],
                verifier,
                trust_remote_code=kwargs.get("trust_remote_code", False),
            ),
            attn_kv_layer_ids=kwargs["attn_kv_layer_ids"],
            block_pattern=kwargs["block_pattern"],
            mamba_num_heads=kwargs["mamba_num_heads"],
            mamba_head_dim=kwargs["mamba_head_dim"],
            mamba_d_state=kwargs["mamba_d_state"],
            mamba_n_groups=kwargs["mamba_n_groups"],
            markov_rank=kwargs["markov_rank"],
            enable_confidence_head=kwargs["enable_confidence_head"],
            confidence_head_with_markov=kwargs["confidence_head_with_markov"],
            speculators_config=SpeculatorsConfig(
                algorithm="mamba_attn_hybrid",
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(verifier),
            ),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        model.init_from_verifier(verifier)
        return model

    def init_from_verifier(self, verifier: str) -> None:
        names = ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm")
        weights = load_model_layers(
            [
                f"model.layers.{layer_id}.self_attn.{name}.weight"
                for layer_id in sorted(set(self.config.attn_kv_layer_ids))
                for name in names
            ],
            verifier,
        )
        for layer_id, layer in zip(
            self.config.attn_kv_layer_ids, self.attention_layers(), strict=True
        ):
            for name in names:
                weight = weights.get(f"model.layers.{layer_id}.self_attn.{name}.weight")
                param = getattr(layer.self_attn, name).weight
                if weight is not None:
                    param.data.copy_(weight)
                param.requires_grad = name.endswith("proj") or weight is None

        embedding = self.embed_tokens.weight
        if embedding[self.mask_token_id].float().norm() < _UNTRAINED_ROW_NORM:
            generator = torch.Generator().manual_seed(0)
            row = torch.randn(embedding.shape[1], generator=generator)
            embedding.data[self.mask_token_id] = row * embedding.float().std()

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        return DSparkDraftModel.get_trainer_kwargs(**kwargs)

    @torch.compiler.disable
    def _attention_masks(self, document_ids, anchor_positions, total_seq_len):
        masks = {}
        for window in self._windows:
            mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
                document_ids=document_ids.squeeze(0),
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                block_size=self.block_size,
                sliding_window=window,
                sliding_window_non_causal=self.config.sliding_window_non_causal,
            )
            masks[window] = _compiled_create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=document_ids.device,
            )
        return masks

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        loss_config: LossConfig,
        position_ids: torch.Tensor | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **_kwargs,
    ):
        total_seq_len = input_ids.shape[1]
        block = self.block_size
        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, device=input_ids.device
            ).unsqueeze(0)

        anchor_positions, anchor_valid = select_anchors(loss_mask, max_anchors, block)
        anchor_valid = anchor_valid & (anchor_positions > 0)
        masks = self._attention_masks(document_ids, anchor_positions, total_seq_len)
        block_indices = get_base_indices_for_anchored_blocks(anchor_positions, block)

        mask_token_ids = torch.full_like(block_indices, self.mask_token_id)
        mask_token_ids[::block] = input_ids[0, anchor_positions]
        hidden = self.embed_tokens(mask_token_ids).unsqueeze(0)
        block_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], block
        )
        position_embeddings = self.rotary_emb(hidden, block_position_ids.unsqueeze(0))

        seed = self.hidden_norm(
            self.fc(hidden_states[0, (anchor_positions - 1).clamp(min=0)])
        )
        initial_states = self.seed_projs[0](seed).view(
            max_anchors,
            self.config.mamba_num_heads,
            self.config.mamba_head_dim,
            self.config.mamba_d_state,
        )
        verifier_kv = [
            kv.transpose(0, 1).unsqueeze(0).chunk(2, dim=-1)
            for kv in kv_cache[0].unbind(1)
        ]

        with torch.no_grad():
            targets = self.verifier_lm_head(
                self.verifier_norm(
                    verifier_last_hidden_states[:, (block_indices - 1) % total_seq_len]
                )
            )

        attention_idx = 0
        for layer in self.layers:
            if isinstance(layer, MambaSubLayer):
                hidden = layer(hidden, initial_states)
            elif isinstance(layer, AttentionSubLayer):
                hidden = layer(
                    hidden,
                    verifier_kv[self._kv_index[attention_idx]],
                    position_embeddings,
                    masks[layer.self_attn.sliding_window],
                )
                attention_idx += 1
            else:
                hidden = layer(hidden)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        block_tokens = input_ids[0, block_indices].view(max_anchors, block)
        prev_token_ids = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
        hidden_blocks = hidden.view(max_anchors, block, -1)
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(max_anchors, block, -1) + markov_bias).view_as(logits)

        confidence_logits = None
        if self.confidence_head is not None:
            features = hidden_blocks.detach()
            if prev_emb is not None and self.config.confidence_head_with_markov:
                features = torch.cat(
                    [features, prev_emb.detach().to(features.dtype)], dim=-1
                )
            confidence_logits = self.confidence_head(features).reshape(1, -1)

        aligned_loss_mask = loss_mask[
            :, block_indices
        ] * anchor_valid.repeat_interleave(block).unsqueeze(0).to(loss_mask.dtype)
        aligned_loss_mask[:, ::block] = 0

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            block,
            loss_config=loss_config,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=False,
        )
        return None, loss, metrics
