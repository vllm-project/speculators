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
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.jetspec import JetSpecSpeculatorConfig
from speculators.models.jetspec.attention import create_causal_anchor_block_mask_mod
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids

from .metrics import compute_metrics


@SpeculatorModel.register("jetspec")
class JetSpecDraftModel(DraftVocabMixin, SpeculatorModel):
    """JetSpec causal parallel draft head.

    Produces logits for all block positions in a single forward pass using
    causal attention within blocks.  Each position conditions on the prefix
    context (via injected verifier KV cache) and all earlier positions in
    its block.  At training time, ground-truth tokens are used (teacher
    forcing); at inference time a tree-causal mask enables tree drafting.
    """

    config_class: ClassVar[type[JetSpecSpeculatorConfig]] = JetSpecSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
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
        config: JetSpecSpeculatorConfig,
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

        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(num_draft_layers)
            ]
        )

        self.norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(tl_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * tl_config.hidden_size,
            tl_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size
        self.kd_temperature = config.kd_temperature
        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "JetSpecDraftModel":
        """Create JetSpec model from training arguments."""
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
        block_size = kwargs.get("block_size", 16)
        kd_temperature = kwargs.get("kd_temperature", 1.0)

        config = JetSpecSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=block_size,
            aux_hidden_state_layer_ids=target_layer_ids,
            kd_temperature=kd_temperature,
            speculators_config=SpeculatorsConfig(
                algorithm="jetspec",
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
        """Get training and validation kwargs for JetSpec."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
        }
        return dict(shared), dict(shared)

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
    ):
        mask_mod, q_len, kv_len = create_causal_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

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
        gamma: float = 4.0,
        max_anchors: int = 3072,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        # --- anchor selection (reuse DFlash utility) ---
        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size
        )

        # --- attention mask (always causal within blocks) ---
        attn_mask = self._create_attention_mask(
            document_ids=document_ids,
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            device=device,
            sliding_window=None,
        )

        # --- block indices ---
        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # [num_anchors * block_size]

        # --- block token embeddings (actual tokens, not mask tokens) ---
        block_token_ids = input_ids[:, anchored_block_indices]
        block_embedding = self.embed_tokens(block_token_ids)
        # [1, num_anchors * block_size, hidden_size]

        # --- project fused hidden states ---
        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        # [1, total_seq_len, hidden_size]

        # --- position embeddings ---
        block_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        full_position_ids = torch.cat(
            [position_ids, block_position_ids.unsqueeze(0)], dim=1
        )
        position_embeddings = self.rotary_emb(hidden_states, full_position_ids)

        # --- targets: next-token prediction (no roll needed) ---
        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            # verifier_logits[j] predicts token at j+1 (standard causal LM)
            # For block position i at anchor+i, the model predicts anchor+i+1
            # Target = verifier_logits[anchor+i] = prediction for anchor+i+1
            targets = verifier_logits[:, anchored_block_indices]
            # [1, num_anchors * block_size, draft_vocab_size]

        # --- forward through decoder layers ---
        for layer in self.layers:
            block_embedding = layer(
                hidden_states=block_embedding,
                target_hidden=fc_output,
                attention_mask=attn_mask,
                position_ids=full_position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(block_embedding)
        logits = self.lm_head(hidden)
        # [1, num_anchors * block_size, vocab_size]

        # --- loss mask (all positions contribute, including pos 0) ---
        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )

        # --- compute loss and metrics ---
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics
