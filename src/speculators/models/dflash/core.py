import warnings
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import (
    AutoTokenizer,  # noqa: PLC0415
    PretrainedConfig,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
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
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
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

        # Use aux_hidden_state_layer_ids from config if present
        if config.aux_hidden_state_layer_ids is not None:
            self.target_layer_ids = config.aux_hidden_state_layer_ids
        else:
            self.target_layer_ids = [
                2,
                num_verifier_layers // 2,
                num_verifier_layers - 3,
            ]

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
            aux_hidden_state_layer_ids=kwargs.get("target_layer_ids"),
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

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

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

    def load_verifier_weights(self):
        """Load verifier weights and mask_token_id from tokenizer."""
        super().load_verifier_weights()

        # Load tokenizer to get mask_token_id with fallbacks
        verifier_config = self.config.speculators_config.verifier
        tokenizer = AutoTokenizer.from_pretrained(verifier_config.name_or_path)
        if tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            token_options = [
                ("pad_token_id", tokenizer.pad_token_id),
                ("eos_token_id", tokenizer.eos_token_id),
                ("unk_token_id", tokenizer.unk_token_id),
            ]
            for token_name, token_id in token_options:
                if token_id is not None:
                    self.mask_token_id = token_id
                    warnings.warn(
                        f"Tokenizer does not have mask_token. "
                        f"Using {token_name}={token_id} as fallback.",
                        stacklevel=2,
                    )
                    break
            else:
                raise ValueError("No suitable special token found in tokenizer")
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
        draft_indices = torch.cumsum(self.t2d.long(), dim=0) - 1  # type: ignore[union-attr,operator]
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
