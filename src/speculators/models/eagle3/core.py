import copy
from typing import ClassVar

import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import AutoConfig, DynamicCache, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
    extend_dense_mask_for_draft_tokens,
    extend_mask_for_draft_tokens,
)
from speculators.models.eagle3.metrics import compute_metrics
from speculators.models.eagle3.model_definitions import model_classes
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import (
    conditional_torch_compile,
    resolve_target_layer_ids,
    strip_verifier_final_layer_id,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig


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

        tl_config = self.config.transformer_layer_config
        self._model_definitions = model_classes[tl_config.model_type]

        # Eagle3-specific: embed_tokens grad depends on config
        self.embed_tokens.weight.requires_grad = self.config.embed_requires_grad

        # FC LAYER
        num_aux = (
            len(config.eagle_aux_hidden_state_layer_ids)
            if config.eagle_aux_hidden_state_layer_ids
            else 3
        )
        self.fc = torch.nn.Linear(
            num_aux * self.hidden_size, self.hidden_size, bias=False
        )

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

        # Sliding window attention support
        self.sliding_window = getattr(tl_config, "sliding_window", None)
        layer_types = getattr(tl_config, "layer_types", None) or []
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(layer_types)
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_layers - len(self.sliding_window_indices))

        # ROTARY EMBEDDINGS
        # Create a modified config for the rotary embedding to use 2x the hidden size
        modified_tl_config = copy.copy(config.transformer_layer_config)
        modified_tl_config.hidden_size *= 2
        self.rotary_emb = self._model_definitions.rotary_emb_class(modified_tl_config)

        # LAYER NORMS
        norm_class = self._model_definitions.norm_class
        self.norm = norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self.verifier_norm = norm_class(self.hidden_size, eps=tl_config.rms_norm_eps)
        self.verifier_norm.weight.requires_grad = False

        if config.norm_before_fc:
            self.input_norm = self._model_definitions.norm_class(
                num_aux * self.hidden_size,
                eps=config.transformer_layer_config.rms_norm_eps,
            )
        else:
            self.input_norm = None

        self.fc_norm: torch.nn.ModuleList | None = None
        if config.fc_norm:
            self.fc_norm = torch.nn.ModuleList(
                [
                    self._model_definitions.norm_class(
                        self.hidden_size,
                        eps=config.transformer_layer_config.rms_norm_eps,
                    )
                    for _ in range(num_aux)
                ]
            )

        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.eagle_aux_hidden_state_layer_ids

    def _build_attn_mask(self, doc_ids_1d, seq_len, device, sliding_window=None):
        mask_mod = create_combined_mask_mod(
            doc_ids_1d,
            seq_len,
            sliding_window=sliding_window,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        verifier_path = model.config.speculators_config.verifier.name_or_path
        if verifier_path is not None:
            model.config.eagle_aux_hidden_state_layer_ids = resolve_target_layer_ids(
                model.config.eagle_aux_hidden_state_layer_ids, verifier_path
            )
        return model

    def load_verifier_weights(self):
        super().load_verifier_weights()

        self.embed_tokens.weight.requires_grad_(self.config.embed_requires_grad)

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

    @conditional_torch_compile
    def forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, num_aux * hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, hidden_size]
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        loss_config: LossConfig | None = None,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)
            # shape: [1, total_seq_len]

        past_key_values = DynamicCache()

        doc_ids_1d = document_ids.squeeze(0).to(device)

        full_attn_mask = (
            self._build_attn_mask(doc_ids_1d, total_seq_len, device)
            if self.uses_full_attn
            else None
        )
        sliding_window_attn_mask = (
            self._build_attn_mask(
                doc_ids_1d,
                total_seq_len,
                device,
                self.sliding_window,
            )
            if self.uses_sliding_window_attn
            else None
        )

        if self.input_norm is not None:
            hidden_states = self.input_norm(hidden_states)
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(len(self.fc_norm), dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)],
                dim=-1,
            )
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

            for layer_idx, decoder_layer in enumerate(self.layers):
                layer_mask = (
                    sliding_window_attn_mask
                    if layer_idx in self.sliding_window_indices
                    else full_attn_mask
                )
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            if self.config.norm_output:
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
            else:
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
                    loss_config=loss_config,
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

            if self._attn_impl == "simple_flex_attention":
                if full_attn_mask is not None:
                    full_attn_mask = extend_mask_for_draft_tokens(full_attn_mask)
                if sliding_window_attn_mask is not None:
                    sliding_window_attn_mask = extend_mask_for_draft_tokens(
                        sliding_window_attn_mask
                    )
            else:
                if full_attn_mask is not None:
                    full_attn_mask = extend_dense_mask_for_draft_tokens(
                        full_attn_mask, total_seq_len
                    )
                if sliding_window_attn_mask is not None:
                    sliding_window_attn_mask = extend_dense_mask_for_draft_tokens(
                        sliding_window_attn_mask, total_seq_len
                    )
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
        if kwargs.get("target_layer_ids") is not None:
            target_layer_ids = strip_verifier_final_layer_id(
                target_layer_ids, kwargs["verifier_name_or_path"]
            )

        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )

        config = Eagle3SpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs["norm_before_residual"],
            norm_before_fc=kwargs.get("norm_before_fc", False),
            fc_norm=kwargs.get("fc_norm", False),
            norm_output=kwargs.get("norm_output", False),
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
        """Get training and validation kwargs for Eagle3.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        train_kwargs = {
            "use_off_policy_tokens": kwargs["use_off_policy_tokens"],
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_config": loss_config,
        }
        val_kwargs = {
            "use_off_policy_tokens": False,
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_config": loss_config,
        }
        return train_kwargs, val_kwargs
