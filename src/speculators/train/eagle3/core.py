# ruff: noqa: ERA001
from typing import ClassVar

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoConfig, DynamicCache

from speculators.config import VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.train.eagle3.attention import (
    create_combined_mask_mod,
    extend_mask_for_draft_tokens,
)
from speculators.train.eagle3.model_definitions import model_classes
from speculators.utils.loading import load_model_layers


def align_for_step(
    logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, total_seq_len]
    ttt_step: int,
):
    # There are no target values for the last ttt_step tokens, so we mask them out
    # before computing the loss/accuracy. Likewise, there are no logits for the first
    # ttt_step tokens, so we mask them out.
    # This is equivalent to shifting the target values by ttt_step + 1 to the left
    # which puts them in the correct position for the generated tokens.
    # e.g.
    # targets_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # logits_indices_ttt_step_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # logits_indices_ttt_step_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # logits_indices_ttt_step_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    # The indices for the loss_mask need to be kept in line with the targets indices
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


@SpeculatorModel.register("eagle3_draft")
class Eagle3DraftModel(SpeculatorModel):
    config_class: ClassVar[type[Eagle3SpeculatorConfig]] = Eagle3SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_lm_head.weight",
        "d2t",
        "t2d",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = ["verifier_lm_head.weight"]  # type: ignore[misc,assignment]

    def __init__(
        self,
        config: Eagle3SpeculatorConfig,
        t2d: torch.Tensor,
        d2t: torch.Tensor,
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
    ):
        super().__init__(
            config=config,
            verifier=None,
            verifier_attachment_mode="train_only",
        )
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.ttt_steps = ttt_steps
        self.ttt_step_loss_decay = ttt_step_loss_decay
        self.register_buffer("t2d", t2d)  # shape: [verifier_vocab_size], bool
        self.register_buffer("d2t", d2t)  # shape: [draft_vocab_size], int offsets
        self.draft_vocab_size = int(t2d.sum(dtype=torch.long).item())
        model_definitions = model_classes[config.transformer_layer_config.model_type]

        self.fc = torch.nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        num_hidden_layers = config.transformer_layer_config.num_hidden_layers
        self.layers = torch.nn.ModuleList(
            [
                model_definitions.decoder_layer_class(
                    config.transformer_layer_config,
                    layer_idx,
                    norm_before_residual=config.norm_before_residual,
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )
        self.norm = model_definitions.norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self.rotary_emb = model_definitions.rotary_emb_class(
            config.transformer_layer_config
        )
        self._setup_embeddings_and_lm_heads(config.speculators_config.verifier, t2d)

    def _setup_embeddings_and_lm_heads(self, config: VerifierConfig, t2d: torch.Tensor):
        if config.name_or_path is None:
            raise ValueError("VerifierConfig `name_or_path` value is required.")
        verifier_model_config = AutoConfig.from_pretrained(config.name_or_path)
        if verifier_model_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Verifier hidden size {verifier_model_config.hidden_size} does not"
                f" match draft hidden size {self.hidden_size}."
            )

        verifier_weights = load_model_layers(
            ["model.embed_tokens.weight", "lm_head.weight"], config.name_or_path
        )

        # EMBEDDINGS
        self.embed_tokens = torch.nn.Embedding(
            verifier_model_config.vocab_size,
            self.hidden_size,
            padding_idx=verifier_model_config.pad_token_id,
        )
        # shape: [verifier_vocab_size, hidden_size]
        default_dtype = self.embed_tokens.weight.dtype

        embed_tokens_sd = {
            "weight": verifier_weights["model.embed_tokens.weight"].to(default_dtype)
        }
        self.embed_tokens.load_state_dict(embed_tokens_sd)
        self.embed_tokens.weight.requires_grad = False

        # LM HEADS
        self.lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        # shape: [hidden_size, draft_vocab_size]
        self.verifier_lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )

        masked_lm_head_weight = verifier_weights["lm_head.weight"].to(
            device=t2d.device, dtype=default_dtype
        )[t2d.to(torch.bool), :]
        if masked_lm_head_weight.shape != self.lm_head.weight.shape:
            raise ValueError(
                f"Masked verifier lm head data shape "
                f"{masked_lm_head_weight.shape} does not match draft "
                f"lm head shape {self.lm_head.weight.shape}"
            )
        self.lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.requires_grad = False

    @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, hidden_size]
        ttt_steps: int | None = None,
        use_off_policy_tokens: bool = False,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if ttt_steps is None:
            ttt_steps = self.ttt_steps
        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)
            # shape: [1, total_seq_len]

        past_key_values = DynamicCache(config=self.config.transformer_layer_config)

        combined_mask_mod = create_combined_mask_mod(lengths.to(device), total_seq_len)
        block_mask = create_block_mask(
            combined_mask_mod,
            B=None,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
        )

        hidden_states = self.fc(hidden_states)
        # shape: [1, total_seq_len, hidden_size]

        original_input_ids = input_ids.detach().clone()
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                target_logits = self.verifier_lm_head(verifier_last_hidden_states)
                # shape: [1, total_seq_len, draft_vocab_size]

        loss = torch.tensor(0.0, device=device)
        prev_correct = (
            loss_mask.clone()
            if loss_mask is not None
            else torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
        )
        draft_tokens = []
        metrics = {}
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
                    attention_mask=block_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            logits = self.lm_head(self.norm(hidden_states))
            # shape: [1, total_seq_len, draft_vocab_size]

            if return_loss:
                s_logits, s_targets, s_loss_mask, s_prev_correct = align_for_step(
                    logits, target_logits, loss_mask, prev_correct, ttt_step
                )
                loss_weight = self.ttt_step_loss_decay**ttt_step
                s_loss = loss_weight * loss_function(s_logits, s_targets, s_loss_mask)
                loss += s_loss

                s_full_acc, s_cond_acc = compute_accuracy(
                    s_logits, s_targets, s_loss_mask, s_prev_correct
                )
                metrics[f"loss_{ttt_step}"] = s_loss.detach().clone()
                metrics[f"full_acc_{ttt_step}"] = s_full_acc
                metrics[f"cond_acc_{ttt_step}"] = s_cond_acc

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())
            # shape: [1, total_seq_len]
            # Use d2t to map draft tokens to verifier tokens.
            # Must be in verifier vocabulary space because we use the full verifier
            # vocabulary in the embedding.
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

            block_mask = extend_mask_for_draft_tokens(block_mask)
            position_ids = position_ids + 1
            # shape: [1, total_seq_len]

        metrics["loss"] = loss.detach().clone()

        if return_loss:
            return draft_tokens, loss, metrics
        else:
            return draft_tokens
