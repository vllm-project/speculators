import torch
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM, DynamicCache

from typing import ClassVar
from torch.nn.attention.flex_attention import create_block_mask

from speculators.model import SpeculatorModel
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.train.eagle3.attention import (
    create_combined_mask_mod,
    extend_mask_for_draft_tokens,
)
from speculators.train.eagle3.model_definitions import model_classes


def load_verifier_embeddings(verifier_model_name_or_path: str):
    verifier_model = AutoModelForCausalLM.from_pretrained(verifier_model_name_or_path)
    return verifier_model.model.embed_tokens.state_dict()


class Eagle3VerifierLMHead(torch.nn.Module):
    def __init__(self, hidden_size: int, draft_vocab_size: int):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden_size, draft_vocab_size, bias=False)
        self.lm_head.weight.requires_grad = False

    def load_verifier_lm_head(
        self, verifier_model_name_or_path: str, t2d: torch.Tensor
    ):
        verifier_model = AutoModelForCausalLM.from_pretrained(
            verifier_model_name_or_path
        )
        verifier_lm_head_data = verifier_model.lm_head.weight.data.to(t2d.device)
        trucated_data = verifier_lm_head_data[t2d, :]
        if trucated_data.shape[0] != self.lm_head.weight.shape[0]:
            raise ValueError(
                f"Truncated verifier lm head data shape {trucated_data.shape} does not match draft lm head shape {self.lm_head.weight.shape}"
            )
        self.lm_head.weight.data = trucated_data

    @torch.no_grad()
    def forward(self, verifier_last_hidden_states: torch.Tensor):
        return self.lm_head(verifier_last_hidden_states)


def align_for_step(
    logits: torch.Tensor,  # shape: [batch_size, total_seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [batch_size, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [batch_size, total_seq_len]
    ttt_step: int,
):
    # We don't have target values for the last ttt_step tokens, so we mask them out on the logit side
    # We shift the target values by ttt_step + 1 to the left because that's the position the generated tokens correspond to
    # e.g.
    # targets_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # logits_indices_ttt_step_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # logits_indices_ttt_step_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # logits_indices_ttt_step_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    # The indices for the loss_mask need to be kept in line with the targets indices
    logits = logits[:, :-ttt_step] if ttt_step > 0 else logits
    # shape: [batch_size, total_seq_len - ttt_step, draft_vocab_size]
    targets = targets[:, ttt_step:]
    # shape: [batch_size, total_seq_len - ttt_step, draft_vocab_size]
    if loss_mask is not None:
        loss_mask = loss_mask[:, ttt_step:]
        # shape: [batch_size, total_seq_len - ttt_step]
    return logits, targets, loss_mask


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,  # shape: [batch_size, total_seq_len - ttt_step, draft_vocab_size]
    targets: torch.Tensor,  # shape: [batch_size, total_seq_len - ttt_step, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [batch_size, total_seq_len - ttt_step]
):
    # Note: logits, targets, and loss_mask are already aligned for the current ttt_step
    target_tokens = torch.argmax(targets, dim=-1)
    predicted_tokens = torch.argmax(logits, dim=-1)
    # shape: [batch_size, total_seq_len - ttt_step]

    correct = predicted_tokens == target_tokens
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    acc = correct.float().sum() / (
        correct.numel() + 1e-5
    )  # avoid NaNs when loss_mask is all False
    return acc


def loss_function(
    logits: torch.Tensor,  # shape: [batch_size, total_seq_len - ttt_step, draft_vocab_size]
    targets: torch.Tensor,  # shape: [batch_size, total_seq_len - ttt_step, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [batch_size, total_seq_len - ttt_step]
):
    # Note: logits, targets, and loss_mask are already aligned for the current ttt_step
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    elementwise_loss = torch.nn.functional.kl_div(
        logits, target_p, reduction="none", log_target=False
    )

    if loss_mask is not None:
        elementwise_loss = elementwise_loss * loss_mask.unsqueeze(-1)
        denominator = loss_mask.sum(dim=1) + 1e-5
    else:
        denominator = logits.shape[1]  # total_seq_len - ttt_step
    batch_loss = torch.sum(elementwise_loss, dim=(1, 2)) / denominator
    # shape: [batch_size]
    return batch_loss.mean()

@SpeculatorModel.register("eagle3_draft")
class Eagle3DraftModel(SpeculatorModel):
    config_class: ClassVar[type[Eagle3SpeculatorConfig]] = Eagle3SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "lm_head.weight",
        "d2t",
        "t2d",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = []  # type: ignore[misc,assignment]
    def __init__(
        self,
        verifier_model_name_or_path: str,
        hidden_size: int,  # Must be same for verifier and draft
        # Vocab mappings
        t2d: torch.Tensor,
        d2t: torch.Tensor,
        decoder_layer_config: PretrainedConfig,
        # Verifier
        verifier_vocab_size: int,
        verifier_pad_token_id: int | None,
        # Draft config
        num_layers: int = 1,
        ttt_steps: int = 3,
    ):
        
        norm_before_residual = True
        from speculators.config import SpeculatorsConfig, VerifierConfig
        from speculators.proposals.greedy import GreedyTokenProposalConfig
        speculator_config = Eagle3SpeculatorConfig(
            transformer_layer_config=decoder_layer_config,
            draft_vocab_size=t2d.sum(dtype=torch.long).item(),
            norm_before_residual=norm_before_residual,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[GreedyTokenProposalConfig(
                    proposal_type="greedy",
                    speculative_tokens=ttt_steps,
                )],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=verifier_model_name_or_path,
                    architectures=["LlamaForCausalLM"], # todo: fix
                ),
            ),
        )
        super().__init__(config=speculator_config, verifier=None, verifier_attachment_mode="train_only")
        self.verifier_model_name_or_path = verifier_model_name_or_path
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder_layer_config = decoder_layer_config
        self.ttt_steps = ttt_steps
        self.register_buffer(
            "t2d", t2d
        )  # shape: [verifier_vocab_size], bool
        self.register_buffer(
            "d2t", d2t
        )  # shape: [draft_vocab_size], int offsets
        self.draft_vocab_size = t2d.sum(dtype=torch.long).item()
        model_definitions = model_classes[decoder_layer_config.model_type]

        self.fc = torch.nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.layers = torch.nn.ModuleList(
            [
                model_definitions.decoder_layer_class(decoder_layer_config, layer_idx, norm_before_residual=norm_before_residual)
                for layer_idx in range(num_layers)
            ]
        )
        self.norm = model_definitions.norm_class(
            hidden_size, eps=decoder_layer_config.rms_norm_eps
        )
        self.rotary_emb = model_definitions.rotary_emb_class(decoder_layer_config)
        self.embed_tokens = torch.nn.Embedding(
            verifier_vocab_size, hidden_size, padding_idx=verifier_pad_token_id
        )
        # shape: [verifier_vocab_size, hidden_size]
        self.embed_tokens.load_state_dict(
            load_verifier_embeddings(verifier_model_name_or_path)
        )
        self.embed_tokens.weight.requires_grad = False

        self.lm_head = torch.nn.Linear(hidden_size, self.draft_vocab_size, bias=False)
        # shape: [hidden_size, draft_vocab_size]

    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        target_logits: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, draft_vocab_size]
        ttt_steps: int | None = None,
        use_off_policy_tokens: bool = False,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        return_loss = target_logits is not None

        if ttt_steps is None:
            ttt_steps = self.ttt_steps
        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)

        past_key_values = DynamicCache(config=self.decoder_layer_config)

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

        loss = torch.tensor(0.0, device=device)
        draft_tokens = []
        accuracy_list = []
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
                    attention_mask=block_mask,  # block_mask_to_dense_attention_mask(block_mask, device, torch.bool),
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            logits = self.lm_head(self.norm(hidden_states))
            # shape: [1, total_seq_len, draft_vocab_size]

            if return_loss:
                s_logits, s_targets, s_loss_mask = align_for_step(
                    logits, target_logits, loss_mask, ttt_step
                )
                loss += loss_function(s_logits, s_targets, s_loss_mask)
                accuracy_list.append(compute_accuracy(s_logits, s_targets, s_loss_mask))

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())
            # shape: [1, total_seq_len]
            # Use d2t to map draft tokens to verifier tokens.
            # Must be in verifier vocabulary space because we use full verifier vocabulary in embedding
            input_ids = input_ids + self.d2t[input_ids]

            if use_off_policy_tokens:
                # Overwrite input_ids with ground truth tokens
                # shift input_ids by 1 to the left and pad with 0
                # note: inputs_ids will no longer line up with verifier_last_hidden_state
                # the draft logits generated from the padded tokens are ignored sliced out for loss calculation
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

        if return_loss:
            return (
                draft_tokens,
                loss,
                torch.tensor(accuracy_list, device=device, dtype=torch.float),
            )
        else:
            return draft_tokens
