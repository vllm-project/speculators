import torch
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM, DynamicCache

from torch.nn.attention.flex_attention import create_block_mask

from speculators.train.eagle3.attention import (
    create_combined_mask_mod,
    extend_mask_for_draft_tokens,
)
from speculators.train.eagle3.model_definitions import model_classes


class Eagle3DraftModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,  # Must be same for verifier and draft
        # Vocab mappings
        t2d_vocab: torch.Tensor,
        d2t_vocab: torch.Tensor,
        decoder_layer_config: PretrainedConfig,
        # Verifier
        verifier_vocab_size: int,
        verifier_pad_token_id: int,
        # Draft config
        num_layers: int = 1,
        ttt_steps: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder_layer_config = decoder_layer_config
        self.ttt_steps = ttt_steps
        self.t2d_vocab = t2d_vocab  # shape: [verifier_vocab_size], bool
        self.d2t_vocab = d2t_vocab  # shape: [draft_vocab_size], int offsets
        self.draft_vocab_size = t2d_vocab.sum(dtype=torch.long).item()
        model_definitions = model_classes[decoder_layer_config.model_type]

        self.fc_layer = torch.nn.Linear(3 * hidden_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [
                model_definitions.decoder_layer_class(decoder_layer_config, layer_idx)
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

        self.verifier_lm_head = torch.nn.Linear(
            hidden_size, self.draft_vocab_size, bias=False
        )
        self.lm_head = torch.nn.Linear(hidden_size, self.draft_vocab_size, bias=False)
        # shape: [hidden_size, draft_vocab_size]

    def load_verifier_lm_head(self, verifier_model_name_or_path: str):
        verifier_model = AutoModelForCausalLM.from_pretrained(
            verifier_model_name_or_path
        )
        verifier_lm_head_data = verifier_model.lm_head.weight.data.to(
            self.t2d_vocab.device
        )
        self.verifier_lm_head.weight.data = verifier_lm_head_data[self.t2d_vocab, :]

    def loss_function(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        targets = torch.nn.functional.log_softmax(targets, dim=-1)
        return torch.nn.functional.kl_div(
            logits, targets, reduction="sum", log_target=True
        ) / (logits.shape[0] * logits.shape[1])

    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, hidden_size]
        ttt_steps: int | None = None,
        use_off_policy_tokens: bool = False,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        return_loss = verifier_last_hidden_states is not None

        if ttt_steps is None:
            ttt_steps = self.ttt_steps
        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)

        past_key_values = DynamicCache(config=self.decoder_layer_config)
        position_ids = (
            torch.cat(
                [
                    torch.arange(length, dtype=torch.long, device=device)
                    for length in lengths
                ],
                dim=0,
            )
            .unsqueeze(0)
            .contiguous()
        )
        # shape: [1, total_seq_len]

        combined_mask_mod = create_combined_mask_mod(lengths.to(device))
        block_mask = create_block_mask(
            combined_mask_mod,
            B=None,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
        )

        hidden_states = self.fc_layer(hidden_states)
        # shape: [1, total_seq_len, hidden_size]

        if return_loss:
            verifier_logits = self.verifier_lm_head(verifier_last_hidden_states)
            loss = torch.tensor(0.0, device=device)

        draft_tokens = []
        for ttt_step in range(ttt_steps):
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

            hidden_states = self.norm(hidden_states)

            logits = self.lm_head(hidden_states)
            # shape: [1, total_seq_len, draft_vocab_size]

            if return_loss:
                loss += self.loss_function(
                    logits[:, : -(ttt_step + 1)], verifier_logits[:, (ttt_step + 1) :]
                )

            input_ids = torch.argmax(logits, dim=-1)
            # shape: [1, total_seq_len]
            # Use d2t to map draft tokens to verifier tokens.
            # Must be in verifier vocabulary space because we use full verifier vocabulary in embedding
            input_ids = input_ids + self.d2t_vocab[input_ids]
            draft_tokens.append(input_ids)

            if use_off_policy_tokens:
                # Overwrite input_ids with ground truth tokens
                # shift input_ids by 1 to the left and pad with 0
                # note: inputs_ids will no longer line up with verifier_last_hidden_state
                # the draft logits generated from the padded tokens are ignored sliced out for loss calculation
                input_ids = torch.cat(
                    [input_ids[:, 1:], input_ids.new_zeros(1, 1)], dim=-1
                )
                # shape: [1, total_seq_len]

            block_mask = extend_mask_for_draft_tokens(block_mask)
            position_ids = position_ids + 1
            # shape: [1, total_seq_len]

        return draft_tokens, loss if return_loss else None
