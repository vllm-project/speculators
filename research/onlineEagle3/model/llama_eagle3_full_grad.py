# This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
# Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)
# It has been modified to speed up the training function by
# using dot products instead of attention masks when running forward passes.
# And to use Llama 3 instead of Llama 2, along with a few other experiments.


# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""updated eagle model."""

import math
import os
from typing import Optional

import numpy as np
import torch.utils.checkpoint
from model.modeling_llama_TTT_eagle3 import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from torch import nn
from transformers.activations import ACT2FN


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config,
        layer_idx,
        disable_input_layernorm=True,
    ) -> None:
        super().__init__(config, layer_idx)
        #print("input layernorm")
        #print(disable_input_layernorm)
        # Skip the input_layernorm
        if disable_input_layernorm:
            del self.input_layernorm
            self.input_layernorm = nn.Identity()


class Model(nn.Module):
    def __init__(
        self,
        config,
        load_emb=False,
        path=None,
        bias=False,
        total_tokens=63,
        depth=5,
        top_k=8,
        threshold=1.0,
        hf_target="meta-llama/Llama-3.1-8B-Instruct",

    ):
        super().__init__()
        # self.verifier=AutoModelForCausalLM.from_pretrained(hf_target, device_map="auto")
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.draft_vocab_size
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )  # noqa: E501
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if load_emb:
            import json

            from safetensors import safe_open

            try:
                with open(os.path.join(path, "model.safetensors.index.json")) as f:
                    index_json = json.loads(f.read())
                    #print(index_json["weight_map"].keys())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(
                    os.path.join(path, emb_path), framework="pt", device="cpu"
                ) as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].to(torch.bfloat16)
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json")) as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].to(torch.bfloat16)
            self.embed_tokens.weight.data = tensor

        import json

        from safetensors import safe_open

        try:
            with open(os.path.join(path, "model.safetensors.index.json")) as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["lm_head.weight"]
            with safe_open(
                os.path.join(path, emb_path), framework="pt", device="cpu"
            ) as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].to(torch.bfloat16)
        except:
            with open(os.path.join(path, "pytorch_model.bin.index.json")) as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(path, emb_path))
            tensor = weights["lm_head.weight"].to(torch.bfloat16)
        mapping = np.load("d2t.npy")
        self.d2t = torch.tensor(mapping)
        mapping = np.load("t2d.npy")
        self.t2d = torch.tensor(mapping)

        # FIX: Use indices of True values, not the boolean mask itself
        t2d_indices = torch.nonzero(self.t2d).squeeze(-1)
        self.lm_head.weight.data = tensor[t2d_indices, :]
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.threshold = math.log(threshold)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx=index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.fc = nn.Linear(
            3 * config.hidden_size, config.hidden_size, bias=bias, dtype=torch.bfloat16
        )
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states_history=None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):

        batch_size, seq_length, hidden_shape = hidden_states.shape
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        if hidden_shape != inputs_embeds.shape[2]:
            hidden_states = self.fc(hidden_states)

        past_key_values_length = 0
        if position_ids is None:
            device = (
                hidden_states.device
                if hidden_states is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        inputs_embeds = self.input_layernorm(inputs_embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat((inputs_embeds, hidden_states), dim=-1)

        prior_hidden_states = []

        for i in range(len(hidden_states_history)):
            prior_hidden_states.append(self.hidden_norm(hidden_states_history[i]))
            prior_hidden_states[i] = torch.cat(
                (inputs_embeds, prior_hidden_states[i]), dim=-1
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for _, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    prior_hidden_states,
                    None,
                    False,
                    False,
                    None,
                    position_embeddings,
                    use_reentrant=False,
                )
            hidden_states = layer_outputs[0]
        return hidden_states

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to
    `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )
