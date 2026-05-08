# MTP (Multi-Token Prediction)

MTP is a speculative decoding approach that finetunes the native multi-token prediction head shipped with models like Qwen3-Next and Qwen3.5. Unlike Eagle-3 and DFlash, which train a separate draft model from scratch, MTP finetunes the existing MTP head on domain-specific data to improve its acceptance rate, using the [FastMTP](https://arxiv.org/abs/2509.18362) methodology.

## How It Works

### Architecture

The MTP head is a single transformer layer that shares the verifier's embedding table and LM head. At each speculative step k:

1. Embed `input_ids[t+k]` and concatenate with the hidden state from the previous step (or the verifier's hidden state for step 0)
2. Project the concatenated representation through `input_proj` (2H -> H)
3. Run through the transformer layer
4. Predict logits for position `t+k+1`

The hidden state output of step k feeds recursively into step k+1.

### Training

MTP finetuning uses pre-extracted verifier hidden states so that only the small MTP head (~100M-400M params) loads during training -- the full verifier (80B+) is never loaded. Training uses an exponential-decay loss from FastMTP (Equation 2): weight for step k is `beta^k / sum(beta^j)`, giving earlier speculative steps more weight.

Default step weights with beta=0.6 and 3 steps: `[0.51, 0.31, 0.18]`.

### Key Differences from Eagle-3

| | MTP | Eagle-3 | | --------------------- | ------------------------------------ | -------------------------------- | | **Draft model** | Finetunes native MTP head | Trains draft layers from scratch | | **Prerequisite** | Model must ship with MTP layers | Any supported model | | **Embedding/LM head** | Shared with verifier (frozen) | Separate (with vocab mapping) | | **Deployment** | Stitch back into verifier checkpoint | Separate speculator checkpoint |

## Supported Models

| Model Family | Architecture | MoE | | ------------ | ---------------------------- | --- | | Qwen3-Next | `Qwen3NextForCausalLM` | Yes | | Qwen3.5 | `Qwen3_5_TextForCausalLM` | No | | Qwen3.5-MoE | `Qwen3_5_MoeTextForCausalLM` | Yes |

## End-to-End Pipeline

The MTP finetuning pipeline has 7 stages:

1. **Response regeneration** -- Generate domain-specific responses with the verifier
2. **Data preparation** -- Tokenize and create loss masks
3. **Hidden states extraction** -- Extract verifier hidden states
4. **Model conversion** -- Extract MTP head from native checkpoint
5. **Finetuning** -- Train the MTP head on domain data
6. **Stitching** -- Reintegrate finetuned weights into verifier checkpoint
7. **Deployment** -- Serve with vLLM

## Research & Citation

MTP finetuning in Speculators follows the FastMTP methodology:

```bibtex
@article{cui2025fastmtp,
  title={FastMTP: Advancing Multi-Token Prediction via Multi-Draft Verification for LLM Acceleration},
  author={Cui, Jiaao and Zheng, Ziang and Yue, Shuzhang and Li, Ao and Liu, Qingrui and Wang, Shengyu and Guo, Bin and Yu, Zhiwen},
  journal={arXiv preprint arXiv:2509.18362},
  year={2025}
}
```

## See Also

- [Train MTP Offline](../tutorials/train_mtp_offline.md) -- Complete finetuning tutorial with GSM8K
