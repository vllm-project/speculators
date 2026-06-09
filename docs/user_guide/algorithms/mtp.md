# MTP (Multi-Token Prediction)

MTP is a speculative decoding method that uses the model's own native multi-token prediction head to draft multiple tokens ahead, which are then verified by the target model in a single forward pass. Unlike Eagle-3 and DFlash which train draft models from scratch, MTP finetuning starts from the model's pre-existing MTP layers -- converting them to speculators format, finetuning on domain-specific data, and stitching the improved weights back. This approach is available for models that ship with native MTP support, such as Qwen3-Next and Qwen3.5.

## How It Works

### Architecture

The MTP head consists of a single prediction layer that takes two inputs at each speculative step: the verifier's hidden states and token embeddings from the ground-truth input. These are normalized separately, concatenated, and projected down to the model's hidden dimension before passing through a standard decoder layer (attention + MLP). The output hidden states feed into the next step recursively, while a shared LM head produces draft logits at each step.

### Training Process

1. **Convert:** Extract the native MTP head from the model checkpoint and convert it to speculators format using `MTPConverter`
2. **Train:** Finetune the MTP layers on domain-specific data. At each step k, the model predicts token t+k+1 given verifier hidden states at position t and ground-truth token embeddings at position t+k. Per-step losses are weighted with exponential decay (default beta=0.6) following FastMTP Equation 2
3. **Stitch:** Merge the finetuned MTP weights back into the original verifier checkpoint for deployment

Only the MTP layers are trainable -- `embed_tokens` and `lm_head` are frozen and shared with the verifier.

### Step Weight Formula

Per-step loss weights follow normalized exponential decay:

```
alpha_k = beta^(k-1) / sum(beta^(j-1) for j=1..K)
```

With the default beta=0.6 and 3 speculative steps, this gives weights `[0.51, 0.31, 0.18]`, emphasizing accuracy on the first speculative token.

## Research & Citation

MTP finetuning is based on the FastMTP method from Tencent: [FastMTP Repository](https://github.com/Tencent-BAC/FastMTP) | [arXiv Paper](https://arxiv.org/abs/2509.18362)

```bibtex
@article{cui2025fastmtp,
  title={FastMTP: Accelerating Multi-Token Prediction via Efficient Speculative Decoding},
  author={Cui, Hao and Diao, Hailin and Wu, Jian and Cheng, Yu},
  journal={arXiv preprint arXiv:2509.18362},
  year={2025}
}
```

## See Also

- [Train MTP Online](../tutorials/train_mtp_online.md) -- Step-by-step finetuning tutorial
- [vLLM Recipes](https://recipes.vllm.ai/) -- Deployment commands for serving MTP models
