# DSpark

DSpark is a speculative decoding method that extends [DFlash](dflash.md)'s block-parallel drafting with two additions: a lightweight sequential head that gives draft positions *inside* a block a dependency on the tokens before them, and a confidence head that predicts how likely each position is to be accepted. Purely parallel drafters propose a whole block in one forward pass but have no inter-token dependencies, so acceptance decays quickly toward the end of the block. DSpark's semi-autoregressive design targets that decay directly, while the confidence signal makes it possible to verify only as far as is worth verifying.

The draft model is a DFlash backbone -- `DSparkDraftModel` subclasses `DFlashDraftModel` -- so the architecture, training pipeline, and vLLM deployment path are the same. It can be paired with any supported verifier.

## Research & Citation

DSpark is based on research from DeepSeek: [arXiv Paper](https://arxiv.org/abs/2607.05147)

```bibtex
@article{cheng2026dspark,
  title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
  author={Cheng, Xin and Yu, Xingkai and Shao, Chenze and Li, Jiashi and Xiong, Yunfan and Qian, Yi and Zhu, Jiaqi and Ma, Shirong and Zhang, Xiaokang and Ye, Jiasheng and others},
  journal={arXiv preprint arXiv:2607.05147},
  year={2026}
}
```
