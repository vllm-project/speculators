# Algorithm Decision Guide

Speculators currently supports three speculative decoding algorithms: **Eagle-3**, **P-EAGLE**, and **DFlash**. All are lossless -- they produce output from the same distribution as the target model.

## How They Differ

**Eagle-3** predicts draft tokens autoregressively, one at a time.

**P-EAGLE** extends Eagle-3 with parallel multi-token prediction across multiple depths, using COD sampling for memory-efficient training.

**DFlash** predicts all draft tokens in a single forward pass using block-based prediction with anchor points.

All algorithms can be paired with any supported verifier model (including quantized variants) -- the draft architecture is independent of the verifier architecture. The draft layers are always trained from scratch, so the choice of draft architecture doesn't constrain which target models you can accelerate.

## Current Support

|                     | Eagle-3       | P-EAGLE             | DFlash              |
| ------------------- | ------------- | ------------------- | ------------------- |
| **Draft layers**    | Llama-style   | Llama-style         | Qwen3-style         |
| **Drafting**        | Autoregressive| Parallel (multi-depth) | Single forward pass |
| **Verifier models** | Any supported | Any supported       | Any supported       |
| **Speculators**     | Mature        | Newer, growing fast | Newer, growing fast |
| **vLLM**            | Mature        | Newer, growing fast | Newer, growing fast |

Eagle-3 has been available longer and has broader support in both Speculators and vLLM. P-EAGLE and DFlash were added more recently and support is improving rapidly.

## Which Should I Use?

If you're unsure, start with Eagle-3 -- it has the most mature tooling and documentation. If you want parallel multi-token prediction with an Eagle-3-based architecture, try P-EAGLE. If you want to experiment with DFlash's single-forward-pass block prediction approach, the training workflow is the same.

For more details on each algorithm, see:

- [Eagle-3](eagle3.md)
- [P-EAGLE](peagle.md)
- [DFlash](dflash.md)
