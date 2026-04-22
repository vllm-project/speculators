# Algorithm Decision Guide

Speculators currently supports two speculative decoding algorithms: **Eagle-3** and **DFlash**. Both are lossless -- they produce output from the same distribution as the target model.

## How They Differ

**Eagle-3** predicts draft tokens autoregressively, one at a time.

**DFlash** predicts all draft tokens in a single forward pass using block-based prediction with anchor points.

Both algorithms can be paired with any supported verifier model (including quantized variants) -- the draft architecture is independent of the verifier architecture. The draft layers are always trained from scratch, so the choice of draft architecture doesn't constrain which target models you can accelerate.

## Current Support

|                     | Eagle-3       | DFlash              |
| ------------------- | ------------- | ------------------- |
| **Draft layers**    | Llama-style   | Qwen3-style         |
| **Verifier models** | Any supported | Any supported       |
| **Speculators**     | Mature        | Newer, growing fast |
| **vLLM**            | Mature        | Newer, growing fast |

Eagle-3 has been available longer and has broader support in both Speculators and vLLM. DFlash was added more recently and support is improving rapidly.

## Which Should I Use?

If you're unsure, start with Eagle-3 -- it has the most mature tooling and documentation. If you want to experiment with DFlash's single-forward-pass approach, the training workflow is the same.

For more details on each algorithm, see:

- [Eagle-3](eagle3.md)
- [DFlash](dflash.md)
