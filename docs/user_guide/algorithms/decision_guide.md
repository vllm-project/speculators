# Algorithm Decision Guide

Speculators currently supports four speculative decoding algorithms: **Eagle-3**, **P-EAGLE**, **DFlash**, and **MTP**. All are lossless -- they produce output from the same distribution as the target model.

## How They Differ

**Eagle-3** predicts draft tokens autoregressively, one at a time.

**P-EAGLE** extends Eagle-3 with parallel multi-token prediction across multiple depths, using COD sampling for memory-efficient training.

**DFlash** predicts all draft tokens in a single forward pass using block-based prediction with anchor points.

**MTP** finetunes the model's native multi-token prediction head on domain-specific data. Unlike the other algorithms, MTP does not train from scratch -- it starts from pre-existing MTP layers and is only available for models with native MTP support.

Eagle-3, P-EAGLE, and DFlash can be paired with any supported verifier model (including quantized variants) -- the draft architecture is independent of the verifier architecture. MTP requires a model with native MTP layers (e.g. Qwen3-Next, Qwen3.5).

## Current Support

|                     | Eagle-3       | P-EAGLE             | DFlash              | MTP                         |
| ------------------- | ------------- | ------------------- | ------------------- | --------------------------- |
| **Draft layers**    | Llama-style   | Llama-style         | Qwen3-style         | Native MTP layers           |
| **Verifier models** | Any supported | Any supported       | Any supported       | Models with native MTP only |
| **Training mode**   | From scratch  | From scratch        | From scratch        | Finetune existing MTP head  |
| **Speculators**     | Mature        | Newer, growing fast | Newer, growing fast | Newer, growing fast         |
| **vLLM**            | Mature        | Newer, growing fast | Newer, growing fast | Newer, growing fast         |

Eagle-3 has been available longer and has broader support in both Speculators and vLLM. P-EAGLE, DFlash, and MTP were added more recently and support is improving rapidly.

## Which Should I Use?

If you're unsure, start with Eagle-3 -- it has the most mature tooling and documentation. If you want parallel multi-token prediction with an Eagle-3-based architecture, try P-EAGLE. If you want to experiment with DFlash's single-forward-pass block prediction approach, the training workflow is the same. If your model already has native MTP layers (e.g. Qwen3-Next, Qwen3.5), MTP finetuning lets you improve the existing MTP head on domain-specific data without training a separate draft model.

For more details on each algorithm, see:

- [Eagle-3](eagle3.md)
- [P-EAGLE](peagle.md)
- [DFlash](dflash.md)
- [MTP](mtp.md)
