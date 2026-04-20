# Algorithms

Speculators supports two speculative decoding algorithms. Both are lossless -- they produce output from the same distribution as the target model.

## [EAGLE-3](eagle3.md)

Predicts draft tokens autoregressively using Llama-style draft layers. The more established algorithm with mature support in both Speculators and vLLM.

## [DFlash](dflash.md)

Predicts all draft tokens in a single forward pass using block-based prediction with Qwen3-style draft layers. Newer, with support improving rapidly.

## Choosing an Algorithm

Both algorithms can be paired with any supported verifier model. For help choosing between them, see the [Decision Guide](decision_guide.md).

## Adding New Algorithms

See the [Developer Guide](../../developer/add_algorithm.md) for instructions on adding custom algorithms to Speculators.
