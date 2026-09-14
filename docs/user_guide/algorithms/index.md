# Algorithms

Speculators supports six speculative decoding algorithms. All are lossless -- they produce output from the same distribution as the target model.

## [Eagle-3](eagle3.md)

Predicts draft tokens autoregressively using Llama-style draft layers. The more established algorithm with mature support in both Speculators and vLLM.

## [P-EAGLE](peagle.md)

Extends Eagle-3 with parallel multi-token prediction across multiple depths, using COD sampling for memory-efficient training.

## [DFlash](dflash.md)

Predicts all draft tokens in a single forward pass using block-based prediction with Qwen3-style draft layers. Newer, with support improving rapidly.

## [DFlash2](dflash2.md)

Adds local dynamic convolutions and a predecessor-conditioned candidate selector to DFlash while retaining one parallel draft-model forward pass. Experimental training support.

## [DSpark](dspark.md)

Extends DFlash with a Markov head for intra-block token dependencies and a confidence head predicting per-position acceptance. Newest, with support improving rapidly.

## [MTP](mtp.md)

Finetunes the model's native multi-token prediction head on domain-specific data. Available for models with built-in MTP support (e.g. Qwen3-Next, Qwen3.5).

## Choosing an Algorithm

Most algorithms can be paired with any supported verifier model. DFlash2 currently requires the verifier's full vocabulary and an unquantized language-model head; MTP requires a model with native MTP layers. For help choosing between them, see the [Decision Guide](decision_guide.md).

## Adding New Algorithms

See the [Developer Guide](../../developer/add_algorithm.md) for instructions on adding custom algorithms to Speculators.
