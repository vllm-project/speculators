# Algorithms

Speculators supports multiple speculative decoding algorithms. Each algorithm provides a different approach to generating draft tokens for verification by the base model.

## Supported Algorithms

### [EAGLE-3](eagle3.md)

The primary speculative decoding algorithm in Speculators. EAGLE-3 uses a lightweight transformer decoder with test-time training (TTT) steps to autoregressively generate draft tokens. It achieves high acceptance rates with minimal parameter overhead.

**Status**: Fully supported — training, inference (vLLM), and pre-trained models available.

### [FastMTP](fast_mtp.md)

A multi-token prediction approach based on DeepSeek-V3's MTP architecture, adapted for speculative decoding. FastMTP uses independent prediction layers for each speculative step, offering a simpler alternative to EAGLE-3's sequential TTT approach.

**Status**: Under development — model definition and training support in progress.

## Adding New Algorithms

See the [Adding New Algorithms](add_new_algorithms.md) guide for instructions on implementing and registering new speculative decoding algorithms in Speculators.
