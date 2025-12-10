# Speculators 0.3.0 release notes

This Speculators 0.3.0 release provides support for training EAGLE3 speculative decoding draft models, including data generation, model training, and validation workflows for use with vLLM.

## New features and enhancements

### Model definitions and training support

<!-- https://issues.redhat.com/browse/INFERENG-2276 -->
#### Added Eagle3 draft model support
Added support for the Eagle3 draft model including all features required for efficient Eagle3 model training.
This model definition is used to train new Eagle3 draft models.

<!-- https://issues.redhat.com/browse/INFERENG-2282 -->
#### Added training, timing, and testing logic for Eagle3 draft model
Training, timing, and testing logic is now integrated into the `Eagle3DraftModel` [forward](https://docs.vllm.ai/projects/speculators/en/latest/reference/speculators/models/?h=forward#speculators.models.EagleSpeculator.forward) method. 
The `forward` method now supports dynamic step counts and computes per-step loss and accuracy.

<!-- https://issues.redhat.com/browse/INFERENG-2283 -->
#### Added training loop for Eagle3 draft models
Full training support is now available for single and multi-layer Eagle3 draft models for both Mixture of Experts (MoE) and non-MoE target models.

### Data generation and processing

<!-- https://issues.redhat.com/browse/INFERENG-2523 -->
#### Added new vLLM data generation prototype
Created a new prototype hidden states generator using vLLM which provides the following:

- Multiprocess executor for efficient batch inference
- Patched model forward pass to capture intermediate hidden states
- Prefill-only mode (`max_tokens=1`)
- Tensor parallelism support
- Automatic KV-cache and memory management

End-to-end scripts include the following:

- [data_generation_offline.py](https://github.com/vllm-project/speculators/blob/main/scripts/data_generation_offline.py): preprocesses data, saves token frequency distribution, generates hidden states
- [build_vocab_mapping.py](https://github.com/vllm-project/speculators/blob/main/scripts/build_vocab_mapping.py): builds t2d and d2t tensors

Supports MoE and non-MoE models. Vision-language support will be added later.

<!-- https://issues.redhat.com/browse/INFERENG-2526 -->
#### Data loading from vLLM generated outputs
Data is saved as individual `data_{index}.pt` files. Each data point contains `input_ids`, `hidden_states`, and `loss_mask`.

<!-- https://issues.redhat.com/browse/INFERENG-3061 -->
#### Typed data generation configuration
Added a new dataclass-based configuration generator (`config_generator.py`) for data generation with full type safety.
The generator ensures full reproducibility for the data generation with a complete set of metadata for the output including package versions, GPU info, cache keys, and CLI parameters.

<!-- https://issues.redhat.com/browse/INFERENG-3050 -->
#### Data generation benchmarking
New Benchmarking data is available for Llama3, Qwen3, and gpt-oss models.

### Attention and performance

<!-- https://issues.redhat.com/browse/INFERENG-2525 -->
#### New FlexAttention attention mask definition
Added new document-masking support enabling fast, memory-efficient Eagle3 draft model training.
This approach exploits sparsity in train-time-test attention masks, providing faster performance and lower memory usage compared to a naive full attention matrix.

### End-to-end workflows

<!-- https://issues.redhat.com/browse/INFERENG-3393 -->
#### New E2E script for data generation and training speculative draft models
A new end-to-end script has been added which runs the full workflow including data generation, vocabulary mapping, and training, under a single configuration.

The script provides a simplified interface for configuring a full training run that can be launched once. Internally, the script runs each step of the process and ensures data flows correctly from each step to the next. 

The following Llama3, Qwen3, and gpt-oss draft model training scripts have been updated:

1. [llama3_8b_sharegpt_5k.py](https://github.com/vllm-project/speculators/blob/main/examples/data_generation_and_training/llama3_8b_sharegpt_5k.py)
2. [gpt_oss_20b_ultrachat_5k.py](https://github.com/vllm-project/speculators/blob/main/examples/data_generation_and_training/gpt_oss_20b_ultrachat_5k.py)
3. [qwen3_8b_sharegpt_ultrachat.py](https://github.com/vllm-project/speculators/blob/main/examples/data_generation_and_training/qwen3_8b_sharegpt_ultrachat.py)

<!-- https://issues.redhat.com/browse/INFERENG-3389 -->
#### New gpt-oss training and validation example script
A new end-to-end training example for gpt-oss has been added:
[gpt_oss_20b_ultrachat_5k.py](https://github.com/vllm-project/speculators/blob/main/examples/data_generation_and_training/gpt_oss_20b_ultrachat_5k.py)

<!-- https://issues.redhat.com/browse/INFERENG-3398 -->
#### New Llama3 training and validation example script
An example covering data generation, training, and vLLM validation is available for Llama 3.3 70B.
<!-- Where is this script? -->
[TEXT](URL)

### Testing and validation

<!-- https://issues.redhat.com/browse/INFERENG-3397 -->
#### Updated vLLM smoke tests
A smoke test trains a single-layer draft model for one epoch using pre-generated hidden states, then runs it in vLLM with a Llama3 8B target model.

<!-- https://issues.redhat.com/browse/INFERENG-3422 -->
#### New vLLM benchmarking framework
A new automated evaluation framework that benchmarks EAGLE3 speculator models using vLLM and GuideLLM has been added.
Pre-configured evaluation configurations are available for the following models:

- Llama-3.1-8B
- Llama-3.3-70B
- gpt-oss-20B
- Qwen3-8B
- Qwen3-32B

The framework can be reviewed in the 
[examples/evaluate/eval-guidellm](https://github.com/vllm-project/speculators/tree/main/examples/evaluate/eval-guidellm) folder.

To run an evaluation:

```cmd
./run_evaluation.sh -c configs/llama-3.1-8b-eagle3.env
```

The command automatically handles vLLM server startup, runs GuideLLM benchmarks, extracts acceptance rate metrics from logs, and cleans up when complete.

The framework supports multiple dataset types including HuggingFace datasets with colon syntax for specific files (e.g., org/dataset:file.jsonl), local files, and directories.
It includes modular bash scripts following best practices with proper error handling and process management, configurable sampling parameters (`temperature`, `top_p`, `top_k`), and outputs detailed metrics including weighted per-position acceptance rates and conditional acceptance probabilities.
Configuration precedence for the evaluation run is as follows and can be easily changed:

1. CLI arguments
2. config file
3. Framework defaults
