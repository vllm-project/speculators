# Model Formats

## What's Covered

In this document, you will learn:

1. The structure and organization of the Speculators model formats
2. How to save, load, and share speculator models across the ecosystem
3. The design principles that guide Speculators' model format implementation

Before reading this document, you should be familiar with the following concepts:

- Basic understanding of machine learning model formats and serialization
- Familiarity with Hugging Face's model hub and Transformers library
- General knowledge of speculative decoding and its benefits

## Overview

Speculators employs a model format designed with ecosystem compatibility and user experience as top priorities. This format serves as a standard and bridge between various speculative decoding implementations and the broader machine learning ecosystem, particularly focusing on compatibility with Hugging Face's model hub, Speculators' implementations built on top of Transformers, and other platforms like vLLM. With the Hugging Face format as a foundation, Speculators enables extensibility and usability for practitioners consuming existing algorithms and researchers developing new ones.

The core components that make up the Speculators model format are the following:

- Hugging Face compatible directory structure and file formats
- Speculators-specific extensions through the `speculators_config` key in the `config.json` file
- Standardized interfaces for saving, loading, and pushing models to the Hugging Face Hub

At the end of this document, examples of configuration files for popular algorithms are provided.

## Components

### Base Model Format

The base model format for Speculators builds directly on the standard Hugging Face model format and represents the specific draft model, ensuring compatibility with existing tools, libraries, and workflows. A typical Speculators model directory contains:

```
model_directory/
├── config.json           # Model configuration file
├── model.safetensors     # Model weights in safetensors format
└── README.md             # Optional documentation
```

Where `config.json` and `model.safetensors` define the draft model and its weights, respectively. Standardizing the `config.json` file as the base for the draft model has the additional benefit for independent draft models, so they only need the speculators-specific addition.

#### `config.json`

The `config.json` file represents specifics about the draft model's architecture, hyperparams, and configurations. It extends the Transformers PretrainedConfig class and serves as the central configuration for a Speculators model, containing all necessary metadata and parameters. Core required keys in the config file include:

- `architectures`: A list containing the model architecture class for the Speculators draft model (e.g., "MLPDraftModel"), or the Hugging Face model class for independent draft models.
- `model_type`: A string identifying the Speculators draft model type (e.g., "mlp_draft_model"), or the Hugging Face model type for independent draft models.
- `torch_dtype`: The data type used for the model weights (e.g., "float32", "bfloat16", "float16")
- `transformers_version`: The version of Transformers used to create the model
- `speculators_version`: The version of Speculators used to create the model
- `inputs`: A list defining the expected inputs to the draft model about the attachment points to the verifier (e.g., "input_ids", "input_embeddings", "layer.0")

Additionally, the file contains implementation-specific keys that define the draft model's architecture and hyperparameters, such as hidden layer dimensions, activation functions, and other configuration parameters depending on the specific Drafter architecture.

Example of a minimal `config.json`:

```json
{
    "architectures": ["MLPDraftModel"],
    "model_type": "mlp_draft_model",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.35.0",
    "speculators_version": "0.1.0",
    "inputs": ["input_embeddings"],
    "...": "..."
}
```

Future format versions may support additional keys for advanced features, such as quantization and compression methods, to further enhance the model's performance and usability.

#### `safetensors` Files

Speculators adopts the `safetensors` format as the standard for storing the draft model's weights, providing multiple benefits:

- **Security**: Unlike pickle-based formats, is secure against arbitrary code execution
- **Efficiency**: Optimized for fast loading times and memory mapping capabilities
- **Compatibility**: Widespread adoption across the Hugging Face ecosystem and related tools
- **Mmap Support**: Allows accessing tensor data without loading the entire model into memory

A typical Speculators model includes one or more safetensors files containing the serialized weights of the draft model:

```
model_directory/
├── model.safetensors           # Single file format for smaller models
# OR
├── model-00001-of-00003.safetensors  # Sharded format for larger models
├── model-00002-of-00003.safetensors
└── model-00003-of-00003.safetensors
```

Future versions of the format may support additional file formats, such as compressed tensors formats, built on top of the safetensors format, to enhance model size and loading performance further.

### Speculators Format Extensions

The Speculators specific extensions to the model format are centralized within the `config.json` file, specifically under the `speculators_config` key. This design choice allows Speculators to maintain compatibility with the Hugging Face model format while providing additional functionality tailored to speculative decoding.

The `speculators_config` dictionary contains subkeys that are broken apart by the core concepts for speculative decoding inference, with the following keys:

- `algorithm`: The name of the speculative decoding algorithm used by the model
- `proposal_methods`: A list of dictionaries defining the supported token proposal strategies for the speculator
- `default_proposal_method`: The default token proposal method to use when generating tokens
- `verifier`: A dictionary defining the target verifier model the speculator was created for

#### Algorithm

The `algorithm` field is a required string that specifies the speculative decoding algorithm used by the model. It serves as a primary identifier for the algorithm with which the speculator model was created and intended to be used. It additionally allows the Speculators library and other tools to automatically load the correct implementation and associated utilities when a model is loaded.

This field must match one of the supported algorithms in the Speculators library, such as:

- `"eagle"`, `"eagle_2"`, `"eagle_3"` - Eagle speculator variants based on Transformer architecture for the draft model
- `"hass"` - Similar to Eagle based on the Transformer architecture for the draft model
- `"mlp_speculator"` - Based on a multi-layer perceptron (MLP) architecture for the draft model
- `"specdec"` - An independent speculator model

Example usage in a config:

```json
{
    "speculators_config": {
        "algorithm": "eagle",
        "...": "..."
    }
}
```

#### Token Proposal Methods

The `proposal_methods` field is a required list of dictionaries defining a speculator's supported token proposal strategies. This field works alongside the `default_proposal_method` key, which specifies which method to use by default.

Token proposal methods determine how a speculator generates candidate tokens and how these tokens are verified with the verifier model. By supporting multiple potential methods and enabling the algorithm to define what should work, they provide adaptability to different use cases and performance requirements. The method-specific parameters are the best defaults to ship the model, but implementations may optionally allow users to override these parameters at runtime.

Each dictionary in the `proposal_methods` list must contain:

- A `proposal_type` key identifying the method (e.g., "greedy", "sample", "tree")
- Method-specific configuration parameters that control the behavior of the proposal strategy

Some common proposal methods include:

- `"greedy"`: Deterministic decoding that selects the highest probability token at each step
- `"sample"`: Samples from the top-p probability mass of the predicted tokens at each step
- `"tree"`: Generates a tree of possible tokens for more likely matches at each step

Example usage in a config:

```json
{
    "proposal_methods": [
        {
            "proposal_type": "greedy",
            "draft_tokens": 5
        },
        {
            "proposal_type": "sample",
            "draft_tokens": 5,
            "temperature": 0.8,
            "top_p": 0.5
        },
        {
            "proposal_type": "tree",
            "tree_type": "static",
            "initial_branching_factor": 4,
            "branching_factor": 2,
            "draft_tokens": 5
        }
    ],
    "default_proposal_method": "greedy"
}
```

#### Verifier

The `verifier` field is a required dictionary defining the target verifier model for which the draft model was created. The dictionary serves two primary purposes:

1. Enabling automatic loading of the associated verifier model if one is not provided at runtime
2. Providing validation parameters to ensure compatibility when using different verifiers with a trained draft model

There are several required and optional keys within the `verifier` dict, enabling the above functionality:

- Keys for automatic loading
  - `name_or_path`: The Hugging Face model ID or local path to automatically load the verifier
- Keys for model architecture compatibility validation
  - `architectures`: List of model architecture classes for compatibility validation
  - `hidden_size`: Hidden dimension size used for compatibility checks
  - `intermediate_size`: Intermediate dimension size for compatibility validation
  - Additional model configuration parameters from the original verifier model
- Keys for tokenizer compatibility validation
  - `vocab_size`: Size of the vocabulary used by the tokenizer
  - `max_position_embeddings`: Maximum position embeddings supported
  - `bos_token_id`: Beginning of sequence token ID
  - `eos_token_id`: End of sequence token ID
  - Additional tokenizer configuration parameters from the original verifier model

Example of a verifier configuration:

```json
{
    "verifier": {
        "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "bos_token_id": 128000,
        "eos_token_id": [
            128001,
            128008,
            128009
        ]
    }
}
```

### Interfaces

The Speculators library provides a set of standardized interfaces for saving, loading, and pushing models to the Hugging Face Hub. These interfaces build on top of and maintain compatibility with the core interfaces in both PyTorch and the Transformers library, ensuring a familiar experience for users of these ecosystems.

#### Speculator Class

The core `Speculator` class that implements speculative decoding algorithms extends PyTorch's `nn.Module` and Transformers' `PushToHubMixin`, providing a consistent interface for model operations:

- `from_pretrained()`: Loads a pretrained speculator model from a local directory or the Hugging Face Hub
- `save_pretrained()`: Saves a speculator model to a local directory with optional uploading to the Hugging Face Hub
- `push_to_hub()`: Directly pushes a model to the Hugging Face Hub for sharing with the community

For detailed usage and examples, refer to the [entrypoints documentation](./entrypoints.md).

#### SpeculatorConfig Class

The `SpeculatorConfig` class extends the Transformers `PretrainedConfig` class, providing compatible APIs with the following methods:

- `from_pretrained()`: Loads a speculator configuration from a local directory or the Hugging Face Hub
- `save_pretrained()`: Saves a speculator configuration to a local directory
- `push_to_hub()`: Directly pushes a configuration to the Hugging Face Hub for sharing with the community

For detailed usage and examples, refer to the [entrypoints documentation](./entrypoints.md).

## Examples

This section provides examples of `config.json` files for popular speculative decoding algorithms.

### Eagle

```json
{
    "architectures": ["TransformerDraftModel"],
    "model_type": "transformer_draft_model",
    "torch_dtype": "bfloat16",
    "transformers_version": "X.X.X",
    "speculators_version": "X.X.X",
    "inputs": ["input_embeddings", "hidden_states[-2]"],
    "inputs_hidden_states_after_layer_norm": false,
    "transformer_architecture": "LlamaDecoderLayer",
    "transformer_input_type": "projection",
    "transformer_include_output_layer_norm": false,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [
        128001,
        128008,
        128009
    ],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "use_cache": true,
    "vocab_size": 128256,
    "speculators_config": {
        "algorithm": "eagle",
        "proposal_methods": [
            {
                "proposal_type": "greedy",
                "draft_tokens": 5
            },
            {
                "proposal_type": "sample",
                "draft_tokens": 5,
                "temperature": 0.8,
                "top_p": 0.5
            },
            {
                "proposal_type": "tree",
                "tree_type": "static",
                "initial_branching_factor": 4,
                "branching_factor": 2,
                "depth": 5
            }
        ],
        "default_proposal_method": "tree",
        "verifier": {
            "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "bos_token_id": 128000,
            "eos_token_id": [
                128001,
                128008,
                128009
            ]
        }
    }
}
```

### Eagle-2

```json
{
    "architectures": ["TransformerDraftModel"],
    "model_type": "transformer_draft_model",
    "torch_dtype": "bfloat16",
    "transformers_version": "X.X.X",
    "speculators_version": "X.X.X",
    "inputs": ["input_embeddings", "hidden_states[-2]"],
    "inputs_hidden_states_after_layer_norm": false,
    "transformer_architecture": "LlamaDecoderLayer",
    "transformer_input_type": "projection",
    "transformer_include_output_layer_norm": false,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [
        128001,
        128008,
        128009
    ],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "use_cache": true,
    "vocab_size": 128256,
    "speculators_config": {
        "algorithm": "eagle_2",
        "proposal_methods": [
            {
                "proposal_type": "greedy",
                "draft_tokens": 5
            },
            {
                "proposal_type": "sample",
                "draft_tokens": 5,
                "temperature": 0.8,
                "top_p": 0.5
            },
            {
                "proposal_type": "tree",
                "tree_type": "context_aware",
                "max_tokens": 64,
                "max_branching_factor": 5,
                "max_depth": 5
            }
        ],
        "default_proposal_method": "tree",
        "verifier": {
            "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "bos_token_id": 128000,
            "eos_token_id": [
                128001,
                128008,
                128009
            ]
        }
    }
}
```

### Eagle-3

```json
{
    "architectures": ["FusedTransformerDraftModel"],
    "model_type": "fused_transformer_draft_model",
    "torch_dtype": "bfloat16",
    "transformers_version": "X.X.X",
    "speculators_version": "X.X.X",
    "inputs": ["input_ids", "hidden[3]", "hidden[12]", "hidden[-2]"],
    "inputs_hidden_states_after_layer_norm": false,
    "transformer_architecture": "LlamaDecoderLayer",
    "transformer_input_type": "projection",
    "transformer_include_output_layer_norm": false,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [
        128001,
        128008,
        128009
    ],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "use_cache": true,
    "vocab_size": 128256,
    "speculators_config": {
        "algorithm": "eagle_3",
        "proposal_methods": [
            {
                "proposal_type": "greedy",
                "draft_tokens": 5
            },
            {
                "proposal_type": "sample",
                "draft_tokens": 5,
                "temperature": 0.8,
                "top_p": 0.5
            },
            {
                "proposal_type": "tree",
                "tree_type": "context_aware",
                "max_tokens": 64,
                "max_branching_factor": 5,
                "max_depth": 5
            }
        ],
        "default_proposal_method": "tree",
        "verifier": {
            "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "bos_token_id": 128000,
            "eos_token_id": [
                128001,
                128008,
                128009
            ]
        }
    }
}
```

### HASS

```json
{
    "architectures": ["TransformerDraftModel"],
    "model_type": "transformer_draft_model",
    "torch_dtype": "bfloat16",
    "transformers_version": "X.X.X",
    "speculators_version": "X.X.X",
    "inputs": ["input_embeddings", "hidden_states[-2]"],
    "inputs_hidden_states_after_layer_norm": false,
    "transformer_architecture": "LlamaDecoderLayer",
    "transformer_input_type": "projection_with_bias",
    "transformer_include_output_layer_norm": false,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [
        128001,
        128008,
        128009
    ],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "use_cache": true,
    "vocab_size": 128256,
    "speculators_config": {
        "algorithm": "hass",
        "proposal_methods": [
            {
                "proposal_type": "greedy",
                "draft_tokens": 5
            },
            {
                "proposal_type": "sample",
                "draft_tokens": 5,
                "temperature": 0.8,
                "top_p": 0.5
            },
            {
                "proposal_type": "tree",
                "tree_type": "context_aware",
                "max_tokens": 64,
                "max_branching_factor": 5,
                "max_depth": 5
            }
        ],
        "default_proposal_method": "tree",
        "verifier": {
            "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "bos_token_id": 128000,
            "eos_token_id": [
                128001,
                128008,
                128009
            ]
        }
    }
}
```

### MLP Speculator

```json
{
    "architectures": ["MLPDraftModel"],
    "model_type": "mlp_draft_model",
    "torch_dtype": "bfloat16",
    "transformers_version": "X.X.X",
    "speculators_version": "X.X.X",
    "inputs": ["input_embeddings", "hidden_states[-1]"],
    "inputs_hidden_states_after_layer_norm": false,
    "hidden_size": 4096,
    "intermediate_size": 4096,
    "vocab_size": 128256,
    "draft_tokens": 5,
    "tie_weights": false,
    "speculators_config": {
        "algorithm": "mlp_speculator",
        "proposal_methods": [
            {
                "proposal_type": "greedy",
                "draft_tokens": 5
            }
        ],
        "default_proposal_method": "greedy",
        "verifier": {
            "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "bos_token_id": 128000,
            "eos_token_id": [
                128001,
                128008,
                128009
            ]
        }
    }
}
```

### SpecDec

```json
{
    "architectures": [
        "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [
        128001,
        128008,
        128009
    ],
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0.dev0",
    "use_cache": true,
    "vocab_size": 128256,
    "speculators_version": "X.X.X",
    "speculators_config": {
        "algorithm": "specdec",
        "proposal_methods": [
            {
                "proposal_type": "greedy",
                "max_new_tokens": 5,
            }
        ],
        "default_proposal_method": "greedy",
        "verifier": {
            "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "bos_token_id": 128000,
            "eos_token_id": [
                128001,
                128008,
                128009
            ]
        }
    },
}
```

## Related Resources

Related Docs:

- [Entrypoints Overview](./entrypoints.md) - This doc provides detailed information about saving and loading speculators.
- [Architecture Overview](./architecture.md) - This doc provides detailed information about the Speculators architecture that powers the entry points.

Related External Resources:

- [Speculative Decoding Overview](https://arxiv.org/abs/2401.07851) - A general-purpose, survey paper on speculative decoding, its benefits, and its applications.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Documentation for the Transformers library, which Speculators integrates with
