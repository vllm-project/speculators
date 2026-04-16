---
weight: -99
---

# Algorithms

Speculators supports multiple speculative decoding algorithms. Each algorithm has its own model architecture, training procedure, and configuration format, but all share the same registry system and deployment pipeline.

## Supported Algorithms

| Algorithm                                                                                      | Registry Key | Status         | Description                                                                                                      |
| ---------------------------------------------------------------------------------------------- | ------------ | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| [EAGLE-3](https://github.com/vllm-project/speculators/tree/main/src/speculators/models/eagle3) | `eagle3`     | ✅ Supported   | Feature-level draft model with vocabulary mapping between draft and target tokenizers                            |
| [FastMTP](fast_mtp.md)                                                                         | `mtp`        | 🚧 In Progress | Multi-token prediction using a single recursive MTP layer ([arXiv:2509.18362](https://arxiv.org/abs/2509.18362)) |

✅ = Supported, 🚧 = In Progress

## Adding New Algorithms

See the [Adding New Algorithms](add_new_algorithms.md) guide for step-by-step instructions on implementing a new speculative decoding algorithm in Speculators.

## How the Registry Works

Speculators uses a class registry pattern to dynamically discover and instantiate algorithm implementations. When you register a config or model class:

```python
@SpeculatorModelConfig.register("my_algo")
class MyAlgoConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["my_algo"] = "my_algo"
```

The training script can then look up your algorithm by name:

```python
model_class = SpeculatorModel.get_class("my_algo")
model = model_class.from_training_args(verifier_config, **args)
```

This keeps algorithm implementations self-contained and avoids modifying shared training infrastructure when adding new algorithms.
