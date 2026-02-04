# Adding New Speculative Decoding Algorithms

This guide explains how to add a new speculative decoding algorithm to the Speculators library.


## Quick Start

Adding a new algorithm requires:

1. **Create algorithm module** under `src/speculators/models`.
2. **Configuration class** with `@register` decorator. When Python imports your module, the `@register("myalgo")` decorator adds your class to a global registry dictionary. The training script looks up `"myalgo"` in the registry to find your class. This is helpful because the training script doesn't need to know about every algorithm and adding a new algorithm doesn't require modifying the training script.

3. **Model class** with `@register` decorator
4. **Training factory methods** as classmethods on the model
5. **CLI arguments** in `train.py`


## Step-by-Step Guide

### 1. Create Algorithm Module

Create a self-contained directory for your algorithm under `src/speculators/models`. See `src/speculators/models/eagle3` as an example. This keeps algorithm logic isolated and maintainable. Each algorithm owns its configuration, model definition, and any custom components. Example file structure:

```
src/speculators/models
|-> eagle3
|-> ...
|-> new_algorithm
    |-> __init__.py
    |-> core.py
    |-> config.py
```

### 2. Implement Configuration Class

Define how your algorithm is configured. The config stores hyperparameters, architectural choices, and other settings. It's serialized when saving models and deserialized when loading them.
In `config.py`, create a configuration class with the `@register` decorator, for example:

```python
from speculators import SpeculatorModelConfig

@SpeculatorModelConfig.register("myalgo")
class MyAlgoSpeculatorConfig(SpeculatorModelConfig):
    speculators_model_type: str = "myalgo"

    # Algorithm-specific parameters
    block_size: int = 8
    num_layers: int = 1
```

**Reference:** See `src/speculators/models/eagle3/config.py` for a complete example.

**Key points:**
- Use `@SpeculatorModelConfig.register("myalgo")` decorator
- Set `speculators_model_type` to match your algorithm name
- Inherit common fields from `SpeculatorModelConfig`
- Add algorithm-specific parameters as needed

### 3. Implement Model Class

Define your algorithm's architecture and training interface. The model class contains model architecture, forward pass logic, and training setup. By implementing the required methods, your algorithm should work seamlessly with the training infrastructure.

In `core.py`, create a model class with the `@register` decorator and required training factory methods.

**`from_training_args()`**: The training script needs a way to build your model from command-line arguments. This classmethod receives training arguments as keyword arguments (**kwargs) and extracts the parameters it needs to construct your model instance. Each algorithm pulls out only the arguments relevant to it. This is similar to how `transformers` uses `.from_pretrained()` - each model knows how to instantiate itself.

**`get_trainer_kwargs()`**: Different algorithms may need different parameters during training vs validation. For example, Eagle3 uses off-policy tokens during training but not validation. This method returns separate kwargs dictionaries for each mode.

**Reference:** See `src/speculators/models/eagle3/core.py` for a complete example.

**Required for the training infrastructure:**

Model attributes:
- `layers`: ModuleList of decoder layers (each layer is individually wrapped by FSDP for distributed training)

Methods:
- `from_training_args(cls, verifier_config, **kwargs)`: Factory method to build from CLI args (receives all args as kwargs)
- `get_trainer_kwargs(**kwargs)`: Returns `(train_kwargs, val_kwargs)` dicts passed to `forward()`
- `forward(...)`: Must return `(output, loss, metrics)` where metrics includes a `"loss"` key

### 4. Export Classes

Make your classes importable from the package. Python's import system requires explicit exports from `__init__.py`. This also provides a clean public API.

In `__init__.py`, export your config and model classes.

``` python
from speculators.models.eagle3.config import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel

__all__ = [
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
]
```

**Reference:** See `src/speculators/models/eagle3/__init__.py`

### 5. Add CLI Arguments (Optional)

Add algorithm-specific command-line arguments to the training script. If your algorithm has unique hyperparameters (like Eagle3's `--ttt-steps` or a custom `--block-size`), users need a way to configure them from the command line. These arguments are passed to your `from_training_args()` method. Only add arguments if your algorithm needs parameters beyond the common ones (verifier path, number of layers, etc.).

**Reference:** See `scripts/train.py`

### 6. Train Your Model

The training script should automatically works with your new algorithm:

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py \
    --speculator-type myalgo \
    --verifier-name-or-path meta-llama/Llama-3.1-8B \
    --num-layers 1 \
    --block-size 8 \
    --data-path ./data \
    --save-path ./checkpoints \
    --epochs 20
```

## How It Works

**The flow during training:**

1. User runs: `python train.py --speculator-type myalgo`
2. Training script calls: `model_class = SpeculatorModel.get_class("myalgo")`
3. Registry returns: `MyAlgoDraftModel` class
4. Script converts args to dict: `vars(args)` and calls: `model_class.from_training_args(verifier_config, **vars(args))`
5. Your factory method extracts the kwargs it needs and builds the model instance
6. Trainer validates the model is registered (via checks in `setup_model()` and `apply_fully_sharded()`)

This pattern is similar to how `transformers` uses `.from_pretrained()` - each model owns its own instantiation logic.

**Reference:** See `scripts/train.py`

## Using Base Components

Shared transformer layer components that can be reused across algorithms. Many speculative decoding algorithms use similar architectural components (decoder layers, attention, normalization). Instead of duplicating code, you can import pre-configured components for different base model architectures.

**When to use:** If your algorithm uses standard transformer components from models like LLaMA or Qwen3, you can import them from `base_components` instead of defining your own. This is especially useful when you only need to customize one layer (like the first layer) while keeping the rest standard.

**Available architectures:** `llama`, `qwen3`

**Reference:**
- Component definitions: `src/speculators/models/base_components.py`
- Usage example: `src/speculators/models/eagle3/model_definitions.py`
