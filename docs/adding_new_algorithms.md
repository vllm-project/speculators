# Adding New Speculative Decoding Algorithms

This guide explains how to add a new speculative decoding algorithm to the Speculators library.

## Quick Start

Adding a new algorithm requires:

1. **Configuration class** with `@register` decorator
2. **Model class** with `@register` decorator
3. **Training factory methods** as classmethods on the model
4. **CLI arguments** in `train.py` (if needed)

## Step-by-Step Guide

### 1. Create Algorithm Module

Create a directory under `src/speculators/models/<algorithm_name>/`:

```bash
mkdir -p src/speculators/models/myalgo
cd src/speculators/models/myalgo
touch __init__.py config.py core.py
```

### 2. Implement Configuration Class

In `config.py`, create a configuration class with the `@register` decorator.

**Reference:** See `src/speculators/models/eagle3/config.py` for a complete example.

- Use `@SpeculatorModelConfig.register("myalgo")` decorator
- Set `speculators_model_type` to a unique identifier
- Inherit common fields from `SpeculatorModelConfig`
- Add algorithm-specific parameters as needed

### 3. Implement Model Class

In `core.py`, create a model class with the `@register` decorator and training factory methods.

**Reference:** See `src/speculators/models/eagle3/core.py` for a complete example:

- Use `@SpeculatorModel.register("myalgo")` decorator
- Set `config_class` to link config and model
- Must have `layers`, `lm_head`, `embed_tokens` attributes for FSDP training
- Implement `from_training_args()` classmethod to create model from CLI args
- Implement `get_trainer_kwargs()` staticmethod to return training/validation kwargs
- `forward()` must return a dict with at least `{"loss": loss}`

### 4. Export Classes

In `__init__.py`, export your config and model classes.

**Reference:** See `src/speculators/models/eagle3/__init__.py`

### 5. Add CLI Arguments (Optional)

If your algorithm needs custom arguments, add them to `scripts/train.py` in the `parse_args()` function.

**Reference:** See `scripts/train.py` lines 230-237 for Eagle3-specific arguments like `--ttt-steps`

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

The training script uses the registry system to dynamically load and instantiate your algorithm. This pattern is similar to how `transformers` uses `.from_pretrained()` - each model owns its own instantiation logic.

**Reference:** See `scripts/train.py` lines 150-178 for how the registry and factory methods are used.

## Using Base Components

The library provides shared transformer components for reuse across algorithms.

**Reference:**
- Component definitions: `src/speculators/models/base_components.py`
- Usage example: `src/speculators/models/eagle3/model_definitions.py` line 54

Available architectures: `llama`, `qwen3`

## Best Practices

1. **Self-contained modules**: Keep all algorithm logic in its own directory
2. **Follow the pattern**: Use the same structure as Eagle3 for consistency
3. **Document your algorithm**: Add docstrings explaining the approach
4. **Test with FSDP**: Ensure distributed training works with `torchrun`
5. **Keep it simple**: Only add complexity when necessary

## Example: Eagle3

See `src/speculators/models/eagle3/` for a complete reference implementation that follows this pattern.
