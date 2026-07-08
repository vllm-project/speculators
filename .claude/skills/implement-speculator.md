---
name: implement-speculator
description: Generate implementation code for a new speculative decoding algorithm in the speculators library and vLLM, following exact codebase patterns.
---

# Implement a New Speculator from Spec

You are implementing a new speculative decoding algorithm based on a spec file. The user provides the algorithm name; the spec is at `.claude/agent_state/specs/<algo_name>.md`.

## Step 1: Load the Spec

Read the implementation spec from `.claude/agent_state/specs/<algo_name>.md`. If it doesn't exist, tell the user to run `/analyze-paper` first.

## Step 2: Read Reference Implementations

Before writing any code, read the reference files to match patterns exactly:

**Always read:**
- `src/speculators/models/__init__.py` — current model registry
- `src/speculators/models/eagle3/__init__.py` — export pattern
- `src/speculators/models/eagle3/config.py` — config class pattern
- `src/speculators/models/eagle3/core.py` — model class pattern (forward, from_training_args, get_trainer_kwargs)
- `scripts/train.py` — CLI args and preprocess_fns dict
- `src/speculators/models/base_components.py` — available transformer components
- `src/speculators/models/metrics.py` — available loss functions

**Read if the spec says to extend an existing model:**
- The parent model's `config.py` and `core.py` (e.g., `models/dflash/core.py` for DFlash variants)

**Read if vLLM changes are needed:**
- `/workspace/vllm/vllm/transformers_utils/configs/speculators/algos.py` — speculator registration
- `/workspace/vllm/vllm/model_executor/models/registry.py` — model registry

## Step 3: Create the Speculators Implementation

Create a new branch for this work:
```bash
cd /workspace/speculators
git checkout -b feat/<algo_name>
```

### 3a. Create `src/speculators/models/<algo_name>/__init__.py`

Follow the eagle3 pattern exactly:
```python
from speculators.models.<algo_name>.config import <AlgoName>SpeculatorConfig
from speculators.models.<algo_name>.core import <AlgoName>DraftModel

__all__ = [
    "<AlgoName>DraftModel",
    "<AlgoName>SpeculatorConfig",
]
```

If there's a custom `shift_batch` or data preprocessing function, export it too.

### 3b. Create `src/speculators/models/<algo_name>/config.py`

Follow the eagle3/config.py pattern:
- Import `SpeculatorModelConfig` from `speculators`
- Use `@SpeculatorModelConfig.register("<algo_name>")`
- Set `speculators_model_type: Literal["<algo_name>"] = "<algo_name>"`
- Include `transformer_layer_config: PretrainedConfig` with the standard `field_serializer` and `field_validator` (copy from eagle3)
- Add algorithm-specific fields from the spec using `Field(default=..., description="...")`
- Set `architectures: list[str]` default

### 3c. Create `src/speculators/models/<algo_name>/core.py`

Follow the eagle3/core.py pattern:
- Use `@SpeculatorModel.register("<algo_name>")`
- Inherit from `DraftVocabMixin, SpeculatorModel` (or parent class if extending)
- Set `config_class` ClassVar
- Set `_keys_to_ignore_on_load_missing` and `_keys_to_ignore_on_save` ClassVars
- Implement `__init__` following the pattern:
  1. Set attention implementation
  2. Call `super().__init__(config=config)`
  3. Call `self._init_vocab(config)`
  4. Create FC/projection layers
  5. Create decoder layers as `self.layers = torch.nn.ModuleList([...])`
  6. Create rotary embeddings
  7. Create layer norms
  8. Call `self.post_init()`
- Implement `target_layer_ids` property
- Implement `forward()` returning `(draft_tokens, loss, metrics)` — the metrics dict MUST contain `"loss_sum"` and `"loss_total"` keys
- Implement `from_training_args(cls, verifier_config, **kwargs)`:
  1. Resolve target layer IDs via `resolve_target_layer_ids()`
  2. Build config with `SpeculatorsConfig` and `VerifierConfig`
  3. Instantiate model, load vocab mappings, load verifier weights
- Implement `get_trainer_kwargs(**kwargs)` returning `(train_kwargs, val_kwargs)`

### 3d. Create optional files

Based on the spec:
- `metrics.py` — if custom loss computation needed (follow `models/eagle3/metrics.py`)
- `data.py` — if custom batch preprocessing needed (follow `models/eagle3/data.py`)
- `model_definitions.py` — if custom decoder layers needed (follow `models/eagle3/model_definitions.py`)
- `attention.py` — if custom attention masks needed (follow `models/eagle3/attention.py`)

### 3e. Modify `src/speculators/models/__init__.py`

Add import line and `__all__` entries following the existing alphabetical pattern.

### 3f. Modify `scripts/train.py`

If the spec requires new CLI arguments:
1. Read the current `parse_args()` function in `scripts/train.py`
2. Add new arguments in the appropriate section (look for algorithm-specific arg groups)
3. If a custom preprocess function exists, add it to the `preprocess_fns` dict

## Step 4: Create the vLLM Implementation

Based on the spec's vLLM section:

### 4a. If mapping to existing inference pathway

Add registration in `/workspace/vllm/vllm/transformers_utils/configs/speculators/algos.py`:
```python
@register_speculator("<algo_name>")
def update_<algo_name>(config, speculators_config):
    # Set architecture, hidden sizes, and algorithm-specific config
    ...
```

Add registry entries in `/workspace/vllm/vllm/model_executor/models/registry.py` to `_SPECULATIVE_DECODING_MODELS`.

### 4b. If new inference method needed

Additionally:
- Create model file at `vllm/model_executor/models/<base>_<algo_name>.py`
- Add to `SpeculativeMethod` in `vllm/config/speculative.py`
- Add routing in `vllm/v1/worker/gpu/spec_decode/__init__.py`
- Add proposer creation in `vllm/v1/worker/gpu_model_runner.py`

## Step 5: Validate

### 5a. Lint and format
```bash
cd /workspace/speculators && make style && make quality
```

Fix any issues found. Common issues:
- Import ordering (ruff will fix)
- Line length (120 char limit)
- Type annotations

### 5b. Run unit tests
```bash
cd /workspace/speculators
/workspace/speculators/.venv/bin/python -m pytest tests/unit/ -x -v 2>&1 | tail -50
```

Check that:
- The new model type registers correctly
- Config serialization/deserialization works
- No import errors

### 5c. Quick forward pass test
Write a minimal test that:
1. Creates a config with small dimensions
2. Instantiates the model
3. Runs a forward pass with dummy data
4. Verifies output shapes

## Step 6: Present Results

Show the user:
1. A summary of files created and modified
2. The full diff (`git diff` and `git diff --cached`)
3. Test results
4. Any issues encountered and how they were resolved
5. Suggest running `/train-speculator <algo_name>` to validate with actual training
