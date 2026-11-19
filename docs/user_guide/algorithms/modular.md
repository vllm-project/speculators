# Modular block models

The `modular` model composes a backbone and a correction module independently. It currently supports the anchored-block family:

| Component             | Choices                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------- |
| Backbone              | DFlash; DFlash2 convolutional decoder layers                                                 |
| Correction            | Identity; DSpark Markov (vanilla, gated, or RNN); DFlash2 candidate selector; XPress refiner |
| Confidence prediction | Optional with identity or Markov correction; Markov features are optional inputs             |

These choices define trainable architectures. Combining separately trained weights does not establish decoding quality. EAGLE3, P-EAGLE, and MTP backbones are outside the current composition interface.

## Configure a training run

Start with [the training configuration](https://github.com/vllm-project/speculators/blob/main/examples/train/modular_dflash2_markov.yaml) and its [component recipe](https://github.com/vllm-project/speculators/blob/main/examples/train/recipes/dflash2_markov.yaml). From the repository root:

```bash
python -m speculators.train --config examples/train/modular_dflash2_markov.yaml
```

Set the verifier, preprocessed data path, captured target layers, and output directory for your run. The normal data preparation and distributed training workflows still apply. The example combines a DFlash2 backbone with Markov correction and a confidence head:

```yaml
backbone:
  kind: dflash2
  conv_kernel_size: 2
  conv_group_size: 16
correction:
  kind: markov
  rank: 128
  markov_head_type: vanilla
confidence:
  enabled: true
  with_correction_features: true
```

The recipe file contains component architecture choices. Shared block layout options, attention settings, losses, optimizer settings, and objective hyperparameters remain in the normal training configuration. The correction and confidence choices select their matching training objective; an explicitly supplied incompatible objective is rejected.

`--composition-config` also works with `--draft-config` to combine a component recipe with a saved transformer decoder configuration. It cannot override `--from-pretrained`, which already specifies the complete architecture. Legacy head architecture flags such as `--markov-rank` and `--conv-group-size` are rejected for modular runs; set them in the component recipe. Objective options such as `--confidence-head-alpha` retain their existing behavior.

The candidate selector currently requires the full verifier vocabulary. The current backbones require matching verifier/draft hidden sizes. Convolution group size must divide the hidden size, and convolution kernel size cannot exceed block size. Confidence features from correction require a Markov correction module. Invalid combinations fail during configuration/model construction.

Training saves the resolved architecture in each checkpoint's `config.json` and in `resolved_model_config.json` at the run root, so changing the source recipe file does not erase the architecture used by that run. Existing command, version, and code-patch provenance also remains available.

## Load and save

Native modular checkpoints use `backbone.*` and `correction.*` parameter names:

```python
from speculators import SpeculatorModel

model = SpeculatorModel.from_pretrained("./output/checkpoint")
model.save_pretrained("./output/native-copy")
```

The existing `dflash`, `dflash2`, `dspark`, and `xpress` model names remain available with their original configuration defaults. These named algorithms save and load flat checkpoints. Modular models save and load hierarchical checkpoints.

Full training resume requires the same Speculators version that wrote the checkpoint. Use `--from-pretrained` to initialize a new run from compatible model weights.

## Serving support

Native modular mixtures currently require a serving implementation that understands their component recipe and proposal behavior. This change does not add native modular execution to vLLM. The `correct()` and `score_candidates()` APIs operate without training targets, but they are component operations rather than a serving scheduler or cache implementation.

When launching vLLM through `scripts/launch_vllm.py`, pass `--provenance-dir` pointing at the associated run's output/checkpoint directory. Include the run's command, checkpoint hashes, and patch artifacts when publishing models or evaluation results.
