# Data Generation and Training

Speculators currently supports training of Eagle3 speculative decoders. For full details on all the steps described below, see [README.md](../../scripts/README.md)

This process is currently broken down into three key steps:

1. Data Generation
2. Vocab Mapping
3. Training

## Data Generation

Generate hidden states for training using vLLM. Dataset values are passed through the target or verifier model and generated hidden states are saved to disk for further use. `scripts/data_generation_offline.py` provides the main entry point for generating training data for Eagle3 models.

Once completed, the following files will be generated on disk:
1. `token_freq.pt` (the token frequency distribution file)
2. `data_config.json` (data metadata)
3. data pt files containing the hidden state values

Note: this process uses vLLM and requires the `datagen` optional install.

## Vocab Mapping

Build `d2t` and `t2d` files from the token frequency distribution file. `scripts/build_vocab_mapping.py` is the main entrypoint for this step.

Once completed, the following files will be generated from this step on disk:
1. `d2t.npy`
2. `t2d.npy`

## Training

Train an Eagle3 draft model or `speculator`.
Currently, training is supported for:

1. Single-Layer and Multi-Layer Draft Models for Non-MoE models
3. Single-Layer and Multi-Layer Draft Models of certain Non-Vision MoEs

For a full list of models with support, see: https://github.com/vllm-project/speculators/blob/main/README.md

`scripts/train.py` provides the main entry point for training Eagle3 models with support for single and multi GPU training using FSDP.

# Examples

The files in this folder provide end-to-end examples which run the three steps listed above for Llama3 and Qwen3 draft
models. If at any point a step fails, you can rerun the script and continue from the last step. Seprate steps may also run using the individual scripts listed above.

The table below shows current benchmarked performance of draft models trained e2e using speculators and validated in vLLM:

<TODO INSERT TABLE>