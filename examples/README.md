# Training

Speculators currently supports training of Eagle, Eagle 3 and HASS-style speculative decoders. This functionality is still under development and includes the necessary data generation and training steps required to produce the models. Examples of the training process can be found under the [`research`](https://github.com/neuralmagic/speculators/tree/main/research) directory for [Eagle 3](https://github.com/neuralmagic/speculators/blob/main/research/eagle3/README.md) and [HASS](https://github.com/neuralmagic/speculators/blob/main/research/hass/README.md).

# Conversion

## Converting models trained through `speculators`

To properly serve the trained model with vLLM, a conversion step is required for the trained speculator. This step is also listed under the `research` directory.

## Converting models from other libraries

Conversion is also supported of speculative decoder models produced by other research libraries. An example bash script to convert the Eagle 3 model, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` can be found under `convert/eagle3`.

## Conversion Updates
Applying conversion will:

1. Extend the model's config.json by adding a speculators_config. This contains proper EAGLE and EAGLE 3 configuration fields
2. Update model.safetensors  with correct embeddings and remapped weights
3. Enable full vLLM compatibility

Once converted, all models can run using `vllm serve </path/to/convered/model>`
