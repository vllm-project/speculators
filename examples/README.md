# Training

Speculators currently supports training of Eagle, Eagle 3 and HASS-style speculative decoders. This functionality is still under development and includes the necessary data generation and training steps required to produce the models. Examples of the training process can be found under the `research` directory

# Conversion

## Converting models trained through `speculators`

To properly serve the trained model with vLLM, a conversion step is required for the trained speculator. This step is also listed under the `research` directory.

## Converting models from other libraries

Conversion is also supported of speculative decoder models produced by other research libraries. An example bash script to convert the Eagle 3 model, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` can be found under `convert/eagle3`.

Applying conversion will:

1. Extends the model's config.json by adding a speculators_config. This contains proper EAGLE and EAGLE 3 configuration fields
2. Update model.safetensors  with correct embeddings and remapped weights
3. Enable full vLLM compatibility

Once converted, all models can run using `vllm serve </path/to/convered/model>`
