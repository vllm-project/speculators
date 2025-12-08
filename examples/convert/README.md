# Running Speculators Models

All models trained through `speculators` include a `speculators_config` in their config.json. These models are in the speculators format and directly runnable in vLLM, using `vllm serve </path/to/speculator/model>` which will apply all the speculative decoding parameters defined in the `speculators_config`.

# Converting models from third-party libraries

It may also be desirable to convert third-party models to the `speculators` format. Conversion is supported of speculative decoder models produced by other research libraries. An example bash script to convert the Eagle3 model, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` can be found under `convert/eagle3`.

Applying conversion will:

1. Extend the model's config.json by adding a speculators_config. This contains proper EAGLE and EAGLE 3 configuration fields
2. Update model.safetensors with correct embeddings and remapped weights
3. Enable full vLLM compatibility

Once converted, all models can run using `vllm serve </path/to/speculator/model>`
