# Serve in vLLM

This tutorial shows you how to deploy your trained speculator model for production inference using vLLM.

After training a speculator, deploying it with vLLM is straightforward. vLLM automatically recognizes the `speculators_config` in your model and enables speculative decoding.

## Basic Serving

```bash
vllm serve ./checkpoints/checkpoint_best
```

That's it! vLLM will:

1. Load your speculator model
2. Read the `speculators_config` from config.json
3. Load the target/verifier model
4. Enable speculative decoding automatically

## Long form command

```bash
vllm serve Qwen/Qwen3-8B \
  -tp 1 \
  --speculative-config '{
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "method": "eagle3"
  }'
```

For the long form command, pass in the target model first (e.g. `Qwen/Qwen3-8B`) and then specify the draft model in the speculative config (e.g. `RedHatAI/Qwen3-8B-speculator.eagle3`).

The long form command can be used to change the `num_speculative_tokens` or to use a different target model with the speculator. This can be used to combine a quantized target model, with a speculative decoding model for even better performance.
