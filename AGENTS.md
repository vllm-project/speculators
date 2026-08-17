# Speculators Agent Instructions

## Launching vLLM

- **For training** (hidden state extraction): always use `scripts/launch_vllm.py`. Never use `python -m vllm.entrypoints.openai.api_server` directly — it misses `--speculative_config` and `--kv_transfer_config` which are required for hidden state extraction.
- **For eval**: launch vLLM directly with `--spec-model` and `--spec-tokens` (e.g. `python -m vllm.entrypoints.openai.api_server --model <target> --spec-model <checkpoint> --spec-tokens <N>`).
