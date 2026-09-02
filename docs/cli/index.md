# CLI Reference

This page provides a comprehensive reference for all command-line interface (CLI) tools available in Speculators.

## Overview

Speculators provides the following CLI commands for different stages of the speculative decoding workflow:

| Command                             | Purpose                                                      | Reference                               |
| ----------------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| `speculators prepare-data`          | Preprocess and tokenize datasets for training                | [→ Details](prepare_data.md)            |
| `speculators generate-offline-data` | Generate hidden states offline using vLLM                    | [→ Details](data_generation_offline.md) |
| `launch_vllm.py`                    | Launch vLLM server configured for hidden states extraction   | [→ Details](launch_vllm.md)             |
| `speculators train`                 | Train speculator models with online or offline hidden states | [→ Details](train.md)                   |
| `speculators regenerate-responses`  | Regenerate dataset responses using a vLLM-served model       | [→ Details](response_regeneration.md)   |
| `speculators stitch-mtp`            | Stitch finetuned MTP weights back into verifier checkpoint   | `speculators stitch-mtp --help`         |
| `speculators convert`               | Convert speculator checkpoints between formats               | `speculators convert --help`            |

## Common Workflows

The diagram below shows the high-level flow for training a speculator model. The offline pipeline runs each stage sequentially, while the online pipeline combines hidden-state extraction and training into a single step.

```mermaid
flowchart TD
    subgraph optional ["Optional: Response Regeneration"]
        A["speculators regenerate-responses\nRegenerate dataset responses for improved model alignment"]
    end

    subgraph offline ["Offline Pipeline"]
        B["speculators prepare-data\nTokenize & format dataset"]
        C["launch_vllm.py\nStart vLLM server"]
        D["speculators generate-offline-data\nExtract hidden states from verifier and cache to disk"]
        E["speculators train\nTrain draft model on saved hidden states"]
    end

    subgraph online ["Online Pipeline"]
        F["speculators prepare-data\nTokenize & format dataset"]
        G["launch_vllm.py\nStart vLLM server"]
        H["speculators train\nExtract hidden states & train in one step"]
    end

    A -- "JSONL conversations" --> B
    A -- "JSONL conversations" --> F
    B --> C --> D -- "hs_i.safetensors files\ncontaining {hidden_states}" --> E
    F --> G --> H

    click B "prepare_data/" _self
    click F "prepare_data/" _self
    click C "launch_vllm/" _self
    click G "launch_vllm/" _self
    click D "data_generation_offline/" _self
    click E "train/" _self
    click A "response_regeneration/" _self
    click H "train/" _self
```
