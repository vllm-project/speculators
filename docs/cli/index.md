# CLI Reference

This page provides a comprehensive reference for all command-line interface (CLI) tools available in Speculators.

## Overview

Speculators provides four main CLI scripts for different stages of the speculative decoding workflow:

| Script                       | Purpose                                                      | Reference                               |
| ---------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| `prepare_data.py`            | Preprocess and tokenize datasets for training                | [→ Details](prepare_data.md)            |
| `data_generation_offline.py` | Generate hidden states offline using vLLM                    | [→ Details](data_generation_offline.md) |
| `launch_vllm.py`             | Launch vLLM server configured for hidden states extraction   | [→ Details](launch_vllm.md)             |
| `train.py`                   | Train speculator models with online or offline hidden states | [→ Details](train.md)                   |

## Common Workflows

The diagram below shows the high-level flow for training a speculator model. The offline pipeline runs each stage sequentially, while the online pipeline combines hidden-state extraction and training into a single step.

```mermaid
flowchart TD
    subgraph optional ["Optional: Response Regeneration"]
        A["response_regeneration/script.py\nRegenerate model responses to dataset prompts for improved model alignment"]
    end

    subgraph offline ["Offline Pipeline"]
        B["prepare_data.py\nTokenize & format dataset"]
        C["launch_vllm.py\nStart vLLM server"]
        D["data_generation_offline.py\nExtract hidden states from verifier and cache to disk"]
        E["train.py \nTrain draft model on saved hidden states"]
    end

    subgraph online ["Online Pipeline"]
        F["prepare_data.py\nTokenize & format dataset"]
        G["launch_vllm.py\nStart vLLM server"]
        H["train.py \nExtract hidden states & train in one step"]
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
    click H "train/" _self
```
