# Response Regeneration

Takes prompts from existing datasets and regenerates responses through a vLLM-served model. The regenerated responses are used to train speculator models for speculative decoding.

## Usage

Start a vLLM server, then run the script against it:

```bash
python script.py --dataset ultrachat --limit 1000
```

The model name is auto-detected from the vLLM server. Output is written to a JSONL file (auto-named based on dataset and model, or set with `--outfile`).

## Supported Datasets

| Name | Dataset | Default Split | Prompt Format |
|------|---------|---------------|---------------|
| `magpie` | [Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered) | `train` | string |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `train_sft` | string |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `train` | string |
| `code_alpaca` | [HuggingFaceH4/CodeAlpaca_20K](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K) | `train` | string |
| `nemotron` | [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | `chat` | messages |
| `tulu3` | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `train` | messages |
| `wildchat` | [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) | `train` | messages |
| `nemotron_cascade` | [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | `train` | messages |
| `nemotron_ifchat` | [nvidia/Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | `reasoning_off` | messages |

### magpie

300K filtered synthetic instructions generated from Llama-3.1 using the Magpie method. Prompts are single-turn instructions stored in a plain string field. Good general-purpose coverage across tasks.

### ultrachat

200K multi-turn dialogues covering a broad range of topics. The `train_sft` split contains the SFT-ready subset. Prompts are plain strings representing the user's opening message.

### gsm8k

Grade-school math word problems with step-by-step solutions. Uses the `main` subset. Useful for training and evaluating mathematical reasoning capabilities.

### code_alpaca

20K code generation prompts based on the Stanford Alpaca format. Each row contains a plain string prompt describing a coding task. Compact and code-focused, making it useful for training on programming-specific distributions.

### nemotron

NVIDIA's large-scale post-training dataset covering multiple domains. Available splits: `chat`, `math`, `code`, `stem`. Uses a conversational messages format; the first user message is extracted as the prompt.

```bash
python script.py --dataset nemotron --split chat
python script.py --dataset nemotron --split math
python script.py --dataset nemotron --split code
python script.py --dataset nemotron --split stem
```

### tulu3

Allen AI's SFT mixture used to train Tulu 3, containing ~939K examples spanning diverse tasks and sources. Uses a conversational messages format. Broad coverage makes it a good default for general-purpose regeneration.

### wildchat

~529K real user conversations collected from ChatGPT and GPT-4. Captures natural user interaction patterns and phrasing, making it useful for training models on realistic chat distributions.

### nemotron_cascade

NVIDIA's Cascade 2 SFT dataset spanning 8 domains with data generated from multiple frontier models. Each domain is a separate subset: `chat`, `conversational_agent`, `instruction_following`, `math`, `safety`, `science`, `swe`, `terminal_agent`. All subsets use a single `train` split.

```bash
python script.py --dataset nemotron_cascade                                    # chat (default)
python script.py --dataset nemotron_cascade --subset math
python script.py --dataset nemotron_cascade --subset instruction_following
python script.py --dataset nemotron_cascade --subset science
python script.py --dataset nemotron_cascade --subset swe
python script.py --dataset nemotron_cascade --subset conversational_agent
python script.py --dataset nemotron_cascade --subset terminal_agent
python script.py --dataset nemotron_cascade --subset safety
```

### nemotron_ifchat

NVIDIA's instruction-following and chat SFT dataset with synthetic dialogues generated from multiple frontier models (Kimi-K2, GLM-4, Qwen3, etc.). Available in two splits: `reasoning_off` (default) for standard responses and `reasoning_on` for chain-of-thought style responses.

```bash
python script.py --dataset nemotron_ifchat                        # non-reasoning
python script.py --dataset nemotron_ifchat --split reasoning_on   # reasoning
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ultrachat` | Dataset to process (see table above) |
| `--split` | dataset-specific | Dataset split to use |
| `--subset` | dataset-specific | Dataset subset/config name |
| `--endpoint` | `http://127.0.0.1:8000/v1/chat/completions` | vLLM chat completions endpoint |
| `--model` | auto-detected | Model name exposed by vLLM |
| `--limit` | none | Stop after N rows |
| `--concurrency` | `64` | Max concurrent requests |
| `--max-tokens` | `8192` | Max tokens per generation |
| `--outfile` | auto-generated | Output JSONL path |
| `--resume` | off | Skip rows already in outfile |
| `--language-filter` | none | Only process rows matching this language (e.g., `EN`) |

## Adding a New Dataset

Add an entry to `DATASET_CONFIGS` in `script.py`:

```python
"my_dataset": {
    "id": "org/dataset-name",       # HuggingFace dataset ID
    "prompt_field": "instruction",  # field containing the prompt
    "default_split": "train",       # default split to use
    "subset": "main",              # optional: dataset config/subset name
},
```

For datasets using a conversational messages format (list of `{role, content}` dicts), set `prompt_field` to the messages field name and add the role/content field mappings:

```python
"my_chat_dataset": {
    "id": "org/dataset-name",
    "prompt_field": "messages",
    "default_split": "train",
    "messages_role_field": "role",
    "messages_content_field": "content",
},
```

The script will automatically extract the first user message as the prompt.
