# Response Regeneration

Takes prompts from existing datasets and regenerates responses through a vLLM-served model. The regenerated responses are used to train speculator models for speculative decoding.

## Usage

Start a vLLM server, then run the script against it:

```bash
python script.py --dataset ultrachat --limit 1000
```

The model name is auto-detected from the vLLM server. Output is written to a JSONL file (auto-named based on dataset and model, or set with `--outfile`).

## Supported Datasets

| Name | Dataset | Default Split | Samples | Size | Prompt Format |
|------|---------|---------------|---------|------|---------------|
| `magpie` | [Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered) | `train` | 300K | 2.1 GB | string |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `train_sft` | 208K | 1.4 GB | string |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `train` | 7.5K | 4 MB | string |
| `code_alpaca` | [HuggingFaceH4/CodeAlpaca_20K](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K) | `train` | 18K | 5 MB | string |
| `nemotron` | [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | `chat` | 1.4M | 6.1 GB | messages |
| `tulu3` | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `train` | 939K | 1.5 GB | messages |
| `wildchat` | [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) | `train` | 529K | 3.1 GB | messages |
| `nemotron_cascade` | [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | `train` | millions | ~593 GB | messages |
| `nemotron_ifchat` | [nvidia/Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | `reasoning_off` | ~2M | ~16 GB | messages |
| `nemotron_ifchat_v1` | [nvidia/Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | `structured_outputs` | — | — | messages |
| `nemotron_agentic` | [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | `interactive_agent` | — | — | messages |
| `nemotron_competitive_v2` | [nvidia/Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) | `competitive_coding_python` | — | — | messages |
| `nemotron_competitive_v1` | [nvidia/Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | `infinibyte_part00` | — | — | messages |
| `nemotron_math` | [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | `high_part00` | — | — | messages |
| `nemotron_science` | [nvidia/Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | `MCQ` | — | — | messages |
| `nemotron_swe` | [nvidia/Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) | `agentless` | — | — | messages |
| `longalign` | [zai-org/LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k) | `train` | 9.9K | 0.6 GB | messages |
| `open_perfectblend` | [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) | `train` | 1.42M | ~4 GB | messages |
| `metamathqa` | [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | `train` | 395K | — | string |
| `ultrainteract` | [openbmb/UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft) | `train` | 289K | — | string |
| `orca_math` | [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | `train` | 200K | — | string |
| `ultrafeedback` | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | `train_sft` | 187K | — | string |
| `evol_codealpaca` | [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | `train` | 111K | — | string |
| `autoif` | [Post-training-Data-Flywheel/AutoIF-instruct-61k](https://huggingface.co/datasets/Post-training-Data-Flywheel/AutoIF-instruct-61k) | `train` | 61K | — | messages |
| `lmsys_arena` | [mlabonne/lmsys-arena-human-preference-55k-sharegpt](https://huggingface.co/datasets/mlabonne/lmsys-arena-human-preference-55k-sharegpt) | `train` | 57K | — | messages |
| `openhermes` | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | `train` | 1M | — | messages |
| `openr1_math` | [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | `train` | 94K–225K | — | messages |
| `numinamath` | [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) | `train` | 72K | — | messages |
| `codeforces_cots` | [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) | `train` | 48K | — | string |
| `codeforces` | [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) | `train` | 10K | — | string |
| `taco` | [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO) | `train` | 25K | — | string |
| `xlam_function_calling` | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | `train` | 60K | — | string |
| `apigen_mt` | [Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | `train` | 5K | — | messages |
| `swe_rebench` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | `train` | 67K | — | messages |

### magpie

300K filtered synthetic instructions generated from Llama-3.1 using the Magpie method. Prompts are single-turn instructions stored in a plain string field. Good general-purpose coverage across tasks.

### ultrachat

200K multi-turn dialogues covering a broad range of topics. The `train_sft` split contains the SFT-ready subset. Prompts are plain strings representing the user's opening message.

### gsm8k

Grade-school math word problems with step-by-step solutions. Uses the `main` subset. Useful for training and evaluating mathematical reasoning capabilities.

### code_alpaca

20K code generation prompts based on the Stanford Alpaca format. Each row contains a plain string prompt describing a coding task. Compact and code-focused, making it useful for training on programming-specific distributions.

### nemotron

NVIDIA's large-scale post-training dataset covering multiple domains. Uses a conversational messages format; the first user message is extracted as the prompt.

| Split | Samples | Size |
|-------|---------|------|
| `chat` | 628K | 3.4 GB |
| `stem` | 355K | 1.2 GB |
| `math` | 239K | 0.5 GB |
| `code` | 175K | 1.0 GB |

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

NVIDIA's Cascade 2 SFT dataset spanning 8 domains with data generated from multiple frontier models. Each domain is a separate subset. All subsets use a single `train` split. Note: the HuggingFace metadata underreports sizes — actual sizes are much larger than shown in the dataset viewer.

| Subset | Size |
|--------|------|
| `chat` | 213 GB |
| `math` | 246 GB |
| `science` | 46 GB |
| `swe` | 36 GB |
| `terminal_agent` | 31 GB |
| `conversational_agent` | 17 GB |
| `instruction_following` | 3.6 GB |
| `safety` | 14 MB |

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

NVIDIA's instruction-following and chat SFT dataset with synthetic dialogues generated from multiple frontier models (Kimi-K2, GLM-4, Qwen3, etc.). Available in two splits for standard and chain-of-thought style responses. ~2M total samples across both splits.

| Split | Size |
|-------|------|
| `reasoning_off` | 6.0 GB |
| `reasoning_on` | 10.1 GB |

```bash
python script.py --dataset nemotron_ifchat                        # non-reasoning
python script.py --dataset nemotron_ifchat --split reasoning_on   # reasoning
```

### nemotron_ifchat_v1

NVIDIA's instruction-following chat v1 dataset. The `structured_outputs` split contains prompts requiring structured output formats.

```bash
python script.py --dataset nemotron_ifchat_v1                              # structured_outputs (default)
```

### nemotron_agentic

NVIDIA's agentic SFT dataset v2 covering interactive agents, tool calling, and search tasks. Select domain via `--split`.

| Split | Description |
|-------|-------------|
| `interactive_agent` | Interactive agent conversations |
| `tool_calling` | Tool calling examples |
| `search` | Search-based agent tasks |

```bash
python script.py --dataset nemotron_agentic                                # interactive_agent (default)
python script.py --dataset nemotron_agentic --split tool_calling
python script.py --dataset nemotron_agentic --split search
```

### nemotron_competitive_v2

NVIDIA's competitive programming SFT dataset v2 with Python and C++ solutions. Select language via `--split`.

| Split | Description |
|-------|-------------|
| `competitive_coding_python` | Python solutions |
| `competitive_coding_cpp` | C++ solutions |

```bash
python script.py --dataset nemotron_competitive_v2                         # Python (default)
python script.py --dataset nemotron_competitive_v2 --split competitive_coding_cpp
```

### nemotron_competitive_v1

NVIDIA's competitive programming dataset v1 with Infinibyte-generated data, split into two parts.

| Split | Description |
|-------|-------------|
| `infinibyte_part00` | Part 0 |
| `infinibyte_part01` | Part 1 |

```bash
python script.py --dataset nemotron_competitive_v1                         # part00 (default)
python script.py --dataset nemotron_competitive_v1 --split infinibyte_part01
```

### nemotron_math

NVIDIA's math SFT dataset v2 with problems at three difficulty levels. The high-difficulty split is further divided into three parts.

| Split | Description |
|-------|-------------|
| `high_part00` | High difficulty, part 0 |
| `high_part01` | High difficulty, part 1 |
| `high_part02` | High difficulty, part 2 |
| `medium` | Medium difficulty |
| `low` | Low difficulty |

```bash
python script.py --dataset nemotron_math                                   # high_part00 (default)
python script.py --dataset nemotron_math --split high_part01
python script.py --dataset nemotron_math --split high_part02
python script.py --dataset nemotron_math --split medium
python script.py --dataset nemotron_math --split low
```

### nemotron_science

NVIDIA's science dataset v1 with multiple-choice questions and reasoning Q&A.

| Split | Description |
|-------|-------------|
| `MCQ` | Multiple-choice questions |
| `RQA` | Reasoning question answering |

```bash
python script.py --dataset nemotron_science                                # MCQ (default)
python script.py --dataset nemotron_science --split RQA
```

### nemotron_swe

NVIDIA's software engineering SFT dataset v2 with agentless problem-solving trajectories based on GitHub issues.

```bash
python script.py --dataset nemotron_swe                                    # agentless (default)
```

### longalign

~10K long-context instruction-following samples ranging from 8k to 64k tokens. Useful for training speculators on long-form generation patterns where token prediction behavior may differ from shorter contexts.

### open_perfectblend

Open-source reproduction of the instruction dataset from "The Perfect Blend: Redefining RLHF with Mixture of Judges." ~1.42M samples blending 8 source datasets covering math, code, chat, and instruction-following. Uses ShareGPT-style conversations with `from`/`value` fields and `human`/`gpt` roles.

| Source | Samples |
|--------|---------|
| meta-math/MetaMathQA | 395K |
| openbmb/UltraInteract_sft | 289K |
| HuggingFaceH4/ultrachat_200k | 208K |
| microsoft/orca-math-word-problems-200k | 200K |
| HuggingFaceH4/ultrafeedback_binarized | 187K |
| theblackcat102/evol-codealpaca-v1 | 111K |
| Post-training-Data-Flywheel/AutoIF-instruct-61k | 61K |
| mlabonne/lmsys-arena-human-preference-55k-sharegpt | 57K |

The individual source datasets are also available separately (see below): `metamathqa`, `ultrainteract`, `ultrachat`, `orca_math`, `ultrafeedback`, `evol_codealpaca`, `autoif`, `lmsys_arena`.

### metamathqa

395K augmented math questions with detailed solutions. Prompts are plain-string queries covering arithmetic, algebra, geometry, and word problems.

```bash
python script.py --dataset metamathqa
```

### ultrainteract

289K instruction-response pairs from UltraInteract spanning math, code, and logic tasks. Prompts are plain-string instructions.

```bash
python script.py --dataset ultrainteract
```

### orca_math

200K math word problems with step-by-step solutions from Microsoft's Orca-Math dataset.

```bash
python script.py --dataset orca_math
```

### ultrafeedback

187K preference-ranked prompts from UltraFeedback. Uses the `train_sft` split with plain-string prompts.

```bash
python script.py --dataset ultrafeedback
```

### evol_codealpaca

111K evolved code instruction-output pairs. Code-focused prompts progressively evolved for complexity.

```bash
python script.py --dataset evol_codealpaca
```

### autoif

61K auto-generated instruction-following examples. Uses standard messages format with role/content pairs.

```bash
python script.py --dataset autoif
```

### lmsys_arena

57K human-preference conversations from LMSYS Chatbot Arena in ShareGPT format with `from`/`value` fields.

```bash
python script.py --dataset lmsys_arena
```

### openhermes

1M instruction and chat examples covering diverse tasks including coding, roleplay, and general knowledge. Uses ShareGPT-style conversations with `from`/`value` fields.

```bash
python script.py --dataset openhermes
```

### openr1_math

High-quality verified R1-style math reasoning traces. Available in three subsets of increasing size. Select via `--subset`.

| Subset | Samples |
|--------|---------|
| `default` | 94K |
| `extended` | 131K |
| `all` | 225K |

```bash
python script.py --dataset openr1_math                                     # default subset (94K)
python script.py --dataset openr1_math --subset extended
python script.py --dataset openr1_math --subset all
```

### numinamath

72K math problems with tool-integrated reasoning (TIR) solutions from the AI-MO competition pipeline. Uses standard messages format.

```bash
python script.py --dataset numinamath
```

### codeforces_cots

Competitive coding problems with R1-style chain-of-thought reasoning traces. Defaults to the `solutions` subset. Multiple subsets available via `--subset`.

| Subset | Description |
|--------|-------------|
| `solutions` | Full solution set |
| `solutions_py` | Python solutions only |
| `solutions_cpp` | C++ solutions only |
| `solutions_decontaminated` | Decontaminated solutions |
| `solutions_w_editorials` | Solutions with editorials |

```bash
python script.py --dataset codeforces_cots                                 # solutions (default)
python script.py --dataset codeforces_cots --subset solutions_py
python script.py --dataset codeforces_cots --subset solutions_decontaminated
```

### codeforces

10K+ Codeforces problems through 2025. Problem descriptions are used as prompts. Available subsets via `--subset`.

| Subset | Description |
|--------|-------------|
| `default` | Full problem set with metadata |
| `verifiable` | Problems with verification tests |
| `verifiable-prompts` | Pre-formatted prompts |

```bash
python script.py --dataset codeforces                                      # default subset
python script.py --dataset codeforces --subset verifiable
```

### taco

25K competitive programming tasks with test cases from multiple online judges. Prompts are plain-string problem descriptions.

```bash
python script.py --dataset taco
```

### xlam_function_calling

60K verified function-calling examples. Each row contains a plain-string `query` and JSON-encoded tool definitions and expected answers.

```bash
python script.py --dataset xlam_function_calling
```

### apigen_mt

5K multi-turn tool-use and agent trajectories. Uses ShareGPT-style conversations with `from`/`value` fields, including `function_call` and `observation` roles.

```bash
python script.py --dataset apigen_mt
```

### swe_rebench

67K realistic SWE agent traces from OpenHands on SWE-bench tasks. Each trajectory contains system/user/assistant/tool turns following the OpenAI function-calling format.

```bash
python script.py --dataset swe_rebench
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

For ShareGPT-style datasets that use `from`/`value` fields with `human`/`gpt` roles instead of `user`/`assistant`, set `messages_user_value` to match the user role label:

```python
"my_sharegpt_dataset": {
    "id": "org/dataset-name",
    "prompt_field": "conversations",
    "default_split": "train",
    "messages_role_field": "from",
    "messages_content_field": "value",
    "messages_user_value": "human",
},
```
