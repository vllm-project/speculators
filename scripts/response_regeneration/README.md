# Response Regeneration

Takes prompts from existing datasets and regenerates responses through a vLLM-served model. The regenerated responses are used to train speculator models for speculative decoding.

## Usage

Start a vLLM server, then run the script against it:

```bash
python script.py --dataset ultrachat --limit 1000
```

The model name is auto-detected from the vLLM server. Output is written to a JSONL file (auto-named based on dataset and model, or set with `--outfile`).

## Supported Datasets

### General Chat & Instruction Following

| Name | Dataset | Default Split | Samples | Size | Prompt Format |
|------|---------|---------------|---------|------|---------------|
| `magpie` | [Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered) | `train` | 300K | 2.1 GB | string |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `train_sft` | 208K | 1.4 GB | string |
| `tulu3` | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `train` | 939K | 1.5 GB | messages |
| `wildchat` | [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) | `train` | 529K | 3.1 GB | messages |
| `openhermes` | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | `train` | 1M | 795 MB | messages |
| `ultrafeedback` | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | `train_sft` | 187K | 650 MB | string |
| `autoif` | [Post-training-Data-Flywheel/AutoIF-instruct-61k](https://huggingface.co/datasets/Post-training-Data-Flywheel/AutoIF-instruct-61k) | `train` | 61K | 63 MB | messages |
| `lmsys_arena` | [mlabonne/lmsys-arena-human-preference-55k-sharegpt](https://huggingface.co/datasets/mlabonne/lmsys-arena-human-preference-55k-sharegpt) | `train` | 57K | 57 MB | messages |
| `longalign` | [zai-org/LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k) | `train` | 9.9K | 0.6 GB | messages |
| `nemotron_ifchat` | [nvidia/Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | `reasoning_off` | ~2M | ~16 GB | messages |
| `nemotron_ifchat_v1` | [nvidia/Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | `structured_outputs` | ~5K | ~38 MB | messages |

### Math & Science

| Name | Dataset | Default Split | Samples | Size | Prompt Format |
|------|---------|---------------|---------|------|---------------|
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `train` | 7.5K | 4 MB | string |
| `metamathqa` | [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | `train` | 395K | 188 MB | string |
| `orca_math` | [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | `train` | 200K | 84 MB | string |
| `openr1_math` | [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | `train` | 94K–225K | 4.2 GB | messages |
| `numinamath` | [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) | `train` | 72K | 148 MB | messages |
| `nemotron_math` | [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | `high_part00` | ~696K | ~13 GB | messages |
| `nemotron_science` | [nvidia/Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | `MCQ` | ~174K | ~742 MB | messages |

### Coding & Competitive Programming

| Name | Dataset | Default Split | Samples | Size | Prompt Format |
|------|---------|---------------|---------|------|---------------|
| `code_alpaca` | [HuggingFaceH4/CodeAlpaca_20K](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K) | `train` | 18K | 5 MB | string |
| `evol_codealpaca` | [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | `train` | 111K | 137 MB | string |
| `codeforces` | [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) | `train` | 10K | 2.8 GB | string |
| `codeforces_cots` | [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) | `train` | 48K | 1.9 GB | string |
| `taco` | [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO) | `train` | 25K | 2.4 GB | string |
| `nemotron_competitive_v2` | [nvidia/Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) | `competitive_coding_python` | ~337K | ~44 GB | messages |
| `nemotron_competitive_v1` | [nvidia/Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | `infinibyte_part00` | ~587K | ~23 GB | messages |

### Agentic, Tool Use & SWE

| Name | Dataset | Default Split | Samples | Size | Prompt Format |
|------|---------|---------------|---------|------|---------------|
| `nemotron_agentic` | [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | `interactive_agent` | ~279K | ~6 GB | messages |
| `xlam_function_calling` | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | `train` | 60K | 96 MB | string |
| `apigen_mt` | [Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | `train` | 5K | 10 MB | messages |
| `nemotron_swe` | [nvidia/Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) | `agentless` | ~210K | ~6 GB | messages |
| `swe_rebench` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | `train` | 67K | 2.1 GB | messages |

### Multi-Domain Blends

| Name | Dataset | Default Split | Samples | Size | Prompt Format |
|------|---------|---------------|---------|------|---------------|
| `nemotron` | [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | `chat` | 1.4M | 6.1 GB | messages |
| `nemotron_cascade` | [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | `train` | millions | ~593 GB | messages |
| `open_perfectblend` | [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) | `train` | 1.42M | ~4 GB | messages |
| `ultrainteract` | [openbmb/UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft) | `train` | 289K | 171 MB | string |

## Dataset Relationships

### NVIDIA Nemotron Family

NVIDIA publishes two large multi-domain datasets (`nemotron`, `nemotron_cascade`) alongside domain-specific component datasets. The component datasets are independent HuggingFace repos — they are not subsets/configs of the multi-domain ones, but cover the same training pipeline.

```
Multi-domain bundles
├── nemotron          (Nemotron-Post-Training-Dataset-v2)  — chat, stem, math, code splits
└── nemotron_cascade  (Nemotron-Cascade-2-SFT-Data)        — 8 domain subsets

Domain-specific components
├── Chat & Instruction Following
│   ├── nemotron_ifchat     (Nemotron-SFT-Instruction-Following-Chat-v2)
│   └── nemotron_ifchat_v1  (Nemotron-Instruction-Following-Chat-v1)
├── Math & Science
│   ├── nemotron_math       (Nemotron-Math-v2)
│   └── nemotron_science    (Nemotron-Science-v1)
├── Coding
│   ├── nemotron_competitive_v2  (Nemotron-SFT-Competitive-Programming-v2)
│   └── nemotron_competitive_v1  (Nemotron-Competitive-Programming-v1)
└── Agentic & SWE
    ├── nemotron_agentic    (Nemotron-SFT-Agentic-v2)
    └── nemotron_swe        (Nemotron-SFT-SWE-v2)
```

### open_perfectblend Sources

`open_perfectblend` blends 8 source datasets. Each source is also available as a standalone config for targeted regeneration:

| Source | Standalone Config | Category |
|--------|-------------------|----------|
| meta-math/MetaMathQA | `metamathqa` | Math |
| openbmb/UltraInteract_sft | `ultrainteract` | Multi-domain |
| HuggingFaceH4/ultrachat_200k | `ultrachat` | Chat |
| microsoft/orca-math-word-problems-200k | `orca_math` | Math |
| HuggingFaceH4/ultrafeedback_binarized | `ultrafeedback` | Chat |
| theblackcat102/evol-codealpaca-v1 | `evol_codealpaca` | Coding |
| Post-training-Data-Flywheel/AutoIF-instruct-61k | `autoif` | Instruction following |
| mlabonne/lmsys-arena-human-preference-55k-sharegpt | `lmsys_arena` | Chat |

## Dataset Details

### General Chat & Instruction Following

#### magpie

300K filtered synthetic instructions generated from Llama-3.1 using the Magpie method. Prompts are single-turn instructions stored in a plain string field. Good general-purpose coverage across tasks.

#### ultrachat

200K multi-turn dialogues covering a broad range of topics. The `train_sft` split contains the SFT-ready subset. Prompts are plain strings representing the user's opening message. Also a source in [open_perfectblend](#open_perfectblend-sources).

#### tulu3

Allen AI's SFT mixture used to train Tulu 3, containing ~939K examples spanning diverse tasks and sources. Uses a conversational messages format. Broad coverage makes it a good default for general-purpose regeneration.

#### wildchat

~529K real user conversations collected from ChatGPT and GPT-4. Captures natural user interaction patterns and phrasing, making it useful for training models on realistic chat distributions.

#### openhermes

1M instruction and chat examples covering diverse tasks including coding, roleplay, and general knowledge. Uses ShareGPT-style conversations with `from`/`value` fields.

```bash
python script.py --dataset openhermes
```

#### ultrafeedback

187K preference-ranked prompts from UltraFeedback. Uses the `train_sft` split with plain-string prompts. Also a source in [open_perfectblend](#open_perfectblend-sources).

```bash
python script.py --dataset ultrafeedback
```

#### autoif

61K auto-generated instruction-following examples. Uses standard messages format with role/content pairs. Also a source in [open_perfectblend](#open_perfectblend-sources).

```bash
python script.py --dataset autoif
```

#### lmsys_arena

57K human-preference conversations from LMSYS Chatbot Arena in ShareGPT format with `from`/`value` fields. Also a source in [open_perfectblend](#open_perfectblend-sources).

```bash
python script.py --dataset lmsys_arena
```

#### longalign

~10K long-context instruction-following samples ranging from 8k to 64k tokens. Useful for training speculators on long-form generation patterns where token prediction behavior may differ from shorter contexts.

#### nemotron_ifchat

NVIDIA's instruction-following and chat SFT dataset with synthetic dialogues generated from multiple frontier models (Kimi-K2, GLM-4, Qwen3, etc.). Available in two splits for standard and chain-of-thought style responses. ~2M total samples across both splits. Part of the [Nemotron family](#nvidia-nemotron-family).

| Split | Size |
|-------|------|
| `reasoning_off` | 6.0 GB |
| `reasoning_on` | 10.1 GB |

```bash
python script.py --dataset nemotron_ifchat                        # non-reasoning
python script.py --dataset nemotron_ifchat --split reasoning_on   # reasoning
```

#### nemotron_ifchat_v1

NVIDIA's instruction-following chat v1 dataset. The `structured_outputs` split contains prompts requiring structured output formats. Part of the [Nemotron family](#nvidia-nemotron-family).

```bash
python script.py --dataset nemotron_ifchat_v1                              # structured_outputs (default)
```

### Math & Science

#### gsm8k

Grade-school math word problems with step-by-step solutions. Uses the `main` subset. Useful for training and evaluating mathematical reasoning capabilities.

#### metamathqa

395K augmented math questions with detailed solutions. Prompts are plain-string queries covering arithmetic, algebra, geometry, and word problems. Also a source in [open_perfectblend](#open_perfectblend-sources).

```bash
python script.py --dataset metamathqa
```

#### orca_math

200K math word problems with step-by-step solutions from Microsoft's Orca-Math dataset. Also a source in [open_perfectblend](#open_perfectblend-sources).

```bash
python script.py --dataset orca_math
```

#### openr1_math

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

#### numinamath

72K math problems with tool-integrated reasoning (TIR) solutions from the AI-MO competition pipeline. Uses standard messages format.

```bash
python script.py --dataset numinamath
```

#### nemotron_math

NVIDIA's math SFT dataset v2 with problems at three difficulty levels. The high-difficulty split is further divided into three parts. Part of the [Nemotron family](#nvidia-nemotron-family).

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

#### nemotron_science

NVIDIA's science dataset v1 with multiple-choice questions and reasoning Q&A. Part of the [Nemotron family](#nvidia-nemotron-family).

| Split | Description |
|-------|-------------|
| `MCQ` | Multiple-choice questions |
| `RQA` | Reasoning question answering |

```bash
python script.py --dataset nemotron_science                                # MCQ (default)
python script.py --dataset nemotron_science --split RQA
```

### Coding & Competitive Programming

#### code_alpaca

20K code generation prompts based on the Stanford Alpaca format. Each row contains a plain string prompt describing a coding task. Compact and code-focused, making it useful for training on programming-specific distributions.

#### evol_codealpaca

111K evolved code instruction-output pairs. Code-focused prompts progressively evolved for complexity. Also a source in [open_perfectblend](#open_perfectblend-sources).

```bash
python script.py --dataset evol_codealpaca
```

#### codeforces

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

#### codeforces_cots

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

#### taco

25K competitive programming tasks with test cases from multiple online judges. Prompts are plain-string problem descriptions.

```bash
python script.py --dataset taco
```

#### nemotron_competitive_v2

NVIDIA's competitive programming SFT dataset v2 with Python and C++ solutions. Select language via `--split`. Part of the [Nemotron family](#nvidia-nemotron-family).

| Split | Description |
|-------|-------------|
| `competitive_coding_python` | Python solutions |
| `competitive_coding_cpp` | C++ solutions |

```bash
python script.py --dataset nemotron_competitive_v2                         # Python (default)
python script.py --dataset nemotron_competitive_v2 --split competitive_coding_cpp
```

#### nemotron_competitive_v1

NVIDIA's competitive programming dataset v1 with Infinibyte-generated data, split into two parts. Part of the [Nemotron family](#nvidia-nemotron-family).

| Split | Description |
|-------|-------------|
| `infinibyte_part00` | Part 0 |
| `infinibyte_part01` | Part 1 |

```bash
python script.py --dataset nemotron_competitive_v1                         # part00 (default)
python script.py --dataset nemotron_competitive_v1 --split infinibyte_part01
```

### Agentic, Tool Use & SWE

#### nemotron_agentic

NVIDIA's agentic SFT dataset v2 covering interactive agents, tool calling, and search tasks. Select domain via `--split`. Part of the [Nemotron family](#nvidia-nemotron-family).

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

#### xlam_function_calling

60K verified function-calling examples. Each row contains a plain-string `query` and JSON-encoded tool definitions and expected answers.

```bash
python script.py --dataset xlam_function_calling
```

#### apigen_mt

5K multi-turn tool-use and agent trajectories. Uses ShareGPT-style conversations with `from`/`value` fields, including `function_call` and `observation` roles.

```bash
python script.py --dataset apigen_mt
```

#### nemotron_swe

NVIDIA's software engineering SFT dataset v2 with agentless problem-solving trajectories based on GitHub issues. Part of the [Nemotron family](#nvidia-nemotron-family).

```bash
python script.py --dataset nemotron_swe                                    # agentless (default)
```

#### swe_rebench

67K realistic SWE agent traces from OpenHands on SWE-bench tasks. Each trajectory contains system/user/assistant/tool turns following the OpenAI function-calling format.

```bash
python script.py --dataset swe_rebench
```

### Multi-Domain Blends

#### nemotron

NVIDIA's large-scale post-training dataset covering multiple domains. Uses a conversational messages format; the first user message is extracted as the prompt. Part of the [Nemotron family](#nvidia-nemotron-family).

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

#### nemotron_cascade

NVIDIA's Cascade 2 SFT dataset spanning 8 domains with data generated from multiple frontier models. Each domain is a separate subset. All subsets use a single `train` split. Note: the HuggingFace metadata underreports sizes — actual sizes are much larger than shown in the dataset viewer. Part of the [Nemotron family](#nvidia-nemotron-family).

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

#### open_perfectblend

Open-source reproduction of the instruction dataset from "The Perfect Blend: Redefining RLHF with Mixture of Judges." ~1.42M samples blending 8 source datasets covering math, code, chat, and instruction-following. Uses ShareGPT-style conversations with `from`/`value` fields and `human`/`gpt` roles. See [open_perfectblend Sources](#open_perfectblend-sources) for the individual source datasets.

#### ultrainteract

289K instruction-response pairs from UltraInteract spanning math, code, and logic tasks. Prompts are plain-string instructions. Also a source in [open_perfectblend](#open_perfectblend-sources).

```bash
python script.py --dataset ultrainteract
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
