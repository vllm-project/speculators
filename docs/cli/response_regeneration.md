# response_regeneration

Regenerates assistant responses in existing datasets using a vLLM-served model. Given a dataset containing user prompts (e.g., Magpie, UltraChat), this pipeline extracts the prompts, sends them to a vLLM server, and produces a new dataset with the original prompts paired with freshly generated responses from the target model. This is useful for creating training data where you want a specific model's outputs in place of the original assistant responses.

The pipeline consists of two scripts:

| Script       | Purpose                                                        |
| ------------ | -------------------------------------------------------------- |
| `run_all.sh` | End-to-end pipeline: starts vLLM, regenerates responses, stops |
| `script.py`  | Standalone response regeneration against a running vLLM server |

## run_all.sh

Orchestrates the entire pipeline: starts a vLLM server (with optional data/tensor parallelism), regenerates responses for the dataset, and stops the server. Uses vLLM's built-in data parallelism (`--data-parallel-size`) for multi-GPU scaling with automatic load balancing.

### Basic Usage

```bash
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dataset magpie
```

### Arguments

- **`--model`** (str, required) Model to serve and use for generation.

- **`--gpus`** (str, default: all visible) Comma-separated GPU IDs (sets `CUDA_VISIBLE_DEVICES`).

- **`--port`** (int, default: `8000`) Server port.

- **`--dp-size`** (int) Number of data parallel replicas (maps to vLLM's `--data-parallel-size`).

- **`--tp-size`** (int) Tensor parallel size per replica (maps to vLLM's `--tensor-parallel-size`).

- **`--keep-server`** (flag) Don't stop the vLLM server after processing completes.

All other arguments are passed through to `script.py`.

### Full Example

```bash
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dp-size 4 --tp-size 2 \
  --dataset magpie \
  --limit 1000 \
  --concurrency 128 \
  --max-tokens 4096
```

## script.py

Extracts user prompts from a dataset, sends them to a vLLM chat completion endpoint, and writes out new prompt-response pairs with the target model's generated responses.

### Features

- **Auto-detects model** from vLLM server (no need to specify `--model`)
- **Resume capability** to skip already-processed rows
- **Async processing** with configurable concurrency

### Basic Usage

```bash
python scripts/response_regeneration/script.py --dataset magpie
```

### Arguments

#### Data Arguments

- **`--dataset`** (str, default: `ultrachat`) Dataset to process. See [Supported Datasets](#supported-datasets) for the full list.

- **`--split`** (str, default: dataset-specific) Dataset split. Defaults to `train` for magpie and `train_sft` for ultrachat.

- **`--limit`** (int, default: `None`) Stop after N rows.

- **`--language-filter`** (str, default: `None`) Only process rows where language matches this value (e.g., `EN`).

#### Server Arguments

- **`--endpoint`** (str, default: `http://127.0.0.1:8000/v1/chat/completions`) vLLM chat completions endpoint.

- **`--model`** (str, default: `None`) Model name exposed by vLLM. Auto-detected from the server if not specified.

#### Generation Arguments

- **`--concurrency`** (int, default: `64`) Max concurrent requests to the vLLM server.

- **`--max-tokens`** (int, default: `8192`) Max tokens for generation.

#### Output Arguments

- **`--outfile`** (str, default: auto-generated) Output JSONL path. If not specified, auto-generated as `{dataset}_{model}.jsonl`.

- **`--resume`** (flag) Skip rows already present in the output file (matched by uuid or index).

### Full Example

```bash
python scripts/response_regeneration/script.py \
  --dataset magpie \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --limit 1000 \
  --concurrency 128 \
  --max-tokens 4096 \
  --outfile magpie_Llama-3.3-70B-Instruct.jsonl \
  --resume
```

## Supported Datasets

See the [response regeneration README](../../scripts/response_regeneration/README.md) for full details, split/subset tables, and dataset relationship diagrams.

### General Chat & Instruction Following

| Dataset | HuggingFace ID | Default Split | Samples | Prompt Format |
| ------- | -------------- | ------------- | ------- | ------------- |
| `magpie` | `Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered` | `train` | 300K | string |
| `ultrachat` | `HuggingFaceH4/ultrachat_200k` | `train_sft` | 208K | string |
| `tulu3` | `allenai/tulu-3-sft-mixture` | `train` | 939K | messages |
| `wildchat` | `allenai/WildChat` | `train` | 529K | messages |
| `openhermes` | `teknium/OpenHermes-2.5` | `train` | 1M | messages |
| `ultrafeedback` | `HuggingFaceH4/ultrafeedback_binarized` | `train_sft` | 187K | string |
| `autoif` | `Post-training-Data-Flywheel/AutoIF-instruct-61k` | `train` | 61K | messages |
| `lmsys_arena` | `mlabonne/lmsys-arena-human-preference-55k-sharegpt` | `train` | 57K | messages |
| `longalign` | `zai-org/LongAlign-10k` | `train` | 9.9K | messages |
| `nemotron_ifchat` | `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` | `reasoning_off` | ~2M | messages |
| `nemotron_ifchat_v1` | `nvidia/Nemotron-Instruction-Following-Chat-v1` | `structured_outputs` | ~5K | messages |

### Math & Science

| Dataset | HuggingFace ID | Default Split | Samples | Prompt Format |
| ------- | -------------- | ------------- | ------- | ------------- |
| `gsm8k` | `openai/gsm8k` | `train` | 7.5K | string |
| `metamathqa` | `meta-math/MetaMathQA` | `train` | 395K | string |
| `orca_math` | `microsoft/orca-math-word-problems-200k` | `train` | 200K | string |
| `openr1_math` | `open-r1/OpenR1-Math-220k` | `train` | 94K–225K | messages |
| `numinamath` | `AI-MO/NuminaMath-TIR` | `train` | 72K | messages |
| `nemotron_math` | `nvidia/Nemotron-Math-v2` | `high_part00` | ~696K | messages |
| `nemotron_science` | `nvidia/Nemotron-Science-v1` | `MCQ` | ~174K | messages |

### Coding & Competitive Programming

| Dataset | HuggingFace ID | Default Split | Samples | Prompt Format |
| ------- | -------------- | ------------- | ------- | ------------- |
| `code_alpaca` | `HuggingFaceH4/CodeAlpaca_20K` | `train` | 18K | string |
| `evol_codealpaca` | `theblackcat102/evol-codealpaca-v1` | `train` | 111K | string |
| `codeforces` | `open-r1/codeforces` | `train` | 10K | string |
| `codeforces_cots` | `open-r1/codeforces-cots` | `train` | 48K | string |
| `taco` | `BAAI/TACO` | `train` | 25K | string |
| `nemotron_competitive_v2` | `nvidia/Nemotron-SFT-Competitive-Programming-v2` | `competitive_coding_python` | ~337K | messages |
| `nemotron_competitive_v1` | `nvidia/Nemotron-Competitive-Programming-v1` | `infinibyte_part00` | ~587K | messages |

### Agentic, Tool Use & SWE

| Dataset | HuggingFace ID | Default Split | Samples | Prompt Format |
| ------- | -------------- | ------------- | ------- | ------------- |
| `nemotron_agentic` | `nvidia/Nemotron-SFT-Agentic-v2` | `interactive_agent` | ~279K | messages |
| `xlam_function_calling` | `Salesforce/xlam-function-calling-60k` | `train` | 60K | string |
| `apigen_mt` | `Salesforce/APIGen-MT-5k` | `train` | 5K | messages |
| `nemotron_swe` | `nvidia/Nemotron-SFT-SWE-v2` | `agentless` | ~210K | messages |
| `swe_rebench` | `nebius/SWE-rebench-openhands-trajectories` | `train` | 67K | messages |

### Multi-Domain Blends

| Dataset | HuggingFace ID | Default Split | Samples | Prompt Format |
| ------- | -------------- | ------------- | ------- | ------------- |
| `nemotron` | `nvidia/Nemotron-Post-Training-Dataset-v2` | `chat` | 1.4M | messages |
| `nemotron_cascade` | `nvidia/Nemotron-Cascade-2-SFT-Data` | `train` | millions | messages |
| `open_perfectblend` | `mlabonne/open-perfectblend` | `train` | 1.42M | messages |
| `ultrainteract` | `openbmb/UltraInteract_sft` | `train` | 289K | string |

## Output Format

Each line of the output JSONL file pairs the original user prompt with the newly generated response, in a conversations format compatible with fine-tuning:

```json
{
  "id": "sample_0",
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "gpt", "value": "The capital of France is Paris."}
  ],
  "metadata": {
    "idx": 0,
    "finish_reason": "stop",
    "latency_s": 1.234,
    "usage": {...},
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
    "reasoning_content": "..."
  }
}
```

- The `id` field uses the dataset's UUID if available, otherwise falls back to `sample_{idx}`.
- The `reasoning_content` field in metadata is only included when the model provides reasoning content (e.g., with reasoning models).

On error, the response conversation turn is omitted and an `error` field is included in metadata:

```json
{
  "id": "sample_0",
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"}
  ],
  "metadata": {
    "idx": 0,
    "error": "ConnectionError(...)",
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions"
  }
}
```

If `--outfile` is not specified, the filename is auto-generated based on dataset and model (e.g., `magpie_Llama-3.3-70B-Instruct.jsonl`).
