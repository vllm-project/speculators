# Evaluating Model Performance

The evaluator has two modes:

- `acceptance`: an acceptance run over the selected datasets that reports draft acceptance rate, acceptance length, and per-position acceptance rates. No baseline is needed.
- `speedup`: closed-loop concurrency sweeps with warmup and repeats. Collect a baseline and a speculative configuration separately, then compare them using `throughput_interactivity.py plot`.

## Setup

Run commands from the repository root. With an editable source-checkout installation, `speculators evaluate` forwards to the same evaluator without duplicating its logic. For example, `speculators evaluate acceptance --help` and `speculators evaluate speedup --help` show the mode arguments. The script form below also works. Evaluation scripts are not bundled in standalone wheels. Install client requirements in your environment:

```bash
uv pip install -r scripts/evaluate/requirements.txt
```

Acceptance uses the experimental vLLM 0.30+ `metrics.speculative_decoding` schema with `--per-request-spec-decode-metrics summary`; pin the server version/commit. Enable `--enable-per-request-metrics` if server timing metrics are also wanted. Speedup uses GuideLLM 0.8.0 for native timed concurrency runs. Use `--guidellm-bin /path/to/guidellm` if the client is installed in a separate environment. Set `--tokenizer` to the verifier tokenizer or local snapshot when the served alias is not a tokenizer repository. For speedup sweeps, disable prefix caching (`--no-enable-prefix-caching` on current vLLM) because the workload cycles through the same prompts.

## Dataset inputs

Both modes take one or more datasets through `--dataset`. They share the same loader and column mappings. MRCR is supported only in acceptance mode for now.

`--dataset` accepts preset names, Hugging Face dataset IDs, or local `.jsonl`, `.json`, `.csv`, or `.parquet` files. Use `--dataset-config`, `--split`, `--data-files`, and `--revision` to select HF configurations, splits, files, and revisions. Loading flags apply globally to every dataset in the invocation. `--max-samples` defaults to 128 per dataset; acceptance sends each selected sample once. Speedup replays a prompt pool until each point reaches its time limit. It does not require a full dataset pass within that time.

### Built-in datasets and blends

`--dataset basic` includes GSM8K, MATH-500, HumanEval, MBPP, AIME 2026, MT-Bench and Arena-Hard; MRCR is separate. Presets supply their source configuration, split, and column mapping, using the current default dataset revision. `--max-samples` applies separately to each dataset; smaller datasets keep all their rows.

| Preset       | Source                                               |
| ------------ | ---------------------------------------------------- |
| `gsm8k`      | `openai/gsm8k`, main/test                            |
| `math500`    | `HuggingFaceH4/MATH-500`, test                       |
| `humaneval`  | `openai/openai_humaneval`, test                      |
| `mbpp`       | `google-research-datasets/mbpp`, full/test           |
| `aime26`     | `math-ai/aime26`, test                               |
| `mtbench`    | `HuggingFaceH4/mt_bench_prompts`, train              |
| `arena-hard` | `lmarena-ai/arena-hard-auto`, v2.0 (750 prompts)     |
| `mrcr`       | `openai/mrcr`, all needle counts                     |
| `speedbench` | NVIDIA data prepared and cached locally on first use |

## Acceptance

```bash
speculators evaluate acceptance \
  --dataset speedbench --dataset-config qualitative \
  --target http://localhost:9123 --model my-model \
  --output-dir results/acceptance
```

```bash
# MRCR: 8 needles.
speculators evaluate acceptance \
  --dataset openai/mrcr --data-files '8needle/*.parquet' \
  --messages-column prompt \
  --target http://localhost:9123 --model my-model \
  --output-dir results/mrcr_8needle
```

MRCR preserves complete histories and omits client output-token caps: `--max-tokens` and body `max_tokens`/`max_completion_tokens` are rejected. Other datasets keep the 8192-token default. Prompt-length binning uses response `prompt_tokens` afterward, without pretokenization or filtering.

Acceptance has a 900-second timeout per request. Three consecutive errors in completion order stop further scheduling across datasets; in-flight requests finish and are saved. Success resets the counter. Errors or missing metrics return exit 1.

## Speedup

```bash
# Baseline server.
speculators evaluate speedup \
  --dataset humaneval --target http://localhost:9124 --model my-model \
  --tokenizer my-verifier-model --label baseline \
  --output-dir results/humaneval_baseline

# Speculative server.
speculators evaluate speedup \
  --dataset humaneval --target http://localhost:9123 --model my-model \
  --tokenizer my-verifier-model --label dspark \
  --output-dir results/humaneval_dspark
```

Speedup stops at the first failed point by default. Use `--keep-going` to continue the remaining points and return exit 1 afterward if any point failed. Each point checks `/metrics` before and after GuideLLM; unavailable metrics or enabled prefix caching fail the point. Individual `concN_rR.json` reports stay on disk, including failed reports. Parsing warns and skips malformed JSON or invalid benchmark data; valid repeats are averaged in `summary.csv` and plots. Use a fresh output directory for each run.

## Graph

```bash
python scripts/evaluate/throughput_interactivity.py plot \
  --series results/humaneval_baseline:Baseline \
  --series results/humaneval_dspark:DSpark \
  --out results/humaneval_speedup.png
```
