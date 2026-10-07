# throughput_interactivity.py

Measures output-token throughput against per-user interactivity for a vLLM (or any OpenAI-compatible) server, in the way SemiAnalysis's InferenceX does, and draws the result. One file, four subcommands. See the [tutorial](../user_guide/tutorials/throughput_interactivity.md) for what the chart means and a worked example.

Requirements: Python 3.10+, `guidellm>=0.7.1` (tested with 0.8.0) for `collect`, and `pillow` for `plot`. `parse` and `validate` use the standard library only.

## Basic Usage

```bash
# run a closed-loop sweep against a server, then parse and validate
# (defaults: 3 repeats, 30 s warmup + 100 s window per point, about 52 minutes here)
python scripts/evaluate/throughput_interactivity.py collect \
  --target http://localhost:8000 --model Qwen/Qwen3.8-27B \
  --dataset RedHatAI/speculator_benchmarks --subset HumanEval --max-tokens 1024 \
  --streams 1,2,4,8,16,32,64,128 --out-dir runs/dspark_humaneval --label dspark

# rebuild the CSV from raw results, check it, draw it
python scripts/evaluate/throughput_interactivity.py parse runs/dspark_humaneval --label dspark --csv dspark.csv
python scripts/evaluate/throughput_interactivity.py validate dspark.csv
python scripts/evaluate/throughput_interactivity.py plot --series baseline.csv:'no speculator' --series dspark.csv:'dspark' --out chart.png
```

## `collect`

Runs one GuideLLM benchmark per point and repeat, snapshots the server's `/metrics` around each run for speculative-decoding acceptance, then parses and validates the directory.

### Server

- **`--target`** (str, required) Server base URL, e.g. `http://localhost:8000`.
- **`--model`** (str, required) Model name as served, sent in the request's `model` field.
- **`--tokenizer`** (str, default: `--model`) Tokenizer path or Hugging Face id GuideLLM uses to count tokens.
- **`--request-format`** (str) `/v1/completions` (default) or `/v1/chat/completions` (default with `--dataset`).

### Load

Give `--streams`, `--rates`, or both.

- **`--streams`** (comma list of int) Closed loop, InferenceX style: requests kept in flight, e.g. `1,2,4,8,16,32,64,128`. Points are named `conc<N>`.
- **`--rates`** (comma list of float) Open loop: constant arrival rates in requests per second, e.g. `0.5,1,2,4`. Points are named `rate<r>`.
- **`--synchronous` / `--no-synchronous`** Also run a single-stream point (`sync`). On by default only when no `--streams` are given, since `--streams 1` is the same point.
- **`--repeats`** (int, default: `3`) Runs per point. The chart draws the mean; `validate` reports the spread.
- **`--max-seconds`** (float, default: `100`) Measurement window per point.
- **`--warmup-seconds`** (float, default: `30`) Warmup per point, excluded from every number.

### Data

Give one data source per run: a dataset subset, random text, or a raw GuideLLM spec. The script exits if flags from two sources are combined.

**A dataset subset** (real prompts, sent through `/v1/chat/completions`):

- **`--dataset`** (str) Hugging Face dataset id, e.g. `RedHatAI/speculator_benchmarks`, or a local directory or `.jsonl` file.
- **`--subset`** (str) File name without `.jsonl`, e.g. `HumanEval` or `math_reasoning`.
- **`--prompt-column`** (str, default: `prompt`) Column holding the prompt.
- **`--dataset-repeat`** (int, default: `0`) Repeat the subset this many times; `0` picks enough rows for 20 requests per stream and at least 5,000 requests. GuideLLM ends a run when its dataset runs out.
- **`--max-tokens`** (int) `max_tokens` per request. Set it with a dataset.
- **`--ignore-eos`** Force every output to `--max-tokens`.

**Random text** (lengths forced, prefix caching defeated; not meaningful for a speculator):

- **`--prompt-tokens`**, **`--output-tokens`** (int) Target prompt and output lengths.
- **`--range-ratio`** (float, default: `0.8`) Each request's lengths drawn uniformly from `ratio × L` to `L`. `1` gives fixed lengths, which make closed-loop streams move in lockstep (finish and restart together).

**A raw GuideLLM spec** (anything GuideLLM accepts, passed through unchanged):

- **`--data`** (str) The data spec, e.g. `kind=synthetic_text,prompt_tokens=1000,output_tokens=1000`.
- **`--data-column-mapper`** (str) Column mapper spec. Also usable with `--dataset`, where it defaults to `kind=generative_column_mapper,column_mappings.text_column=<prompt column>`.

### Output and control

- **`--out-dir`** (str, required) Directory for the raw results: `<point>_r<repeat>.json`, `<point>_r<repeat>.metrics.json`, `_data/` for materialized datasets, and `bench_command.txt`.
- **`--label`** (str, required) Series name written into the CSV's `model` column.
- **`--csv`** (str, default: `<out-dir>/<label>.csv`) Where to write the CSV.
- **`--metrics-url`** (str, default: `<target>/metrics`) Prometheus endpoint for acceptance counters.
- **`--no-metrics`** Do not record acceptance.
- **`--guidellm-bin`** (str, default: `guidellm`) GuideLLM executable.
- **`--guidellm-arg`** (str, repeatable) Extra argument appended to every GuideLLM command.
- **`--dry-run`** Print the GuideLLM commands and exit. Do this first.
- **`--overwrite`** Re-run points whose JSON exists.
- **`--keep-going`** Continue after a failed point.

## `parse`

```bash
python scripts/evaluate/throughput_interactivity.py parse <json_dir> --label <name> --csv <out.csv>
```

Reads every `*.json` under `json_dir`, one row per benchmark, named from `<point>_r<N>.json`, and renames synchronous and concurrent benchmarks `sync` and `conc<N>` whatever the file is called. Acceptance sidecars are merged in when present. Prints the validation report.

## `validate`

```bash
python scripts/evaluate/throughput_interactivity.py validate <csv>
```

Prints one row per point, averaged over repeats, then the warnings listed in the [tutorial](../user_guide/tutorials/throughput_interactivity.md#read-the-validation-report).

## `plot`

- **`--series`** (`CSV[:NAME[:#COLOR]]`, repeatable, required) One curve per CSV; the first is drawn on top.
- **`--out`** (str, required) PNG path.
- **`--x`** (`itl` | `little`, default: `itl`) X axis: `1000 / mean ITL` (InferenceX), or throughput divided by requests in flight.
- **`--title`**, **`--subtitle`**, **`--xlabel`**, **`--ylabel`**, **`--note`** Chart text. Put the hardware, versions and workload in the subtitle.
- **`--label-format`** (str, default: `auto`) Point labels: `auto` gives `single stream`, `<N> concurrent` or `<r> req/s`; otherwise a format string over `{rps}`, `{conc}` and `{streams}`.
- **`--label-first-only`** Label only the first series, for curves that overlap.
- **`--drop`** (repeatable) Leave out a point by name, e.g. `conc128`.
- **`--min-concurrency`**, **`--max-concurrency`** (float) Keep only points in this range.
- **`--force-legend`** Draw the legend for a single series.

## CSV columns

| Column                                                                                      | Meaning                                                                                                                                                       |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`, `point`, `offered_rps`, `offered_concurrency`, `repeat`, `source_json`, `strategy` | Identifiers. `point` is `sync`, `conc<N>` or `rate<r>`; `offered_concurrency` is N (1 for `sync`).                                                            |
| `measured_duration_s`                                                                       | Measurement window length.                                                                                                                                    |
| `successful_requests`, `errored_requests`, `incomplete_requests`                            | Sizes of GuideLLM's three request lists.                                                                                                                      |
| `started_requests`, `achieved_rps`                                                          | Requests started inside the window, and that count per second.                                                                                                |
| `completed_rps`                                                                             | Requests finished inside the window, per second.                                                                                                              |
| `aggregate_output_tps`                                                                      | Output tokens generated inside the window, per second. The y axis.                                                                                            |
| `mean_active_concurrency`                                                                   | In-flight time inside the window divided by the window.                                                                                                       |
| `interactivity_tps_user`                                                                    | `aggregate_output_tps / mean_active_concurrency` (Little's law; `plot --x little`).                                                                           |
| `mean_itl_ms`, `interactivity_itl_tps_user`                                                 | Mean inter-token latency of successful requests, and `1000 / mean_itl_ms`. The x axis.                                                                        |
| `mean_output_tokens`, `mean_prompt_tokens`, `mean_cached_tokens`                            | Means over successful requests; cached = prompt tokens served from the prefix cache (GuideLLM 0.8+).                                                          |
| `mean_output_tokens_per_iteration`                                                          | Output tokens per streamed chunk: about 1 without speculative decoding, the accepted length plus one with it.                                                 |
| `median_ttft_ms`, `p99_ttft_ms`, `median_itl_ms`                                            | Over successful requests.                                                                                                                                     |
| `arrival_burstiness`                                                                        | Coefficient of variation of request starts per second inside the window; above 2 means lockstep waves.                                                        |
| `acceptance_length`, `num_drafts`, `num_accepted_tokens`                                    | From the server's speculative-decoding counters over the run: `1 + accepted / drafts`, as in `evaluate.py`. Empty without a speculator or without `/metrics`. |
| `stop_reason`                                                                               | Why GuideLLM stopped, normally `max_duration`.                                                                                                                |

## Measurement rules

- Everything is counted inside the measurement window `[measure_start_time, measure_end_time]` of GuideLLM's scheduler metrics. Output tokens come from every request that overlaps the window, successful or incomplete, prorated over `[first token, last token]`. In-flight time is each request's lifetime clipped to the window, errored requests included.
- `achieved_rps` counts requests started inside the window, not GuideLLM's `requests_made`, which also counts requests that were queued but never sent.
- The synchronous point measures about 0.9995 concurrent, not 1.0; do not filter on `>= 1`.
- With a dataset the subset is repeated and shuffled before the run, never streamed once, and `mean_cached_tokens` shows how much of it the server served from its prefix cache.
