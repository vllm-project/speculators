# throughput_interactivity.py

Measures output-token throughput against per-user interactivity for a vLLM server, in the way SemiAnalysis's InferenceX does, and draws the result. One file, four subcommands. See the [tutorial](../user_guide/tutorials/throughput_interactivity.md) for what the chart means and a worked example.

Requirements: Python 3.10+, `guidellm>=0.8.0` for `collect`, and `matplotlib` for `plot`. `parse` and `validate` use the standard library only.

## Basic Usage

```bash
# run a closed-loop sweep against a server, then parse and validate
# (defaults: 3 repeats, 30 s warmup + 100 s window per point, about 52 minutes here)
python scripts/evaluate/throughput_interactivity.py collect \
  --target http://localhost:8000 --model Qwen/Qwen3.8-27B \
  --subset HumanEval --max-tokens 1024 \
  --streams 1,2,4,8,16,32,64,128 --out-dir runs/dspark_humaneval --label dspark

# rebuild the CSV from raw results, check it, draw it
python scripts/evaluate/throughput_interactivity.py parse runs/dspark_humaneval --label dspark --csv dspark.csv
python scripts/evaluate/throughput_interactivity.py validate dspark.csv
python scripts/evaluate/throughput_interactivity.py plot --series baseline.csv:'no speculator' --series dspark.csv:'dspark' --out chart.png
```

## `collect`

Runs one GuideLLM benchmark per point and repeat, samples the server's `/metrics` during each run for speculative-decoding acceptance and the scheduler's running and waiting counts, then parses and validates the directory. The server must have prefix caching off: `collect` reads the setting from `/metrics` and refuses to start while it is on.

### Server

- **`--target`** (str, required) vLLM server base URL, e.g. `http://localhost:8000`. `/metrics` must be served there too.
- **`--model`** (str, required) Model name as served, sent in the request's `model` field.
- **`--tokenizer`** (str, default: `--model`) Tokenizer path or Hugging Face id GuideLLM uses to count tokens.
- **`--extra-body`** (`KEY=JSON`, repeatable) Extra field sent in every request body, e.g. `temperature=0` for greedy decoding. Without it the server's sampling defaults apply; vLLM takes them from the model's `generation_config.json` (for `Qwen/Qwen3.8-27B`: temperature 1.0, top-p 0.95, top-k 20). Acceptance length depends on sampling, so state it with the result and keep it identical across configurations.

### Load

Closed loop, InferenceX style: N requests are kept in flight at all times, and each stream sends its next request the moment the previous one completes.

- **`--streams`** (comma list of int, required) The values of N, e.g. `1,2,4,8,16,32,64,128`. Points are named `conc<N>`.
- **`--repeats`** (int, default: `3`) Runs per point. The chart draws the mean; `validate` reports the spread.
- **`--max-seconds`** (float, default: `100`) Measurement window per point.
- **`--warmup-seconds`** (float, default: `30`) Warmup per point, excluded from every number. Give 0 or at least 1: GuideLLM reads a value below 1 as a fraction of the run, so the script refuses it.

### Data

Real prompts, sent through `/v1/chat/completions`. Random text is not offered: a drafter has nothing to predict in it.

- **`--dataset`** (str, default: `RedHatAI/speculator_benchmarks`) Hugging Face dataset id, or a local directory or `.jsonl` file.
- **`--subset`** (str, required) File name without `.jsonl`, e.g. `HumanEval` or `math_reasoning`.
- **`--prompt-column`** (str, default: `prompt`) Column holding the prompt.
- **`--dataset-repeat`** (int, default: `0`) Repeat the subset this many times; `0` picks enough rows for 20 requests per stream and at least 5,000 requests. GuideLLM ends a run when its dataset runs out.
- **`--max-tokens`** (int, required) `max_tokens` per request, e.g. `1024`.

### Output and control

- **`--out-dir`** (str, required) Directory for the raw results: `<point>_r<repeat>.json`, `<point>_r<repeat>.metrics.json` (the `/metrics` samples around the window, the acceptance deltas and the gauge averages), `_data/` for the materialized dataset, and `bench_command.txt`, which gets one block appended per invocation.
- **`--label`** (str, required) Series name written into the CSV's `model` column.
- **`--csv`** (str, default: `<out-dir>/<label>.csv`) Where to write the CSV.
- **`--guidellm-bin`** (str, default: `guidellm`) GuideLLM executable.
- **`--dry-run`** Print the GuideLLM commands and exit. Do this first.
- **`--overwrite`** Re-run points whose JSON exists.
- **`--keep-going`** Run the remaining points after a failed one. The exit status is still non-zero, and the failed points are missing from the CSV.

## `parse`

```bash
python scripts/evaluate/throughput_interactivity.py parse <json_dir> --label <name> --csv <out.csv>
```

Reads every `*.json` under `json_dir`, one row per benchmark, named `conc<N>` from the benchmark's own strategy, with the repeat taken from `<point>_r<N>.json`. Server-metrics sidecars are merged in when present. Prints the validation report.

## `validate`

```bash
python scripts/evaluate/throughput_interactivity.py validate <csv>
```

Prints one row per point, averaged over repeats, then the warnings listed in the [tutorial](../user_guide/tutorials/throughput_interactivity.md#read-the-validation-report).

## `plot`

- **`--series`** (`CSV[:NAME[:#COLOR]]`, repeatable, required) One curve per CSV; later series are drawn over earlier ones.
- **`--out`** (str, required) PNG path.
- **`--x`** (`itl` | `little`, default: `itl`) X axis: `1000 / mean ITL` (InferenceX), or throughput divided by requests in flight.
- **`--title`**, **`--subtitle`**, **`--note`** Chart text. Put the hardware, versions and workload in the subtitle; the note is one grey line under the x-axis label.

Points are labelled `N=<streams>`; the legend appears when there is more than one series. To leave a point off a chart, delete its JSON and re-run `parse`.

## CSV columns

| Column                                                                  | Meaning                                                                                                                                                                                                       |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`, `point`, `streams`, `repeat`, `source_json`                    | Identifiers. `point` is `conc<N>` and `streams` is N.                                                                                                                                                         |
| `measured_duration_s`                                                   | Measurement window length.                                                                                                                                                                                    |
| `successful_requests`, `errored_requests`, `incomplete_requests`        | Sizes of GuideLLM's three request lists, warmup included.                                                                                                                                                     |
| `measured_requests`                                                     | Successful requests that completed inside the window; every per-request statistic below is over these.                                                                                                        |
| `completed_rps`                                                         | Requests finished inside the window, per second.                                                                                                                                                              |
| `aggregate_output_tps`                                                  | Output tokens generated inside the window, per second. The y axis.                                                                                                                                            |
| `mean_active_concurrency`                                               | In-flight time inside the window divided by the window.                                                                                                                                                       |
| `interactivity_tps_user`                                                | `aggregate_output_tps / mean_active_concurrency` (Little's law; `plot --x little`).                                                                                                                           |
| `mean_itl_ms`, `interactivity_itl_tps_user`                             | Mean inter-token latency of the requests that completed inside the window, and `1000 / mean_itl_ms`. The x axis.                                                                                              |
| `mean_output_tokens`, `mean_prompt_tokens`                              | Means over the requests that completed inside the window.                                                                                                                                                     |
| `mean_output_tokens_per_iteration`                                      | Output tokens per streamed chunk: about 1 without speculative decoding, the accepted length plus one with it.                                                                                                 |
| `median_ttft_ms`, `p99_ttft_ms`, `median_itl_ms`                        | Over the requests that completed inside the window.                                                                                                                                                           |
| `acceptance_length`, `num_drafts`, `num_accepted_tokens`                | From the server's speculative-decoding counters between the `/metrics` samples nearest the window's start and end: `1 + accepted / drafts`, as in `evaluate.py`. Empty without a speculator.                  |
| `mean_running_requests`, `mean_waiting_requests`, `mean_kv_cache_usage` | vLLM's `num_requests_running`, `num_requests_waiting` and `kv_cache_usage_perc` gauges averaged over the samples inside the window. Check 4 uses the waiting count to tell a server queue from a slow client. |
| `stop_reason`                                                           | Why GuideLLM stopped, normally `max_duration`.                                                                                                                                                                |

## Measurement rules

- Everything is counted inside the measurement window `[measure_start_time, measure_end_time]` of GuideLLM's scheduler metrics. Output tokens come from every request that overlaps the window, successful or incomplete, prorated over `[first token, last token]`. In-flight time is each request's lifetime clipped to the window, errored requests included.
- Per-request statistics (ITL, TTFT, output and prompt lengths, tokens per iteration) are over the requests that completed inside the window. GuideLLM's `successful` list also holds requests that finished during warmup, a third of the list at high N.
- The server's counters are differenced between the `/metrics` samples nearest the window's start and end (the 2 s sampling interval bounds the error), and the gauges are averaged over the samples inside it. The whole-run difference is kept in the sidecar under `whole_run`.
- The subset is repeated and shuffled before the run, never streamed once, so prefix caching must be off. `collect` reads `enable_prefix_caching` from vLLM's `cache_config_info` metric and refuses to start while it is on, and fails any point during which the server's prefix-cache counters moved.
