# Throughput vs. Interactivity

This tutorial shows how to measure what a speculator buys in production terms: how many output tokens a GPU produces per second (throughput) against how fast each user receives them (interactivity), over the whole range of load. It is the chart SemiAnalysis publishes for every GPU on [InferenceX](https://inferencex.semianalysis.com), produced here for a vLLM server with and without a speculator, on real prompts.

The tool is one helper script, `scripts/evaluate/throughput_interactivity.py`, with four subcommands: `collect` runs the sweep with [GuideLLM](https://github.com/vllm-project/guidellm), `parse` turns the raw results into a CSV, `validate` checks the sweep for the usual ways a benchmark lies, and `plot` draws the chart. It complements [Evaluating Model Performance](evaluating_performance.md): that page measures acceptance rates and latency against request rate; this one measures the throughput and per-user speed trade-off at fixed concurrency, the way InferenceX does.

## What the chart shows

Each point is one load level. For every point the script measures, over a window after a warmup:

- **y axis, output token throughput (tok/s):** output tokens the server generated inside the window, divided by the window length. All requests that overlap the window count, including those still running when it ends; a request's tokens are spread evenly between its first and last token, and only the part inside the window counts.
- **x axis, interactivity (tok/s/user):** 1000 divided by the mean inter-token latency in milliseconds. This is the decode speed one user sees once tokens are flowing. GuideLLM's `inter_token_latency_ms` is `(last token - first token) / (output tokens - 1)`, the same formula as InferenceX's TPOT, so the numbers are comparable with the InferenceX dashboard. Queue wait and time to first token are reported separately.

Moving left along a curve means more concurrent requests: total throughput rises and each user's stream slows down, because every decode step streams the full weights through HBM whether the batch holds 1 sequence or 128. Speculative decoding changes the shape of the curve: a step now emits several tokens, so at low concurrency each user gets tokens much faster, and at high concurrency, where the GPU is already busy, the gain shrinks. Comparing the two curves at equal throughput, or at equal interactivity, is the honest way to read what a speculator delivers.

The CSV also carries an older definition, `interactivity_tps_user = throughput / mean requests in flight` (Little's law), which counts queue wait and TTFT against the user. `plot --x little` draws it. Below capacity the two agree within a few percent.

## How a sweep runs

`collect` runs one GuideLLM benchmark per point and repeat:

1. **Load model.** The default is closed loop, as InferenceX: `--streams 1,2,4,8,...` keeps exactly N requests in flight, and each stream sends its next request the moment the previous one completes. Concurrency is the knob, the server is never over-queued, and the sweep can go as deep into saturation as `max-num-seqs` allows. `--rates r,...` gives open-loop points instead (requests arrive on a timer whatever the server does), which answers "what arrival rate can this server absorb" but cannot measure past capacity.
2. **Warmup and window.** Each point runs for `--warmup-seconds` plus `--max-seconds`. GuideLLM excludes the warmup, and the parser counts only what happened inside the window, so cold start and the ramp-up of the batch are not in the numbers. Make the window several request lifetimes long; `validate` tells you when it was too short.
3. **Acceptance.** Around every run the script snapshots the server's `/metrics` counters and stores the speculative-decoding acceptance over the run (`num_drafts`, accepted tokens, acceptance length, per-position acceptance) in a sidecar file next to the raw JSON. The CSV carries `acceptance_length` per point, so you can see acceptance fall or hold as concurrency grows. It also records `mean_output_tokens_per_iteration`, the tokens per streamed chunk seen by the client, as a cross-check.
4. **Provenance.** `bench_command.txt` in the output directory records the timestamp, full command, git SHA, GuideLLM, vLLM and speculators versions, and the server's `/version` and `/v1/models` responses.

Existing results are skipped unless `--overwrite`, so a sweep can be resumed or extended into the same directory, and the CSV is rebuilt from every JSON found there.

## What is supported

| Input             | Flags                                                                           | Notes                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A dataset subset  | `--dataset RedHatAI/speculator_benchmarks --subset HumanEval --max-tokens 1024` | Real prompts through `/v1/chat/completions`. The subset is fetched once, repeated enough times for the sweep (GuideLLM stops when a dataset runs out), shuffled with a fixed seed and written under `<out-dir>/_data/`. Any subset name, a local directory or a local `.jsonl` works; `--prompt-column` names the column. Add `--ignore-eos` to force the output length. |
| Random text       | `--prompt-tokens 1000 --output-tokens 1000 [--range-ratio 0.8]`                 | Lengths drawn uniformly from 80% to 100% of the target, the InferenceX convention. Random tokens defeat prefix caching, which is right for a hardware number and wrong for a speculator: a drafter has nothing to predict in random text, so use a dataset to measure one.                                                                                               |
| Any GuideLLM spec | `--data kind=...`                                                               | Passed through unchanged.                                                                                                                                                                                                                                                                                                                                                |

Both load models (`--streams`, `--rates`), repeats (`--repeats`), a single-stream point (`--synchronous`), and any OpenAI-compatible server work. GuideLLM 0.7.1 or newer is required; the script was tested with 0.8.0.

## Run the example

[`examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh`](https://github.com/vllm-project/speculators/blob/main/examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh) benchmarks `Qwen/Qwen3.8-27B` alone and with [`RedHatAI/Qwen3.8-27B-speculator.dspark`](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) on the HumanEval and math_reasoning subsets of `RedHatAI/speculator_benchmarks`:

```bash
pip install "guidellm>=0.8.0" pillow
CUDA_VISIBLE_DEVICES=0 bash examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh
```

It launches vLLM for each configuration, sweeps N = 1 to 128 on each dataset, stops the server, and draws one chart per dataset. The sections below were produced by exactly this procedure.

## Run it by hand

1. Serve the model. For the speculator configuration:

   ```bash
   vllm serve Qwen/Qwen3.8-27B --port 8010 --max-model-len 16384 \
     --max-num-seqs 256 --max-num-batched-tokens 16384 \
     --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dspark","num_speculative_tokens":7,"method":"dspark"}'
   ```

2. Collect one sweep per server configuration. Use `--dry-run` first to see the GuideLLM commands.

   ```bash
   python scripts/evaluate/throughput_interactivity.py collect \
     --target http://localhost:8010 --model Qwen/Qwen3.8-27B \
     --dataset RedHatAI/speculator_benchmarks --subset HumanEval --max-tokens 1024 \
     --streams 1,2,4,8,16,32,64,128 --max-seconds 90 --warmup-seconds 30 --repeats 3 \
     --out-dir runs/dspark_humaneval --label dspark --csv dspark_humaneval.csv --keep-going
   ```

   For the largest N run a second `collect` into the same `--out-dir` with a longer warmup and window; each point's JSON is kept and the CSV is rebuilt.

3. Validate before you believe it:

   ```bash
   python scripts/evaluate/throughput_interactivity.py validate dspark_humaneval.csv
   ```

4. Plot the configurations you want to compare:

   ```bash
   python scripts/evaluate/throughput_interactivity.py plot \
     --series baseline_humaneval.csv:'no speculator':'#7a8c3f' \
     --series dspark_humaneval.csv:'dspark, 7 draft tokens':'#b52513' \
     --title 'Qwen3.8-27B, HumanEval' --subtitle '1x B200, vLLM 0.30, max_tokens 1024' \
     --out humaneval.png
   ```

## Read the validation report

`validate` prints one row per point (offered load, achieved rate, throughput, concurrency, both interactivity definitions, ITL, TTFT, acceptance length, arrival burstiness, and the spread across repeats) and warns about eight known problems. Each of them produced a wrong chart at least once:

1. **Achieved rate does not track the offered rate** (open loop): the generator is not sending what its label says.
2. **Several open-loop points share one concurrency:** a cap on the client or the server, not capacity.
3. **A run ended on `requests_exhausted`:** the dataset ran out; raise `--dataset-repeat`.
4. **Tokens per completed request do not match the output length:** the window was too short for steady state; lengthen the warmup and the window.
5. **Repeats disagree by more than 5%:** differences smaller than that are noise.
6. **Requests start in synchronized waves:** closed-loop streams with identical lengths finish and restart together, prefill in one burst and decode with nothing else in the batch, which overstates throughput and ITL. Jitter random lengths or use a dataset.
7. **Prompts served from the prefix cache:** a small subset repeated many times gets cheaper prefill than its length suggests. Disable prefix caching on the server for a clean number, or report it as a cached workload.
8. **Closed-loop points queue inside the server:** more streams than the server can hold wait inside vLLM, so TTFT jumps while ITL does not. The cause is the batch cap (`max-num-seqs`) or a full KV cache; raise the cap, free KV cache memory, or stop the sweep below that N.

## Choosing N and windows

- Pin the server's scheduler limits and set the batch cap at or above the largest N: `--max-num-seqs 256 --max-num-batched-tokens 16384` for a sweep to N = 128. Left unset, vLLM derives them from GPU memory (`max-num-seqs` 256 to 1,024, batched tokens 2,048 to 16,384), so the same script measures a different scheduler on a different host. InferenceX pins the batch size in every recipe, usually at or above the concurrency of the point, with a large chunked-prefill budget. Raise the cap if you want the left end of the curve to show the GPU rather than one configuration. When the cap is above N and check 8 still fires, the KV cache is the usual limit: sample vLLM's `num_requests_running`, `num_requests_waiting` and `kv_cache_usage_perc` gauges while the point runs. In the example below, the speculator server's cache was full at 73 running requests (see `examples/evaluate/results/qwen3_8_27b_dspark/diagnostic_pinned_limits/`).
- A smaller `--max-num-batched-tokens` gives better ITL, because fewer prefill tokens interrupt decode steps; a larger one gives better TTFT and throughput. vLLM's tuning guide recommends above 8,192 for throughput.
- At 1,024 output tokens, a request takes a few seconds at N = 1 and tens of seconds at N = 128, so the high-N points need a 60 s warmup and a 120 s window or more.
- Run the configurations one at a time on a quiet machine, keep the server flags identical apart from the speculator, and repeat every point before quoting a difference of a few percent.

## Results: Qwen3.8-27B with and without its DSpark speculator

Produced by the example script's procedure on 2026-10-06, one B200 per server, vLLM 0.30.1rc1 nightly, GuideLLM 0.8.0, `max_tokens` 1024, N = 1 to 128 requests in flight, one run per point. CSVs, provenance and the validation reports are under `examples/evaluate/results/qwen3_8_27b_dspark/`.

![Throughput vs. interactivity, HumanEval](../../assets/throughput_interactivity_qwen3_8_27b_humaneval.png)

![Throughput vs. interactivity, math_reasoning](../../assets/throughput_interactivity_qwen3_8_27b_math_reasoning.png)

**What the two charts say.** With one request in flight the speculator multiplies each user's speed by 2.9 on HumanEval (263 against 92 tok/s per user) and by 4.0 on math_reasoning (373 against 92), in line with its measured acceptance lengths of 3.4 and 4.7. The gain shrinks as concurrency grows, because the verifier's batch becomes compute-bound and every drafted token still has to be verified: at 64 in flight the per-user gain is 1.16x on HumanEval and 1.54x on math_reasoning, with 10% and 36% more throughput. At 128 in flight the plain server produces more tokens per second on HumanEval (5,359 against 3,967) and about the same on math_reasoning (4,977 against 4,730); the speculator configuration has reached its capacity between 64 and 128 streams, its KV cache is full at about 73 running requests, and `validate` flags its 128-stream points as queueing inside the server (median TTFT 11.0 s and 3.4 s). Acceptance length is flat across the sweep (3.36 to 3.45 on HumanEval, 4.49 to 4.67 on math_reasoning), so the shrinking gain is the GPU running out of compute, not the drafter getting worse.

**Reading the chart at equal user experience.** Pick a per-user speed and read the throughput each configuration can deliver at it. Around 80 tok/s per user, the plain server runs about 16 requests in flight (1,200 tok/s on either dataset), while the speculator server delivers about 3,800 tok/s on HumanEval (between 32 and 64 in flight) and 4,700 tok/s on math_reasoning (64 in flight): three to four times the output per GPU for the same experience.

**Caveats.** One run per point, as the example is configured; raise `REPEATS` to 3 before quoting a difference of a few percent. Output lengths are the model's own (no `--ignore-eos`): HumanEval responses average 720 to 800 tokens and math_reasoning responses 260 to 350 tokens, so the two datasets exercise different prompt-to-output ratios, and a shorter answer spends a larger share of its life in prefill. The results were measured with vLLM's defaults on this GPU (`max-num-seqs` 1,024, `max-num-batched-tokens` 16,384); the example script now pins `max-num-seqs` 256 and 16,384 batched tokens, and a rerun of N = 96 and 128 with those limits reproduced the speculator server's numbers (3,897 tok/s at 57 tok/s per user at N = 128). Its queueing is not the batch cap: vLLM's gauges showed at most 74 running requests, the rest waiting, and the KV cache 99.9% full. The drafter adds 11 KV cache groups and shrinks the cache from 1.36M to 544k token-equivalents, and the hybrid model reserves per-request state, so one B200 holds about 73 concurrent requests in this configuration. A quantized target such as `RedHatAI/Qwen3.8-27B-NVFP4` frees memory for the cache.

### HumanEval

| N in flight | no speculator: tok/s | tok/s/user | dspark: tok/s | tok/s/user | per-user gain | throughput gain | acceptance length | dspark median TTFT |
| ----------- | -------------------- | ---------- | ------------- | ---------- | ------------- | --------------- | ----------------- | ------------------ |
| 1           | 92                   | 92         | 247           | 263        | 2.85x         | 2.70x           | 3.40              | 46 ms              |
| 2           | 177                  | 89         | 469           | 252        | 2.84x         | 2.65x           | 3.36              | 61 ms              |
| 4           | 346                  | 87         | 909           | 236        | 2.72x         | 2.63x           | 3.44              | 63 ms              |
| 8           | 671                  | 84         | 1,624         | 213        | 2.53x         | 2.42x           | 3.44              | 65 ms              |
| 16          | 1,250                | 79         | 2,601         | 173        | 2.21x         | 2.08x           | 3.45              | 72 ms              |
| 32          | 2,238                | 70         | 3,549         | 117        | 1.67x         | 1.59x           | 3.43              | 103 ms             |
| 64          | 3,666                | 58         | 4,034         | 67         | 1.16x         | 1.10x           | 3.44              | 178 ms             |
| 128         | 5,359                | 42         | 3,967         | 60         | 1.41x         | 0.74x           | 3.43              | 11.0 s             |

### math_reasoning

| N in flight | no speculator: tok/s | tok/s/user | dspark: tok/s | tok/s/user | per-user gain | throughput gain | acceptance length | dspark median TTFT |
| ----------- | -------------------- | ---------- | ------------- | ---------- | ------------- | --------------- | ----------------- | ------------------ |
| 1           | 91                   | 92         | 332           | 373        | 4.03x         | 3.64x           | 4.59              | 46 ms              |
| 2           | 175                  | 89         | 603           | 355        | 3.99x         | 3.44x           | 4.49              | 61 ms              |
| 4           | 341                  | 87         | 1,120         | 326        | 3.76x         | 3.28x           | 4.61              | 62 ms              |
| 8           | 654                  | 83         | 1,967         | 286        | 3.44x         | 3.01x           | 4.65              | 64 ms              |
| 16          | 1,196                | 76         | 3,116         | 227        | 2.98x         | 2.61x           | 4.61              | 71 ms              |
| 32          | 2,102                | 67         | 4,189         | 150        | 2.25x         | 1.99x           | 4.67              | 110 ms             |
| 64          | 3,436                | 55         | 4,684         | 84         | 1.54x         | 1.36x           | 4.67              | 191 ms             |
| 128         | 4,977                | 40         | 4,730         | 75         | 1.90x         | 0.95x           | 4.67              | 3.4 s              |

Interactivity is 1000 / mean inter-token latency. "Per-user gain" and "throughput gain" are the speculator configuration divided by the plain one at the same N.

See [throughput_interactivity.py](../../cli/throughput_interactivity.md) for the full command-line reference and the CSV columns.
