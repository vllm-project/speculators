# Throughput vs. Interactivity

This tutorial shows how to measure what a speculator buys in production terms: how many output tokens a GPU produces per second (throughput) against how fast each user receives them (interactivity), over the whole range of load. It is the chart SemiAnalysis publishes for every GPU on [InferenceX](https://inferencex.semianalysis.com), produced here for a vLLM server with one or more speculators, on real prompts.

The tool is one helper script, `scripts/evaluate/throughput_interactivity.py`, with four subcommands: `collect` runs the sweep with [GuideLLM](https://github.com/vllm-project/guidellm), `parse` turns the raw results into a CSV, `validate` checks the sweep for the usual ways a benchmark lies, and `plot` draws the chart. It complements [Evaluating Model Performance](evaluating_performance.md): that page measures acceptance rates and latency against request rate; this one measures the throughput and per-user speed trade-off at fixed concurrency, the way InferenceX does.

## What the chart shows

Each point is one load level. For every point the script measures, over a window after a warmup:

- **y axis, output token throughput (tok/s):** output tokens the server generated inside the window, divided by the window length. All requests that overlap the window count, including those still running when it ends; a request's tokens are spread evenly between its first and last token, and only the part inside the window counts.
- **x axis, interactivity (tok/s/user):** 1000 divided by the mean inter-token latency in milliseconds. This is the decode speed one user sees once tokens are flowing. GuideLLM's `inter_token_latency_ms` is `(last token - first token) / (output tokens - 1)`, the same formula as InferenceX's TPOT, so the numbers are comparable with the InferenceX dashboard. Queue wait and time to first token are reported separately.

Moving left along a curve means more concurrent requests: total throughput rises and each user's stream slows down, because every decode step streams the full weights through HBM whether the batch holds 1 sequence or 128. Speculative decoding changes the shape of the curve: a step now emits several tokens, so at low concurrency each user gets tokens much faster, and at high concurrency, where the GPU is already busy, the gain shrinks. Comparing the curves at equal throughput, or at equal interactivity, is the honest way to read what a speculator delivers.

The CSV also carries a second definition, `interactivity_tps_user = throughput / mean requests in flight` (Little's law), which counts queue wait and TTFT against the user. `plot --x little` draws it. Below capacity the two agree within a few percent.

## How a sweep runs

`collect` runs one GuideLLM benchmark per point and repeat:

1. **Load model.** The default is closed loop, as InferenceX: `--streams 1,2,4,8,...` keeps exactly N requests in flight, and each stream sends its next request the moment the previous one completes. N, streams, requests in flight and concurrency are the same number on this page. Concurrency is the knob, the server is never over-queued, and the sweep can go as deep into saturation as `max-num-seqs` allows. `--rates r,...` adds open-loop points (requests arrive on a timer whatever the server does); either or both may be given. Open loop answers "what arrival rate can this server absorb" but cannot measure past capacity.
2. **Warmup and window.** Each point runs for `--warmup-seconds` plus `--max-seconds`. GuideLLM excludes the warmup, and the parser counts only what happened inside the window, so cold start and the ramp-up of the batch are not in the numbers. Make the window several request lifetimes long; `validate` tells you when it was too short.
3. **Acceptance.** Around every run the script snapshots the server's `/metrics` counters and stores the speculative-decoding acceptance over the run (`num_drafts`, accepted tokens, acceptance length, which is the accepted draft tokens per step plus one, i.e. the tokens emitted per verifier step, and per-position acceptance) in a sidecar file next to the raw JSON. The CSV carries `acceptance_length` per point, so you can see acceptance fall or hold as concurrency grows. The same snapshots give `prefix_cache_hit_rate`, the share of prompt tokens the server served from its prefix cache. It also records `mean_output_tokens_per_iteration`, the tokens per streamed chunk seen by the client, as a cross-check.
4. **Provenance.** `bench_command.txt` in the output directory records the timestamp, full command, git SHA, GuideLLM, vLLM and speculators versions, and the server's `/version` and `/v1/models` responses.

Existing results are skipped unless `--overwrite`, so a sweep can be resumed or extended into the same directory, and the CSV is rebuilt from every JSON found there.

## What is supported

| Input             | Flags                                                                           | Notes                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A dataset subset  | `--dataset RedHatAI/speculator_benchmarks --subset HumanEval --max-tokens 1024` | Real prompts through `/v1/chat/completions`. The subset is fetched once, repeated enough times for the sweep (GuideLLM stops when a dataset runs out), shuffled with a fixed seed and written under `<out-dir>/_data/`. Any subset name, a local directory or a local `.jsonl` works; `--prompt-column` names the column. Add `--ignore-eos` to force the output length. |
| Random text       | `--prompt-tokens 1000 --output-tokens 1000 [--range-ratio 0.8]`                 | Lengths drawn uniformly from 80% to 100% of the target, the InferenceX convention. Random tokens defeat prefix caching, which is right for a hardware number and wrong for a speculator: a drafter has nothing to predict in random text, so use a dataset to measure one.                                                                                               |
| Any GuideLLM spec | `--data kind=...`                                                               | Passed through unchanged.                                                                                                                                                                                                                                                                                                                                                |

Both load models (`--streams`, `--rates`), repeats (`--repeats`), a single-stream point (`--synchronous`), and any OpenAI-compatible server work. Requests carry no sampling parameters unless `--extra-body` adds them (`--extra-body temperature=0` for greedy decoding), so by default the server's own defaults apply, which for vLLM come from the model's `generation_config.json`; acceptance length depends on them, so keep them identical across the configurations you compare and name them in the chart's subtitle. GuideLLM 0.8.0 or newer is recommended and is what the script was tested with. Over HTTP, GuideLLM does not report `mean_cached_tokens` (its OpenAI backend never reads `prompt_tokens_details`), so check 7 reads the server's prefix-cache counters from `/metrics` instead; serve with `--no-enable-prefix-caching` rather than depending on the check.

## Run the example

[`examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh`](https://github.com/vllm-project/speculators/blob/main/examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh) benchmarks `Qwen/Qwen3.8-27B` alone, with its DSpark speculator [`RedHatAI/Qwen3.8-27B-speculator.dspark`](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) (7 draft tokens), and with its MTP head, the multi-token-prediction head shipped in the checkpoint (2 draft tokens; the head has one layer, so vLLM's own default would be 1, and the vLLM recipe for this model suggests 3, so treat the count as a knob), on the HumanEval and math_reasoning subsets of `RedHatAI/speculator_benchmarks`:

```bash
pip install "guidellm>=0.8.0" pillow
CUDA_VISIBLE_DEVICES=0 bash examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh
```

It launches vLLM for each configuration, sweeps N = 1 to 128 on each dataset, stops the server, and draws one chart per dataset. The sections below were produced by exactly this procedure.

## Run it by hand

1. Serve each configuration in turn with identical flags apart from `--speculative-config`: the plain model (omit the flag), DSpark (below), and MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`). Turn prefix caching off: the repeated prompts would otherwise be prefilled from cache, and GuideLLM cannot see cache hits over HTTP. Serve on one GPU, as the model would be deployed in production, not on every GPU in the box.

   ```bash
   vllm serve Qwen/Qwen3.8-27B --port 8010 --max-model-len 16384 \
     --max-num-seqs 256 --max-num-batched-tokens 16384 \
     --no-enable-prefix-caching \
     --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dspark","num_speculative_tokens":7,"method":"dspark"}'
   ```

2. Collect one sweep per server configuration, each into its own `--out-dir` and with its own `--label` (`baseline`, `dspark`, `mtp`). Use `--dry-run` first to see the GuideLLM commands. Points whose JSON already exists are skipped, so the short-window and long-window runs below go into the same directory and the CSV is rebuilt from all of them.

   ```bash
   python scripts/evaluate/throughput_interactivity.py collect \
     --target http://localhost:8010 --model Qwen/Qwen3.8-27B \
     --dataset RedHatAI/speculator_benchmarks --subset HumanEval --max-tokens 1024 \
     --streams 1,2,4,8,16 --max-seconds 90 --warmup-seconds 30 --repeats 3 \
     --out-dir runs/dspark_humaneval --label dspark --csv dspark_humaneval.csv --keep-going
   python scripts/evaluate/throughput_interactivity.py collect \
     --target http://localhost:8010 --model Qwen/Qwen3.8-27B \
     --dataset RedHatAI/speculator_benchmarks --subset HumanEval --max-tokens 1024 \
     --streams 32,64,128 --max-seconds 120 --warmup-seconds 60 --repeats 3 \
     --out-dir runs/dspark_humaneval --label dspark --csv dspark_humaneval.csv --keep-going
   ```

   The high-N points (32 and up) get the longer warmup and window because their requests take tens of seconds. To change a point that already ran, pass `--overwrite`.

3. Validate before you believe it:

   ```bash
   python scripts/evaluate/throughput_interactivity.py validate dspark_humaneval.csv
   ```

4. Plot the configurations you want to compare:

   ```bash
   python scripts/evaluate/throughput_interactivity.py plot \
     --series baseline_humaneval.csv:'no speculator':'#7a8c3f' \
     --series dspark_humaneval.csv:'dspark, 7 draft tokens':'#b52513' \
     --series mtp_humaneval.csv:'MTP head, 2 draft tokens':'#2c6e9b' \
     --title 'Qwen3.8-27B, HumanEval' --subtitle '1x B200, vLLM 0.30, max_tokens 1024' \
     --out humaneval.png
   ```

## Read the validation report

`validate` prints one row per point (offered load, achieved rate, throughput, concurrency, both interactivity definitions, ITL, TTFT, acceptance length, arrival burstiness, and the spread across repeats) and warns about eight known problems. Each of them produced a wrong chart at least once:

1. **Achieved rate does not track the offered rate** (open loop): the generator is not sending what its label says.
2. **Several open-loop points share one concurrency:** a cap on the client or the server, not capacity.
3. **A run ended on `requests_exhausted`:** the dataset ran out; set `--dataset-repeat` higher than the automatic choice (20 requests per stream, at least 5,000 requests).
4. **Tokens per completed request do not match the output length:** the window was too short for steady state; lengthen the warmup and the window.
5. **Repeats disagree by more than 5%:** differences smaller than that are noise.
6. **Requests start in synchronized waves:** closed-loop streams with identical lengths finish and restart together, prefill in one burst and decode with nothing else in the batch, which overstates throughput and interactivity (ITL comes out lower than under mixed prefill and decode). Jitter random lengths or use a dataset.
7. **Prompts served from the prefix cache** (`prefix_cache_hit_rate` from the server's counters): a small subset repeated many times gets cheaper prefill than its length suggests. Serve with `--no-enable-prefix-caching` for a clean number, or report it as a cached workload.
8. **Closed-loop points queue inside the server:** more streams than the server can hold wait inside vLLM, so TTFT jumps while ITL does not. The cause is the batch cap (`max-num-seqs`) or a full KV cache; raise the cap, free KV cache memory, or stop the sweep below that N.

## Choosing N and windows

- Keep every server flag identical apart from the drafter, and serve with `--no-enable-prefix-caching`, or the repeated prompts are prefilled from cache. One GPU per server, sized the way the model is deployed in production (`TP=1` for a 27B dense model); running the configurations on separate GPUs of one host at the same time is fine, spreading one server over every free GPU is not, because it measures a deployment nobody runs.
- Pin the server's scheduler limits and set the batch cap at or above the largest N: `--max-num-seqs 256 --max-num-batched-tokens 16384` for a sweep to N = 128. Left unset, vLLM derives them from GPU memory (`max-num-seqs` 256 to 1,024, batched tokens 2,048 to 16,384), so the same script measures a different scheduler on a different host. InferenceX pins the batch size in every recipe, usually at or above the concurrency of the point, with a large chunked-prefill budget. Raise the cap if you want the left end of the curve to show the GPU rather than one configuration. When the cap is above N and check 8 still fires, the KV cache is the usual limit: sample vLLM's `num_requests_running`, `num_requests_waiting` and `kv_cache_usage_perc` gauges while the point runs. In the example, the DSpark server held about 74 running requests on one B200 before queueing.
- A smaller `--max-num-batched-tokens` gives better ITL, because fewer prefill tokens interrupt decode steps; a larger one gives better TTFT and throughput. vLLM's tuning guide recommends above 8,192 for throughput.
- At 1,024 output tokens, a request takes a few seconds at N = 1 and tens of seconds at N = 128, so the high-N points need a 60 s warmup and a 120 s window or more.
- Run the configurations one at a time on a quiet machine, keep the server flags identical apart from the speculator, and repeat every point before quoting a difference of a few percent.

## What the example produces

Two charts, one per dataset, with the plain model, the DSpark speculator (7 draft tokens) and the checkpoint's MTP head (2 draft tokens). These were measured on one B200 per server with vLLM 0.30.1rc1 and GuideLLM 0.8.0, one run per point, before the example turned off prefix caching, so their prefill was partly served from cache; your numbers will differ with hardware, vLLM version and server flags, which is why the example records all of them in `bench_command.txt` and the serve logs.

![Throughput vs. interactivity, HumanEval](../../assets/throughput_interactivity_qwen3_8_27b_humaneval.png)

![Throughput vs. interactivity, math_reasoning](../../assets/throughput_interactivity_qwen3_8_27b_math_reasoning.png)

How to read them: at one request in flight, DSpark gives each user about three times the plain model's speed on HumanEval and four times on math_reasoning, in line with its acceptance length; MTP with 2 draft tokens gives about two times. Moving left, every gain shrinks as the verifier batch fills, and the curves cross: with these settings DSpark delivers the most throughput wherever each user must get more than roughly 100 to 125 tok/s, while MTP's cheaper drafts deliver the most output per GPU at lower per-user speeds and never queue. Pick the per-user speed you must guarantee, read the throughput each configuration gives at it, and that is the comparison that matters for a deployment. `validate` flagged DSpark's N = 128 points as queueing inside the server: with the drafter loaded, the KV cache on one B200 held about 74 requests, so that point measures queueing plus decode.

See [throughput_interactivity.py](../../cli/throughput_interactivity.md) for the full command-line reference and the CSV columns.
