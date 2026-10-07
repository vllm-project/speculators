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
3. **Acceptance.** Around every run the script snapshots the server's `/metrics` counters and stores the speculative-decoding acceptance over the run (`num_drafts`, accepted tokens, acceptance length, which is the accepted draft tokens per step plus one, i.e. the tokens emitted per verifier step, and per-position acceptance) in a sidecar file next to the raw JSON. The CSV carries `acceptance_length` per point, so you can see acceptance fall or hold as concurrency grows. It also records `mean_output_tokens_per_iteration`, the tokens per streamed chunk seen by the client, as a cross-check.
4. **Provenance.** `bench_command.txt` in the output directory records the timestamp, full command, git SHA, GuideLLM, vLLM and speculators versions, and the server's `/version` and `/v1/models` responses.

Existing results are skipped unless `--overwrite`, so a sweep can be resumed or extended into the same directory, and the CSV is rebuilt from every JSON found there.

## What is supported

| Input             | Flags                                                                           | Notes                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A dataset subset  | `--dataset RedHatAI/speculator_benchmarks --subset HumanEval --max-tokens 1024` | Real prompts through `/v1/chat/completions`. The subset is fetched once, repeated enough times for the sweep (GuideLLM stops when a dataset runs out), shuffled with a fixed seed and written under `<out-dir>/_data/`. Any subset name, a local directory or a local `.jsonl` works; `--prompt-column` names the column. Add `--ignore-eos` to force the output length. |
| Random text       | `--prompt-tokens 1000 --output-tokens 1000 [--range-ratio 0.8]`                 | Lengths drawn uniformly from 80% to 100% of the target, the InferenceX convention. Random tokens defeat prefix caching, which is right for a hardware number and wrong for a speculator: a drafter has nothing to predict in random text, so use a dataset to measure one.                                                                                               |
| Any GuideLLM spec | `--data kind=...`                                                               | Passed through unchanged.                                                                                                                                                                                                                                                                                                                                                |

Both load models (`--streams`, `--rates`), repeats (`--repeats`), a single-stream point (`--synchronous`), and any OpenAI-compatible server work. GuideLLM 0.8.0 or newer is recommended and is what the script was tested with; 0.7.1 works but does not report `mean_cached_tokens`, so check 7 stays silent.

## Run the example

[`examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh`](https://github.com/vllm-project/speculators/blob/main/examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh) benchmarks `Qwen/Qwen3.8-27B` alone, with its DSpark speculator [`RedHatAI/Qwen3.8-27B-speculator.dspark`](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) (7 draft tokens), and with its MTP head, the multi-token-prediction head shipped in the checkpoint (2 draft tokens, as Qwen's model cards recommend), on the HumanEval and math_reasoning subsets of `RedHatAI/speculator_benchmarks`:

```bash
pip install "guidellm>=0.8.0" pillow
CUDA_VISIBLE_DEVICES=0 bash examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh
```

It launches vLLM for each configuration, sweeps N = 1 to 128 on each dataset, stops the server, and draws one chart per dataset. The sections below were produced by exactly this procedure.

## Run it by hand

1. Serve each configuration in turn with identical flags apart from `--speculative-config`: the plain model (omit the flag), DSpark (below), and MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`).

   ```bash
   vllm serve Qwen/Qwen3.8-27B --port 8010 --max-model-len 16384 \
     --max-num-seqs 256 --max-num-batched-tokens 16384 \
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
7. **Prompts served from the prefix cache:** a small subset repeated many times gets cheaper prefill than its length suggests. Disable prefix caching on the server for a clean number, or report it as a cached workload.
8. **Closed-loop points queue inside the server:** more streams than the server can hold wait inside vLLM, so TTFT jumps while ITL does not. The cause is the batch cap (`max-num-seqs`) or a full KV cache; raise the cap, free KV cache memory, or stop the sweep below that N.

## Choosing N and windows

- Pin the server's scheduler limits and set the batch cap at or above the largest N: `--max-num-seqs 256 --max-num-batched-tokens 16384` for a sweep to N = 128. Left unset, vLLM derives them from GPU memory (`max-num-seqs` 256 to 1,024, batched tokens 2,048 to 16,384), so the same script measures a different scheduler on a different host. InferenceX pins the batch size in every recipe, usually at or above the concurrency of the point, with a large chunked-prefill budget. Raise the cap if you want the left end of the curve to show the GPU rather than one configuration. When the cap is above N and check 8 still fires, the KV cache is the usual limit: sample vLLM's `num_requests_running`, `num_requests_waiting` and `kv_cache_usage_perc` gauges while the point runs. In the results below, the DSpark server's gauges never showed more than 74 running requests (see `examples/evaluate/results/qwen3_8_27b_dspark/diagnostic_pinned_limits/`).
- A smaller `--max-num-batched-tokens` gives better ITL, because fewer prefill tokens interrupt decode steps; a larger one gives better TTFT and throughput. vLLM's tuning guide recommends above 8,192 for throughput.
- At 1,024 output tokens, a request takes a few seconds at N = 1 and tens of seconds at N = 128, so the high-N points need a 60 s warmup and a 120 s window or more.
- Run the configurations one at a time on a quiet machine, keep the server flags identical apart from the speculator, and repeat every point before quoting a difference of a few percent.

## Results: Qwen3.8-27B alone, with its DSpark speculator, and with its MTP head

Measured on 2026-10-06 with the example script's `collect` calls: one B200 per server, vLLM 0.30.1rc1 nightly, GuideLLM 0.8.0, `max_tokens` 1024, N = 1 to 128 requests in flight, one run per point. Two differences from running the example as is: the plain and DSpark servers were started without the scheduler limits the example now pins, and they ran at the same time on two GPUs of one host (see the caveats). In the files and tables, `baseline` and "no speculator" both mean the plain target model, and "MTP head" is the multi-token-prediction head shipped in the Qwen3.8-27B checkpoint. CSVs, provenance and the validation reports are under `examples/evaluate/results/qwen3_8_27b_dspark/`.

![Throughput vs. interactivity, HumanEval](../../assets/throughput_interactivity_qwen3_8_27b_humaneval.png)

![Throughput vs. interactivity, math_reasoning](../../assets/throughput_interactivity_qwen3_8_27b_math_reasoning.png)

**What the two charts say.** With one request in flight, DSpark multiplies each user's speed by 2.9 on HumanEval (263 against 92 tok/s per user) and by 4.0 on math_reasoning (373 against 92); the model's own MTP head at 2 draft tokens gives 2.0x and 2.2x (185 and 207 tok/s per user). The measured acceptance lengths explain the gap: 3.4 to 4.7 for DSpark against 2.3 to 2.5 for MTP. Every gain shrinks as concurrency grows; the usual explanation is that the verifier's batch becomes compute-bound while every drafted token still has to be verified: at 64 in flight DSpark gives 1.16x and 1.54x per user with 10% and 36% more throughput, MTP 1.44x and 1.55x per user with +41% and +48% throughput. At 128 in flight the plain server produces 5,359 tok/s on HumanEval against 3,967 with DSpark and 6,129 with MTP, and 4,977 on math_reasoning against 4,730 and 6,078. The DSpark configuration reaches its capacity between 64 and 128 in flight: its KV cache is full (vLLM's gauges never showed more than 74 running requests), and `validate` flags its N = 128 points as queueing inside the server (median TTFT 11.0 s and 3.4 s); the MTP configuration's median TTFT at 128 is 0.16 s and 0.16 s. MTP, whose 2-token drafts cost little verification and whose 0.16 s TTFT at N = 128 indicates that its cache held all 128 requests (its gauges were not sampled), keeps gaining throughput up to N = 128 and ends above the plain server on both datasets. Acceptance length is flat across the sweep for both speculators, so the shrinking gains are not the drafters getting worse; the GPU running out of compute is the likely cause.

**Reading the chart at equal user experience.** Pick a per-user speed and read the throughput each configuration can deliver at it. Around 80 tok/s per user, the plain server runs about 16 requests in flight (1,200 tok/s on either dataset); the DSpark server delivers about 3,800 tok/s on HumanEval (between 32 and 64 in flight) and 4,700 on math_reasoning (64 in flight); the MTP server delivers about 5,100 tok/s on both (64 in flight). The ranking depends on the target speed: above about 125 tok/s per user on HumanEval and about 100 on math_reasoning, DSpark delivers more throughput at that speed than MTP, and serves more users at it; at 80 tok/s per user MTP's cheaper drafts give the most output per GPU.

**Caveats.** One run per point, as the example is configured; raise `REPEATS` to 3 before quoting a difference of a few percent. `validate` flagged one point, MTP at N = 1 on HumanEval: the 90 s window held 894 generated tokens per completed request against a mean output of 798, so that point is a short-window estimate. Output lengths are the model's own (no `--ignore-eos`): HumanEval responses average 720 to 800 tokens and math_reasoning responses 260 to 350 tokens, so the two datasets exercise different prompt-to-output ratios, and a shorter answer spends a larger share of its life in prefill. The plain and DSpark results were measured with vLLM's defaults on this GPU (`max-num-seqs` 1,024, `max-num-batched-tokens` 16,384) and the MTP results with the limits the example now pins (256 and 16,384); a rerun of DSpark at N = 96 and 128 with the pinned limits reproduced its numbers (3,897 tok/s at 57 tok/s per user at N = 128), and vLLM's gauges showed its queueing is a full KV cache, not the batch cap: never more than 74 running requests, the rest waiting, cache 99.9% full. Loading the drafter shrinks the KV cache from 1.36M to 544k token slots (vLLM reports KV capacity in token slots; the drafter's layers add cache groups that share that memory), so one B200 holds no more than 74 concurrent requests in that configuration. A quantized target such as `RedHatAI/Qwen3.8-27B-NVFP4` frees memory for the cache. MTP uses 2 draft tokens because Qwen's model cards for this architecture recommend `"num_speculative_tokens":2`; vLLM's bare default for the one-layer MTP head would be 1, and more draft tokens were not measured.

### HumanEval

| N in flight | no speculator: tok/s | tok/s/user | DSpark: tok/s | tok/s/user | acceptance | MTP: tok/s | tok/s/user | acceptance |
| ----------- | -------------------- | ---------- | ------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 1           | 92                   | 92         | 247           | 263        | 3.40       | 179        | 185        | 2.25       |
| 2           | 177                  | 89         | 469           | 252        | 3.36       | 361        | 185        | 2.27       |
| 4           | 346                  | 87         | 909           | 236        | 3.44       | 687        | 176        | 2.28       |
| 8           | 671                  | 84         | 1,624         | 213        | 3.44       | 1,273      | 163        | 2.29       |
| 16          | 1,250                | 79         | 2,601         | 173        | 3.45       | 2,249      | 145        | 2.26       |
| 32          | 2,238                | 70         | 3,549         | 117        | 3.43       | 3,707      | 119        | 2.27       |
| 64          | 3,666                | 58         | 4,034         | 67         | 3.44       | 5,174      | 83         | 2.28       |
| 128         | 5,359                | 42         | 3,967         | 60         | 3.43       | 6,129      | 49         | 2.27       |

### math_reasoning

| N in flight | no speculator: tok/s | tok/s/user | DSpark: tok/s | tok/s/user | acceptance | MTP: tok/s | tok/s/user | acceptance |
| ----------- | -------------------- | ---------- | ------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 1           | 91                   | 92         | 332           | 373        | 4.59       | 197        | 207        | 2.51       |
| 2           | 175                  | 89         | 603           | 355        | 4.49       | 385        | 206        | 2.51       |
| 4           | 341                  | 87         | 1,120         | 326        | 4.61       | 725        | 193        | 2.53       |
| 8           | 654                  | 83         | 1,967         | 286        | 4.65       | 1,302      | 175        | 2.52       |
| 16          | 1,196                | 76         | 3,116         | 227        | 4.61       | 2,251      | 151        | 2.53       |
| 32          | 2,102                | 67         | 4,189         | 150        | 4.67       | 3,671      | 122        | 2.53       |
| 64          | 3,436                | 55         | 4,684         | 84         | 4.67       | 5,093      | 85         | 2.53       |
| 128         | 4,977                | 40         | 4,730         | 75         | 4.67       | 6,078      | 50         | 2.54       |

Interactivity is 1000 / mean inter-token latency. Acceptance is the mean accepted draft length plus one, from vLLM's counters over the run.

See [throughput_interactivity.py](../../cli/throughput_interactivity.md) for the full command-line reference and the CSV columns.
