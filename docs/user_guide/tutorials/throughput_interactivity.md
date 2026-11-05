# Throughput vs. Interactivity

Use the [evaluation guide](evaluating_performance.md#serve-the-model) to serve `RedHatAI/Qwen3.8-27B-speculator.dspark`, then run:

```bash
python scripts/evaluate/evaluate.py speedup \
  --dataset humaneval speed_bench_math \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/throughput-dspark

aiperf plot output/throughput-dspark --output output/throughput-plots
```

Run the same datasets and settings against each server configuration, using a separate output directory for each. See [Evaluating Model Performance](evaluating_performance.md#speedup-and-plotting) for more command examples.

## Historical results

The graphs below were collected with GuideLLM, using the historical datasets and settings described here. They were not produced by the AIPerf command above; the collectors use different metric definitions and measurement windows.

Two charts, one per dataset, with the plain model, the DSpark speculator (7 draft tokens) and the checkpoint's MTP head (2 draft tokens). These were measured on one B200 per server with vLLM 0.30.1rc1 and GuideLLM 0.8.0, one run per point, before the example turned off prefix caching, so their prefill was partly served from cache; your numbers will differ with hardware, vLLM version and server flags, which is why the example records all of them in `bench_command.txt` and the serve logs.

![Throughput vs. interactivity, HumanEval](../../assets/throughput_interactivity_qwen3_8_27b_humaneval.png)

![Throughput vs. interactivity, math_reasoning](../../assets/throughput_interactivity_qwen3_8_27b_math_reasoning.png)

How to read them: at one request in flight, DSpark gives each user about three times the plain model's speed on HumanEval and four times on math_reasoning, in line with its acceptance length; MTP with 2 draft tokens gives about two times. Moving left, every gain shrinks as the verifier batch fills, and the curves cross: with these settings DSpark delivers the most throughput wherever each user must get more than roughly 100 to 125 tok/s, while MTP's cheaper drafts deliver the most output per GPU at lower per-user speeds and never queue. Pick the per-user speed you must guarantee, read the throughput each configuration gives at it, and that is the comparison that matters for a deployment. `validate` flagged DSpark's N = 128 points as queueing inside the server: with the drafter loaded, the KV cache on one B200 held about 74 requests, so that point measures queueing plus decode.

### The same three configurations on an H100

The same procedure on one H100 80GB per server, with vLLM 0.29.0 and GuideLLM 0.8.0, prefix caching off, one run per point, and every request sent with the model's default sampling (temperature 1.0, top-p 0.95, top-k 20, the values in its `generation_config.json`, passed with `--extra-body` so that all three servers see the same):

![Throughput vs. interactivity, HumanEval, H100](../../assets/throughput_interactivity_qwen3_8_27b_h100_humaneval.png)

![Throughput vs. interactivity, math_reasoning, H100](../../assets/throughput_interactivity_qwen3_8_27b_h100_math_reasoning.png)

At one request in flight the picture matches the B200: DSpark gives each user 3.0 times the plain model's speed on HumanEval and 3.9 times on math_reasoning, MTP with 2 draft tokens 2.0 and 2.1 times. The right side of the chart is different, and the cause is the KV cache, not the drafter. Qwen3.8-27B is a hybrid model whose 48 linear-attention layers keep about 160 MB of recurrent state per request, and vLLM holds one copy of that state per draft token so it can roll back rejected drafts (`1 + num_speculative_tokens` copies). A request therefore costs about 0.2 GB of KV cache without a speculator, 0.55 GB with MTP and 1.4 GB with DSpark. After the weights, an 80 GB H100 leaves 17 GiB of KV cache to the plain server, 15 GiB to MTP and 10 GiB to DSpark, so the three servers hold about 88, 28 and 7 running requests. Past that, `validate` reports check 4 (DSpark from N = 16, MTP from N = 32, the plain server at N = 128): throughput plateaus at about 1,000 tok/s for DSpark and 1,900 for MTP while the ITL-based interactivity stays flat, because once a request streams it is measured like any other, and `plot --x little` shows the same points with the queue wait counted against the user. On a B200 the same per-request cost leaves room for about 73 DSpark requests, which is why only its N = 128 point queued there. Both runs say the same thing about the drafters; the H100 run also says how much memory each configuration needs before concurrency, not the drafter, sets the throughput.
