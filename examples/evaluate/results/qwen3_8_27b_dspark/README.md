# Qwen3.8-27B with and without `RedHatAI/Qwen3.8-27B-speculator.dspark`

Results of `examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh`'s procedure, measured on 2026-10-06. The charts are in `docs/assets/throughput_interactivity_qwen3_8_27b_*.png` and the discussion in the [tutorial](../../../docs/user_guide/tutorials/throughput_interactivity.md#results-qwen38-27b-with-and-without-its-dspark-speculator).

| file                                  | contents                                                                                                                               |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `<config>_<subset>.csv`               | one row per point: `baseline` is `Qwen/Qwen3.8-27B` alone, `dspark` adds the speculator with 7 draft tokens                            |
| `<config>_<subset>_validate.txt`      | the `validate` table and warnings for that sweep                                                                                       |
| `<config>_<subset>_bench_command.txt` | provenance written by `collect`: command, versions, server version and model list                                                      |
| `serve_<config>_command.txt`          | the exact `vllm serve` command of each server                                                                                          |
| `run_sweeps_used.sh`                  | the driver that produced these files: the same `collect` calls as the example script, with the two servers run in parallel on two GPUs |

Environment: one NVIDIA B200 per server (TP=1), vLLM 0.30.1rc1.dev418 nightly, GuideLLM 0.8.0, Python 3.12, `--max-model-len 16384`, default `max-num-seqs` and `max-num-batched-tokens`. One run per point.

**What the two charts say.** With one request in flight the speculator multiplies each user's speed by 2.9 on HumanEval (263 against 92 tok/s per user) and by 4.0 on math_reasoning (373 against 92), in line with its measured acceptance lengths of 3.4 and 4.7. The gain shrinks as concurrency grows, because the verifier's batch becomes compute-bound and every drafted token still has to be verified: at 64 in flight the per-user gain is 1.16x on HumanEval and 1.54x on math_reasoning, with 10% and 36% more throughput. At 128 in flight the plain server produces more tokens per second on HumanEval (5,359 against 3,967) and about the same on math_reasoning (4,977 against 4,730); the speculator configuration has reached its capacity between 64 and 128 streams, and `validate` flags its 128-stream points as queueing inside the server (median TTFT 11.0 s and 3.4 s). Acceptance length is flat across the sweep (3.36 to 3.45 on HumanEval, 4.49 to 4.67 on math_reasoning), so the shrinking gain is the GPU running out of compute, not the drafter getting worse.

**Reading the chart at equal user experience.** Pick a per-user speed and read the throughput each configuration can deliver at it. Around 80 tok/s per user, the plain server runs about 16 requests in flight (1,200 tok/s on either dataset), while the speculator server delivers about 3,800 tok/s on HumanEval (between 32 and 64 in flight) and 4,700 tok/s on math_reasoning (64 in flight): three to four times the output per GPU for the same experience.

**Caveats.** One run per point, as the example is configured; raise `REPEATS` to 3 before quoting a difference of a few percent. Output lengths are the model's own (no `--ignore-eos`): HumanEval responses average 720 to 800 tokens and math_reasoning responses 260 to 350 tokens, so the two datasets exercise different prompt-to-output ratios, and a shorter answer spends a larger share of its life in prefill. Both servers ran with default `max-num-seqs` and `max-num-batched-tokens` (16,384); the speculator server was not tuned for high concurrency.

## HumanEval

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

## math_reasoning

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
