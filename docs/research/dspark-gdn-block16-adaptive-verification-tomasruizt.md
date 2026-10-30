---
title: "DSpark and GDN: does adaptive verification help with a larger draft?"
tags:
  - DSpark
  - Adaptive verification
  - Qwen3.8
  - GDN
  - vLLM
---

<div class="entry-byline" markdown>
[![tomasruizt](https://github.com/tomasruizt.png?size=64){ .entry-avatar }](https://github.com/tomasruizt)
**[@tomasruizt](https://github.com/tomasruizt)** posted on **2026-10-27**
</div>

# DSpark and GDN: does adaptive verification help with a larger draft?

## Summary

Adaptive verification (AV) improved throughput over fixed K=15 by 5–25% with Qwen3.8-27B and RadixArk's DSpark v2 drafter in a parallel sweep on separate GPU pairs. Profiling at concurrency 1 showed that verifying fewer tokens saves GDN recurrence work.

GDN supported a 16-token verification window through its Triton fallback under FULL CUDA graph replay, beyond the fused CUDA kernel's limit of 8. The fused kernel's width guard remained in place.

## Results

We compared no speculation, fixed K=15, and K=15 with AV, each on two B300s (TP2, BF16), using SPEED-Bench `throughput_1k`. The three methods ran concurrently on separate GPU pairs. Each setting has three repeats. Tables show medians, and the plot connects means with individual runs shown faintly.

The methods used different GPU pairs and shared host resources, so the measured speedups may include hardware-placement and contention effects. A sequential comparison on the same GPU pair would separate those effects from the choice of decoding method.

![Throughput versus inverse TPOT for target-only, fixed K=15, and AV](../assets/research/dspark-gdn-block16/throughput-inverse-tpot.png)

Higher and further right is better. The x-axis is inverse mean request TPOT in tok/s/user, and the y-axis is aggregate output throughput in tok/s.

AV exceeded both fixed K=15 and target-only throughput at all four measured concurrencies under these conditions.

| Concurrency | Target-only tok/s | Fixed K=15 tok/s | AV tok/s | AV / target-only | AV / fixed |
| ----------: | ----------------: | ---------------: | -------: | ---------------: | ---------: |
|           1 |               143 |              251 |      285 |            2.00× |      1.14× |
|          16 |             1,534 |            2,602 |    2,735 |            1.78× |      1.05× |
|          64 |             3,295 |            2,956 |    3,514 |            1.07× |      1.19× |
|         128 |             3,510 |            3,141 |    3,935 |            1.12× |      1.25× |

<details markdown="1">
<summary>TPOT and acceptance length by concurrency</summary>

The acceptance lengths were modest on this workload, roughly three tokens per step. AL includes the bonus/replacement token, so an AL of 3 means about two accepted draft tokens plus one token from the target.

AL can change with concurrency because timed runs cover different mixes of requests, and AV can also change how many draft tokens it verifies.

| Concurrency | Target-only TPOT (ms) | Fixed TPOT (ms) | AV TPOT (ms) | Fixed AL | AV AL |
| ----------: | --------------------: | --------------: | -----------: | -------: | ----: |
|           1 |                 6.940 |           4.015 |        3.538 |     2.59 |  2.71 |
|          16 |                 9.696 |           5.563 |        5.389 |     3.39 |  3.11 |
|          64 |                16.627 |          18.009 |       13.098 |     3.19 |  3.24 |
|         128 |                27.265 |          28.942 |       19.729 |     3.17 |  2.92 |

</details>

### Why AV helps even at concurrency 1

The c1 result surprised us. With only one request, extra draft tokens are often assumed to be almost free because they reuse the target's weights. Why would verifying fewer of them help?

We checked with CUDA events and Nsight Systems, running the same eight SPEED-Bench prompts with and without AV on a fourth GPU pair. AV reduced the average number of positions verified from 16 to 5.91, while producing almost the same number of output tokens per step. Target time fell from 8.82 to 7.48 ms, with a smaller drafter-time change from 1.57 to 1.50 ms. The drafter still proposed all 15 candidates. Most of the saving came from the target doing less verification work.

GDN explains much of that saving. It updates recurrent state for each verified token, so extra positions still cost time at c1. Across the 48 GDN layers, recurrence time fell from 1.109 to 0.478 ms per step per GPU. That accounts for 47% of the target-time reduction in the Nsight trace.

<details markdown="1">
<summary>C1 query lengths, output tokens, and stage timings</summary>

| Per decode step                     | Fixed K=15 |       AV |
| ----------------------------------- | ---------: | -------: |
| Mean target query length            |         16 |     5.91 |
| Mean emitted tokens                 |       2.60 |     2.62 |
| Target GPU time                     |    8.82 ms |  7.48 ms |
| Drafter GPU time                    |    1.57 ms |  1.50 ms |
| Total worker GPU time               |   10.83 ms |  9.45 ms |
| GDN recurrence kernel time, per GPU |   1.109 ms | 0.478 ms |

</details>

Here is the broader Nsight breakdown. Each row adds up that family's kernel time and reports it per step per GPU. The last column shows its share of the total net saving of 0.910 ms. Negative values mean that family became more expensive with AV.

| Kernel family                               | Fixed SD (ms/step/GPU) | AV (ms/step/GPU) | Saving (ms/step/GPU) | % of total kernel-time saving |
| ------------------------------------------- | ---------------------: | ---------------: | -------------------: | ----------------------------: |
| GDN recurrence                              |                  1.109 |            0.478 |                0.631 |                         69.3% |
| Matrix multiplication and split-K reduction |                  6.409 |            6.102 |                0.308 |                         33.8% |
| All-reduce                                  |                  0.984 |            0.958 |                0.026 |                          2.9% |
| Causal convolution                          |                  0.245 |            0.194 |                0.050 |                          5.5% |
| Normalization                               |                  0.165 |            0.158 |                0.007 |                          0.8% |
| Standard attention                          |                  0.329 |            0.340 |               −0.011 |                         −1.2% |
| Other kernels                               |                  1.331 |            1.433 |               −0.101 |                        −11.1% |
| **Total**                                   |             **10.572** |        **9.662** |            **0.910** |                    **100.0%** |

**GDN recurrence and matrix multiplication provide most of the saving.** Convolution and communication contribute smaller gains. Standard attention does not become faster.

These kernel totals cover the whole capture, including drafter and prefill work. Some kernels overlap, so a 1 ms reduction in their summed durations does not necessarily save 1 ms of elapsed time. The profiles also used a different B300 pair from the main sweep. They show why AV helps, but do not account for every part of the sweep's speedup.

## What is still missing?

We would next compare smaller fixed K values and a simple policy that changes K with batch size. Those comparisons would tell us whether AV is the best choice, rather than just a better choice than always verifying 15 drafts.

The three methods ran in parallel on separate GPU pairs and shared host resources. A sequential comparison on the same pair would remove that confound. We only profiled c1 and have not located a compute-bound crossover or measured which operations determine performance at higher concurrency.

## Appendix

<details markdown="1">
<summary>Why test a 16-token verification window?</summary>

GDN's fused CUDA kernel supports query lengths up to 8, but its Triton fallback handles longer queries. CUDA graph support therefore needs to allow longer queries while keeping the fused kernel's own guard in place. [vLLM PR #60229](https://github.com/vllm-project/vllm/pull/60229) provides that support for adaptive verification.

The implementation question is whether longer queries execute correctly through Triton under CUDA graph replay with real requests.

We chose a DSpark checkpoint trained with block size 16 and explicitly set K=15. That gives the target up to 16 input tokens per request: one existing token plus 15 draft tokens. This goes beyond the fused kernel's limit and gives AV room to choose how much of the draft to verify.

There was also a performance question. A large draft may help when few requests are running, but verifying it in full can waste work at higher concurrency. Can AV avoid that waste? To find out, we compared three choices: no speculation, fixed K=15, and K=15 with AV.

</details>

<details markdown="1">
<summary>Models, hardware, and benchmark setup</summary>

- Target: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), with 48 GDN layers and 16 standard attention layers.
- Drafter: [RadixArk's DSpark v2](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/tree/b9a5dbdf03bc999c6c73c426b19c2d9041cea393). Its training block size is 16, but its serving default is 7, so we explicitly selected K=15. We used the published weights without further training.
- Hardware: two B300 GPUs per method, TP2, BF16 for both models. Eight GPUs in total, including the separate diagnostic/profile pair.
- Code: [555978f8cf](https://github.com/vllm-project/vllm/commit/555978f8cf2cd21b22adba5d09d5f6e9df357714) (PR #60229 snapshot from 2026-10-27), without local source changes.
- Workload: [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) `throughput_1k`, measured with AIPerf 0.13.0. Temperature 0, thinking disabled, natural EOS, output capped at 1,024 tokens.

We tested concurrency 1, 16, 64, and 128, with three repeats per setting. We kept each method's server running across its repeats to avoid repeated startup costs, and disabled prefix caching so later runs would not benefit from earlier prompts. Fixed K used GPUs 0–1, target-only used 2–3, and AV used 4–5 concurrently, using the same shuffle seed of 42. Diagnostics and c1 profiles used GPUs 6–7.

All 36 runs completed successfully: 6,772 requests, with no errors, cancellations, or preemptions. We measured performance without profiling hooks. Separate runs checked which code paths executed and where GPU time was spent. To compare steady serving performance, we left out the initial fill and final drain.

</details>

<details markdown="1">
<summary>Correctness and execution checks</summary>

| GSM8K, first 100 test questions, five-shot | Correct |
| ------------------------------------------ | ------: |
| Target-only                                |  99/100 |
| Fixed K=15                                 |  99/100 |
| AV, K=15                                   |  99/100 |

We also needed to check that the longer verification path still produced sensible answers. All three methods gave the same numeric answer on every one of the 100 questions. Fixed SD and AV had identical text on 81 questions. None were invalid or hit the output cap. This is a small correctness check.

To check the original implementation question, we recorded FULL graph replay with query lengths up to 16, including requests of different lengths in the same batch. The trace confirmed the Triton GDN fallback. We also exercised mixed prefill/decode batches, and all 65 GDN metadata tests passed. The fused CUDA kernel's width guard remained in place throughout.

</details>

<details markdown="1">
<summary>Reproduction commands and data</summary>

The commands and pinned model revisions are linked below. We extended the graph capture sizes because K=15 creates larger target and drafter batches. Without enough graph coverage, eager execution could obscure the comparison.

- [Server commands, software versions, pinned model downloads, and benchmark instructions](../assets/research/dspark-gdn-block16/vllm_command.txt)
- [Exact vLLM source commit](https://github.com/vllm-project/vllm/commit/555978f8cf2cd21b22adba5d09d5f6e9df357714) (PR #60229 snapshot from 2026-10-27, without local source changes)
- AIPerf configs: [target-only](../assets/research/dspark-gdn-block16/baseline-sweep.yaml), [fixed K=15](../assets/research/dspark-gdn-block16/fixed-sweep.yaml), [AV](../assets/research/dspark-gdn-block16/adaptive-sweep.yaml)
- GSM8K: [evaluation command](../assets/research/dspark-gdn-block16/eval_command.txt) and [evaluation client](../assets/research/dspark-gdn-block16/eval_gsm8k.py.txt) (save as `eval_gsm8k.py`)
- [All 36 per-run measurements](../assets/research/dspark-gdn-block16/measurements.csv)
- C1 profiles: [CUDA-event stage timings](../assets/research/dspark-gdn-block16/c1-stage-timings.csv) and [Nsight GDN recurrence timings](../assets/research/dspark-gdn-block16/c1-gdn-timings.json). CPU and GPU stage durations overlap and must not be added together.

</details>

<details markdown="1">
<summary>Measurement and software details</summary>

We used AIPerf's native SPEED-Bench loader. Mean input lengths per run ranged from 1,062 to 1,089 tokens, and mean output lengths from 606 to 929 tokens. All runs passed checks for completion, server saturation, queueing, preemptions, and a usable steady window.

The throughput window starts at the concurrency-th request completion and ends at the last request submission. We estimate tokens inside it by distributing each request's output count evenly across its streamed chunks. TPOT is the mean request TPOT for requests submitted inside the window. The shortest window was 10.59 seconds. AL uses server counter deltas over the full client measurement interval: `1 + accepted draft tokens / speculative steps`.

GSM8K used the first five training examples as demonstrations, concurrency 16, temperature 0, seed 42, thinking disabled, and a 1,024-token cap. Runtime checks confirmed `sample_from_anchor=True`, 15 drafter query tokens per request, and confidence-head use on both TP ranks. The sampling configuration was `draft_sample_method=probabilistic` with temperature 0.

The stage timings use CUDA events with Nsight capture stopped. The kernel breakdown comes from the Nsight trace. Both profile GPUs held 2,032 MHz during the trace passes. One fixed-K event-only clock sample was 1,897 MHz. Client TPOT differed by less than 0.3% with Nsight capture on versus off. GDN ran 48 recurrence calls per GPU per decode step, averaging 23.10 µs each without AV and 9.96 µs with AV.

Software: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, Transformers 5.17.0, Triton 3.7.1, AIPerf 0.13.0. The recorded Python source ran with precompiled vLLM extensions `0.30.1rc1.dev17+g8d3f2b65c.precompiled`. Each B300 SXM6 reported 275,040 MiB of memory, with driver 610.43.02. We did not use Speculators to train or serve these checkpoints.

</details>

<details markdown="1">
<summary>Additional measurement limitations</summary>

- Output lengths and the requests included in each measurement differ, so throughput and mean TPOT are not exact reciprocals at a fixed nominal concurrency.
- The model revisions are pinned, but we did not generate weight checksum manifests or record the cached SPEED-Bench dataset's revision and checksum.
- Fixed-K GPU clocks dropped under high-concurrency load (down to 1,485 MHz at c64), while the target-only and AV pairs stayed at 2,032 MHz.

</details>
