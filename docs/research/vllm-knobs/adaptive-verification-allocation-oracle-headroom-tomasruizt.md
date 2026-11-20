---
title: "Adaptive verification: oracle headroom of the per-request draft allocation, and why per-request uncertainty does not capture it (@tomasruizt)"
tags:
  - DFlash2
  - Adaptive verification
  - Acceptance estimation
  - Qwen3
  - vLLM
---

<div class="entry-byline" markdown>
[![tomasruizt](https://github.com/tomasruizt.png?size=64){ .entry-avatar }](https://github.com/tomasruizt)
**[@tomasruizt](https://github.com/tomasruizt)** posted on **2026-11-20**
</div>

# Adaptive verification: oracle headroom of the per-request draft allocation, and why per-request uncertainty does not capture it (@tomasruizt)

Joint work with [Shubhra Pandit](https://github.com/shubhra).

## Summary

Inside each step, adaptive verification (AV) gives its draft budget to the (request, position) slots with the highest survival probability, the running product of the online acceptance estimator's per-position probabilities ([vllm#52228](https://github.com/vllm-project/vllm/pull/52228)). We measured how many more tokens per step a perfect ranking would accept at the same budget, and therefore the same step cost. The oracle gains +0.4% to +3.3% where AV verifies 60% to 90% of drafts, and **+6.6% to +8.8%** where AV trims hard: Qwen3.8-27B at c256 (58% verified) and Qwen3-4B at c512 under the rejection-sampler logits cap (34% verified). No per-request uncertainty policy captures any of it (-0.5% to +0.1%). Even a per-request offset fitted in hindsight on the request's full history gains at most +0.2%. The gap is per draft, not per request.

![Oracle and per-request headroom vs. the fraction of drafts verified](../../assets/research/av-allocation-oracle-headroom.png)

Each point is one model, workload and concurrency. Filled: at the budget AV chose. Hollow: at a budget forced to 25%, 50% or 75% of the drafts. Gray crosses: the hindsight per-request offset at the same budgets.

## Motivation

An earlier experiment by Shubhra Pandit ([shubhra/vllm#1](https://github.com/shubhra/vllm/pull/1)) added an uncertainty bonus to the estimator's logit and saw no change in throughput or acceptance length. That bonus is one value per draft position, shared by every request, and it decays with the lifetime label count of the position. With about 100 labels per position per step it drops to ~0.09 logits after the first step, so it never changed which requests get the verification slots. Per draft position, verified counts and match rates in the two arms agree within 1 to 2%.

The open question was whether uncertainty **per request**, which changes the allocation inside a step, could help. A request that keeps getting 1 or 2 slots never reveals whether its deep drafts would have been accepted, so its estimate is never corrected. Before building such a policy we measured the upper bound.

## Setup

- Targets: `Qwen/Qwen3-4B` (BF16) and `Qwen/Qwen3.8-27B` (BF16). Drafters: `RedHatAI/Qwen3-4B-speculator.dflash2` (K=8) and `RedHatAI/Qwen3.8-27B-speculator.dflash2` (K=7), `"method":"dflash"`, AV on.
- Evaluation: `RedHatAI/speculator_benchmarks`. Either one subset per run (HumanEval, math_reasoning, writing) or all 9 subsets in one seeded shuffle ("mixed", 924 prompts), so that every batch mixes code, math, writing, QA, RAG, summarization, translation and tool calls. Temperature 0, max 1,024 output tokens, 4 x c requests per concurrency (at least 128).
- Serving: vLLM main `89e860de00` (2026-11-17) plus a trace patch, branch [`av-alloc-headroom`](https://github.com/tomasruizt/vllm/tree/av-alloc-headroom) (`3138435083`). TP1 for the 4B, TP1 and TP2 for the 27B, prefix caching off, `--max-num-seqs` equal to the highest concurrency of the run.
- CUDA graphs cover every tested step: `max_cudagraph_capture_size = max_num_seqs x (K+1)`. The Blackwell default of 1,024 tokens sends the 4B c128 verify-all step (1,152 tokens) to eager mode.
- Client: AIPerf 0.12.0, closed loop, one server per mode swept over the concurrencies.
- Software and hardware: PyTorch 2.14.0+cu132, 1x or 2x NVIDIA B200 (183 GB) per server.

### Method

1. **Trace.** For every verification step the patch logs each request's per-position acceptance probabilities (the ones AV ranks), its scheduled and admitted draft counts, the number of accepted drafts, and the budget AV chose.
2. **Verify all.** With `AV_TRACE_VERIFY_ALL=1` the server still computes AV's budget but verifies every draft, so the true accepted length `a` of each request is known. At temperature 0 a width `w` would have accepted `min(w, a)`.
3. **Replay.** At each step's budget `B`, replay allocation policies and count accepted tokens. The metric is output tokens per step (requests + accepted drafts) relative to AV's own ranking. Same `B` means the same step cost, so this is an upper bound on the throughput change.
   - `oracle`: knows `a`, accepts `min(B, sum a)`.
   - `calib`: AV's ranking with a per-request logit offset `u_r`, learned online by Newton steps from only the labels the policy itself verified (censored, like production). `calib_fast` forgets old labels (decay 0.8) to follow phases within a request.
   - `UCB`: `u_r + 1.0 x sd(u_r)`, so rarely verified requests get explored. `Thompson`: `u_r` sampled from its posterior.
   - `hindsight`: each request's offset fitted on all of its uncensored labels. Not causal, it bounds every policy that shifts one offset per request.
4. **Check the replay.** A second server runs normal AV with the same trace. Replaying AV's ranking reproduces the widths production admitted in 99.8% to 99.96% of steps in every run.

All policies stay lossless: they depend on the drafter's distribution (`q_max`) and on outcomes of earlier steps, never on the drafted token being trimmed.

## Reproduction

```bash
# Normal AV. For the oracle trace add AV_TRACE_VERIFY_ALL=1.
# AV_TRACE_MAX_CHUNK_BYTES lifts the 1GB rejection-sampler buffer (see Observations).
AV_TRACE_PATH=trace.jsonl AV_TRACE_MAX_CHUNK_BYTES=8589934592 \
vllm serve Qwen/Qwen3.8-27B --max-model-len 16384 --max-num-seqs 256 --tensor-parallel-size 2 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dflash2","num_speculative_tokens":7,"method":"dflash","enable_adaptive_verification":true}' \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --compilation-config '{"max_cudagraph_capture_size": 2048}'

vllm serve Qwen/Qwen3-4B --max-model-len 16384 --max-num-seqs 512 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --speculative-config '{"model":"RedHatAI/Qwen3-4B-speculator.dflash2","num_speculative_tokens":8,"method":"dflash","enable_adaptive_verification":true}' \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --compilation-config '{"max_cudagraph_capture_size": 4608}'

aiperf profile --model Qwen/Qwen3-4B --tokenizer Qwen/Qwen3-4B --url http://127.0.0.1:8000 \
  --endpoint-type chat --streaming --input-file mixed.jsonl --custom-dataset-type single_turn \
  --dataset-sampling-strategy sequential --concurrency 512 --request-count 2048 \
  --extra-inputs max_tokens:1024 --extra-inputs temperature:0
```

`mixed.jsonl` holds one `{"text": prompt}` per line: the `prompt` field of all 9 subsets of `RedHatAI/speculator_benchmarks`, shuffled with `random.Random(0)`.

## Results

Settings with headroom. Mixed workload, thinking on, full CUDA graph coverage. Tokens per step vs. AV's ranking at the same budget:

| Target          | c (mean running) | Budget                    | Drafts verified |          Oracle | Hindsight offset |            calib |              UCB |         Thompson |
| --------------- | ---------------: | ------------------------- | --------------: | --------------: | ---------------: | ---------------: | ---------------: | ---------------: |
| Qwen3.8-27B TP2 |         128 (86) | AV                        |             68% |           +3.3% |           +0.10% |           -0.09% |           -0.09% |           -0.29% |
| Qwen3.8-27B TP2 |        256 (128) | AV                        |             58% |       **+6.6%** |           +0.09% |           -0.12% |           -0.13% |           -0.34% |
| Qwen3-4B        |          64 (50) | AV                        |             28% |           +3.2% |           +0.05% |           -0.04% |           -0.04% |           -0.22% |
| Qwen3-4B        |         128 (97) | AV                        |             76% |           +0.5% |           +0.03% |           -0.02% |           -0.02% |           -0.08% |
| Qwen3-4B        |        256 (194) | AV                        |             76% |           +0.8% |           +0.06% |           -0.02% |           -0.02% |           -0.13% |
| Qwen3-4B        |        512 (393) | AV, cap lifted            |             68% |           +2.1% |           +0.09% |           -0.05% |           -0.07% |           -0.20% |
| Qwen3-4B        |        512 (393) | AV, production logits cap |             34% |       **+8.8%** |           +0.12% |           -0.11% |           -0.12% |           -0.41% |
| both            |        64 to 512 | forced to 50%             |             50% | +9.2% to +10.7% | +0.11% to +0.22% | -0.20% to -0.04% | -0.19% to -0.03% | -0.53% to -0.35% |

Earlier runs, which do not change the conclusion:

- Qwen3-4B, one subset per run, thinking on, c16 to c64 (default graphs): oracle +0.4% to +1.3% at c32 to c64 (AV verifies 63% to 90%), +3.9% to +6.6% at c16 (24% to 49%). Per-request policies -0.3% to +0.06%, hindsight at most +0.12%.
- Qwen3-4B and Qwen3.8-27B (TP1), mixed, thinking off, c32 to c128: oracle +1.1% to +3.0%, hindsight at most +0.08%.
- Qwen3.8-27B TP1, mixed, thinking on, c64 to c256: oracle +0.6% to +1.2%. The KV cache saturates at 49 to 56 running requests on average, so AV verifies 89% to 94% of drafts.

Why per-request information does not help. Residual = observed minus predicted acceptance on the uncensored labels:

| Workload             | sd of a request's mean residual | Split-half correlation of a request's residual | Step-to-step correlation |
| -------------------- | ------------------------------: | ---------------------------------------------: | -----------------------: |
| Qwen3-4B, one subset |                  0.014 to 0.033 |                                   0.10 to 0.77 |           0.014 to 0.052 |
| Qwen3-4B, mixed      |                  0.027 to 0.046 |                                   0.24 to 0.68 |           0.027 to 0.045 |
| Qwen3.8-27B, mixed   |                  0.024 to 0.027 |                                   0.06 to 0.28 |           0.011 to 0.025 |

Requests do have a stable offset (split-half correlation up to 0.7 on mixed and writing batches), but it is a few hundredths in probability, too small to reorder the slots that decide the budget. Which draft gets rejected changes from step to step.

## Observations

- **Headroom tracks how hard AV trims.** At 60% to 90% verified, AV's ranking keeps 96% to 99.5% of the tokens the oracle accepts. The oracle gap peaks near a 50% budget (+9% to +11%) and shrinks again at 25%, where every policy only verifies the first, mostly accepted, positions.
- **The rejection-sampler buffer caps the AV budget.** AV never verifies more than `2^30 / (vocab x 4)` logits per step minus one per request: 1,766 for Qwen3-4B (152K vocab) and 1,081 for Qwen3.8-27B (248K vocab). For the 4B at c512 this cap, not the cost model, sets the budget to 34% of drafts. For the 27B it binds from about 130 running requests. It also truncates a verify-all trace, so for the oracle we lifted it (`AV_TRACE_MAX_CHUNK_BYTES`) and reapplied it offline.
- **Censoring does not bias the production estimator.** On the drafts AV verifies, predicted and observed acceptance agree within 0.006 at every position (4B, c512), and its mean predictions differ from the verify-all estimator's by at most 0.02 at deep positions. So there is no blind spot for exploration to fix.
- **The estimator already extracts what `q_max` offers.** A per-position 20-bin recalibration of the same feature, fitted at c32 and c64 and tested at c128, changes tokens per step by -0.06% to +0.02%, while the oracle has +0.4% to +9.7% at the same budgets.
- **The replay keeps each request's drafts as they happened under verify-all.** A different allocation would change what gets drafted later. It also measures tokens per step, not throughput: an end-to-end gain would come on top of AV's existing cost model.
- **Next:** new lossless per-draft features, evaluated on these traces in the settings with headroom. Candidates: the target's entropy at the last accepted position (known before drafting), and the drafter's entropy and top-2 margin combined with `q_max` in one logistic model. [vllm#52228](https://github.com/vllm-project/vllm/pull/52228) compared them one at a time.
- **Not tested:** temperature > 0, H100 or other GPUs, DFlash, DSpark and its trained confidence head, MoE targets, and end-to-end throughput of any new policy.

## References

- [vllm-project/vllm#52228](https://github.com/vllm-project/vllm/pull/52228) (online acceptance estimator), [vllm-project/vllm#51303](https://github.com/vllm-project/vllm/issues/51303) (adaptive verification tracking issue)
- [shubhra/vllm#1](https://github.com/shubhra/vllm/pull/1) (check-count uncertainty term)
- [DFlash adaptive verification on Qwen3.5-35B-A3B: CUDA graph capture sizes](dflash-adaptive-verification-cuda-graphs-tomasruizt.md)
