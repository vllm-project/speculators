---
title: "DSpark acceptance on Mistral Large 4 NVFP4: bundled drafter accepts ~2.2-2.5 of 3 draft tokens"
tags:
  - DSpark
  - Mistral Large 4
  - NVFP4
  - vLLM
---

<div class="entry-byline" markdown>
[![MeganEFlynn](https://github.com/MeganEFlynn.png?size=64){ .entry-avatar }](https://github.com/MeganEFlynn)
**[@MeganEFlynn](https://github.com/MeganEFlynn)** posted on **2026-10-30**
</div>

# DSpark acceptance on Mistral Large 4 NVFP4: bundled drafter accepts ~2.2–2.5 of 3 draft tokens

## Summary

Serving `mistralai/Mistral-Large-4-1T-A52B-NVFP4` (1T-A52B MoE, day-0 vLLM support) with its bundled DSpark drafter at depth 3 on 4 × B300, the measured acceptance length is 3.16 (HumanEval) and 3.31 (math_reasoning) at temperature 1.0 / top-p 0.95, and 3.33 / 3.48 greedy. The drafter's trained depth is exactly 3; requesting `num_speculative_tokens=8` prevents the server from starting.

## Setup

- Target model: `mistralai/Mistral-Large-4-1T-A52B-NVFP4`, revision `bb6d00fe249b0e876f0cffc6b85014be02de3e68`, NVFP4 experts + FP8 attention/shared experts (568.15 GiB, 32 shards)
- Draft model: DSpark drafter bundled in the same checkpoint (`mtp_dspark.*` tensors); no separate draft path or conversion
- Algorithm: DSpark, `num_speculative_tokens=3` (the only supported depth — see Observations)
- Evaluation: `RedHatAI/speculator_benchmarks`, subsets HumanEval and math_reasoning, up to 80 requests per subset, max-concurrency 1; two sampling settings: temperature 1.0 / top-p 0.95 (Mistral-recommended) and temperature 0.0 / top-p 1.0 (greedy)
- Serving: `vllm/vllm-openai:mistral-large-4` image, `--tensor-parallel-size 4 --kv-cache-dtype fp8_ds_mla --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 8`; engine selected FLASHMLA_SPARSE + FP8 block-scaled GEMMs + NVFP4 MoE (FLASHINFER_TRTLLM)
- Software: Speculators commit `07b4fcf`, guidellm 0.8.1, client Python 3.11
- Hardware: 4 × NVIDIA B300 SXM6 AC (275 GB each), TP4

**Acceptance-length definition:** `1 + accepted_draft_tokens / drafting_steps`, i.e. it includes the single guaranteed verifier token. Maximum is 4 at draft depth 3. Per-position acceptance rates are conditional on all earlier draft positions being accepted.

## Reproduction

Server (note: the `vllm/vllm-openai` image entrypoint is already `vllm serve`, so the model and flags are passed as container args):

```bash
docker run -d --gpus '"device=0,1,2,3"' --ipc=host -v /data:/data \
  -e HF_HOME=/data/hf-home -e HF_HUB_OFFLINE=1 -p 8000:8000 \
  vllm/vllm-openai:mistral-large-4 \
  mistralai/Mistral-Large-4-1T-A52B-NVFP4 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_ds_mla --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-seqs 8 \
  --tool-call-parser mistral --enable-auto-tool-choice --reasoning-parser mistral
```

Client, from the Speculators repository root (Python 3.11+ environment with `guidellm` on `PATH`):

```bash
python scripts/evaluate/evaluate.py throughput \
  --target http://localhost:8000/v1 \
  --dataset RedHatAI/speculator_benchmarks \
  --subsets HumanEval,math_reasoning \
  --max-requests 80 --max-concurrency 1 \
  --gen-kwargs '{"temperature":1.0,"top_p":0.95}' \
  --output-dir ./results/le_chonk_nvfp4_dspark        # sampled
python scripts/evaluate/evaluate.py throughput \
  --target http://localhost:8000/v1 \
  --dataset RedHatAI/speculator_benchmarks \
  --subsets HumanEval,math_reasoning \
  --max-requests 80 --max-concurrency 1 \
  --gen-kwargs '{"temperature":0.0,"top_p":1.0}' \
  --output-dir ./results/le_chonk_nvfp4_dspark_greedy  # greedy
```

## Results

Acceptance length (includes verifier token; max 4). Target-only baseline rows are N/A: without a drafter there are no draft tokens to accept, and the `vllm:spec_decode_*` counters are absent from `/metrics`.

| Configuration | Subset | Drafts | Accepted draft tokens | Acceptance length | Pos 0 | Pos 1 | Pos 2 |
|---------------|--------|--------|-----------------------|-------------------|-------|-------|-------|
| DSpark depth 3, T=1.0 / top-p 0.95 | HumanEval | 45,688 | 98,715 | 3.161 | 0.850 | 0.713 | 0.598 |
| DSpark depth 3, T=1.0 / top-p 0.95 | math_reasoning | 10,485 | 24,253 | 3.313 | 0.883 | 0.767 | 0.664 |
| DSpark depth 3, greedy | HumanEval | 40,614 | 94,627 | 3.330 | 0.890 | 0.775 | 0.664 |
| DSpark depth 3, greedy | math_reasoning | 9,601 | 23,851 | 3.484 | 0.917 | 0.828 | 0.740 |
| Target only (no drafter) | — | N/A | N/A | N/A | N/A | N/A | N/A |

Throughput/speedup vs. a target-only baseline is being collected with `throughput_interactivity.py` (streams 1–32) and will be appended; concurrency-1 informational output rates were 232.5 tok/s (sampled) and 252.0 tok/s (greedy).

Speculator-disabled reference (same flags minus `--speculative-config`, `--max-num-seqs 32`, prefix caching off, `--max-num-batched-tokens 8192`): the evaluator confirms no `vllm:spec_decode_*` counters exist, so acceptance length is N/A by construction. On the same workload (HumanEval + math_reasoning, 80 requests per subset, concurrency 1) the target-only server produced 105.0 output tok/s at T=1.0/top-p 0.95, versus 232.5 tok/s with DSpark depth 3 — ~2.2× at concurrency 1. A controlled speedup comparison across concurrency levels (streams 1–32) follows below.

## Observations

- The bundled drafter accepts ~2.2–2.5 of its 3 drafted tokens per step at both sampling settings, i.e. depth 3 is well-matched to this target. `num_draft_tokens / num_drafts` was 3.000 in every run, matching the configured depth.
- Greedy decoding raises acceptance length by ~0.17 on both subsets versus temperature 1.0 / top-p 0.95, with per-position rates rising uniformly (math_reasoning pos-0: 0.883 → 0.917).
- math_reasoning accepts consistently better than HumanEval at every position.
- **Depth 8 is unsupported.** Launching with `--speculative-config '{"method":"dspark","num_speculative_tokens":8}'` fails during drafter load with `AssertionError: Position-specific DSpark masks need num_speculative_tokens equal to the trained depth 3, got 8` (`vllm/models/mistral_large4/nvidia/dspark.py`, `assert num_spec == self.block_size`). The drafter's position-specific mask embeddings are trained at depth exactly 3, so no other depth can be served for this checkpoint.
- Validation: per-subset counter deltas summed exactly to the server-wide `vllm:spec_decode_*` totals during the sampled run, confirming no unrelated traffic reached the server during measurement.
