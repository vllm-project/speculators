---
title: "Static draft depth across batch and context on Gemma-4 31B FP8 (@seongyun1104)"
tags:
  - Draft model
  - Gemma4
  - vLLM
---

<div class="entry-byline" markdown>
[![seongyun1104](https://github.com/seongyun1104.png?size=64){ .entry-avatar }](https://github.com/seongyun1104)
**[@seongyun1104](https://github.com/seongyun1104)** posted on **2026-11-11**
</div>

# Static draft depth across batch and context on Gemma-4 31B FP8 (@seongyun1104)

## Summary

Under the pre-specified decision rule, we found no resolved reversal in draft-depth rankings across context lengths at any tested batch size, using delivered milliseconds per committed token (median per-request TPOT divided by batch size). At batch 32, end-to-end output throughput ranked K3 above K5 at 1k and K5 above K3 at 2k in both launches. We did not apply the reversal rule to throughput.

The question was whether, at a fixed batch size, the best static draft depth K changes with context length. That would be the case for a context axis in the K schedule ([vllm-project/vllm#48627](https://github.com/vllm-project/vllm/issues/48627)). Measured at temperature 0 on random-token prompts; see Observations for what that does to acceptance.

## Setup

- Target model: [`prithivMLmods/gemma-4-31B-it-qat-FP8`](https://huggingface.co/prithivMLmods/gemma-4-31B-it-qat-FP8), revision `1245be0199c1b792597ed7ea1ac5897269e66a89`, FP8 (compressed-tensors W8A8). The weights byte count was checked before the run.
- Draft model: [`google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant), revision `96d4c8ca3cb38c107a8478587878124895d1e844`, unquantized as published. The weights byte count was checked before the run.
- Algorithm: vLLM draft-model speculation with static `num_speculative_tokens` K.
- Knob under test: K ∈ {0, 1, 3, 5}, plus a target-only baseline.
  - `OFF`: no drafter (target-only baseline).
  - `Z`: drafter loaded, K = 0 at every batch size, via `num_speculative_tokens_per_batch_size [[1, 64, 0]]`.
  - `K1`, `K3`, `K5`: static K.
  - Everything else is identical across arms: target, drafter, attention backend, prefix caching, max model length, max sequences, memory utilization, and workload.
- Evaluation: `vllm bench serve`, random dataset, input length = context (1,024 / 2,048 / 4,096 / 16,384 tokens), output 640 tokens with `ignore_eos`, temperature 0 (greedy). `num_prompts` = `max_concurrency` = batch size, request rate `inf`. Batch sizes 1, 8, 16 and 32, with 10 (batch, context) shapes.
- Serving:
  - vLLM `7ecc83a9877552a5f4be31c133ab8502dd40f50e` (precompiled wheel `vllm-0.33.0rc1.dev784+g7ecc83a98`) plus `bs_ctx_bd09764c45.patch` (applied tree `eeb37ef757`).
  - Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`), TP 1, `--attention-backend TRITON_ATTN`, `--no-enable-prefix-caching`, `--max-num-seqs 64`, `--max-model-len 34816`, `--gpu-memory-utilization 0.90`.
  - Launch command for every arm: `cmd.json` in each launch directory (linked under Reproduction).
- Software: Speculators not used. torch 2.14.0+cu130, transformers 5.22.1.
- Hardware: 1× NVIDIA H100 NVL, 95,830 MiB, driver 595.91.07. One host for all arms.
- Launches: each config launched twice in the registered order K3, K1, K5, OFF, Z, Z, OFF, K5, K1, K3, with all 10 shapes per launch. Every launch completed and passed its gates.

## Reproduction

All files below are in [DepthChart at commit `173f63ae13`](https://github.com/seongyun1104/depthchart/tree/173f63ae138c8fdb369ab88b7eb9cf29529e28c1/mrv2_ctx_20261105). Paths are relative to `mrv2_ctx_20261105/`, and the full write-up is [`public_20261119/REPORT.md`](https://github.com/seongyun1104/depthchart/blob/173f63ae138c8fdb369ab88b7eb9cf29529e28c1/mrv2_ctx_20261105/public_20261119/REPORT.md).

- Pre-registration: `costmap_20261107/STAGE1_R2_PREREG.md` (tag `costmap-stage1-r2-prereg-20261111`). It keeps §1–§7 of `costmap_20261107/STAGE1_ODD_PREREG.md`, which defines the metrics and makes delivered ms per token primary.
- Raw outputs (bench JSON per cell, MRv2 step traces, server counters, `cmd.json` per launch, `EVIDENCE_SHA256SUMS`): `costmap_20261107/run_20261111r2/evidence/costmap_out/`.
- `vllm.patch` equivalent: `smoke/box/bs_ctx_bd09764c45.patch`, sha256 `85f25f7e…6278d7`, applied to `7ecc83a9`.
- GPU packet: `bash public_20261119/build_r2_packet.sh` rebuilds the 24-file r2 packet from the public repository and checks its manifest against the registered `dd8f8c69…3d4a`. The packet holds the patch, the wheel pin (`smoke/box/provision_mrv2.sh`), the model-revision pins and weight byte checks (`smoke/box/preflight_mrv2.sh`), the step-trace hook (`smoke/trace/`), the runner and the box script (`costmap_20261107/stage1/run_stage1_box.sh`). The box runs that script on an H100 NVL with the packet unpacked under `/root`; renting the box is not scripted.
- `checkpoint_sha256.txt` and `drafter_checkpoint_sha256.txt` equivalent: `public_20261119/model_files_after_r2.tsv`, written after this run for the same revisions. The run itself checked byte counts.
- `scripts/launch_vllm.py --provenance-dir` was not used; the files above stand in for its artifacts.
- Regenerate every table offline (no GPU):

```bash
cd mrv2_ctx_20261105
python costmap_20261107/run_20261111r2/analysis/export_csv.py costmap_20261107/run_20261111r2
python public_20261119/build_hub_table.py      # the Results table below
python k_rank_r2_20261119/k_rank.py            # the ranking check (rules: k_rank_r2_20261119/RULES.md)
bash public_20261119/build_r2_packet.sh /tmp/costmap_packet_r2.tar   # GPU packet; manifest must match
```

## Results

Each row is the mean of the two launches.

- Acceptance length = `1 + accepted speculative tokens / draft iterations`, from the server's spec-decode counters over the whole bench run. The verifier bonus token is included, and the ceiling is K + 1.
- Output tokens/s = total output tokens / bench duration. The window is all requests of the cell, from the first request sent to the last completion, so it includes prefill and TTFT.
- Median TPOT, mean ITL and median ITL are as reported by `vllm bench serve`. TPOT is per request and excludes the first token; ITL is over inter-token intervals.
- Throughput vs. OFF = output tokens/s over the target-only (`OFF`) output tokens/s at the same shape, both two-launch means.
- Delivered ms per token = median TPOT / batch size. This is r2's pre-registered primary metric. The ranking criterion on it was fixed before the reanalysis, not before the GPU run.
- Step period = median interval between adjacent steady decode steps in the MRv2 trace. A steady step is one where all B requests are in the batch, no row is prefilling, and every row is verified at width K (none for K = 0). Each cell needs at least 40 such intervals.
- `OFF` is the target-only baseline.

| Batch (concurrency) | Context | Config |   K |      Acceptance length | Output tokens/s | Throughput vs. OFF | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Step period (ms) | Delivered ms per token |
| ------------------: | ------: | ------ | --: | ---------------------: | --------------: | -----------------: | ---------------: | ------------: | --------------: | ---------------: | ---------------------: |
|                   1 |      1k | OFF    |   — |       N/A (no drafter) |            70.2 |              1.00x |            14.07 |         14.07 |           14.07 |            14.07 |                 14.066 |
|                   1 |      1k | Z      |   0 | N/A (K = 0, no drafts) |            67.6 |              0.96x |            14.61 |         14.61 |           14.61 |            14.61 |                 14.614 |
|                   1 |      1k | K1     |   1 |                   1.99 |           132.8 |              1.89x |             7.35 |         14.63 |           14.60 |            14.59 |                  7.350 |
|                   1 |      1k | K3     |   3 |                   3.94 |           239.2 |              3.41x |             3.99 |         15.76 |           15.71 |            15.70 |                  3.995 |
|                   1 |      1k | K5     |   5 |                   5.86 |           326.6 |              4.65x |             2.87 |         16.84 |           16.78 |            16.78 |                  2.873 |
|                   1 |      4k | OFF    |   — |       N/A (no drafter) |            64.0 |              1.00x |            14.77 |         14.77 |           14.75 |            14.75 |                 14.769 |
|                   1 |      4k | Z      |   0 | N/A (K = 0, no drafts) |            61.6 |              0.96x |            15.38 |         15.38 |           15.33 |            15.33 |                 15.381 |
|                   1 |      4k | K1     |   1 |                   2.00 |           115.7 |              1.81x |             7.77 |         15.51 |           15.44 |            15.44 |                  7.769 |
|                   1 |      4k | K3     |   3 |                   3.94 |           192.9 |              3.02x |             4.30 |         16.86 |           16.73 |            16.72 |                  4.301 |
|                   1 |      4k | K5     |   5 |                   5.78 |           248.8 |              3.89x |             3.15 |         18.11 |           17.93 |            17.92 |                  3.145 |
|                   1 |     16k | OFF    |   — |       N/A (no drafter) |            44.4 |              1.00x |            17.37 |         17.37 |           17.29 |            17.28 |                 17.365 |
|                   1 |     16k | Z      |   0 | N/A (K = 0, no drafts) |            42.6 |              0.96x |            18.30 |         18.30 |           18.04 |            18.05 |                 18.297 |
|                   1 |     16k | K1     |   1 |                   1.98 |            68.5 |              1.54x |             9.37 |         18.54 |           18.15 |            18.15 |                  9.374 |
|                   1 |     16k | K3     |   3 |                   3.83 |            94.2 |              2.12x |             5.41 |         20.69 |           20.04 |            20.02 |                  5.406 |
|                   1 |     16k | K5     |   5 |                   5.54 |           107.8 |              2.43x |             4.09 |         22.52 |           21.60 |            21.58 |                  4.089 |
|                   8 |      1k | OFF    |   — |       N/A (no drafter) |           428.0 |              1.00x |            17.21 |         17.38 |           17.25 |            17.25 |                  2.152 |
|                   8 |      1k | Z      |   0 | N/A (K = 0, no drafts) |           410.6 |              0.96x |            17.96 |         17.95 |           18.02 |            18.04 |                  2.244 |
|                   8 |      1k | K1     |   1 |                   1.99 |           746.9 |              1.75x |             9.07 |         18.03 |           18.05 |            18.06 |                  1.134 |
|                   8 |      1k | K3     |   3 |                   3.89 |          1040.6 |              2.43x |             5.87 |         22.84 |           22.77 |            22.76 |                  0.734 |
|                   8 |      1k | K5     |   5 |                   5.69 |          1317.4 |              3.08x |             4.16 |         24.77 |           23.72 |            23.78 |                  0.521 |
|                   8 |      4k | OFF    |   — |       N/A (no drafter) |           313.5 |              1.00x |            20.09 |         20.54 |           18.38 |            18.39 |                  2.511 |
|                   8 |      4k | Z      |   0 | N/A (K = 0, no drafts) |           295.6 |              0.94x |            21.46 |         21.92 |           19.61 |            19.61 |                  2.682 |
|                   8 |      4k | K1     |   1 |                   1.99 |           463.5 |              1.48x |            11.69 |         24.10 |           19.55 |            19.55 |                  1.461 |
|                   8 |      4k | K3     |   3 |                   3.89 |           570.0 |              1.82x |             8.38 |         34.13 |           25.22 |            25.20 |                  1.048 |
|                   8 |      4k | K5     |   5 |                   5.61 |           652.3 |              2.08x |             6.60 |         39.47 |           26.83 |            26.83 |                  0.825 |
|                   8 |     16k | OFF    |   — |       N/A (no drafter) |           122.3 |              1.00x |            39.83 |         40.27 |           23.78 |            23.79 |                  4.979 |
|                   8 |     16k | Z      |   0 | N/A (K = 0, no drafts) |           116.1 |              0.95x |            41.96 |         42.48 |           25.09 |            25.08 |                  5.245 |
|                   8 |     16k | K1     |   1 |                   1.99 |           143.5 |              1.17x |            29.19 |         59.05 |           25.22 |            25.20 |                  3.649 |
|                   8 |     16k | K3     |   3 |                   3.90 |           153.3 |              1.25x |            25.33 |        100.60 |           34.01 |            33.97 |                  3.167 |
|                   8 |     16k | K5     |   5 |                   5.50 |           159.1 |              1.30x |            23.79 |        131.58 |           38.89 |            38.95 |                  2.974 |
|                  16 |      1k | OFF    |   — |       N/A (no drafter) |           659.1 |              1.00x |            21.88 |         21.97 |           21.26 |            21.27 |                  1.367 |
|                  16 |      1k | Z      |   0 | N/A (K = 0, no drafts) |           628.0 |              0.95x |            23.04 |         23.14 |           22.39 |            22.40 |                  1.440 |
|                  16 |      1k | K1     |   1 |                   1.86 |           701.4 |              1.06x |            13.75 |         25.49 |           24.67 |            24.73 |                  0.859 |
|                  16 |      1k | K3     |   3 |                   3.24 |           730.8 |              1.11x |             7.02 |         25.72 |           26.91 |            27.06 |                  0.439 |
|                  16 |      1k | K5     |   5 |                   4.37 |           702.9 |              1.07x |             7.10 |         32.27 |           32.08 |            32.37 |                  0.444 |
|                  16 |      4k | OFF    |   — |       N/A (no drafter) |           427.2 |              1.00x |            28.28 |         28.49 |           22.93 |            22.94 |                  1.767 |
|                  16 |      4k | Z      |   0 | N/A (K = 0, no drafts) |           413.2 |              0.97x |            29.38 |         29.59 |           23.89 |            23.90 |                  1.836 |
|                  16 |      4k | K1     |   1 |                   1.86 |           447.8 |              1.05x |            19.77 |         36.41 |           26.42 |            26.44 |                  1.236 |
|                  16 |      4k | K3     |   3 |                   3.26 |           454.9 |              1.06x |            14.50 |         46.60 |           30.37 |            30.51 |                  0.906 |
|                  16 |      4k | K5     |   5 |                   4.23 |           441.4 |              1.03x |            13.42 |         55.65 |           36.42 |            36.69 |                  0.839 |
|                  32 |      1k | OFF    |   — |       N/A (no drafter) |           874.1 |              1.00x |            31.74 |         31.84 |           30.54 |            30.54 |                  0.992 |
|                  32 |      1k | Z      |   0 | N/A (K = 0, no drafts) |           849.0 |              0.97x |            32.95 |         33.37 |           31.56 |            31.57 |                  1.030 |
|                  32 |      1k | K1     |   1 |                   1.94 |          1239.6 |              1.42x |            17.46 |         34.72 |           31.60 |            31.61 |                  0.546 |
|                  32 |      1k | K3     |   3 |                   3.37 |          1264.3 |              1.45x |            11.19 |         40.65 |           38.60 |            38.70 |                  0.350 |
|                  32 |      1k | K5     |   5 |                   4.47 |          1204.8 |              1.38x |             9.97 |         50.28 |           50.16 |            50.43 |                  0.311 |
|                  32 |      2k | OFF    |   — |       N/A (no drafter) |           716.1 |              1.00x |            36.83 |         36.91 |           31.69 |            31.69 |                  1.151 |
|                  32 |      2k | Z      |   0 | N/A (K = 0, no drafts) |           678.0 |              0.95x |            39.02 |         39.09 |           33.52 |            33.53 |                  1.219 |
|                  32 |      2k | K1     |   1 |                   1.98 |          1029.7 |              1.44x |            22.07 |         43.82 |           33.11 |            33.11 |                  0.690 |
|                  32 |      2k | K3     |   3 |                   3.78 |          1228.1 |              1.71x |            14.40 |         59.17 |           41.16 |            41.16 |                  0.450 |
|                  32 |      2k | K5     |   5 |                   5.40 |          1248.4 |              1.74x |            14.59 |         80.45 |           53.84 |            53.92 |                  0.456 |

Best option by delivered ms per committed token, among Z, K1, K3 and K5. A reversal had to be resolved by r2's launch-spread rule at both contexts; `≈` means the two options were not separated once the variation between the two launches was taken into account. — = not measured.

| Batch | 1k      | 2k      | 4k  | 16k |
| ----- | ------- | ------- | --- | --- |
| 1     | K5      | —       | K5  | K5  |
| 8     | K5      | —       | K5  | K5  |
| 16    | K3 ≈ K5 | —       | K5  | —   |
| 32    | K5      | K3 ≈ K5 | —   | —   |

No pair of options was resolved in opposite orders at two contexts of the same batch, and no batch changed its resolved best option with context.

## Observations

- **Throughput ranks differently from delivered ms per token.** By end-to-end output tokens/s, K3 was first at batch 16 in both launches at both 1k and 4k. At batch 32, the K5/K3 throughput ratio was 0.937 and 0.969 at 1k and 1.023 and 1.010 at 2k (launch 0, launch 1), so K3 and K5 swapped observed order between the two contexts in both launches. This is an observed crossing of two options, not a resolved change of best K:
  - the reversal rule was not applied to throughput;
  - the best option by throughput at b32·1k differed between launches (K3, then K1);
  - throughput includes prefill and TTFT, whose share changes with context. We note it rather than test it after the fact; any rule applied now would be exploratory.
- **Greedy acceptance is an optimistic reference.** All numbers are at temperature 0 (greedy) on random-token prompts with `ignore_eos`. Acceptance length here was 1.86–2.00 at K1, 3.24–3.94 at K3 and 4.23–5.86 at K5.
  - Greedy acceptance is generally higher than sampled acceptance. These values are therefore an optimistic reference for what this target/drafter pair would reach with sampling on these prompts at these K.
  - Sampling was not measured.
  - Higher acceptance favours large K, so with sampling or real prompts the ranking could move. This run cannot say which way.
- **K5 is the largest K tested**, so "best = K5" may sit at the edge of the grid. At batch 8, K5's margin over K3 by delivered ms per token narrows with context (log ratio 0.34 → 0.24 → 0.06), but does not reverse inside the grid.
- **Earlier campaign, different setup, not pooled.** In July 2026 (vLLM 0.27.1, V1 runner, concurrency 256), a batch × context schedule met its pre-registered bar against a batch-only schedule (1.29–1.36× at three of four prefix lengths). Against static K=3 it showed no consistent throughput advantage: −11.64 %, +1.94 %, +1.42 % and −0.73 % at prefix 400 / 900 / 1,900 / 4,000. The full table and its corrections are in DepthChart's [`EVIDENCE_LEDGER.md`](https://github.com/seongyun1104/depthchart/blob/173f63ae138c8fdb369ab88b7eb9cf29529e28c1/EVIDENCE_LEDGER.md).
- **Step-cost side result.** In r2, K1 at batch 16 took 10.4–10.6% longer per step than Z (K = 0 with the drafter loaded). In the follow-up server comparison, all six launches recorded 32 target graph tokens for that cell. Applying the (16, 32\] branch of [vllm-project/vllm#66852](https://github.com/vllm-project/vllm/pull/66852) lowered that step by 7.6 % at 1k and 6.5 % at 4k on one H100 NVL, while eight control cells stayed within ±0.3 %. This is a reproduction credited to #66852. Details: [`public_20261119/REPORT.md`](https://github.com/seongyun1104/depthchart/blob/173f63ae138c8fdb369ab88b7eb9cf29529e28c1/mrv2_ctx_20261105/public_20261119/REPORT.md).
- **Not tested:**
  - batch 64 and above, K2, K4, or K > 5;
  - contexts beyond 16k, or 2k and 16k at every batch;
  - sampling (temperature > 0) or real prompts;
  - other model sizes or drafters, other GPUs, TP > 1;
  - adaptive or dynamic K;
  - whether a context-aware schedule beats a well-tuned batch-only schedule and the best static K. The implementation PR [vllm-project/vllm#48944](https://github.com/vllm-project/vllm/pull/48944) is not supported or refuted by this entry beyond the absence of a reversal here.
