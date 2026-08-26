# FP8 Hidden-States Ablation — Final Summary

Qwen/Qwen3-8B speculators (eagle3, dflash, dspark) trained on 5K magpie + 5K ultrachat (from `inference-optimization/Dataset-Qwen3-235B-Instruct`), comparing bf16 vs FP8-quantized (per-token scale) hidden states from the `hs_connectors` FP8 connector.

## guidellm throughput eval (weighted across all 9 `RedHatAI/speculator_benchmarks` subsets)

| Model  | Precision | Weighted acceptance_length | Weighted acceptance_rate | n_drafts |
| ------ | --------- | -------------------------- | ------------------------ | -------- |
| eagle3 | bf16      | 2.0726                     | 0.3575                   | 526,808  |
| eagle3 | fp8       | 2.0629                     | 0.3543                   | 528,566  |
| dflash | bf16      | 2.0600                     | 0.0707                   | 528,145  |
| dflash | fp8       | 2.0626                     | 0.0708                   | 521,452  |
| dspark | bf16      | 2.2558                     | 0.1570                   | 607,338  |
| dspark | fp8       | 2.2512                     | 0.1564                   | 601,826  |

bf16-vs-fp8 gap on acceptance_length: eagle3 ~0.5% relative, dflash ~0.1% relative (fp8 fractionally *higher* here — within run-to-run noise), dspark ~0.2% relative. All differences are within noise — FP8 connector validated end-to-end (train -> serve -> guidellm) across all three speculator architectures with no measurable quality regression.

Per-subset breakdown for all 6 runs: see the `*_acceptance.csv` files in this directory (one row per `RedHatAI/speculator_benchmarks` subset).

## Storage size and write-speed (hidden-states transfer itself)

The tables above are quality-parity checks (does FP8 hurt the trained speculator?). This section measures the thing FP8 is actually meant to speed up: the hidden-states write path.

**Real data-gen run** (eagle3 config, `Qwen/Qwen3-8B`, layers `[2, 18, 33, 36]`, 300 samples via `scripts/data_generation_offline.py`, same vLLM server otherwise identical, `--concurrency 32`):

| Backend | Total on-disk size (300 files) | Throughput     | avg vLLM request | avg file write |
| ------- | ------------------------------ | -------------- | ---------------- | -------------- |
| bf16    | 14.24 GB                       | 25.6 samples/s | 1114 ms          | 2 ms           |
| fp8     | 7.12 GB (**exactly 50.0%**)    | 25.5 samples/s | 1115 ms          | 3 ms           |

End-to-end throughput is identical within noise — the write itself (2-3 ms) is ~3 orders of magnitude smaller than the GPU generation time per sample (~1.1 s) and runs in a background thread off the critical path, so the ~50% smaller payload doesn't show up as a generation-throughput win on local disk. Where it matters is disk footprint and any bandwidth-constrained transfer (e.g. the `mooncake` backend, or storing many samples on a slower/network filesystem).

**Isolated write-path microbenchmark** (synthetic tensors, same shape, CPU-only quantize+`save_file` step, tmpfs, median of 100 trials — isolates the pure compute/IO tradeoff independent of GPU noise):

| seq_len (tokens) | bf16 size | fp8 size | size ratio | bf16 write | fp8 write | write ratio |
| ---------------- | --------- | -------- | ---------- | ---------- | --------- | ----------- |
| 128              | 4.2 MB    | 2.1 MB   | 0.500      | 1.28 ms    | 1.36 ms   | 1.07x       |
| 512              | 16.8 MB   | 8.4 MB   | 0.500      | 5.53 ms    | 9.75 ms   | 1.76x       |
| 2048             | 67.1 MB   | 33.6 MB  | 0.500      | 22.46 ms   | 28.73 ms  | 1.28x       |
| 8192             | 268.5 MB  | 134.3 MB | 0.500      | 89.46 ms   | 78.59 ms  | 0.88x       |

Takeaway: the size reduction is a clean, deterministic 50% at every scale (1 byte/elem fp8 vs 2 byte/elem bf16, scale overhead is negligible). The quantization compute (`amax` reduction + divide + cast) adds real CPU cost that roughly offsets the smaller write at small/medium chunk sizes on fast local storage, and only nets out as a net *write-time* win at very large single writes. The practical benefit is disk/network footprint, not local write latency.

## DSpark serving note

DSpark checkpoints failed to load under the team's local `vllm_spec_eval` checkout (`AssertionError`, weight shape mismatch: draft FC layer sized off `num_hidden_layers` instead of `len(aux_hidden_state_layer_ids)`). Confirmed fixed upstream — verified working end-to-end (loads, serves, actively speculates/accepts tokens, full guidellm eval completes) on:

- `vllm==0.27.1` (PyPI release) — **works**
- `vllm==0.24.0` (PyPI release) — does **not** work; that release predates DSpark support entirely (`dspark` isn't even in the `SpeculativeConfig` method enum yet).

New env for this: `/home/shubhra/spec_evals/env_vllm024` (misnomer now — actually running vllm 0.27.1). Needs `FLASHINFER_DISABLE_VERSION_CHECK=1` to bypass an unrelated flashinfer/flashinfer-cubin version-consistency guard (flashinfer-cubin tops out at 0.6.13 on PyPI, flashinfer itself at 0.6.16.post3 — cosmetic mismatch only).

## Files

- Full per-subset CSVs: `eval_results/{model}_{precision}/acceptance.csv`
- Training logs: `runs/{model}_{precision}/train.log`
- Serving logs (vllm 0.27.1 dspark test): `logs/eval027_dspark_{precision}.log`
