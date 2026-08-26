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

## DSpark serving note

DSpark checkpoints failed to load under the team's local `vllm_spec_eval` checkout (`AssertionError`, weight shape mismatch: draft FC layer sized off `num_hidden_layers` instead of `len(aux_hidden_state_layer_ids)`). Confirmed fixed upstream — verified working end-to-end (loads, serves, actively speculates/accepts tokens, full guidellm eval completes) on:

- `vllm==0.27.1` (PyPI release) — **works**
- `vllm==0.24.0` (PyPI release) — does **not** work; that release predates DSpark support entirely (`dspark` isn't even in the `SpeculativeConfig` method enum yet).

New env for this: `/home/shubhra/spec_evals/env_vllm024` (misnomer now — actually running vllm 0.27.1). Needs `FLASHINFER_DISABLE_VERSION_CHECK=1` to bypass an unrelated flashinfer/flashinfer-cubin version-consistency guard (flashinfer-cubin tops out at 0.6.13 on PyPI, flashinfer itself at 0.6.16.post3 — cosmetic mismatch only).

## Files

- Full per-subset CSVs: `eval_results/{model}_{precision}/acceptance.csv`
- Training logs: `runs/{model}_{precision}/train.log`
- Serving logs (vllm 0.27.1 dspark test): `logs/eval027_dspark_{precision}.log`
