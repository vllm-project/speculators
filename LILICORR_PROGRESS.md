# LiLiCorr Implementation Progress

Intermediate status of the LiLiCorr (arXiv:2608.20530) training support on the `lilicorr` branch. Implementation is complete and validated end-to-end (training → export → vLLM serving); scale-up runs are in progress.

## What's implemented

- `src/speculators/models/lilicorr/` — LiLiCorr draft model: DFlash backbone + candidate-lattice reranker head (K=8, 2-layer/8-head transformer over the top-8 lattice, logit scale 8.0), joint loss = DFlash block loss + 0.25·CE + 0.25·penalty (paper's `base` variant).
- `examples/train/lilicorr_qwen3_8b_online.sh` — paper-recipe training example (Qwen3-8B verifier, 5 draft layers fed by verifier layers 2/10/18/26/34, block_size 16, 512 anchors / 3072-token seq, fixed-exp-decay γ=7, lr 6e-4, cosine + 4% warmup, full verifier vocabulary for vLLM serving).
- `tests/` — unit tests for the head + `tests/integration/models/test_lilicorr_vllm_parity.py`, which pins the checkpoint contract against vLLM's serving-side head (submodule names, feature order, attention-bias geometry) to 1e-5.

Serving uses vLLM's existing `LiLiCorrDraftModel` (`method: dflash`, `draft_sample_method: greedy`, 15 speculative tokens). Checkpoints are written in vLLM naming directly; no d2t, full target vocab.

## Results so far

### 5k-sample run (Qwen3-8B, tutorial_regen conversations, 90/10 split)

5000 samples → 4347 sequences after the 3072-token filter; global batch 2 on 2×A100-80GB (~1.07 s/step) with a 2-GPU vLLM data-parallel server for online hidden-state extraction. **Paused mid-epoch 3 of 6** (checkpoints for epochs 0–2 saved; resume = re-run the example, it reloads the latest checkpoint).

Validation (held-out 10%), teacher-forced:

| epoch | pos-1 acc | pos-2 acc | pos-3 acc | backbone eal | lilicorr_eal | top1_eal (argmax) |
| ----- | --------- | --------- | --------- | ------------ | ------------ | ----------------- |
| 0     | 44.6%     | 26.7%     | 17.9%     | 1.673        | 1.799        | 1.674             |
| 1     | 52.8%     | 32.2%     | 21.6%     | 1.867        | 2.026        | 1.868             |
| 2     | 57.8%     | 36.1%     | 24.3%     | 2.009        | 2.193        | 2.011             |

The reranker path holds a ~9% EAL edge over per-slot argmax.

### Served acceptance (epoch-1 checkpoint, vLLM, greedy)

250 GSM8K test questions × 256 tokens (chat mode, thinking traces), 33k draft rounds of 15 tokens:

| position         | 1     | 2     | 3    | 4    | 5    |
| ---------------- | ----- | ----- | ---- | ---- | ---- |
| acceptance/round | 54.9% | 23.7% | 8.7% | 3.5% | 1.3% |

Mean acceptance length **1.93** (= 1 + Σ per-position rates, arithmetic checks exactly). Served position-1 (54.9%) agrees with teacher-forced val (52.8%), so the serving path is faithful. For comparison, the earlier 300-sample smoke run served AL was 1.25 (pos-2 2.9%) on its 8-prompt set.

## Known deltas vs the NVIDIA ModelOpt reference trainer

- Draft in-block attention: speculators trains causal-in-block on sliding-window draft layers; ModelOpt's lilicorr recipe defaults to bidirectional-in-block with full context. ModelOpt checkpoints can still be served in vLLM by setting `layer_types=["full_attention"]*5` in the converted draft config.
- Optimizer: Muon+AdamW hybrid (speculators) vs fp32-master-weight AdamW (ModelOpt recipe).
- Supervision masking: pre-tokenized loss_mask vs chat-template answer_only_loss (~6 tokens/record difference).
- A side-by-side comparison run of both trainers on the same 5k data is in progress.

## Artifacts

Training checkpoints/logs live outside git (2.1 GB per epoch checkpoint): container-local under `/tmp/lilicorr_check/full/checkpoints/{0,1,2}/` (val metrics in `val_metrics.json` per epoch). Reproduce with `examples/train/lilicorr_qwen3_8b_online.sh` (see header for GPU layout).
