# A/B test: LiLiCorr training code regression check (Qwen3.5-4B, 100k subset)

**Date:** 2026-10-29 → 2026-10-30 **Question:** Did a code change between the successful Qwen3.5-4B full-scale run (finished 2026-10-21) and the Qwen3.8-27B runs cause the 27B draft to train worse than the 4B draft?

## Setup

Two training runs, identical in every respect except the speculators code tree:

|               | run-old                                                | run-new                                            |
| ------------- | ------------------------------------------------------ | -------------------------------------------------- |
| Commit        | `377ee2f` (state at 4B full-scale completion)          | `b4830c0` (state used by the 27B fresh-argmax run) |
| Verifier      | Qwen3.5-4B                                             | Qwen3.5-4B                                         |
| Data          | same 100,401-row subset                                | same                                               |
| Hidden states | same live vLLM extraction server (online fetch+delete) | same                                               |
| Hardware      | 2×A100-80GB training ranks + shared 2-GPU server       | same                                               |

Data: `shanjiaz/qwen3_5_4b_perfectblend_regen` (`open-perfectblend_Qwen3.5-4B.jsonl`, sha256 `a833a0faf8c050a87fd8cc60cd6630691937d4d520d638417deb483b45cb03c2`), pretokenized `input_ids`/`loss_mask` rows. A deterministic 100k subset was drawn by hashing row index (`sha256("ab4b-subset-<i>") % 1,729,783 < 100,000`), then `speculators prepare-data --seq-length 3072` (passthrough path; 97,248 rows / 236.4M tokens survived truncation).

Recipe (4B run's verified hyperparameters): lilicorr, block 16, γ=7, 512 anchors, seq 3072, 5 draft layers on verifier layers 2/9/16/23/30, mask token 248077 (verified unused in the 4B tokenizer), lr 6e-4 Muon, cosine + 4% warmup, 1 epoch, seed 42, `--on-missing generate`.

Only 2 training ranks were used (vs 6 in the full-scale run), so absolute numbers are below the full-scale reference; the comparison is old-vs-new under identical conditions.

## Known code deltas (377ee2f → b4830c0, `src/` only)

- `losses/utils.py`: loss denominator changed from raw masked-token count to decay-weighted mask mass (ModelOpt parity). For block 16 / γ=7 this re-weights the printed loss by ~2.26×.
- `lilicorr/core.py`: `lilicorr_target_source` toggle added, default **argmax** at `b4830c0` — but `377ee2f` had no toggle and always used argmax, so supervision is identical in both runs.
- Chunked checkpointed loss (`loss_chunk_anchors`), dflash `compute_heads` chunking, `lilicorr_lr` param group, optimizer prefix split.
- Data pipeline (`cli/`, `data_generation/`, `train/data.py`, `train/dataloader.py`) is byte-identical between the two commits.

## Results

Final held-out validation (teacher-forced, 10% split), 1 epoch over the 100k subset:

| Metric           | run-old (377ee2f) | run-new (b4830c0) |
| ---------------- | ----------------- | ----------------- |
| val pos_1 acc    | 0.882             | 0.885             |
| val pos_5 acc    | 0.554             | 0.563             |
| val pos_8 acc    | 0.396             | 0.410             |
| val pos_12 acc   | 0.253             | 0.269             |
| val pos_15 acc   | 0.183             | 0.196             |
| val lilicorr_eal | 7.035             | **7.100**         |
| val top1_eal     | 5.989             | 6.157             |
| val oracle_eal   | 9.369             | 9.579             |
| val lilicorr_ce  | 0.366             | 0.375             |
| val dflash_loss  | 0.917             | 1.986             |

As expected, absolute loss values differ (~2.2× on dflash_loss) because of the denominator change; every acceptance-quality metric is equal or slightly better on the new code.

## Conclusion

**No code regression.** The new code (`b4830c0`) matches or slightly beats the old code on every acceptance metric under identical data, verifier, and hyperparameters. The loss-scale difference is the intentional decay-weighted normalization, not a training-quality change.

Therefore the 27B run's weaker acceptance (vs the 4B full-scale run) is **not** explained by code changes between 2026-10-21 and 2026-10-28. Remaining candidate causes, to be tested separately:

1. Verifier difficulty: Qwen3.8-27B is intrinsically harder to draft (sharper argmax, longer reasoning chains).
2. Training progress: the 27B run was stopped at ~11% of 2 epochs (pos_1 ~0.72 and climbing); the 4B numbers are post-full-epoch.
3. Geometry: the 27B run used block 8 / γ=3 / seq 8192 / layers 4..60, not the 4B-proven block 16 / γ=7 / seq 3072.
4. Data: 27B run used the public mirror corpus, not the private original.

A follow-up run isolates (1): Qwen3.8-27B verifier on the same 100k-scale subset with the 4B A/B's recipe geometry, new code.
