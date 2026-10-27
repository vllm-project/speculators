# Experiment Log — LiLiCorr Draft Model (Qwen3.8-27B verifier)

Running record of experiments, configuration changes, and results for the
LiLiCorr speculative-decoding draft model training. Companion to
[HANDOFF.md](HANDOFF.md) (setup/operations) — this file covers *what we tried
and what happened*.

- **Dates:** 2026-10-26 → ongoing
- **Verifier:** Qwen3.8-27B (`/tmp/models/Qwen3.8-27B`), served via vLLM 0.32.0
- **Hardware:** 8× GPU — extraction server on GPUs 0–1, FSDP trainer (6 ranks) on GPUs 2–7
- **Eval:** HumanEval via `scripts/evaluate/evaluate.py` (throughput mode, default
  sampling temperature), draft served with `num_speculative_tokens=15`,
  `draft_sample_method=greedy`

---

## 1. Configuration evolution (pre-launch, paper-fidelity work)

Each change below was made to align with arXiv:2608.20530v2 (Table 4) and/or
NVIDIA ModelOpt's reference implementation (`hf_lilicorr.py` / `hf_dflash.py`),
before any long training run.

| # | Change | From → To | Why | Result |
|---|--------|-----------|-----|--------|
| 1 | Separate head LR | single LR → `lilicorr_lr` param group split | allow independent tuning of LiLiCorr head vs DFlash backbone | feature added (`optimizers.py`, `schema.py`); unused in final recipe (both use `lr`) |
| 2 | Optimizer | `muon` → `adamw` | paper Table 4 | in `full.yaml` |
| 3 | Learning rate | `1.0e-3` → `6.0e-4` | paper Table 4 | in `full.yaml` |
| 4 | Scheduler | `cosine` → `linear` | paper Table 4 | in `full.yaml` |
| 5 | Loss fn | `{"ce":0.1,"tv":0.9}` → `ce` | paper Table 4 | in `full.yaml` |
| 6 | `total_seq_len` | `8192` → `4096` → `6144` | iterative; 6144 balances memory vs block coverage | final: 6144 |
| 7 | Target definition | argmax of verifier logits → **`input_ids`** (toggle `lilicorr_target_source`) | ModelOpt uses the sampled token actually in the sequence as ground truth; our training data was *sampled* (temp 1.0, top_p 0.95, top_k 20), so verifier argmax routinely disagrees with the actual next token — argmax silently trains against wrong targets | toggle added; **default flipped to `input_ids` before Run 2** (user decision, 2026-10-27) |
| 8 | Loss normalization | raw mask count → decay-weighted sum in denominator | ModelOpt divides by the decay-weighted token count; raw count under-weighted the loss by **2.26×** for block_size=16, gamma=7 | fixed in `losses/utils.py` + lilicorr chunked path |
| 9 | Checkpoint frequency | 5% of epoch (~5449 steps) | per user | `checkpoint_freq: 0.05` |

Full audit vs ModelOpt and a full paper-hyperparameter re-verification were
done on top of these; no further discrepancies found.

## 2. Run 1 — argmax targets (2026-10-26 evening → 2026-10-27 ~12:00)

**Config:** the recipe above but with `lilicorr_target_source="argmax"` (then
the default).

- Trained through the 5% checkpoint (step 5449) and the 10% checkpoint
  (step 10898, landed 04:54 Oct 27).
- **Ad-hoc probe serving** (step-5449 checkpoint, single requests, tiny
  samples): mean acceptance length **~2.2–2.6** (2.27 / 2.33 / 2.53 / 2.56
  across probes; per-position acceptance started ~0.63–0.68 at position 0).
- **Automated HumanEval at step 10898: FAILED** — every request 400'd.
  Root cause: eval server ran `--max-model-len 2048` while `evaluate.py`'s
  throughput mode hardcodes `max_tokens=4096` into a backend field that
  `--gen-kwargs` does **not** override (confirmed by direct reproduction).
  Result: "No successful throughput requests". No valid HumanEval number
  exists for Run 1.
- **Run ended in checkpoint corruption** at ~step 16347: the eval loop's
  pause (`pkill`) landed mid-`optimizer_state_dict.pt` write.
  `PytorchStreamReader failed reading zip archive` on resume; model weights
  were intact but optimizer state was not, and no backup existed.
  Archived: `/home/MeganEFlynn/archived_checkpoints/runfull_checkpoints_corrupted_20261027_120706/`
  (+ logs in `..._logs_corrupted_run_20261027_120723/`).

**Lessons fixed in response:**
- Eval loop now watches `training_state.json` mtime (written *last* in the
  save sequence — the commit marker) instead of `model.safetensors`.
- Eval server now uses `--max-model-len 8192`.
- Verified-backup rotation added (`checkpoints/0_prev`, promoted only after
  confirmed resume).

## 3. Smoke test (2026-10-27 11:20)

`smoke.yaml`: 40 steps, `checkpoint_freq: 0.0001` (checkpoint every ~11
steps), isolated save/log dirs. Purpose: validate the full
pause → GPU-clear → eval-server → HumanEval → resume → promote pipeline in
minutes before committing to a multi-day run.

**Result: PASS** — checkpoint at step 11 detected, training paused, GPUs
cleared, eval server healthy, HumanEval ran, training resumed cleanly.

## 4. Run 2 — input_ids targets (2026-10-27 ~12:30 → ongoing)

**Config:** identical to Run 1 except `lilicorr_target_source="input_ids"`
(the corrected ground-truth definition). Fresh from step 0, no resume.

| Checkpoint | Step | % of epoch | Acceptance length | Pos-0 accept | Notes |
|------------|------|-----------|-------------------|--------------|-------|
| 1 | 5449 | 5% | **2.562** | 0.583 | first fully-valid measurement of the entire pipeline |
| 2 | ~10898 | 10% | — | — | pending |

**Step-5449 per-position acceptance rates** (HumanEval, 100,798 drafts,
157,447 accepted tokens):

| Pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|-----|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|
| Rate | .583 | .339 | .208 | .134 | .090 | .063 | .045 | .032 | .022 | .016 | .011 | .008 | .005 | .003 | .002 |

Observations:
- Decay is near-geometric (~0.65–0.67 ratio/position), summing to the 2.56
  acceptance length. No anomalous cliffs.
- Tail beyond position ~6 is nearly dead (<5%) at this stage — expected to
  fill in as the selector head trains.
- Pipeline worked end-to-end: pause, GPU clear (first attempt), eval,
  verified resume, promotion to `checkpoints/0_prev`.

*(This table is updated as each checkpoint's eval completes; raw results live
in `runfull/humaneval_each_checkpoint/step_<N>/` under the run directory.)*

## 5. Infrastructure findings (each cost real debugging time)

| Symptom | Root cause | Fix |
|---|---|---|
| Training loss exactly `0.00e+00` | stale non-editable `hs_connectors` from PyPI (incompatible API) shadowing local workspace copy | `pip install -e ./hs_connectors` |
| vLLM rejects `speculators`-format lilicorr config | config-translation glue missing from installed vLLM (model/worker code existed) | patch `algos.py`/`base.py` in site-packages (documented in HANDOFF.md) |
| 369 GB `/tmp` fill-ups | orphaned hidden-state files from failed generations | periodic `rm -rf /tmp/hidden_states/*`; watch disk |
| Extraction-server OOM during KV profiling | `--max-num-batched-tokens 16384` | reduced to 8192 |
| `torch_call_dispatcher` ABI crash | FlashAttention-3 vs torch ABI mismatch | `--attention-config '{"flash_attn_version": 2}'` |
| Silent worker death | CUDA graph capture crash | `--enforce-eager` |
| HumanEval all-400s | `--max-model-len 2048` < evaluate.py's hardcoded `max_tokens=4096` | `--max-model-len 8192` (see Run 1) |
| Checkpoint corruption | `pkill` mid-save; checkpointer has no atomic staging | watch `training_state.json`; verified-backup rotation (see Run 1) |
| Orphaned GPU memory after kills | vLLM/torch workers renamed via `setproctitle` (`VLLM::Worker` etc.) escape `pkill -f` | kill by PID from `nvidia-smi --query-compute-apps`, poll until memory clear |

## 6. Open items

- [ ] Collect 10% checkpoint (step ~10898) HumanEval result for Run 2 and
      compare the acceptance-length trend against 5%.
- [ ] Continue through the epoch; watch whether position-0 acceptance and
      the per-position decay ratio improve.
- [ ] (Optional) re-verify against the paper's reported acceptance-length
      trajectory for this verifier/model pair.
