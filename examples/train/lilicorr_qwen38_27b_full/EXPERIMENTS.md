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
| 2 | 10898 | 10% | **2.728** | — | |
| 3 | 16347 | 15% | **2.956** | — | |
| 4 | 21796 | 20% | **3.041** | — | |
| 5 | 27245 | 25% | **3.130** | — | run stopped here (2026-10-28) to free GPUs for the plan.md run; checkpoint verified + backed up to `checkpoints/0_prev`, resumable |

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

## 5. Run 3 — plan.md recipe, block 8 (2026-10-28 11:40 → ongoing)

**Config:** fresh draft per `~/plan.md` — block **8** (7 speculative tokens),
candidate K=8, **gamma 3**, `total_seq_len` **8192**, **Muon** (2D weights) +
AdamW (rest) both at **6e-4** (Muon wd 0.1, AdamW wd 0), one **cosine**
schedule across 2 epochs, 4% warmup, `checkpoint_freq 0.25`, pure CE,
`input_ids` targets. Branch `lilicorr-qwen38-fresh-2ep` (worktree
`~/speculators-b8`) from `6c88b7e` + FP32 checkpoint-saving patch (`dd330a5`).
Run dir: `run_b8_2ep`. Note: epoch ≈ 92.6k steps at seq-len 8192 (vs ≈ 109k at
6144), so 25% lands at step 23148. Pace ~1,550 steps/h (≈70% faster than
Run 2). Evals run manually (no auto loop, per plan.md §5) with
`num_speculative_tokens=7`.

| Checkpoint | Step | % of epoch 1 | Acceptance length | Pos-0 accept | Notes |
|------------|------|-----------|-------------------|--------------|-------|
| 1 | 23148 | 25% | **3.025** (of 7) | 0.653 | per-position: .653/.436/.302/.221/.169/.135/.109 |
| 2 | ~46296 | 50% | — | — | pending |

vs Run 2 at the same 25%-of-epoch point: 3.025-of-7 vs 3.130-of-15 —
near-identical acceptance length with less than half the drafted tokens
(43% vs 21% per-slot efficiency), and better position-0 (0.653 vs 0.583).

Operational note: the first launch attempt died at step 4 — the switchover
script ran in a tmux session that exited on completion, SIGHUP-ing its
background children (extraction server + trainer). Fixed by giving each
long-lived process its own persistent tmux session (`planb8-vllm`,
`planb8-train`).

**Paused 2026-10-29 ~13:05 EDT** (user request, to set up the input_ids-vs-
argmax target comparison — see [TARGET_SOURCE_COMPARISON.md](TARGET_SOURCE_COMPARISON.md)).
Training had reached ~step 38.4k, but the graceful interrupted-checkpoint
save failed: SIGTERM delivered directly to the workers (SIGINT to torchrun
hadn't propagated) collapsed the TCPStore before the save completed, so
`checkpoints/interrupted/` was never written and steps 23148→~38.4k are lost.
The verified step-23148 checkpoint (`checkpoints/0` + `0_prev`) is intact and
resumable. GPUs and extraction server fully torn down.

## 6. Run 4 — argmax A/B (2026-10-29 13:22 → ongoing)

**Config:** identical to Run 3 except `lilicorr_target_source="argmax"`
(branch `lilicorr-qwen38-fresh-2ep-argmax`, commit `b6271a8` — the default
flipped in `core.py::get_trainer_kwargs`). Run dir: `run_b8_2ep_argmax`.
Purpose: the controlled input_ids-vs-argmax comparison the Run 1 failure
left unanswered — see
[TARGET_SOURCE_COMPARISON.md](TARGET_SOURCE_COMPARISON.md) for the full
writeup.

| Checkpoint | Step | % of epoch 1 | Acceptance length | Pos-0 accept | Notes |
|------------|------|-----------|-------------------|--------------|-------|
| 1 | 23148 | 25% | **3.040** (of 7) | 0.662 | per-position: .662/.440/.305/.221/.169/.135/.109; eval ran automatically overnight, training resumed + backup promoted |

**Result: statistically indistinguishable from Run 3's input_ids side at the
matched step** (3.040 vs 3.025 acceptance, every per-position rate within
~0.01, argmax marginally ahead everywhere, train loss equal at 2.77). The
"argmax trains against wrong targets" concern does not show up in served
acceptance at this stage; `input_ids` stays the default on principle
(matches ModelOpt, correct for sampled data). Caveat: one seed per side, one
checkpoint, one benchmark.

Run 4 continues training past the 25% point (auto-resumed after the eval).

## 7. Infrastructure findings (each cost real debugging time)

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

## 8. Open items

- [ ] Collect 10% checkpoint (step ~10898) HumanEval result for Run 2 and
      compare the acceptance-length trend against 5%.
- [ ] Continue through the epoch; watch whether position-0 acceptance and
      the per-position decay ratio improve.
- [ ] (Optional) re-verify against the paper's reported acceptance-length
      trajectory for this verifier/model pair.
