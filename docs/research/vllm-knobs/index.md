# vLLM knobs

Serving-side configuration measured against a fixed checkpoint: adaptive verification, draft budget / `num_speculative_tokens`, CUDA graph capture sizes, attention backend, parallelism, prefix caching, concurrency sweeps.

Nothing is trained here. If the conclusion is about a model rather than a flag, it belongs in [model evaluations](../model-evaluations/index.md).

Start from [`TEMPLATE.md`](TEMPLATE.md). The conventions that apply to every entry are in the [hub overview](../index.md).

## What we have learned

- [Adaptive verification vs. fixed draft budget on Gemma4-31B-it](adaptive-verification-dflash-dflash2-dspark-gemma4-shubhra.md) — AV matched a fixed budget at N≤16 and pulled ahead mid-range: +10.0% throughput / +11.8% interactivity for DFlash at N=64, with DFlash2 and DSpark close behind. At N=128 gains narrow to +1.3–4.7%; repeat spread there was under 1% of the mean, and the entry notes the smaller gains may be noise at that single point. 3 repeats at N=32–128. Acceptance length fell under load while the fixed budget's stayed flat.
- [DFlash adaptive verification on Qwen3.5-35B-A3B: CUDA graph capture sizes](dflash-adaptive-verification-cuda-graphs-tomasruizt.md) — with default capture sizes AV *reduced* throughput by 6.6% at c128, because steps containing a prefill exceed the 1,024-token capture limit and run eagerly. Raising capture sizes to 6,144 lifted AV-off throughput 24% at c128, after which AV helped at every concurrency (+0.7–3.3%). Worth reading before trusting any AV measurement taken at default capture sizes.
- [DSpark + GDN at block 16: does AV help with a larger draft?](dspark-gdn-block16-adaptive-verification-tomasruizt.md) — AV measured 5–25% higher throughput than fixed K=15 on Qwen3.8-27B across four concurrencies. The three methods ran concurrently on separate GPU pairs sharing host resources, so the margin may include placement and contention effects; the entry flags this and recommends a sequential rerun. Profiling at c1 found AV verified 5.91 positions per step versus 16, with GDN recurrence accounting for 47% of the target-time reduction — that profile ran on a different GPU pair and does not account for the whole sweep speedup.

## Pin vLLM precisely

Defaults move between releases and have reversed conclusions before — a capture-size default turned a 6.6% adaptive-verification regression into a gain across the board. A result here is only meaningful against a known build.

- vLLM version **and** commit SHA.
- The full launch command, verbatim, for every configuration including the baseline.
- Any patch applied, as `vllm.patch`.

Always pass `--provenance-dir` to `scripts/launch_vllm.py`, pointed at the run's output directory.

## Vary one thing

Everything except the knob under test must be identical across arms, and the entry must say so explicitly.

- Hold the attention backend fixed across baseline and all arms; force it rather than relying on auto-selection.
- If one configuration requires an extra flag to run at all, apply that flag to every arm including the baseline, and explain why in Observations.
- Prefix caching, max model length, and tensor-parallel size are part of the configuration, not incidental.

## Required in Results

- A target-only baseline under the same flags.
- The concurrency range tested, with per-concurrency numbers rather than a single headline. Serving knobs routinely help at one end of the range and hurt at the other; a single number hides the crossover.
- Total throughput and per-user interactivity (TPOT / ITL) as separate columns.
- The steady-state measurement window defined: which requests are counted, when the window opens and closes.
- Where a knob changes acceptance as well as speed, report both — the mechanism usually shows up as a trade between accepted tokens per step and step cost.

## Limitations

State the concurrency levels, draft budgets, and model sizes you did **not** test. Extrapolating a serving result past the tested range is how these conclusions get misused.
