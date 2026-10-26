---
name: speculators-evaluation
description: Run Speculators acceptance evaluations and vLLM concurrency sweeps with AIPerf, prepare SPEED-Bench or MRCR inputs, and report acceptance length/rates and throughput-versus-interactivity curves. Use for dataset selection, benchmark runs, server setup, and baseline/speculative comparisons. Does not perform model training.
---

# Speculators evaluation

Use the checkout's AIPerf-based evaluator to collect decoding statistics and performance curves. A request for commands, a report, or one dataset does not authorize additional runs. Follow the user's selected host, environment, scope, and comparator. If the user pauses monitoring, keep it paused until explicitly resumed; do not create follow-ups unless requested.

## Interface and responsibilities

Read applicable `AGENTS.md`, inspect the branch, and consult `docs/user_guide/tutorials/evaluating_performance.md` and current help before constructing commands. This skill describes the AIPerf implementation; do not apply its flags to an older checkout.

```bash
python scripts/evaluate/evaluate.py acceptance --help
python scripts/evaluate/evaluate.py speedup --help
python scripts/evaluate/evaluate.py summarize --help
```

From the repository root, install in the selected client environment with `pip install -r scripts/evaluate/requirements.txt` when installation is needed and authorized. The editable adapter package pins **AIPerf 0.13.0** and supports Python 3.11–3.13. Use the direct Python entrypoint; do not assume a `speculators evaluate` console command exists.

- `evaluate.py`: CLI parsing and read-only inspection of forwarded options, mode defaults, `ALIASES`, dataset/native command dispatch, root-level provenance, and acceptance CSV reporting. There is no separate acceptance client or `acceptance.py`.
- `eval_datasets/__init__.py`: empty package marker for native plugin discovery.
- `eval_datasets/speedbench.py`: SPEED-Bench arguments, pinned preparation, and per-category conversation counts for native run budgets.
- `eval_datasets/mrcr.py`: Hugging Face loader preserving complete histories.
- `eval_datasets/plugins.yaml` and `pyproject.toml`: register the MRCR loader through AIPerf's plugin entrypoint; they are not a run configuration.

AIPerf owns requests, scheduling, native metrics, exports, and plots. Keep CLI orchestration in `evaluate.py`, source-specific preparation/loading in the dataset modules, and the package initializer empty. Calculate reports offline; prefer small changes to these owners over adding another runner or timing estimator.

## Server and comparison setup

Reuse an appropriate existing endpoint. For an authorized launch, check resource availability, use the selected existing vLLM environment, and record the command, logs, checkpoint revisions, and effective startup configuration. If using `scripts/launch_vllm.py`, follow repository provenance requirements. Check `/health` and `/v1/models` before sending an authorized workload.

Acceptance requires a build supporting `--per-request-spec-decode-metrics summary` with a response schema compatible with AIPerf. This integration was exercised with vLLM 0.31.0; do not infer a minimum supported release from that test or assume experimental metrics remain stable. Record the exact build.

For DSpark, default to seven speculative tokens with adaptive verification off; for a supported native MTP comparator, default to three. Explicit session settings override these defaults. Enable adaptive verification only when requested and supported by the checkpoint/build. Prefer native MTP for an unspecified comparator when its weights and implementation are available; otherwise use the same verifier with speculative decoding disabled. Do not silently substitute after a requested comparator fails.

Match GPU type/count, TP, weights/KV precision, context/RoPE, scheduler limits, generation settings, workload, and measurement budgets. Disable prefix caching for comparisons. The wrapper does **not** validate prefix-cache counters or enforce zero queueing: verify configuration and disclose resource limits when interpreting high-concurrency points. Verify effective context/RoPE for both verifier and draft/MTP head; use run-local configuration copies rather than editing shared caches.

Serialize workloads on each endpoint. Matched servers can run in parallel on separate authorized GPU allocations and ports. Stop only processes owned by the requested run. Preserve failed measurements; do not automatically retry them or change settings to obtain a result.

## Defaults and configuration

Both modes use chat endpoints, sequential sampling, server-reported token counts, raw exports, and this generation object:

```json
{"temperature":0,"chat_template_kwargs":{"enable_thinking":true},"max_completion_tokens":32768}
```

`--extra-inputs` replaces that entire object. Include every desired generation setting when overriding it. The same cap applies to MRCR. There is no wrapper `--body`, `--max-tokens`, or `--max-samples` flag.

| Mode       | Defaults                                                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Acceptance | Nonstreaming, concurrency 64, request timeout 900 seconds.                                                                                                      |
| Speedup    | Streaming; concurrency 1,4,8,16,32,64,128; three profile runs; 30 seconds warmup per point; 100 seconds measurement; zero grace period; native automatic plots. |

Other options pass through to `aiperf profile`; use canonical long names. The wrapper supports CLI commands only and rejects YAML configuration flags and alternate conversation-budget spellings. Use `--num-conversations` here; run `aiperf profile --config FILE.yaml` directly for YAML. Use a fresh `--output-artifact-dir`; the wrapper adds no resume/overwrite policy.

## Datasets and budgets

Use `--dataset` to select datasets. When omitted, acceptance defaults to SPEED-Bench qualitative and speedup defaults to HumanEval.

Use `--dataset NAME [NAME ...]`. Each selected dataset runs sequentially with the same profile flags and budget. Each dataset writes native results under `<root>/<dataset>/`, with acceptance CSVs at the root.

- `gsm8k`, `math500`, `humaneval`, `mbpp`, and `mtbench` map to native `spec_al_*` loaders. Other native public dataset names work. Without an explicit budget these use AIPerf's **ten-request** default. Prompt templates and multi-turn handling are native.
- AIME26 and Arena-Hard have no equivalent native public presets in this version. Use prepared native custom inputs when requested; do not silently omit them from an expected seven-dataset evaluation. There is no `basic` expansion.
- `qualitative` aliases `speed_bench_qualitative` and stays one aggregate profile. The inspected source contains 880 conversations across 11 categories and 1,138 user turns. For per-category acceptance, use AIPerf's bundled `speed_bench_sweep` YAML template and `speed-bench-report`; the wrapper does not expand categories or launch report commands.
- Named qualitative categories such as `speed_bench_coding` and `speed_bench_math` default to their full selected conversation count (currently 80 each). Selecting categories explicitly launches separate runs. Throughput categories use names such as `speed_bench_throughput_8k_high_entropy`; acceptance retains the native request budget.
- `--request-count` limits model calls and may cut a multi-turn conversation short; `--num-conversations` lets conversations finish their turns. Explicit budgets apply per selected dataset and suppress its automatic full-conversation budget. `qualitative --request-count 8` makes eight calls across the aggregate split. Native YAML sweeps define their own budgets; use `sessions: 80` per category for full qualitative histories. Do not call a request budget a unique-source cap.

When `--dataset` selects SPEED-Bench without `--input-file`, prepare only the needed split using NVIDIA NeMo Skills commit `5ac8609a56ac941540b10c92e68d556e6343cd4c`. Cache under `$HF_HOME/speculators/speedbench/<prepare-commit>/` (default `~/.cache/huggingface`). AIPerf loads and filters categories; the adapter only counts source conversations to set CLI acceptance budgets. Keep prepared prompts local and respect source access/redistribution restrictions.

For a full comparison, state the dataset selection and budgets first, then run acceptance, matched timed sweeps, and separately requested long-context acceptance. Reuse the user's requested sequence. Do not silently substitute the old GuideLLM seven-dataset workflow or source-cap semantics.

## Acceptance commands

Set `MODEL` to the served alias and `TARGET` to the selected server base URL. All qualitative conversations/categories:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset qualitative --model "$MODEL" --url "$TARGET" \
  --output-artifact-dir output/qualitative
```

A short multi-dataset check, with eight model calls per dataset:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset gsm8k math500 --request-count 8 --concurrency 2 \
  --model "$MODEL" --url "$TARGET" \
  --output-artifact-dir output/acceptance-smoke
```

MRCR streams `openai/mrcr` through the registered `mrcr` loader. `needles` accepts 2, 4, or 8 (default 8); `max_samples` caps source rows (default 100); `revision` can pin a Hugging Face commit. Source cap and request budget are separate. Each prompt becomes one request containing its complete message history, including historical assistant replies. No `/tokenize`, truncation, context filtering, or regeneration of earlier turns occurs.

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset mrcr --dataset-filter needles=8 --dataset-filter max_samples=2 \
  --request-count 2 --concurrency 1 --request-timeout-seconds 3600 \
  --model "$MODEL" --url "$TARGET" --output-artifact-dir output/mrcr
```

Preserve and report over-context errors. This mode measures decoding statistics, not MRCR answer accuracy.

## Results and failure behavior

Native files remain intact: `profile_export_aiperf.json`/`.csv`, `profile_export.jsonl` request records, and `profile_export_raw.jsonl` payloads/responses. Sweep points live under `profile_runs/trial_*/concurrency_*/`. Keep the root-level `eval_command.txt`, each dataset's expanded `aiperf_command.txt`, and server/checkpoint provenance.

Reporting reads each native summary and profiling request JSONL once after the runs. It excludes warmup and duplicate phase exports, and writes:

- `acceptance.csv`: pooled acceptance statistics per native run.
- Native `aiperf speed-bench-report` commands produce `acceptance_length_by_category.csv` and `acceptance_rate_by_category.csv` separately; these are not generated by the wrapper.
- `acceptance_by_prompt_length.csv`: statistics and exported request counts per actual prompt-length bin, starting at 8,192 and doubling through 1,048,576, plus overflow and `unknown`.

Pool counters before division: length = `1 + accepted_draft_tokens / num_spec_steps`; rate = `100 * accepted_draft_tokens / proposed_draft_tokens`; position k = `100 * steps_accepting_at_least_k / num_spec_steps`. Numeric position columns start at 1. Missing values stay blank. Native per-request averages can differ from these pooled values.

Category reporting is delegated to AIPerf 0.13.0's `speed-bench-report`, using separate category runs and their `server_metrics_export.json`. Keep native server-metrics collection enabled and expose `/metrics`. Native acceptance rates are fractions (0–1); the native `Overall` is an arithmetic mean across category values. Do not label it as pooled per-request acceptance. Per-position rates for each category remain in `acceptance.csv`; prompt-bin counts retain exported errors with `error_isl` or `unknown`, excluding their acceptance counters. No request-ID/category join or custom category aggregation remains.

A native subprocess failure preserves artifacts and continues subsequent selected datasets, returning nonzero afterward. Preparation/download/process-launch exceptions can abort before final summaries. Validation returns nonzero for incomplete native reports, request errors, or no successful requests; acceptance additionally requires speculative metrics and usable raw records covering every success. Partial CSVs may remain. There is no custom three-failure cutoff, retry, or per-dataset checkpoint policy.

## Native SPEED-Bench category workflow

Prepare qualitative with `python scripts/evaluate/eval_datasets/speedbench.py`, which prints its local path. Generate the upstream config with `aiperf config init -t speed_bench_sweep -o speed-bench.yaml --model "$MODEL" --url "$TARGET"`. Set the file dataset path, model/endpoint, server metrics URL, generation settings, and budget in that native YAML. It does not inherit wrapper presets. For our acceptance settings, use nonstreaming chat, timeout 900, server token counts, sequential sampling, raw exports, the standard generation object, no warmup, and a profiling phase with `sessions: 80` and `concurrency: 64` per category.

Use `aiperf config validate` and `aiperf config expand` to check the 11-category plan without launching requests. Run `aiperf profile --config speed-bench.yaml`. Point `aiperf speed-bench-report` at the resulting variation run directories for each of `--metric accept_length` and `--metric accept_rate`. Follow the concrete commands in the evaluation tutorial. Then use `python scripts/evaluate/evaluate.py summarize --output-artifact-dir RUN_ROOT` for the additional pooled per-run, prompt-length, and position CSVs. This postprocessing launches no benchmark and preserves original profiling provenance.

## Speedup and graphs

```bash
python scripts/evaluate/evaluate.py speedup \
  --dataset humaneval --model "$MODEL" --url "$TARGET" \
  --output-artifact-dir output/humaneval-dspark
```

Override native defaults when requested, for example `--concurrency 1,8 --num-profile-runs 2 --warmup-duration 5 --benchmark-duration 30`. Speedup does not add a full-qualitative conversation budget, launch a comparator, or compute baseline-relative speedup ratios. Keep compared runs in distinct directories and use only matched settings.

Native interactivity `output_token_throughput_per_user.avg` is the mean of per-request reciprocals of inter-token latency. The previous reciprocal-mean axis would be `1000 / inter_token_latency.avg` when latency is in milliseconds; these are different statistics. Y is native total `output_token_throughput.avg`. With server counts and zero grace, unfinished requests' partial output is excluded. Do not claim equivalence to GuideLLM's partial-window accounting or mix collectors as matched results.

```bash
aiperf plot output/humaneval-dspark --output output/plots
```

Native plotting accepts `--config FILE.yaml` on `aiperf plot`; PNG export needs Chrome/Chromium (`BROWSER_PATH` if necessary). Automatic plot failures are nonfatal unless `--plot-required` is supplied. In AIPerf 0.13.0, **static multi-run PNG export chooses p50 when present even for a requested avg statistic**, and variation-derived `experiment_group` values can merge separate deployments at the same concurrency. Verify the actual plotted statistic and grouping; do not present a mean-interactivity comparison as validated merely because a PNG exists. Dashboard statistic selection is separate. An offline plot from verified native exports can satisfy an exact chart request without changing measurement code.

## Delivery

Report the measured selection, unique conversations versus model calls, successes/errors/missing metrics, pooled acceptance length/rate, and numeric position rates. For MRCR, include needle/source selection, prompt bins, maximum successful prompt length, and context-limit failures. Derive counts from saved records; do not assume the requested budget completed.

For every performance chart, state the actual X/Y statistics, GPU count/type per server, TP, precision, context, draft length/adaptive setting, sampling/output cap, prefix caching, warmup/window/grace, completed request counts, and repeats. Label only configurations actually measured. Compare speedup ratios only at clearly defined matched points.

Link summaries, graphs, native archives, commands, and provenance in the requested report location. Verify numbers and image links. Separate CPU/fake-server plumbing tests from real model measurements, and acceptance statistics from task accuracy. Do not distribute restricted prompts or raw exports containing them in a public report. Saving artifacts does not authorize commits, pushes, PRs, posting, or publication.
