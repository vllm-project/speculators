# Independent consumer fan-out benchmark

This fixture runs two fresh-server scenarios in order:

1. one vLLM hidden-state producer and one single-process trainer (`1p1c`);
2. one fresh producer and three independently launched single-process trainers
   (`1p3c`).

The launcher rejects distributed launchers and any DP, TP, PP, SP, or process-count
option other than one. It also requires distinct physical GPU indices for every role
that overlaps in time. `CUDA_VISIBLE_DEVICES` and distributed rank variables are owned
by the launcher and cannot be supplied by a role config.

Each trainer receives its own accounting endpoint through the `{endpoint}` command
placeholder. Other placeholders are `{consumer_id}`, `{output_dir}`, `{scenario}`, and
the scenario-local `{shared_artifacts_dir}`.
The proxy forwards non-streaming OpenAI requests to vLLM and records a completion only
when the response is successful and contains a hidden-state artifact path. It stores
only a digest of the request identity, never the returned artifact path.

Start from `config.example.json`, replace the model and preprocessed-data placeholders,
and keep each consumer command as a direct, single-process `scripts/train.py` launch.
Run the fixture from the repository root:

```bash
python scripts/benchmark_independent_consumers.py \
  benchmarks/independent_consumer_fanout/config.example.json \
  --run-directory /tmp/speculators-fanout-run \
  --report /tmp/speculators-fanout-report.json
```

The run directory must not already exist. Role logs remain there and the compact report
contains the exact command/configuration, package versions, request and valid-completion
counts, shared-sample multiplicity, a common post-warmup throughput window, native
per-consumer `profile/step_ms` summaries, makespan, and sampled GPU memory. Environment
values are omitted from the report. The command exits nonzero
if a role fails, a GPU is shared or already occupied, a completion is malformed, sample
multiplicity is ambiguous, or the common steady-state window is too small.

For the unshared baseline,
`expected_service_completions_per_shared_sample` is one for `1p1c` and three for
`1p3c`. A publish-once implementation changes the latter to one; the logical consumer
commands and all other workload settings must remain equivalent.

To measure publish-once fan-out, pass the same cache to every consumer with
`--shared-hidden-states-path {shared_artifacts_dir}` and set the `1p3c` expected service
multiplicity to one. The report then includes aggregate logical request, hit, miss,
coalesced-waiter, retry, publish, failure, cleanup, and timeout counters under
`shared_artifact_cache`. The run fails closed unless three logical requests correspond
to every service completion, exactly one miss is published, the other two requests hit,
and all failure, retry, cleanup, and timeout counters are zero. Baseline scenarios that
do not use the shared-cache placeholder remain valid without cache accounting.

In publish-once mode, `per_consumer_completions` and the steady-state per-consumer
completion map identify which consumer owned each service miss. They do not represent
logical trainer progress: cache logical-request totals and each independent consumer's
`consumer_step_times` provide that evidence. The report labels both maps with
`service_request_owner` to make this distinction explicit.
Despite their names, `warmup_completions_per_consumer` and
`minimum_steady_completions_per_consumer` are service-wide publication thresholds in
this mode; the corresponding step fields apply separately to every consumer.
