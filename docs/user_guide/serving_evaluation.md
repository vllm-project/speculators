# Serving evaluation during training

Teacher-forced validation and serving exercise different draft contexts. To measure real speculative decoding acceptance during training, configure an external evaluator command. By default, evaluation runs **asynchronously** on rank 0 in a background thread against a frozen weights-only snapshot so training continues immediately without blocking on serving startup or benchmark latency.

## vLLM / AIPerf example

Install the [evaluation dependencies](tutorials/evaluating_performance.md#install) in the evaluator's Python environment. From a repository checkout, add this to the training configuration:

```yaml
trainer:
  checkpoint_freq: 1
  checkpoint_eval_command: >-
    python examples/train/evaluate_checkpoint.py
    --verifier Qwen/Qwen3-8B
    --cuda-visible-devices 7
    --dataset gsm8k
    --request-count 8
    --spec-tokens 7
  checkpoint_eval_steps: 500
  checkpoint_eval_mode: async
  checkpoint_eval_timeout: 300
```

Reserve the evaluation GPUs separately: for example, run training with `CUDA_VISIBLE_DEVICES=0,1` and reserve device 7 for the command above. The evaluator explicitly selects its own devices; it does not free or reuse the training model's memory. For a separate host or environment, supply a different command that implements the protocol below.

The example starts a new server with `scripts/launch_vllm.py eval`, selects the checkpoint's saved algorithm (overridable with `--spec-method`), verifies its unique model identity through `/v1/models`, runs `scripts/evaluate/evaluate.py acceptance`, and shuts down the server. It requests per-request speculative metrics and disables prefix caching. Acceptance values are the existing evaluator's pooled histogram statistics, kept separate for each dataset; rates with `_pct` suffixes are percentages. Use the same verifier, sampling parameters, prompt set, draft budget and concurrency when comparing checkpoints. Acceptance alone does not establish a latency improvement.

## Execution modes and trigger cadence

- **`checkpoint_eval_steps` (optional, weights-only export):** When set to a positive integer `N`, every `N` optimizer steps the trainer exports only model weights and config (`checkpointer.save_model`) to a temporary snapshot under `<save_path>/serving_eval/_snapshots/` without writing optimizer/scheduler state or modifying training checkpoints (`<save_path>/<epoch>/` or `checkpoint_best`). When unset (`None`), serving evaluation triggers on each saved training checkpoint selected by `checkpoint_freq` and `save_best` (copying model files—excluding optimizer/scheduler state—into a temporary snapshot in `async` mode so later in-place saves or `save_best` pruning cannot corrupt an in-flight evaluation).
- **`checkpoint_eval_mode: async` (default):** After the brief synchronous weights export or snapshot, training resumes immediately across all ranks while rank 0 runs the evaluator in a background thread. At most one active evaluation and one latest pending snapshot are kept (if newer snapshots arrive while the evaluator is busy, the pending slot is replaced and the superseded snapshot is removed), keeping the evaluation GPU utilized without building an unbounded queue. Temporary snapshot weights are deleted after evaluation completes; artifacts and provenance stay under `<save_path>/serving_eval/`. Completed metrics are flushed on the main thread at subsequent steps and drained at the end of training, logged against their original optimizer step.
- **`checkpoint_eval_mode: sync`:** Pauses training until rank 0 finishes evaluating. Peers wait through the distributed store (outside training collectives), so evaluation can exceed the training collective timeout. Useful for deterministic debugging or custom evaluators that intentionally coordinate shared GPU resources.

## Evaluator protocol

`checkpoint_eval_command` is split into argv without a shell. Use `bash path/to/script.sh` if a shell script is needed. The evaluator receives:

| Environment variable          | Value                                            |
| ----------------------------- | ------------------------------------------------ |
| `SPECULATORS_EVAL_CHECKPOINT` | Absolute path to the evaluated weights directory |
| `SPECULATORS_EVAL_STEP`       | Optimizer step identifying the evaluated weights |
| `SPECULATORS_EVAL_OUTPUT_DIR` | Fresh artifact directory for this invocation     |

Write a nonempty mapping of finite numeric scalars to `$SPECULATORS_EVAL_OUTPUT_DIR/metrics.json`, then exit successfully. The result must be a regular file of at most 1 MiB. For example:

```json
{"gsm8k/acceptance_length": 3.25, "gsm8k/acceptance_rate_pos_1_pct": 87.5}
```

The trainer records these as `serving_eval/gsm8k/acceptance_length` and `serving_eval/gsm8k/acceptance_rate_pos_1_pct` at the checkpoint's optimizer step. TensorBoard and MLflow log out-of-order steps natively; for Weights & Biases, asynchronous results that finish after later training steps are logged against a dedicated `serving_eval/step` x-axis so they are never dropped.

Each invocation retains `evaluation.json` (checkpoint, step and command), `evaluator.log`, `metrics.json`, and any evaluator artifacts under `<save_path>/serving_eval/`. These history directories are preserved even when later sub-epoch saves overwrite `<save_path>/<epoch>/`, `save_best: true` prunes non-best epochs, or temporary async snapshots are cleaned up; rely on the launcher's `drafter_checkpoint_sha256.txt` provenance to verify weight identity. By default the launcher hashes local verifier and drafter weights on every evaluation. Pass `--no-hash-checkpoints` to the example to avoid re-hashing unchanged local verifier weights and record file sizes and modification times for both models instead.

Inherited torchrun rank/rendezvous variables are removed before launch. `checkpoint_eval_timeout` defaults to 300 seconds and includes model startup, benchmark and teardown. An evaluation failure or timeout warns, retains its logs and lets training continue. Interrupting training (`SIGTERM` / `SIGINT`) cancels any active evaluator process group, removes temporary snapshots, and lets workers save their `interrupted` checkpoint. Evaluator commands must keep their workers in the original process group.

Serving evaluation is disabled by default. When `checkpoint_eval_steps` is unset, sub-epoch saves are evaluated immediately and epoch-end saves are evaluated after validation and any best-checkpoint updates and retention, so resuming after an interrupted evaluation restores the recorded best loss. Repeated triggers at the same `(epoch, step)` are evaluated once. Interrupted saves are skipped.
