# Serving evaluation during training

Teacher-forced validation and serving exercise different draft contexts. To compare saved checkpoints with actual speculative decoding, configure an external evaluator command. It runs on rank 0 after a complete checkpoint save and logs scalar results under `serving_eval` at that checkpoint's optimizer step.

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
  checkpoint_eval_timeout: 300
```

Reserve the evaluation GPUs separately: for example, run training with `CUDA_VISIBLE_DEVICES=0,1` and reserve device 7 for the command above. The evaluator explicitly selects its own devices; it does not free or reuse the training model's memory. For a separate host or environment, supply a different command that implements the protocol below.

The example starts a new server with `scripts/launch_vllm.py eval`, selects the checkpoint's saved algorithm (overridable with `--spec-method`), verifies its unique model identity through `/v1/models`, runs `scripts/evaluate/evaluate.py acceptance`, and shuts down the server. It requests per-request speculative metrics and disables prefix caching. Acceptance values are the existing evaluator's pooled histogram statistics, kept separate for each dataset; rates with `_pct` suffixes are percentages. Use the same verifier, sampling parameters, prompt set, draft budget and concurrency when comparing checkpoints. Acceptance alone does not establish a latency improvement.

## Evaluator protocol

`checkpoint_eval_command` is split into argv without a shell. Use `bash path/to/script.sh` if a shell script is needed. The evaluator receives:

| Environment variable          | Value                                            |
| ----------------------------- | ------------------------------------------------ |
| `SPECULATORS_EVAL_CHECKPOINT` | Absolute path to the saved checkpoint            |
| `SPECULATORS_EVAL_STEP`       | Optimizer step identifying the evaluated weights |
| `SPECULATORS_EVAL_OUTPUT_DIR` | Fresh artifact directory for this invocation     |

Write a nonempty mapping of finite numeric scalars to `$SPECULATORS_EVAL_OUTPUT_DIR/metrics.json`, then exit successfully. The result must be a regular file of at most 1 MiB. For example:

```json
{"gsm8k/acceptance_length": 3.25, "gsm8k/acceptance_rate_pos_1_pct": 87.5}
```

The trainer records these as `serving_eval/gsm8k/acceptance_length` and `serving_eval/gsm8k/acceptance_rate_pos_1_pct`. It also keeps `evaluation.json` (checkpoint, step and command), `evaluator.log`, the returned metrics, and any evaluator artifacts under `<save_path>/serving_eval/`. The example retains vLLM launch provenance, checkpoint hashes, and AIPerf's command, raw records and summaries.

Inherited torchrun rank/rendezvous variables are removed before launch. All training ranks wait for evaluation before the checkpoint can be overwritten or removed. Set `checkpoint_eval_timeout` shorter than the training process-group timeout; the default is 300 seconds and includes model startup, benchmark and teardown. A failed or timed-out evaluation warns and retains its logs without stopping training or logging successful metrics. The launched process group is cleaned, including descendant serving workers; evaluator commands must not detach workers into another session.

This option is disabled by default. It evaluates checkpoints actually saved by the existing `checkpoint_freq` / `save_best` policy, including sub-epoch saves; repeated saves at the same optimizer step are evaluated once. Interrupted saves are skipped. Teacher-forced validation and `checkpoint_best` selection remain unchanged. No GPU serving results are implied by enabling this hook.
