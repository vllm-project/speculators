# Speculators — Agent Guide

## Provenance

Training, eval, and vLLM-launch scripts write reproducibility artifacts so runs can be recreated later.

- **Training** (`src/speculators/train/utils.py`): `save_train_command()` writes `train_command.txt` (timestamp, git SHA, world size, package versions, full argv) into the checkpoint directory.
- **Eval** (`scripts/evaluate/evaluate.py`): `save_eval_provenance()` writes `eval_command.txt` (timestamp, git SHA, package versions, full argv) into the output directory.
- **vLLM launch** (`scripts/launch_vllm.py`): `_save_vllm_provenance()` writes `vllm_command.txt`, `vllm.patch`, and `checkpoint_sha256.txt` into `--provenance-dir`. Two subcommands:
  - `launch_vllm.py train MODEL` (default): hidden-states extraction for training data generation.
  - `launch_vllm.py eval MODEL --spec-model DRAFTER`: speculative decoding serving for evaluation.

When publishing a model or eval results (HuggingFace, GitHub comments, etc.), always include the provenance artifacts (`train_command.txt`, `eval_command.txt`, `vllm_command.txt`, patches) so the run can be reproduced.
