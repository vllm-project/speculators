# Speculators — Agent Guide

## Provenance

Training and eval scripts write reproducibility artifacts so runs can be recreated later.

- **Training** (`src/speculators/train/utils.py`): `save_train_command()` writes `train_command.txt` (timestamp, git SHA, world size, package versions, full argv) into the checkpoint directory.
- **Eval** (`scripts/evaluate/evaluate.py`): `save_eval_provenance()` writes `eval_command.txt` (timestamp, git SHA, package versions, full argv) into the output directory.

When publishing a model or eval results (HuggingFace, GitHub comments, etc.), always include the provenance artifacts so the run can be reproduced.
