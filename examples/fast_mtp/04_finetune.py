"""Fine-tune a converted FastMTP speculator on pre-generated hidden state data.

Prerequisites
-------------
1. A converted FastMTP checkpoint on disk (see ``01_convert_checkpoint.py``).
2. A directory of ``.pt`` training files (see ``02_generate_data.py``).

Usage
-----
Edit the constants below, then run::

    python examples/fast_mtp/04_finetune.py
"""

from speculators.models.fast_mtp import FastMTPSpeculator
from speculators.train.fast_mtp_trainer_utils import build_fast_mtp_trainer

# ── Configuration ─────────────────────────────────────────────────────────────

SPECULATOR_PATH = "./output/qwen3_next_80b_mtp_speculators"
DATA_DIR = "/mnt/data/rahul-tuli/datasets/Qwen3-Next-80B-A3B-Instruct_ultrachat"
OUTPUT_DIR = "./output/qwen3_next_80b_mtp_finetuned"

MAX_LEN = 4096
LR = 5e-5
NUM_EPOCHS = 3
BATCH_SIZE = 1
TRAIN_RATIO = 0.9
SCHEDULER_TYPE = "cosine"

# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print(f"Loading speculator from {SPECULATOR_PATH}")
    model = FastMTPSpeculator.from_pretrained(SPECULATOR_PATH)
    model.train()

    print(f"Data directory : {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    trainer = build_fast_mtp_trainer(
        model=model,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        max_len=MAX_LEN,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        scheduler_type=SCHEDULER_TYPE,
    )

    print(
        f"Training on {len(trainer.train_loader.dataset)} samples, "  # type: ignore[arg-type]
        f"validating on {len(trainer.val_loader.dataset)} samples"  # type: ignore[union-attr,arg-type]
    )
    trainer.run_training()

    print(f"\nDone. Checkpoints saved to {OUTPUT_DIR}")
