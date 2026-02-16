
import sys
from pathlib import Path

# Add scripts directory to path so we can import the run_e2e function.
scripts_path = Path(__file__).absolute().parent.parent.parent / "scripts"
sys.path.append(str(scripts_path))

from gen_and_train import (  # noqa: E402
    DataGenArgs,
    TrainArgs,
    VocabMappingArgs,
    run_e2e,
)

### Example E2E run for Llama 3.1 8B on 20k samples from ShareGPT ###

# Note: With just 20k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model.

# Timing (on 2x NVIDIA H100 80GB GPUs)
# Data Generation: 839 seconds
# Vocab Mapping: 6 seconds
# Training: 1254 seconds
# Total: 2099 seconds (35 mins)

# Results on MT-Bench:
# first token accuracy: 0.40
# second token accuracy: 0.13
# third token accuracy: 0.04
# average acceptance length: 1.57


if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "Qwen/Qwen3-8B"
    OUTPUT_PATH = "./output/qwen3_8b_sharegpt_20k"
    TOTAL_SEQ_LEN = 8192

    # Data Generation
    data_gen_args_sharegpt = DataGenArgs(
        train_data_path="sharegpt",
        seq_length=TOTAL_SEQ_LEN,
        max_samples=40000,  # Only use 5000 samples from ShareGPT
    )

    # Vocab Mapping
    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=32000,  # Use an 32k draft vocabulary
        target_vocab_size=151936,  # From https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json#L29
    )

    # Training
    train_args = TrainArgs(
        logger="wandb",
        lr=3e-5,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="qwen3_8b_sharegpt_20k",
        epochs=10,
    )

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=data_gen_args_sharegpt,
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )
