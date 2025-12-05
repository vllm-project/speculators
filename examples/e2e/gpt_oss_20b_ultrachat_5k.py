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

### Example E2E run for GPT-OSS 20B on 5k samples from UltraChat ###

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model.

# Because this is a thinking model, we use "turn dropout" which randomly truncates
# training conversations. This is because thinking models only use the last response
# when training (via loss masking). By randomly truncating the conversations, the model
# learns to generalize to both short and long conversations.

# Timing (on 4x NVIDIA H100 80GB GPUs)
# Data Generation: TBD
# Vocab Mapping: TBD
# Training: TBD
# Total: TBD

# Results on MT-Bench:
# first token accuracy: TBD
# second token accuracy: TBD
# third token accuracy: TBD
# average acceptance length: TBD


if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "openai/gpt-oss-20b"
    OUTPUT_PATH = "./output/gpt_oss_20b_ultrachat_5k"
    TOTAL_SEQ_LEN = 8192

    # Data Generation
    data_gen_args_ultrachat = DataGenArgs(
        train_data_path="ultrachat",
        max_model_len=TOTAL_SEQ_LEN,
        seq_length=TOTAL_SEQ_LEN,
        max_samples=5000,  # Only use 5000 samples from UltraChat
        turn_dropout=True,  # Turn dropout enabled here
    )

    # Vocab Mapping
    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=32000,  # Use a 32k draft vocabulary
        target_vocab_size=201088,  # From https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
    )

    # Training
    train_args = TrainArgs(
        logger="trackio",
        lr=3e-5,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="gpt_oss_20b_ultrachat_5k",
        epochs=10,
    )

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=data_gen_args_ultrachat,
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )
