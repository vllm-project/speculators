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

### Example E2E full run for Qwen 3 8B on ShareGPT and UltraChat ###

# Note: This is a full training run using all the data from ShareGPT (~130k samples) and
# UltraChat (~200k samples).

# Because this is a thinking model, we use "turn dropout" which randomly truncates
# training conversations. This is because thinking models only use the last response
# when training (via loss masking). By randomly truncating the conversations, the model
# learns to generalize to both short and long conversations.

# Timing (on 4x NVIDIA H100 80GB GPUs)
# Data Generation: ~15 hours
# Vocab Mapping: 6 seconds
# Training: ~8 hours
# Total: ~23 hours

# Results on MT-Bench:
# first token accuracy: 0.58
# second token accuracy: 0.28
# third token accuracy: 0.13
# average acceptance length: 1.98


if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    OUTPUT_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    TOTAL_SEQ_LEN = 8192
    TOTAL_SEQ_LEN2 = 2048

    # Data Generation
    data_gen_args_sharegpt = DataGenArgs(
        train_data_path="sharegpt",
        seq_length=TOTAL_SEQ_LEN,
        turn_dropout=True,  # Turn dropout enabled here
    )
    '''
    data_gen_args_ultrachat = DataGenArgs(
        train_data_path="ultrachat",
        seq_length=TOTAL_SEQ_LEN,
        turn_dropout=True,  # Turn dropout enabled here
    )
    '''

    # Vocab Mapping
    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=64000,  # Use an 32k draft vocabulary
        target_vocab_size=151936,  # From https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json#L29
    )

    # Training
    train_args = TrainArgs(
        logger="trackio",
        lr=3e-5,
        total_seq_len=TOTAL_SEQ_LEN2,
        run_name="qwen3_30b_instruct_finetune",
        epochs=10,
        pretrained_model_path="RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3",
    )

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=[data_gen_args_sharegpt],
        vocab_mapping_args=None,
        train_args=train_args,
    )