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

### Example E2E run for Llama 3.3 70B on ShareGPT and Ultrachat ###


if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "meta-llama/Llama-3.3-70B-Instruct"
    OUTPUT_PATH = "./output/llama3_3_70b_sharegpt_ultrachat"
    TOTAL_SEQ_LEN = 8_192

    # Data Generation
    data_gen_args_sharegpt = DataGenArgs(
        train_data_path="sharegpt",
        max_model_len=TOTAL_SEQ_LEN,
        seq_length=TOTAL_SEQ_LEN,
    )

    data_gen_args_ultrachat = DataGenArgs(
        train_data_path="ultrachat",
        max_model_len=TOTAL_SEQ_LEN,
        seq_length=TOTAL_SEQ_LEN,
    )

    # Vocab Mapping
    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=32_000,
        target_vocab_size=128_256,  # From https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json#L37
    )

    # Training
    train_args = TrainArgs(
        logger="trackio",
        lr=3e-5,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="llama3_3_70b_sharegpt_ultrachat",
        epochs=10,
    )

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=[data_gen_args_sharegpt, data_gen_args_ultrachat],
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )
