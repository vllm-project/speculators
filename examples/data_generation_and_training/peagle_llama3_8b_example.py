import sys
from pathlib import Path

# Add scripts directory to path so we can import the run_e2e function.
scripts_path = Path(__file__).absolute().parent.parent.parent / "scripts"
sys.path.append(str(scripts_path))

from gen_and_train import (  # noqa: E402
    DataGenArgs,
    TrainArgs,
    run_e2e,
)

### Example E2E run for P-EAGLE on Llama 3.1 8B with ShareGPT ###

# P-EAGLE (Parallel EAGLE) extends EAGLE-3 with parallel multi-token prediction using
# Conditional-On-Distribution (COD) sampling for memory-efficient training.


if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    OUTPUT_PATH = "./output/peagle_llama3_8b_sharegpt_5k"
    # Use smaller sequence length for initial testing to avoid OOM
    TOTAL_SEQ_LEN = 2048

    # Data Generation (same as EAGLE-3)
    data_gen_args_sharegpt = DataGenArgs(
        train_data_path="sharegpt",
        seq_length=TOTAL_SEQ_LEN,
        max_samples=5000,  # Only use 5000 samples from ShareGPT
    )

    # Vocab Mapping - set to None to use full vocabulary (no reduction)
    vocab_mapping_args = None

    train_args = TrainArgs(
        logger="trackio",
        lr=3e-5,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="peagle_llama3_8b_sharegpt_5k",
        epochs=10,
        num_layers=4,
        speculator_type="peagle",  # Use P-EAGLE instead of EAGLE-3
        para_depths=10,
        down_sample_ratio=0.7,
        down_sample_ratio_min=0.2,
        max_seq_len=2048,
        ptd_token_id=128255,
    )

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=data_gen_args_sharegpt,
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )
