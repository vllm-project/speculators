import shutil
import sys
from pathlib import Path

# Add scripts directory to path so we can import the run_e2e function.
scripts_path = Path(__file__).absolute().parent.parent.parent.parent / "scripts"
sys.path.append(str(scripts_path))

from gen_and_train import (  # type: ignore[import-not-found] # noqa: E402
    DataGenArgs,
    TrainArgs,
    VocabMappingArgs,
    run_e2e,
)

from tests.e2e.vllm.utils import run_vllm_engine  # noqa: E402


def test_gen_train_acceptance(tmp_path: Path, monkeypatch):
    VERIFIER_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    OUTPUT_PATH = tmp_path / "llama3_8b_sharegpt_5k"
    TOTAL_SEQ_LEN = 8192
    NUM_EPOCHS = 5

    # Data Generation
    data_gen_args_sharegpt = DataGenArgs(
        train_data_path="sharegpt",
        seq_length=TOTAL_SEQ_LEN,
        max_samples=5000,
    )

    # Vocab Mapping
    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=8192,  # Use a very small draft vocabulary for this example
        target_vocab_size=128256,  # From https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json#L37
    )

    # Training
    train_args = TrainArgs(
        lr=3e-4,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="test_gen_train_acceptance",
        epochs=NUM_EPOCHS,
    )

    # Use local environment for training run
    monkeypatch.setenv("LOCAL_TRAIN_ENV", "1")

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=data_gen_args_sharegpt,
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )

    # Final checkpoint path
    FINAL_CHECKPOINT_PATH = OUTPUT_PATH / "checkpoints" / str(NUM_EPOCHS - 1)

    prompts = [
        "Write a binary search algorithm in Python",
        "Explain the concept of quantum computing",
        "Code a simple web server in Python",
    ]

    run_vllm_engine(
        str(FINAL_CHECKPOINT_PATH),
        tmp_path,
        prompts,
        max_tokens=512,
        ignore_eos=True,
        acceptance_thresholds=[0.30, 0.07, 0.01],
    )

    # Forcibly clean up
    shutil.rmtree(str(OUTPUT_PATH))
