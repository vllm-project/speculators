import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

# Add scripts directory to path
scripts_path = Path(__file__).absolute().parent.parent.parent / "scripts"
sys.path.append(str(scripts_path))

from gen_and_train import prepare_args, run_script  # noqa: E402

### Example E2E run for Llama 3.1 8B on 50 samples from ShareGPT ###


if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    HF_DATASET_NAME = "nm-testing/sharegpt_llama3_8b_hidden_states"
    OUTPUT_PATH = Path("./output/llama3_8b_sharegpt_5k")
    TOTAL_SEQ_LEN = 8192

    # Download HuggingFace dataset with pre-generated hidden states
    data_dir = OUTPUT_PATH / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_DATASET_NAME,
        repo_type="dataset",
        local_dir=data_dir,
        local_dir_use_symlinks=False,
    )

    # Training arguments
    checkpoint_path = OUTPUT_PATH / "checkpoints"
    log_path = OUTPUT_PATH / "logs"

    train_args = {
        "verifier_name_or_path": VERIFIER_NAME_OR_PATH,
        "data_path": str(data_dir),
        "save_path": str(checkpoint_path),
        "log_dir": str(log_path),
        "total_seq_len": TOTAL_SEQ_LEN,
        "lr": 3e-5,
        "epochs": 10,
        "run_name": "llama3_8b_sharegpt_5k",
        "logger": "trackio",
        "data_format_version": 1,
    }

    # Prepare command line arguments
    train_args_list = prepare_args(train_args)

    device_count = torch.accelerator.device_count()
    python_alt = f"torchrun --standalone --nproc_per_node={device_count}"

    # Resolve logger requirements
    packages = ["."]
    if train_args["logger"]:
        loggers = train_args["logger"].split(",")
        packages.extend([logger.strip() for logger in loggers])

    # Run training using run_script helper
    run_script(
        script_name="train.py",
        script_args=train_args_list,
        requires=packages,
        python_alt=python_alt,
    )
