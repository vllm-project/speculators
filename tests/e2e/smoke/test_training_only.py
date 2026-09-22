import subprocess
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from loguru import logger

from tests.e2e.utils import run_vllm_engine


class TestTrainvLLM:
    """
    An e2e test which trains a speculator model using pre-computed hidden states
    and runs the trained model in vLLM.
    """

    def _run_training(self, script_path: str, args_dict: dict):
        cmd = [
            "python",
            script_path,
        ]

        for key, value in args_dict.items():
            flag = f"--{key}"

            if value is True:
                cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        logger.info("CMD:")
        logger.info(" ".join(cmd))
        return subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @pytest.mark.smoke
    def test_train_vllm_engine(
        self, tmp_path: Path, prompts: list[list[dict[str, str]]]
    ):
        MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
        DATASET = "inference-optimization/speculators-ci-datasets"
        SAVE_PATH = str(tmp_path / "checkpoints")

        # Fetch pre-computed hidden states (token_freq.pt is bundled)
        data_dir = (
            Path(
                snapshot_download(
                    repo_id=DATASET,
                    repo_type="dataset",
                    allow_patterns=["llama3_8b_hidden_states/*"],
                )
            )
            / "llama3_8b_hidden_states"
        )

        training_args = {
            "lr": 3e-5,
            "total-seq-len": 8192,
            "epochs": 1,
            "verifier-name-or-path": MODEL_PATH,
            "data-path": str(data_dir),
            "hidden-states-path": str(data_dir / "hidden_states"),
            "save-path": SAVE_PATH,
            "log-dir": str(tmp_path / "logs"),
            "on-missing": "raise",
        }
        # Train draft model for one epoch
        p = self._run_training("scripts/train.py", training_args)
        p.wait()

        stdout, stderr = p.communicate()

        if p.returncode != 0:
            print(stdout)  # noqa: T201
            print(stderr)  # noqa: T201

        assert p.returncode == 0

        # Verify train_command.txt was saved and copied into epoch checkpoint dir
        assert Path(SAVE_PATH, "train_command.txt").exists()
        assert Path(SAVE_PATH, "0", "train_command.txt").exists()

        # Run trained speculator in vLLM
        run_vllm_engine(model_path=SAVE_PATH + "/0", tmp_path=tmp_path, prompts=prompts)
