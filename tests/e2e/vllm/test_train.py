import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from loguru import logger

from speculators.train.vocab_mapping import (
    build_vocab_mappings_from_distribution,
)
from tests.e2e.vllm.utils import run_vllm_engine


class TestTrainvLLM:
    """
    An e2e test which trains a speculator model using pre-computed hidden states
    and runs the trained model in vLLM.
    """

    def _generate_t2d_d2t(self, token_freq: str, d2t_path: Path, t2d_path: Path):
        token_freq_dict = torch.load(
            Path(token_freq) / "token_freq.pt", weights_only=True
        )
        d2t, t2d = build_vocab_mappings_from_distribution(
            token_freq_dict=token_freq_dict,
            draft_vocab_size=8192,
            target_vocab_size=128256,
        )

        np.save(d2t_path, d2t.cpu().numpy())
        np.save(t2d_path, t2d.cpu().numpy())

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
    def test_train_vllm_engine(self, tmp_path: Path, prompts: list[str]):
        MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
        DATASET_PATH = "nm-testing/sharegpt_llama3_8b_hidden_states"
        TOKEN_FREQ_PATH = "nm-testing/sharegpt_llama3_8b_token_freq"
        SAVE_PATH = str(tmp_path / "checkpoints")

        # 1. Fetch pre-computed hidden states and token frequency data
        local_dir = snapshot_download(repo_id=DATASET_PATH, repo_type="dataset")
        token_freq = snapshot_download(repo_id=TOKEN_FREQ_PATH, repo_type="dataset")

        d2t_path = tmp_path / "d2t.npy"
        t2d_path = tmp_path / "t2d.npy"

        # 2. Generate t2d and d2t files
        self._generate_t2d_d2t(token_freq, d2t_path=d2t_path, t2d_path=t2d_path)

        training_args = {
            "lr": 3e-5,
            "total-seq-len": 8192,
            "epochs": 1,
            "verifier-name-or-path": MODEL_PATH,
            "data-path": local_dir,
            "save-path": SAVE_PATH,
            "data-format-version": 1,
            "log-dir": str(tmp_path / "logs"),
            "d2t-path": str(tmp_path / "d2t.npy"),
            "t2d-path": str(tmp_path / "t2d.npy"),
        }
        # 3. Train draft model for one epoch
        p = self._run_training("scripts/train.py", training_args)
        p.wait()

        stdout, stderr = p.communicate()

        if p.returncode != 0:
            print(stdout)  # noqa: T201
            print(stderr)  # noqa: T201

        assert p.returncode == 0

        # 4. Run trained speculator in vLLM
        # TODO: is there a way to get the checkpoint folder directly?
        run_vllm_engine(model_path=SAVE_PATH + "/0", tmp_path=tmp_path, prompts=prompts)
