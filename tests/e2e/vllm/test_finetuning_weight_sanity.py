"""E2E test: verify finetuning modifies weights but keeps them ~95% similar."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from speculators.utils.loading import load_full_state_dict


@pytest.mark.e2e
@pytest.mark.slow
def test_finetuning_weight_sanity(tmp_path: Path):
    """Verify finetuning changes weights but keeps them similar (low LR)."""
    PRETRAINED = "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"
    DATASET = "nm-testing/sharegpt_llama3_8b_hidden_states"

    # Load initial weights
    initial = load_full_state_dict(PRETRAINED)

    # Run short training with low LR
    data_dir = snapshot_download(repo_id=DATASET, repo_type="dataset")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train.py",
            "--pretrained-model-path",
            PRETRAINED,
            "--verifier-name-or-path",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--data-path",
            data_dir,
            "--save-path",
            str(tmp_path / "ckpt"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--epochs",
            "1",
            "--lr",
            "1e-5",
            "--total-seq-len",
            "2048",
            "--data-format-version",
            "1",
            "--num-workers",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    # Load finetuned weights
    ckpt_dir = next((tmp_path / "ckpt").glob("*"))
    finetuned = {}
    for f in ckpt_dir.glob("*.safetensors"):
        finetuned.update(load_file(str(f)))

    # Verify same keys
    assert set(initial.keys()) == set(finetuned.keys())

    # Compute similarities
    similarities = []
    for key in initial:
        w1 = initial[key].flatten().float()
        w2 = finetuned[key].flatten().float()
        sim = torch.nn.functional.cosine_similarity(
            w1.unsqueeze(0), w2.unsqueeze(0)
        ).item()
        similarities.append(sim)

    avg_sim = sum(similarities) / len(similarities)

    # Assert: changed but similar (~95%)
    assert 0.93 <= avg_sim <= 0.9999, f"Expected similarity ~95%, got {avg_sim:.2%}"
    assert avg_sim < 0.9999, "Weights unchanged (too similar)"
