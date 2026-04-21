"""E2E test: verify finetuning modifies weights but keeps them close (bounded rel L1).

Uses relative L1 distance: ||a-b||_1 / (||b||_1 + eps) per tensor.
All trainable tensors must have rel_l1 <= REL_L1_MAX;
at least a few must have rel_l1 > REL_L1_MIN to ensure they changed.

To see all logs (per-tensor distances, frozen checks, etc.), run:
  pytest tests/e2e/vllm/test_finetuning_weight_sanity.py -s --log-cli-level=INFO
Without -s, pytest captures output and you won't see logger output until failure.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from speculators import Eagle3DraftModel

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.slow
def test_finetuning_weight_sanity(tmp_path: Path):
    """Verify finetuning changes weights but keeps rel L1 distance bounded (low LR)."""
    # Ensure logs are visible when running with -s (no capture)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    # Learning rate used for training (shared for log and CLI).
    FROZEN_KEY_PATTERNS = ("d2t", "embed_tokens.weight", "t2d")
    # Relative distance thresholds: all trainable tensors must have rel_l1 <= REL_L1_MAX
    # at least MIN_CHANGED tensors must have rel_l1 > REL_L1_MIN
    # to ensure weights actually changed.
    LR = "1e-5"
    REL_L1_MAX = 0.05
    REL_L1_MIN = 1e-4
    MIN_CHANGED = 3
    EPS = 1e-12
    PRETRAINED = "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"
    DATASET = "nm-testing/sharegpt_llama3_8b_hidden_states"

    # Get initial state dict
    model = Eagle3DraftModel.from_pretrained(PRETRAINED)
    initial_sd = model.state_dict()
    # Remove verifier weights which aren't saved in checkpoints
    del initial_sd["verifier_norm.weight"]
    del initial_sd["verifier_lm_head.weight"]
    del model

    # Run short training with low LR for single epoch
    logger.info("Downloading dataset %s", DATASET)
    data_dir = snapshot_download(repo_id=DATASET, repo_type="dataset")
    logger.info("Dataset at %s", data_dir)
    logger.info(
        "Running training (1 epoch, lr=%s, save_path=%s)", LR, tmp_path / "ckpt"
    )
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train.py",
            "--from-pretrained",
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
            "2",
            "--lr",
            LR,
            "--total-seq-len",
            "2048",
            "--num-workers",
            "2",
            "--legacy-data",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error(
            "Training failed (returncode=%d). stderr:\n%s",
            result.returncode,
            result.stderr,
        )
        if result.stdout:
            logger.debug("Training stdout:\n%s", result.stdout)
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    logger.info(
        "Training finished. Loading finetuned weights from %s", tmp_path / "ckpt"
    )
    ckpt_dir = next((tmp_path / "ckpt").glob("*"))
    finetuned_sd = {}
    for f in ckpt_dir.glob("*.safetensors"):
        finetuned_sd.update(load_file(str(f)))
    logger.info("Loaded %d parameter tensors from checkpoint", len(finetuned_sd))

    # Verify same keys
    assert set(initial_sd.keys()) == set(finetuned_sd.keys())

    # These tensors must remain identical (frozen / not trained)
    num_changed = 0
    for key in sorted(initial_sd.keys()):
        assert initial_sd[key].shape == finetuned_sd[key].shape, (
            f"Shape mismatch for {key}: "
            f"initial {initial_sd[key].shape} vs finetuned {finetuned_sd[key].shape}"
        )

        if any(pat in key for pat in FROZEN_KEY_PATTERNS):
            assert torch.equal(initial_sd[key], finetuned_sd[key]), (
                f"Tensor {key} must stay identical after finetuning (frozen); "
                f"initial and finetuned differ"
            )
            logger.info("  [frozen] %s: identical", key)
        else:
            # Trainable
            diff = initial_sd[key] - finetuned_sd[key]
            l1_norm_finetuned = finetuned_sd[key].abs().sum() + EPS
            rel_l1 = (diff.abs().sum() / l1_norm_finetuned).item()
            max_abs = diff.abs().max().item()
            mean_abs = diff.abs().mean().item()
            logger.info(
                "  %s: rel_l1=%.3e  max|Δ|=%.3e  mean|Δ|=%.3e",
                key,
                rel_l1,
                max_abs,
                mean_abs,
            )
            assert rel_l1 <= REL_L1_MAX, (
                f"Tensor {key} has rel_l1={rel_l1:.4e} > {REL_L1_MAX} "
                f"(weights changed too much)"
            )

            if rel_l1 >= REL_L1_MIN:
                num_changed += 1

    assert num_changed >= MIN_CHANGED, (
        f"Expected at least {MIN_CHANGED} tensors with rel_l1 > {REL_L1_MIN}, "
        f"got {num_changed}."
    )
