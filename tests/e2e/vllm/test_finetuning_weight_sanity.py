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

from speculators.utils.loading import load_full_state_dict

logger = logging.getLogger(__name__)

# Learning rate used for training (shared for log and CLI).
LR = "1e-4"
FROZEN_KEY_PATTERNS = ("d2t", "embed_tokens.weight", "t2d")

# Relative distance thresholds: all trainable tensors must have rel_l1 <= REL_L1_MAX;
# at least MIN_CHANGED tensors must have rel_l1 > REL_L1_MIN
# to ensure weights actually changed.
REL_L1_MAX = 0.05
REL_L1_MIN = 1e-4
MIN_CHANGED = 3
EPS = 1e-12


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

    PRETRAINED = "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"
    DATASET = "nm-testing/sharegpt_llama3_8b_hidden_states"

    logger.info("Loading initial weights from %s", PRETRAINED)
    initial = load_full_state_dict(PRETRAINED)
    logger.info("Loaded %d parameter tensors", len(initial))

    # Run short training with low LR
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
            LR,
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
    finetuned = {}
    for f in ckpt_dir.glob("*.safetensors"):
        finetuned.update(load_file(str(f)))
    logger.info("Loaded %d parameter tensors from checkpoint", len(finetuned))

    # Verify same keys
    assert set(initial.keys()) == set(finetuned.keys())

    # Verify same shapes for each key
    for key in initial:
        assert initial[key].shape == finetuned[key].shape, (
            f"Shape mismatch for {key}: "
            f"initial {initial[key].shape} vs finetuned {finetuned[key].shape}"
        )

    # These tensors must remain identical (frozen / not trained)
    for key in sorted(initial.keys()):
        if any(pat in key for pat in FROZEN_KEY_PATTERNS):
            same = torch.equal(initial[key], finetuned[key])
            assert same, (
                f"Tensor {key} must stay identical after finetuning (frozen); "
                f"initial and finetuned differ"
            )
            logger.info("  [frozen] %s: identical", key)

    # Relative L1 distance per tensor (skip frozen: they are already checked identical)
    trainable_keys = [
        k for k in initial if not any(p in k for p in FROZEN_KEY_PATTERNS)
    ]
    logger.info(
        "Computing relative L1 distance for %d trainable tensors (skip %d frozen)",
        len(trainable_keys),
        len(initial) - len(trainable_keys),
    )
    key_to_rel_l1 = {}
    for key in sorted(trainable_keys):
        a = initial[key].flatten().float()
        b = finetuned[key].flatten().float()
        d = a - b
        norm_b_l1 = b.abs().sum().item() + EPS
        rel_l1 = d.abs().sum().item() / norm_b_l1
        max_abs = d.abs().max().item()
        mean_abs = d.abs().mean().item()
        key_to_rel_l1[key] = rel_l1
        logger.info(
            "  %s: rel_l1=%.3e  max|Δ|=%.3e  mean|Δ|=%.3e",
            key, rel_l1, max_abs, mean_abs,
        )

    # All trainable tensors must have relative L1 distance <= REL_L1_MAX
    for key, rel_l1 in key_to_rel_l1.items():
        assert rel_l1 <= REL_L1_MAX, (
            f"Tensor {key} has rel_l1={rel_l1:.4e} > {REL_L1_MAX} "
            f"(weights changed too much)"
        )

    # At least MIN_CHANGED tensors must have rel_l1 > REL_L1_MIN (actually changed)
    changed_keys = [k for k, r in key_to_rel_l1.items() if r > REL_L1_MIN]
    rel_l1_vals = list(key_to_rel_l1.values())
    summary = (
        f"rel_l1: min={min(rel_l1_vals):.3e}, max={max(rel_l1_vals):.3e}, "
        f"count_changed(rel_l1>{REL_L1_MIN})={len(changed_keys)}"
    )
    logger.info("Summary: %s", summary)
    logger.info(
        "Tensors with rel_l1 > %s (count=%d): %s",
        REL_L1_MIN, len(changed_keys),
        [(k, key_to_rel_l1[k]) for k in changed_keys],
    )
    assert len(changed_keys) >= MIN_CHANGED, (
        f"Expected at least {MIN_CHANGED} tensors with rel_l1 > {REL_L1_MIN}, "
        f"got {len(changed_keys)}. {summary}"
    )
