"""E2E test: convert → load → forward → perturb → stitch round-trip for MTP.

Validates that the full MTP pipeline works end-to-end:
1. Convert Qwen/Qwen3.5-0.8B to a speculators MTP checkpoint
2. Load the converted speculator and run a forward pass on GPU
3. Perturb trainable weights to emulate fine-tuning
4. Save the perturbed speculator
5. Stitch the perturbed weights back into the verifier via subprocess
6. Verify MTP weights changed, non-MTP weights unchanged, shapes preserved
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from speculators import SpeculatorModel
from speculators.convert.mtp import MTPConverter
from tests.conftest import requires_cuda, requires_transformers_version
from tests.e2e.utils import run_vllm_engine

logger = logging.getLogger(__name__)

VERIFIER = "Qwen/Qwen3.5-0.8B"
MTP_KEY_PREFIX = "mtp."
NON_MTP_SPOT_CHECK_KEYS = [
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
]


def _load_safetensors_dir(path: Path) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(load_file(str(f), device="cpu"))
    return weights


@pytest.mark.e2e
@pytest.mark.slow
@requires_cuda
@requires_transformers_version("5.2.0")
def test_mtp_roundtrip(tmp_path: Path, seed):
    """Full MTP pipeline: convert, load, forward, perturb, stitch, verify."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    converted_path = tmp_path / "converted"
    perturbed_path = tmp_path / "perturbed"
    stitched_path = tmp_path / "stitched"

    verifier_local_path = Path(snapshot_download(repo_id=VERIFIER, repo_type="model"))
    logger.info("Verifier cached at %s", verifier_local_path)

    # -- Step 1: Convert --
    logger.info("Converting %s to MTP speculator format", VERIFIER)
    converter = MTPConverter()
    converter.convert(
        input_path=VERIFIER,
        output_path=str(converted_path),
        base_model=VERIFIER,
        num_speculative_steps=3,
        validate=False,
    )
    assert converted_path.exists()
    assert any(converted_path.glob("*.safetensors"))
    logger.info("Conversion complete: %s", converted_path)

    # -- Step 2: Load --
    logger.info("Loading converted speculator")
    model: SpeculatorModel = SpeculatorModel.from_pretrained(str(converted_path))  # type: ignore[assignment]
    model = model.cuda()  # type: ignore[call-arg]
    logger.info("Speculator loaded on GPU")

    # -- Step 3: Forward pass --
    logger.info("Running forward pass")
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).cuda()
    hidden_states = torch.randn(
        batch_size, seq_len, model.config.hidden_size, dtype=model.dtype
    ).cuda()

    with torch.no_grad():
        logits_list, loss, _metrics = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
        )

    assert len(logits_list) == model.config.num_speculative_steps
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    logger.info(
        "Forward pass OK: %d logit tensors, loss=%.4f", len(logits_list), loss.item()
    )

    # -- Step 4: Perturb trainable weights --
    model = model.cpu()
    logger.info("Perturbing trainable MTP weights (seed=%d, scale=0.01)", seed)
    num_perturbed = 0
    for _name, param in model.named_parameters():
        if param.requires_grad:
            noise = torch.randn_like(param) * 0.01
            param.data.add_(noise)
            num_perturbed += 1
    assert num_perturbed > 0, "No trainable parameters found"
    logger.info("Perturbed %d parameter tensors", num_perturbed)

    # -- Step 5: Save perturbed --
    logger.info("Saving perturbed speculator to %s", perturbed_path)
    model.save_pretrained(str(perturbed_path))
    del model
    assert any(perturbed_path.glob("*.safetensors"))

    # -- Step 6: Stitch via subprocess --
    logger.info("Stitching perturbed weights back into verifier")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/stitch_mtp.py",
            str(perturbed_path),
            str(verifier_local_path),
            "--output-path",
            str(stitched_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error(
            "Stitch failed (returncode=%d). stderr:\n%s",
            result.returncode,
            result.stderr,
        )
        if result.stdout:
            logger.debug("Stitch stdout:\n%s", result.stdout)
    assert result.returncode == 0, f"Stitch script failed:\n{result.stderr}"
    assert stitched_path.exists()
    logger.info("Stitch complete: %s", stitched_path)

    # -- Step 7-9: Verify stitched weights --
    original_weights = _load_safetensors_dir(verifier_local_path)
    stitched_weights = _load_safetensors_dir(stitched_path)

    mtp_keys = [k for k in original_weights if k.startswith(MTP_KEY_PREFIX)]
    assert len(mtp_keys) > 0, "No MTP keys found in original verifier"

    # Step 7: MTP weights should differ (perturbation was applied)
    mtp_changed = 0
    for key in mtp_keys:
        assert key in stitched_weights, f"MTP key {key} missing from stitched output"
        if not torch.equal(original_weights[key], stitched_weights[key]):
            mtp_changed += 1
    assert mtp_changed > 0, (
        "No MTP weights changed after stitching — perturbation was not reflected"
    )
    logger.info(
        "%d / %d MTP weight tensors changed after stitching", mtp_changed, len(mtp_keys)
    )

    # Step 8: Non-MTP weights should be identical
    for key in NON_MTP_SPOT_CHECK_KEYS:
        assert key in stitched_weights, (
            f"Non-MTP key {key} missing from stitched output"
        )
        assert torch.equal(original_weights[key], stitched_weights[key]), (
            f"Non-MTP weight {key} was modified during stitching"
        )
    logger.info(
        "Non-MTP spot-check weights are identical (verified %d keys)",
        len(NON_MTP_SPOT_CHECK_KEYS),
    )

    # Step 9: All MTP tensor shapes preserved
    for key in mtp_keys:
        assert original_weights[key].shape == stitched_weights[key].shape, (
            f"Shape mismatch for {key}: "
            f"original {original_weights[key].shape} vs "
            f"stitched {stitched_weights[key].shape}"
        )
    logger.info("All %d MTP tensor shapes preserved", len(mtp_keys))

    # -- Step 10: Deploy stitched checkpoint with vLLM MTP --
    logger.info("Running vLLM MTP inference on stitched checkpoint")
    prompts = [
        [{"role": "user", "content": "Write a binary search function in python"}],
    ]
    run_vllm_engine(
        model_path=str(stitched_path),
        tmp_path=tmp_path,
        prompts=prompts,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
        max_model_len=256,
        enforce_eager=True,
    )
    logger.info("vLLM MTP inference passed")
