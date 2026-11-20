"""Numerical contracts recorded before the DFlash backbone extraction.

To regenerate, run this file with --record using speculators from reference
revision 988ed16b1f91315942049da3093d88dba70c45d9 on PYTHONPATH. Never regenerate
the fixture from the implementation under test to accommodate a refactor.
"""

import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.xpress import XPressDraftModel, XPressSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.optimizers import split_named_params_for_muon

REFERENCE_REVISION = "988ed16b1f91315942049da3093d88dba70c45d9"
REFERENCE_PATH = Path(__file__).parent / "data" / "dflash_family_reference.json"
RECIPES = {
    "dflash": (DFlashDraftModel, DFlashSpeculatorConfig, {}),
    "dflash2": (
        DFlash2DraftModel,
        DFlash2SpeculatorConfig,
        {"conv_group_size": 4, "selector_rank": 4, "selector_top_k": 4},
    ),
    "dspark": (DSparkDraftModel, DSparkSpeculatorConfig, {"markov_rank": 4}),
    "xpress": (XPressDraftModel, XPressSpeculatorConfig, {"xpress_rank": 4}),
}


def legacy_name(name, recipe):
    """Compare physical tensors independently of their new owning module."""
    if name.startswith("correction."):
        prefix = {
            "dspark": "markov_head",
            "dflash2": "candidate_selector",
            "xpress": "refiner_head",
        }[recipe]
        return prefix + name.removeprefix("correction")
    return name.removeprefix("backbone.")


def make_contract_model(recipe, sample_from_anchor):
    model_class, config_class, extra = RECIPES[recipe]
    config = config_class(
        transformer_layer_config=Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            _attn_implementation="eager",  # type: ignore[call-arg]
        ),
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
        speculators_config=SpeculatorsConfig(
            algorithm=recipe,
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
        **extra,
    )
    model = model_class(config).float()
    # Depend on tensor identity, not construction order or RNG consumption.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            offset = int(
                hashlib.sha256(legacy_name(name, recipe).encode()).hexdigest()[:4], 16
            )
            values = torch.sin(torch.arange(parameter.numel()).float() + offset)
            values = values.reshape(parameter.shape)
            parameter.copy_(1 + 0.02 * values if "norm" in name else 0.1 * values)
    return model


def summarize(tensor):
    values = tensor.detach().float().flatten()
    return {"probe": values[:8].tolist(), "norm": values.norm().item()}


def contract_report(recipe, sample_from_anchor, model=None):
    if model is None:
        model = make_contract_model(recipe, sample_from_anchor)
    captured = {}

    def capture_projection(module, args, output):
        captured["hidden"] = summarize(args[0])
        captured["base_logits"] = summarize(output)

    handle = model.lm_head.register_forward_hook(capture_projection)
    module = importlib.import_module(type(model).__module__)
    original_metrics = module.compute_metrics

    def capture_metrics(*args, **kwargs):
        logits = kwargs.get("unary_logits") if not args else args[0]
        captured["objective_logits"] = summarize(logits)
        if "correction" in kwargs:
            captured["candidate_logits"] = summarize(
                kwargs["correction"].candidate_logits
            )
        elif "candidate_logits" in kwargs:
            captured["candidate_logits"] = summarize(kwargs["candidate_logits"])
        return original_metrics(*args, **kwargs)

    # Two packed documents and nontrivial position IDs exercise rotary alignment.
    batch = {
        "hidden_states": torch.cos(torch.arange(24 * 32).float()).view(1, 24, 32),
        "input_ids": (torch.arange(24).view(1, -1) * 7) % 32,
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1] * 3]).float(),
        "verifier_last_hidden_states": torch.sin(torch.arange(24 * 16).float()).view(
            1, 24, 16
        ),
        "document_ids": torch.tensor([[0] * 12 + [1] * 12]),
        "position_ids": torch.tensor([list(range(12)) * 2]),
    }
    # Reset immediately before sampling so constructor RNG never affects anchors.
    torch.manual_seed(42)
    with patch.object(module, "compute_metrics", capture_metrics):
        _, loss, metrics = model(
            **batch,
            max_anchors=3,
            loss_config=resolve_loss_config("kl_div", "eager"),
        )
    handle.remove()
    loss.backward()
    muon, adamw = split_named_params_for_muon(model)
    return {
        "keys": sorted(legacy_name(name, recipe) for name in model.state_dict()),
        "trainable": sorted(
            legacy_name(name, recipe)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        "omitted": sorted(model._keys_to_ignore_on_save),
        "muon": sorted(legacy_name(name, recipe) for name, _ in muon),
        "adamw": sorted(legacy_name(name, recipe) for name, _ in adamw),
        "outputs": captured,
        "loss": loss.item(),
        "metrics": {
            key: value.item()
            for key, value in metrics.items()
            if key in {"eal_sum", "eal_total", "full_acc_sum", "full_acc_total"}
        },
        "gradients": {
            legacy_name(name, recipe): summarize(parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
    }


def assert_report_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_report_close(actual[key], expected[key])
    elif isinstance(expected, list) and expected and isinstance(expected[0], str):
        assert actual == expected
    else:
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(expected), rtol=2e-4, atol=2e-6
        )


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_legacy_recipe_numerical_contract(recipe, sample_from_anchor):
    reference = json.loads(REFERENCE_PATH.read_text())
    assert reference["revision"] == REFERENCE_REVISION
    actual = contract_report(recipe, sample_from_anchor)
    expected = reference["recipes"][f"{recipe}:{sample_from_anchor}"]
    if recipe == "dflash2":
        # The old accuracy counted unary predictions. The selector tests cover
        # the new accuracy, including a target injected outside the original top-k.
        for report in (actual, expected):
            report["metrics"].pop("full_acc_sum")
    elif recipe == "xpress":
        # The pre-refactor fixture predates verifier-based XPress greedy metrics.
        for key in ("full_acc_sum", "full_acc_total", "eal_sum", "eal_total"):
            actual["metrics"].pop(key)
    assert_report_close(actual, expected)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", required=True)
    parser.parse_args()
    torch.set_num_threads(1)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(
            {
                "revision": REFERENCE_REVISION,
                "torch": torch.__version__,
                "recipes": {
                    f"{recipe}:{sample}": contract_report(recipe, sample)
                    for recipe in RECIPES
                    for sample in (False, True)
                },
            },
            indent=2,
        )
        + "\n"
    )
