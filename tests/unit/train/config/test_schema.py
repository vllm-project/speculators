"""Data-model seam tests for the train config schema.

These exercise the schema at its two pure seams -- ``flatten()`` and
``from_flat()`` -- not CLI parsing or YAML. Backward compatibility against the
real parser is proven separately by the example-recipe tests, not a golden
``vars(args)`` snapshot here.
"""

import pytest

from speculators.train.config import TrainConfig
from speculators.train.config.schema import (
    CONFIG_DESTS,
    DraftArgs,
    OptimizerArgs,
)


def test_constructs_from_defaults():
    # The whole point of the schema seam: a config exists with no inputs.
    TrainConfig()


def test_flatten_covers_exactly_the_schema_fields():
    # flatten() emits every schema dest and nothing else; consumers bind the flat
    # dict by name (**kwargs / args.<field>), so the key set is the contract.
    flat = TrainConfig().flatten()
    assert set(flat) == CONFIG_DESTS
    # Order is deterministic (declaration order) so the run.yaml dump stays stable.
    assert list(flat) == list(TrainConfig(speculator_type="dflash").flatten())


def test_flatten_resolves_eagle3_derived_defaults():
    # Mirrors the tail of the pre-refactor parse_args for the default (eagle3) run.
    flat = TrainConfig().flatten()
    assert flat["draft_arch"] == "llama"
    assert flat["norm_before_fc"] is True
    assert flat["norm_output"] is True
    assert flat["muon_lr"] == pytest.approx(10 * flat["lr"])


def test_flatten_resolves_non_eagle3_derived_defaults():
    flat = TrainConfig(speculator_type="dflash").flatten()
    assert flat["draft_arch"] == "qwen3"
    assert flat["norm_before_fc"] is False
    assert flat["norm_output"] is False


def test_from_flat_inverts_flatten():
    cfg = TrainConfig(
        speculator_type="dspark",
        draft=DraftArgs(num_layers=4, full_attention_indices=[2, 18, 33]),
        optimizer=OptimizerArgs(lr=3e-4),
    )
    assert TrainConfig.from_flat(cfg.flatten()) == cfg


def test_from_flat_default_round_trip():
    cfg = TrainConfig()
    assert TrainConfig.from_flat(cfg.flatten()) == cfg


def test_from_flat_ignores_non_config_keys():
    flat = TrainConfig().flatten()
    flat["config"] = "run.yaml"
    flat["dump_config"] = True
    recovered = TrainConfig.from_flat(flat)
    assert recovered == TrainConfig()


def test_from_flat_accepts_partial_working_dict():
    recovered = TrainConfig.from_flat({"lr": 5e-4, "num_layers": 6})
    assert recovered.optimizer.lr == pytest.approx(5e-4)
    assert recovered.draft.num_layers == 6
    # Untouched fields fall back to their schema defaults.
    assert recovered.trainer.epochs == 20
