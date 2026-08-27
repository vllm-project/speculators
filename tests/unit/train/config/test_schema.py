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
    DFlash2Args,
    DFlashArgs,
    DraftArgs,
    LossArgs,
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


def test_flatten_resolves_dflash_derived_defaults():
    # Best-practices recipe from https://github.com/vllm-project/speculators/issues/979:
    # dflash gets 5 draft layers, D-PACE weighting, CE loss, and block_size=16
    # out of the box.
    flat = TrainConfig(speculator_type="dflash").flatten()
    assert flat["num_layers"] == 5
    assert flat["per_position_loss_weight"] == "dpace"
    assert flat["loss_fn"] == "ce"
    assert flat["block_size"] == 16
    assert flat["sliding_window_non_causal"] is False


def test_flatten_resolves_dflash2_derived_defaults():
    flat = TrainConfig(speculator_type="dflash2").flatten()
    assert flat["num_layers"] == 5
    assert flat["per_position_loss_weight"] == "fixed-exp-decay"
    assert flat["loss_fn"] == "kl_div"
    assert flat["block_size"] == 8
    assert flat["conv_kernel_size"] == 2
    assert flat["conv_group_size"] == 16
    assert flat["selector_rank"] == 256
    assert flat["selector_top_k"] == 16
    assert flat["selector_loss_alpha"] == pytest.approx(1.0)
    assert flat["sliding_window_non_causal"] is True


def test_flatten_leaves_non_dflash_derived_defaults_unchanged():
    # DSpark shares only the DFlash layer default; the remaining derived defaults
    # keep their pre-existing behavior.
    for speculator_type in ("eagle3", "dspark", "peagle", "mtp"):
        flat = TrainConfig(speculator_type=speculator_type).flatten()
        assert flat["num_layers"] == (5 if speculator_type == "dspark" else 1)
        assert flat["per_position_loss_weight"] == "fixed-exp-decay"
        assert flat["loss_fn"] == "kl_div"
        assert flat["block_size"] == 8
        assert flat["sliding_window_non_causal"] is False


def test_dflash_derived_defaults_do_not_override_explicit_values():
    cfg = TrainConfig(
        speculator_type="dflash",
        draft=DraftArgs(num_layers=3),
        loss=LossArgs(loss_fn="kl_div"),
        dflash=DFlashArgs(per_position_loss_weight="fixed-exp-decay", block_size=8),
    )
    assert cfg.draft.num_layers == 3
    assert cfg.loss.loss_fn == "kl_div"
    assert cfg.dflash.per_position_loss_weight == "fixed-exp-decay"
    assert cfg.dflash.block_size == 8


def test_dflash2_explicit_values_override_defaults():
    cfg = TrainConfig(
        speculator_type="dflash2",
        draft=DraftArgs(num_layers=3, sliding_window_non_causal=False),
        loss=LossArgs(loss_fn="ce"),
        dflash=DFlashArgs(block_size=4),
        dflash2=DFlash2Args(
            conv_kernel_size=3,
            conv_group_size=8,
            selector_rank=128,
            selector_top_k=32,
            selector_loss_alpha=0.25,
        ),
    )
    flat = cfg.flatten()
    assert flat["num_layers"] == 3
    assert flat["loss_fn"] == "ce"
    assert flat["block_size"] == 4
    assert flat["sliding_window_non_causal"] is False
    assert flat["conv_kernel_size"] == 3
    assert flat["conv_group_size"] == 8
    assert flat["selector_rank"] == 128
    assert flat["selector_top_k"] == 32
    assert flat["selector_loss_alpha"] == pytest.approx(0.25)


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
