"""Proof-of-equivalence: the refactored config layer must reproduce the exact
flat working-dict the pre-refactor argparse parser produced.

The expected values live in ``golden_flat_dict.json`` as one ``_baseline`` (the
all-defaults flat working-dict) plus, per invocation, only the keys that *differ*
from that baseline. This keeps the fixture legible -- each invocation shows just
what it changes -- instead of re-listing ~78 defaults 15 times. Every value was
captured once from ``origin/main`` before this refactor and is a frozen snapshot
of pre-refactor behaviour.

Values are compared with ``==`` (so ``120 == 120.0``), which is the correct
definition of "same input" for the downstream model layer, which consumes this
flat dict via ``**kwargs``. If any case fails, the model layer would observe a
behavioural change.

Regenerating (only if an intentional default change lands): re-capture against
the pre-refactor parser at **this branch's base commit** -- i.e.
``git show $(git merge-base HEAD origin/main):scripts/train.py`` (restore its
``src/speculators/utils/argparse_utils.py`` too) -- NOT against a bare
``origin/main``, which drifts ahead of the base and would silently re-baseline the
"zero functional change" proof against a different parser. Edit ``golden_flat_dict``
and ``golden_cli_flags.json`` in the same commit so the fixture diff documents the
behavioural change on purpose.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts.train import parse_args
from speculators.train.config import CONFIG_DESTS, add_config_cli_arguments
from tests.unit.train.golden_cli_matrix import INVOCATIONS, full_argv

GOLDEN = json.loads((Path(__file__).parent / "golden_flat_dict.json").read_text())
BASELINE = GOLDEN["_baseline"]
GOLDEN_FLAGS = set(
    json.loads((Path(__file__).parent / "golden_cli_flags.json").read_text())
)


def test_baseline_covers_every_config_dest():
    # The baseline is the single source of default values; its key set must be
    # exactly the config surface. Deltas are validated to be a subset per case.
    assert set(BASELINE) == set(CONFIG_DESTS)


def test_generated_flag_strings_match_pre_refactor():
    # Value-equivalence (below) compares dests + values, never the option strings
    # users actually type. This guards the "no flag renamed/removed" constraint
    # directly: the generated CLI's option-string set must equal the pre-refactor
    # parser's frozen snapshot. A schema field rename (which keeps the dest stable
    # in some cases, or which the golden values wouldn't flag) turns this red.
    parser = argparse.ArgumentParser()
    add_config_cli_arguments(parser)
    ours = {opt for action in parser._actions for opt in action.option_strings}
    assert ours == GOLDEN_FLAGS


@pytest.mark.parametrize(
    ("name", "tail"), INVOCATIONS, ids=[name for name, _ in INVOCATIONS]
)
def test_flat_dict_matches_pre_refactor(name, tail):
    delta = GOLDEN[name]
    # Guard the fixture itself: every delta key is a real dest and genuinely
    # differs from the baseline (a stray "delta" equal to the default would be
    # dead weight and mask drift).
    for key, value in delta.items():
        assert key in CONFIG_DESTS, f"{name}: delta key '{key}' is not a config dest"
        assert BASELINE[key] != value, f"{name}: delta key '{key}' equals the baseline"

    expected = {**BASELINE, **delta}

    with mock.patch.object(sys, "argv", full_argv(tail)):
        args = parse_args()
    flat = vars(args)

    # A dropped dest must fail loudly rather than slip through as ``None == None``
    # (which ``flat.get`` would allow for a None-valued default like
    # target_layer_ids).
    missing = set(expected) - set(flat)
    assert not missing, f"{name}: flat working-dict dropped dests: {missing}"

    mismatches = {
        key: (flat[key], expected[key])
        for key in expected
        if flat[key] != expected[key]
    }
    assert not mismatches, f"{name}: flat working-dict drifted: {mismatches}"

    # Only the two new run-mode keys may be added; nothing else.
    extra = set(flat) - set(expected)
    assert extra <= {"config", "dump_config"}, f"{name}: unexpected extra keys {extra}"
