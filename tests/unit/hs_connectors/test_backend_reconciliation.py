"""Drift-guard: every backend train-arg must be auto-registered.

Backend train-args are introspected from each backend's ``add_train_args`` and
auto-registered into the parser by ``resolution.py``.  This test verifies that
the auto-registration mechanism works: every dest a backend registers is
present in ``_BACKEND_DESTS``, and none collides with ``CONFIG_DESTS``.
"""

import argparse

import pytest

from hs_connectors import HiddenStatesBackend
from speculators.train.config.resolution import _BACKEND_DESTS
from speculators.train.config.schema import CONFIG_DESTS


def _backend_train_arg_dests(backend_cls: type[HiddenStatesBackend]) -> set[str]:
    """The argparse dests a backend registers via ``add_train_args``."""
    scratch_parser = argparse.ArgumentParser()
    backend_cls.add_train_args(scratch_parser)
    return {
        action.dest
        for action in scratch_parser._actions
        if action.dest not in ("help", argparse.SUPPRESS)
    }


def test_registry_is_populated():
    assert HiddenStatesBackend.registry, "no backends registered"
    assert "file" in HiddenStatesBackend.registry


def test_no_backend_schema_collision():
    """Backend dests must not collide with schema fields."""
    overlap = _BACKEND_DESTS & CONFIG_DESTS
    assert not overlap, f"Backend dests overlap with CONFIG_DESTS: {overlap}"


@pytest.mark.parametrize(
    ("name", "backend_cls"),
    sorted(HiddenStatesBackend.registry.items()),
    ids=sorted(HiddenStatesBackend.registry),
)
def test_backend_train_args_are_auto_registered(
    name: str, backend_cls: type[HiddenStatesBackend]
):
    dests = _backend_train_arg_dests(backend_cls)
    missing = sorted(dests - _BACKEND_DESTS)
    assert not missing, (
        f"Backend '{name}' registers train-arg dest(s) {missing} via add_train_args "
        f"that are not in _BACKEND_DESTS. This means _collect_backend_args() in "
        f"resolution.py did not pick them up."
    )
