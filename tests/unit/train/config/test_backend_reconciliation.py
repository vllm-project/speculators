"""Drift-guard: every backend train-arg must be mirrored as a schema field.

``hs_connectors`` is speculators-agnostic (argparse-based, no pydantic) because
vLLM consumes it standalone. So speculators no longer builds its parser by
calling each backend's ``add_train_args``; the schema generates the whole flag
surface, and each backend's train-args are *mirrored* as schema fields (e.g.
``FileBackend``'s ``--hidden-states-path`` is mirrored as ``DataArgs``'
``hidden_states_path``).

That mirroring is the risk this test guards. :func:`resolution.resolve` filters
the parsed namespace down to :data:`~schema.CONFIG_DESTS`, so a backend train-arg
with no matching schema field would be *silently dropped* during resolution. A
new backend (NIXL/RDMA is on the roadmap) could introduce that gap without any
error. This test makes the gap loud: for every registered backend it collects the
argparse dests ``add_train_args`` registers and asserts each one is a config dest.
"""

import argparse

import pytest

from hs_connectors import HiddenStatesBackend
from speculators.train.config.schema import CONFIG_DESTS


def _backend_train_arg_dests(backend_cls: type[HiddenStatesBackend]) -> set[str]:
    """The argparse dests a backend registers via ``add_train_args``.

    Introspected from a throwaway parser's ``_actions``, skipping argparse's
    auto-added ``help`` action (and any suppressed dest) so only genuine
    backend train-args remain.
    """
    scratch_parser = argparse.ArgumentParser()
    backend_cls.add_train_args(scratch_parser)
    return {
        action.dest
        for action in scratch_parser._actions
        if action.dest not in ("help", argparse.SUPPRESS)
    }


def test_registry_is_populated_with_file_backend():
    # Prove the parametrized test below is exercising something real: the registry
    # is non-empty and the one backend that exists today mirrors its train-arg.
    assert HiddenStatesBackend.registry, "no backends registered"
    assert "file" in HiddenStatesBackend.registry
    assert "hidden_states_path" in _backend_train_arg_dests(
        HiddenStatesBackend.registry["file"]
    )
    assert "hidden_states_path" in CONFIG_DESTS


@pytest.mark.parametrize(
    ("name", "backend_cls"),
    sorted(HiddenStatesBackend.registry.items()),
    ids=sorted(HiddenStatesBackend.registry),
)
def test_backend_train_args_are_mirrored_in_schema(
    name: str, backend_cls: type[HiddenStatesBackend]
):
    # Every dest a backend adds via add_train_args must have a matching schema
    # field, or resolve() (which filters to CONFIG_DESTS) drops the value.
    dests = _backend_train_arg_dests(backend_cls)
    missing = sorted(dests - CONFIG_DESTS)
    assert not missing, (
        f"Backend '{name}' registers train-arg dest(s) {missing} via add_train_args "
        f"that have no matching schema field. speculators mirrors backend train-args "
        f"as schema fields; add them (e.g. to DataArgs) or resolve() will silently "
        f"drop them (filtered by CONFIG_DESTS)."
    )
