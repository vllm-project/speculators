"""Unit tests for the argparse helpers in the Speculators library."""

import argparse

import pytest

from speculators.utils.argparse_utils import explicitly_provided_dests


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--draft-arch", type=str, default="llama")
    parser.add_argument("--flag", action="store_true")
    return parser


@pytest.mark.smoke
def test_returns_only_options_present_on_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--num-layers", "5"])

    provided = explicitly_provided_dests(_parser(), ["num_layers", "draft_arch"])

    assert provided == {"num_layers"}


@pytest.mark.smoke
def test_value_equal_to_default_still_counts_as_provided(monkeypatch):
    # Passing the default value explicitly must still be detected (the whole point:
    # provided-based, not value-vs-default comparison).
    monkeypatch.setattr("sys.argv", ["prog", "--draft-arch", "llama"])

    provided = explicitly_provided_dests(_parser(), ["num_layers", "draft_arch"])

    assert provided == {"draft_arch"}


@pytest.mark.smoke
def test_empty_when_nothing_provided(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])

    provided = explicitly_provided_dests(_parser(), ["num_layers", "draft_arch"])

    assert provided == set()


@pytest.mark.smoke
def test_store_true_flag_detected(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--flag"])

    provided = explicitly_provided_dests(_parser(), ["flag", "num_layers"])

    assert provided == {"flag"}
