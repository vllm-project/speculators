import logging

import pytest

from speculators.train.logger import IsRank0Filter


def _record(**extra):
    record = logging.LogRecord(
        "speculators", logging.INFO, __file__, 0, "msg", None, None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


@pytest.fixture
def clean_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_global_rank0_filter_passes_only_global_rank0(monkeypatch, clean_rank_env):
    # Multi-node: a non-zero global rank that happens to be local_rank 0
    # must still be filtered out (the bug this guards against).
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert IsRank0Filter().filter(_record()) is False

    monkeypatch.setenv("RANK", "0")
    assert IsRank0Filter().filter(_record()) is True


def test_override_bypasses_filter(clean_rank_env, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert IsRank0Filter().filter(_record(override_rank0_filter=True)) is True
