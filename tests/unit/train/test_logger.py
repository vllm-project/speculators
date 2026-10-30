import logging
import queue
from logging.handlers import QueueListener

import pytest

from speculators.train.logger import (
    FormatDictFilter,
    IsRank0Filter,
    _SnapshotQueueHandler,
    setup_root_logger,
)


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


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _through_listener(records):
    """Run records through the queue handler and a listener."""
    capture = _Capture()
    capture.addFilter(FormatDictFilter())
    log_queue: queue.SimpleQueue = queue.SimpleQueue()
    listener = QueueListener(log_queue, capture, respect_handler_level=True)
    listener.start()
    handler = _SnapshotQueueHandler(log_queue)
    try:
        for record in records:
            handler.handle(record)
    finally:
        listener.stop()
    return capture.lines


def test_queue_handler_keeps_the_flat_metric_formatting():
    """Dict messages render as ``k=v`` even though rendering moved to a thread.

    ``QueueHandler.prepare`` would stringify the message first, leaving
    FormatDictFilter a str; downstream tooling greps the flat form.
    """
    record = _record()
    record.msg = {"train": {"loss": 1.5, "step": 7}}

    (line,) = _through_listener([record])

    assert line == "train/loss=1.500, train/step=7"


def test_queue_handler_snapshots_the_metric_dict():
    """The queued record holds the values from log time, not render time.

    The listener formats on another thread, so a metrics dict reused across
    steps would otherwise print later values against the earlier step.
    """
    metrics = {"train": {"loss": 1.5}}
    record = _record()
    record.msg = metrics

    handler = _SnapshotQueueHandler(queue.SimpleQueue())
    prepared = handler.prepare(record)
    metrics["train"]["loss"] = 99.0

    assert prepared.msg == {"train/loss": 1.5}


def test_setup_root_logger_leaves_a_configured_root_alone():
    """A root that already has handlers keeps them and gets no listener.

    ``basicConfig`` is a no-op there, so an unconditional listener would have
    nothing feeding it and would never stop.
    """
    root = logging.getLogger()
    existing = list(root.handlers)
    try:
        root.handlers = [logging.NullHandler()]
        setup_root_logger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.NullHandler)
    finally:
        root.handlers = existing
