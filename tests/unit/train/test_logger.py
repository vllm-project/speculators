import logging
import queue
from logging.handlers import QueueHandler, QueueListener

import pytest

from speculators.train.logger import (
    FormatDictFilter,
    IsRank0Filter,
    _AsyncConsoleHandler,
    _SnapshotQueueHandler,
    setup_metric_logger,
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
    """``QueueHandler.prepare`` would stringify the message, leaving
    FormatDictFilter a str; downstream tooling greps the ``k=v`` form."""
    record = _record()
    record.msg = {"train": {"loss": 1.5, "step": 7}}

    (line,) = _through_listener([record])

    assert line == "train/loss=1.500, train/step=7"


def test_queue_handler_snapshots_the_metric_dict():
    """The listener formats on another thread, so a metrics dict reused
    across steps would otherwise print later values for this step."""
    metrics = {"train": {"loss": 1.5}}
    record = _record()
    record.msg = metrics

    handler = _SnapshotQueueHandler(queue.SimpleQueue())
    prepared = handler.prepare(record)
    metrics["train"]["loss"] = 99.0

    assert prepared.msg == {"train/loss": 1.5}


@pytest.fixture
def restore_logging():
    """Put the root and metric loggers back as the session found them."""
    root = logging.getLogger()
    metrics = logging.getLogger("speculators.metrics")
    saved = (list(root.handlers), list(metrics.handlers), metrics.propagate)
    yield
    for handler in metrics.handlers:
        handler.close()
    root.handlers, metrics.handlers, metrics.propagate = saved


def test_the_root_console_handler_renders_inline(restore_logging):
    """Everything but metrics renders on the calling thread, so a queue
    cannot still be holding a warning or a traceback at exit."""
    root = logging.getLogger()
    root.handlers = []

    setup_root_logger()

    assert root.handlers
    assert not any(isinstance(h, QueueHandler) for h in root.handlers)


def test_metrics_render_off_thread_and_skip_the_root(restore_logging, tmp_path):
    """Only the hot path is async. Without ``propagate=False`` the root would
    render every metric a second time, synchronously."""
    setup_metric_logger(loggers="", run_name="run", output_dir=str(tmp_path))

    metrics = logging.getLogger("speculators.metrics")
    console = [h for h in metrics.handlers if isinstance(h, _AsyncConsoleHandler)]

    assert len(console) == 1
    assert metrics.propagate is False


def test_the_rank_filter_runs_before_the_record_is_queued(monkeypatch, clean_rank_env):
    """With the filter on the listener's handler instead, every rank would
    copy, flatten and enqueue each metric only to discard it."""
    handler = _AsyncConsoleHandler()
    handler.addFilter(IsRank0Filter())
    prepared = []
    original = handler.prepare

    def spy(record):
        prepared.append(record)
        return original(record)

    monkeypatch.setattr(handler, "prepare", spy)

    try:
        monkeypatch.setenv("RANK", "1")
        handler.handle(_record())
        assert prepared == []

        monkeypatch.setenv("RANK", "0")
        handler.handle(_record())
        assert len(prepared) == 1
    finally:
        handler.close()


def test_closing_the_console_handler_drains_the_queue():
    """``logging.shutdown`` closes handlers at exit; queued metrics must print."""
    handler = _AsyncConsoleHandler()
    listener = handler._listener
    assert listener is not None
    capture = _Capture()
    capture.addFilter(FormatDictFilter())
    listener.handlers = (capture,)
    record = _record()
    record.msg = {"train": {"loss": 1.5}}

    handler.handle(record)
    handler.close()

    assert capture.lines == ["train/loss=1.500"]
    assert handler._listener is None
