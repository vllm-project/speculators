from types import SimpleNamespace

import pytest

from tests.e2e import utils


class _FakeResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_wait_for_server_requires_stable_health(monkeypatch):
    clock = [0.0]
    statuses = iter([503, 200, 200, 200, 200])

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(utils.time, "sleep", advance)
    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(next(statuses)),
    )

    utils.wait_for_server(
        8000,
        timeout=20,
        poll_interval=1,
        readiness_stability=3,
    )

    assert clock[0] == 4


def test_wait_for_server_checks_process_during_stability(monkeypatch):
    clock = [0.0]
    polls = iter([None, None, 1])
    process = SimpleNamespace(returncode=1, poll=lambda: next(polls))

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(utils.time, "sleep", advance)
    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(200),
    )

    with pytest.raises(RuntimeError, match="exited with code 1"):
        utils.wait_for_server(
            8000,
            timeout=20,
            poll_interval=1,
            readiness_stability=3,
            process=process,
        )
