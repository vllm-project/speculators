from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from tests.e2e import utils

if TYPE_CHECKING:
    import subprocess


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
    process = cast(
        "subprocess.Popen",
        SimpleNamespace(returncode=1, poll=lambda: next(polls)),
    )

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


def test_launch_vllm_server_uses_an_isolated_process_group(monkeypatch, tmp_path):
    process = Mock(pid=1234, returncode=None)
    process.poll.return_value = None
    popen = Mock(return_value=process)

    monkeypatch.setattr(utils.subprocess, "Popen", popen)
    monkeypatch.setattr(utils, "wait_for_server", Mock())

    result = utils.launch_vllm_server(
        "model",
        8000,
        str(tmp_path / "hidden_states"),
    )

    assert result is process
    assert popen.call_args.kwargs["start_new_session"] is True


def test_stop_vllm_server_cleans_up_the_process_group(monkeypatch):
    process = Mock(pid=1234, returncode=-15)
    process.poll.return_value = None
    process.wait.return_value = None
    killpg = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(utils.os, "killpg", killpg)

    utils.stop_vllm_server(process)

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=30)
    assert killpg.call_args_list[0].args == (1234, utils.signal.SIGTERM)
    assert killpg.call_args_list[1].args == (1234, 0)
