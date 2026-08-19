"""E2E smoke test for the HTTP hidden-states producer/consumer loop.

A writer standing in for the vLLM connector writes a safetensors payload under
the ``.lock`` flock protocol to a directory served by ``scripts/serve_hs.py``;
the test then reads it back via ``HttpTransfer`` (standing in for the trainer
on another node) and validates shape and the lock-blocking semantics.

Only depends on the standard library, so it always runs; the server is
launched automatically on a scratch port.
"""

import fcntl
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

hs_connectors = pytest.importorskip(
    "hs_connectors.transfer", reason="hs_connectors not installed"
)
HttpTransfer = hs_connectors.HttpTransfer

REPO_ROOT = Path(__file__).resolve().parents[3]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def serve_hs(tmp_path: Path):
    """Launch ``scripts/serve_hs.py`` on a scratch port; yield its base URL."""
    port = _free_port()
    root = tmp_path / "hidden_states"
    root.mkdir()
    serve_script = str(REPO_ROOT / "scripts" / "serve_hs.py")
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, repo-controlled script
        [
            sys.executable,
            serve_script,
            "--root",
            str(root),
            "--port",
            str(port),
            "--no-sweeper",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("serve_hs.py exited during startup") from None
            time.sleep(0.05)
    else:
        proc.terminate()
        raise RuntimeError("serve_hs.py did not start within 10s")
    yield base, root
    proc.terminate()
    proc.wait(timeout=5)


def _write_under_lock(root: Path, name: str, tensors: dict[str, torch.Tensor]):
    """Write ``tensors`` the way the vLLM connector does: flock held during write."""
    data_path = root / name
    lock_path = Path(str(data_path) + ".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        save_file(tensors, str(data_path))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return str(data_path)


@pytest.mark.e2e
def test_http_hidden_states_roundtrip(serve_hs):
    """Producer writes under flock; HttpTransfer GETs, validates, and DELETEs."""
    base, root = serve_hs
    hs = torch.randn(32, 4, 16, dtype=torch.bfloat16)
    token_ids = torch.randint(0, 1000, (32,))
    handle = _write_under_lock(
        root, "req_test.safetensors", {"hidden_states": hs, "token_ids": token_ids}
    )

    transfer = HttpTransfer(base, root, timeout=10.0)
    out = transfer.get_generated(handle)

    assert out is not None, "get_generated returned None for an existing sample"
    assert torch.equal(out["hidden_states"], hs)
    assert torch.equal(out["token_ids"], token_ids)

    transfer.delete(handle)
    assert not (root / "req_test.safetensors").exists()
    assert not (root / "req_test.safetensors.lock").exists()


@pytest.mark.e2e
def test_http_get_blocks_until_lock_released(serve_hs):
    """The server must hold the GET until the connector releases the flock."""
    base, root = serve_hs
    hs = torch.randn(8, 2, 16)
    data_path = root / "req_locked.safetensors"
    lock_path = Path(str(data_path) + ".lock")

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    save_file({"hidden_states": hs}, str(data_path))

    hold_seconds = 1.0
    threading.Timer(hold_seconds, lambda: fcntl.flock(fd, fcntl.LOCK_UN)).start()
    try:
        transfer = HttpTransfer(base, root, timeout=10.0)
        start = time.monotonic()
        out = transfer.get_generated(str(data_path))
        elapsed = time.monotonic() - start

        assert out is not None
        assert torch.equal(out["hidden_states"], hs)
        assert elapsed >= hold_seconds, (
            f"GET returned in {elapsed:.2f}s, before the {hold_seconds}s lock release"
        )
    finally:
        os.close(fd)


@pytest.mark.e2e
def test_http_get_rejected_name_maps_404_to_none(serve_hs):
    """A 404 must surface as ``None``, not an exception, so the trainer can retry.

    (The server answers 404 immediately for names it refuses to serve; for a
    valid name whose write has not started yet it holds the GET until the
    connector's lock file appears — see ``serve_hs.py``.)
    """
    base, root = serve_hs
    transfer = HttpTransfer(base, root, timeout=10.0)
    assert transfer.get_generated(str(root / "not_a_safetensors_file.txt")) is None
