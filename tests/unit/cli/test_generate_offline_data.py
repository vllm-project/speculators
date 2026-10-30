"""Tests for the offline hidden-states generation worker.

``_worker`` is driven directly against a fake vLLM client: the only thing stubbed
out is the network call that produces a safetensors file, so the real staging,
validation and publication path runs against a real filesystem.
"""

import asyncio
import importlib
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from speculators.data_generation.offline import get_existing_hidden_state_indices

# speculators.cli re-exports the command function under the module's own name,
# so reach the module itself to patch its globals.
mod = importlib.import_module("speculators.cli.generate_offline_data")


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    path = tmp_path / "hidden_states"
    path.mkdir()
    return path


def _payload(token_ids: list[int], fill: float = 0.0) -> dict[str, torch.Tensor]:
    """A hidden-states payload for ``token_ids`` in the layout training reads."""
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        # [seq_len, num_layers, hidden_size]
        "hidden_states": torch.full((len(token_ids), 4, 2), fill, dtype=torch.bfloat16),
    }


def _token_ids(path: Path) -> list[int]:
    return load_file(path)["token_ids"].tolist()


def _run_worker(
    monkeypatch,
    tmp_path: Path,
    output_dir: Path,
    items: list[dict[str, Any]],
    replies: list[dict[str, torch.Tensor]],
    *,
    validate_outputs: bool = True,
) -> tuple[dict[str, Any], list[int]]:
    """Run ``_worker`` over ``items`` against a fake server.

    The server saves ``replies[i]`` as the generated file for ``items[i]``.

    Returns the stats dict and the skipped indices.
    """
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    served = iter(replies)

    async def fake_generate_hidden_states_async(_client, _model, item, **_kwargs):
        source_path = server_dir / f"generated_{item['idx']}.safetensors"
        save_file(next(served), source_path)
        return str(source_path), {}

    monkeypatch.setattr(
        mod, "generate_hidden_states_async", fake_generate_hidden_states_async
    )

    stats: dict[str, Any] = {
        "ok": 0,
        "errors": 0,
        "total_vllm_s": 0.0,
        "total_write_s": 0.0,
        "start_time": time.perf_counter(),
    }
    skipped_indices: list[int] = []

    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        for item in items:
            await queue.put(item)
        await queue.put(None)
        with tqdm(total=len(items), disable=True) as pbar:
            await mod._worker(
                client=None,
                model="fake-model",
                queue=queue,
                pbar=pbar,
                vllm_semaphore=asyncio.Semaphore(1),
                write_semaphore=asyncio.Semaphore(1),
                hidden_states_output_dir=output_dir,
                validate_outputs=validate_outputs,
                request_timeout=None,
                max_retries=0,
                fail_on_error=False,
                skipped_indices=skipped_indices,
                cancel_event=asyncio.Event(),
                failure_tracker=None,
                stats=stats,
            )

    asyncio.run(scenario())
    return stats, skipped_indices


@pytest.mark.parametrize(
    "defect",
    [
        pytest.param(lambda ids: _payload([9] * len(ids)), id="wrong-tokens"),
        # Training re-checks token ids on cached files but not the values, so a
        # non-finite file left under its final name would reach training as is.
        pytest.param(lambda ids: _payload(ids, fill=float("nan")), id="non-finite"),
    ],
)
def test_validation_failure_does_not_publish(monkeypatch, tmp_path, output_dir, defect):
    # `hs_<idx>.safetensors` is what the dataloader reads and what resumption
    # counts as done, so a failed sample must leave it as it was: absent for
    # index 0, the previous good file for index 1.
    items = [
        {"idx": 0, "input_ids": [1, 2, 3]},
        {"idx": 1, "input_ids": [4, 5]},
    ]
    previous = output_dir / "hs_1.safetensors"
    save_file(_payload([4, 5]), previous)
    previous_bytes = previous.read_bytes()

    stats, skipped = _run_worker(
        monkeypatch,
        tmp_path,
        output_dir,
        items,
        [defect(item["input_ids"]) for item in items],
    )

    assert stats["errors"] == 2
    assert stats["ok"] == 0
    assert skipped == [0, 1]
    assert get_existing_hidden_state_indices(output_dir) == [1]
    assert previous.read_bytes() == previous_bytes
    # Nothing is left staged.
    assert sorted(p.name for p in output_dir.iterdir()) == ["hs_1.safetensors"]


def test_valid_samples_are_published(monkeypatch, tmp_path, output_dir):
    items = [
        {"idx": 0, "input_ids": [1, 2, 3]},
        {"idx": 1, "input_ids": [4, 5]},
    ]

    stats, skipped = _run_worker(
        monkeypatch,
        tmp_path,
        output_dir,
        items,
        [_payload([1, 2, 3]), _payload([4, 5])],
    )

    assert stats["errors"] == 0
    assert stats["ok"] == 2
    assert skipped == []
    assert _token_ids(output_dir / "hs_0.safetensors") == [1, 2, 3]
    assert _token_ids(output_dir / "hs_1.safetensors") == [4, 5]
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "hs_0.safetensors",
        "hs_1.safetensors",
    ]


def test_publishes_without_validation_when_disabled(monkeypatch, tmp_path, output_dir):
    # Without --validate-outputs the file is published unchecked.
    items = [{"idx": 0, "input_ids": [1, 2, 3]}]

    stats, skipped = _run_worker(
        monkeypatch,
        tmp_path,
        output_dir,
        items,
        [_payload([9, 9, 9])],
        validate_outputs=False,
    )

    assert stats["errors"] == 0
    assert stats["ok"] == 1
    assert skipped == []
    assert _token_ids(output_dir / "hs_0.safetensors") == [9, 9, 9]


# ---------------------------------------------------------------------------
# _publish_hidden_states: the file-operation boundary. The worker tests above
# only observe the end state; these pin down what happens on the way there.
# ---------------------------------------------------------------------------


def _staged(tmp_path: Path, output_dir: Path, token_ids: list[int]) -> tuple[str, Path]:
    """A server-side file ready to publish, and the target name for index 0."""
    source = tmp_path / "server" / "generated.safetensors"
    source.parent.mkdir()
    save_file(_payload(token_ids), source)
    return str(source), output_dir / "hs_0.safetensors"


def test_target_is_absent_while_validation_runs(monkeypatch, tmp_path, output_dir):
    # The final name appearing is the commit point, so validation must run
    # before it. Publishing first and rolling back on failure would pass the
    # worker tests above, but a process that dies mid-check (for example when
    # another worker hits --fail-on-error, which calls os._exit) would leave an
    # unvalidated file published.
    source, target = _staged(tmp_path, output_dir, [1, 2, 3])
    seen: dict[str, Any] = {}

    def probe(_data, _tokens):
        seen["target_exists"] = target.exists()
        seen["finished_indices"] = get_existing_hidden_state_indices(output_dir)
        seen["files"] = [p.name for p in output_dir.iterdir()]

    monkeypatch.setattr(mod, "check_hidden_states", probe)
    mod._publish_hidden_states(source, target, [1, 2, 3])

    assert seen["target_exists"] is False
    assert seen["finished_indices"] == []
    assert seen["files"] == [".hs_0.safetensors.tmp"]
    assert _token_ids(target) == [1, 2, 3]


def test_staged_file_left_by_a_crash_is_replaced(tmp_path, output_dir):
    # A process killed between staging and the rename leaves the staged file
    # behind. Redoing the index must replace it rather than leave another one.
    source, target = _staged(tmp_path, output_dir, [1, 2, 3])
    (output_dir / ".hs_0.safetensors.tmp").write_bytes(b"left by a crash")

    mod._publish_hidden_states(source, target, [1, 2, 3])

    assert [p.name for p in output_dir.iterdir()] == ["hs_0.safetensors"]
    assert _token_ids(target) == [1, 2, 3]


def test_interrupted_copy_publishes_nothing(monkeypatch, tmp_path, output_dir):
    # A cross-filesystem move degrades to copy-then-delete. If the copy dies
    # part-way, the partial bytes must not surface under the final name.
    source, target = _staged(tmp_path, output_dir, [1, 2, 3])

    def interrupted_move(_src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "move", interrupted_move)
    with pytest.raises(OSError, match="No space left"):
        mod._publish_hidden_states(source, target, [1, 2, 3])

    assert list(output_dir.iterdir()) == []


def test_failed_rename_leaves_nothing_staged(tmp_path, output_dir):
    source, target = _staged(tmp_path, output_dir, [1, 2, 3])
    target.mkdir()  # a file cannot be renamed over a directory

    with pytest.raises(OSError):
        mod._publish_hidden_states(source, target, [1, 2, 3])

    assert [p.name for p in output_dir.iterdir()] == ["hs_0.safetensors"]
    assert target.is_dir()
