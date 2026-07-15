import errno
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.data_generation_offline import (
    _align_and_write_hidden_states,
    parse_args as parse_offline_args,
)
from speculators.data_generation.offline import (
    InvalidHiddenStateCacheError,
    align_hidden_states_to_tokens,
    atomic_move_safetensors,
    atomic_save_safetensors,
    durable_unlink_safetensors,
    get_existing_hidden_state_indices,
    get_indices_to_process,
    hidden_states_file_sha256,
    validate_hidden_states_file_contents,
    validate_hidden_states_path,
    validate_hidden_states_tensors,
)


def _hidden_state_tensors(
    tokens: list[int],
    *,
    value: float | None = None,
) -> dict[str, torch.Tensor]:
    shape = (len(tokens), 2, 3)
    hidden_states = (
        torch.arange(len(tokens) * 6, dtype=torch.float32).reshape(shape)
        if value is None
        else torch.full(shape, value, dtype=torch.float32)
    )
    return {
        "token_ids": torch.tensor(tokens, dtype=torch.long),
        "hidden_states": hidden_states,
    }


def _save_hidden_states(
    path: Path,
    tokens: list[int],
    *,
    value: float | None = None,
) -> None:
    save_file(_hidden_state_tensors(tokens, value=value), path)


def _cache_paths(
    tmp_path: Path,
    *,
    same_root: bool = False,
) -> tuple[Path, Path, Path, Path]:
    source_root = tmp_path / "cache" if same_root else tmp_path / "source"
    target_root = source_root if same_root else tmp_path / "target"
    source_root.mkdir()
    if target_root != source_root:
        target_root.mkdir()
    return (
        source_root,
        target_root,
        source_root / "response.safetensors",
        target_root / "hs_0.safetensors",
    )


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--max-retries", "-1", "expected a non-negative integer"),
        ("--concurrency", "0", "positive integer"),
        ("--concurrency", "-1", "positive integer"),
        ("--request-timeout", "0", "finite positive number"),
        ("--request-timeout", "-1", "finite positive number"),
        ("--request-timeout", "nan", "finite positive number"),
        ("--request-timeout", "inf", "finite positive number"),
        ("--request-timeout", "-inf", "finite positive number"),
    ],
)
def test_offline_cli_rejects_invalid_values(monkeypatch, capsys, flag, value, message):
    monkeypatch.setattr("sys.argv", ["data_generation_offline.py", flag, value])
    with pytest.raises(SystemExit) as exc_info:
        parse_offline_args()
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_align_hidden_states_to_tokens_truncates_prefix_match():
    data = _hidden_state_tensors([1, 2, 3, 4, 5])
    aligned, truncated = align_hidden_states_to_tokens(
        data,
        [1, 2, 3],
        allow_prefix_truncation=True,
    )
    assert truncated is True
    assert aligned["token_ids"].tolist() == [1, 2, 3]
    assert aligned["hidden_states"].shape == (3, 2, 3)
    assert torch.equal(aligned["hidden_states"], data["hidden_states"][:3])


def test_align_hidden_states_to_tokens_rejects_non_prefix_mismatch():
    data = _hidden_state_tensors([1, 9, 3])
    with pytest.raises(ValueError, match="Token ids don't match"):
        align_hidden_states_to_tokens(
            data,
            [1, 2, 3],
            allow_prefix_truncation=True,
        )


def test_align_and_write_fast_path_moves_without_full_load(tmp_path, monkeypatch):
    source_root, target_root, source, target = _cache_paths(tmp_path)
    _save_hidden_states(source, [1, 2, 3])

    def raise_if_loaded(*args, **kwargs):
        del args, kwargs
        raise AssertionError("load_file should not be called")

    fsynced = []
    monkeypatch.setattr("scripts.data_generation_offline.load_file", raise_if_loaded)
    monkeypatch.setattr(
        "speculators.data_generation.offline._fsync_directory",
        lambda path: fsynced.append(Path(path).resolve()),
    )
    _align_and_write_hidden_states(
        source,
        target,
        [1, 2, 3],
        source_root=source_root,
        target_root=target_root,
        allow_prefix_truncation=False,
        validate_outputs=False,
    )
    assert not source.exists()
    assert load_file(target)["token_ids"].tolist() == [1, 2, 3]
    assert set(fsynced) == {source_root.resolve(), target_root.resolve()}


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("missing-hidden-states", ValueError, "missing required tensors"),
        ("wrong-tokens", InvalidHiddenStateCacheError, "do not match"),
    ],
)
def test_align_and_write_fast_path_rejects_invalid_source(
    tmp_path,
    case,
    error,
    message,
):
    source_root, target_root, source, target = _cache_paths(tmp_path)
    if case == "missing-hidden-states":
        save_file({"token_ids": torch.tensor([1, 2, 3])}, source)
    else:
        _save_hidden_states(source, [9, 9, 9])

    with pytest.raises(error, match=message):
        _align_and_write_hidden_states(
            source,
            target,
            [1, 2, 3],
            source_root=source_root,
            target_root=target_root,
            allow_prefix_truncation=False,
            validate_outputs=False,
        )
    assert source.is_file()
    assert not target.exists()


def test_align_and_write_fast_path_restores_source_replacement(tmp_path, monkeypatch):
    source_root, target_root, source, target = _cache_paths(tmp_path)
    replacement = source_root / "replacement.safetensors"
    _save_hidden_states(source, [1, 2, 3])
    _save_hidden_states(replacement, [1, 2, 3], value=42)
    replacement_sha256 = hidden_states_file_sha256(
        replacement,
        allowed_root=source_root,
    )
    real_replace = os.replace

    def swap_during_direct_rename(source_path, target_path):
        if Path(source_path) == source and Path(target_path) == target:
            real_replace(replacement, source)
        return real_replace(source_path, target_path)

    monkeypatch.setattr(
        "speculators.data_generation.offline.os.replace",
        swap_during_direct_rename,
    )
    with pytest.raises(FileExistsError, match="changed during the atomic move"):
        _align_and_write_hidden_states(
            source,
            target,
            [1, 2, 3],
            source_root=source_root,
            target_root=target_root,
            allow_prefix_truncation=False,
            validate_outputs=False,
        )
    assert not target.exists()
    assert hidden_states_file_sha256(source, allowed_root=source_root) == (
        replacement_sha256
    )


def test_validate_hidden_states_path_accepts_file_and_pending_leaf(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    source = root / "request.safetensors"
    source.write_bytes(b"hidden states")
    pending = root / "pending.safetensors"

    assert validate_hidden_states_path(source, root) == source.resolve()
    assert validate_hidden_states_path(
        pending,
        root,
        require_exists=False,
    ) == pending.resolve()
    with pytest.raises(ValueError, match="does not exist"):
        validate_hidden_states_path(pending, root)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("relative", "must be absolute"),
        ("suffix", "must end in .safetensors"),
        ("directory", "not a regular file"),
        ("outside", "outside the allowed root"),
        ("traversal", "outside the allowed root"),
        ("file-symlink", "symlink component"),
        ("parent-symlink", "symlink component"),
    ],
)
def test_validate_hidden_states_path_rejects_unsafe_paths(tmp_path, case, message):
    root = tmp_path / "shared"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    if case == "relative":
        candidate = "relative.safetensors"
    elif case == "suffix":
        candidate = root / "source.bin"
        candidate.write_bytes(b"hidden states")
    elif case == "directory":
        candidate = root / "directory.safetensors"
        candidate.mkdir()
    elif case in {"outside", "traversal"}:
        outside_file = outside / "outside.safetensors"
        outside_file.write_bytes(b"hidden states")
        if case == "outside":
            candidate = outside_file
        else:
            (root / "nested").mkdir()
            candidate = root / "nested" / ".." / ".." / outside.name / outside_file.name
    else:
        real_dir = root / "real"
        real_dir.mkdir()
        real_file = real_dir / "source.safetensors"
        real_file.write_bytes(b"hidden states")
        if case == "file-symlink":
            candidate = root / "linked-file.safetensors"
            candidate.symlink_to(real_file)
        else:
            linked_dir = root / "linked-dir"
            linked_dir.symlink_to(real_dir, target_is_directory=True)
            candidate = linked_dir / real_file.name

    with pytest.raises(ValueError, match=message):
        validate_hidden_states_path(candidate, root)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("token-rank", "one-dimensional"),
        ("empty-tokens", "must not be empty"),
        ("token-dtype", "integer dtype"),
        ("hidden-rank", "exactly three non-empty dimensions"),
        ("hidden-empty", "exactly three non-empty dimensions"),
        ("hidden-dtype", "floating dtype"),
        ("length", "sequence length does not match"),
    ],
)
def test_validate_hidden_states_file_rejects_invalid_structure(tmp_path, case, message):
    token_ids = torch.tensor([1, 2], dtype=torch.long)
    hidden_states = torch.randn(2, 2, 3)
    if case == "token-rank":
        token_ids = token_ids.unsqueeze(0)
    elif case == "empty-tokens":
        token_ids = torch.tensor([], dtype=torch.long)
        hidden_states = torch.empty(0, 2, 3)
    elif case == "token-dtype":
        token_ids = token_ids.float()
    elif case == "hidden-rank":
        hidden_states = torch.randn(2, 3)
    elif case == "hidden-empty":
        hidden_states = torch.empty(2, 0, 3)
    elif case == "hidden-dtype":
        hidden_states = torch.ones(2, 2, 3, dtype=torch.long)
    else:
        hidden_states = torch.randn(1, 2, 3)

    target = tmp_path / "hs_0.safetensors"
    save_file({"token_ids": token_ids, "hidden_states": hidden_states}, target)
    with pytest.raises(InvalidHiddenStateCacheError, match=message):
        validate_hidden_states_file_contents(target, tmp_path)


def test_hidden_state_validators_accept_mtp_final_layer_only(tmp_path):
    tensors = {
        "token_ids": torch.tensor([1, 2], dtype=torch.long),
        "hidden_states": torch.randn(2, 1, 3),
    }
    target = tmp_path / "hs_0.safetensors"
    save_file(tensors, target)
    assert validate_hidden_states_file_contents(target, tmp_path) == [1, 2]
    validate_hidden_states_tensors(tensors)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_validate_hidden_states_file_streams_nonfinite_values(tmp_path, bad_value):
    target = tmp_path / "hs_0.safetensors"
    tensors = _hidden_state_tensors(list(range(257)))
    tensors["hidden_states"][200, 1, 2] = bad_value
    save_file(tensors, target)

    validate_hidden_states_file_contents(target, tmp_path, validate_values=False)
    with pytest.raises(InvalidHiddenStateCacheError, match="NaN or Inf"):
        validate_hidden_states_file_contents(target, tmp_path, validate_values=True)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("outside", "outside the allowed root"),
        ("managed", "another managed cache entry"),
        ("nested-managed", "another managed cache entry"),
    ],
)
def test_align_and_write_rejects_unsafe_source(tmp_path, case, message):
    broad_root = tmp_path / "shared"
    broad_root.mkdir()
    target_root = broad_root
    source_root = broad_root
    target = target_root / "hs_0.safetensors"
    if case == "outside":
        source = tmp_path / "outside.safetensors"
    elif case == "managed":
        source = target_root / "hs_1.safetensors"
    else:
        target_root = broad_root / "output"
        target_root.mkdir()
        source = target_root / "hs_1.safetensors"
        target = target_root / "hs_0.safetensors"
    _save_hidden_states(source, [1, 2, 3])

    with pytest.raises(ValueError, match=message):
        _align_and_write_hidden_states(
            source,
            target,
            [1, 2, 3],
            source_root=source_root,
            target_root=target_root,
            allow_prefix_truncation=False,
            validate_outputs=False,
        )
    assert source.is_file()
    assert not target.exists()


def test_align_and_write_truncated_source_equals_target(tmp_path, monkeypatch):
    root, _, source, _ = _cache_paths(tmp_path, same_root=True)
    source = root / "hs_0.safetensors"
    _save_hidden_states(source, [1, 2, 3, 4])
    fsynced = []
    monkeypatch.setattr(
        "speculators.data_generation.offline._fsync_directory",
        lambda path: fsynced.append(Path(path).resolve()),
    )
    _align_and_write_hidden_states(
        source,
        source,
        [1, 2, 3],
        source_root=root,
        target_root=root,
        allow_prefix_truncation=True,
        validate_outputs=False,
    )
    loaded = load_file(source)
    assert loaded["token_ids"].tolist() == [1, 2, 3]
    assert loaded["hidden_states"].shape == (3, 2, 3)
    assert fsynced == [root.resolve()]
    assert not list(root.glob(".*.tmp"))


def test_truncated_write_commits_before_source_unlink(tmp_path, monkeypatch):
    source_root, target_root, source, target = _cache_paths(tmp_path)
    _save_hidden_states(source, [1, 2, 3, 4])
    real_unlink = durable_unlink_safetensors
    observations = []

    def checked_unlink(file_path, *, allowed_root, expected_sha256=None):
        observations.append(load_file(target)["token_ids"].tolist())
        real_unlink(
            file_path,
            allowed_root=allowed_root,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(
        "scripts.data_generation_offline.durable_unlink_safetensors",
        checked_unlink,
    )
    _align_and_write_hidden_states(
        source,
        target,
        [1, 2, 3],
        source_root=source_root,
        target_root=target_root,
        allow_prefix_truncation=True,
        validate_outputs=False,
    )
    assert observations == [[1, 2, 3]]
    assert load_file(target)["token_ids"].tolist() == [1, 2, 3]
    assert not source.exists()


def test_atomic_save_interruption_preserves_target_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    _save_hidden_states(target, [1])

    def interrupted_save(tensors, temp_path):
        del tensors
        temp_path.write_bytes(b"partial")
        raise OSError("simulated interrupted save")

    monkeypatch.setattr(
        "speculators.data_generation.offline._save_safetensors",
        interrupted_save,
    )
    with pytest.raises(OSError, match="simulated interrupted save"):
        atomic_save_safetensors(
            _hidden_state_tensors([2]),
            target,
            allowed_root=root,
        )
    assert load_file(target)["token_ids"].tolist() == [1]
    assert not list(root.glob(".*.tmp"))


@pytest.mark.parametrize("replace_source", [False, True], ids=["success", "race"])
def test_atomic_move_cross_device_copy(tmp_path, monkeypatch, replace_source):
    source_root, target_root, source, target = _cache_paths(tmp_path)
    replacement = source_root / "replacement.safetensors"
    _save_hidden_states(source, [1, 2, 3])
    _save_hidden_states(replacement, [9, 9, 9])
    expected_sha256 = hidden_states_file_sha256(source, allowed_root=source_root)
    real_replace = os.replace
    real_copyfileobj = shutil.copyfileobj
    direct_move_attempts = 0

    def force_exdev(source_path, target_path):
        nonlocal direct_move_attempts
        if Path(source_path) == source and Path(target_path) == target:
            direct_move_attempts += 1
            raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(source_path, target_path)

    def copy_then_maybe_replace(source_file, target_file, length):
        real_copyfileobj(source_file, target_file, length)
        if replace_source:
            real_replace(replacement, source)

    monkeypatch.setattr("speculators.data_generation.offline.os.replace", force_exdev)
    monkeypatch.setattr(
        "speculators.data_generation.offline.shutil.copyfileobj",
        copy_then_maybe_replace,
    )

    def move():
        validation = (
            {
                "expected_source_sha256": expected_sha256,
                "expected_tokens": [1, 2, 3],
            }
            if replace_source
            else {}
        )
        return atomic_move_safetensors(
            source,
            target,
            source_root=source_root,
            target_root=target_root,
            **validation,
        )

    if replace_source:
        with pytest.raises(FileExistsError, match="changed after it was copied"):
            move()
        assert load_file(source)["token_ids"].tolist() == [9, 9, 9]
        assert not target.exists()
    else:
        assert move() == target.resolve()
        assert not source.exists()
        assert load_file(target)["token_ids"].tolist() == [1, 2, 3]
    assert direct_move_attempts == 1
    assert not list(target_root.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("operation", "identical"),
    [
        ("move", False),
        ("move", True),
        ("save", False),
        ("save", True),
    ],
)
def test_atomic_existing_target_is_rejected_or_idempotent(
    tmp_path,
    operation,
    identical,
):
    source_root, target_root, source, target = _cache_paths(tmp_path)
    tensors = _hidden_state_tensors([7, 8])
    if operation == "move":
        save_file(tensors, source)
        if identical:
            target.write_bytes(source.read_bytes())
        else:
            _save_hidden_states(target, [0])
    else:
        save_file(tensors if identical else _hidden_state_tensors([0]), target)
    original_target = target.read_bytes()

    def commit():
        if operation == "move":
            return atomic_move_safetensors(
                source,
                target,
                source_root=source_root,
                target_root=target_root,
            )
        return atomic_save_safetensors(
            tensors,
            target,
            allowed_root=target_root,
        )

    if identical:
        assert commit() == target.resolve()
        assert target.read_bytes() == original_target
        assert load_file(target)["token_ids"].tolist() == [7, 8]
        if operation == "move":
            assert not source.exists()
    else:
        with pytest.raises(FileExistsError, match="Refusing to replace"):
            commit()
        assert load_file(target)["token_ids"].tolist() == [0]
        if operation == "move":
            assert load_file(source)["token_ids"].tolist() == [7, 8]


@pytest.mark.parametrize("operation", ["move", "unlink"])
def test_atomic_operations_fsync_parent_once(tmp_path, monkeypatch, operation):
    root, _, source, target = _cache_paths(tmp_path, same_root=True)
    source.write_bytes(b"hidden states")
    fsynced = []
    monkeypatch.setattr(
        "speculators.data_generation.offline._fsync_directory",
        lambda path: fsynced.append(Path(path).resolve()),
    )
    if operation == "move":
        atomic_move_safetensors(
            source,
            target,
            source_root=root,
            target_root=root,
        )
        assert target.read_bytes() == b"hidden states"
    else:
        durable_unlink_safetensors(source, allowed_root=root)
    assert not source.exists()
    assert fsynced == [root.resolve()]


def test_durable_unlink_preserves_source_replacement(tmp_path):
    root, _, source, _ = _cache_paths(tmp_path, same_root=True)
    replacement = root / "replacement.safetensors"
    _save_hidden_states(source, [1, 2, 3])
    expected_sha256 = hidden_states_file_sha256(source, allowed_root=root)
    _save_hidden_states(replacement, [9, 9, 9])
    os.replace(replacement, source)

    with pytest.raises(FileExistsError, match="changed after it was validated"):
        durable_unlink_safetensors(
            source,
            allowed_root=root,
            expected_sha256=expected_sha256,
        )
    assert load_file(source)["token_ids"].tolist() == [9, 9, 9]
    assert not list(root.glob(".*.delete.safetensors"))


def test_in_place_atomic_save_rejects_target_changed_after_read(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    _save_hidden_states(target, [1, 2, 3])
    expected_sha256 = hidden_states_file_sha256(target, allowed_root=root)
    _save_hidden_states(target, [9, 9, 9])

    with pytest.raises(FileExistsError, match="changed after it was read"):
        atomic_save_safetensors(
            _hidden_state_tensors([1, 2]),
            target,
            allowed_root=root,
            allow_replace=True,
            expected_existing_sha256=expected_sha256,
        )
    assert load_file(target)["token_ids"].tolist() == [9, 9, 9]


def test_atomic_save_rejects_symlink_commit_lock(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    outside_lock = tmp_path / "outside.lock"
    outside_lock.write_text("do not follow")
    (root / ".hs_0.safetensors.commit.lock").symlink_to(outside_lock)

    with pytest.raises(ValueError, match="commit lock is a symlink"):
        atomic_save_safetensors(
            _hidden_state_tensors([1]),
            target,
            allowed_root=root,
        )
    assert not target.exists()
    assert outside_lock.read_text() == "do not follow"
    assert not list(root.glob(".*.tmp"))


def test_concurrent_atomic_saves_reject_loser_without_clobber(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    variants = [_hidden_state_tensors([value] * 128, value=value) for value in (11, 22)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                atomic_save_safetensors,
                tensors,
                target,
                allowed_root=root,
            )
            for tensors in variants
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError as error:
                outcomes.append(error)

    loaded = load_file(target)
    final_value = loaded["token_ids"][0].item()
    assert final_value in {11, 22}
    assert torch.all(loaded["token_ids"] == final_value)
    assert torch.all(loaded["hidden_states"] == float(final_value))
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    assert not list(root.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("num_samples", "max_samples", "existing", "world_size", "rank", "expected"),
    [
        (10, None, [], 1, 0, list(range(10))),
        (10, 5, [], 1, 0, list(range(5))),
        (5, 10, [], 1, 0, list(range(5))),
        (10, None, [2, 5, 7], 1, 0, [0, 1, 3, 4, 6, 8, 9]),
        (5, None, list(range(5)), 1, 0, []),
        (10, None, [], 2, 0, [0, 1, 2, 3, 4]),
        (10, None, [], 2, 1, [5, 6, 7, 8, 9]),
        (10, None, [], 3, 0, [0, 1, 2, 3]),
        (10, None, [], 3, 1, [4, 5, 6]),
        (10, None, [], 3, 2, [7, 8, 9]),
        (100, 10, [], 2, 0, [0, 1, 2, 3, 4]),
        (100, 10, [], 2, 1, [5, 6, 7, 8, 9]),
        (10, None, [1, 3], 2, 0, [0, 2, 4]),
        (10, None, [0, 1, 2, 3, 4], 2, 0, []),
        (5, None, list(range(10)), 1, 0, []),
    ],
)
def test_get_indices_to_process_cases(
    num_samples,
    max_samples,
    existing,
    world_size,
    rank,
    expected,
):
    assert get_indices_to_process(
        num_samples,
        max_samples,
        existing,
        world_size,
        rank,
    ) == expected


def test_get_indices_to_process_has_full_nonoverlapping_coverage():
    chunks = [
        get_indices_to_process(17, None, [], world_size=4, rank=rank)
        for rank in range(4)
    ]
    all_indices = [index for chunk in chunks for index in chunk]
    assert sorted(all_indices) == list(range(17))
    assert len(all_indices) == len(set(all_indices))


@pytest.mark.parametrize("existing_directory", [False, True])
def test_get_existing_hidden_state_indices_empty(tmp_path, existing_directory):
    output = tmp_path / "output"
    if existing_directory:
        output.mkdir()
    assert get_existing_hidden_state_indices(output) == []


def test_get_existing_hidden_state_indices_finds_sorted_files(tmp_path):
    for index in [9, 2, 5, 0]:
        _save_hidden_states(tmp_path / f"hs_{index}.safetensors", [index])
    assert get_existing_hidden_state_indices(tmp_path) == [0, 2, 5, 9]


def test_persistent_commit_lock_is_not_a_resume_candidate(tmp_path):
    target = tmp_path / "hs_0.safetensors"
    _save_hidden_states(target, [0])
    assert get_existing_hidden_state_indices(tmp_path) == [0]
    assert (tmp_path / ".hs_0.safetensors.commit.lock").is_file()
    assert get_existing_hidden_state_indices(tmp_path) == [0]


def test_get_existing_hidden_state_indices_ignores_unrelated_files(tmp_path):
    _save_hidden_states(tmp_path / "hs_0.safetensors", [0])
    ignored = ["hs_abc.safetensors", "hs_.safetensors", "other_file.txt", "hs_1.pt"]
    for name in ignored:
        (tmp_path / name).touch()
    assert get_existing_hidden_state_indices(tmp_path) == [0]
    assert all((tmp_path / name).exists() for name in ignored)


def test_resume_quarantines_noncanonical_numeric_filename(tmp_path):
    noncanonical = tmp_path / "hs_00.safetensors"
    _save_hidden_states(noncanonical, [0])
    assert get_existing_hidden_state_indices(tmp_path) == []
    assert not noncanonical.exists()
    assert len(list((tmp_path / "invalid").glob("*.json"))) == 1


def test_resume_quarantines_invalid_entries_and_wrong_tokens(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.safetensors"
    _save_hidden_states(outside, [5])
    (tmp_path / "hs_0.safetensors").touch()
    (tmp_path / "hs_1.safetensors").mkdir()
    save_file(
        {"hidden_states": torch.randn(1, 2, 3)},
        tmp_path / "hs_2.safetensors",
    )
    save_file(
        {
            "token_ids": torch.tensor([3, 3], dtype=torch.long),
            "hidden_states": torch.randn(1, 2, 3),
        },
        tmp_path / "hs_3.safetensors",
    )
    _save_hidden_states(tmp_path / "hs_4.safetensors", [999])
    (tmp_path / "hs_5.safetensors").symlink_to(outside)
    _save_hidden_states(tmp_path / "hs_6.safetensors", [6])

    assert get_existing_hidden_state_indices(
        tmp_path,
        expected_tokens_for_index=lambda index: [index],
    ) == [6]
    assert all(
        not os.path.lexists(tmp_path / f"hs_{index}.safetensors")
        for index in range(6)
    )
    assert outside.is_file()
    invalid_dir = tmp_path / "invalid"
    quarantined = [
        path
        for path in invalid_dir.iterdir()
        if ".invalid-" in path.name and not path.name.endswith(".json")
    ]
    records = list(invalid_dir.glob("*.json"))
    assert len(quarantined) == len(records) == 6
    assert any(path.is_symlink() for path in quarantined)
    for record_path in records:
        record = json.loads(record_path.read_text())
        assert record["timestamp"].endswith("+00:00")
        assert record["reason"]
        assert record["original_path"].startswith(str(tmp_path))


def test_resume_quarantines_cache_index_outside_dataset(tmp_path):
    target = tmp_path / "hs_7.safetensors"
    _save_hidden_states(target, [7])

    def expected_tokens(index):
        raise InvalidHiddenStateCacheError(
            f"cache index {index} is outside the current dataset"
        )

    assert get_existing_hidden_state_indices(
        tmp_path,
        expected_tokens_for_index=expected_tokens,
    ) == []
    assert not target.exists()
    assert len(list((tmp_path / "invalid").glob("*.json"))) == 1


def test_resume_quarantines_nonfinite_hidden_states(tmp_path):
    target = tmp_path / "hs_0.safetensors"
    tensors = _hidden_state_tensors([0])
    tensors["hidden_states"][0, 0, 0] = float("nan")
    save_file(tensors, target)

    assert get_existing_hidden_state_indices(tmp_path) == []
    assert not target.exists()
    records = list((tmp_path / "invalid").glob("*.json"))
    assert len(records) == 1
    assert "NaN or Inf" in json.loads(records[0].read_text())["reason"]


def test_resume_callback_infrastructure_error_aborts_without_quarantine(tmp_path):
    target = tmp_path / "hs_0.safetensors"
    _save_hidden_states(target, [0])

    def expected_tokens(index):
        raise RuntimeError(f"dataset backend unavailable for index {index}")

    with pytest.raises(RuntimeError, match="dataset backend unavailable"):
        get_existing_hidden_state_indices(
            tmp_path,
            expected_tokens_for_index=expected_tokens,
        )
    assert target.is_file()
    assert not (tmp_path / "invalid").exists()
