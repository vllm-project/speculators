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


def _save_hidden_states(path: Path, tokens: list[int]) -> None:
    save_file(
        {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "hidden_states": torch.randn(len(tokens), 2, 3),
        },
        path,
    )


def test_offline_cli_rejects_negative_max_retries_before_runtime(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "sys.argv",
        ["data_generation_offline.py", "--max-retries", "-1"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_offline_args()

    assert exc_info.value.code == 2
    assert "expected a non-negative integer" in capsys.readouterr().err


@pytest.mark.parametrize("invalid_concurrency", ["0", "-1"])
def test_offline_cli_rejects_nonpositive_concurrency_before_runtime(
    monkeypatch,
    capsys,
    invalid_concurrency,
):
    monkeypatch.setattr(
        "sys.argv",
        ["data_generation_offline.py", "--concurrency", invalid_concurrency],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_offline_args()

    assert exc_info.value.code == 2
    assert "positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("invalid_timeout", ["0", "-1", "nan", "inf", "-inf"])
def test_offline_cli_rejects_nonpositive_or_nonfinite_request_timeout(
    monkeypatch,
    capsys,
    invalid_timeout,
):
    monkeypatch.setattr(
        "sys.argv",
        ["data_generation_offline.py", "--request-timeout", invalid_timeout],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_offline_args()

    assert exc_info.value.code == 2
    assert "finite positive number" in capsys.readouterr().err


def test_align_hidden_states_to_tokens_truncates_prefix_match():
    """Multimodal prefix matches are trimmed to the preprocessed token length."""
    data = {
        "token_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        "hidden_states": torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3),
    }

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
    """Token drift should fail instead of aligning unrelated hidden states."""
    data = {
        "token_ids": torch.tensor([1, 9, 3], dtype=torch.long),
        "hidden_states": torch.zeros(3, 2, 3),
    }

    with pytest.raises(ValueError, match="Token ids don't match"):
        align_hidden_states_to_tokens(
            data,
            [1, 2, 3],
            allow_prefix_truncation=True,
        )


def test_align_and_write_hidden_states_moves_without_load_when_validation_disabled(
    tmp_path,
    monkeypatch,
):
    """Fast-path generation uses the structure gate without a full tensor load."""
    source_root = tmp_path / "shared-source"
    source_root.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    source_path = source_root / "source.safetensors"
    target_path = output_root / "target.safetensors"
    _save_hidden_states(source_path, [1, 2, 3])

    def raise_if_loaded(*args, **kwargs):
        del args, kwargs
        raise AssertionError("load_file should not be called")

    monkeypatch.setattr("scripts.data_generation_offline.load_file", raise_if_loaded)
    fsynced_directories = []
    monkeypatch.setattr(
        "speculators.data_generation.offline._fsync_directory",
        lambda path: fsynced_directories.append(Path(path).resolve()),
    )

    _align_and_write_hidden_states(
        source_path,
        target_path,
        [1, 2, 3],
        source_root=source_root,
        target_root=output_root,
        allow_prefix_truncation=False,
        validate_outputs=False,
    )

    assert not source_path.exists()
    assert load_file(target_path)["token_ids"].tolist() == [1, 2, 3]
    assert set(fsynced_directories) == {
        source_root.resolve(),
        output_root.resolve(),
    }


def test_align_and_write_fast_path_rejects_structurally_invalid_source(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    target = target_root / "hs_0.safetensors"
    save_file({"token_ids": torch.tensor([1, 2, 3])}, source)

    with pytest.raises(ValueError, match="missing required tensors"):
        _align_and_write_hidden_states(
            source,
            target,
            [999],
            source_root=source_root,
            target_root=target_root,
            allow_prefix_truncation=False,
            validate_outputs=False,
        )

    assert source.is_file()
    assert not target.exists()


def test_align_and_write_fast_path_rejects_wrong_sample_tokens(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    target = target_root / "hs_0.safetensors"
    _save_hidden_states(source, [9, 9, 9])

    with pytest.raises(InvalidHiddenStateCacheError, match="do not match"):
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


def test_align_and_write_fast_path_restores_same_token_source_replacement(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    replacement = source_root / "replacement.safetensors"
    target = target_root / "hs_0.safetensors"
    _save_hidden_states(source, [1, 2, 3])
    save_file(
        {
            "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "hidden_states": torch.full((3, 2, 3), 42.0),
        },
        replacement,
    )
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
    assert hidden_states_file_sha256(
        source,
        allowed_root=source_root,
    ) == replacement_sha256


def test_validate_hidden_states_path_accepts_regular_file_under_root(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    source_path = root / "request.safetensors"
    source_path.write_bytes(b"hidden states")

    assert validate_hidden_states_path(source_path, root) == source_path.resolve()


def test_validate_hidden_states_path_rejects_relative_path(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()

    with pytest.raises(ValueError, match="must be absolute"):
        validate_hidden_states_path("relative.safetensors", root)


def test_validate_hidden_states_path_rejects_wrong_suffix_or_non_file(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    wrong_suffix = root / "source.bin"
    wrong_suffix.write_bytes(b"hidden states")
    directory = root / "directory.safetensors"
    directory.mkdir()

    with pytest.raises(ValueError, match="must end in .safetensors"):
        validate_hidden_states_path(wrong_suffix, root)
    with pytest.raises(ValueError, match="not a regular file"):
        validate_hidden_states_path(directory, root)


def test_validate_hidden_states_path_pre_lock_allows_missing_leaf(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    pending_path = root / "pending.safetensors"

    assert validate_hidden_states_path(
        pending_path,
        root,
        require_exists=False,
    ) == pending_path.resolve()
    with pytest.raises(ValueError, match="does not exist"):
        validate_hidden_states_path(pending_path, root)


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda root, outside: outside / "outside.safetensors",
        lambda root, outside: (
            root / "nested" / ".." / ".." / outside.name / "outside.safetensors"
        ),
    ],
)
def test_validate_hidden_states_path_rejects_paths_outside_root(
    tmp_path,
    path_factory,
):
    root = tmp_path / "shared"
    root.mkdir()
    (root / "nested").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_path = outside / "outside.safetensors"
    outside_path.write_bytes(b"hidden states")

    candidate = path_factory(root, outside)
    with pytest.raises(ValueError, match="outside the allowed root"):
        validate_hidden_states_path(candidate, root)


@pytest.mark.parametrize("link_parent", [False, True])
def test_validate_hidden_states_path_rejects_symlink_components(
    tmp_path,
    link_parent,
):
    root = tmp_path / "shared"
    root.mkdir()
    real_dir = root / "real"
    real_dir.mkdir()
    real_file = real_dir / "source.safetensors"
    real_file.write_bytes(b"hidden states")

    if link_parent:
        linked_dir = root / "linked-dir"
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        candidate = linked_dir / real_file.name
    else:
        candidate = root / "linked-file.safetensors"
        candidate.symlink_to(real_file)

    with pytest.raises(ValueError, match="symlink component"):
        validate_hidden_states_path(candidate, root)


@pytest.mark.parametrize(
    ("token_ids", "hidden_states", "error"),
    [
        (
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.randn(1, 2, 3),
            "one-dimensional",
        ),
        (
            torch.tensor([], dtype=torch.long),
            torch.empty(0, 2, 3),
            "must not be empty",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.randn(2, 2, 3),
            "integer dtype",
        ),
        (
            torch.tensor([1, 2], dtype=torch.long),
            torch.randn(2, 3),
            "exactly three non-empty dimensions",
        ),
        (
            torch.tensor([1, 2], dtype=torch.long),
            torch.empty(2, 0, 3),
            "exactly three non-empty dimensions",
        ),
        (
            torch.tensor([1, 2], dtype=torch.long),
            torch.ones(2, 2, 3, dtype=torch.long),
            "floating dtype",
        ),
        (
            torch.tensor([1, 2], dtype=torch.long),
            torch.randn(1, 2, 3),
            "sequence length does not match",
        ),
    ],
    ids=[
        "token-rank",
        "empty-token-sequence",
        "token-dtype",
        "hidden-rank",
        "hidden-empty-dimension",
        "hidden-dtype",
        "sequence-length",
    ],
)
def test_validate_hidden_states_file_contents_rejects_invalid_structure(
    tmp_path,
    token_ids,
    hidden_states,
    error,
):
    target = tmp_path / "hs_0.safetensors"
    save_file(
        {"token_ids": token_ids, "hidden_states": hidden_states},
        target,
    )

    with pytest.raises(InvalidHiddenStateCacheError, match=error):
        validate_hidden_states_file_contents(target, tmp_path)


def test_generic_hidden_state_validators_accept_mtp_final_layer_only(tmp_path):
    """MTP intentionally extracts one verifier-final layer and no aux layers."""
    tensors = {
        "token_ids": torch.tensor([1, 2], dtype=torch.long),
        "hidden_states": torch.randn(2, 1, 3),
    }
    target = tmp_path / "hs_0.safetensors"
    save_file(tensors, target)

    assert validate_hidden_states_file_contents(target, tmp_path) == [1, 2]
    validate_hidden_states_tensors(tensors)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_validate_hidden_states_file_contents_streams_nonfinite_values(
    tmp_path,
    bad_value,
):
    target = tmp_path / "hs_0.safetensors"
    hidden_states = torch.randn(257, 2, 3)
    hidden_states[200, 1, 2] = bad_value
    save_file(
        {
            "token_ids": torch.arange(257, dtype=torch.long),
            "hidden_states": hidden_states,
        },
        target,
    )

    # The cheap source-file structure gate does not force a full tensor read.
    validate_hidden_states_file_contents(target, tmp_path, validate_values=False)
    with pytest.raises(InvalidHiddenStateCacheError, match="NaN or Inf"):
        validate_hidden_states_file_contents(target, tmp_path, validate_values=True)


def test_align_and_write_hidden_states_rejects_source_outside_root(tmp_path):
    source_root = tmp_path / "shared"
    source_root.mkdir()
    source_path = tmp_path / "outside.safetensors"
    source_path.write_bytes(b"do not move")
    target_path = source_root / "target.safetensors"

    with pytest.raises(ValueError, match="outside the allowed root"):
        _align_and_write_hidden_states(
            source_path,
            target_path,
            [1, 2, 3],
            source_root=source_root,
            target_root=source_root,
            allow_prefix_truncation=False,
            validate_outputs=False,
        )

    assert source_path.read_bytes() == b"do not move"
    assert not target_path.exists()


def test_align_and_write_rejects_other_managed_cache_source(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    other_target = root / "hs_1.safetensors"
    requested_target = root / "hs_0.safetensors"
    _save_hidden_states(other_target, [1, 2, 3])

    with pytest.raises(ValueError, match="another managed cache entry"):
        _align_and_write_hidden_states(
            other_target,
            requested_target,
            [1, 2, 3],
            source_root=root,
            target_root=root,
            allow_prefix_truncation=False,
            validate_outputs=False,
        )

    assert other_target.is_file()
    assert not requested_target.exists()


def test_align_and_write_rejects_managed_source_under_broader_source_root(tmp_path):
    broad_source_root = tmp_path / "shared"
    broad_source_root.mkdir()
    target_root = broad_source_root / "output"
    target_root.mkdir()
    other_target = target_root / "hs_1.safetensors"
    requested_target = target_root / "hs_0.safetensors"
    _save_hidden_states(other_target, [1, 2, 3])

    with pytest.raises(ValueError, match="another managed cache entry"):
        _align_and_write_hidden_states(
            other_target,
            requested_target,
            [1, 2, 3],
            source_root=broad_source_root,
            target_root=target_root,
            allow_prefix_truncation=False,
            validate_outputs=False,
        )

    assert other_target.is_file()
    assert not requested_target.exists()


def test_align_and_write_hidden_states_truncated_source_equals_target(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "shared"
    root.mkdir()
    file_path = root / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "hidden_states": torch.randn(4, 2, 3),
        },
        file_path,
    )
    fsynced_directories = []
    monkeypatch.setattr(
        "speculators.data_generation.offline._fsync_directory",
        lambda path: fsynced_directories.append(Path(path).resolve()),
    )

    _align_and_write_hidden_states(
        file_path,
        file_path,
        [1, 2, 3],
        source_root=root,
        target_root=root,
        allow_prefix_truncation=True,
        validate_outputs=False,
    )

    loaded = load_file(file_path)
    assert loaded["token_ids"].tolist() == [1, 2, 3]
    assert loaded["hidden_states"].shape == (3, 2, 3)
    assert fsynced_directories == [root.resolve()]
    assert not list(root.glob(".*.tmp"))


def test_truncated_write_commits_target_before_durable_source_unlink(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    target = target_root / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "hidden_states": torch.randn(4, 2, 3),
        },
        source,
    )
    real_unlink = durable_unlink_safetensors
    unlink_observations = []

    def checked_unlink(file_path, *, allowed_root, expected_sha256=None):
        committed = load_file(target)
        unlink_observations.append(committed["token_ids"].tolist())
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

    assert unlink_observations == [[1, 2, 3]]
    assert not source.exists()
    assert load_file(target)["token_ids"].tolist() == [1, 2, 3]


def test_atomic_save_interruption_preserves_old_target_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([1], dtype=torch.long),
            "hidden_states": torch.ones(1, 1, 1),
        },
        target,
    )

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
            {
                "token_ids": torch.tensor([2], dtype=torch.long),
                "hidden_states": torch.zeros(1, 1, 1),
            },
            target,
            allowed_root=root,
        )

    assert load_file(target)["token_ids"].tolist() == [1]
    assert not list(root.glob(".*.tmp"))


def test_atomic_move_uses_cross_device_copy_fallback(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    target = target_root / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([7, 8], dtype=torch.long),
            "hidden_states": torch.randn(2, 1, 2),
        },
        source,
    )
    real_replace = os.replace
    direct_move_attempts = 0

    def replace_with_exdev_once(source_path, target_path):
        nonlocal direct_move_attempts
        if Path(source_path) == source and Path(target_path) == target:
            direct_move_attempts += 1
            raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(
        "speculators.data_generation.offline.os.replace",
        replace_with_exdev_once,
    )

    committed = atomic_move_safetensors(
        source,
        target,
        source_root=source_root,
        target_root=target_root,
    )

    assert committed == target.resolve()
    assert direct_move_attempts == 1
    assert not source.exists()
    assert load_file(target)["token_ids"].tolist() == [7, 8]
    assert not list(target_root.glob(".*.tmp"))


def test_cross_device_move_preserves_source_replacement_after_copy(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    replacement = source_root / "replacement.safetensors"
    target = target_root / "hs_0.safetensors"
    _save_hidden_states(source, [1, 2, 3])
    _save_hidden_states(replacement, [9, 9, 9])
    expected_sha256 = hidden_states_file_sha256(source, allowed_root=source_root)

    real_replace = os.replace

    def force_initial_exdev(source_path, target_path):
        if Path(source_path) == source and Path(target_path) == target:
            raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(
        "speculators.data_generation.offline.os.replace",
        force_initial_exdev,
    )
    real_copyfileobj = shutil.copyfileobj

    def copy_then_replace_source(source_file, target_file, length):
        real_copyfileobj(source_file, target_file, length)
        real_replace(replacement, source)

    monkeypatch.setattr(
        "speculators.data_generation.offline.shutil.copyfileobj",
        copy_then_replace_source,
    )

    with pytest.raises(FileExistsError, match="changed after it was copied"):
        atomic_move_safetensors(
            source,
            target,
            source_root=source_root,
            target_root=target_root,
            expected_source_sha256=expected_sha256,
            expected_tokens=[1, 2, 3],
        )

    assert load_file(source)["token_ids"].tolist() == [9, 9, 9]
    assert not target.exists()
    assert not list(target_root.glob(".*.tmp"))


def test_atomic_move_rejects_different_existing_target_and_preserves_source(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    target = target_root / "hs_0.safetensors"
    _save_hidden_states(source, [7, 8])
    _save_hidden_states(target, [0])

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        atomic_move_safetensors(
            source,
            target,
            source_root=source_root,
            target_root=target_root,
        )

    assert load_file(source)["token_ids"].tolist() == [7, 8]
    assert load_file(target)["token_ids"].tolist() == [0]


def test_atomic_move_identical_existing_target_is_idempotent(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    source = source_root / "response.safetensors"
    target = target_root / "hs_0.safetensors"
    _save_hidden_states(source, [7, 8])
    target.write_bytes(source.read_bytes())

    committed = atomic_move_safetensors(
        source,
        target,
        source_root=source_root,
        target_root=target_root,
    )

    assert committed == target.resolve()
    assert not source.exists()
    assert load_file(target)["token_ids"].tolist() == [7, 8]


def test_atomic_move_same_directory_fsyncs_parent_once(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    source = root / "response.safetensors"
    target = root / "hs_0.safetensors"
    source.write_bytes(b"hidden states")
    fsynced_directories = []
    monkeypatch.setattr(
        "speculators.data_generation.offline._fsync_directory",
        lambda path: fsynced_directories.append(Path(path).resolve()),
    )

    atomic_move_safetensors(
        source,
        target,
        source_root=root,
        target_root=root,
    )

    assert fsynced_directories == [root.resolve()]
    assert target.read_bytes() == b"hidden states"


def test_durable_unlink_validates_and_fsyncs_parent(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    source = root / "response.safetensors"
    source.write_bytes(b"hidden states")
    fsynced_directories = []
    monkeypatch.setattr(
        "speculators.data_generation.offline._fsync_directory",
        lambda path: fsynced_directories.append(Path(path).resolve()),
    )

    durable_unlink_safetensors(source, allowed_root=root)

    assert not source.exists()
    assert fsynced_directories == [root.resolve()]


def test_durable_unlink_preserves_regular_source_replacement(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    source = root / "response.safetensors"
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


def test_atomic_save_rejects_different_existing_target(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    _save_hidden_states(target, [1])

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        atomic_save_safetensors(
            {
                "token_ids": torch.tensor([2], dtype=torch.long),
                "hidden_states": torch.zeros(1, 2, 3),
            },
            target,
            allowed_root=root,
        )

    assert load_file(target)["token_ids"].tolist() == [1]


def test_in_place_atomic_save_rejects_target_changed_after_read(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    _save_hidden_states(target, [1, 2, 3])
    expected_sha256 = hidden_states_file_sha256(target, allowed_root=root)

    # Simulate another writer committing a different target after the caller
    # loaded the old source but before its prefix rewrite acquired the lock.
    _save_hidden_states(target, [9, 9, 9])

    with pytest.raises(FileExistsError, match="changed after it was read"):
        atomic_save_safetensors(
            {
                "token_ids": torch.tensor([1, 2], dtype=torch.long),
                "hidden_states": torch.zeros(2, 2, 3),
            },
            target,
            allowed_root=root,
            allow_replace=True,
            expected_existing_sha256=expected_sha256,
        )

    assert load_file(target)["token_ids"].tolist() == [9, 9, 9]


def test_atomic_save_identical_existing_target_is_idempotent(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    tensors = {
        "token_ids": torch.tensor([1, 2], dtype=torch.long),
        "hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
    }
    save_file(tensors, target)
    original_bytes = target.read_bytes()

    committed = atomic_save_safetensors(tensors, target, allowed_root=root)

    assert committed == target.resolve()
    assert target.read_bytes() == original_bytes


def test_atomic_save_rejects_symlink_commit_lock(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    outside_lock = tmp_path / "outside.lock"
    outside_lock.write_text("do not follow")
    lock_path = root / ".hs_0.safetensors.commit.lock"
    lock_path.symlink_to(outside_lock)

    with pytest.raises(ValueError, match="commit lock is a symlink"):
        atomic_save_safetensors(
            {
                "token_ids": torch.tensor([1], dtype=torch.long),
                "hidden_states": torch.ones(1, 2, 3),
            },
            target,
            allowed_root=root,
        )

    assert not target.exists()
    assert outside_lock.read_text() == "do not follow"
    assert not list(root.glob(".*.tmp"))


def test_concurrent_different_atomic_saves_reject_loser_without_clobber(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    target = root / "hs_0.safetensors"
    variants = [
        {
            "token_ids": torch.full((128,), value, dtype=torch.long),
            "hidden_states": torch.full((128, 2, 3), float(value)),
        }
        for value in (11, 22)
    ]

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
            except FileExistsError as e:
                outcomes.append(e)

    loaded = load_file(target)
    final_value = loaded["token_ids"][0].item()
    assert final_value in {11, 22}
    assert torch.all(loaded["token_ids"] == final_value)
    assert torch.all(loaded["hidden_states"] == float(final_value))
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    assert not list(root.glob(".*.tmp"))


# ===== get_indices_to_process Tests =====


class TestGetIndicesToProcess:
    def test_single_node_no_max_samples(self):
        result = get_indices_to_process(10, None, [], world_size=1, rank=0)
        assert result == list(range(10))

    def test_single_node_with_max_samples(self):
        result = get_indices_to_process(10, 5, [], world_size=1, rank=0)
        assert result == [0, 1, 2, 3, 4]

    def test_single_node_max_samples_exceeds_num_samples(self):
        result = get_indices_to_process(5, 10, [], world_size=1, rank=0)
        assert result == list(range(5))

    def test_single_node_with_existing(self):
        result = get_indices_to_process(10, None, [2, 5, 7], world_size=1, rank=0)
        assert result == [0, 1, 3, 4, 6, 8, 9]

    def test_all_samples_already_processed(self):
        result = get_indices_to_process(5, None, list(range(5)), world_size=1, rank=0)
        assert result == []

    def test_multi_node_even_split(self):
        r0 = get_indices_to_process(10, None, [], world_size=2, rank=0)
        r1 = get_indices_to_process(10, None, [], world_size=2, rank=1)
        assert r0 == [0, 1, 2, 3, 4]
        assert r1 == [5, 6, 7, 8, 9]

    def test_multi_node_uneven_split(self):
        r0 = get_indices_to_process(10, None, [], world_size=3, rank=0)
        r1 = get_indices_to_process(10, None, [], world_size=3, rank=1)
        r2 = get_indices_to_process(10, None, [], world_size=3, rank=2)
        assert r0 == [0, 1, 2, 3]
        assert r1 == [4, 5, 6]
        assert r2 == [7, 8, 9]

    def test_multi_node_no_overlap_and_full_coverage(self):
        num_samples = 17
        world_size = 4
        all_indices = []
        for rank in range(world_size):
            chunk = get_indices_to_process(
                num_samples, None, [], world_size=world_size, rank=rank
            )
            all_indices.extend(chunk)
        assert sorted(all_indices) == list(range(num_samples))
        assert len(all_indices) == len(set(all_indices))

    def test_multi_node_with_max_samples(self):
        r0 = get_indices_to_process(100, 10, [], world_size=2, rank=0)
        r1 = get_indices_to_process(100, 10, [], world_size=2, rank=1)
        assert r0 == [0, 1, 2, 3, 4]
        assert r1 == [5, 6, 7, 8, 9]

    def test_multi_node_with_existing(self):
        result = get_indices_to_process(10, None, [1, 3], world_size=2, rank=0)
        assert result == [0, 2, 4]

    def test_multi_node_rank_fully_processed(self):
        result = get_indices_to_process(10, None, [0, 1, 2, 3, 4], world_size=2, rank=0)
        assert result == []

    def test_existing_exceeds_num_samples(self):
        result = get_indices_to_process(5, None, list(range(10)), world_size=1, rank=0)
        assert result == []


# ===== get_existing_hidden_state_indices Tests =====


class TestGetExistingHiddenStateIndices:
    def test_nonexistent_directory(self, tmp_path):
        result = get_existing_hidden_state_indices(tmp_path / "nonexistent")
        assert result == []

    def test_empty_directory(self, tmp_path):
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == []

    def test_finds_safetensor_files(self, tmp_path):
        _save_hidden_states(tmp_path / "hs_0.safetensors", [0])
        _save_hidden_states(tmp_path / "hs_3.safetensors", [3])
        _save_hidden_states(tmp_path / "hs_7.safetensors", [7])
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0, 3, 7]

    def test_persistent_commit_lock_is_not_a_resume_candidate(self, tmp_path):
        target = tmp_path / "hs_0.safetensors"
        _save_hidden_states(target, [0])

        assert get_existing_hidden_state_indices(tmp_path) == [0]
        assert (tmp_path / ".hs_0.safetensors.commit.lock").is_file()
        assert get_existing_hidden_state_indices(tmp_path) == [0]

    def test_ignores_non_numeric_suffixes(self, tmp_path):
        _save_hidden_states(tmp_path / "hs_0.safetensors", [0])
        (tmp_path / "hs_abc.safetensors").touch()
        (tmp_path / "hs_.safetensors").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0]
        assert (tmp_path / "hs_abc.safetensors").exists()
        assert (tmp_path / "hs_.safetensors").exists()

    def test_quarantines_noncanonical_numeric_filename(self, tmp_path):
        noncanonical = tmp_path / "hs_00.safetensors"
        _save_hidden_states(noncanonical, [0])

        assert get_existing_hidden_state_indices(tmp_path) == []
        assert not noncanonical.exists()
        assert len(list((tmp_path / "invalid").glob("*.json"))) == 1

    def test_ignores_unrelated_files(self, tmp_path):
        _save_hidden_states(tmp_path / "hs_0.safetensors", [0])
        (tmp_path / "other_file.txt").touch()
        (tmp_path / "hs_1.pt").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0]

    def test_results_are_sorted(self, tmp_path):
        for i in [9, 2, 5, 0]:
            _save_hidden_states(tmp_path / f"hs_{i}.safetensors", [i])
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0, 2, 5, 9]

    def test_quarantines_invalid_entries_and_wrong_dataset_tokens(self, tmp_path):
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

        result = get_existing_hidden_state_indices(
            tmp_path,
            expected_tokens_for_index=lambda index: [index],
        )

        assert result == [6]
        for index in range(6):
            assert not os.path.lexists(tmp_path / f"hs_{index}.safetensors")
        assert outside.is_file()

        invalid_dir = tmp_path / "invalid"
        quarantined = [
            path
            for path in invalid_dir.iterdir()
            if ".invalid-" in path.name and not path.name.endswith(".json")
        ]
        records = list(invalid_dir.glob("*.json"))
        assert len(quarantined) == 6
        assert len(records) == 6
        assert any(path.is_symlink() for path in quarantined)
        for record_path in records:
            record = json.loads(record_path.read_text())
            assert record["timestamp"].endswith("+00:00")
            assert record["reason"]
            assert record["original_path"].startswith(str(tmp_path))

    def test_quarantines_cache_index_outside_current_dataset(self, tmp_path):
        _save_hidden_states(tmp_path / "hs_7.safetensors", [7])

        def expected_tokens(index):
            raise InvalidHiddenStateCacheError(
                f"cache index {index} is outside the current dataset"
            )

        result = get_existing_hidden_state_indices(
            tmp_path,
            expected_tokens_for_index=expected_tokens,
        )

        assert result == []
        assert not (tmp_path / "hs_7.safetensors").exists()
        assert len(list((tmp_path / "invalid").glob("*.json"))) == 1

    def test_resume_quarantines_nonfinite_hidden_states(self, tmp_path):
        target = tmp_path / "hs_0.safetensors"
        hidden_states = torch.randn(1, 2, 3)
        hidden_states[0, 0, 0] = float("nan")
        save_file(
            {
                "token_ids": torch.tensor([0], dtype=torch.long),
                "hidden_states": hidden_states,
            },
            target,
        )

        assert get_existing_hidden_state_indices(tmp_path) == []
        assert not target.exists()
        records = list((tmp_path / "invalid").glob("*.json"))
        assert len(records) == 1
        assert "NaN or Inf" in json.loads(records[0].read_text())["reason"]

    def test_callback_infrastructure_error_aborts_without_quarantine(self, tmp_path):
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
