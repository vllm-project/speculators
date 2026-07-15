import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts import pipeline_provenance as provenance


@pytest.mark.parametrize(
    "revision", ["main", "d5776d4", "g" * 40, "a" * 39, "a" * 41]
)
def test_rejects_nonimmutable_dataset_revisions(revision: str):
    with pytest.raises(provenance.ProvenanceError, match="full 40-character"):
        provenance.validate_full_commit_sha(revision)


def test_normalizes_full_dataset_revision():
    assert provenance.validate_full_commit_sha("A" * 40) == "a" * 40


@pytest.mark.parametrize("field_name", ["SOURCE_REVISION", "MODEL_REVISION"])
def test_rejects_unresolved_source_and_model_revisions(field_name: str):
    with pytest.raises(provenance.ProvenanceError, match=field_name):
        provenance.validate_full_commit_sha("unresolved", field_name)


def test_validates_exact_local_model_snapshot(tmp_path: Path):
    revision = "b" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    assert (
        provenance.validate_local_model_snapshot(snapshot, revision)
        == snapshot.resolve()
    )


@pytest.mark.parametrize("failure", ["wrong-revision", "remote-identifier"])
def test_rejects_nonexact_local_model_snapshot(tmp_path: Path, failure: str):
    revision = "b" * 40
    if failure == "wrong-revision":
        snapshot = tmp_path / ("a" * 40)
        snapshot.mkdir()
        error = "exact local HF snapshot"
    else:
        snapshot = Path("Qwen/Qwen3-VL-4B-Instruct")
        error = "local snapshot directory"
    with pytest.raises(provenance.ProvenanceError, match=error):
        provenance.validate_local_model_snapshot(snapshot, revision)


def _qwen3_vl_4b_config() -> dict:
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


@pytest.mark.parametrize("valid", [True, False])
def test_validates_exact_qwen3_vl_4b_architecture(tmp_path: Path, valid: bool):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    config = _qwen3_vl_4b_config()
    if not valid:
        config["text_config"]["hidden_size"] = 2048
    (snapshot / "config.json").write_text(json.dumps(config))
    if valid:
        assert provenance.validate_qwen3_vl_4b_snapshot(snapshot) == snapshot.resolve()
    else:
        with pytest.raises(provenance.ProvenanceError, match="Qwen3-VL-4B-Instruct"):
            provenance.validate_qwen3_vl_4b_snapshot(snapshot)


def _create_git_checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "tracked.txt").write_text("candidate\n")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(checkout), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "candidate",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, revision


def test_source_revision_is_bound_to_clean_executing_checkout(tmp_path: Path):
    checkout, revision = _create_git_checkout(tmp_path)
    (checkout / "runtime-output").mkdir()
    (checkout / "runtime-output" / "result.bin").write_bytes(b"artifact")
    assert provenance.validate_source_checkout(checkout, revision) == revision


def test_source_revision_rejects_wrong_or_dirty_checkout(tmp_path: Path):
    checkout, revision = _create_git_checkout(tmp_path)
    with pytest.raises(provenance.ProvenanceError, match="does not match"):
        provenance.validate_source_checkout(checkout, "f" * 40)
    (checkout / "tracked.txt").write_text("modified\n")
    with pytest.raises(provenance.ProvenanceError, match="not clean"):
        provenance.validate_source_checkout(checkout, revision)
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.txt"], check=True)
    with pytest.raises(provenance.ProvenanceError, match="not clean"):
        provenance.validate_source_checkout(checkout, revision)


def _pipeline_fields() -> dict[str, str]:
    return {
        "attempt_label": "x01",
        "dataset_repo": "owner/dataset",
        "dataset_revision": "a" * 40,
        "epochs": "5",
        "max_samples": "5000",
        "model": "owner/model",
        "model_revision": "b" * 40,
        "num_train_gpus": "1",
        "seq_length": "4096",
        "source_revision": "c" * 40,
        "vllm_extra_args": "--enforce-eager",
        "vllm_max_model_len": "5120",
    }


def _completed_artifact(tmp_path: Path, *, with_guard: bool = False):
    artifact = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    guard = tmp_path / "guard"
    guards = [guard] if with_guard else []
    assert provenance.claim_artifact(manifest, artifact, fields, guards) == "run"
    artifact.mkdir()
    result = artifact / "result.bin"
    result.write_bytes(b"complete")
    if with_guard:
        guard.mkdir()
        (guard / "hs_0.safetensors").write_bytes(b"hidden")
    provenance.complete_artifact(manifest, artifact, fields, guards)
    return artifact, manifest, fields, guard, result


def test_fingerprint_is_order_independent_and_covers_required_fields():
    fields = _pipeline_fields()
    expected = provenance.fingerprint_fields(fields)
    assert provenance.fingerprint_fields(dict(reversed(fields.items()))) == expected
    for key, value in fields.items():
        changed = fields | {key: f"{value}-changed"}
        assert provenance.fingerprint_fields(changed) != expected


def test_manifest_claim_complete_and_exact_reuse(tmp_path: Path):
    artifact = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact, fields) == "run"
    assert json.loads(manifest.read_text())["state"] == "in_progress"
    artifact.mkdir()
    (artifact / "result.bin").write_bytes(b"complete")
    provenance.complete_artifact(manifest, artifact, fields)

    completed = json.loads(manifest.read_text())
    aggregate = completed["content_aggregates"][str(artifact.resolve())]
    digest = hashlib.sha256(b"complete").hexdigest()
    record = f"result.bin\0{len(b'complete')}\0{digest}\0".encode()
    assert completed["state"] == "complete"
    assert (aggregate["file_count"], aggregate["total_size"]) == (1, len(b"complete"))
    assert aggregate["sha256"] == hashlib.sha256(record).hexdigest()
    assert provenance.claim_artifact(manifest, artifact, fields) == "reuse"


@pytest.mark.parametrize("mutation", ["replace", "append", "remove", "add"])
def test_completed_manifest_rejects_content_mutation(tmp_path: Path, mutation: str):
    artifact, manifest, fields, _, result = _completed_artifact(tmp_path)
    if mutation == "replace":
        result.write_bytes(b"tampered")
    elif mutation == "append":
        result.write_bytes(b"complete-extra")
    elif mutation == "remove":
        result.unlink()
        (artifact / "replacement.bin").write_bytes(b"complete")
    else:
        (artifact / "extra.bin").write_bytes(b"extra")
    with pytest.raises(provenance.ProvenanceError, match="content"):
        provenance.claim_artifact(manifest, artifact, fields)


def test_completed_manifest_hashes_and_revalidates_guard_directory(tmp_path: Path):
    artifact, manifest, fields, guard, _ = _completed_artifact(
        tmp_path, with_guard=True
    )
    completed = json.loads(manifest.read_text())
    assert set(completed["content_aggregates"]) == {
        str(artifact.resolve()), str(guard.resolve())
    }
    (guard / "hs_0.safetensors").write_bytes(b"changed")
    with pytest.raises(provenance.ProvenanceError, match="content"):
        provenance.claim_artifact(manifest, artifact, fields, [guard])


@pytest.mark.parametrize("unsafe_name", ["partial.tmp", "download.incomplete"])
def test_complete_rejects_temporary_artifact_files(tmp_path: Path, unsafe_name: str):
    artifact = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact, fields) == "run"
    artifact.mkdir()
    (artifact / unsafe_name).write_bytes(b"partial")
    with pytest.raises(provenance.ProvenanceError, match="temporary"):
        provenance.complete_artifact(manifest, artifact, fields)


def test_complete_rejects_recursive_artifact_symlink(tmp_path: Path):
    artifact = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact, fields) == "run"
    artifact.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (artifact / "linked.bin").symlink_to(outside)
    with pytest.raises(provenance.ProvenanceError, match="symbolic link"):
        provenance.complete_artifact(manifest, artifact, fields)


@pytest.mark.parametrize("symlinked_path", ["artifact", "manifest"])
def test_claim_rejects_symlinked_ancestor(tmp_path: Path, symlinked_path: str):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    artifact = (
        linked_parent / "artifact"
        if symlinked_path == "artifact"
        else tmp_path / "artifact"
    )
    manifest = (
        linked_parent / "artifact.provenance.json"
        if symlinked_path == "manifest"
        else tmp_path / "artifact.provenance.json"
    )
    with pytest.raises(provenance.ProvenanceError, match="symbolic-link component"):
        provenance.claim_artifact(manifest, artifact, _pipeline_fields())


def test_manifest_must_be_outside_hashed_artifact(tmp_path: Path):
    artifact = tmp_path / "artifact"
    with pytest.raises(provenance.ProvenanceError, match="self-reference"):
        provenance.claim_artifact(
            artifact / "provenance.json", artifact, _pipeline_fields()
        )


@pytest.mark.parametrize(
    "field",
    [
        "dataset_revision", "epochs", "max_samples", "model", "model_revision",
        "num_train_gpus", "seq_length", "source_revision", "vllm_extra_args",
        "vllm_max_model_len",
    ],
)
def test_manifest_mismatch_fails_closed(tmp_path: Path, field: str):
    artifact, manifest, fields, _, _ = _completed_artifact(tmp_path)
    with pytest.raises(provenance.ProvenanceError, match="mismatch"):
        provenance.claim_artifact(manifest, artifact, fields | {field: "different"})


def test_existing_artifact_without_manifest_fails_closed(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "stale.arrow").write_bytes(b"stale")
    with pytest.raises(provenance.ProvenanceError, match="without a provenance"):
        provenance.claim_artifact(
            tmp_path / "missing.provenance.json", artifact, _pipeline_fields()
        )


def test_unfinished_claim_cannot_be_reused(tmp_path: Path):
    artifact = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact, fields) == "run"
    with pytest.raises(provenance.ProvenanceError, match="unfinished"):
        provenance.claim_artifact(manifest, artifact, fields)


@pytest.mark.parametrize("artifact_state", ["missing", "empty", "symlink"])
def test_completed_manifest_requires_intact_artifact(
    tmp_path: Path, artifact_state: str
):
    artifact, manifest, fields, _, result = _completed_artifact(tmp_path)
    result.unlink()
    if artifact_state != "empty":
        artifact.rmdir()
    if artifact_state == "symlink":
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        (replacement / "result.bin").write_bytes(b"replacement")
        artifact.symlink_to(replacement, target_is_directory=True)
    for operation in (
        lambda: provenance.claim_artifact(manifest, artifact, fields),
        lambda: provenance.complete_artifact(manifest, artifact, fields),
    ):
        with pytest.raises(provenance.ProvenanceError, match="artifact"):
            operation()


def test_completed_manifest_requires_intact_guard_directories(tmp_path: Path):
    artifact, manifest, fields, guard, _ = _completed_artifact(
        tmp_path, with_guard=True
    )
    (guard / "hs_0.safetensors").unlink()
    with pytest.raises(provenance.ProvenanceError, match="empty artifact"):
        provenance.claim_artifact(manifest, artifact, fields, [guard])


def _valid_messages(image_path: Path) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_path.resolve().as_uri()},
                },
                {"type": "text", "text": "Describe the image."},
            ],
        },
        {"role": "assistant", "content": "A test image."},
    ]


class _FakeDataset:
    def __init__(self, rows: list[dict], column_names: list[str] | None = None):
        self.rows = rows
        self.column_names = column_names or list(rows[0])

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@pytest.fixture
def prepared_case(tmp_path: Path):
    prepared = tmp_path / "prepared"
    dataset = tmp_path / "dataset"
    prepared.mkdir()
    dataset.mkdir()
    image = dataset / "image.png"
    image.write_bytes(b"image")
    row = {
        "input_ids": [1, 2],
        "loss_mask": [0, 1],
        "messages": _valid_messages(image),
        "seq_len": 2,
    }
    return prepared, dataset, image, row


def _validate_prepared(
    monkeypatch, prepared_case, rows, *, max_samples=1, seq_length=2
):
    prepared, dataset, _, _ = prepared_case
    monkeypatch.setattr("datasets.load_from_disk", lambda _path: _FakeDataset(rows))
    return provenance.validate_prepared_dataset(
        prepared, dataset_dir=dataset, max_samples=max_samples, seq_length=seq_length
    )


def test_validates_prepared_row_count_columns_images_and_sequence_length(
    prepared_case, monkeypatch
):
    _, _, image, row = prepared_case
    second = {
        "input_ids": [3],
        "loss_mask": [True],
        "messages": _valid_messages(image),
        "seq_len": 1,
    }
    assert _validate_prepared(
        monkeypatch, prepared_case, [row, second], max_samples=2
    ) == 2


def test_rejects_prepared_sequence_over_approved_length(prepared_case, monkeypatch):
    *_, row = prepared_case
    row.update(input_ids=[1, 2, 3], loss_mask=[0, 1, 1], seq_len=3)
    with pytest.raises(provenance.ProvenanceError, match="sequence length"):
        _validate_prepared(monkeypatch, prepared_case, [row], seq_length=2)


def test_rejects_prepared_row_count_below_approved_exact_count(
    prepared_case, monkeypatch
):
    *_, row = prepared_case
    with pytest.raises(provenance.ProvenanceError, match="approved exact count"):
        _validate_prepared(monkeypatch, prepared_case, [row], max_samples=2)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("seq-len", "seq_len"),
        ("no-trainable", "no trainable token"),
        ("nonbinary-mask", "binary values"),
        ("empty-messages", "messages must be a non-empty list"),
        ("missing-seq-len", "seq_len"),
    ],
)
def test_rejects_invalid_prepared_row_contract(
    prepared_case, monkeypatch, mutation: str, error: str
):
    *_, row = prepared_case
    if mutation == "seq-len":
        row["seq_len"] = 1
    elif mutation == "no-trainable":
        row["loss_mask"] = [0, 0]
    elif mutation == "nonbinary-mask":
        row["loss_mask"] = [0, 2]
    elif mutation == "empty-messages":
        row["messages"] = []
    else:
        row.pop("seq_len")
    with pytest.raises(provenance.ProvenanceError, match=error):
        _validate_prepared(monkeypatch, prepared_case, [row])


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("outside", "escapes dataset_dir"),
        ("symlink", "symbolic link"),
        ("multiple", "exactly one user image"),
    ],
)
def test_rejects_invalid_prepared_image(
    prepared_case, monkeypatch, mutation: str, error: str
):
    _, dataset, image, row = prepared_case
    messages = row["messages"]
    if mutation == "outside":
        target = dataset.parent / "outside.png"
        target.write_bytes(b"outside")
        messages[0]["content"][0]["image_url"]["url"] = target.resolve().as_uri()
    elif mutation == "symlink":
        target = dataset / "linked.png"
        target.symlink_to(image)
        messages[0]["content"][0]["image_url"]["url"] = target.absolute().as_uri()
    else:
        messages[0]["content"].append(
            {"type": "image_url", "image_url": {"url": image.resolve().as_uri()}}
        )
    with pytest.raises(provenance.ProvenanceError, match=error):
        _validate_prepared(monkeypatch, prepared_case, [row])


def _write_runtime_fixture(
    tmp_path: Path, *, max_samples: int = 2, epochs: int = 2
) -> tuple[Path, Path, Path, list[dict]]:
    prepared = tmp_path / "prepared"
    hidden = tmp_path / "hidden-states"
    checkpoints = tmp_path / "checkpoints"
    for path in (prepared, hidden, checkpoints):
        path.mkdir()
    rows = []
    for index in range(max_samples):
        tokens = [index + 1, index + 2]
        rows.append({"input_ids": tokens})
        save_file(
            {
                "token_ids": torch.tensor(tokens, dtype=torch.long),
                "hidden_states": torch.randn(len(tokens), 2, 3),
            },
            hidden / f"hs_{index}.safetensors",
        )
        (hidden / f".hs_{index}.safetensors.commit.lock").touch()
    for epoch in range(epochs):
        epoch_dir = checkpoints / str(epoch)
        epoch_dir.mkdir()
        (epoch_dir / "config.json").write_text(json.dumps({"model_type": "test"}))
        save_file({"weight": torch.ones(2, 2)}, epoch_dir / "model.safetensors")
        torch.save(
            {"state": {}, "param_groups": [{"params": []}]},
            epoch_dir / "optimizer_state_dict.pt",
        )
        (epoch_dir / "training_state.json").write_text(
            json.dumps({"epoch": epoch, "local_step": 0, "global_step": epoch + 1})
        )
    (checkpoints / "train_command.txt").write_text("torchrun train.py\n")
    return checkpoints, hidden, prepared, rows


def _validate_runtime(monkeypatch, runtime_case):
    checkpoints, hidden, prepared, rows = runtime_case
    monkeypatch.setattr(
        "datasets.load_from_disk", lambda _path: _FakeDataset(rows)
    )
    return provenance.validate_runtime_artifacts(
        checkpoints, hidden, prepared, max_samples=2, epochs=2
    )


@pytest.mark.parametrize("extra_source_lock", [False, True])
def test_validates_exact_runtime_hidden_states_and_completed_checkpoints(
    tmp_path: Path, monkeypatch, extra_source_lock: bool
):
    runtime_case = _write_runtime_fixture(tmp_path)
    if extra_source_lock:
        (runtime_case[1] / ".vllm-response-abc.safetensors.commit.lock").touch()
    assert _validate_runtime(monkeypatch, runtime_case) == 2


def test_runtime_rejects_active_noncanonical_source_commit_lock(tmp_path, monkeypatch):
    runtime_case = _write_runtime_fixture(tmp_path)
    lock = runtime_case[1] / ".vllm-response-abc.safetensors.commit.lock"
    lock.touch()
    descriptor = os.open(lock, os.O_RDONLY)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(provenance.ProvenanceError, match="still active"):
            _validate_runtime(monkeypatch, runtime_case)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing-hidden", "cache set is not exact"),
        ("token-mismatch", "hidden-state cache"),
        ("nan-hidden", "hidden-state cache"),
        ("inf-model", "NaN or Inf"),
        ("bad-optimizer", "not loadable"),
        ("missing-epoch", "epoch set is incomplete"),
        ("mid-epoch", "mid-epoch"),
    ],
)
def test_runtime_rejects_corrupt_hidden_state_or_checkpoint(
    tmp_path: Path, monkeypatch, mutation: str, error: str
):
    runtime_case = _write_runtime_fixture(tmp_path)
    checkpoints, hidden, _, rows = runtime_case
    hidden_path = hidden / "hs_1.safetensors"
    if mutation == "missing-hidden":
        hidden_path.unlink()
    elif mutation in {"token-mismatch", "nan-hidden"}:
        hidden_path.unlink()
        states = torch.randn(2, 2, 3)
        if mutation == "nan-hidden":
            states[1, 1, 1] = torch.nan
        tokens = [99, 100] if mutation == "token-mismatch" else rows[1]["input_ids"]
        save_file(
            {"token_ids": torch.tensor(tokens), "hidden_states": states}, hidden_path
        )
    elif mutation == "inf-model":
        model = checkpoints / "0" / "model.safetensors"
        model.unlink()
        save_file({"weight": torch.tensor([[1.0, torch.inf]])}, model)
    elif mutation == "bad-optimizer":
        (checkpoints / "0" / "optimizer_state_dict.pt").write_bytes(b"not a checkpoint")
    elif mutation == "missing-epoch":
        for path in (checkpoints / "1").iterdir():
            path.unlink()
        (checkpoints / "1").rmdir()
    else:
        (checkpoints / "1" / "training_state.json").write_text(
            json.dumps({"epoch": 1, "local_step": 3, "global_step": 4})
        )
    with pytest.raises(provenance.ProvenanceError, match=error):
        _validate_runtime(monkeypatch, runtime_case)


def _write_shard(data_dir: Path, index: int, total: int, *, width: int = 5) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    shard = data_dir / f"train-{index:0{width}d}-of-{total:0{width}d}.parquet"
    shard.write_bytes(b"PAR1")


def test_validates_complete_train_parquet_shards(tmp_path: Path):
    data = tmp_path / "dataset" / "data"
    _write_shard(data, 1, 2)
    _write_shard(data, 0, 2)
    shards = provenance.validate_train_parquet_shards(tmp_path / "dataset")
    assert [path.name for path in shards] == [
        "train-00000-of-00002.parquet", "train-00001-of-00002.parquet"
    ]


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("mixed-totals", "mixed totals"),
        ("missing-index", "incomplete or duplicated"),
        ("duplicate-index", "incomplete or duplicated"),
        ("stale-file", "stale shard mix"),
    ],
)
def test_rejects_invalid_train_parquet_shards(tmp_path: Path, failure: str, error: str):
    data = tmp_path / "dataset" / "data"
    if failure == "mixed-totals":
        _write_shard(data, 0, 2)
        _write_shard(data, 1, 3)
    elif failure == "missing-index":
        _write_shard(data, 0, 2)
    elif failure == "duplicate-index":
        _write_shard(data, 0, 2, width=1)
        _write_shard(data, 0, 2)
    else:
        _write_shard(data, 0, 1)
        (data / "train-old.parquet").write_bytes(b"stale")
    with pytest.raises(provenance.ProvenanceError, match=error):
        provenance.validate_train_parquet_shards(tmp_path / "dataset")
