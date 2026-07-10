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
    "revision",
    ["main", "d5776d4", "g" * 40, "a" * 39, "a" * 41],
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


def test_rejects_model_snapshot_whose_basename_does_not_match_revision(
    tmp_path: Path,
):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()

    with pytest.raises(provenance.ProvenanceError, match="exact local HF snapshot"):
        provenance.validate_local_model_snapshot(snapshot, "b" * 40)


def test_rejects_remote_model_identifier_as_snapshot():
    with pytest.raises(provenance.ProvenanceError, match="local snapshot directory"):
        provenance.validate_local_model_snapshot(
            Path("Qwen/Qwen3-VL-4B-Instruct"),
            "b" * 40,
        )


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


def test_validates_qwen3_vl_4b_architecture_metadata(tmp_path: Path):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(_qwen3_vl_4b_config()))

    assert provenance.validate_qwen3_vl_4b_snapshot(snapshot) == snapshot.resolve()


def test_rejects_non_4b_qwen3_vl_snapshot_metadata(tmp_path: Path):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    config = _qwen3_vl_4b_config()
    config["text_config"]["hidden_size"] = 2048
    (snapshot / "config.json").write_text(json.dumps(config))

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
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "candidate",
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
    # Default pipeline artifacts are intentionally untracked and may live below
    # the checkout; they do not change the source tree represented by HEAD.
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

    subprocess.run(
        ["git", "-C", str(checkout), "add", "tracked.txt"],
        check=True,
    )
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


def _install_fake_prepared_dataset(
    monkeypatch: pytest.MonkeyPatch,
    row: dict,
    *,
    column_names: list[str] | None = None,
) -> None:
    class FakeDataset:
        def __init__(self):
            self.column_names = column_names or list(row)

        def __len__(self):
            return 1

        def __iter__(self):
            yield row

    monkeypatch.setattr("datasets.load_from_disk", lambda _path: FakeDataset())


def _install_indexable_prepared_dataset(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict],
) -> None:
    class FakeDataset:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return rows[index]

    monkeypatch.setattr("datasets.load_from_disk", lambda _path: FakeDataset())


def _write_runtime_fixture(
    tmp_path: Path,
    *,
    max_samples: int = 2,
    epochs: int = 2,
) -> tuple[Path, Path, Path, list[dict]]:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    hidden_states_dir = tmp_path / "hidden-states"
    hidden_states_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    rows = []
    for index in range(max_samples):
        tokens = [index + 1, index + 2]
        rows.append({"input_ids": tokens})
        save_file(
            {
                "token_ids": torch.tensor(tokens, dtype=torch.long),
                "hidden_states": torch.randn(len(tokens), 2, 3),
            },
            hidden_states_dir / f"hs_{index}.safetensors",
        )
        (hidden_states_dir / f".hs_{index}.safetensors.commit.lock").touch()

    for epoch in range(epochs):
        epoch_dir = checkpoint_dir / str(epoch)
        epoch_dir.mkdir()
        (epoch_dir / "config.json").write_text(json.dumps({"model_type": "test"}))
        save_file(
            {"weight": torch.ones(2, 2)},
            epoch_dir / "model.safetensors",
        )
        torch.save(
            {"state": {}, "param_groups": [{"params": []}]},
            epoch_dir / "optimizer_state_dict.pt",
        )
        (epoch_dir / "training_state.json").write_text(
            json.dumps(
                {"epoch": epoch, "local_step": 0, "global_step": epoch + 1}
            )
        )
    (checkpoint_dir / "train_command.txt").write_text("torchrun train.py\n")
    return checkpoint_dir, hidden_states_dir, prepared_dir, rows


def test_fingerprint_is_order_independent_and_covers_required_fields():
    fields = _pipeline_fields()
    expected = provenance.fingerprint_fields(fields)

    assert provenance.fingerprint_fields(dict(reversed(fields.items()))) == expected
    for key, value in fields.items():
        changed = dict(fields)
        changed[key] = f"{value}-changed"
        assert provenance.fingerprint_fields(changed) != expected


def test_manifest_claim_complete_and_exact_reuse(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()

    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "run"
    assert json.loads(manifest.read_text())["state"] == "in_progress"
    artifact_dir.mkdir()
    (artifact_dir / "result.bin").write_bytes(b"complete")

    provenance.complete_artifact(manifest, artifact_dir, fields)

    completed = json.loads(manifest.read_text())
    assert completed["state"] == "complete"
    aggregate = completed["content_aggregates"][str(artifact_dir.resolve())]
    assert aggregate["file_count"] == 1
    assert aggregate["total_size"] == len(b"complete")
    file_digest = hashlib.sha256(b"complete").hexdigest()
    expected_record = f"result.bin\0{len(b'complete')}\0{file_digest}\0".encode()
    assert aggregate["sha256"] == hashlib.sha256(expected_record).hexdigest()
    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "reuse"


@pytest.mark.parametrize("mutation", ["replace", "append", "remove", "add"])
def test_completed_manifest_rejects_content_mutation(
    tmp_path: Path,
    mutation: str,
):
    artifact_dir = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "run"
    artifact_dir.mkdir()
    result = artifact_dir / "result.bin"
    result.write_bytes(b"complete")
    provenance.complete_artifact(manifest, artifact_dir, fields)

    if mutation == "replace":
        result.write_bytes(b"tampered")
    elif mutation == "append":
        result.write_bytes(b"complete-extra")
    elif mutation == "remove":
        result.unlink()
        (artifact_dir / "replacement.bin").write_bytes(b"complete")
    else:
        (artifact_dir / "extra.bin").write_bytes(b"extra")

    with pytest.raises(provenance.ProvenanceError, match="content"):
        provenance.claim_artifact(manifest, artifact_dir, fields)


def test_completed_manifest_hashes_and_revalidates_guard_directory(tmp_path: Path):
    artifact_dir = tmp_path / "checkpoints"
    guard_dir = tmp_path / "hidden-states"
    manifest = tmp_path / "runtime.provenance.json"
    fields = _pipeline_fields()
    assert (
        provenance.claim_artifact(manifest, artifact_dir, fields, [guard_dir])
        == "run"
    )
    artifact_dir.mkdir()
    guard_dir.mkdir()
    (artifact_dir / "model.safetensors").write_bytes(b"model")
    hidden_state = guard_dir / "hs_0.safetensors"
    hidden_state.write_bytes(b"hidden")
    provenance.complete_artifact(manifest, artifact_dir, fields, [guard_dir])

    completed = json.loads(manifest.read_text())
    assert set(completed["content_aggregates"]) == {
        str(artifact_dir.resolve()),
        str(guard_dir.resolve()),
    }
    hidden_state.write_bytes(b"changed")
    with pytest.raises(provenance.ProvenanceError, match="content"):
        provenance.claim_artifact(manifest, artifact_dir, fields, [guard_dir])


@pytest.mark.parametrize("unsafe_name", ["partial.tmp", "download.incomplete"])
def test_complete_rejects_temporary_artifact_files(
    tmp_path: Path,
    unsafe_name: str,
):
    artifact_dir = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "run"
    artifact_dir.mkdir()
    (artifact_dir / unsafe_name).write_bytes(b"partial")

    with pytest.raises(provenance.ProvenanceError, match="temporary"):
        provenance.complete_artifact(manifest, artifact_dir, fields)


def test_complete_rejects_recursive_artifact_symlink(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "run"
    artifact_dir.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (artifact_dir / "linked.bin").symlink_to(outside)

    with pytest.raises(provenance.ProvenanceError, match="symbolic link"):
        provenance.complete_artifact(manifest, artifact_dir, fields)


def test_claim_rejects_artifact_with_symlinked_ancestor(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(provenance.ProvenanceError, match="symbolic-link component"):
        provenance.claim_artifact(
            tmp_path / "artifact.provenance.json",
            linked_parent / "artifact",
            _pipeline_fields(),
        )


def test_claim_rejects_manifest_with_symlinked_ancestor(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(provenance.ProvenanceError, match="symbolic-link component"):
        provenance.claim_artifact(
            linked_parent / "artifact.provenance.json",
            tmp_path / "artifact",
            _pipeline_fields(),
        )


def test_manifest_must_be_outside_hashed_artifact(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    manifest = artifact_dir / "provenance.json"

    with pytest.raises(provenance.ProvenanceError, match="self-reference"):
        provenance.claim_artifact(manifest, artifact_dir, _pipeline_fields())


@pytest.mark.parametrize(
    "field",
    [
        "dataset_revision",
        "epochs",
        "max_samples",
        "model",
        "model_revision",
        "num_train_gpus",
        "seq_length",
        "source_revision",
        "vllm_extra_args",
        "vllm_max_model_len",
    ],
)
def test_manifest_mismatch_fails_closed(tmp_path: Path, field: str):
    artifact_dir = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "run"
    artifact_dir.mkdir()
    (artifact_dir / "result.bin").write_bytes(b"complete")
    provenance.complete_artifact(manifest, artifact_dir, fields)
    changed = dict(fields)
    changed[field] = f"different-{fields[field]}"

    with pytest.raises(provenance.ProvenanceError, match="mismatch"):
        provenance.claim_artifact(manifest, artifact_dir, changed)


def test_existing_artifact_without_manifest_fails_closed(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "stale.arrow").write_bytes(b"stale")

    with pytest.raises(provenance.ProvenanceError, match="without a provenance"):
        provenance.claim_artifact(
            tmp_path / "missing.provenance.json",
            artifact_dir,
            _pipeline_fields(),
        )


def test_unfinished_claim_cannot_be_reused(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "run"

    with pytest.raises(provenance.ProvenanceError, match="unfinished"):
        provenance.claim_artifact(manifest, artifact_dir, fields)


@pytest.mark.parametrize("artifact_state", ["missing", "empty", "symlink"])
def test_completed_manifest_requires_intact_artifact(
    tmp_path: Path,
    artifact_state: str,
):
    artifact_dir = tmp_path / "artifact"
    manifest = tmp_path / "artifact.provenance.json"
    fields = _pipeline_fields()
    assert provenance.claim_artifact(manifest, artifact_dir, fields) == "run"
    artifact_dir.mkdir()
    (artifact_dir / "result.bin").write_bytes(b"complete")
    provenance.complete_artifact(manifest, artifact_dir, fields)

    (artifact_dir / "result.bin").unlink()
    if artifact_state == "missing":
        artifact_dir.rmdir()
    elif artifact_state == "symlink":
        artifact_dir.rmdir()
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        (replacement / "result.bin").write_bytes(b"replacement")
        artifact_dir.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(provenance.ProvenanceError, match="artifact"):
        provenance.claim_artifact(manifest, artifact_dir, fields)
    with pytest.raises(provenance.ProvenanceError, match="artifact"):
        provenance.complete_artifact(manifest, artifact_dir, fields)


def test_completed_manifest_requires_intact_guard_directories(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    hidden_states_dir = tmp_path / "hidden-states"
    manifest = tmp_path / "runtime.provenance.json"
    fields = _pipeline_fields()
    assert (
        provenance.claim_artifact(
            manifest,
            checkpoint_dir,
            fields,
            [hidden_states_dir],
        )
        == "run"
    )
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    hidden_states_dir.mkdir()
    hidden_state = hidden_states_dir / "hs_0.safetensors"
    hidden_state.write_bytes(b"hidden")
    provenance.complete_artifact(
        manifest,
        checkpoint_dir,
        fields,
        [hidden_states_dir],
    )
    hidden_state.unlink()

    with pytest.raises(provenance.ProvenanceError, match="empty artifact"):
        provenance.claim_artifact(
            manifest,
            checkpoint_dir,
            fields,
            [hidden_states_dir],
        )


def test_validates_prepared_row_count_columns_and_sequence_length(
    tmp_path: Path,
    monkeypatch,
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "image.png"
    image_path.write_bytes(b"image")

    class FakeDataset:
        column_names = ["input_ids", "loss_mask", "messages", "seq_len"]

        def __len__(self):
            return 2

        def __iter__(self):
            yield {
                "input_ids": [1, 2],
                "loss_mask": [False, True],
                "messages": _valid_messages(image_path),
                "seq_len": 2,
            }
            yield {
                "input_ids": [3],
                "loss_mask": [1],
                "messages": _valid_messages(image_path),
                "seq_len": 1,
            }

    monkeypatch.setattr("datasets.load_from_disk", lambda _path: FakeDataset())

    assert (
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=2,
            seq_length=2,
        )
        == 2
    )


def test_rejects_prepared_sequence_over_approved_length(tmp_path: Path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "image.png"
    image_path.write_bytes(b"image")

    class FakeDataset:
        column_names = ["input_ids", "loss_mask", "messages", "seq_len"]

        def __len__(self):
            return 1

        def __iter__(self):
            yield {
                "input_ids": [1, 2, 3],
                "loss_mask": [0, 1, 1],
                "messages": _valid_messages(image_path),
                "seq_len": 3,
            }

    monkeypatch.setattr("datasets.load_from_disk", lambda _path: FakeDataset())

    with pytest.raises(provenance.ProvenanceError, match="sequence length"):
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=1,
            seq_length=2,
        )


def test_rejects_prepared_row_count_below_approved_exact_count(
    tmp_path: Path,
    monkeypatch,
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "image.png"
    image_path.write_bytes(b"image")

    class FakeDataset:
        column_names = ["input_ids", "loss_mask", "messages", "seq_len"]

        def __len__(self):
            return 1

        def __iter__(self):
            yield {
                "input_ids": [1],
                "loss_mask": [1],
                "messages": _valid_messages(image_path),
                "seq_len": 1,
            }

    monkeypatch.setattr("datasets.load_from_disk", lambda _path: FakeDataset())

    with pytest.raises(provenance.ProvenanceError, match="approved exact count"):
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=2,
            seq_length=2,
        )


@pytest.mark.parametrize(
    ("row_change", "error"),
    [
        ({"seq_len": 1}, "seq_len"),
        ({"loss_mask": [0, 0]}, "no trainable token"),
        ({"loss_mask": [0, 2]}, "binary values"),
        ({"messages": []}, "messages must be a non-empty list"),
    ],
)
def test_rejects_invalid_prepared_row_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row_change: dict,
    error: str,
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "image.png"
    image_path.write_bytes(b"image")
    row = {
        "input_ids": [1, 2],
        "loss_mask": [0, 1],
        "messages": _valid_messages(image_path),
        "seq_len": 2,
    }
    row.update(row_change)
    _install_fake_prepared_dataset(monkeypatch, row)

    with pytest.raises(provenance.ProvenanceError, match=error):
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=1,
            seq_length=2,
        )


def test_rejects_prepared_dataset_missing_required_seq_len(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "image.png"
    image_path.write_bytes(b"image")
    row = {
        "input_ids": [1],
        "loss_mask": [1],
        "messages": _valid_messages(image_path),
    }
    _install_fake_prepared_dataset(monkeypatch, row)

    with pytest.raises(provenance.ProvenanceError, match="seq_len"):
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=1,
            seq_length=2,
        )


def test_rejects_prepared_image_outside_dataset_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(b"outside")
    row = {
        "input_ids": [1],
        "loss_mask": [1],
        "messages": _valid_messages(outside_image),
        "seq_len": 1,
    }
    _install_fake_prepared_dataset(monkeypatch, row)

    with pytest.raises(provenance.ProvenanceError, match="escapes dataset_dir"):
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=1,
            seq_length=2,
        )


def test_rejects_prepared_image_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    real_image = dataset_dir / "real.png"
    real_image.write_bytes(b"image")
    linked_image = dataset_dir / "linked.png"
    linked_image.symlink_to(real_image)
    messages = _valid_messages(real_image)
    messages[0]["content"][0]["image_url"]["url"] = linked_image.absolute().as_uri()
    row = {
        "input_ids": [1],
        "loss_mask": [1],
        "messages": messages,
        "seq_len": 1,
    }
    _install_fake_prepared_dataset(monkeypatch, row)

    with pytest.raises(provenance.ProvenanceError, match="symbolic link"):
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=1,
            seq_length=2,
        )


def test_rejects_prepared_row_with_multiple_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "image.png"
    image_path.write_bytes(b"image")
    messages = _valid_messages(image_path)
    messages[0]["content"].append(
        {
            "type": "image_url",
            "image_url": {"url": image_path.resolve().as_uri()},
        }
    )
    row = {
        "input_ids": [1],
        "loss_mask": [1],
        "messages": messages,
        "seq_len": 1,
    }
    _install_fake_prepared_dataset(monkeypatch, row)

    with pytest.raises(provenance.ProvenanceError, match="exactly one user image"):
        provenance.validate_prepared_dataset(
            prepared_dir,
            dataset_dir=dataset_dir,
            max_samples=1,
            seq_length=2,
        )


def test_validates_exact_runtime_hidden_states_and_completed_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)

    assert (
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )
        == 2
    )


def test_runtime_accepts_inactive_noncanonical_source_commit_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """VL prefix repair leaves an inode-stable lock for its staging source."""
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    (hidden_states_dir / ".vllm-response-abc.safetensors.commit.lock").touch()

    assert (
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )
        == 2
    )


def test_runtime_rejects_active_noncanonical_source_commit_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    lock_path = hidden_states_dir / ".vllm-response-abc.safetensors.commit.lock"
    lock_path.touch()
    descriptor = os.open(lock_path, os.O_RDONLY)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(provenance.ProvenanceError, match="still active"):
            provenance.validate_runtime_artifacts(
                checkpoint_dir,
                hidden_states_dir,
                prepared_dir,
                max_samples=2,
                epochs=2,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_runtime_rejects_missing_canonical_hidden_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    (hidden_states_dir / "hs_1.safetensors").unlink()

    with pytest.raises(provenance.ProvenanceError, match="cache set is not exact"):
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )


def test_runtime_rejects_hidden_state_token_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    (hidden_states_dir / "hs_1.safetensors").unlink()
    save_file(
        {
            "token_ids": torch.tensor([99, 100], dtype=torch.long),
            "hidden_states": torch.randn(2, 2, 3),
        },
        hidden_states_dir / "hs_1.safetensors",
    )

    with pytest.raises(provenance.ProvenanceError, match="hidden-state cache"):
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )


def test_runtime_rejects_nonfinite_hidden_state_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    hidden_states = torch.randn(2, 2, 3)
    hidden_states[1, 1, 1] = torch.nan
    (hidden_states_dir / "hs_1.safetensors").unlink()
    save_file(
        {
            "token_ids": torch.tensor(rows[1]["input_ids"], dtype=torch.long),
            "hidden_states": hidden_states,
        },
        hidden_states_dir / "hs_1.safetensors",
    )

    with pytest.raises(provenance.ProvenanceError, match="hidden-state cache"):
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )


def test_runtime_rejects_nonfinite_model_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    (checkpoint_dir / "0" / "model.safetensors").unlink()
    save_file(
        {"weight": torch.tensor([[1.0, torch.inf]])},
        checkpoint_dir / "0" / "model.safetensors",
    )

    with pytest.raises(provenance.ProvenanceError, match="NaN or Inf"):
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )


def test_runtime_rejects_unloadable_optimizer_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    (checkpoint_dir / "0" / "optimizer_state_dict.pt").write_bytes(
        b"not a torch checkpoint"
    )

    with pytest.raises(provenance.ProvenanceError, match="not loadable"):
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )


def test_runtime_rejects_incomplete_checkpoint_epoch_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    for file_path in (checkpoint_dir / "1").iterdir():
        file_path.unlink()
    (checkpoint_dir / "1").rmdir()

    with pytest.raises(provenance.ProvenanceError, match="epoch set is incomplete"):
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )


def test_runtime_rejects_mid_epoch_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir, hidden_states_dir, prepared_dir, rows = _write_runtime_fixture(
        tmp_path
    )
    _install_indexable_prepared_dataset(monkeypatch, rows)
    (checkpoint_dir / "1" / "training_state.json").write_text(
        json.dumps({"epoch": 1, "local_step": 3, "global_step": 4})
    )

    with pytest.raises(provenance.ProvenanceError, match="mid-epoch"):
        provenance.validate_runtime_artifacts(
            checkpoint_dir,
            hidden_states_dir,
            prepared_dir,
            max_samples=2,
            epochs=2,
        )


def _write_shard(data_dir: Path, index: int, total: int, *, width: int = 5) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    name = f"train-{index:0{width}d}-of-{total:0{width}d}.parquet"
    (data_dir / name).write_bytes(b"PAR1")


def test_validates_complete_train_parquet_shards(tmp_path: Path):
    data_dir = tmp_path / "dataset" / "data"
    _write_shard(data_dir, 1, 2)
    _write_shard(data_dir, 0, 2)

    shards = provenance.validate_train_parquet_shards(tmp_path / "dataset")

    assert [path.name for path in shards] == [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]


def test_rejects_mixed_parquet_shard_totals(tmp_path: Path):
    data_dir = tmp_path / "dataset" / "data"
    _write_shard(data_dir, 0, 2)
    _write_shard(data_dir, 1, 3)

    with pytest.raises(provenance.ProvenanceError, match="mixed totals"):
        provenance.validate_train_parquet_shards(tmp_path / "dataset")


def test_rejects_missing_parquet_shard_index(tmp_path: Path):
    data_dir = tmp_path / "dataset" / "data"
    _write_shard(data_dir, 0, 2)

    with pytest.raises(provenance.ProvenanceError, match="incomplete or duplicated"):
        provenance.validate_train_parquet_shards(tmp_path / "dataset")


def test_rejects_duplicate_parquet_shard_index(tmp_path: Path):
    data_dir = tmp_path / "dataset" / "data"
    _write_shard(data_dir, 0, 2, width=1)
    _write_shard(data_dir, 0, 2, width=5)

    with pytest.raises(provenance.ProvenanceError, match="incomplete or duplicated"):
        provenance.validate_train_parquet_shards(tmp_path / "dataset")


def test_rejects_unexpected_stale_parquet_file(tmp_path: Path):
    data_dir = tmp_path / "dataset" / "data"
    _write_shard(data_dir, 0, 1)
    (data_dir / "train-old.parquet").write_bytes(b"stale")

    with pytest.raises(provenance.ProvenanceError, match="stale shard mix"):
        provenance.validate_train_parquet_shards(tmp_path / "dataset")
