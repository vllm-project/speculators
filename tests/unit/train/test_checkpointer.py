import shutil
from pathlib import Path

import pytest

from speculators.train.checkpointer import BaseCheckpointer, SingleGPUCheckpointer


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path):
    """Create a temporary checkpoint directory."""
    return tmp_path / "checkpoints"


class TestCleanupOldCheckpoints:
    """Tests for BaseCheckpointer.cleanup_old_checkpoints."""

    def _create_checkpoint_dirs(
        self, base: Path, epoch_nums: list[int]
    ) -> list[Path]:
        """Helper to create fake checkpoint directories."""
        dirs = []
        for num in epoch_nums:
            d = base / str(num)
            d.mkdir(parents=True, exist_ok=True)
            # Create a marker file
            (d / "model.safetensors").touch()
            dirs.append(d)
        return dirs

    def test_cleanup_keeps_max_checkpoints(self, tmp_checkpoint_dir: Path):
        """Only the most recent checkpoints are kept."""
        self._create_checkpoint_dirs(tmp_checkpoint_dir, [0, 1, 2, 3, 4])
        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)

        ckpt.cleanup_old_checkpoints(max_checkpoints=2)

        remaining = sorted(
            int(d.name) for d in tmp_checkpoint_dir.iterdir() if d.is_dir()
        )
        assert remaining == [3, 4]

    def test_cleanup_no_op_when_fewer_than_max(self, tmp_checkpoint_dir: Path):
        """No directories are removed when count <= max_checkpoints."""
        self._create_checkpoint_dirs(tmp_checkpoint_dir, [0, 1])
        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)

        ckpt.cleanup_old_checkpoints(max_checkpoints=5)

        remaining = sorted(
            int(d.name) for d in tmp_checkpoint_dir.iterdir() if d.is_dir()
        )
        assert remaining == [0, 1]

    def test_cleanup_no_op_when_equal_to_max(self, tmp_checkpoint_dir: Path):
        """No directories are removed when count == max_checkpoints."""
        self._create_checkpoint_dirs(tmp_checkpoint_dir, [0, 1, 2])
        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)

        ckpt.cleanup_old_checkpoints(max_checkpoints=3)

        remaining = sorted(
            int(d.name) for d in tmp_checkpoint_dir.iterdir() if d.is_dir()
        )
        assert remaining == [0, 1, 2]

    def test_cleanup_ignores_non_numeric_dirs(self, tmp_checkpoint_dir: Path):
        """Non-numeric directories are not counted or removed."""
        self._create_checkpoint_dirs(tmp_checkpoint_dir, [0, 1, 2, 3])
        # Create a non-numeric dir
        (tmp_checkpoint_dir / "best").mkdir(parents=True, exist_ok=True)

        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)
        ckpt.cleanup_old_checkpoints(max_checkpoints=2)

        remaining = {d.name for d in tmp_checkpoint_dir.iterdir() if d.is_dir()}
        assert remaining == {"2", "3", "best"}

    def test_cleanup_with_nonexistent_path(self, tmp_checkpoint_dir: Path):
        """No error when checkpoint path doesn't exist."""
        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)
        # Should not raise
        ckpt.cleanup_old_checkpoints(max_checkpoints=2)

    def test_cleanup_max_one(self, tmp_checkpoint_dir: Path):
        """Setting max_checkpoints=1 keeps only the latest."""
        self._create_checkpoint_dirs(tmp_checkpoint_dir, [0, 1, 2])
        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)

        ckpt.cleanup_old_checkpoints(max_checkpoints=1)

        remaining = sorted(
            int(d.name) for d in tmp_checkpoint_dir.iterdir() if d.is_dir()
        )
        assert remaining == [2]


class TestGetPreviousEpoch:
    """Tests for BaseCheckpointer._get_previous_epoch."""

    def test_no_checkpoints(self, tmp_checkpoint_dir: Path):
        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)
        assert ckpt.previous_epoch == -1

    def test_finds_highest_epoch(self, tmp_checkpoint_dir: Path):
        for num in [0, 1, 5, 3]:
            (tmp_checkpoint_dir / str(num)).mkdir(parents=True, exist_ok=True)

        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)
        assert ckpt.previous_epoch == 5

    def test_ignores_non_numeric(self, tmp_checkpoint_dir: Path):
        (tmp_checkpoint_dir / "0").mkdir(parents=True, exist_ok=True)
        (tmp_checkpoint_dir / "best").mkdir(parents=True, exist_ok=True)

        ckpt = SingleGPUCheckpointer(tmp_checkpoint_dir)
        assert ckpt.previous_epoch == 0
