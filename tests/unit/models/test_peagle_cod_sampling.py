"""Unit tests for P-EAGLE COD sampling utilities."""

import pytest
import torch

from speculators.models.peagle.cod_sampling import (
    build_depth_position_ids,
    cod_sample,
    compute_cod_statistics,
)


class TestCodSample:
    """Tests for the cod_sample function."""

    def test_basic_sampling(self):
        """Test basic COD sampling with a simple loss mask."""
        loss_mask = torch.tensor([True, False, True, True, True, False, True])
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=3)

        assert len(indices) == 3
        # Depth 0 should retain all valid positions
        assert torch.equal(indices[0], torch.tensor([0, 2, 3, 4, 6]))
        # Deeper depths should have fewer positions
        assert len(indices[1]) <= len(indices[0])
        assert len(indices[2]) <= len(indices[1])

    def test_depth_0_retains_all_valid(self):
        """Depth 0 should always retain all valid positions."""
        loss_mask = torch.tensor([True, True, False, True, False, True])
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=4)

        expected = torch.tensor([0, 1, 3, 5])
        assert torch.equal(indices[0], expected)

    def test_geometric_decay_count(self):
        """Verify geometric decay in position counts."""
        # Use a large sequence to make statistics reliable
        n = 1000
        loss_mask = torch.ones(n, dtype=torch.bool)
        r = 0.5
        K = 4

        generator = torch.Generator().manual_seed(42)
        indices = cod_sample(
            loss_mask, down_sample_ratio=r, num_depths=K, generator=generator
        )

        # Depth k should retain approximately n * r^k positions
        for k in range(K):
            expected = int(n * r**k)
            actual = len(indices[k])
            if k == 0:
                assert actual == n
            else:
                assert actual == expected

    def test_single_depth(self):
        """num_depths=1 should only return depth 0 (all valid positions)."""
        loss_mask = torch.tensor([True, False, True, True])
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=1)

        assert len(indices) == 1
        assert torch.equal(indices[0], torch.tensor([0, 2, 3]))

    def test_empty_loss_mask(self):
        """Test with no valid positions."""
        loss_mask = torch.zeros(10, dtype=torch.bool)
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=3)

        assert len(indices) == 3
        for k in range(3):
            assert indices[k].numel() == 0

    def test_all_valid_positions(self):
        """Test with all positions valid."""
        loss_mask = torch.ones(10, dtype=torch.bool)
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=2)

        assert len(indices) == 2
        assert len(indices[0]) == 10
        assert len(indices[1]) == 5  # floor(10 * 0.5)

    def test_sorted_order(self):
        """Sampled indices should be in sorted (causal) order."""
        loss_mask = torch.ones(100, dtype=torch.bool)
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=5)

        for k in range(5):
            if indices[k].numel() > 1:
                diffs = indices[k][1:] - indices[k][:-1]
                assert (diffs > 0).all(), f"Depth {k} indices not sorted"

    def test_indices_are_valid_positions(self):
        """All returned indices should be positions where loss_mask is True."""
        loss_mask = torch.tensor(
            [True, False, True, False, True, True, False, True, True, False]
        )
        valid = torch.nonzero(loss_mask, as_tuple=False).squeeze(-1)
        valid_set = set(valid.tolist())

        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=4)

        for k in range(4):
            for idx in indices[k].tolist():
                assert idx in valid_set, f"Index {idx} at depth {k} is not valid"

    def test_reproducibility_with_generator(self):
        """Same generator seed should produce same results."""
        loss_mask = torch.ones(50, dtype=torch.bool)

        gen1 = torch.Generator().manual_seed(123)
        indices1 = cod_sample(
            loss_mask, down_sample_ratio=0.5, num_depths=4, generator=gen1
        )

        gen2 = torch.Generator().manual_seed(123)
        indices2 = cod_sample(
            loss_mask, down_sample_ratio=0.5, num_depths=4, generator=gen2
        )

        for k in range(4):
            assert torch.equal(indices1[k], indices2[k])

    def test_down_sample_ratio_min(self):
        """Minimum retention rate should prevent zero positions at deep depths."""
        loss_mask = torch.ones(100, dtype=torch.bool)
        # With r=0.1 and K=10, depth 9 would have 100 * 0.1^9 ≈ 0
        # but with min=0.05, depth 9 should have at least 100 * 0.05 = 5
        indices = cod_sample(
            loss_mask,
            down_sample_ratio=0.1,
            num_depths=10,
            down_sample_ratio_min=0.05,
        )

        for k in range(10):
            expected_min = int(100 * 0.05)
            if k == 0:
                assert len(indices[k]) == 100
            else:
                # With min, should have at least floor(100 * 0.05) = 5
                actual_ratio = max(0.1**k, 0.05)
                expected = int(100 * actual_ratio)
                assert len(indices[k]) == expected

    def test_invalid_down_sample_ratio(self):
        """Should raise ValueError for invalid ratio values."""
        loss_mask = torch.ones(10, dtype=torch.bool)

        with pytest.raises(ValueError, match="down_sample_ratio must be in"):
            cod_sample(loss_mask, down_sample_ratio=0.0, num_depths=2)

        with pytest.raises(ValueError, match="down_sample_ratio must be in"):
            cod_sample(loss_mask, down_sample_ratio=1.0, num_depths=2)

        with pytest.raises(ValueError, match="down_sample_ratio must be in"):
            cod_sample(loss_mask, down_sample_ratio=-0.5, num_depths=2)

        with pytest.raises(ValueError, match="down_sample_ratio must be in"):
            cod_sample(loss_mask, down_sample_ratio=1.5, num_depths=2)

    def test_invalid_num_depths(self):
        """Should raise ValueError for num_depths < 1."""
        loss_mask = torch.ones(10, dtype=torch.bool)

        with pytest.raises(ValueError, match="num_depths must be >= 1"):
            cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=0)

    def test_invalid_min_ratio(self):
        """Should raise ValueError for invalid min ratio."""
        loss_mask = torch.ones(10, dtype=torch.bool)

        with pytest.raises(ValueError, match="down_sample_ratio_min must be in"):
            cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=2,
                       down_sample_ratio_min=-0.1)

        with pytest.raises(ValueError, match="down_sample_ratio_min must be in"):
            cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=2,
                       down_sample_ratio_min=1.5)

    def test_very_small_sequence(self):
        """Test with single-element sequence."""
        loss_mask = torch.tensor([True])
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=3)

        assert len(indices) == 3
        assert torch.equal(indices[0], torch.tensor([0]))
        # floor(1 * 0.5) = 0, so deeper depths should be empty
        assert indices[1].numel() == 0
        assert indices[2].numel() == 0

    def test_high_retention_rate(self):
        """Test with retention rate close to 1."""
        loss_mask = torch.ones(20, dtype=torch.bool)
        indices = cod_sample(loss_mask, down_sample_ratio=0.9, num_depths=3)

        # All depths should retain most positions
        assert len(indices[0]) == 20
        assert len(indices[1]) == 18  # floor(20 * 0.9)
        assert len(indices[2]) == 16  # floor(20 * 0.81)

    def test_low_retention_rate(self):
        """Test with very low retention rate."""
        loss_mask = torch.ones(100, dtype=torch.bool)
        indices = cod_sample(loss_mask, down_sample_ratio=0.1, num_depths=4)

        assert len(indices[0]) == 100
        assert len(indices[1]) == 10  # floor(100 * 0.1)
        assert len(indices[2]) == 1   # floor(100 * 0.01)
        assert len(indices[3]) == 0   # floor(100 * 0.001) = 0

    def test_no_duplicate_indices(self):
        """Each depth should have unique indices."""
        loss_mask = torch.ones(50, dtype=torch.bool)
        indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=5)

        for k in range(5):
            unique = torch.unique(indices[k])
            assert len(unique) == len(indices[k]), f"Duplicates at depth {k}"


class TestComputeCodStatistics:
    """Tests for compute_cod_statistics."""

    def test_basic_statistics(self):
        """Test basic statistics computation."""
        indices = [
            torch.tensor([0, 1, 2, 3, 4]),
            torch.tensor([1, 3]),
            torch.tensor([2]),
        ]
        stats = compute_cod_statistics(indices, seq_len=10)

        assert stats["num_depths"] == 3
        assert stats["seq_len"] == 10
        assert stats["per_depth_counts"] == [5, 2, 1]
        assert stats["total_positions"] == 8
        assert stats["naive_total"] == 15
        assert abs(stats["compression_ratio"] - 8 / 15) < 1e-5

    def test_empty_indices(self):
        """Test with empty depth indices."""
        indices = [torch.tensor([0, 1, 2]), torch.tensor([])]
        stats = compute_cod_statistics(indices, seq_len=5)

        assert stats["total_positions"] == 3
        assert stats["per_depth_counts"] == [3, 0]

    def test_single_depth(self):
        """Test with single depth."""
        indices = [torch.tensor([0, 1, 2, 3])]
        stats = compute_cod_statistics(indices, seq_len=5)

        assert stats["compression_ratio"] == 1.0


class TestBuildDepthPositionIds:
    """Tests for build_depth_position_ids."""

    def test_basic_position_ids(self):
        """Test basic position ID building."""
        indices = [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([1, 3]),
            torch.tensor([2]),
        ]
        pos_ids = build_depth_position_ids(indices, seq_len=5)

        expected = torch.tensor([0, 1, 2, 3, 1, 3, 2])
        assert torch.equal(pos_ids, expected)

    def test_empty_depths(self):
        """Test with some empty depths."""
        indices = [
            torch.tensor([0, 1]),
            torch.tensor([], dtype=torch.long),
            torch.tensor([0]),
        ]
        pos_ids = build_depth_position_ids(indices, seq_len=3)

        expected = torch.tensor([0, 1, 0])
        assert torch.equal(pos_ids, expected)

    def test_all_empty(self):
        """Test with all empty indices."""
        indices = []
        pos_ids = build_depth_position_ids(indices, seq_len=0)

        assert pos_ids.numel() == 0

    def test_device_placement(self):
        """Test that output respects device parameter."""
        indices = [torch.tensor([0, 1, 2])]
        pos_ids = build_depth_position_ids(indices, seq_len=3, device=torch.device("cpu"))

        assert pos_ids.device == torch.device("cpu")
