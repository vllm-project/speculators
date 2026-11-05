import pytest
import torch
from safetensors.torch import save_file

from speculators.data_generation.offline import (
    check_hidden_states,
    find_corrupt_hidden_state_indices,
    get_existing_hidden_state_indices,
    get_indices_to_process,
)


def test_check_hidden_states_reports_non_finite_values():
    hidden_states = torch.zeros(3, 4, 2, dtype=torch.bfloat16)
    hidden_states[1, 2, 0] = torch.nan

    with pytest.raises(
        ValueError,
        match=r"min=nan, max=nan",
    ):
        check_hidden_states(
            {"token_ids": torch.tensor([1, 2, 3]), "hidden_states": hidden_states},
            [1, 2, 3],
        )


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
        (tmp_path / "hs_0.safetensors").touch()
        (tmp_path / "hs_3.safetensors").touch()
        (tmp_path / "hs_7.safetensors").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0, 3, 7]

    def test_ignores_non_numeric_suffixes(self, tmp_path):
        (tmp_path / "hs_0.safetensors").touch()
        (tmp_path / "hs_abc.safetensors").touch()
        (tmp_path / "hs_.safetensors").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0]

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "hs_0.safetensors").touch()
        (tmp_path / "other_file.txt").touch()
        (tmp_path / "hs_1.pt").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0]

    def test_results_are_sorted(self, tmp_path):
        for i in [9, 2, 5, 0]:
            (tmp_path / f"hs_{i}.safetensors").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0, 2, 5, 9]


# ===== find_corrupt_hidden_state_indices Tests =====


def _write_sample(path, seq_len=4):
    save_file(
        {
            "token_ids": torch.arange(seq_len, dtype=torch.int64),
            "hidden_states": torch.zeros(seq_len, 3, 2, dtype=torch.bfloat16),
        },
        path,
    )


def _truncate(path):
    path.write_bytes(path.read_bytes()[:-1])


class TestFindCorruptHiddenStateIndices:
    def test_healthy_file_is_not_reported(self, tmp_path):
        _write_sample(tmp_path / "hs_0.safetensors")
        assert find_corrupt_hidden_state_indices(tmp_path, [0]) == {}

    def test_truncated_file_is_reported_with_a_reason(self, tmp_path):
        path = tmp_path / "hs_0.safetensors"
        _write_sample(path)
        _truncate(path)

        corrupt = find_corrupt_hidden_state_indices(tmp_path, [0])

        assert list(corrupt) == [0]
        assert corrupt[0]

    def test_extra_trailing_bytes_are_reported(self, tmp_path):
        # Buffer coverage has to be exact in both directions, so a file that
        # grew is just as unreadable as one that was cut short.
        path = tmp_path / "hs_0.safetensors"
        _write_sample(path)
        path.write_bytes(path.read_bytes() + b"\x00")

        assert list(find_corrupt_hidden_state_indices(tmp_path, [0])) == [0]

    def test_empty_file_is_reported(self, tmp_path):
        (tmp_path / "hs_0.safetensors").touch()
        assert list(find_corrupt_hidden_state_indices(tmp_path, [0])) == [0]

    def test_absent_file_is_skipped(self, tmp_path):
        # Absence already means "not generated", which is not corruption.
        assert find_corrupt_hidden_state_indices(tmp_path, [7]) == {}

    def test_absent_directory_is_skipped(self, tmp_path):
        assert find_corrupt_hidden_state_indices(tmp_path / "nope", [0, 1]) == {}

    def test_only_requested_indices_are_checked(self, tmp_path):
        _write_sample(tmp_path / "hs_0.safetensors")
        broken = tmp_path / "hs_1.safetensors"
        _write_sample(broken)
        _truncate(broken)

        assert find_corrupt_hidden_state_indices(tmp_path, [0]) == {}
        assert list(find_corrupt_hidden_state_indices(tmp_path, [0, 1])) == [1]

    def test_corrupt_index_rejoins_the_work_queue(self, tmp_path):
        # The resume path must not treat mere presence as "done", otherwise an
        # interrupted run leaves a file that is skipped forever.
        _write_sample(tmp_path / "hs_0.safetensors")
        broken = tmp_path / "hs_1.safetensors"
        _write_sample(broken)
        _truncate(broken)
        _write_sample(tmp_path / "hs_2.safetensors")

        existing = get_existing_hidden_state_indices(tmp_path)
        assert existing == [0, 1, 2]

        corrupt = find_corrupt_hidden_state_indices(tmp_path, existing)
        existing = [i for i in existing if i not in corrupt]

        assert get_indices_to_process(3, None, existing, world_size=1, rank=0) == [1]
