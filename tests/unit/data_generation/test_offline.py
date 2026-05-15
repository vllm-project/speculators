from speculators.data_generation.offline import (
    get_existing_hidden_state_indices,
    get_indices_to_process,
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
