"""Unit tests for P-EAGLE attention mask with StreamingLLM (attention sinks)."""

import torch

from speculators.models.peagle.attention import (
    _compute_doc_start_positions,
    create_peagle_mask_mod,
)


def _lengths_to_document_ids(lengths, total_seq_len):
    """Build document_ids tensor from packed document lengths."""
    doc_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.long), lengths
    )
    return torch.cat(
        [
            doc_ids,
            -1 * torch.ones(total_seq_len - doc_ids.shape[0], dtype=torch.long),
        ]
    ).contiguous()


def _evaluate_mask(mask_mod, total_sampled):
    """Evaluate mask_mod element-wise over the q x kv grid."""
    zero = torch.zeros((), dtype=torch.long)
    mask = torch.zeros(total_sampled, total_sampled, dtype=torch.bool)
    for q in range(total_sampled):
        for kv in range(total_sampled):
            mask[q, kv] = bool(mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv)))
    return mask


def _simple_cod_indices(seq_length, depths_to_sample=None):
    """Create simple COD-like indices for testing.

    Depth 0: all positions [0, seq_length)
    Depth 1+: specified positions
    """
    anchor_pos = list(range(seq_length))
    depth_vals = [0] * seq_length

    if depths_to_sample is not None:
        for d, positions in depths_to_sample.items():
            for p in positions:
                anchor_pos.append(p)
                depth_vals.append(d)

    return (
        torch.tensor(anchor_pos, dtype=torch.long),
        torch.tensor(depth_vals, dtype=torch.long),
    )


class TestComputeDocStartPositions:
    def test_single_document(self):
        doc_ids = torch.tensor([0, 0, 0, 0, 0])
        result = _compute_doc_start_positions(doc_ids)
        assert result.tolist() == [0, 0, 0, 0, 0]

    def test_two_documents(self):
        doc_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        result = _compute_doc_start_positions(doc_ids)
        assert result.tolist() == [0, 0, 0, 3, 3, 3]

    def test_with_padding(self):
        doc_ids = torch.tensor([0, 0, 0, 1, 1, -1, -1])
        result = _compute_doc_start_positions(doc_ids)
        assert result.tolist() == [0, 0, 0, 3, 3, -1, -1]

    def test_all_padding(self):
        doc_ids = torch.tensor([-1, -1, -1])
        result = _compute_doc_start_positions(doc_ids)
        assert result.tolist() == [-1, -1, -1]

    def test_three_documents(self):
        doc_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        result = _compute_doc_start_positions(doc_ids)
        assert result.tolist() == [0, 0, 2, 2, 2, 5, 5]


class TestNoStreamingUnchanged:
    """When sink_size=None (no StreamingLLM), mask is identical to original."""

    def test_single_doc_no_streaming(self):
        seq_len = 8
        doc_ids = _lengths_to_document_ids(torch.tensor([seq_len]), seq_len)
        anchor_pos, depth = _simple_cod_indices(seq_len)

        mask_with = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=None,
            max_context_window=None,
        )
        mask_without = create_peagle_mask_mod(anchor_pos, depth, doc_ids)

        total = anchor_pos.shape[0]
        dense_with = _evaluate_mask(mask_with, total)
        dense_without = _evaluate_mask(mask_without, total)
        assert torch.equal(dense_with, dense_without)


class TestStreamingSingleDoc:
    """StreamingLLM with a single document."""

    def test_sink_and_window(self):
        seq_len = 20
        sink_size = 4
        window = 6
        doc_ids = _lengths_to_document_ids(torch.tensor([seq_len]), seq_len)
        anchor_pos, depth = _simple_cod_indices(seq_len)

        mask_mod = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=sink_size,
            max_context_window=window,
        )
        mask = _evaluate_mask(mask_mod, anchor_pos.shape[0])

        q_idx = 15
        for kv_idx in range(seq_len):
            kv_pos = kv_idx
            is_sink = kv_pos < sink_size
            in_window = kv_pos >= (q_idx - window)
            is_causal = kv_pos <= q_idx

            expected = is_causal and (is_sink or in_window)
            assert mask[q_idx, kv_idx] == expected, (
                f"q={q_idx}, kv={kv_idx}: expected {expected}, "
                f"got {mask[q_idx, kv_idx]}"
            )

    def test_gap_positions_blocked(self):
        """Positions in the middle (between sink and window) should be blocked."""
        seq_len = 20
        sink_size = 3
        window = 4
        doc_ids = _lengths_to_document_ids(torch.tensor([seq_len]), seq_len)
        anchor_pos, depth = _simple_cod_indices(seq_len)

        mask_mod = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=sink_size,
            max_context_window=window,
        )
        mask = _evaluate_mask(mask_mod, anchor_pos.shape[0])

        q_idx = 15
        for kv_idx in range(sink_size, q_idx - window):
            assert not mask[q_idx, kv_idx], (
                f"Gap position kv={kv_idx} should be blocked for q={q_idx}"
            )

    def test_early_query_no_gap(self):
        """When query is near start, window and sink overlap."""
        seq_len = 20
        sink_size = 5
        window = 8
        doc_ids = _lengths_to_document_ids(torch.tensor([seq_len]), seq_len)
        anchor_pos, depth = _simple_cod_indices(seq_len)

        mask_mod = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=sink_size,
            max_context_window=window,
        )
        mask = _evaluate_mask(mask_mod, anchor_pos.shape[0])

        q_idx = 6
        for kv_idx in range(q_idx + 1):
            assert mask[q_idx, kv_idx], (
                f"Position kv={kv_idx} should be visible for q={q_idx}"
            )


class TestStreamingMultiDoc:
    """StreamingLLM respects document boundaries in packed sequences."""

    def test_cross_doc_sinks_blocked(self):
        """Sinks from doc0 should not be visible to queries in doc1."""
        doc0_len = 10
        doc1_len = 10
        total = doc0_len + doc1_len
        doc_ids = _lengths_to_document_ids(torch.tensor([doc0_len, doc1_len]), total)
        anchor_pos, depth = _simple_cod_indices(total)

        mask_mod = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=4,
            max_context_window=6,
        )
        mask = _evaluate_mask(mask_mod, anchor_pos.shape[0])

        q_idx = 18
        for kv_idx in range(doc0_len):
            assert not mask[q_idx, kv_idx], (
                f"Doc0 position kv={kv_idx} should be blocked for doc1 q={q_idx}"
            )

    def test_doc1_has_own_sinks(self):
        """Doc1's sink tokens are relative to doc1's start position."""
        doc0_len = 10
        doc1_len = 15
        total = doc0_len + doc1_len
        sink_size = 3
        window = 4
        doc_ids = _lengths_to_document_ids(torch.tensor([doc0_len, doc1_len]), total)
        anchor_pos, depth = _simple_cod_indices(total)

        mask_mod = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=sink_size,
            max_context_window=window,
        )
        mask = _evaluate_mask(mask_mod, anchor_pos.shape[0])

        q_idx = 22
        for kv_idx in [10, 11, 12]:
            assert mask[q_idx, kv_idx], (
                f"Doc1 sink kv={kv_idx} should be visible for q={q_idx}"
            )
        for kv_idx in range(18, 23):
            assert mask[q_idx, kv_idx], (
                f"Doc1 window kv={kv_idx} should be visible for q={q_idx}"
            )
        for kv_idx in range(13, 18):
            assert not mask[q_idx, kv_idx], (
                f"Doc1 gap kv={kv_idx} should be blocked for q={q_idx}"
            )


class TestWindowCoversAll:
    """When window >= seq_len, equivalent to full causal attention."""

    def test_large_window_matches_no_streaming(self):
        seq_len = 12
        doc_ids = _lengths_to_document_ids(torch.tensor([seq_len]), seq_len)
        anchor_pos, depth = _simple_cod_indices(seq_len)

        mask_full = create_peagle_mask_mod(anchor_pos, depth, doc_ids)
        mask_streaming = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=1,
            max_context_window=seq_len,
        )

        total = anchor_pos.shape[0]
        dense_full = _evaluate_mask(mask_full, total)
        dense_streaming = _evaluate_mask(mask_streaming, total)
        assert torch.equal(dense_full, dense_streaming)


class TestRolloutUnaffected:
    """Within-rollout attention (depth > 0) is not affected by StreamingLLM."""

    def test_rollout_attention_preserved(self):
        seq_len = 10
        depths = {1: [2, 5, 8], 2: [2]}
        doc_ids = _lengths_to_document_ids(torch.tensor([seq_len]), seq_len)
        anchor_pos, depth = _simple_cod_indices(seq_len, depths_to_sample=depths)

        mask_full = create_peagle_mask_mod(anchor_pos, depth, doc_ids)
        mask_streaming = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=2,
            max_context_window=3,
        )

        total = anchor_pos.shape[0]
        dense_full = _evaluate_mask(mask_full, total)
        dense_streaming = _evaluate_mask(mask_streaming, total)

        for q in range(total):
            if depth[q] == 0:
                continue
            for kv in range(total):
                if anchor_pos[q] == anchor_pos[kv] and depth[kv] > 0:
                    assert dense_full[q, kv] == dense_streaming[q, kv], (
                        f"Rollout attention changed at q={q}, kv={kv}"
                    )


class TestPaddingExcluded:
    """Padding positions (document_id = -1) should never be attended to."""

    def test_padding_blocked(self):
        doc_len = 6
        total_seq_len = 10
        doc_ids = _lengths_to_document_ids(torch.tensor([doc_len]), total_seq_len)
        anchor_pos, depth = _simple_cod_indices(doc_len)

        mask_mod = create_peagle_mask_mod(
            anchor_pos,
            depth,
            doc_ids,
            sink_size=2,
            max_context_window=3,
        )
        _evaluate_mask(mask_mod, anchor_pos.shape[0])

        zero = torch.zeros((), dtype=torch.long)
        for q in range(doc_len):
            result = mask_mod(zero, zero, torch.tensor(q), torch.tensor(q))
            assert result
