import pytest
import torch

from speculators.train.trainer import _BatchPrefetcher


def _batches(n):
    return [{"x": torch.full((4,), float(i)), "meta": i} for i in range(n)]


def test_disabled_prefetcher_yields_batches_unstaged():
    batches = _batches(3)
    prefetcher = _BatchPrefetcher(batches, device=0, enabled=False)
    seen = []
    for batch, staged in prefetcher:
        prefetcher.prefetch()
        assert staged is None
        seen.append(batch["meta"])
    assert seen == [0, 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefetched_batches_keep_order_and_values():
    batches = _batches(4)
    prefetcher = _BatchPrefetcher(batches, device=0, enabled=True)
    seen = []
    for i, (batch, staged) in enumerate(prefetcher):
        if i == 0:
            assert staged is None
        else:
            assert staged["x"].is_cuda
            assert "meta" not in staged
            torch.testing.assert_close(staged["x"].cpu(), batch["x"])
        seen.append(batch["meta"])
        prefetcher.prefetch()
    assert seen == [0, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefetch_error_is_raised_when_reached():
    def loader():
        yield {"x": torch.zeros(2)}
        raise RuntimeError("bad batch")

    prefetcher = _BatchPrefetcher(loader(), device=0, enabled=True)
    iterator = iter(prefetcher)
    next(iterator)
    prefetcher.prefetch()
    with pytest.raises(RuntimeError, match="bad batch"):
        next(iterator)
