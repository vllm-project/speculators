"""Host-to-device prefetch of the next training batch."""

import pytest
import torch

from speculators.train.trainer import _H2DPrefetcher


class _CountingLoader:
    """Records how many batches have been pulled out of it."""

    def __init__(self, num_batches: int):
        self.num_batches = num_batches
        self.produced = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            self.produced += 1
            yield {"input_ids": torch.full((2,), i), "error_records": i}


def test_yields_every_batch_in_order():
    prefetcher = _H2DPrefetcher(_CountingLoader(4), torch.device("cpu"))

    seen = [p.wait()["input_ids"][0].item() for p in prefetcher]

    assert seen == [0, 1, 2, 3]


def test_keeps_non_tensor_entries_and_the_host_batch():
    """``error_records`` is a plain int, and the loop still reads the CPU batch.

    The host batch also has to stay referenced for the lifetime of the copy:
    it is pinned memory the DataLoader would otherwise be free to reuse.
    """
    (prefetched,) = list(_H2DPrefetcher(_CountingLoader(1), torch.device("cpu")))

    assert prefetched.cpu["error_records"] == 0
    assert prefetched.wait()["error_records"] == 0
    assert prefetched.cpu["input_ids"].device.type == "cpu"


def test_len_passes_through():
    assert len(_H2DPrefetcher(_CountingLoader(7), torch.device("cpu"))) == 7


def test_stop_prefetch_does_not_pull_another_batch():
    """``max_steps`` ends the loop, so the next batch must not be fetched.

    Without this the iterator would pull one more batch out of the loader and
    copy it to the device for a step that never runs.
    """
    loader = _CountingLoader(8)
    prefetcher = _H2DPrefetcher(loader, torch.device("cpu"))

    consumed = 0
    for _ in prefetcher:
        consumed += 1
        if consumed == 3:
            prefetcher.stop_prefetch()

    assert consumed == 3
    assert loader.produced == 3


def test_a_second_epoch_prefetches_again():
    loader = _CountingLoader(3)
    prefetcher = _H2DPrefetcher(loader, torch.device("cpu"))
    for _ in prefetcher:
        prefetcher.stop_prefetch()

    assert sum(1 for _ in prefetcher) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_copies_land_on_the_device_through_the_side_stream():
    """The values must be correct even though the copy ran on another stream.

    ``wait`` is what makes that true: the compute stream has to be told to wait
    on the copy's event before it reads the tensors.
    """
    prefetcher = _H2DPrefetcher(_CountingLoader(3), torch.device("cuda", 0))

    for expected, prefetched in enumerate(prefetcher):
        assert prefetched.event is not None
        gpu = prefetched.wait()
        assert gpu["input_ids"].device.type == "cuda"
        assert torch.equal(gpu["input_ids"].cpu(), torch.full((2,), expected))
