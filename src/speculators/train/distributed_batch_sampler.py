"""
MIT License

Copyright (c) 2023 One

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted from https://github.com/imoneoi/multipack_sampler.
"""

# Standard
import warnings
from heapq import heapreplace
from typing import NamedTuple

import numpy as np

# Third Party
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Sampler


## Multipack Distributed Batch Sampler
class _Bin(NamedTuple):
    """Helper named tuple for `lpt_packed_batch`"""

    fill: int  # sum of items in _Bin
    rank: int  # device rank _Bin is associated with


def _lpt_packed_batch(
    lengths: np.ndarray, max_len: int, num_replicas: int, start_index: int, rank: int
) -> None | list:
    """
    Check if lengths can be distributed into `num_replicas` machines with at most
    `max_len` tokens per machine and return local rank's batch.

    Uses the LPT (Longest processing time first scheduling) algorithm
    Time: O(|lengths| log |lengths| + |lengths| log replicas)

    Returns:
    `None` if unable to find a valid packing. Otherwise, return the batch indices that
    correspond to `rank`.
    """

    # Greedily assign lengths (in decreasing order) to the least full rank until they
    # are all assigned or we run out of space.
    local_batch = []
    heap = [_Bin(0, i) for i in range(num_replicas)]

    # sort in descending order
    indices = np.argsort(lengths)[::-1]

    for idx, size in zip(indices, lengths[indices]):
        new_fill = heap[0].fill + size
        if new_fill > max_len:
            # Size doesn't fit in least full batch (or any others), report failure.
            return None

        if heap[0].rank == rank:
            # minimum bucket corresponds to the local rank -> add idx to local batch
            local_batch.append(start_index + idx)

        _ = heapreplace(heap, _Bin(new_fill, heap[0].rank))

    return local_batch


def _assign_to_packed_batches(
    lengths: np.ndarray, max_len: int, rank: int, replicas: int
) -> list[NDArray]:
    """Distribute lengths to batches across all ranks, while respecting max_length.
    Uses a binary search + LPT algorithm.

    Args:
        lengths (np.ndarray): array of dataset sample lengths
        max_len (int): maximum allowed sum of lengths in batch
        rank (int): local rank to collect batches for
        replicas (int): world size to distribute batches to

    Returns:
        tuple[list, int, int]:
            - list of np.arrays containing the indices for each batch on the local rank
            - sum of dataset lengths included (total sum of lengths in dataset minus any
              that were dropped at end of dataset)
            - total token capacity if each batch maxed out max_length
    """

    lengths_so_far = 0
    ind = 0
    result = []
    lengths_cumsum = np.cumsum(lengths)

    # binary search for max integer x such that the next x elements in shuffled lengths
    # array can be packed into `replicas` batches.
    # Add the local rank's batch to `result` and repeat until end of dataset
    while True:
        if len(lengths) - ind < replicas:
            # Not enough lengths left to pack into `num_replicas` batches
            # Break and drop whatever lengths we have left
            break

        # binary search in [1, 1 + upper bound for x)
        left = 1
        right = 1 + np.searchsorted(
            lengths_cumsum[ind:], lengths_so_far + max_len * replicas, "right"
        )

        batch = None
        while right - left > 1 and right > replicas:
            mid = (left + right) // 2
            batch = _lpt_packed_batch(
                lengths[ind : ind + mid], max_len, replicas, ind, rank
            )
            if batch is None:
                right = mid
            else:
                left = mid

        if batch is None:
            batch = _lpt_packed_batch(
                lengths[ind : ind + left], max_len, replicas, ind, rank
            )

        ind += left
        lengths_so_far = lengths_cumsum[ind - 1]

        # append only result for local rank (already filtered in lpt_packed_batch)
        result.append(batch)

    return result


class MultipackDistributedBatchSamplerV2(Sampler):
    def __init__(
        self,
        batch_max_length: int,
        lengths: ArrayLike,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        """Efficient distributed packing sampler for linear attention style models

        Args:
            batch_max_length (int): max number of tokens in a single batch per device
            lengths (ArrayLike[int]): the lengths of each sample in the dataset
            num_replicas (int): The number of replicas to split the dataset across.
            rank (int): The local rank to collect batches for.
            seed (int, optional): Seed for RNG, must be the same on all ranks. Default 0
        """
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.batch_max_length = batch_max_length
        self.lengths = np.array(lengths)

        self.valid_indices = np.nonzero(self.lengths <= self.batch_max_length)[0]
        if self.rank == 0 and len(self.valid_indices) < len(self.lengths):
            msg = (
                f"Dropping {len(self.lengths) - len(self.valid_indices)}"
                f"/{len(self.lengths)} samples longer than batch_max_length. "
                "Ensure that the right max_batch_length is used during data processing."
            )
            warnings.warn(msg, stacklevel=1)

        self._cached_generated_batches = (-1, [])

    def __iter__(self):
        batches = self._generate_batches(self.epoch)
        return iter(batches)

    def __len__(self):
        batches = self._generate_batches(self.epoch)
        return len(batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _generate_batches(self, epoch: int) -> list[NDArray]:
        """Generate batches for local rank

        Returns:
            list[NDArray]: list of np.arrays containing the indices for each batch on
            the local rank
        """
        if self._cached_generated_batches[0] == epoch:
            return self._cached_generated_batches[1]

        rng = np.random.default_rng(seed=self.seed + epoch)
        indices = rng.permutation(self.valid_indices)

        batches = _assign_to_packed_batches(
            self.lengths[indices], self.batch_max_length, self.rank, self.num_replicas
        )

        # The indices in batches are relative to the shuffled self.lengths[indices]
        # Translate them so that they are instead relative to the overall unshuffled
        # self.lengths array.
        batches = [indices[batch] for batch in batches]

        # Cache result
        self._cached_generated_batches = (epoch, batches)
        return batches
