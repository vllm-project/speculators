import contextlib
import os

import torch

from speculators.data_generation.logging_utils import PipelineLogger

log = PipelineLogger(__name__)


# Based on vLLM's util with the same name
@contextlib.contextmanager
def set_default_torch_num_threads(num_threads: int | None = None):
    """
    Sets the default number of threads for PyTorch to the given value.

    `None` means using the value of the environment variable `OMP_NUM_THREADS`
    (or `1` if that is not available).
    """
    if num_threads is None:
        num_threads = 1

        try:
            num_threads = int(os.environ["OMP_NUM_THREADS"])
        except KeyError:
            log.debug(
                f"OMP_NUM_THREADS is not set; "
                f"defaulting Torch threads to {num_threads}.",
            )
        except ValueError:
            log.warning(
                f"OMP_NUM_THREADS is invalid; "
                f"defaulting Torch threads to {num_threads}.",
            )

    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    try:
        yield
    finally:
        torch.set_num_threads(old_num_threads)
