from pathlib import Path
from typing import Union

from torch.nn import Module


def load_model(source: Union[str, Path, Module]) -> Module:  # noqa: ARG001
    """
    Load a PyTorch / Transformers model from the specified source.

    :param source: The source from which to load the model.
        This can be a path to a local model directory, a Hugging Face model id/name,
        or a custom model class (no op in this case).
    :return: The loaded model.
    """
    raise NotImplementedError(
        "Loading models from a path or Hugging Face Hub is not yet implemented."
    )
