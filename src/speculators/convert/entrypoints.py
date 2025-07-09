"""
Unified CLI interface for checkpoint conversion.
"""

import os
from pathlib import Path
from typing import Literal, Optional, Union

from speculators.convert.converters import SpeculatorConverter
from speculators.model import SpeculatorModel
from speculators.utils import check_download_model_checkpoint

__all__ = ["convert_model"]


def convert_model(
    model: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]] = None,
    config: Optional[Union[str, os.PathLike]] = None,
    verifier: Optional[Union[str, os.PathLike]] = None,
    verifier_attachment_mode: Literal["detached", "full", "train_only"] = "detached",
    validate_device: Optional[Union[str, int]] = None,
    algorithm: Literal["auto", "eagle", "eagle2", "hass"] = "auto",
    algorithm_kwargs: Optional[dict] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
) -> SpeculatorModel:
    model = check_download_model_checkpoint(
        model,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
    )
    if not config:
        config = model / "config.json"
    if not algorithm_kwargs:
        algorithm_kwargs = {}

    ConverterClass = SpeculatorConverter.resolve_converter(  # noqa: N806
        algorithm,
        model=model,
        config=config,
        verifier=verifier,
        **algorithm_kwargs,
    )
    converter = ConverterClass(
        model=model,
        config=config,
        verifier=verifier,
        **algorithm_kwargs,
    )

    return converter(
        output_path=output_path,
        validate_device=validate_device,
        verifier_attachment_mode=verifier_attachment_mode,
    )
