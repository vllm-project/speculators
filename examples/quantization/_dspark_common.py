"""Shared checkpoint handling and DSpark support for calibrated quantization."""

import argparse
import hashlib
import inspect
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.transformers.compression.mtp import has_mtp_in_checkpoint
from transformers import PretrainedConfig

from speculators import SpeculatorModel
from speculators.models.dspark import DSparkDraftModel

IGNORE = [
    r"re:.*(^|\.)embed_tokens$",
    "lm_head",
    "verifier_lm_head",
    r"re:.*(^|\.)markov_head\..*",
    r"re:.*(^|\.)confidence_head\..*",
]


def check_unquantized(config):
    # Fail closed even for compressed-tensors input; these examples start with BF16.
    if isinstance(config, dict):
        if config.get("quantization_config") is not None:
            raise ValueError("Use an unquantized DSpark checkpoint for these examples")
        if any(
            config.get(name, 0)
            for name in (
                "num_mtp_layers",
                "mtp_num_hidden_layers",
                "num_nextn_predict_layers",
            )
        ):
            raise ValueError("MTP checkpoints are not supported by these examples")
        for value in config.values():
            check_unquantized(value)
    elif isinstance(config, list):
        for value in config:
            check_unquantized(value)


class DSparkOneshot(Oneshot):
    """Strict DSpark-only validation until compressor supports speculator configs.

    This overrides a version-specific method. It intentionally rejects MTP and all
    quantized inputs instead of duplicating compressor's MTP/quantization handling.
    """

    def validate_model(self, model):
        validator = inspect.getsource(Oneshot.validate_model).encode()
        expected = "1ad821d8941818e7468087f65e3a0953d23332cac43b551f6766b6a839b3ccbb"
        if hashlib.sha256(validator).hexdigest() != expected:
            raise RuntimeError(
                "Unknown llm-compressor validator; review the pinned integration"
            )
        if not isinstance(model, DSparkDraftModel):
            raise TypeError("Only DSpark models are supported")
        if hasattr(model, "mtp") or has_mtp_in_checkpoint(model):
            raise ValueError("MTP checkpoints are not supported by this example")
        raw, _ = PretrainedConfig.get_config_dict(
            model.config.name_or_path, local_files_only=True
        )
        check_unquantized(raw)
        check_unquantized(
            {"quantization_config": getattr(model.config, "quantization_config", None)}
        )


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--model", required=True, help="DSpark Hub ID or local checkpoint"
    )
    result.add_argument(
        "--revision",
        default="87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce",
        help="Immutable Hub commit",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def checkpoint_path(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    local = Path(args.model)
    checkpoint = (
        str(local.resolve())
        if local.is_dir()
        else snapshot_download(args.model, revision=args.revision)
    )
    if args.output_dir.resolve().is_relative_to(Path(checkpoint).resolve()):
        raise ValueError("Output must be outside the input checkpoint")
    raw, _ = PretrainedConfig.get_config_dict(checkpoint, local_files_only=True)
    check_unquantized(raw)
    if raw.get("speculators_model_type") != "dspark":
        raise TypeError("Only DSpark models are supported")
    return checkpoint


def load_model(args):
    checkpoint = checkpoint_path(args)
    model = SpeculatorModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    if not isinstance(model, DSparkDraftModel):
        raise TypeError("Only DSpark models are supported")
    model.eval()
    return model
