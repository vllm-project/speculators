"""Quantize a DSpark drafter with NVFP4 GPTQ and iMatrix calibration."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from types import MethodType

import torch
from compressed_tensors.quantization import preset_name_to_scheme
from dspark_calibration import calibration_batches, calibration_forward
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoTokenizer, PretrainedConfig

from speculators import SpeculatorModel
from speculators.models.dspark import DSparkDraftModel

IGNORE = (
    r"re:.*(^|\.)embed_tokens$",
    r"re:.*(^|\.)lm_head$",
    r"re:.*(^|\.)verifier_lm_head$",
    r"re:.*(^|\.)markov_head\..*",
    r"re:.*(^|\.)confidence_head\..*",
)


class DSparkOneshot(Oneshot):
    """Run one-shot quantization with DSpark-specific validation."""

    def validate_model(self, model: object) -> None:
        """Reject non-DSpark models and checkpoints that are already quantized.

        Args:
            model: Model instance supplied by llmcompressor.
        """
        # llmcompressor 0.15 cannot load a speculators config with AutoConfig.
        if not isinstance(model, DSparkDraftModel):
            raise TypeError("Only DSpark models are supported")
        raw, _ = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path=model.config.name_or_path
        )
        if (
            raw.get("quantization_config") is not None
            or raw.get("transformer_layer_config", {}).get("quantization_config")
            is not None
            or getattr(model.config, "quantization_config", None) is not None
        ):
            raise ValueError("Use an unquantized DSpark checkpoint")


def _nvfp4_recipe() -> GPTQModifier:
    """Build the strict-iMatrix NVFP4 GPTQ recipe.

    Returns:
        Configured GPTQ modifier with DSpark exclusions.
    """
    scheme = preset_name_to_scheme(name="NVFP4", targets=["Linear"])
    scheme.weights.observer = "imatrix_mse"
    scheme.weights.observer_kwargs = {"strict": True}
    return GPTQModifier(config_groups={"group_0": scheme}, ignore=IGNORE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence; defaults to the process arguments.

    Returns:
        Parsed model, calibration, and quantization options.
    """
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, help="DSpark Hub ID or local checkpoint")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--data-dir", type=Path, required=True)
    cli.add_argument("--hidden-states-dir", type=Path, required=True)
    cli.add_argument("--samples", type=int, default=1024)
    cli.add_argument("--sequence-length", type=int, default=2048)
    cli.add_argument("--max-anchors", type=int, default=32)
    cli.add_argument("--seed", type=int, default=42)
    return cli.parse_args(args=argv)


def quantize(args: argparse.Namespace) -> None:
    """Quantize a DSpark checkpoint with strict-iMatrix NVFP4 GPTQ.

    Args:
        args: Parsed command-line arguments.
    """
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed_all(seed=args.seed)
    model = SpeculatorModel.from_pretrained(
        pretrained_model_name_or_path=args.model, torch_dtype=torch.bfloat16
    ).eval()
    if not isinstance(model, DSparkDraftModel):
        raise TypeError("Only DSpark models are supported")
    verifier = model.config.speculators_config.verifier.name_or_path
    if verifier is None:
        raise ValueError("The DSpark checkpoint must identify its verifier")
    layers = model.target_layer_ids
    loader = calibration_batches(
        data_dir=args.data_dir,
        hidden_states_dir=args.hidden_states_dir,
        count=args.samples,
        max_length=args.sequence_length,
        max_anchors=args.max_anchors,
        layer_ids=layers,
        width=model.fc.in_features // len(layers),
        block_size=model.block_size,
    )
    print(f"Calibrating on prepared rows 0 through {args.samples - 1}", flush=True)
    model.forward = MethodType(calibration_forward, model)
    recipe = _nvfp4_recipe()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=verifier)
    DSparkOneshot(
        model=model,
        processor=tokenizer,
        dataset=loader,
        recipe=recipe,
        pipeline="basic",
        output_dir=str(args.output_dir),
    )()


def main() -> None:
    """Run the DSpark NVFP4 quantization command-line interface."""
    quantize(args=parse_args())


if __name__ == "__main__":
    main()
