"""Quantize a DSpark drafter with NVFP4 GPTQ and iMatrix calibration."""

import argparse
import random
from pathlib import Path
from types import MethodType

import torch
from dspark_calibration import calibration_batches, calibration_forward
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoTokenizer, PretrainedConfig

from speculators import SpeculatorModel
from speculators.models.dspark import DSparkDraftModel


class DSparkOneshot(Oneshot):
    def validate_model(self, model):
        # llmcompressor 0.15 cannot load a speculators config with AutoConfig.
        if not isinstance(model, DSparkDraftModel):
            raise TypeError("Only DSpark models are supported")
        raw, _ = PretrainedConfig.get_config_dict(model.config.name_or_path)
        if (
            raw.get("quantization_config") is not None
            or raw.get("transformer_layer_config", {}).get("quantization_config")
            is not None
            or getattr(model.config, "quantization_config", None) is not None
        ):
            raise ValueError("Use an unquantized DSpark checkpoint")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, help="DSpark Hub ID or local checkpoint")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--data-dir", type=Path, required=True)
    cli.add_argument("--hidden-states-dir", type=Path, required=True)
    cli.add_argument("--samples", type=int, default=1024)
    cli.add_argument("--sequence-length", type=int, default=2048)
    cli.add_argument("--max-anchors", type=int, default=32)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = SpeculatorModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).eval()
    verifier = model.config.speculators_config.verifier.name_or_path
    layers = model.target_layer_ids
    loader = calibration_batches(
        args.data_dir,
        args.hidden_states_dir,
        count=args.samples,
        max_length=args.sequence_length,
        max_anchors=args.max_anchors,
        layer_ids=layers,
        width=model.fc.in_features // len(layers),
        block_size=model.block_size,
    )
    print(f"Calibrating on prepared rows 0 through {args.samples - 1}", flush=True)
    model.forward = MethodType(calibration_forward, model)
    recipe = GPTQModifier(
        targets="Linear",
        scheme="NVFP4",
        weight_observer="imatrix_mse",
        ignore=[
            "Embedding",
            r"re:.*(^|\.)embed_tokens$",
            "lm_head",
            "verifier_lm_head",
            r"re:.*(^|\.)markov_head\..*",
            r"re:.*(^|\.)confidence_head\..*",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(verifier)
    DSparkOneshot(
        model=model,
        processor=tokenizer,
        dataset=loader,
        recipe=recipe,
        pipeline="basic",
        output_dir=str(args.output_dir),
    )()


if __name__ == "__main__":
    main()
