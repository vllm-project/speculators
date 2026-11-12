"""Quantize a DSpark drafter with NVFP4 GPTQ and iMatrix calibration."""

import random
from pathlib import Path
from types import MethodType

import torch
from _dspark_calibration import calibration_batches, calibration_forward, capture_layers
from _dspark_common import IGNORE, DSparkOneshot, load_model, parser
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoTokenizer


def main():
    cli = parser(__doc__)
    cli.add_argument("--data-dir", type=Path, required=True)
    cli.add_argument("--hidden-states-dir", type=Path, required=True)
    cli.add_argument(
        "--extraction-provenance",
        type=Path,
        required=True,
        help="vllm_command.txt for this cache",
    )
    cli.add_argument("--samples", type=int, default=1024)
    cli.add_argument("--sequence-length", type=int, default=2048)
    cli.add_argument("--max-anchors", type=int, default=32)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = load_model(args)
    verifier = model.config.speculators_config.verifier.name_or_path
    layers = model.target_layer_ids
    captured = capture_layers(args.extraction_provenance, verifier)
    loader = calibration_batches(
        args.data_dir,
        args.hidden_states_dir,
        count=args.samples,
        max_length=args.sequence_length,
        max_anchors=args.max_anchors,
        layer_ids=layers,
        captured_layer_ids=captured,
        width=model.fc.in_features // len(layers),
        block_size=model.block_size,
    )
    print(f"Calibrating on prepared rows 0 through {args.samples - 1}", flush=True)
    model.forward = MethodType(calibration_forward, model)
    recipe = GPTQModifier(
        targets="Linear", scheme="NVFP4", weight_observer="imatrix_mse", ignore=IGNORE
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
