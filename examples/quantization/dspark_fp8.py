"""Quantize a DSpark drafter with dynamic FP8 weights and activations."""

from _dspark_common import IGNORE, DSparkOneshot, load_model, parser
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    args = parser(__doc__).parse_args()
    model = load_model(args)
    # Replace FP8_DYNAMIC with FP8_BLOCK for blockwise FP8.
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    DSparkOneshot(model=model, recipe=recipe, output_dir=str(args.output_dir))()


if __name__ == "__main__":
    main()
