"""Quantize a DSpark drafter with dynamic FP8 weights and activations."""

from _dspark_common import IGNORE, checkpoint_path, parser
from llmcompressor import model_free_ptq


def main():
    args = parser(__doc__).parse_args()
    checkpoint = checkpoint_path(args)
    # Replace FP8_DYNAMIC with FP8_BLOCK for blockwise FP8.
    model_free_ptq(
        model_stub=checkpoint,
        save_directory=str(args.output_dir),
        scheme="FP8_DYNAMIC",
        ignore=IGNORE,
    )


if __name__ == "__main__":
    main()
