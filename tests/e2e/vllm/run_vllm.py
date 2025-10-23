import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams  # type: ignore[import-not-found]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampling-params-args",
        type=str,
        required=True,
        help="JSON-serialized kwargs for SamplingParams instantiation",
    )
    parser.add_argument(
        "--llm-args",
        type=str,
        required=True,
        help="JSON-serialized kwargs for LLM instantiation",
    )
    parser.add_argument(
        "--prompts", type=str, required=True, help="JSON-serialized prompts"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="File to save the JSON-serialized results (outputs’ token IDs)",
    )
    return parser.parse_args()


def run_vllm(args: argparse.Namespace):
    sampling_params = SamplingParams(**json.loads(args.sampling_params_args))
    llm = LLM(**json.loads(args.llm_args))
    return llm.generate(json.loads(args.prompts), sampling_params)


if __name__ == "__main__":
    args = parse_args()
    outputs = run_vllm(args)

    # only token IDs (presence, count, type) were validated, so serialize/return those
    output_token_ids = []
    for output in outputs:
        output_token_ids.append(output.outputs[0].token_ids)

    with args.results_file.open("w", encoding="utf-8") as f:
        json.dump(output_token_ids, f)
