import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
from vllm.v1.metrics.reader import Counter, Metric, Vector


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


def extract_metrics(raw_metrics: list[Metric]) -> dict:
    metrics_dict = {}
    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens
    for metric in raw_metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    metrics_dict["total_num_output_tokens"] = total_num_output_tokens
    metrics_dict["num_drafts"] = num_drafts
    metrics_dict["num_draft_tokens"] = num_draft_tokens
    metrics_dict["num_accepted_tokens"] = num_accepted_tokens
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    metrics_dict["acceptance_length"] = acceptance_length
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        metrics_dict[f"acceptance_at_token_{i}"] = acceptance_rate

    return metrics_dict


def run_vllm(args: argparse.Namespace):
    sampling_params = SamplingParams(**json.loads(args.sampling_params_args))
    llm = LLM(**json.loads(args.llm_args), disable_log_stats=False)
    outputs = llm.generate(json.loads(args.prompts), sampling_params)
    metrics_dict = extract_metrics(llm.get_metrics())
    return outputs, metrics_dict


if __name__ == "__main__":
    args = parse_args()
    outputs, metrics_dict = run_vllm(args)

    # only token IDs (presence, count, type) were validated, so serialize/return those
    output_token_ids = []
    for output in outputs:
        output_token_ids.append(output.outputs[0].token_ids)

    results_dict = {
        "outputs": output_token_ids,
        "metrics": metrics_dict,
    }

    with args.results_file.open("w", encoding="utf-8") as f:
        json.dump(results_dict, f)
