import argparse
import importlib
import json
import os
import types
from pathlib import Path

# Set environment variables before importing vLLM
os.environ["VLLM_USE_DEEP_GEMM"] = "0"


def _workaround_vllm_torch210() -> None:
    """Patch torch 2.10.0 / vLLM nightly incompatibility before loading vLLM.

    In torch 2.10.0, ``torch._inductor`` exposes ``standalone_compile`` as a
    wrapper function that shadows the submodule of the same name.  vLLM nightly
    patches ``torch._inductor.standalone_compile.FakeTensorMode`` via
    ``unittest.mock.patch``, but ``mock`` resolves the dotted path through
    ``getattr``, landing on the wrapper function rather than the module.
    Functions have no ``FakeTensorMode`` attribute, so ``mock`` raises
    ``AttributeError`` and vLLM fails to start.

    Fix: replace the attribute with a callable subclass of the actual submodule
    so that ``mock.patch`` patches ``FakeTensorMode`` in the correct module
    namespace (which the submodule reads at runtime via ``LOAD_GLOBAL``) while
    callers that invoke ``standalone_compile(graph, ...)`` still reach the
    original wrapper via ``__call__``.

    Upstream issue: https://github.com/pytorch/pytorch/issues/176562
    """
    try:
        import torch._inductor as _ti  # noqa: PLC0415
    except ImportError:
        return

    sc_fn = getattr(_ti, "standalone_compile", None)
    if not callable(sc_fn) or hasattr(sc_fn, "FakeTensorMode"):
        return  # not affected or already patched

    try:
        sc_mod = importlib.import_module("torch._inductor.standalone_compile")
    except ImportError:
        return

    if not hasattr(sc_mod, "FakeTensorMode"):
        return  # unexpected module layout — leave untouched

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return sc_fn(*args, **kwargs)  # type: ignore[misc]

    sc_mod.__class__ = _CallableModule
    _ti.standalone_compile = sc_mod  # type: ignore[assignment]


# Apply the workaround before importing vLLM: the fix must be in place before
# vLLM's compilation backend resolves torch._inductor.standalone_compile.
_workaround_vllm_torch210()

from vllm import LLM, SamplingParams  # type: ignore[import-not-found]  # noqa: E402
from vllm.v1.metrics.reader import Counter, Metric, Vector  # noqa: E402


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


def extract_metrics(raw_metrics: list[Metric], total_num_output_tokens: int) -> dict:
    metrics_dict: dict[str, int | float] = {}
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts: list[float] = []
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
            if len(acceptance_counts) < len(metric.values):
                acceptance_counts = acceptance_counts + [0.0] * (
                    len(metric.values) - len(acceptance_counts)
                )
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
    outputs = llm.chat(json.loads(args.prompts), sampling_params)
    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    metrics_dict = extract_metrics(llm.get_metrics(), total_num_output_tokens)
    return outputs, metrics_dict


if __name__ == "__main__":
    args = parse_args()
    outputs, metrics_dict = run_vllm(args)

    # only token IDs (presence, count, type) were validated, so serialize/return those
    output_token_ids = [output.outputs[0].token_ids for output in outputs]

    results_dict = {
        "outputs": output_token_ids,
        "metrics": metrics_dict,
    }

    with args.results_file.open("w", encoding="utf-8") as f:
        json.dump(results_dict, f)
