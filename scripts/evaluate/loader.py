"""Select dataset loaders and combine their common evaluation requests."""

from __future__ import annotations

import argparse  # noqa: TC003
import json
from itertools import islice

from eval_datasets import basic, mrcr, speedbench


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Configure dataset mapping and selection for both evaluation modes."""
    group = parser.add_argument_group("dataset loading")
    group.add_argument(
        "--dataset-config",
        help="Hugging Face dataset configuration, e.g. main for GSM8K; "
        "omit to use the default configuration (all configurations for speedbench)",
    )
    group.add_argument("--split", help="Source split (preset default, otherwise train)")
    group.add_argument(
        "--data-files", nargs="+", help="Source file paths or HF file patterns"
    )
    group.add_argument("--revision", help="Hugging Face dataset revision")
    columns = group.add_mutually_exclusive_group()
    columns.add_argument(
        "--prompt-column",
        help="Column containing plain text, wrapped as one user message "
        "(default: prompt); use --messages-column for an existing conversation",
    )
    columns.add_argument(
        "--messages-column",
        help="Column containing a message list (or JSON-encoded message list); "
        "preserves conversation roles and turns, e.g. MRCR",
    )
    columns.add_argument(
        "--prompt-template", help="Text template, e.g. {instruction}\\n{input}"
    )
    group.add_argument(
        "--body",
        type=json.loads,
        default='{"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}',
        help="Generation settings (default: temperature=0, enable_thinking=true); "
        "--body replaces this default",
    )
    group.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum source rows per dataset (default: 128)",
    )
    group.add_argument(
        "--needles",
        type=int,
        choices=(2, 4, 8),
        nargs="+",
        help="MRCR needle counts (preset default: 8)",
    )
    group.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=0,
        help="MRCR minimum rendered prompt length, inclusive",
    )
    group.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="MRCR maximum rendered prompt length "
        "(default: server context minus output budget)",
    )


def _load_datasets(args: argparse.Namespace) -> dict:
    """Select loaders and merge their named, lazy conversation iterators."""
    source_flags = (
        "dataset_config",
        "data_files",
        "split",
        "revision",
        "prompt_column",
        "messages_column",
        "prompt_template",
    )
    if (len(args.dataset) > 1 or "basic" in args.dataset) and any(
        getattr(args, flag, None) is not None for flag in source_flags
    ):
        raise ValueError(
            "Source-specific loading flags require a single dataset; "
            "run differently configured datasets separately"
        )
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if "speedbench" in args.dataset and any(
        getattr(args, flag, None) is not None
        for flag in (
            "split",
            "revision",
            "prompt_column",
            "messages_column",
            "prompt_template",
        )
    ):
        raise ValueError(
            "SPEED-Bench prepared inputs support --dataset-config and --data-files, "
            "not split/revision/column overrides"
        )
    has_mrcr = any(name in {"mrcr", "openai/mrcr"} for name in args.dataset)
    if not has_mrcr and (
        args.needles or args.min_prompt_tokens or args.max_prompt_tokens is not None
    ):
        raise ValueError("Needle and prompt-length selection flags require MRCR")
    datasets: dict = {}
    for source in args.dataset:
        match source:
            case "mrcr" | "openai/mrcr":
                selected = mrcr.load(args)
            case "speedbench":
                selected = speedbench.load(args)
            case _:
                selected = basic.load(source, args)
        for name, conversations in selected.items():
            datasets.setdefault(name, conversations)
    return datasets


def _requests(datasets: dict, args: argparse.Namespace):
    """Build API requests once, after dataset-specific prompt preparation."""
    body = {
        **args.body,
        "model": args.model,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.max_tokens is not None:
        body.pop("max_completion_tokens", None)
        body["max_tokens"] = args.max_tokens
    for name, conversations in datasets.items():
        for index, sample in enumerate(islice(conversations, args.max_samples)):
            yield {
                "id": sample.get("id", f"{name}:{index}"),
                "dataset": name,
                "category": sample["category"],
                "body": {**body, "messages": sample["messages"]},
            }


def load_evaluation_requests(args: argparse.Namespace) -> list[dict]:
    """Load and combine requests for either evaluation mode."""
    datasets = _load_datasets(args)
    results = list(_requests(datasets, args))
    if not results:
        raise ValueError("no requests in the selected datasets")
    return results
