"""Select dataset loaders and combine their common evaluation requests."""

from __future__ import annotations

import argparse  # noqa: TC003
import json
from itertools import islice

from eval_datasets import basic, speedbench


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


def _load_datasets(args: argparse.Namespace) -> dict:
    """Select loaders and merge their named, lazy conversation iterators."""
    has_mrcr = any(name in {"mrcr", "openai/mrcr"} for name in args.dataset)
    if has_mrcr and (
        args.max_tokens is not None
        or "max_tokens" in args.body
        or "max_completion_tokens" in args.body
    ):
        raise ValueError(
            "MRCR does not allow --max-tokens or output-token limits in --body"
        )
    datasets: dict = {}
    for source in args.dataset:
        match source:
            case "mrcr" | "openai/mrcr":
                selected = basic.load("mrcr", args)
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
    for name, conversations in datasets.items():
        dataset_body = dict(body)
        if name != "mrcr":
            dataset_body.pop("max_completion_tokens", None)
            dataset_body["max_tokens"] = (
                args.max_tokens if args.max_tokens is not None else 8192
            )
        for index, sample in enumerate(islice(conversations, args.max_samples)):
            yield {
                "id": f"{name}:{index}",
                "dataset": name,
                "category": sample["category"],
                "body": {**dataset_body, "messages": sample["messages"]},
            }


def load_evaluation_requests(args: argparse.Namespace) -> list[dict]:
    """Load and combine requests for either evaluation mode."""
    datasets = _load_datasets(args)
    results = list(_requests(datasets, args))
    if not results:
        raise ValueError("no requests in the selected datasets")
    return results
