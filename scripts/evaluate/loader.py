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
    group.add_argument(
        "--category",
        nargs="+",
        help="Category labels to select before --max-samples",
    )
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
        help="Generation settings (default: temperature=0, enable_thinking=true, "
        "max_completion_tokens=32768); --body replaces this default",
    )
    group.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum source rows per dataset (default: 128)",
    )


def _load_datasets(args: argparse.Namespace) -> dict:
    """Select loaders and merge their named, lazy conversation iterators."""
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
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True},
        "max_completion_tokens": 32768,
    }
    if args.body is not None:
        body = dict(args.body)
    body.update(model=args.model, stream=True, stream_options={"include_usage": True})
    for name, conversations in datasets.items():
        selected = (
            sample
            for sample in conversations
            if not args.category or sample["category"] in args.category
        )
        for index, sample in enumerate(islice(selected, args.max_samples)):
            yield {
                "id": f"{name}:{index}",
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
