"""Basic presets and text/conversation mapping for HF datasets and local files."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET_PRESETS: dict[str, dict] = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "split": "test",
        "prompt_column": "question",
    },
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "prompt_column": "problem",
    },
    "humaneval": {
        "dataset": "openai/openai_humaneval",
        "split": "test",
        "prompt_column": "prompt",
    },
    "mbpp": {
        "dataset": "google-research-datasets/mbpp",
        "dataset_config": "full",
        "split": "test",
        "prompt_column": "text",
    },
    "aime26": {
        "dataset": "math-ai/aime26",
        "split": "test",
        "prompt_column": "problem",
    },
    "mtbench": {
        "dataset": "HuggingFaceH4/mt_bench_prompts",
        "split": "train",
        "prompt_template": "{prompt[0]}",
    },
    "arena-hard": {
        "dataset": "lmarena-ai/arena-hard-auto",
        "split": "train",
        "prompt_column": "prompt",
        "data_files": ["data/arena-hard-v2.0/question.jsonl"],
    },
    "mrcr": {
        "dataset": "openai/mrcr",
        "split": "train",
        "messages_column": "prompt",
    },
}


BASIC_DATASETS = (
    "gsm8k",
    "math500",
    "humaneval",
    "mbpp",
    "aime26",
    "mtbench",
    "arena-hard",
)


def dataset_rows(
    source: str, *, dataset_config=None, split="train", data_files=None, revision=None
):
    """Stream source rows without constructing evaluation requests."""
    local = Path(source)
    if local.is_file():
        file_format = local.suffix.lower().lstrip(".")
        return load_dataset(
            "json" if file_format == "jsonl" else file_format,
            data_files=str(local),
            split="train",
            streaming=True,
        )
    return load_dataset(
        source,
        name=dataset_config,
        split=split,
        data_files=data_files,
        revision=revision,
        streaming=True,
    )


def _column(row: dict, name: str):
    """Allow both literal column names and dotted paths into nested records."""
    if name in row:
        return row[name]
    for part in name.split("."):
        row = row[part]
    return row


def _conversations(name: str, args: argparse.Namespace):
    options = {"dataset": name, "split": "train", **DATASET_PRESETS.get(name, {})}
    for key in ("dataset_config", "split", "data_files"):
        if (value := getattr(args, key)) is not None:
            options[key] = value
    if any(
        getattr(args, key) is not None
        for key in ("messages_column", "prompt_column", "prompt_template")
    ):
        for key in ("messages_column", "prompt_column", "prompt_template"):
            options[key] = getattr(args, key)
    rows = dataset_rows(
        options["dataset"],
        dataset_config=options.get("dataset_config"),
        split=options["split"],
        data_files=options.get("data_files"),
        revision=args.revision,
    )
    messages_column = options.get("messages_column")
    prompt_column = options.get("prompt_column") or "prompt"
    prompt_template = options.get("prompt_template")
    for row in rows:
        if messages_column:
            messages = _column(row, messages_column)
            messages = json.loads(messages) if isinstance(messages, str) else messages
        else:
            prompt = (
                prompt_template.format_map(row)
                if prompt_template is not None
                else _column(row, prompt_column)
            )
            messages = [{"role": "user", "content": prompt}]
        yield {"messages": messages, "category": row.get("category")}


def load(name: str, args: argparse.Namespace) -> dict:
    """Return named conversation iterators; expand the full basic blend here."""
    names = BASIC_DATASETS if name == "basic" else (name,)
    return {dataset: _conversations(dataset, args) for dataset in names}
