"""Select MRCR by needles and server-rendered length without truncating history."""

import argparse
import json
import sys

import httpx

from .basic import dataset_rows


def load(args: argparse.Namespace) -> dict:
    """Filter before the sample cap, reserving generation space in the context."""
    if args.needles and args.data_files:
        raise ValueError("Choose --needles or --data-files for MRCR, not both")
    if args.dataset_config:
        raise ValueError("MRCR uses --needles or --data-files, not --dataset-config")
    if (
        args.prompt_column
        or args.prompt_template
        or args.messages_column not in (None, "prompt")
    ):
        raise ValueError("MRCR requires the full conversation in the prompt column")
    target = args.target.rstrip("/").removesuffix("/v1")
    response = httpx.get(target + "/v1/models", timeout=30, trust_env=False)
    response.raise_for_status()
    model = next((m for m in response.json()["data"] if m["id"] == args.model), None)
    if model is None or not model.get("max_model_len"):
        raise ValueError(
            "MRCR requires the served model's max_model_len from /v1/models"
        )
    output = args.max_tokens
    if output is None:
        output = args.body.get("max_tokens", args.body.get("max_completion_tokens", 0))
    budget = model["max_model_len"] - output
    upper = args.max_prompt_tokens if args.max_prompt_tokens is not None else budget
    if not 0 <= args.min_prompt_tokens <= upper <= budget:
        raise ValueError(
            "MRCR prompt-length bounds must fit server context minus the output budget"
        )
    if not hasattr(args, "dataset_selection"):
        args.dataset_selection = {}

    if args.data_files:
        return {
            "mrcr": _conversations(args, "mrcr", args.data_files, model, upper, output)
        }
    return {
        f"mrcr/{n}needle": _conversations(
            args, f"mrcr/{n}needle", [f"{n}needle/*.parquet"], model, upper, output
        )
        for n in dict.fromkeys(args.needles or [8])
    }


def _conversations(args, name, files, model, upper, output):
    stats = {
        "scanned": 0,
        "filtered_by_length": 0,
        "selected": 0,
        "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": upper,
        "server_context": model["max_model_len"],
        "output_budget": output,
    }
    args.dataset_selection[name] = stats
    rows = dataset_rows(
        "openai/mrcr",
        split=args.split or "train",
        data_files=files,
        revision=args.revision,
    )
    with httpx.Client(timeout=120, trust_env=False) as client:
        for index, row in enumerate(rows):
            stats["scanned"] += 1
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            payload = {
                "model": args.model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": args.body.get("chat_template_kwargs", {}),
            }
            if "tools" in args.body:
                payload["tools"] = args.body["tools"]
            response = client.post(
                args.target.rstrip("/").removesuffix("/v1") + "/tokenize", json=payload
            )
            if response.is_error:
                raise ValueError(
                    f"MRCR /tokenize HTTP {response.status_code}: "
                    f"{response.text[:8192]}"
                )
            count = response.json()["count"]
            if not args.min_prompt_tokens <= count <= upper:
                stats["filtered_by_length"] += 1
                continue
            stats["selected"] += 1
            yield {
                "id": f"{name}:{index}",
                "messages": messages,
                "category": row.get("category"),
            }
    print(
        f"{name}: selected {stats['selected']}; "
        f"filtered {stats['filtered_by_length']} by rendered prompt length",
        file=sys.stderr,
    )
