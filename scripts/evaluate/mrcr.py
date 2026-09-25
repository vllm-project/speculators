"""MRCR data preparation for the evaluation path.

Renders every MRCR conversation through the target server's chat template
using the root ``/tokenize`` and ``/detokenize`` endpoints, then rebuckets
the rows by rendered prompt token count. GuideLLM consumes each selected
bucket as a single-turn ``/v1/completions`` workload: the fixed assistant
messages are part of the rendered prompt, so only the final answer is
generated.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger("evaluate")

__all__ = ["BUCKETS", "MRCR_DATASET", "parse_buckets", "parse_needles", "prepare_mrcr"]

MRCR_DATASET = "openai/mrcr"
_RENDER_WORKERS = 24
_HTTP_TIMEOUT = 600

# (label, low, high] over rendered prompt token counts
BUCKETS = (
    ("4096-8192", 4095, 8192),  # smallest bucket inclusive
    ("8193-16384", 8192, 16384),
    ("16385-32768", 16384, 32768),
    ("32769-65536", 32768, 65536),
    ("65537-131072", 65536, 131072),
    ("131073-262144", 131072, 262144),
    ("262145-524288", 262144, 524288),
    ("524289-1048576", 524288, 1048576),
)


def parse_needles(spec: str) -> list[int]:
    """Parse a needle selection: a comma list of ``2``, ``4``, ``8``."""
    try:
        values = [int(v) for v in spec.strip().split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid needle selection: {spec!r}"
        ) from None
    for value in values:
        if value not in (2, 4, 8):
            raise argparse.ArgumentTypeError(
                f"needle count must be 2, 4, or 8: {value}"
            )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"duplicate needle count: {spec!r}")
    return values


def parse_buckets(spec: str) -> list[tuple[str, int, int]]:
    """Parse a bucket selection: a comma list of numbers ``1``-``8`` or labels."""
    by_label = {bucket[0]: bucket for bucket in BUCKETS}
    selected: list[tuple[str, int, int]] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if part.isdigit():
            index = int(part)
            if not 1 <= index <= len(BUCKETS):
                raise argparse.ArgumentTypeError(
                    f"bucket number must be 1-{len(BUCKETS)}: {part}"
                )
            bucket = BUCKETS[index - 1]
        elif part in by_label:
            bucket = by_label[part]
        else:
            raise argparse.ArgumentTypeError(f"unknown bucket: {part!r}")
        if bucket in selected:
            raise argparse.ArgumentTypeError(
                f"bucket selected more than once: {part}"
            )
        selected.append(bucket)
    if not selected:
        raise argparse.ArgumentTypeError("empty bucket selection")
    return selected


def _root_url(target: str) -> str:
    return target.rstrip("/").removesuffix("/v1")


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
        return json.load(response)


def _model_info(target: str) -> tuple[str, int | None]:
    """Return the (model id, max_model_len) served at *target*."""
    with urllib.request.urlopen(f"{_root_url(target)}/v1/models", timeout=30) as resp:
        model = json.load(resp)["data"][0]
    return model["id"], model.get("max_model_len")


def _render_row(root: str, model: str, messages: list[dict]) -> tuple[str, int]:
    """Render one conversation to its exact server-side prompt string."""
    tokens = _post_json(f"{root}/tokenize", {"model": model, "messages": messages})
    detokenized = _post_json(
        f"{root}/detokenize", {"model": model, "tokens": tokens["tokens"]}
    )
    return detokenized["prompt"], tokens["count"]


def _ensure_rendered(
    target: str, model: str, n_needles: int, cache_dir: Path
) -> Path:
    """Render every row of one needle count once and cache it as JSONL."""
    path = cache_dir / f"rendered_{n_needles}needle.jsonl"
    if path.exists():
        logger.info("[mrcr] using cached %s", path)
        return path

    frames = [
        pd.read_parquet(
            hf_hub_download(
                MRCR_DATASET,
                f"{n_needles}needle/{n_needles}needle_{i}.parquet",
                repo_type="dataset",
            )
        )
        for i in (0, 1)
    ]
    prompts = pd.concat(frames, ignore_index=True)["prompt"].tolist()
    logger.info("[mrcr] rendering %d %d-needle rows", len(prompts), n_needles)

    root = _root_url(target)

    def render(item: tuple[int, str]) -> dict | None:
        index, prompt_json = item
        try:
            prompt, count = _render_row(root, model, json.loads(prompt_json))
        except Exception as error:  # noqa: BLE001
            logger.warning("[mrcr] render failed for row %d: %s", index, error)
            return None
        return {"prompt": prompt, "rendered_tokens": count}

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".jsonl.tmp")
    failed = 0
    written = 0
    with ThreadPoolExecutor(max_workers=_RENDER_WORKERS) as pool, tmp_path.open(
        "w"
    ) as file:
        for row in pool.map(render, enumerate(prompts)):
            if row is None:
                failed += 1
                continue
            file.write(json.dumps(row) + "\n")
            written += 1
            if written % 100 == 0:
                logger.info("[mrcr] rendered %d/%d rows", written, len(prompts))
    tmp_path.rename(path)
    if failed:
        logger.warning("[mrcr] %d/%d rows failed to render", failed, len(prompts))
    return path


def prepare_mrcr(
    target: str,
    needles: list[int],
    buckets: list[tuple[str, int, int]],
    data_dir: Path,
    artifacts_dir: Path,
    max_samples: int | None = None,
) -> list[tuple[str, Path]]:
    """Return ``(label, jsonl_path)`` pairs for each selected needle x bucket.

    Buckets with no rows (for example, buckets above the model context
    window) are reported and skipped.
    """
    model, max_model_len = _model_info(target)
    cache_dir = Path(data_dir) / model.replace("/", "_")
    out_dir = Path(artifacts_dir) / "mrcr"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, Path]] = []
    for n_needles in needles:
        rendered = _ensure_rendered(target, model, n_needles, cache_dir)

        # Stream rows to their bucket files in one pass; buckets are disjoint
        # and rows arrive in dataset order, so each file keeps the first rows
        # of its bucket without holding the dataset in memory.
        counts = {label: 0 for label, _, _ in buckets}
        open_files: dict[str, object] = {}
        with rendered.open() as file, contextlib.ExitStack() as stack:
            for line in file:
                row = json.loads(line)
                tokens = row["rendered_tokens"]
                for label, low, high in buckets:
                    if not low < tokens <= high:
                        continue
                    if max_model_len and tokens > max_model_len:
                        break
                    if max_samples is not None and counts[label] >= max_samples:
                        break
                    if label not in open_files:
                        open_files[label] = stack.enter_context(
                            (out_dir / f"{n_needles}needle_{label}.jsonl").open("w")
                        )
                    open_files[label].write(json.dumps({"prompt": row["prompt"]}) + "\n")
                    counts[label] += 1
                    break

        for label, _, _ in buckets:
            count = counts[label]
            if not count:
                logger.warning(
                    "[mrcr] %dneedle/%s: no rows (model context: %s)",
                    n_needles,
                    label,
                    max_model_len,
                )
                continue
            bucket_path = out_dir / f"{n_needles}needle_{label}.jsonl"
            logger.info(
                "[mrcr] %dneedle/%s: %d rows -> %s",
                n_needles,
                label,
                count,
                bucket_path,
            )
            pairs.append((f"mrcr/{n_needles}needle/{label}", bucket_path))
    return pairs
