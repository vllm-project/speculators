"""MRCR long-context prompt source for speculative-decoding acceptance benchmarking.

Loads OpenAI's ``openai/mrcr`` 2-needle dataset and buckets samples by
prompt+answer token count, using the same bin edges and ``o200k_base``
tiktoken encoding OpenAI uses for its own leaderboard. Each bucket becomes an
Inspect AI ``Task`` that is run against the target server to drive realistic
long-context requests.

Correctness is intentionally not graded: MRCR is used here purely as a source
of long, multi-turn prompts to see how spec-decode acceptance holds up as
context length grows, not to measure retrieval accuracy.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import tiktoken
from huggingface_hub import hf_hub_download

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate

from perf_utils import CsvWriter, acceptance_csv_columns, extract_spec_decode_metrics, print_acceptance_report

logger = logging.getLogger("evaluate")

MRCR_REPO = "openai/mrcr"
MRCR_ENCODING = "o200k_base"
MRCR_SHARDS = ("2needle/2needle_0.parquet", "2needle/2needle_1.parquet")

# OpenAI's own bin edges, in tokens: first bin is closed on both ends,
# remaining bins are left-open/right-closed.
MRCR_BIN_EDGES = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)

_ROLE_TO_MESSAGE = {
    "system": ChatMessageSystem,
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
}


def _bucket_label(n_tokens: int) -> str | None:
    lo0, hi0 = MRCR_BIN_EDGES[0], MRCR_BIN_EDGES[1]
    if lo0 <= n_tokens <= hi0:
        return f"{lo0}-{hi0}"
    for lo, hi in zip(MRCR_BIN_EDGES[1:], MRCR_BIN_EDGES[2:]):
        if lo < n_tokens <= hi:
            return f"{lo}-{hi}"
    return None


def _count_tokens(enc: tiktoken.Encoding, messages: list[dict], answer: str) -> int:
    total = sum(len(enc.encode(m["content"])) for m in messages)
    return total + len(enc.encode(answer))


def _iter_records():
    for shard in MRCR_SHARDS:
        path = hf_hub_download(MRCR_REPO, shard, repo_type="dataset")
        yield from pd.read_parquet(path).to_dict("records")


def load_mrcr_buckets(
    max_context: int,
    max_samples_per_bucket: int | None = None,
) -> dict[str, list[Sample]]:
    """Download the MRCR 2-needle dataset and group samples into context-length buckets.

    Buckets whose lower edge exceeds *max_context* are dropped entirely, since
    those prompts would exceed a server started with a smaller max context.
    """
    enc = tiktoken.get_encoding(MRCR_ENCODING)
    buckets: dict[str, list[Sample]] = {}

    for record in _iter_records():
        messages = json.loads(record["prompt"])
        n_tokens = _count_tokens(enc, messages, record["answer"])
        if n_tokens > max_context:
            continue
        label = _bucket_label(n_tokens)
        if label is None:
            continue

        bucket = buckets.setdefault(label, [])
        if max_samples_per_bucket is not None and len(bucket) >= max_samples_per_bucket:
            continue

        bucket.append(
            Sample(
                input=[_ROLE_TO_MESSAGE[m["role"]](content=m["content"]) for m in messages],
                target="",
                metadata={
                    "bucket": label,
                    "n_tokens": n_tokens,
                    "random_string_to_prepend": record["random_string_to_prepend"],
                },
            )
        )

    for label, samples in buckets.items():
        logger.info("  mrcr/2needle/%s: %d samples", label, len(samples))

    return dict(sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0])))


def build_task(samples: list[Sample]) -> Task:
    return Task(dataset=MemoryDataset(samples), solver=[generate()])


def real_token_count(root_url: str, model: str, sample: Sample) -> int:
    """Tokenize *sample* through the target server's own ``/tokenize`` endpoint.

    Unlike the o200k_base counts used for bucketing, this reflects the actual
    tokenizer and chat template the server will use, and (unlike the
    ``/v1/chat/completions/render`` endpoint) doesn't reject oversized prompts —
    it just reports how long they really are.
    """
    messages = [{"role": m.role, "content": m.content} for m in sample.input]
    body = json.dumps({"model": model, "messages": messages}).encode()
    req = Request(
        f"{root_url.rstrip('/')}/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=180) as resp:  # noqa: S310
        return json.loads(resp.read())["count"]


def warn_if_oversized(
    root_url: str,
    model: str,
    subset: str,
    samples: list[Sample],
    max_model_len: int,
) -> None:
    """Log a warning for any sample whose real tokenized length exceeds *max_model_len*.

    MRCR's own bucket labels are based on OpenAI's o200k_base tokenizer, which
    can undercount relative to the model actually being served (different
    vocab, plus chat-template overhead) — this checks against the real count.
    """
    oversized = []
    for sample in samples:
        try:
            n_tokens = real_token_count(root_url, model, sample)
        except (URLError, OSError, KeyError) as e:
            logger.warning("[%s] Could not tokenize sample for length check: %s", subset, e)
            continue
        if n_tokens > max_model_len:
            oversized.append(n_tokens)

    if oversized:
        logger.warning(
            "[%s] %d/%d samples exceed the server's max_model_len=%d once "
            "tokenized by the real model (largest: %d tokens, bucket label "
            "was computed with o200k_base and may undercount)",
            subset,
            len(oversized),
            len(samples),
            max_model_len,
            max(oversized),
        )


def run_mrcr(
    target: str,
    model_info: dict | None,
    metrics_url: str,
    output_dir: Path,
    max_concurrency: int,
    max_context: int,
    max_samples_per_bucket: int,
    require_metrics,
) -> None:
    """Run MRCR long-context evaluation.

    Drives MRCR prompts through the target server bucket-by-bucket, diffing
    spec-decode Prometheus counters around each bucket's run.
    """
    if model_info is None:
        logger.error("Could not determine served model info from %s/models", target)
        sys.exit(1)
    model_name = model_info["id"]
    max_model_len = model_info.get("max_model_len")
    logger.info(
        "Server reports model=%s max_model_len=%s", model_name, max_model_len
    )
    logger.info(
        "Loading MRCR 2-needle dataset (max_context=%d, max_samples_per_bucket=%s)...",
        max_context,
        max_samples_per_bucket,
    )
    buckets = load_mrcr_buckets(
        max_context=max_context,
        max_samples_per_bucket=max_samples_per_bucket,
    )
    if not buckets:
        logger.error(
            "No MRCR buckets available at or below --mrcr-max-context=%d",
            max_context,
        )
        sys.exit(1)

    root_url = target.rstrip("/").removesuffix("/v1")
    log_dir = output_dir / "artifacts" / "mrcr_logs"
    acceptance_csv: CsvWriter | None = None

    for label, samples in buckets.items():
        subset = f"mrcr/2needle/{label}"
        logger.info("[%s] Starting (%d samples)", subset, len(samples))

        if max_model_len is not None:
            warn_if_oversized(root_url, model_name, subset, samples, max_model_len)

        baseline = require_metrics(metrics_url)
        inspect_eval(
            build_task(samples),
            model=f"vllm/{model_name}",
            model_base_url=target,
            max_connections=max_concurrency,
            log_dir=str(log_dir / label.replace("-", "_")),
            display="none",
            fail_on_error=False,
        )
        current = require_metrics(metrics_url)

        spec = extract_spec_decode_metrics(current, baseline_metrics=baseline)
        if not spec or spec.get("num_drafts", 0) <= 0:
            logger.warning("[%s] No speculative decoding metrics found", subset)
            continue

        spec["subset"] = subset
        print_acceptance_report(spec)
        if acceptance_csv is None:
            acceptance_csv = CsvWriter(
                output_dir / "acceptance.csv",
                ["subset"] + acceptance_csv_columns(spec),
            )
        acceptance_csv.append(spec)
        logger.info("[%s] Complete", subset)

    if acceptance_csv is None:
        logger.error("No acceptance metrics collected from any MRCR bucket")
        sys.exit(1)
