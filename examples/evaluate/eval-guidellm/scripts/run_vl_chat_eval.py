#!/usr/bin/env python3
"""Run a simple multimodal chat benchmark against vLLM OpenAI endpoint.

This script is intentionally lightweight and avoids external dependencies.
It is designed to be invoked by `run_guidellm.sh` when a dataset file is
detected as multimodal.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


EXIT_NOT_MULTIMODAL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VL chat evaluation")
    parser.add_argument("--target", type=str, required=True, help="Target base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--dataset-file", type=str, required=True, help="Local JSON/JSONL dataset file")
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Optional image root directory used to resolve relative image paths",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Per-request output JSONL path")
    parser.add_argument("--summary-file", type=str, required=True, help="Summary JSON output path")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _normalize_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("data", "items", "samples", "questions", "annotations"):
                value = obj.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
            return [obj]

    raise ValueError(f"Unsupported dataset format: {path}")


def _extract_text_from_content(content: Any) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)
    return None


def _extract_ref_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        ref = value.strip()
        return ref or None

    if not isinstance(value, dict):
        return None

    for key in ("image", "image_url", "url", "path", "file", "image_path", "image_file"):
        nested = value.get(key)
        if key == "image_url" and isinstance(nested, dict):
            nested = nested.get("url")
        ref = _extract_ref_from_value(nested)
        if ref:
            return ref
    return None


def _collect_image_refs_from_content(content: Any) -> list[str]:
    refs: list[str] = []
    if not isinstance(content, list):
        return refs

    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).lower()
        candidates: list[Any] = []
        if item_type == "image":
            candidates.append(item.get("image"))
        elif item_type == "image_url":
            candidates.append(item.get("image_url"))

        for key in ("image", "image_url", "url", "path", "file", "image_path"):
            candidates.append(item.get(key))

        for candidate in candidates:
            ref = _extract_ref_from_value(candidate)
            if ref:
                refs.append(ref)

    return refs


def _extract_prompt(record: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "instruction", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            if role != "user":
                continue
            text = _extract_text_from_content(msg.get("content"))
            if text:
                return text

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", ""))).lower()
            if role not in {"human", "user"}:
                continue
            value = turn.get("value", turn.get("content"))
            text = _extract_text_from_content(value)
            if text:
                return text

    return ""


def _extract_image_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []

    for key in ("image", "img", "image_path", "image_file", "image_name", "image_url"):
        ref = _extract_ref_from_value(record.get(key))
        if ref:
            refs.append(ref)

    for key in ("images", "image_paths", "image_files"):
        value = record.get(key)
        if isinstance(value, list):
            for one in value:
                ref = _extract_ref_from_value(one)
                if ref:
                    refs.append(ref)

    mm_data = record.get("multi_modal_data")
    if isinstance(mm_data, dict):
        images = mm_data.get("image")
        if isinstance(images, list):
            for image in images:
                ref = _extract_ref_from_value(image)
                if ref:
                    refs.append(ref)
        else:
            ref = _extract_ref_from_value(images)
            if ref:
                refs.append(ref)

    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                refs.extend(_collect_image_refs_from_content(msg.get("content")))

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content", turn.get("value"))
            refs.extend(_collect_image_refs_from_content(content))

    dedup: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        dedup.append(ref)
    return dedup


def _existing_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


def _resolve_local_image_path(
    image_ref: str,
    dataset_dir: Path,
    image_root: Path | None,
) -> Path | None:
    raw_ref = image_ref.strip()
    if not raw_ref:
        return None

    path = Path(raw_ref).expanduser()
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        if image_root is not None:
            candidates.append(image_root / path)
        dataset_dir = dataset_dir.resolve()
        candidates.append(dataset_dir / path)
        for parent in dataset_dir.parents:
            candidates.append(parent / path)
        candidates.append(Path.cwd() / path)

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if _existing_file(resolved):
            return resolved
    return None


def _to_image_url(image_ref: str, dataset_dir: Path, image_root: Path | None) -> str | None:
    ref = image_ref.strip()
    if not ref:
        return None

    if ref.startswith(("http://", "https://", "data:")):
        return ref

    if ref.startswith("file://"):
        parsed = urlparse(ref)
        local_path = Path(unquote(parsed.path))
        if _existing_file(local_path):
            return ref
        return None

    parsed = urlparse(ref)
    if parsed.scheme and len(parsed.scheme) > 1:
        return None

    local_path = _resolve_local_image_path(ref, dataset_dir, image_root)
    if local_path is None:
        return None

    return f"file://{quote(str(local_path), safe='/:._-')}"


def _is_multimodal_record(record: dict[str, Any]) -> bool:
    return len(_extract_image_refs(record)) > 0


def _sample_id(record: dict[str, Any], idx: int) -> str:
    for key in ("id", "sample_id", "uid", "question_id", "qid"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return f"sample_{idx}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def main() -> int:
    args = parse_args()
    target = _normalize_url(args.target)
    dataset_file = Path(args.dataset_file)
    dataset_dir = dataset_file.parent
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
    output_file = Path(args.output_file)
    summary_file = Path(args.summary_file)

    records = _read_json_or_jsonl(dataset_file)
    if not records:
        raise ValueError(f"Dataset has no records: {dataset_file}")

    multimodal_records = [r for r in records if _is_multimodal_record(r)]
    if not multimodal_records:
        print(f"[INFO] No multimodal records found in {dataset_file}", flush=True)
        return EXIT_NOT_MULTIMODAL

    model_resp = _request_json("GET", f"{target}/models", payload=None, timeout=args.timeout)
    model_items = model_resp.get("data") if isinstance(model_resp, dict) else None
    model_id = None
    if isinstance(model_items, list):
        for item in model_items:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_id = item["id"]
                break
    if not model_id:
        raise ValueError(f"Failed to resolve model id from {target}/models")

    print(
        f"[INFO] Running VL eval: records={len(multimodal_records)} model={model_id}",
        flush=True,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    latency_ms_values: list[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    success = 0
    failed = 0
    skipped_invalid_image_ref = 0
    invalid_image_ref_total = 0
    started = time.perf_counter()

    with output_file.open("w", encoding="utf-8") as fout:
        for idx, record in enumerate(multimodal_records):
            sample_id = _sample_id(record, idx)
            prompt = _extract_prompt(record)
            image_refs = _extract_image_refs(record)
            image_urls: list[str] = []
            invalid_image_refs: list[str] = []
            for ref in image_refs:
                url = _to_image_url(ref, dataset_dir, image_root)
                if url is None:
                    invalid_image_refs.append(ref)
                    continue
                image_urls.append(url)

            invalid_image_ref_total += len(invalid_image_refs)

            if not image_urls:
                failed += 1
                skipped_invalid_image_ref += 1
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "status": "error",
                            "error": "no_resolvable_image_ref",
                            "skip_reason": "invalid_image_ref",
                            "invalid_image_refs": invalid_image_refs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            content: list[dict[str, Any]] = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})
            if not content:
                failed += 1
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "status": "error",
                            "error": "empty multimodal content",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            payload: dict[str, Any] = {
                "model": model_id,
                "messages": [{"role": "user", "content": content}],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "stream": False,
            }

            try:
                t0 = time.perf_counter()
                response = _request_json(
                    "POST",
                    f"{target}/chat/completions",
                    payload=payload,
                    timeout=args.timeout,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                usage = response.get("usage") if isinstance(response, dict) else {}
                if not isinstance(usage, dict):
                    usage = {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                total_tokens = int(
                    usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
                )

                success += 1
                latency_ms_values.append(latency_ms)
                prompt_tokens_total += prompt_tokens
                completion_tokens_total += completion_tokens
                total_tokens_total += total_tokens

                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "status": "ok",
                            "latency_ms": round(latency_ms, 3),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "num_images": len(image_urls),
                            "invalid_image_refs": invalid_image_refs,
                            "response_id": response.get("id"),
                            "finish_reason": (
                                (((response.get("choices") or [None])[0] or {}).get("finish_reason"))
                                if isinstance(response, dict)
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "status": "error",
                            "error": str(exc),
                            "num_images": len(image_urls),
                            "invalid_image_refs": invalid_image_refs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if (idx + 1) % 50 == 0:
                print(
                    f"[INFO] progress={idx + 1}/{len(multimodal_records)} "
                    f"success={success} failed={failed}",
                    flush=True,
                )

    elapsed_s = max(time.perf_counter() - started, 1e-9)
    summary = {
        "dataset_file": str(dataset_file),
        "image_root": str(image_root) if image_root is not None else None,
        "target": target,
        "model": model_id,
        "num_records": len(multimodal_records),
        "success": success,
        "failed": failed,
        "success_rate": (success / len(multimodal_records)) if multimodal_records else 0.0,
        "data_quality": {
            "invalid_image_ref_total": invalid_image_ref_total,
            "skipped_invalid_image_ref": skipped_invalid_image_ref,
        },
        "latency_ms": {
            "avg": statistics.fmean(latency_ms_values) if latency_ms_values else 0.0,
            "p50": _percentile(latency_ms_values, 0.50),
            "p95": _percentile(latency_ms_values, 0.95),
            "max": max(latency_ms_values) if latency_ms_values else 0.0,
        },
        "tokens": {
            "prompt_total": prompt_tokens_total,
            "completion_total": completion_tokens_total,
            "all_total": total_tokens_total,
            "all_tokens_per_sec": total_tokens_total / elapsed_s,
            "completion_tokens_per_sec": completion_tokens_total / elapsed_s,
        },
        "throughput": {
            "elapsed_s": elapsed_s,
            "requests_per_sec": success / elapsed_s,
        },
    }

    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[INFO] VL eval complete: success={success} failed={failed} "
        f"output={output_file} summary={summary_file}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
