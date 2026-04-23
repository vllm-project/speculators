#!/usr/bin/env python3
"""Convert a custombench video CSV into a speculators-compatible JSONL.

Output schema (one JSON object per line):
    {
      "conversations": [
        {"role": "user", "content": [
            {"type": "video", "video": "<abs local mp4 path>",
             "fps": 4, "max_frames": 36, "max_pixels": 409600},
            {"type": "text", "text": "<PROMPT>"}
        ]},
    {"role": "assistant", "content": [{"type": "text", "text": "<caption>"}]}
      ]
    }

The *assistant* turn is mandatory for DFlash training: it provides the
ground-truth tokens that the draft model learns to reproduce. We take it
from a column produced by `infer_demo.py` (default: ``abo_zh_caption``).

Note on assistant.content shape: both the user and assistant turns use the
list-of-segments form ``[{"type": "text", "text": ...}]`` because the
Qwen3-Omni processor-level chat template expects every turn's ``content`` to
be iterable of segment dicts. Writing the assistant turn as a bare string
triggers ``string indices must be integers, not 'str'`` inside
``processor.apply_chat_template`` during preprocessing.

Usage:
    python build_custombench_jsonl.py \
        --csv /path/to/custombench_with_captions.csv \
        --dataset-path /path/to/videos/ \
        --out /path/to/custombench_train.jsonl \
        --caption-col abo_zh_caption \
        --max-samples 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

# Prompts kept identical to infer_demo.py so the assistant JSON columns
# (abo_zh_caption / abo_en_caption) align with the correct system prompt.
PROMPT_JSON_CHN_MORETAGS = """你是一位资深的AI视频分析专家。你的任务是分析视频，并以一个单一、有效的JSON对象格式返回你的分析结果。所有输出，包括JSON内的所有文本值，都必须使用简体中文。严格按照以下JSON结构输出，不要包含任何额外的文字。{"short_caption": "","medium_caption": "","long_caption": "","long_long_caption": "","background": "","shot_type": "","shot_angle": "","composition": "","light": "","style": "","color_palette": "","atmosphere": "","camera_movement": "","subjects": "","actions": ""}"""  # noqa: E501, RUF001

PROMPT_JSON_EN_MORETAGS = """You are a senior AI video analysis expert. Your task is to analyze the video and return your analysis results in a single, valid JSON object format. All outputs, including all text values within the JSON, must be in English. Strictly follow the JSON structure below, without any additional text.{"short_caption": "","medium_caption": "","long_caption": "","long_long_caption": "","background": "","shot_type": "","shot_angle": "","composition": "","light": "","style": "","color_palette": "","atmosphere": "","camera_movement": "","subjects": "","actions": ""}"""  # noqa: E501

LANGUAGE_PROMPTS = {
    "zh": PROMPT_JSON_CHN_MORETAGS,
    "en": PROMPT_JSON_EN_MORETAGS,
}


def _resolve_video_path(dataset_path: Path, url_cell: str) -> Path:
    """Mirror infer_demo.py's path-resolution: filename portion of the URL."""
    return dataset_path / Path(urlparse(url_cell).path).name


def _build_conversation(
    video_abs_path: str,
    prompt_text: str,
    assistant_text: str,
    fps: int,
    max_frames: int,
    max_pixels: int,
) -> dict:
    return {
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_abs_path,
                        "fps": fps,
                        "max_frames": max_frames,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url-col", default="cos_signed_url")
    parser.add_argument("--id-col", default="videoid")
    parser.add_argument(
        "--caption-col",
        default="abo_zh_caption",
        help="CSV column storing the target model's caption JSON string.",
    )
    parser.add_argument(
        "--language",
        choices=("zh", "en"),
        default="zh",
        help="Selects the user prompt (must match the caption-col language).",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=36)
    parser.add_argument("--max-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--require-video-exists",
        action="store_true",
        help="Skip rows whose resolved video path does not exist on disk.",
    )
    return parser.parse_args()


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()

    prompt_text = LANGUAGE_PROMPTS[args.language]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_caption = 0
    skipped_missing_video = 0

    with args.csv.open(encoding="utf-8-sig") as fin, args.out.open(
        "w", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if args.caption_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"CSV is missing required column '{args.caption_col}'. "
                f"Run infer_demo.py first to materialize target captions, or "
                f"pass --caption-col to match your column."
            )

        for row in reader:
            if args.max_samples is not None and written >= args.max_samples:
                break

            assistant_text = (row.get(args.caption_col) or "").strip()
            if not assistant_text:
                skipped_no_caption += 1
                continue

            video_path = _resolve_video_path(args.dataset_path, row[args.url_col])
            if args.require_video_exists and not video_path.is_file():
                skipped_missing_video += 1
                continue

            sample = _build_conversation(
                video_abs_path=str(video_path),
                prompt_text=prompt_text,
                assistant_text=assistant_text,
                fps=args.fps,
                max_frames=args.max_frames,
                max_pixels=args.max_pixels,
            )
            fout.write(json.dumps(sample, ensure_ascii=False))
            fout.write("\n")
            written += 1

    print(
        f"[build_custombench_jsonl] wrote {written} samples to {args.out} "
        f"(skipped: no_caption={skipped_no_caption}, "
        f"missing_video={skipped_missing_video})"
    )
    if written == 0:
        raise SystemExit("No samples written; aborting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
