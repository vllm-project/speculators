#!/usr/bin/env python3
"""Convert infer_demo-style CSV -> Qwen-VL-compatible multimodal JSONL.

Each CSV row becomes ONE dialog whose human turn uses a STRUCTURED content
list ([{"type":"image", ...}, {"type":"text", ...}]) instead of the legacy
``"<image>\n<prompt>"`` string.  This is required on both sides of the
DFlash data-gen pipeline:

    * ``speculators/preprocessing.py::_is_multimodal_batch`` only returns
      True when at least one turn's content is a list that contains a dict
      with ``type in {"image","video","audio"}``.  A plain string like
      ``"<image>\\n..."`` leaves ``messages_json`` empty and silently
      routes the sample through the TEXT-ONLY ``/v1/completions`` path
      (no sidecar, no vision placeholders) -> draft model is trained on
      a corrupted teacher signal.
    * ``qwen_vl_utils.fetch_image`` consumes the ``min_pixels`` /
      ``max_pixels`` keys on each media segment, so carrying them here
      guarantees byte-identical smart_resize behaviour with
      ``infer_demo.py`` (factor = 16*2 = 32, window = [256*256, 1024*1024]).

Output record shape::

    {
      "id": "<row_id>-<lang_idx>",
      "image": "/abs/path",                 # kept for BC with legacy tools
      "conversations": [
        {"from": "human", "value": [
            {"type": "image", "image": "/abs/path",
             "min_pixels": 65536, "max_pixels": 1048576},
            {"type": "text",  "text":  "<prompt>"}
        ]},
        {"from": "gpt",   "value": "<caption>"},
      ]
    }
"""
import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm


# Mirror infer_demo.py: 256**2 lower bound, 1024**2 upper bound.
DEFAULT_MIN_PIXELS = 256 * 256
DEFAULT_MAX_PIXELS = 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--lang", default="cn", choices=["cn", "en", "both"])
    ap.add_argument("--id-col", default="index")
    ap.add_argument("--img-col", default="img_path")
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    args = ap.parse_args()

    with Path(args.csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_f = Path(args.out_jsonl)
    out_f.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    with out_f.open("w", encoding="utf-8") as out:
        for row in tqdm(rows, desc="convert"):
            img_path = row.get(args.img_col, "").strip()
            if not img_path or not Path(img_path).exists():
                continue

            # Use absolute paths so downstream workers with a different CWD
            # (datasets.map spawns isolated processes) still resolve correctly.
            img_abs = str(Path(img_path).resolve())

            prompts = []
            if args.lang in ("cn", "both"):
                prompts.append(row["prompt_cn"])
            if args.lang in ("en", "both"):
                prompts.append(row["prompt_en"])

            for idx, prompt_text in enumerate(prompts):
                # Structured content list: required to trip the multimodal
                # branch in _preprocess_batch AND to carry per-sample
                # min/max-pixels through to qwen_vl_utils.fetch_image.
                user_content = [
                    {
                        "type": "image",
                        "image": img_abs,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                    },
                    {"type": "text", "text": prompt_text},
                ]
                record = {
                    "id": f"{row[args.id_col]}-{idx}",
                    # Top-level "image" retained for any legacy consumer that
                    # greps for it; preprocessing.py now reads from the
                    # structured content list instead.
                    "image": img_abs,
                    "conversations": [
                        {"from": "human", "value": user_content},
                        {
                            "from": "gpt",
                            "value": row.get("caption", row.get("hy_ocr_info", "")),
                        },
                    ],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_kept += 1

    print(f"wrote {n_kept} samples -> {out_f}")


if __name__ == "__main__":
    main()