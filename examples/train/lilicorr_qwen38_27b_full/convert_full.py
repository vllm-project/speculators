#!/usr/bin/env python
"""Stage 1: convert the full Qwen3.8-27B regenerated perfectblend jsonl to
speculators training arrow shards.

Same passthrough pipeline as the 5k prep (right-truncate input_ids/loss_mask
to MAX_LEN, drop zero-supervision rows after truncation, mask-id + vocab
bounds checks), parallelized over byte ranges of the *uncompressed* jsonl the
same way the 4B full run was:
  - worker i seeks to start = size*i//n, skips the first partial line (owned
    by worker i-1), reads lines while their start offset <= end
  - output: pyarrow IPC shard + per-worker stats json

Columns: input_ids (int64 list), loss_mask (int64 list), seq_len (int64).
Stage 2 (write_final_full.py) hand-writes the save_to_disk layout since the
~4.4B-token table exceeds the 2.15B-element combine limit.
"""

import json
import os
import sys

import pyarrow as pa

SRC = "/tmp/lilicorr_check/qwen38-perfectblend/raw/plain/train.jsonl"
SHARD_DIR = "/tmp/lilicorr_check/qwen38-perfectblend/full_shards"
STATS_DIR = "/tmp/lilicorr_check/qwen38-perfectblend/full_stats"
MAX_LEN = 8192
MASK_ID = 248077  # auto-resolved mask token; must not appear in data
VOCAB = 248320

LEN_BUCKETS = [512, 1024, 2048, 4096, 6144, 8192, 1 << 62]


def worker(i: int, n: int) -> None:
    size = os.path.getsize(SRC)
    start = size * i // n
    end = size * (i + 1) // n

    stats = {
        "rows": 0,
        "kept": 0,
        "dropped_unsupervised": 0,
        "dropped_mismatch": 0,
        "parse_errors": 0,
        "rows_truncated": 0,
        "orig_tokens": 0,
        "kept_tokens": 0,
        "sup_tokens": 0,
        "mask_id_hits": 0,
        "max_id": 0,
        "min_id": 1 << 40,
        "len_hist": [0] * len(LEN_BUCKETS),
    }

    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int64())),
            ("loss_mask", pa.list_(pa.int64())),
            ("seq_len", pa.int64()),
        ]
    )
    out_path = f"{SHARD_DIR}/shard_{i:03d}.arrow"
    ids_buf: list[list[int]] = []
    mask_buf: list[list[int]] = []
    sl_buf: list[int] = []

    def flush():
        if not ids_buf:
            return
        batch = pa.record_batch(
            [
                pa.array(ids_buf, type=pa.list_(pa.int64())),
                pa.array(mask_buf, type=pa.list_(pa.int64())),
                pa.array(sl_buf, type=pa.int64()),
            ],
            schema=schema,
        )
        writer.write_batch(batch)
        ids_buf.clear()
        mask_buf.clear()
        sl_buf.clear()

    with open(SRC, "rb") as f, pa.ipc.new_file(out_path, schema) as writer:
        f.seek(start)
        if i > 0:
            f.readline()  # skip the partial line owned by the previous worker
        while True:
            pos = f.tell()
            if pos > end:
                break
            line = f.readline()
            if not line:
                break
            stats["rows"] += 1
            try:
                d = json.loads(line)
            except Exception:
                stats["parse_errors"] += 1
                continue
            ids = d["input_ids"]
            mask = d["loss_mask"]
            if len(ids) != len(mask):
                stats["dropped_mismatch"] += 1
                continue
            stats["orig_tokens"] += len(ids)
            for b, edge in enumerate(LEN_BUCKETS):
                if len(ids) <= edge:
                    stats["len_hist"][b] += 1
                    break
            if len(ids) > MAX_LEN:
                stats["rows_truncated"] += 1
                ids = ids[:MAX_LEN]
                mask = mask[:MAX_LEN]
            sup = sum(mask)
            if sup == 0:
                stats["dropped_unsupervised"] += 1
                continue
            if MASK_ID in ids:
                stats["mask_id_hits"] += 1
            row_max = max(ids)
            row_min = min(ids)
            stats["max_id"] = max(stats["max_id"], row_max)
            stats["min_id"] = min(stats["min_id"], row_min)
            stats["kept"] += 1
            stats["kept_tokens"] += len(ids)
            stats["sup_tokens"] += sup
            ids_buf.append(ids)
            mask_buf.append(mask)
            sl_buf.append(len(ids))
            if len(ids_buf) >= 4096:
                flush()
        flush()

    with open(f"{STATS_DIR}/stats_{i:03d}.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    os.makedirs(SHARD_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    i, n = int(sys.argv[1]), int(sys.argv[2])
    worker(i, n)
