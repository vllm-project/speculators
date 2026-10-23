#!/usr/bin/env python
"""Stage 2: write the final full-dataset speculators save_to_disk layout by hand.

datasets.Dataset can't be constructed over a >2.1B-element list<int64> table
(fingerprint pickling + int32 offset overflow), so this writes the
save_to_disk layout directly:
    data-XXXXX-of-XXXXX.arrow   (ArrowWriter stream format)
    dataset_info.json           (DatasetInfo.write_to_directory)
    state.json                  (fixed fingerprint, torch format)

Row order = fixed random permutation (seed 0 applied twice, mirroring
prepare-data's raw+combined shuffles; any fixed permutation is equivalent
since the packer reshuffles per epoch).

Usage: write_final_full.py <out_dir>
"""

import glob
import json
import os
import random
import sys

import datasets
import pyarrow as pa
from datasets.arrow_writer import ArrowWriter

SHARD_GLOB = "/tmp/lilicorr_check/qwen38-perfectblend/full_shards/shard_*.arrow"
OUT = sys.argv[1]
ROWS_PER_SHARD = 26_160  # ~67 shards, ~64M tokens / ~1GB each

features = datasets.Features(
    {
        "input_ids": datasets.List(datasets.Value("int64")),
        "loss_mask": datasets.List(datasets.Value("int64")),
        "seq_len": datasets.Value("int64"),
    }
)
print("features:", features)

tables = []
total = 0
for fp in sorted(glob.glob(SHARD_GLOB)):
    with pa.memory_map(fp, "rb") as src:
        t = pa.ipc.open_file(src).read_all()
    tables.append(t)
    total += t.num_rows
print(f"source rows: {total:,}")

rng = random.Random(0)
perm = list(range(total))
rng.shuffle(perm)
rng.shuffle(perm)

# global row -> (table_idx, row_idx)
counts = [t.num_rows for t in tables]
prefix = [0]
for c in counts:
    prefix.append(prefix[-1] + c)

os.makedirs(OUT, exist_ok=True)
num_shards = (total + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
written_rows = 0
total_bytes = 0
for s in range(num_shards):
    lo = s * ROWS_PER_SHARD
    hi = min(total, lo + ROWS_PER_SHARD)
    ids_buf, mask_buf, sl_buf = [], [], []
    for g in perm[lo:hi]:
        ti = 0
        # binary search the owning table
        a, b = 0, len(tables) - 1
        while a < b:
            mid = (a + b + 1) // 2
            if prefix[mid] <= g:
                a = mid
            else:
                b = mid - 1
        ti = a
        ri = g - prefix[ti]
        row = tables[ti].slice(ri, 1).to_pylist()[0]
        ids_buf.append(row["input_ids"])
        mask_buf.append(row["loss_mask"])
        sl_buf.append(row["seq_len"])
    shard_path = os.path.join(OUT, f"data-{s:05d}-of-{num_shards:05d}.arrow")
    writer = ArrowWriter(features=features, path=shard_path)
    writer.write_batch({"input_ids": ids_buf, "loss_mask": mask_buf, "seq_len": sl_buf})
    num_examples, num_bytes = writer.finalize()
    written_rows += num_examples
    total_bytes += num_bytes
    print(
        f"shard {s + 1}/{num_shards}: rows={num_examples:,} "
        f"bytes={num_bytes:,} ({num_bytes / 1e9:.2f} GB)",
        flush=True,
    )

info = datasets.DatasetInfo(features=features)
info.write_to_directory(OUT)

state = {
    "_data_files": [
        {"filename": f"data-{s:05d}-of-{num_shards:05d}.arrow"}
        for s in range(num_shards)
    ],
    "_fingerprint": "a1b2c3d4e5f60718",
    "_format_columns": ["input_ids", "loss_mask", "seq_len"],
    "_format_kwargs": {},
    "_format_type": "torch",
    "_output_all_columns": False,
    "_split": None,
}
with open(os.path.join(OUT, "state.json"), "w") as f:
    json.dump(state, f, indent=2)

print(f"DONE: {written_rows:,} rows, {total_bytes / 1e9:.1f} GB -> {OUT}")

# round-trip verification
back = datasets.load_from_disk(OUT)
print("reload:", f"{len(back):,}", "rows, format:", back.format)
assert len(back) == written_rows
row = back[0]
ids, mask, sl = row["input_ids"], row["loss_mask"], int(row["seq_len"])
print(
    "row0:",
    tuple(ids.shape),
    tuple(mask.shape),
    sl,
    "sup:",
    int(mask.sum()),
)
assert ids.shape == mask.shape == (sl,)
lens = back.with_format(None)["seq_len"]
print(
    f"seq_len ok: n={len(lens):,} min={min(lens)} max={max(lens)} "
    f"total_tokens={sum(lens):,}"
)
