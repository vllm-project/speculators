#!/usr/bin/env python
"""Scan stage-1 shards for rows with out-of-vocab token ids.

Uses flattened values + numpy reduceat over list offsets (vectorized, no
per-row python). Reports per-shard counts, global row ids, and samples of
the offending rows for diagnosis.
"""

import glob

import numpy as np
import pyarrow as pa

VOCAB = 248320
SHARDS = sorted(
    glob.glob("/tmp/lilicorr_check/qwen38-perfectblend/full_shards/shard_*.arrow")
)

total_rows = 0
bad_global = []
samples = []
for path in SHARDS:
    with pa.memory_map(path, "rb") as src:
        t = pa.ipc.open_file(src).read_all()
    col = t.column("input_ids")
    arr = col.combine_chunks()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.chunk(0)
    offsets = np.asarray(arr.offsets, dtype=np.int64)
    values = np.asarray(arr.values, dtype=np.int64)
    row_max = np.maximum.reduceat(values, offsets[:-1])
    row_min = np.minimum.reduceat(values, offsets[:-1])
    bad_local = np.nonzero((row_max >= VOCAB) | (row_min < 0))[0]
    n = t.num_rows
    for li in bad_local:
        g = total_rows + int(li)
        bad_global.append(g)
        if len(samples) < 8:
            lo, hi = offsets[li], offsets[li + 1]
            vals = values[lo:hi]
            bad_pos = np.nonzero((vals >= VOCAB) | (vals < 0))[0]
            samples.append(
                {
                    "global_row": g,
                    "shard": path.split("/")[-1],
                    "seq_len": int(hi - lo),
                    "n_bad_tokens": int(len(bad_pos)),
                    "bad_values": [int(v) for v in vals[bad_pos[:5]]],
                    "first_bad_pos": int(bad_pos[0]),
                    "context": [
                        int(v) for v in vals[max(0, bad_pos[0] - 4) : bad_pos[0] + 5]
                    ],
                }
            )
    total_rows += n

print(f"scanned {total_rows:,} rows across {len(SHARDS)} shards")
print(f"BAD ROWS: {len(bad_global)}")
if bad_global:
    shards_hit = {}
    for g in bad_global:
        s = g // 36244  # approximate rows per shard, refined below if needed
        shards_hit[s] = shards_hit.get(s, 0) + 1
    print("bad global rows:", bad_global[:50])
    import json

    print(json.dumps(samples, indent=1))
