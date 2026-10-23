#!/usr/bin/env python
"""Stage-1 driver: run all convert_full.py workers in parallel, then
aggregate their stats and enforce the same safety assertions as the 5k prep
(mask-id collisions, vocab bounds, expected row count)."""

import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_full import worker

N = 48
EXPECTED_RECORDS = 1_884_616
MASK_ID = 248077
VOCAB = 248320

if __name__ == "__main__":
    os.makedirs("/tmp/lilicorr_check/qwen38-perfectblend/full_shards", exist_ok=True)
    os.makedirs("/tmp/lilicorr_check/qwen38-perfectblend/full_stats", exist_ok=True)
    with mp.Pool(N) as pool:
        pool.starmap(worker, [(i, N) for i in range(N)])

    tot = {}
    for i in range(N):
        with open(
            f"/tmp/lilicorr_check/qwen38-perfectblend/full_stats/stats_{i:03d}.json"
        ) as f:
            s = json.load(f)
        for k, v in s.items():
            if k == "len_hist":
                tot[k] = [a + b for a, b in zip(tot.get(k, [0] * len(v)), v)]
            elif k == "min_id":
                tot[k] = min(tot.get(k, 1 << 40), v)
            elif k == "max_id":
                tot[k] = max(tot.get(k, 0), v)
            else:
                tot[k] = tot.get(k, 0) + v

    print(json.dumps({k: v for k, v in tot.items() if k != "len_hist"}, indent=1))
    print(
        "len_hist:",
        dict(zip([512, 1024, 2048, 4096, 6144, 8192, "inf"], tot["len_hist"])),
    )

    assert tot["rows"] == EXPECTED_RECORDS, (
        f"scanned {tot['rows']} != {EXPECTED_RECORDS}"
    )
    assert tot["mask_id_hits"] == 0, "MASK_ID collides with data!"
    assert tot["max_id"] < VOCAB, f"max_id {tot['max_id']} >= vocab {VOCAB}"
    assert tot["kept"] > 1_500_000, f"only {tot['kept']} rows kept"
    print(
        f"STAGE1 OK: {tot['kept']:,} rows, {tot['kept_tokens']:,} tokens, "
        f"{100 * tot['sup_tokens'] / tot['kept_tokens']:.1f}% supervised"
    )
