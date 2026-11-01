# Plot saved sweeps

```bash
python scripts/evaluate/plot.py \
  --series 'output/dflash-off:DFlash2 (8 tokens, adaptive off)' \
           'output/dflash-on:DFlash2 (8 tokens, adaptive on)' \
           'output/mtp:Native MTP (3 tokens)' \
  --output output/comparison.png
```

One dataset per directory. Repeats are averaged at each concurrency. Image export requires Chrome/Chromium.
