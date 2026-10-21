# throughput_interactivity.py

Plots timed GuideLLM benchmark JSON files (or legacy vLLM batch files separately) produced by `evaluate.py speedup`. Collection and dataset loading live in the two-mode evaluator. See [Evaluating Model Performance](../user_guide/tutorials/evaluating_performance.md) for collection options and the [graph tutorial](../user_guide/tutorials/throughput_interactivity.md) for interpreting curves. Plotting requires `matplotlib`.

```bash
speculators evaluate speedup \
  --dataset humaneval --target http://localhost:8000 --model my-model \
  --max-tokens 1024 --streams 1,2,4,8,16,32,64,128 \
  --output-dir runs/dspark_humaneval --label dspark

python scripts/evaluate/throughput_interactivity.py plot \
  --series runs/baseline_humaneval:Baseline \
  --series runs/dspark_humaneval:DSpark \
  --out chart.png
```

- **`--series`** (`DIRECTORY[:NAME[:#COLOR]]`, repeatable, required): one curve per result directory. Reads `conc*_r*.json`; the default legend name is the evaluator's saved label. Later series draw over earlier ones.
- **`--out`**: output image path.
- **`--title`**, **`--subtitle`**, **`--note`**: chart text. Put hardware, versions, and workload in the subtitle.

Points are labelled `N=<concurrency>`. Y is GuideLLM's native total output-token throughput over the measurement window. X is `1 / mean TPOT (seconds)` in tokens/s/user; raw stream-chunk ITL is not used. Repeats at the same concurrency are averaged after calculating X for each run. Runs with failed requests are rejected. Collect each dataset separately for one chart per dataset.

The CSV input, `validate` command, and alternative `--x little` axis belonged to the old window collector and have been removed.

Collection defaults to `--warmup-seconds 30 --max-seconds 100 --repeats 3`. This is a time budget per point, not a repeated-request quota. Timed and legacy batch measurements cannot be mixed on one chart.
