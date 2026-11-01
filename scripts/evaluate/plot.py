#!/usr/bin/env python3
"""Compare saved AIPerf sweeps with mean curves and no confidence decorations."""
# mypy: disable-error-code=import-untyped

import argparse
from pathlib import Path

import pandas as pd
from aiperf.plot.constants import PlotTheme
from aiperf.plot.core.data_loader import DataLoader
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.dashboard.utils import runs_to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="+", required=True, metavar="DIRECTORY:LABEL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    x, y = "output_token_throughput_per_user", "output_token_throughput"
    frames = []
    for value in args.series:
        directory, label = value.rsplit(":", 1)
        runs = [
            DataLoader().load_run(path.parent, load_per_request_data=False)
            for path in sorted(Path(directory).rglob("profile_export_aiperf.json"))
            if "phases" not in path.relative_to(directory).parts
        ]
        result = runs_to_dataframe(runs, x, "avg", y, "avg")
        if result["x_stat_actual"] != "avg" or result["y_stat_actual"] != "avg":
            raise ValueError(f"{label}: native average metrics are missing")
        frame = result["df"].groupby("concurrency", as_index=False)[[x, y]].mean()
        frames.append(frame.assign(series=label))
    figure = PlotGenerator(theme=PlotTheme.LIGHT).create_scatter_line_plot(
        pd.concat(frames),
        x,
        y,
        group_by="series",
        title=args.title,
        x_label="Interactivity (tokens/s/user)",
        y_label="Output throughput (tokens/s)",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(args.output)


if __name__ == "__main__":
    main()
