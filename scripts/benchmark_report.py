#!/usr/bin/env python3
"""HTML and Markdown report generation for benchmark results.

Split out of ``benchmark.py``, which had grown past 1800 lines with this
code inline. ``benchmark.py report`` imports :func:`report_benchmark` from
here; everything in this module is presentation and reads nothing but an
already-loaded result dict.
"""

from __future__ import annotations

import json
import statistics
import sys
from html import escape as html_escape
from pathlib import Path

from benchmark import (  # type: ignore[import-not-found]
    BENCHMARK_VERSION,
    KERNEL_TOP_N,
    _get_gpu_name,
    _short_kernel_name,
    compute_aggregate_throughput,
    compute_statistics,
)

# Kernels below this share of device time are merged into one "Other" slice.
# A profiled step launches dozens of kernels and all but a handful are under a
# millisecond; drawn individually they are an unreadable fringe of hairlines.
KERNEL_PIE_MIN_PCT = 2.0


def _unique_labels(labels: list[str]) -> list[str]:
    """Suffix repeated labels: plotly merges pie slices that share a label.

    Elision can still collide on kernels that differ only in the elided middle,
    and a silent merge would misattribute their time.
    """
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return out


def _wrap_kernel_name(name: str, width: int = 60) -> str:
    """Break a kernel name across lines so a hover tooltip stays on screen.

    Escaped chunk by chunk: the result is interpolated into a plotly hover
    template, which renders it as HTML, and the breaks are the only markup
    this should contribute.
    """
    return "<br>".join(
        html_escape(name[i : i + width]) for i in range(0, len(name), width)
    )


# The phases a step is drawn as, in the order they happen. ``fetch_ms`` is
# deliberately absent: it spans queue + H2D, so charting it alongside them
# would double-count the wait. ``bwd_ms`` likewise contains ``clip_ms``, which
# the builders below subtract out before plotting.
_PHASE_COLORS = {
    "queue_ms": "#636EFA",
    "h2d_ms": "#AB63FA",
    "fwd_ms": "#00CC96",
    "bwd_ms": "#FFA15A",
    "clip_ms": "#FECB52",
    "opt_ms": "#EF553B",
    "other_ms": "#B6B6B6",
}

_PHASE_LABELS = {
    "queue_ms": "DataLoader wait",
    "h2d_ms": "H2D transfer",
    "fwd_ms": "Forward",
    "bwd_ms": "Backward",
    "clip_ms": "Grad clip",
    "opt_ms": "Optimizer",
    "other_ms": "Other",
}

_MEMORY_MARK_LABELS = {
    "queue": "After queue",
    "fetch": "After H2D",
    "fwd": "After forward",
    "pre_clip": "After backward",
    "bwd": "After clip",
    "opt": "After optimizer",
}


def _phase_series(per_step) -> list[tuple[str, str, list[float]]]:
    """Per-step series for every drawn phase, with grad clip split out of bwd.

    The trainer times ``bwd_ms`` from the end of forward to the end of the
    clip, so the clip sits inside it; stacking both as measured would count it
    twice.
    """
    series = []
    for key, label in _PHASE_LABELS.items():
        if key == "bwd_ms":
            values = [s["bwd_ms"] - s["clip_ms"] for s in per_step]
        elif key == "other_ms":
            # The named phases do not tile the step: the trainer times
            # ``queue_ms`` to the end of the queue wait but starts ``h2d_ms``
            # at ``pre_h2d``, so the recovery bookkeeping in between belongs to
            # no phase. Draw it explicitly so the stack sums to ``step_ms``.
            values = [
                max(
                    s["step_ms"]
                    - (
                        s["queue_ms"]
                        + s["h2d_ms"]
                        + s["fwd_ms"]
                        + s["bwd_ms"]
                        + s["opt_ms"]
                    ),
                    0.0,
                )
                for s in per_step
            ]
        else:
            values = [s[key] for s in per_step]
        series.append((key, label, values))
    return series


def _build_timing_traces(per_step):
    """Build stacked area traces for timing breakdown."""
    import plotly.graph_objects as go  # noqa: PLC0415

    steps = list(range(len(per_step)))
    return [
        go.Scatter(
            x=steps,
            y=values,
            name=label,
            mode="lines",
            stackgroup="timing",
            line={"width": 0.5, "color": _PHASE_COLORS[key]},
        )
        for key, label, values in _phase_series(per_step)
    ]


def _build_memory_traces(per_step):
    """Build line traces for memory at phase boundaries."""
    import plotly.graph_objects as go  # noqa: PLC0415

    steps = list(range(len(per_step)))
    traces = []
    for mark, label in _MEMORY_MARK_LABELS.items():
        values = [s.get("memory_mb", {}).get(mark) for s in per_step]
        if not any(v is not None for v in values):
            continue
        traces.append(go.Scatter(x=steps, y=values, mode="lines", name=label))
    return traces


def _get_phase_means(per_step):
    """Mean and std of each phase duration over the measured steps."""
    return [
        (
            key,
            label,
            statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        for key, label, values in _phase_series(per_step)
    ]


def _build_timing_fig(per_step):
    """Mean time per training phase as a horizontal bar chart."""
    import plotly.graph_objects as go  # noqa: PLC0415

    phase_stats = _get_phase_means(per_step)
    total_ms = sum(mean for _, _, mean, _ in phase_stats)

    fig = go.Figure()
    for key, label, mean_val, std_val in reversed(phase_stats):
        pct = mean_val / total_ms * 100 if total_ms > 0 else 0
        fig.add_trace(
            go.Bar(
                y=[label],
                x=[mean_val],
                error_x={"type": "data", "array": [std_val], "visible": True},
                orientation="h",
                name=label,
                marker_color=_PHASE_COLORS.get(key, "#999"),
                text=f"{mean_val:.1f}ms ({pct:.1f}%)",
                textposition="auto",
                showlegend=False,
            )
        )
    fig.update_layout(barmode="stack", yaxis={"categoryorder": "array"})
    return fig


def _timing_sub(per_step) -> str:
    """Key numbers for the timing section: step time, throughput, spread."""
    step_stats = compute_statistics([s["step_ms"] for s in per_step])
    # Use the same time-weighted figure the Markdown summary and ``compare``
    # report. ``mean(tokens_per_s)`` would overweight short, fast steps and
    # make the HTML headline disagree with every other view of the same run.
    eff_tps = compute_aggregate_throughput(per_step)["effective_rank0_tokens_per_s"]
    return (
        f"{step_stats['mean']:.1f} ms/step "
        f"[{step_stats['ci95_lower']:.1f}, {step_stats['ci95_upper']:.1f}] · "
        f"{eff_tps:.0f} tok/s effective · "
        f"n={step_stats['count']} steps · brackets are 95% CI, "
        "error bars ±1 std"
    )


def _build_memory_fig(per_step):
    """Mean allocated memory at each phase boundary, with std error bars."""
    import plotly.graph_objects as go  # noqa: PLC0415

    means = []
    stds = []
    for mark in _MEMORY_MARK_LABELS:
        vals = [
            s["memory_mb"][mark]
            for s in per_step
            if "memory_mb" in s and mark in s["memory_mb"]
        ]
        means.append(statistics.mean(vals) if vals else 0)
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)

    fig = go.Figure(
        go.Bar(
            x=list(_MEMORY_MARK_LABELS.values()),
            y=means,
            error_y={"type": "data", "array": stds, "visible": True},
            marker_color="#636EFA",
            text=[f"{m:.0f}" for m in means],
            textposition="outside",
            # The section margins are tight; without this the label on the
            # tallest bar is clipped at the plot edge.
            cliponaxis=False,
            showlegend=False,
        )
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def _memory_sub(results) -> str:
    """Key numbers for the memory section: the peaks torch reports per rank.

    The bars come from rank 0's per-step marks, so the peaks are qualified
    with the rank they describe rather than presented as a run-wide figure.
    """
    per_rank = results["memory"]["per_rank"]
    alloc = max(r["peak_allocated_mb"] for r in per_rank)
    reserved = max(r["peak_reserved_mb"] for r in per_rank)
    scope = "rank 0" if len(per_rank) == 1 else f"max over {len(per_rank)} ranks"
    return (
        f"{alloc:.0f} MB peak allocated · {reserved:.0f} MB peak reserved "
        f"({scope}) · bars are rank 0's mean at each phase boundary, "
        "error bars ±1 std"
    )


def _build_appendix_fig(per_step):
    """Build per-step raw data charts for the appendix."""
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    steps = list(range(len(per_step)))
    has_memory = any("memory_mb" in s for s in per_step)
    n_rows = 4 if has_memory else 3

    titles = [
        "Per-Step Timing Breakdown (ms)",
        "Per-Step Throughput (tokens/s)",
        "Per-Step Time (ms)",
    ]
    if has_memory:
        titles.insert(2, "Per-Step GPU Memory Allocated (logical) (MB)")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.06,
    )

    for trace in _build_timing_traces(per_step):
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[s["tokens_per_s"] for s in per_step],
            mode="lines+markers",
            name="tokens/s",
            marker={"size": 3},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(rangemode="tozero", row=2, col=1)

    if has_memory:
        for trace in _build_memory_traces(per_step):
            fig.add_trace(trace, row=3, col=1)
        fig.update_yaxes(rangemode="tozero", row=3, col=1)

    step_row = n_rows
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[s["step_ms"] for s in per_step],
            mode="lines+markers",
            name="step_ms",
            marker={"size": 3},
            line={"color": "#19D3F3"},
            showlegend=False,
        ),
        row=step_row,
        col=1,
    )
    fig.update_yaxes(rangemode="tozero", row=step_row, col=1)
    fig.update_xaxes(title_text="Step", row=n_rows, col=1)

    return fig, n_rows


def _kernel_markdown_lines(kernels: dict | None) -> list[str]:
    """Markdown table of the top GPU kernels, or nothing when not profiled."""
    if not kernels or not kernels.get("kernels"):
        return []
    total = kernels.get("total_device_ms_per_step", 0)
    lines = [
        f"## Appendix: Top {KERNEL_TOP_N} GPU Kernels",
        "",
        (
            f"{total:.1f} ms GPU time/step · {len(kernels['kernels'])} kernels"
            " · self time, averaged over the profiled steps"
        ),
        "",
        "| Kernel | ms/step | % GPU | calls/step |",
        "|--------|---------|-------|------------|",
    ]
    for k in kernels["kernels"][:KERNEL_TOP_N]:
        pct = k["ms_per_step"] / total * 100 if total else 0
        lines.append(
            f"| `{_short_kernel_name(k['name'])}` | {k['ms_per_step']:.2f}"
            f" | {pct:.1f}% | {k['calls_per_step']:.1f} |"
        )
    lines.append("")
    return lines


def _write_markdown_summary(results: dict, md_path: Path) -> None:
    """Write a markdown summary of benchmark results."""
    cfg = results["config"]
    prov = results["provenance"]
    mem = results["memory"]
    timing = results["timing"]
    per_step = results.get("per_step")
    aggregate = results["aggregate"]

    spec = cfg.get("speculator_type", "?")
    verifier = cfg.get("verifier_name_or_path", "?")
    gpu_desc = f"{cfg.get('num_gpus_used', 1)} x {_get_gpu_name(results)}"
    steps_desc = (
        f"{cfg.get('warmup_steps', '?')} warmup"
        f" + {cfg.get('measured_steps', '?')} measured"
    )
    lines = [
        f"# Benchmark: {spec} / {verifier}",
        "",
        f"- **seq_len**: {cfg.get('total_seq_len', '?')}",
        f"- **GPUs**: {gpu_desc}",
        f"- **optimizer**: {cfg.get('optimizer', '?')}",
        f"- **dtype**: {cfg.get('hidden_states_dtype', '?')}",
        f"- **synthetic**: {cfg.get('synthetic_data', '?')}",
        f"- **steps**: {steps_desc}",
        f"- **git**: `{prov.get('git_sha', 'unknown')[:12]}`",
        f"- **date**: {prov.get('timestamp', '?')[:10]}",
        "",
        "## Timing",
        "",
        "| Metric | Mean | Std | 95% CI | Min | Max |",
        "|--------|------|-----|--------|-----|-----|",
    ]

    for key, stats in timing.items():
        lines.append(
            f"| {key} | {stats['mean']:.2f} | {stats['std']:.2f}"
            f" | [{stats['ci95_lower']:.2f}, {stats['ci95_upper']:.2f}]"
            f" | {stats['min']:.2f} | {stats['max']:.2f} |"
        )

    eff = aggregate["effective_rank0_tokens_per_s"]
    dur = aggregate["measured_time_s"]
    lines += [
        "",
        f"**Effective throughput**: {eff:.0f} tokens/s over {dur:.1f}s",
        "",
        "## Memory",
        "",
        "| Rank | Peak allocated (MB) | Peak reserved (MB) |",
        "|------|---------------------|--------------------|",
    ]
    for r in mem["per_rank"]:
        lines.append(
            f"| {r['rank']} | {r['peak_allocated_mb']:.0f}"
            f" | {r['peak_reserved_mb']:.0f} |"
        )
    if mem["phases"]:
        phases = ", ".join(f"{p}={mb:.0f}" for p, mb in mem["phases"].items())
        lines += ["", f"**Phase peaks** (MB allocated, rank 0): {phases}"]
    lines.append("")

    # Phase breakdown percentages
    if per_step:
        phase_stats = _get_phase_means(per_step)
        total_ms = sum(m for _, _, m, _ in phase_stats)
        if total_ms > 0:
            lines += ["## Phase Breakdown", ""]
            lines.append("| Phase | Mean (ms) | % |")
            lines.append("|-------|-----------|---|")
            for _, label, mean_val, _ in phase_stats:
                pct = mean_val / total_ms * 100
                lines.append(f"| {label} | {mean_val:.1f} | {pct:.1f}% |")
            lines.append("")

    lines += _kernel_markdown_lines(results.get("kernels"))

    md_path.write_text("\n".join(lines))
    print(f"Markdown summary written to {md_path}")


def _build_kernel_pie(kernels: dict):
    """Pie of device time per kernel, with the sub-2% tail merged into "Other"."""
    import plotly.graph_objects as go  # noqa: PLC0415

    total = kernels.get("total_device_ms_per_step", 0) or 1.0
    labels: list[str] = []
    values: list[float] = []
    hovers: list[str] = []
    tail_ms = 0.0
    tail_n = 0

    for k in kernels["kernels"]:
        if k["ms_per_step"] / total * 100 < KERNEL_PIE_MIN_PCT:
            tail_ms += k["ms_per_step"]
            tail_n += 1
            continue
        # Plotly renders legend text as HTML, so the name is escaped here too.
        labels.append(html_escape(_short_kernel_name(k["name"])))
        values.append(k["ms_per_step"])
        # The legend label is clipped, so the full name lives in the hover.
        hovers.append(
            f"<b>{_wrap_kernel_name(k['name'])}</b><br>"
            f"{k['ms_per_step']:.2f} ms/step · "
            f"{k['calls_per_step']:.1f} calls/step"
        )

    if tail_n:
        labels.append(f"Other ({tail_n} kernels)")
        values.append(tail_ms)
        hovers.append(
            f"<b>{tail_n} kernels</b>, each under {KERNEL_PIE_MIN_PCT:.0f}% "
            f"of device time<br>{tail_ms:.2f} ms/step combined"
        )

    fig = go.Figure(
        go.Pie(
            labels=_unique_labels(labels),
            values=values,
            customdata=hovers,
            hovertemplate="%{customdata}<extra></extra>",
            texttemplate="%{percent}",
            textposition="inside",
            insidetextorientation="horizontal",
            sort=False,  # already sorted by time; keep "Other" last
            direction="clockwise",
            marker={"line": {"color": "#fff", "width": 1}},
        )
    )
    fig.update_layout(
        legend={"font": {"family": "ui-monospace, monospace", "size": 10}}
    )
    return fig


def _kernel_name_cell(name: str) -> str:
    """Kernel name as HTML, carrying the full name in a tooltip when clipped.

    Deliberately a ``span.kname`` rather than a ``<code>``: viewers and browser
    extensions routinely restyle ``code`` -- a dark chip behind every cell is
    the common one -- and a class we style ourselves cannot be hijacked that
    way. Elided names get a dotted underline and a help cursor so it is visible
    that there is more to hover for; whole ones stay plain.
    """
    short = _short_kernel_name(name)
    if short == name:
        return f'<span class="kname">{html_escape(name)}</span>'
    return (
        f'<span class="kname trunc" title="{html_escape(name)}">'
        f"{html_escape(short)}</span>"
    )


def _section_html(title: str, sub: str, fig, height: int, *, with_js=False) -> str:
    """One report section: heading, a line of key data, then its chart.

    Every body section goes through here so the three of them stay visually
    identical; plotly's own subplot titles and annotations cannot match the
    surrounding page, which is why the figures carry neither.

    ``with_js`` inlines the plotly bundle and must be set on exactly one
    section. Inlined rather than pulled from the CDN: reports are read from
    air-gapped clusters and out of CI artifact archives, where a remote script
    tag leaves a page of blank boxes.
    """
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin={"t": 30, "b": 40, "l": 20, "r": 20},
    )
    fig_html = fig.to_html(full_html=False, include_plotlyjs=with_js)
    return f"""
<h2 class="sec">{title}</h2>
<p class="sub">{sub}</p>
{fig_html}"""


def _kernel_html(kernels: dict | None) -> tuple[str, str]:
    """Kernel-time pie for the body and top-N table for the appendix.

    Returns ``("", "")`` when the run was not profiled.
    """
    if not kernels or not kernels.get("kernels"):
        return "", ""
    entries = kernels["kernels"]
    total = kernels.get("total_device_ms_per_step", 0)
    shown = sum(
        1
        for k in entries
        if k["ms_per_step"] / (total or 1) * 100 >= KERNEL_PIE_MIN_PCT
    )
    body = _section_html(
        "GPU Kernel Breakdown",
        f"{total:.1f} ms GPU time/step · {len(entries)} kernels · "
        f"{shown} at or above {KERNEL_PIE_MIN_PCT:.0f}%, rest grouped as "
        '"Other" · hover for full name, ms/step and calls/step',
        _build_kernel_pie(kernels),
        height=420,
    )

    rows = "".join(
        f"<tr><td>{_kernel_name_cell(k['name'])}</td>"
        f"<td>{k['ms_per_step']:.2f}</td>"
        f"<td>{(k['ms_per_step'] / total * 100 if total else 0):.1f}%</td>"
        f"<td>{k['calls_per_step']:.1f}</td></tr>"
        for k in entries[:KERNEL_TOP_N]
    )
    appendix = f"""
<details>
<summary>Appendix: Top {KERNEL_TOP_N} GPU Kernels</summary>
<table class="kernels">
<thead><tr><th>Kernel</th><th>ms/step</th><th>% GPU</th><th>calls/step</th></tr>
</thead><tbody>{rows}</tbody></table>
</details>"""
    return body, appendix


def _report_paths(result_path: str, output_path: str | None) -> tuple[Path, Path]:
    """Resolve the input, HTML and Markdown paths, refusing to collide.

    The two outputs are derived from one name, so a ``--output`` that already
    ends in ``.md`` -- or a result file named ``.html`` -- would otherwise have
    one write silently clobber another file the command was asked to read or
    produce.
    """
    json_path = Path(result_path)
    html_path = (
        Path(output_path)
        if output_path
        else json_path.with_name(f"{json_path.stem}_report.html")
    )
    md_path = html_path.with_suffix(".md")
    paths = (json_path, html_path, md_path)
    if len({p.resolve() for p in paths}) != len(paths):
        raise SystemExit(
            f"Report paths collide: input {json_path}, HTML {html_path}, "
            f"Markdown {md_path}. Pass --output with a distinct .html name."
        )
    return html_path, md_path


def report_benchmark(result_path: str, output_path: str | None = None) -> None:
    """Generate an interactive HTML report from benchmark results."""
    try:
        import plotly  # noqa: PLC0415, F401
    except ImportError:
        print("plotly is required: pip install plotly", file=sys.stderr)
        sys.exit(1)

    with open(result_path) as f:
        results = json.load(f)

    version = results.get("benchmark_version")
    if version != BENCHMARK_VERSION:
        raise SystemExit(
            f"Incompatible benchmark version: the file reports {version}, this "
            f"script reads {BENCHMARK_VERSION}. Re-run the benchmark with this "
            "version of the script to report on it."
        )

    per_step = results.get("per_step")
    if not per_step:
        print(
            "No per-step data found. Re-run benchmark without --no-per-step.",
            file=sys.stderr,
        )
        sys.exit(1)

    html_path, md_path = _report_paths(result_path, output_path)

    # Everything below comes from the result file, which the report renders as
    # HTML; escape it rather than trust whoever produced the JSON.
    cfg = results["config"]
    prov = results["provenance"]
    heading = " &middot; ".join(
        html_escape(str(part))
        for part in (
            cfg.get("speculator_type", "?"),
            cfg.get("verifier_name_or_path", "?"),
            f"seq_len={cfg.get('total_seq_len', '?')}",
            f"{cfg.get('num_gpus_used', 1)} GPU(s)",
        )
    )
    sub = " &middot; ".join(
        html_escape(str(part))
        for part in (
            f"{cfg.get('measured_steps', '?')} measured steps",
            _get_gpu_name(results),
            prov.get("git_sha", "unknown")[:12],
        )
    )

    # --- Body sections: timing, memory, kernels ---
    timing_html = _section_html(
        "Step Timing Breakdown",
        _timing_sub(per_step),
        _build_timing_fig(per_step),
        height=340,
        with_js=True,  # first figure on the page carries the plotly bundle
    )
    memory_html = ""
    if any("memory_mb" in s for s in per_step):
        memory_html = _section_html(
            "GPU Memory Allocated (logical)",
            _memory_sub(results),
            _build_memory_fig(per_step),
            height=340,
        )

    # --- Appendix: per-step raw data ---
    appendix_fig, n_rows = _build_appendix_fig(per_step)
    appendix_fig.update_layout(
        height=300 * n_rows,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    appendix_html = appendix_fig.to_html(full_html=False, include_plotlyjs=False)

    # --- Kernel breakdown (only present when the run used --profile) ---
    kernel_html, kernel_appendix_html = _kernel_html(results.get("kernels"))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Benchmark Report</title>
<!-- The plotly figures render on a white template, so the page is light-only.
     Without this the browser paints a dark canvas in dark mode and leaves the
     text at its (dark grey) authored colour. -->
<meta name="color-scheme" content="light">
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1200px;
         margin: 0 auto; padding: 20px; color: #24292f; background: #fff; }}
  h1 {{ font-size: 1.3em; margin-bottom: 0; }}
  h2.sec {{ font-size: 1.1em; font-weight: 600; margin: 34px 0 0; }}
  .sub {{ color: #656d76; font-size: 0.85em; max-width: 900px;
          margin: 4px 0 12px; }}
  details {{ margin-top: 30px; }}
  summary {{ cursor: pointer; font-size: 1.1em; font-weight: 600;
             padding: 8px 0; }}
  table.kernels {{ border-collapse: collapse; font-size: 0.85em; width: 100%; }}
  table.kernels th, table.kernels td {{ text-align: right; padding: 5px 8px;
                                        border-bottom: 1px solid #d8dee4; }}
  table.kernels th {{ color: #656d76; font-weight: 600; }}
  table.kernels tbody tr:nth-child(odd) {{ background: #f6f8fa; }}
  table.kernels th:first-child, table.kernels td:first-child
      {{ text-align: left; }}
  .kname {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.95em; color: #0550ae; background: none;
            white-space: nowrap; padding: 0; border-radius: 0; }}
  .kname.trunc {{ border-bottom: 1px dotted #8c959f; cursor: help; }}
</style>
</head><body>
<h1>{heading}</h1>
<div class="sub">{sub}</div>
{timing_html}
{memory_html}
{kernel_html}
{kernel_appendix_html}
<details>
<summary>Appendix: Per-Step Raw Data</summary>
{appendix_html}
</details>
</body></html>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html)
    print(f"Report written to {html_path}")

    _write_markdown_summary(results, md_path)
