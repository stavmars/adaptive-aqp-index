"""Layer 3 -- chart templates over the Layer-2 reductions.

Thin renderers shared by the exploratory driver (and, later, a figure manifest).
Each takes an already-loaded frame (or a per-cell summary) plus the axis to vary
-- the comparison series, usually the method -- and returns a matplotlib Figure.
No renderer re-reads files or re-implements a reduction; series colour/marker/
linestyle come from `style`, so a method looks the same across every figure.

Three templates cover the views the explorer needs:
  * `trajectory` -- one line per method along the query sequence (per-query or
    cumulative cost);
  * `bar`        -- one bar per method of a single reduced scalar;
  * `sweep`      -- one line per method across a numeric experiment axis.
"""
from __future__ import annotations

from . import aggregate, style

# Readable axis labels for the metrics the templates plot.
METRIC_LABEL = {
    "latency_ms": "latency (ms)",
    "measure_reads": "measure values read",
    "cum_ms": "cumulative time (ms)",
    "total_reads": "measure values read",
    "total_latency_ms": "query time (ms)",
    "lat_p50": "latency p50 (ms)",
    "lat_p95": "latency p95 (ms)",
    "lat_p99": "latency p99 (ms)",
    "speedup_vs_scan": "speedup over scan",
    "within_eb": "fraction within error bound",
    "coverage": "CI coverage",
    "init_ms": "init time (ms)",
}


def _label(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric)


def _per_query_median(frame, metric: str):
    """Per (method, query) value, taking the median over the replicate runs."""
    pqc = aggregate.per_query_cost(frame)
    return (pqc.groupby(["method", "query_ordinal"], dropna=False)[metric]
               .median().reset_index())


def trajectory(frame, *, metric="latency_ms", cumulative=False, title=None,
               width="single"):
    """x = query index, one line per method. `cumulative` sums the per-query
    metric along the sequence and starts each method at its init offset (the
    up-front build), so the curves are build-inclusive."""
    import numpy as np
    import matplotlib.pyplot as plt

    style.apply()
    pq = _per_query_median(frame, metric)
    init = (aggregate.per_query_cost(frame).groupby("method")["init_ms"].median()
            if cumulative else None)

    fig, ax = plt.subplots(figsize=style.figure_size(width))
    for method, g in pq.groupby("method", dropna=False):
        g = g.sort_values("query_ordinal")
        y = g[metric].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y) + float(init.get(method, 0.0))
        st = style.method_style(method)
        ax.plot(g["query_ordinal"], y, color=st["color"], linestyle=st["linestyle"],
                linewidth=1.0, label=st["label"])
    ax.set_xlabel("query")
    ax.set_ylabel(("cumulative " if cumulative else "") + _label(metric))
    if title:
        ax.set_title(title)
    ax.legend(fontsize=6, framealpha=0.6)
    return fig


def bar(summary, *, metric, title=None, width="single", ylabel=None):
    """x = method, one bar per method of a single reduced scalar from a
    `cell_summary` (one row per method)."""
    import matplotlib.pyplot as plt

    style.apply()
    s = summary.dropna(subset=[metric]).sort_values(metric)
    fig, ax = plt.subplots(figsize=style.figure_size(width))
    colors = [style.method_style(m)["color"] for m in s["method"]]
    ax.bar(s["method"].astype(str), s[metric], color=colors)
    ax.set_ylabel(ylabel or _label(metric))
    ax.tick_params(axis="x", rotation=45)
    if title:
        ax.set_title(title)
    return fig


def sweep(summary, *, x_axis, metric, title=None, width="single"):
    """x = a numeric experiment axis (nm/str/eb/n), one line per method, from a
    `cell_summary` that varies only `x_axis`."""
    import matplotlib.pyplot as plt

    style.apply()
    s = summary.dropna(subset=[metric]).copy()
    s[x_axis] = s[x_axis].astype(float)
    fig, ax = plt.subplots(figsize=style.figure_size(width))
    for method, g in s.sort_values(x_axis).groupby("method", dropna=False):
        st = style.method_style(method)
        ax.plot(g[x_axis], g[metric], color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=3, linewidth=1.0, label=st["label"])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(_label(metric))
    if title:
        ax.set_title(title)
    ax.legend(fontsize=6, framealpha=0.6)
    return fig
