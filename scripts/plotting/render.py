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


# Legend placement, shared by every template (set once from the driver):
#   "outside" -- a frameless legend below the axes, so it never eats the data
#                canvas (and several plots tile cleanly on a page);
#   "none"    -- no per-figure legend (use the standalone legend.pdf instead).
LEGEND = "outside"

# Whether to draw the per-axes title (set once from the driver). On by default;
# turn off (--title off) to produce bare figures whose context (dataset, nm, eb)
# is given in the surrounding caption instead.
TITLES = True


def _draw_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if LEGEND == "none" or not handles:
        return
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(len(labels), 4), fontsize=6, frameon=False,
              handlelength=1.6, columnspacing=1.2)


def legend_only(methods, *, width="full", ncol=None):
    """A standalone figure containing only the per-method key (no axes), to drop
    once into a page of subplots rendered with `--legend none`."""
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    style.apply()
    handles = []
    for method in methods:
        st = style.method_style(method)
        handles.append(Line2D([0], [0], color=st["color"], marker=st["marker"],
                              linestyle=st["linestyle"], markersize=4,
                              label=st["label"]))
    w = style.WIDTH_FULL if width == "full" else style.WIDTH_SINGLE
    fig = plt.figure(figsize=(w, 0.5))
    fig.legend(handles=handles, loc="center", ncol=ncol or len(methods),
               fontsize=7, frameon=False)
    return fig


def _label(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric)


def _log_label(label: str) -> str:
    """Mark an axis label as log scale, merged into the units parenthesis when
    present: 'latency (ms)' -> 'latency (ms, log scale)'; 'speedup over scan' ->
    'speedup over scan (log scale)'."""
    label = label or ""
    if label.endswith(")"):
        return label[:-1] + ", log scale)"
    return (label + " (log scale)").strip()


def _per_query_median(frame, metric: str):
    """Per (method, query) value, taking the median over the replicate runs."""
    pqc = aggregate.per_query_cost(frame)
    return (pqc.groupby(["method", "query_ordinal"], dropna=False)[metric]
               .median().reset_index())


def trajectory(frame, *, metric="latency_ms", cumulative=False, title=None,
               width="single", logy=True, head=None, linewidth=1.0):
    """x = query index, one line per method. `cumulative` sums the per-query
    metric along the sequence and starts each method at its init offset (the
    up-front build), so the curves are build-inclusive. `logy` (default on, since
    times span orders of magnitude) is applied only when all values are positive,
    so the convergence of cheap methods is not flattened against the axis. `head`
    limits the per-query view to the first N queries -- the convergence region --
    so a long noisy steady-state tail does not swamp the readable part."""
    import numpy as np
    import matplotlib.pyplot as plt

    style.apply()
    pq = _per_query_median(frame, metric)
    if head is not None:
        pq = pq[pq["query_ordinal"] < head]
    init = (aggregate.per_query_cost(frame).groupby("method")["init_ms"].median()
            if cumulative else None)

    fig, ax = plt.subplots(figsize=style.figure_size(width))
    all_positive = True
    for method, g in pq.groupby("method", dropna=False):
        g = g.sort_values("query_ordinal")
        y = g[metric].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y) + float(init.get(method, 0.0))
        if y.size and y.min() <= 0:
            all_positive = False
        st = style.method_style(method)
        ax.plot(g["query_ordinal"], y, color=st["color"], linestyle=st["linestyle"],
                linewidth=linewidth, label=st["label"])
    applied_log = logy and all_positive
    if applied_log:
        ax.set_yscale("log")
    ax.set_xlabel("query")
    ylab = ("cumulative " if cumulative else "") + _label(metric)
    ax.set_ylabel(_log_label(ylab) if applied_log else ylab)
    if title and TITLES:
        ax.set_title(title)
    _draw_legend(ax)
    return fig


def bar(summary, *, metric, title=None, width="single", ylabel=None, logy=False):
    """x = method, one bar per method of a single reduced scalar from a
    `cell_summary` (one row per method). `logy` for counts/times that span orders
    of magnitude (applied only when every bar is positive)."""
    import matplotlib.pyplot as plt

    style.apply()
    s = summary.dropna(subset=[metric]).sort_values(metric)
    fig, ax = plt.subplots(figsize=style.figure_size(width))
    colors = [style.method_style(m)["color"] for m in s["method"]]
    ax.bar(s["method"].astype(str), s[metric], color=colors)
    applied_log = logy and (s[metric] > 0).all()
    if applied_log:
        ax.set_yscale("log")
    ylab = ylabel or _label(metric)
    ax.set_ylabel(_log_label(ylab) if applied_log else ylab)
    ax.tick_params(axis="x", rotation=45)
    if title and TITLES:
        ax.set_title(title)
    return fig


def grouped_bar(summary, *, metrics, title=None, width="single", ylabel=None,
                logy=False):
    """x = method, a small group of bars per method (e.g. latency p50/p95/p99)."""
    import numpy as np
    import matplotlib.pyplot as plt

    style.apply()
    s = summary.copy()
    methods = [str(m) for m in s["method"]]
    x = np.arange(len(methods))
    bar_w = 0.8 / max(1, len(metrics))
    fig, ax = plt.subplots(figsize=style.figure_size(width))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * bar_w, s[metric], width=bar_w, label=_label(metric))
    applied_log = logy and all((s[m] > 0).all() for m in metrics)
    if applied_log:
        ax.set_yscale("log")
    ax.set_xticks(x + bar_w * (len(metrics) - 1) / 2.0)
    ax.set_xticklabels(methods, rotation=45)
    ylab = ylabel or ""
    ax.set_ylabel(_log_label(ylab) if applied_log else ylab)
    if title and TITLES:
        ax.set_title(title)
    _draw_legend(ax)
    return fig


def scatter(points, *, x, y, by="method", xlabel=None, ylabel=None,
            diagonal=False, title=None, width="single"):
    """One point per row, grouped (coloured) by `by`. `diagonal` adds the y = x
    reference line (the achieved-vs-targeted-error guarantee line)."""
    import matplotlib.pyplot as plt

    style.apply()
    fig, ax = plt.subplots(figsize=style.figure_size(width))
    for key, g in points.groupby(by, dropna=False):
        st = style.method_style(key) if by == "method" else {
            "color": None, "marker": "o", "label": str(key)}
        ax.scatter(g[x], g[y], s=6, alpha=0.55, color=st["color"],
                   marker=st["marker"], label=st["label"])
    if diagonal and len(points):
        lo = float(min(points[x].min(), points[y].min()))
        hi = float(max(points[x].max(), points[y].max()))
        ax.plot([lo, hi], [lo, hi], color="0.5", linewidth=0.6, linestyle="--",
                label="y = x")
    ax.set_xlabel(xlabel or _label(x))
    ax.set_ylabel(ylabel or _label(y))
    if title and TITLES:
        ax.set_title(title)
    _draw_legend(ax)
    return fig


def quantiles(q, *, x, bands, by="method", xlabel=None, ylabel=None,
              diagonal=False, title=None, width="single"):
    """Quantile compliance plot: per `by` group, one line per quantile column in
    `bands` (the first drawn solid, later ones dashed and lighter) over the
    numeric axis `x`, with an optional y = x reference. Log-log when positive,
    since error targets span orders of magnitude. Used for achieved-vs-targeted
    error: the upper band sitting on or below the diagonal is the probabilistic
    guarantee holding."""
    import matplotlib.pyplot as plt

    style.apply()
    s = q.copy()
    s[x] = s[x].astype(float)
    fig, ax = plt.subplots(figsize=style.figure_size(width))
    linestyles = ["-", "--", ":"]
    alphas = [1.0, 0.55, 0.4]
    for key, g in s.sort_values(x).groupby(by, dropna=False):
        st = style.method_style(key) if by == "method" else {
            "color": None, "marker": "o", "label": str(key)}
        for i, band in enumerate(bands):
            ax.plot(g[x], g[band], color=st["color"],
                    linestyle=linestyles[i % len(linestyles)],
                    alpha=alphas[i % len(alphas)], marker=st["marker"],
                    markersize=3, linewidth=1.0,
                    label=f"{st['label']} {band}")
    if diagonal and len(s):
        vals = [s[x]] + [s[b] for b in bands]
        lo = float(min(v.min() for v in vals))
        hi = float(max(v.max() for v in vals))
        ax.plot([lo, hi], [lo, hi], color="0.5", linewidth=0.6, linestyle="--",
                label="y = x")
    if (s[x] > 0).all() and all((s[b] > 0).all() for b in bands):
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or _label(x))
    ax.set_ylabel(ylabel or _label(bands[0]))
    if title and TITLES:
        ax.set_title(title)
    _draw_legend(ax)
    return fig


def sweep(summary, *, x_axis, metric, title=None, width="single",
          logx=False, logy=False):
    """x = a numeric experiment axis (nm/str/eb/n), one line per method, from a
    `cell_summary` that varies only `x_axis`. `logx`/`logy` for axes that span
    orders of magnitude (applied only when the values are positive)."""
    import matplotlib.pyplot as plt

    style.apply()
    s = summary.dropna(subset=[metric]).copy()
    s[x_axis] = s[x_axis].astype(float)
    fig, ax = plt.subplots(figsize=style.figure_size(width))
    for method, g in s.sort_values(x_axis).groupby("method", dropna=False):
        st = style.method_style(method)
        ax.plot(g[x_axis], g[metric], color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=3, linewidth=1.0, label=st["label"])
    xlog = logx and (s[x_axis] > 0).all()
    ylog = logy and (s[metric] > 0).all()
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(_log_label(x_axis) if xlog else x_axis)
    ax.set_ylabel(_log_label(_label(metric)) if ylog else _label(metric))
    if title and TITLES:
        ax.set_title(title)
    _draw_legend(ax)
    return fig
