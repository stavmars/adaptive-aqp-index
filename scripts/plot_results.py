#!/usr/bin/env python3
"""Discovery-driven exploratory plots from the results matrix.

Reuses the analysis core (load -> aggregate -> render) and, for each
`(dataset, workload)` present, draws a standard set of views over whatever
methods exist -- inferring the axis roles rather than declaring them:

  * the facet is `(dataset, workload)`;
  * the replicate axis `run_id` is always reduced away;
  * exactly one remaining axis may vary -- normally `method` (a method
    comparison), or a single numeric axis like `nm`/`str`/`eb` (a sweep).

If two or more axes still vary within a facet the comparison is confounded, so
the facet is skipped with a message naming the offending axes (a hard-constraint
axis such as `mem` is never auto-varied). Absent cells never crash a run; only
corrupt input fails (in Layer 1). This is the quick look during experimentation;
declared, manifest-driven figures come separately.

Declared figures (the figure manifest in `plotting/figures.py`) are the other
mode: each selects its own cells, so they are declared rather than inferred.
`--all` renders every figure, `--figure <id>` one,
`--plan <id>` into a plan-named directory; a figure whose cells are absent is
skipped, or fails under `--strict`.

Usage:
    scripts/plot_results.py --explore
    scripts/plot_results.py --all
    scripts/plot_results.py --figure cumulative_time --dataset taxi
    scripts/plot_results.py --plan substrate_sweep --strict
"""
from __future__ import annotations

import argparse
from pathlib import Path

import a3i_config
from plotting import aggregate, figures, load, render

# Axes a facet may legitimately vary on (besides the dataset/workload facet and
# the always-reduced run_id). `substrate` is intentionally excluded: it is bound
# to the method (each method has one substrate), so it co-varies with a method
# comparison rather than being an independent axis.
COMPARE_AXES = ("method", "nm", "str", "eb", "n")
# Hard-constraint axes: kept fixed unless a declared sweep varies them, never
# auto-varied in exploration.
HARD_AXES = ("mem",)


# Canonical method order for the shared legend.
METHOD_ORDER = ("scan", "kd", "kd_agg", "adkd", "adkd_agg", "adkd_sampling", "a3i")


def _write_legend(frame, out_dir: Path):
    """Emit a standalone legend.pdf with the per-method key present in `frame`,
    so a page of subplots rendered with --legend none can share one legend."""
    import matplotlib.pyplot as plt

    present = set(frame["method"].dropna().unique())
    methods = [m for m in METHOD_ORDER if m in present]
    if not methods:
        return
    fig = render.legend_only(methods)
    path = out_dir / "legend.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


class AmbiguousComparison(ValueError):
    """A facet whose cells differ on more than one axis -- not safely comparable."""


def varying_axes(frame):
    """The comparison axes that take more than one value in `frame`.

    Counts distinct *non-null* values, so `eb` (absent for exact methods) is not
    treated as varying merely because exact and approximate methods are mixed.
    """
    return [a for a in COMPARE_AXES + HARD_AXES
            if frame[a].dropna().nunique() > 1]


def explore_facet(frame, out_dir: Path, eb: float, *, width="single"):
    """Render the standard views for one `(dataset, workload)` facet.

    Returns the list of written paths. Raises `AmbiguousComparison` if the facet
    is not reducible to a single varied axis.
    """
    dataset = frame["dataset"].iloc[0]
    workload = frame["workload"].iloc[0]
    tag = f"{dataset}__{workload}"

    varying = varying_axes(frame)
    hard = [a for a in varying if a in HARD_AXES]
    if hard:
        raise AmbiguousComparison(
            f"{tag}: hard-constraint axis varies {hard}; use an explicit sweep")
    non_method = [a for a in varying if a != "method"]
    if len(varying) >= 2:
        raise AmbiguousComparison(
            f"{tag}: multiple axes vary {varying}; fix all but one to compare")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate.cell_summary(frame, eb)
    figs = {}
    if not non_method:
        # method comparison (or a single method): the standard cost views.
        figs["cumulative_time"] = render.trajectory(
            frame, metric="latency_ms", cumulative=True, title=tag, width=width)
        figs["measure_reads"] = render.bar(
            summary, metric="total_reads", title=tag, width=width)
        figs["latency_p50"] = render.bar(
            summary, metric="lat_p50", title=tag, width=width)
    else:
        # one numeric axis varies (method fixed): a sweep of cumulative cost.
        figs[f"sweep_{non_method[0]}"] = render.sweep(
            summary, x_axis=non_method[0], metric="cum_ms", title=tag, width=width)

    import matplotlib.pyplot as plt
    written = []
    for name, fig in figs.items():
        path = out_dir / f"{tag}__{name}.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def explore(frame, out_dir: Path, eb: float):
    """Render every facet present; skip an ambiguous/absent one with a message."""
    if frame.empty:
        print("no results found")
        return []
    written = []
    for (dataset, workload), sub in frame.groupby(["dataset", "workload"], dropna=False):
        try:
            paths = explore_facet(sub, out_dir, eb)
        except AmbiguousComparison as exc:
            print(f"skip {exc}")
            continue
        for path in paths:
            print(f"wrote {path}")
        written.extend(paths)
    if written:
        _write_legend(frame, out_dir)
    return written


def render_figures(frame, out_dir: Path, eb: float, nm, *, only=None, strict=False):
    """Render the declared figures (or just `only`) into `out_dir`.

    `eb`/`nm` are the within-facet axes the comparison figures pin to. A figure
    whose required cells are absent is skipped with a message; under `strict` it
    raises instead (the gate for a complete figure set).
    """
    import matplotlib.pyplot as plt

    versions = aggregate.mixed_versions(frame)
    if versions:
        print(f"warning: results span multiple engine builds {versions}; "
              "cross-build comparisons may be unfair")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [figures.by_id(only)] if only else list(figures.FIGURES)
    written = []
    for spec in specs:
        items = spec.build(frame, eb, nm)
        if not items:
            message = f"{spec.id}: required cells absent"
            if strict:
                raise figures.MissingCells(message)
            print(f"skip {message}")
            continue
        for suffix, fig in items:
            path = out_dir / f"{spec.id}__{suffix}.pdf"
            fig.savefig(path)
            plt.close(fig)
            written.append(path)
            print(f"wrote {path}")
    if written:
        _write_legend(frame, out_dir)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plots from the results matrix.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--explore", action="store_true",
                      help="discovery-driven exploratory views")
    mode.add_argument("--all", action="store_true",
                      help="render every declared figure")
    mode.add_argument("--figure", default=None,
                      help="render one declared figure by id")
    mode.add_argument("--plan", default=None,
                      help="render the declared figures into a plan-named dir")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (a mode-specific default otherwise)")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--eb", type=float, default=figures.PIN_DEFAULTS["eb"],
                    help="error bound the comparison figures pin to "
                         f"(default {figures.PIN_DEFAULTS['eb']})")
    ap.add_argument("--nm", type=int, default=figures.PIN_DEFAULTS["nm"],
                    help="number of measures the comparison figures pin to; "
                         f"falls back to the largest present (default {figures.PIN_DEFAULTS['nm']})")
    ap.add_argument("--strict", action="store_true",
                    help="fail if a declared figure's required cells are absent")
    ap.add_argument("--per-query-head", type=int, default=None,
                    help="cap the per-query figure to the first N queries "
                         "(default: the full run)")
    ap.add_argument("--legend", choices=["outside", "none"], default="outside",
                    help="per-figure legend: below the axes (default) or omitted "
                         "(a standalone legend.pdf is always written either way)")
    ap.add_argument("--title", choices=["on", "off"], default="on",
                    help="per-figure title with dataset/nm/eb context (default on); "
                         "turn off for bare figures captioned externally")
    args = ap.parse_args(argv)

    figures.PER_QUERY_HEAD = args.per_query_head
    render.LEGEND = args.legend
    render.TITLES = args.title == "on"

    frame = load.load_frame(
        a3i_config.results_root(args.results_root),
        datasets=[args.dataset] if args.dataset else None,
        workloads=[args.workload] if args.workload else None)

    if args.explore:
        explore(frame, Path(args.out_dir or "experiments/plots/explore"), args.eb)
    else:
        default = (f"experiments/plots/{args.plan}" if args.plan
                   else "experiments/plots/figures")
        render_figures(frame, Path(args.out_dir or default), args.eb, args.nm,
                       only=args.figure or None, strict=args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
