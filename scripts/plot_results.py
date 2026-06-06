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

Usage:
    scripts/plot_results.py --explore
    scripts/plot_results.py --explore --dataset synth10_100M --out-dir /tmp/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import a3i_config
from plotting import aggregate, load, render

# Axes a facet may legitimately vary on (besides the dataset/workload facet and
# the always-reduced run_id). `substrate` is intentionally excluded: it is bound
# to the method (each method has one substrate), so it co-varies with a method
# comparison rather than being an independent axis.
COMPARE_AXES = ("method", "nm", "str", "eb", "n")
# Hard-constraint axes: kept fixed unless a declared sweep varies them, never
# auto-varied in exploration.
HARD_AXES = ("mem",)


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
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Exploratory plots from the results matrix.")
    ap.add_argument("--explore", action="store_true",
                    help="render discovery-driven exploratory views (required)")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out-dir", default="experiments/plots/explore",
                    help="output directory (default experiments/plots/explore)")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--eb", type=float, default=0.01,
                    help="error bound for within-eb / accuracy summaries")
    args = ap.parse_args(argv)

    if not args.explore:
        ap.error("specify --explore")

    frame = load.load_frame(
        a3i_config.results_root(args.results_root),
        datasets=[args.dataset] if args.dataset else None,
        workloads=[args.workload] if args.workload else None)
    explore(frame, Path(args.out_dir), args.eb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
