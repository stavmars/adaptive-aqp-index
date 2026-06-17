#!/usr/bin/env python3
"""Partition-size (threshold) sweep: cost + accuracy tables and figures.

No aggregation of its own -- analyze_results.py already reduced runs->queries->
cell for cost, and validate_results.py reduced runs->oracle for accuracy. This
reshapes both along partition_size.

Cost (from summary.csv): a per-method sensitivity summary, the per-metric
relative tables (one workload per row, '*' = that row's best), and an absolute
magnitude table.

Accuracy (from validate_results.py output, if present): the oracle-validated
within-eb fraction, CI coverage, and worst relative error per partition_size,
plus a flatness check (across-size span vs across-run noise) and a per-measure
under-coverage callout. Answers whether the threshold is a pure cost knob.

With --figures it also writes the paper figures under the plots dir.

Usage:
    scripts/analyze_results.py                       # writes summary.csv
    scripts/validate_results.py                      # writes validation/*
    scripts/analyze_threshold.py                     # cost + accuracy tables
    scripts/analyze_threshold.py --metric cum_ms     # focus the relative tables
    scripts/analyze_threshold.py --figures           # + PDFs to experiments/plots/threshold/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# One cost curve over partition_size per row (workload ids are self-describing,
# so dataset is dropped from the label). mem keeps in-mem and on-disk separate.
GROUP = ["method", "workload", "eb", "mem"]
LEVEL = ["method", "eb", "mem"]
METRICS = ["total_reads", "cum_ms", "total_latency_ms", "lat_p50"]
# (metric, axis label) for the figures, in panel order.
FIG_METRICS = [("cum_ms", "cumulative time"),
               ("total_reads", "measure reads"),
               ("lat_p50", "p50 query latency")]
METHOD_ORDER = ["a3i", "adkd_agg", "kd_agg"]
MARKERS = {"a3i": "o", "adkd_agg": "s", "kd_agg": "^"}
SIZES = [128, 256, 512, 1024, 2048, 4096]
DEFAULT_SIZE = 1024

# Accuracy view: oracle-validated metrics per cell, from validate_results.py.
# The approximate method we tune is a3i; the exact methods carry no error.
VAL_DEFAULT = "experiments/validation"
ACC_METHOD = "a3i"
NOMINAL_COVERAGE = 0.95
LOW_COVERAGE = 0.93  # below this a per-measure cell is called out
# (column, label) for the accuracy tables/figure, in display order.
ACC_METRICS = [("approx_within_eb_frac", "within-eb fraction"),
               ("coverage_frac", "CI coverage"),
               ("max_rel_err", "worst rel. error")]


def load(summary: Path) -> pd.DataFrame:
    df = pd.read_csv(summary)
    df["eb"] = df["eb"].fillna("exact")          # exact methods carry no eb
    df = df.dropna(subset=["partition_size"]).copy()
    df["partition_size"] = df["partition_size"].astype(int)
    df = df[df.mem == "INMEM"]
    df = df[df.groupby(GROUP)["partition_size"].transform("nunique") > 1]  # swept only
    if df.empty:
        raise SystemExit("no swept cells (no row varies partition_size)")
    return df


def relative(df: pd.DataFrame, metric: str, index=None) -> pd.DataFrame:
    """Each row's curve over partition_size divided by its own minimum
    (1.00 = best for that row; units cancel, rows comparable)."""
    piv = df.dropna(subset=[metric]).pivot_table(
        index=index or GROUP, columns="partition_size", values=metric)
    return piv.div(piv.min(axis=1), axis=0)


# --------------------------------------------------------------------- tables

def print_tables(df: pd.DataFrame, metric_filter: str | None) -> None:
    print("Values are cost / that row's best:  1.00* = best,  1.20 = 20% worse.")
    print("Reading a row:  all ~1.00 => size barely matters (the '*' is noise);")
    print("  '*' at 128/4096 => no sweet spot;  '*' mid-range with a dip => real best.\n")

    rows = []
    for m in [x for x in METRICS if x in df.columns]:
        r = relative(df, m)
        mean = r.groupby(level="method").mean()
        spread = (r.max(axis=1) - 1).groupby(level="method").mean() * 100
        for meth in mean.index:
            rows.append(dict(method=meth, metric=m, best=int(mean.loc[meth].idxmin()),
                             spread_pct=round(float(spread.loc[meth]), 1)))
    print("=== per-method best size + sensitivity (spread%, mean over workloads) ===")
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    for m in ([metric_filter] if metric_filter else [x for x in METRICS if x in df.columns]):
        rel = relative(df, m)
        def fmt(r):
            b = r.idxmin()
            return pd.Series({c: f"{r[c]:.2f}" + ("*" if c == b else " ") for c in rel.columns})
        print(f"=== {m}: each size vs the row's best ===")
        print(rel.apply(fmt, axis=1).to_string(), "\n")

    # Absolute magnitude on one workload -- the a3i << kd_agg << adkd_agg ordering.
    t = df[df.workload == "taxi_random"]
    if len(t):
        print("=== absolute total_reads (millions), taxi_random ===")
        for meth, sub in [("a3i (eb=0.05)", t[(t.method == "a3i") & (t.eb == 0.05)]),
                          ("a3i (eb=0.01)", t[(t.method == "a3i") & (t.eb == 0.01)]),
                          ("kd_agg", t[t.method == "kd_agg"]),
                          ("adkd_agg", t[t.method == "adkd_agg"])]:
            if len(sub):
                s = sub.set_index("partition_size")["total_reads"].sort_index()
                print(f"  {meth:14}", {int(k): round(v / 1e6, 1) for k, v in s.items()})


# ------------------------------------------------------------------- accuracy

def load_validation(val_root: Path) -> pd.DataFrame:
    """Per-cell oracle-validated accuracy for the tuned method, swept cells only.
    Empty frame when validate_results.py has not been run."""
    files = sorted(val_root.glob("*/*/validation_summary.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.rename(columns={"error_bound": "eb"})
    df = df[df.method == ACC_METHOD].dropna(subset=["partition_size"]).copy()
    df["partition_size"] = df["partition_size"].astype(int)
    swept = df.groupby(["workload", "eb"])["partition_size"].transform("nunique") > 1
    return df[swept]


def load_coverage(val_root: Path) -> pd.DataFrame:
    """Per-(aggregate, measure) coverage for the tuned method -- the granularity
    that distinguishes an intrinsic (all-size) miss from a localized one."""
    files = sorted(val_root.glob("*/*/validation_coverage.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.rename(columns={"error_bound": "eb"})
    df = df[df.method == ACC_METHOD].dropna(subset=["partition_size"]).copy()
    df["partition_size"] = df["partition_size"].astype(int)
    return df


def print_accuracy(val: pd.DataFrame, cov: pd.DataFrame) -> None:
    print(f"\nAccuracy of {ACC_METHOD}, oracle-validated, mean over error bounds "
          f"and runs.\n  within-eb = share of queries with |est-truth|/|truth| "
          f"<= eb;\n  coverage  = share of CIs that bracket truth (nominal "
          f"{NOMINAL_COVERAGE:.2f});  worst rel. error = max over queries.\n")

    for metric, label in ACC_METRICS:
        piv = val.pivot_table(index="workload", columns="partition_size",
                              values=metric).reindex(columns=SIZES)
        print(f"=== {label} ({metric}): workload x partition_size ===")
        print(piv.round(4).to_string(), "\n")

    # Flatness check: is the variation over sizes within run-to-run noise? For
    # each (workload, eb) compare the span of the per-size run-means (size span)
    # to the largest within-size span over runs (run span). ratio < 1 => the
    # threshold moves accuracy less than re-running the same cell does.
    print("=== flatness check: across-size span vs across-run span ===")
    print("ratio < 1 => size variation is within run-to-run noise (a3i).\n")
    rows = []
    for metric, label in ACC_METRICS:
        g = val.groupby(["workload", "eb", "partition_size"])[metric]
        size_means = g.mean().groupby(level=["workload", "eb"])
        size_span = size_means.max() - size_means.min()
        run_span = (g.max() - g.min()).groupby(level=["workload", "eb"]).max()
        ratio = (size_span / run_span.replace(0, float("nan"))).mean()
        rows.append(dict(metric=metric, size_span=round(size_span.mean(), 4),
                         run_span=round(run_span.mean(), 4),
                         ratio=round(float(ratio), 2)))
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    # Per-measure under-coverage: which (workload, aggregate, measure) fall below
    # nominal, and at how many of the swept cells -- a high low/total ratio means
    # the miss is intrinsic to the measure (every size), a low one means it is
    # localized to a particular size.
    if len(cov):
        n_sizes = cov["partition_size"].nunique()
        g = cov.groupby(["workload", "aggregate", "measure"])
        summ = g.agg(min_cov=("mean_coverage", "min"),
                     mean_cov=("mean_coverage", "mean"),
                     cells_low=("mean_coverage", lambda s: int((s < LOW_COVERAGE).sum())),
                     cells=("mean_coverage", "size"))
        flagged = summ[summ.min_cov < LOW_COVERAGE].sort_values("min_cov")
        print(f"=== per-measure under-coverage (< {LOW_COVERAGE:.2f}); "
              f"{n_sizes} sizes x eb per group ===")
        if flagged.empty:
            print("  none\n")
        else:
            print("cells_low ~ cells => intrinsic (all sizes); "
                  "cells_low small => localized to a size.\n")
            print(flagged.round(3).to_string(), "\n")

        # Per-measure within-eb: the looser practical guarantee. We expect it
        # >= the confidence (every approximate answer should land within eb at
        # the nominal rate); measures below that flag a real (eb, confidence)
        # violation, the same heavy-tail effect that drives under-coverage.
        if "mean_within_eb" in cov.columns:
            gw = cov.dropna(subset=["mean_within_eb"]).groupby(
                ["workload", "aggregate", "measure"])
            wsumm = gw.agg(
                min_within=("mean_within_eb", "min"),
                mean_within=("mean_within_eb", "mean"),
                cells_low=("mean_within_eb", lambda s: int((s < NOMINAL_COVERAGE).sum())),
                cells=("mean_within_eb", "size"))
            wflag = wsumm[wsumm.min_within < NOMINAL_COVERAGE].sort_values("min_within")
            print(f"=== per-measure within-eb below confidence ({NOMINAL_COVERAGE:.2f}); "
                  f"{n_sizes} sizes x eb per group ===")
            if wflag.empty:
                print("  none\n")
            else:
                print(wflag.round(3).to_string(), "\n")


# -------------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 150})
    return plt


def fig_sensitivity(df: pd.DataFrame, out: Path) -> None:
    """Main figure: mean relative cost over workloads, one panel per metric."""
    plt = _style()
    fig, axes = plt.subplots(1, len(FIG_METRICS), figsize=(10.5, 3.0))
    for ax, (metric, label) in zip(axes, FIG_METRICS):
        mean = relative(df, metric).groupby(level="method").mean()
        for meth in METHOD_ORDER:
            if meth in mean.index:
                ax.plot(SIZES, mean.loc[meth].reindex(SIZES).values,
                        marker=MARKERS[meth], label=meth, linewidth=1.4, markersize=4)
        ax.set_xscale("log", base=2); ax.set_xticks(SIZES)
        ax.set_xticklabels([str(s) for s in SIZES], rotation=45)
        ax.set_title(label); ax.set_xlabel("partition size (rows/leaf)")
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax.axvline(DEFAULT_SIZE, color="crimson", lw=0.9, ls="--", alpha=0.7)
    axes[0].set_ylabel("cost / method's best\n(mean over workloads)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def fig_per_workload(df: pd.DataFrame, out: Path) -> None:
    """Per-workload grid: rows = workload, cols = metric, one line per method,
    each curve normalized to its own best within that workload."""
    plt = _style()
    workloads = sorted(df.workload.unique())
    nrows, ncols = len(workloads), len(FIG_METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0, 2.1 * nrows), sharex=True)
    for i, wl in enumerate(workloads):
        sub = df[df.workload == wl]
        for j, (metric, label) in enumerate(FIG_METRICS):
            ax = axes[i][j]
            rel = relative(sub, metric, index=["method", "eb"])  # per (method,eb) in this wl
            for meth in METHOD_ORDER:
                rows = rel[rel.index.get_level_values("method") == meth]
                if len(rows):
                    ax.plot(SIZES, rows.mean(axis=0).reindex(SIZES).values,
                            marker=MARKERS[meth], label=meth, linewidth=1.1, markersize=3)
            ax.set_xscale("log", base=2); ax.set_xticks(SIZES)
            ax.axhline(1.0, color="grey", lw=0.5, ls=":")
            ax.axvline(DEFAULT_SIZE, color="crimson", lw=0.8, ls="--", alpha=0.6)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(wl.replace("_", "\n"), fontsize=7)
            if i == nrows - 1:
                ax.set_xticklabels([str(s) for s in SIZES], rotation=45)
                ax.set_xlabel("partition size")
            else:
                ax.set_xticklabels([])
    axes[0][0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle("Cost vs partition size, per workload  (each curve / its own best)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99)); fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def fig_accuracy(val: pd.DataFrame, out: Path) -> None:
    """Accuracy vs partition size: absolute within-eb and coverage per workload,
    with the nominal-coverage reference. Unlike cost, accuracy is already a rate,
    so the curves are not normalized."""
    plt = _style()
    panels = [("approx_within_eb_frac", "within-eb fraction", (0.95, 1.005)),
              ("coverage_frac", "CI coverage", (0.80, 1.005))]
    fig, axes = plt.subplots(1, len(panels), figsize=(9.0, 3.2))
    workloads = sorted(val.workload.unique())
    colors = plt.cm.tab10.colors
    for ax, (metric, label, ylim) in zip(axes, panels):
        piv = val.pivot_table(index="workload", columns="partition_size",
                              values=metric).reindex(columns=SIZES)
        for i, wl in enumerate(workloads):
            ax.plot(SIZES, piv.loc[wl].values, marker="o", markersize=3,
                    linewidth=1.1, color=colors[i % 10], label=wl.replace("_", " "))
        ax.set_xscale("log", base=2); ax.set_xticks(SIZES)
        ax.set_xticklabels([str(s) for s in SIZES], rotation=45)
        ax.set_title(label); ax.set_xlabel("partition size (rows/leaf)")
        ax.set_ylim(*ylim)
        ax.axvline(DEFAULT_SIZE, color="crimson", lw=0.9, ls="--", alpha=0.7)
        if metric == "coverage_frac":
            ax.axhline(NOMINAL_COVERAGE, color="grey", lw=0.8, ls=":")
    axes[0].set_ylabel(f"{ACC_METHOD} (mean over eb, runs)")
    axes[-1].legend(frameon=False, fontsize=6, loc="lower left")
    fig.suptitle(f"{ACC_METHOD} accuracy vs partition size (oracle-validated)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Partition-size sweep tables + figures.")
    ap.add_argument("--summary", default="experiments/analysis/summary.csv")
    ap.add_argument("--metric", choices=METRICS, default=None,
                    help="restrict the per-metric relative tables to one metric")
    ap.add_argument("--figures", action="store_true",
                    help="also write the paper figures to --plots-dir")
    ap.add_argument("--plots-dir", default="experiments/plots/threshold")
    ap.add_argument("--validation-root", default=VAL_DEFAULT,
                    help="validate_results.py output root for the accuracy view")
    args = ap.parse_args()

    df = load(Path(args.summary))
    print_tables(df, args.metric)

    val = load_validation(Path(args.validation_root))
    cov = load_coverage(Path(args.validation_root))
    if len(val):
        print_accuracy(val, cov)
    else:
        print("\n(no swept accuracy cells under "
              f"{args.validation_root}; run validate_results.py for the "
              "accuracy view.)")

    if args.figures:
        out = Path(args.plots_dir); out.mkdir(parents=True, exist_ok=True)
        fig_sensitivity(df, out / "threshold_sensitivity.pdf")
        fig_per_workload(df, out / "threshold_per_workload.pdf")
        if len(val):
            fig_accuracy(val, out / "threshold_accuracy.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
