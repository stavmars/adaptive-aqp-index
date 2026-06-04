#!/usr/bin/env python3
"""Summarize experiment results and check expected per-method behaviour.

An offline consumer of the `qresults`/`runmeta` files. It checks answer *correctness* against the oracle;
this checks *behaviour and performance* against expectations.

For each `(dataset, workload)` it prints a per-method summary (cumulative time,
init, latency percentiles, measure reads, speedup over scan, within-eb /
coverage) and runs a curated set of expectation checks. It works for a partial results matrix: a check whose inputs are absent
is skipped with a note.

Emits `summary.csv` and `findings.csv` under the analysis root.

Usage:
    scripts/analyze_results.py
    scripts/analyze_results.py --dataset synth10_100M
    scripts/analyze_results.py --results-root experiments/results --eb 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import a3i_config
from plotting import load, aggregate

EXACT_METHODS = ("kd", "kd_agg", "adkd", "adkd_agg")
EXACT_TOL = 1e-6          # exact methods must match the oracle this tightly
COVERAGE_SLACK = 0.07     # coverage may sit this far below nominal before WARN


def _row(sub, method):
    r = sub[sub["method"] == method]
    return None if r.empty else r.iloc[0]


def diagnostics(sub, eb, nominal, dataset) -> list[dict]:
    """Expectation checks for one (dataset, workload) summary slice."""
    findings: list[dict] = []

    def add(name, ok, detail, severity="FAIL"):
        findings.append({"check": name,
                         "status": "PASS" if ok else severity, "detail": detail})

    def g(method, col):
        r = _row(sub, method)
        return None if r is None else r[col]

    def have(*ms):
        return all(_row(sub, m) is not None for m in ms)

    # --- cross-method work ordering (deterministic for exact methods) ---------
    if have("scan"):
        worst = sub["total_reads"].max()
        add("reads <= scan", worst <= g("scan", "total_reads") + 2,
            f"max method reads {worst:,.0f} vs scan {g('scan','total_reads'):,.0f}")
    if have("kd", "kd_agg"):
        add("kd_agg <= kd", g("kd_agg", "total_reads") <= g("kd", "total_reads"),
            f"{g('kd_agg','total_reads'):,.0f} vs {g('kd','total_reads'):,.0f}")
        add("eager kd_agg << kd",
            g("kd_agg", "total_reads") < 0.9 * g("kd", "total_reads"),
            f"{g('kd_agg','total_reads'):,.0f} vs {g('kd','total_reads'):,.0f}",
            severity="WARN")
    if have("adkd", "adkd_agg"):
        add("adkd_agg <= adkd (reuse)",
            g("adkd_agg", "total_reads") <= g("adkd", "total_reads"),
            f"{g('adkd_agg','total_reads'):,.0f} vs {g('adkd','total_reads'):,.0f}")
    if have("a3i", "adkd_agg"):
        add("a3i <= adkd_agg (sampling)",
            g("a3i", "total_reads") <= g("adkd_agg", "total_reads"),
            f"{g('a3i','total_reads'):,.0f} vs {g('adkd_agg','total_reads'):,.0f}")
    if have("a3i", "adkd_sampling"):
        add("a3i <= adkd_sampling (reuse)",
            g("a3i", "total_reads") <= g("adkd_sampling", "total_reads"),
            f"{g('a3i','total_reads'):,.0f} vs {g('adkd_sampling','total_reads'):,.0f}")

    # --- cumulative-time ladder (latency carries noise -> WARN) ---------------
    if have("a3i", "adkd_agg", "adkd"):
        add("cum a3i <= adkd_agg <= adkd",
            g("a3i", "cum_ms") <= g("adkd_agg", "cum_ms") <= g("adkd", "cum_ms"),
            f"{g('a3i','cum_ms')/1e3:.1f} <= {g('adkd_agg','cum_ms')/1e3:.1f} "
            f"<= {g('adkd','cum_ms')/1e3:.1f}s", severity="WARN")
    if have("a3i", "scan"):
        add("a3i beats scan", g("a3i", "cum_ms") < g("scan", "cum_ms"),
            f"{g('a3i','cum_ms')/1e3:.1f}s vs {g('scan','cum_ms')/1e3:.1f}s "
            f"({g('scan','cum_ms')/g('a3i','cum_ms'):.0f}x)")

    # --- init split ----------------------------------------------------------
    if have("kd_agg"):
        add("kd_agg has build init", g("kd_agg", "init_ms") > 0,
            f"{g('kd_agg','init_ms')/1e3:.1f}s", severity="WARN")
    if have("a3i"):
        add("a3i ~no init", g("a3i", "init_ms") < 100,
            f"{g('a3i','init_ms'):.0f}ms", severity="WARN")

    # --- exact methods match the oracle --------------------------------------
    for m in EXACT_METHODS:
        if have(m) and g(m, "max_relerr") is not None:
            add(f"{m} == scan (exact)", g(m, "max_relerr") < EXACT_TOL,
                f"max_relerr={g(m,'max_relerr'):.1e}")

    # --- accuracy ------------------------------------------------------------
    for m in ("adkd_sampling", "a3i"):
        cov = g(m, "coverage")
        if have(m) and cov is not None and cov == cov:  # not NaN
            add(f"{m} coverage >= ~nominal", cov >= nominal - COVERAGE_SLACK,
                f"coverage={cov:.0%} (nominal {nominal:.0%})", severity="WARN")
    if have("a3i", "adkd_sampling"):
        ca, cs = g("a3i", "coverage"), g("adkd_sampling", "coverage")
        if ca == ca and cs == cs:
            add("a3i ~ adkd_sampling coverage parity", abs(ca - cs) < 0.1,
                f"{ca:.0%} vs {cs:.0%}", severity="WARN")

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize + sanity-check A3I results.")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--analysis-root", default=None)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--eb", type=float, default=0.01, help="error bound for within-eb")
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    results_root = a3i_config.results_root(args.results_root)
    analysis_root = (Path(args.analysis_root) if args.analysis_root
                     else a3i_config.REPO_ROOT / "experiments" / "analysis")

    frame = load.load_frame(results_root,
                            datasets=[args.dataset] if args.dataset else None,
                            workloads=[args.workload] if args.workload else None)
    if frame.empty:
        print(f"no results under {results_root}")
        return 0

    summary = aggregate.cell_summary(frame, args.eb)
    analysis_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis_root / "summary.csv", index=False)

    tot = {"PASS": 0, "WARN": 0, "FAIL": 0}
    all_findings: list[dict] = []
    order = ["scan", "kd", "kd_agg", "adkd", "adkd_agg", "adkd_sampling", "a3i"]
    for (ds, wl), sub in summary.groupby(["dataset", "workload"], dropna=False):
        sub = sub.set_index("method").reindex(
            [m for m in order if m in sub["method"].values]).reset_index()
        print(f"\n=== {ds} / {wl} ===")
        print(f"  {'method':14}{'init_s':>8}{'cum_s':>9}{'reads':>15}"
              f"{'p50_ms':>9}{'within_eb':>10}{'cov':>6}{'speedup':>8}")
        for _, r in sub.iterrows():
            cov = "" if r["coverage"] != r["coverage"] else f"{r['coverage']:.0%}"
            we = "" if r["within_eb"] != r["within_eb"] else f"{r['within_eb']:.0%}"
            sp = "" if r["speedup_vs_scan"] != r["speedup_vs_scan"] else f"{r['speedup_vs_scan']:.0f}x"
            print(f"  {r['method']:14}{r['init_ms']/1e3:8.1f}{r['cum_ms']/1e3:9.1f}"
                  f"{r['total_reads']:15,.0f}{r['lat_p50']:9.1f}{we:>10}{cov:>6}{sp:>8}")
        fs = diagnostics(sub, args.eb, args.confidence, ds)
        for f in fs:
            tot[f["status"]] += 1
            f.update(dataset=ds, workload=wl)
            all_findings.append(f)
            if f["status"] != "PASS":
                print(f"     [{f['status']}] {f['check']}: {f['detail']}")

    import csv as _csv
    with open(analysis_root / "findings.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["dataset", "workload", "check",
                                            "status", "detail"])
        w.writeheader()
        w.writerows(all_findings)

    print(f"\n==== {tot['PASS']} pass, {tot['WARN']} warn, {tot['FAIL']} fail ====")
    print(f"wrote {analysis_root}/summary.csv and findings.csv")
    return 1 if tot["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
