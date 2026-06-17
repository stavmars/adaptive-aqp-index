#!/usr/bin/env python3
"""Summarize experiment performance and check expected per-method behaviour.

An offline consumer of the `qresults`/`runmeta` files. It reports *behaviour and
performance* only -- answer correctness, error-vs-bound, and CI coverage are
owned by `validate_results.py`, which judges them per (measure, aggregate) over
replicate runs rather than pooled per cell.

For each `(dataset, workload)` it prints a per-method summary (init, cumulative
time, measure reads, median/p95 latency, speedup over scan) and runs a curated
set of behaviour/performance expectation checks. It works for a partial results
matrix: a check whose inputs are absent is skipped with a note.

Emits `summary.csv` and `findings.csv` under the analysis root.

Usage:
    scripts/analyze_results.py
    scripts/analyze_results.py --dataset synth10_100M
    scripts/analyze_results.py --results-root experiments/results --eb 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import a3i_config
from plotting import load, aggregate



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
    # The exact, non-aggregated methods must read exactly the qualifying values,
    # so their read counts coincide (the read-side of exact==scan).
    exact_full = [m for m in ("scan", "kd", "akd") if _row(sub, m) is not None]
    if len(exact_full) >= 2:
        vals = {m: g(m, "total_reads") for m in exact_full}
        add("scan = kd = akd (exact reads)",
            max(vals.values()) - min(vals.values()) <= 2,
            ", ".join(f"{m}={v:,.0f}" for m, v in vals.items()))
    if have("kd", "kd_agg"):
        add("kd_agg <= kd", g("kd_agg", "total_reads") <= g("kd", "total_reads"),
            f"{g('kd_agg','total_reads'):,.0f} vs {g('kd','total_reads'):,.0f}")
        add("eager kd_agg << kd",
            g("kd_agg", "total_reads") < 0.9 * g("kd", "total_reads"),
            f"{g('kd_agg','total_reads'):,.0f} vs {g('kd','total_reads'):,.0f}",
            severity="WARN")
    if have("akd", "akd_agg"):
        add("akd_agg <= akd (reuse)",
            g("akd_agg", "total_reads") <= g("akd", "total_reads"),
            f"{g('akd_agg','total_reads'):,.0f} vs {g('akd','total_reads'):,.0f}")
    # Sampling methods draw a subset of the qualifying rows, so they never read
    # more than the exact full-read methods (scan = kd = akd).
    full_reads = next((g(m, "total_reads") for m in ("kd", "akd", "scan")
                       if _row(sub, m) is not None), None)
    if full_reads is not None:
        for m in ("a3i_akd", "akd_sampling"):
            if _row(sub, m) is not None:
                add(f"{m} <= exact reads (sample within population)",
                    g(m, "total_reads") <= full_reads + 2,
                    f"{g(m,'total_reads'):,.0f} vs full {full_reads:,.0f}")
    # a3i vs the lazy-aggregate method on reads: a soft expectation that holds in
    # practice (a3i samples; akd_agg reads the rows of not-yet-summarised
    # partitions). Not asserted against kd_agg, which materialises every summary
    # via a full up-front build -- on low-selectivity / multi-measure cells a3i
    # routinely reads more, so there is no expected order; a3i's win over the
    # eager methods is in total time (below), not reads.
    if have("a3i_akd", "akd_agg"):
        add("a3i_akd <= akd_agg (reads)",
            g("a3i_akd", "total_reads") <= g("akd_agg", "total_reads"),
            f"{g('a3i_akd','total_reads'):,.0f} vs {g('akd_agg','total_reads'):,.0f}",
            severity="WARN")
    if have("a3i_akd", "akd_sampling"):
        # Reuse helps on overlapping workloads; on a non-overlapping (random) one
        # the two sampling paths can tie or a3i can read marginally more, so this
        # is a soft expectation, not a strict invariant.
        add("a3i_akd <= akd_sampling (reuse)",
            g("a3i_akd", "total_reads") <= g("akd_sampling", "total_reads") * 1.05,
            f"{g('a3i_akd','total_reads'):,.0f} vs {g('akd_sampling','total_reads'):,.0f}",
            severity="WARN")

    # --- total time (init-inclusive cum_ms; the headline AQP claim) ----------
    # a3i must beat a full scan end-to-end -- the premise of doing AQP at all.
    if have("a3i_akd", "scan"):
        add("a3i_akd beats scan", g("a3i_akd", "cum_ms") < g("scan", "cum_ms"),
            f"{g('a3i_akd','cum_ms')/1e3:.1f}s vs {g('scan','cum_ms')/1e3:.1f}s "
            f"({g('scan','cum_ms')/g('a3i_akd','cum_ms'):.0f}x)")
    # a3i should also be the fastest end-to-end among the index methods: its value
    # is total time at a bounded error, where the eager methods' build cost tells.
    # Soft (WARN), not an invariant -- a very large query count amortises a
    # one-time build, so a cheap-per-query method could in principle catch up.
    # Subsumes the old a3i <= akd_agg <= akd ladder and adds kd / kd_agg.
    a3i_cum = g("a3i_akd", "cum_ms")
    others = {m: g(m, "cum_ms") for m in
              ("kd", "kd_agg", "akd", "akd_agg", "akd_sampling")
              if _row(sub, m) is not None}
    if a3i_cum is not None and others:
        faster = {m: v for m, v in others.items() if v < a3i_cum}
        add("a3i_akd lowest total time", not faster,
            (f"a3i_akd {a3i_cum/1e3:.1f}s; faster: "
             + ", ".join(f"{m} {v/1e3:.1f}s" for m, v in faster.items())) if faster
            else f"a3i_akd {a3i_cum/1e3:.1f}s <= all "
                 f"(next {min(others.values())/1e3:.1f}s)",
            severity="WARN")

    # --- init split ----------------------------------------------------------
    if have("kd_agg"):
        add("kd_agg has build init", g("kd_agg", "init_ms") > 0,
            f"{g('kd_agg','init_ms')/1e3:.1f}s", severity="WARN")
    if have("a3i_akd"):
        add("a3i_akd ~no init", g("a3i_akd", "init_ms") < 100,
            f"{g('a3i_akd','init_ms'):.0f}ms", severity="WARN")

    # Answer correctness, error-vs-bound, and CI coverage are deliberately NOT
    # judged here. They are owned by validate_results.py, which checks them
    # per (measure, aggregate) over replicate runs -- the statistically sound
    # grain. Pooling them per cell (as a quick analyzer column would) lets a
    # well-behaved measure mask an under-covered one and treats a single run's
    # dependent queries as independent trials. This script judges behaviour and
    # performance only.

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize + sanity-check A3I results.")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--analysis-root", default=None)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--eb", type=float, default=None,
                    help="restrict to a single error bound; by default every eb "
                         "present is analysed (exact methods join each eb's slice)")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--strict-version", action="store_true",
                    help="fail (exit 1) if any compared slice mixes engine builds")
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

    # Compare methods only within a comparable slice: every axis fixed except the
    # method. Exact methods carry no eb (NaN) and join the slice at *every* eb;
    # the approximate methods are taken one eb at a time. So the analysis runs
    # once per (slice, eb) -- every error bound present is checked, no --eb needed
    # (pass --eb only to restrict to one).
    slice_keys = ["dataset", "workload", "nm", "mem", "partition_size", "n"]
    order = ["scan", "kd", "kd_agg", "akd", "akd_agg", "akd_sampling", "a3i_akd"]

    if args.eb is not None:
        ebs = [args.eb]
    else:
        ebs = sorted(frame.loc[frame["eb"].notna(), "eb"].unique().tolist()) or [0.01]

    all_versions = sorted(set(frame["engine_build_version"].dropna().tolist()))
    if len(all_versions) > 1:
        print(f"warning: results span multiple engine builds {all_versions}")

    summaries: list = []
    all_findings: list[dict] = []
    tot = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for eb in ebs:
        comparable = frame[frame["eb"].isna() | (frame["eb"] == eb)]
        summ = aggregate.cost_cell_summary(comparable)

        # Per-slice engine-build spread (a mixed slice is an unfair comparison).
        slice_versions = (comparable.groupby(slice_keys, dropna=False)
                          ["engine_build_version"]
                          .agg(lambda s: sorted(set(s.dropna()))))
        summaries.append(summ)

        for keys, grp in summ.groupby(slice_keys, dropna=False):
            ds, wl, nm, mem, psize, n = keys
            # An eb context is only meaningful where an approximate method ran at
            # it; an exact-only slice is already covered under its own eb.
            if not grp["method"].isin(("a3i_akd", "akd_sampling")).any():
                continue
            sub = grp.set_index("method").reindex(
                [m for m in order if m in grp["method"].values]).reset_index()
            print(f"\n=== {ds} / {wl}  (nm={nm}, eb={eb:g}, "
                  f"partition_size={psize}, mem={mem}, n={n}) ===")
            print(f"  {'method':14}{'init_s':>8}{'cum_s':>9}{'reads':>15}"
                  f"{'p50_ms':>9}{'p95_ms':>9}{'speedup':>8}{'scan%':>7}")
            for _, r in sub.iterrows():
                sp = "" if r["speedup_vs_scan"] != r["speedup_vs_scan"] else f"{r['speedup_vs_scan']:.0f}x"
                # scan% = share of queries served through the sequential-scan
                # path; blank for an in-memory cell.
                scf = r.get("scan_frac", float("nan"))
                sc = "" if scf != scf else f"{scf*100:.0f}%"
                print(f"  {r['method']:14}{r['init_ms']/1e3:8.1f}{r['cum_ms']/1e3:9.1f}"
                      f"{r['total_reads']:15,.0f}{r['lat_p50']:9.1f}{r['lat_p95']:9.1f}"
                      f"{sp:>8}{sc:>7}")
            fs = diagnostics(sub, eb, args.confidence, ds)
            vers = list(slice_versions.get(keys, []))
            if len(vers) > 1:
                fs.append({"check": "single engine version",
                           "status": "FAIL" if args.strict_version else "WARN",
                           "detail": f"slice mixes engine builds {vers}"})
            for f in fs:
                tot[f["status"]] += 1
                f.update(dataset=ds, workload=wl, nm=nm, mem=mem,
                         partition_size=psize, n=n, eb=eb)
                all_findings.append(f)
                if f["status"] != "PASS":
                    print(f"     [{f['status']}] {f['check']} "
                          f"(nm={nm}, eb={eb:g}): {f['detail']}")

    analysis_root.mkdir(parents=True, exist_ok=True)
    # Exact methods are evaluated in every eb's slice; keep one row per cell.
    summary = (pd.concat(summaries, ignore_index=True)
               .drop_duplicates(subset=slice_keys + ["method", "eb"]))
    # Lead with the cell identity, then the headline cost metrics, so the file
    # opens on what matters; any remaining columns keep their order at the end.
    lead = ["dataset", "workload", "method", "substrate", "nm", "mem",
            "partition_size", "n", "eb",
            "init_ms", "cum_ms", "total_latency_ms",
            "lat_p50", "lat_p95", "lat_p99", "total_reads", "speedup_vs_scan"]
    ordered = [c for c in lead if c in summary.columns]
    ordered += [c for c in summary.columns if c not in ordered]
    summary = summary[ordered]
    summary.to_csv(analysis_root / "summary.csv", index=False)

    # Read monotonicity in the error bound: an approximate method must not read
    # MORE at a looser eb than at a tighter one within the same cell -- a looser
    # accuracy target needs no more sampling. Such a violation is a WARN, not a FAIL:
    # the answer still meets eb (correctness is done in validate_results.py),
    # but extra reads at a looser target signal an inefficiency (e.g. a round-
    # budget give-up that reads a residual whole). Reads also depend on the
    # adaptive session (cracking/reuse) and the draw seed, so it is a warning,
    # not an invariant. Compared across eb, so it runs once here, not per slice.
    approx = summary[summary["method"].isin(("a3i_akd", "akd_sampling"))
                     & summary["eb"].notna()]
    for keys, grp in approx.groupby(slice_keys + ["method"], dropna=False):
        g = grp.sort_values("eb")  # ascending eb = tighter -> looser
        rows = list(g[["eb", "total_reads"]].itertuples(index=False))
        for tight, loose in zip(rows, rows[1:]):
            if loose.total_reads > tight.total_reads:
                ds, wl, nm, mem, psize, n, method = keys
                tot["WARN"] += 1
                all_findings.append({
                    "check": "reads monotone in eb", "status": "WARN",
                    "detail": (f"{method}: eb={loose.eb:g} reads "
                               f"{loose.total_reads:,.0f} > eb={tight.eb:g} reads "
                               f"{tight.total_reads:,.0f} (looser bound read more)"),
                    "dataset": ds, "workload": wl, "nm": nm, "mem": mem,
                    "partition_size": psize, "n": n, "eb": loose.eb})
                print(f"     [WARN] reads monotone in eb ({ds}/{wl} {method}): "
                      f"eb={loose.eb:g} {loose.total_reads:,.0f} > "
                      f"eb={tight.eb:g} {tight.total_reads:,.0f}")

    import csv as _csv
    with open(analysis_root / "findings.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["dataset", "workload", "nm", "mem",
                                            "partition_size", "n", "eb", "check", "status",
                                            "detail"])
        w.writeheader()
        w.writerows(all_findings)

    print(f"\n==== {tot['PASS']} pass, {tot['WARN']} warn, {tot['FAIL']} fail ====")
    print(f"wrote {analysis_root}/summary.csv and findings.csv")
    return 1 if tot["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
