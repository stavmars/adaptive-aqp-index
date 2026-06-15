#!/usr/bin/env python3
"""Unit tests for the analysis core (plotting/load + plotting/aggregate) and the
analyze_results diagnostics. No engine, no matplotlib, tiny in-process fixtures.

Run directly (python3 tests/test_analysis.py) or via ctest (label unit). Covers:
  (i)  load rejects malformed / short / duplicate input loudly;
  (ii) the cost-collapse rule + reductions + the oracle join are correct;
  (iii) diagnostics fire on a planted-bad fixture and pass on a good one;
        the tools run on a partial matrix without crashing.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

try:
    from plotting import load, aggregate          # noqa: E402
    from plotting.load import LoadError, EXPECTED_HEADER  # noqa: E402
    import analyze_results                          # noqa: E402
    _HAVE_DEPS = True
except ImportError:                                 # pandas not installed
    _HAVE_DEPS = False
    EXPECTED_HEADER = []


def _aggs(measures, est, exact=True, ci=None, count_star=100.0):
    out = []
    for m in measures:
        for agg in ("SUM", "COUNT", "AVG"):
            e = est(agg, m)
            lo, hi = (None, None) if ci is None else ci(agg, m, e)
            out.append({"aggregate": agg, "measure": m, "estimate": e,
                        "ci_low": lo, "ci_high": hi,
                        "relative_half_width": 0.0, "effective_df": 0.0,
                        "exact": exact})
    out.append({"aggregate": "COUNT_STAR", "measure": "*", "estimate": count_star,
                "ci_low": count_star, "ci_high": count_star,
                "relative_half_width": 0.0, "effective_df": 0.0, "exact": True})
    return out


def write_cell(root, dataset, workload, substrate, method, key, run, n,
               aggs_for_q, latency=1.0, reads=10, init_ms=0.0,
               header=None, ordinals=None, version="test"):
    d = Path(root) / dataset / workload / substrate / method
    d.mkdir(parents=True, exist_ok=True)
    qf = d / f"qresults_{key}_run{run}.csv"
    hdr = header if header is not None else EXPECTED_HEADER
    ords = ordinals if ordinals is not None else range(n)
    with open(qf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for q in ords:
            rowd = {c: 0 for c in EXPECTED_HEADER}
            rowd.update(query_ordinal=q, method=method, substrate=substrate,
                        dataset=dataset, workload=workload,
                        query_rect='{"lower":[0,0],"upper":[1,1]}',
                        aggregates=json.dumps(aggs_for_q(q)),
                        target_satisfied="true", status="exact",
                        exactify_cause="none", pre_exactification_error_bound=0.0,
                        sampling_seed=run, latency_ms=latency, measure_reads=reads)
            w.writerow([rowd[c] for c in EXPECTED_HEADER])
    meta = dict(engine_build_version=version, cold=True, max_queries=n, run_id=run,
                sampling_seed=run, confidence=0.95, num_measures=len(("m0", "m1")),
                sort_gather_by_row_id=True, init_ms=init_ms, measure_storage="eager")
    (d / f"runmeta_{key}_run{run}.json").write_text(json.dumps(meta))
    return qf


EXACT3 = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.0, exact=True)   # noqa: E731
KEY = "mcols2_memINMEM_n3_ps1024"
KEYE = "err0.01_mcols2_memINMEM_n3_ps1024"


@unittest.skipUnless(_HAVE_DEPS, "pandas not available")
class TestLoad(unittest.TestCase):
    def test_valid_loads(self):
        with tempfile.TemporaryDirectory() as t:
            write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3)
            fr = load.load_frame(t)
            self.assertFalse(fr.empty)
            # 3 queries x 7 aggregate rows
            self.assertEqual(len(fr), 21)
            self.assertEqual(set(fr["method"]), {"scan"})
            self.assertEqual(fr["nm"].iloc[0], 2)
            self.assertEqual(fr["n"].iloc[0], 3)

    def test_bad_header_raises(self):
        with tempfile.TemporaryDirectory() as t:
            write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3,
                       header=EXPECTED_HEADER[:-1])  # drop a column
            with self.assertRaises(LoadError):
                load.load_frame(t)

    def test_short_noncontiguous_raises(self):
        with tempfile.TemporaryDirectory() as t:
            write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3,
                       ordinals=[0, 1, 3])  # gap -> not one-per-row contiguous
            with self.assertRaises(LoadError):
                load.load_frame(t)

    def test_missing_runmeta_stamp_raises(self):
        with tempfile.TemporaryDirectory() as t:
            qf = write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3)
            mp = qf.with_name(qf.name.replace("qresults_", "runmeta_")
                              .replace(".csv", ".json"))
            meta = json.loads(mp.read_text()); del meta["cold"]
            mp.write_text(json.dumps(meta))
            with self.assertRaises(LoadError):
                load.load_frame(t)

    def test_duplicate_cell_raises(self):
        with tempfile.TemporaryDirectory() as t:
            write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3)
            # permuted key tokens -> same parsed identity, different filename
            write_cell(t, "ds", "wl", "n_a", "scan",
                       "ps1024_mcols2_n3_memINMEM", 0, 3, EXACT3)
            with self.assertRaises(LoadError):
                load.load_frame(t)


@unittest.skipUnless(_HAVE_DEPS, "pandas not available")
class TestAggregate(unittest.TestCase):
    def test_cost_collapse(self):
        # 3 queries, each exploded to 7 aggregate rows, latency 5 each.
        with tempfile.TemporaryDirectory() as t:
            write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3,
                       latency=5.0, reads=10)
            fr = load.load_frame(t)
            pqc = aggregate.per_query_cost(fr)
            self.assertEqual(len(pqc), 3)               # one row per query, not 21
            cs = aggregate.cost_summary(fr)
            # total latency must be 3*5 = 15, NOT 21*5 = 105
            self.assertAlmostEqual(cs.iloc[0]["total_latency_ms"], 15.0)
            self.assertEqual(cs.iloc[0]["total_reads"], 30)  # 3*10, not 21*10

    def test_oracle_join_error_and_coverage(self):
        with tempfile.TemporaryDirectory() as t:
            # oracle: every aggregate truth = 100
            write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3)
            # approx a3i: estimate 110, interval [95,115] (covers 100), not exact
            approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 110.0, exact=False,  # noqa: E731
                                     ci=lambda a, m, e: (95.0, 115.0))
            write_cell(t, "ds", "wl", "adaptive_kd", "a3i", KEYE, 0, 3, approx)
            fr = load.load_frame(t)
            m = aggregate.with_error(fr)
            a = m[m["method"] == "a3i"]
            # relative error |110-100|/100 = 0.1 on the SUM/COUNT/AVG rows
            sub = a[a["aggregate"] != "COUNT_STAR"]
            self.assertTrue(((sub["error"] - 0.1).abs() < 1e-9).all())
            # covered: 95 <= 100 <= 115 -> True on every sampled row
            self.assertTrue(sub["covered"].dropna().all())
            self.assertEqual(sub["covered"].notna().sum(), len(sub))


def _good_matrix(t):
    """A healthy (dataset, workload) where every expected relationship holds."""
    # reads: scan high, kd full, kd_agg << kd, adkd full, adkd_agg < adkd,
    #        a3i < adkd_agg and < adkd_sampling. latency mirrors reads.
    cfg = {  # method: (substrate, key, reads, latency, init_ms, exact, est, ci)
        "scan":          ("n_a",         KEY,  1000, 10.0, 0.0,   True),
        "kd":            ("static_kd",   KEY,  1000, 8.0,  500.0, True),
        "kd_agg":        ("static_kd",   KEY,  50,   1.0,  700.0, True),
        "adkd":          ("adaptive_kd", KEY,  1000, 8.0,  0.0,   True),
        "adkd_agg":      ("adaptive_kd", KEY,  150,  2.0,  0.0,   True),
    }
    for m, (sub, key, reads, lat, init, _ex) in cfg.items():
        write_cell(t, "ds", "wl", sub, m, key, 0, 3, EXACT3,
                   latency=lat, reads=reads, init_ms=init)
    # sampling methods: approximate, covering intervals, fewer reads
    approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.5, exact=False,   # noqa: E731
                             ci=lambda a, m, e: (95.0, 105.0))
    write_cell(t, "ds", "wl", "adaptive_kd", "adkd_sampling", KEYE, 0, 3, approx,
               latency=3.0, reads=120, init_ms=0.0)
    write_cell(t, "ds", "wl", "adaptive_kd", "a3i", KEYE, 0, 3, approx,
               latency=1.5, reads=60, init_ms=0.0)


@unittest.skipUnless(_HAVE_DEPS, "pandas not available")
class TestDiagnostics(unittest.TestCase):
    def _findings(self, t):
        fr = load.load_frame(t)
        summ = aggregate.cell_summary(fr, eb=0.01)
        sub = summ[(summ["dataset"] == "ds") & (summ["workload"] == "wl")]
        return analyze_results.diagnostics(sub, eb=0.01, nominal=0.95, dataset="ds")

    def test_good_matrix_no_fail(self):
        with tempfile.TemporaryDirectory() as t:
            _good_matrix(t)
            fails = [f for f in self._findings(t) if f["status"] == "FAIL"]
            self.assertEqual(fails, [], f"unexpected FAILs: {fails}")

    def test_reads_above_agg_warns(self):
        with tempfile.TemporaryDirectory() as t:
            _good_matrix(t)
            # a3i reads MORE than the aggregate methods: no guaranteed order
            # (a3i samples; kd_agg/adkd_agg answer from summaries), so a WARN, not
            # a FAIL. Still below the exact full-read methods (1000), so the
            # sample-within-population check stays green.
            import shutil
            shutil.rmtree(Path(t) / "ds/wl/adaptive_kd/a3i")
            approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.5, exact=False,  # noqa: E731
                                     ci=lambda a, m, e: (95.0, 105.0))
            write_cell(t, "ds", "wl", "adaptive_kd", "a3i", KEYE, 0, 3, approx,
                       latency=1.5, reads=500, init_ms=0.0)  # > adkd_agg (150)
            names = {f["check"]: f["status"] for f in self._findings(t)}
            self.assertEqual(names.get("a3i <= adkd_agg (reads)"), "WARN")
            # kd_agg has no expected read order vs a3i, so it is not checked.
            self.assertIsNone(names.get("a3i <= kd_agg (reads)"))
            self.assertEqual(
                names.get("a3i <= exact reads (sample within population)"), "PASS")

    def test_sampling_exceeds_population_fires(self):
        with tempfile.TemporaryDirectory() as t:
            _good_matrix(t)
            # a sample cannot exceed the qualifying population (the exact reads).
            import shutil
            shutil.rmtree(Path(t) / "ds/wl/adaptive_kd/a3i")
            approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.5, exact=False,  # noqa: E731
                                     ci=lambda a, m, e: (95.0, 105.0))
            write_cell(t, "ds", "wl", "adaptive_kd", "a3i", KEYE, 0, 3, approx,
                       latency=1.5, reads=5000, init_ms=0.0)  # > full reads (1000)
            names = {f["check"]: f["status"] for f in self._findings(t)}
            self.assertEqual(
                names.get("a3i <= exact reads (sample within population)"), "FAIL")

    def test_exact_reads_disagree_fires(self):
        with tempfile.TemporaryDirectory() as t:
            _good_matrix(t)
            # the exact non-agg methods must read the same qualifying set.
            import shutil
            shutil.rmtree(Path(t) / "ds/wl/static_kd/kd")
            write_cell(t, "ds", "wl", "static_kd", "kd", KEY, 0, 3, EXACT3,
                       latency=8.0, reads=900, init_ms=500.0)  # != scan/adkd (1000)
            names = {f["check"]: f["status"] for f in self._findings(t)}
            self.assertEqual(names.get("scan = kd = adkd (exact reads)"), "FAIL")

    def test_a3i_not_fastest_warns(self):
        with tempfile.TemporaryDirectory() as t:
            _good_matrix(t)
            # a3i slower end-to-end than an index method (but still < scan): the
            # total-time lead is a soft expectation, so a WARN, not a FAIL.
            import shutil
            shutil.rmtree(Path(t) / "ds/wl/adaptive_kd/a3i")
            approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.5, exact=False,  # noqa: E731
                                     ci=lambda a, m, e: (95.0, 105.0))
            write_cell(t, "ds", "wl", "adaptive_kd", "a3i", KEYE, 0, 3, approx,
                       latency=3.0, reads=60, init_ms=0.0)  # cum 9s > adkd_agg 6s, < scan 30s
            names = {f["check"]: f["status"] for f in self._findings(t)}
            self.assertEqual(names.get("a3i lowest total time"), "WARN")
            self.assertEqual(names.get("a3i beats scan"), "PASS")

    def test_partial_matrix_no_crash(self):
        # only scan + a3i present -> pair checks needing other methods are skipped,
        # nothing raises.
        with tempfile.TemporaryDirectory() as t:
            write_cell(t, "ds", "wl", "n_a", "scan", KEY, 0, 3, EXACT3,
                       latency=10.0, reads=1000)
            approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.5, exact=False,  # noqa: E731
                                     ci=lambda a, m, e: (95.0, 105.0))
            write_cell(t, "ds", "wl", "adaptive_kd", "a3i", KEYE, 0, 3, approx,
                       latency=1.5, reads=60)
            findings = self._findings(t)
            self.assertTrue(any(f["check"] == "a3i beats scan" for f in findings))


@unittest.skipUnless(_HAVE_DEPS, "pandas not available")
class TestMultiNm(unittest.TestCase):
    def test_multiple_nm_per_workload_no_crash(self):
        # A (dataset, workload) with two nm values has the same method twice;
        # comparison must slice on nm, not collide methods. (Regression: a
        # groupby(dataset,workload)+reindex(method) crashed on duplicate labels.)
        approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.5, exact=False,  # noqa: E731
                                 ci=lambda a, m, e: (95.0, 105.0))
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as a:
            for k, ek in (("mcols1_memINMEM_n3_ps1024", "err0.01_mcols1_memINMEM_n3_ps1024"),
                          ("mcols2_memINMEM_n3_ps1024", "err0.01_mcols2_memINMEM_n3_ps1024")):
                write_cell(t, "ds", "wl", "n_a", "scan", k, 0, 3, EXACT3,
                           latency=10.0, reads=1000)
                write_cell(t, "ds", "wl", "adaptive_kd", "a3i", ek, 0, 3, approx,
                           latency=1.5, reads=60)
            argv = sys.argv
            sys.argv = ["analyze_results.py", "--results-root", t,
                        "--analysis-root", a]
            try:
                rc = analyze_results.main()      # pre-fix: raised ValueError
            finally:
                sys.argv = argv
            self.assertIn(rc, (0, 1))
            self.assertTrue((Path(a) / "summary.csv").is_file())

    def test_mixed_engine_versions_flagged(self):
        # Two cells in one comparable slice stamped with different engine builds:
        # a WARN by default (exit 0), a FAIL under --strict-version (exit 1).
        approx = lambda q: _aggs(["m0", "m1"], lambda a, m: 100.5, exact=False,  # noqa: E731
                                 ci=lambda a, m, e: (95.0, 105.0))
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as a:
            write_cell(t, "ds", "wl", "n_a", "scan", "mcols1_memINMEM_n3_ps1024",
                       0, 3, EXACT3, latency=10.0, reads=1000, version="0.1.0")
            write_cell(t, "ds", "wl", "adaptive_kd", "a3i",
                       "err0.01_mcols1_memINMEM_n3_ps1024", 0, 3, approx,
                       latency=1.5, reads=60, version="0.2.0")
            argv = sys.argv
            try:
                sys.argv = ["analyze_results.py", "--results-root", t,
                            "--analysis-root", a]
                self.assertEqual(analyze_results.main(), 0)
                self.assertIn("single engine version",
                              (Path(a) / "findings.csv").read_text())
                sys.argv = ["analyze_results.py", "--results-root", t,
                            "--analysis-root", a, "--strict-version"]
                self.assertEqual(analyze_results.main(), 1)
            finally:
                sys.argv = argv


if __name__ == "__main__":
    unittest.main(verbosity=2)
