#!/usr/bin/env python3
"""Tests for the declared figure manifest and its driver. No engine; tiny
in-memory frames. Run directly (python3 tests/test_figures.py) or via ctest
(label unit). Skips itself if pandas/matplotlib are unavailable. Covers:
  (i)   the manifest ids resolve via by_id;
  (ii)  the headline cumulative-time figure renders;
  (iii) a cold/mem-mismatched comparison raises;
  (iv)  a capped + uncapped comparison clips to the shared query prefix;
  (v)   a figure whose cells are absent skips by default and raises under strict.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

try:
    import pandas as pd
    import plot_results as pr
    from plotting import aggregate, figures
    from plotting.load import COST_COLS
    _HAVE_DEPS = True
except ImportError:                  # pandas / matplotlib not installed
    _HAVE_DEPS = False


def _rows(method, substrate, *, nm=4, eb=float("nan"), exact=True,
          mem="inmem", n=5, nq=5):
    out = []
    for q in range(nq):
        for agg, meas, est in (("SUM", "m0", 10.0 + q), ("COUNT_STAR", "*", 100.0)):
            row = dict(dataset="d", workload="w", method=method, substrate=substrate,
                       nm=nm, mem=mem, partition_size=1024, n=n, eb=eb, run_id=0,
                       outlier_budget=float("nan") if exact else 0.0,
                       query_ordinal=q, aggregate=agg, measure=meas, estimate=est,
                       ci_low=est, ci_high=est, exact=exact, init_ms=0.0, cold=True,
                       engine_build_version="v1", measure_storage="eager")
            for c in COST_COLS:
                row[c] = 1.0 if c in ("latency_ms", "measure_reads") else 0
            out.append(row)
    return out


def _frame(*chunks):
    rows = []
    for c in chunks:
        rows += c
    return pd.DataFrame(rows)


SCAN = dict(method="scan", substrate="n_a", exact=True, eb=float("nan"))
A3I = dict(method="a3i_akd", substrate="adaptive_kd", exact=False, eb=0.01)
ADKD = dict(method="akd", substrate="adaptive_kd", exact=True, eb=float("nan"))


@unittest.skipUnless(_HAVE_DEPS, "pandas/matplotlib not available")
class Figures(unittest.TestCase):
    def test_manifest_ids_resolve(self):
        self.assertTrue(figures.FIGURES)
        for fig in figures.FIGURES:
            self.assertTrue(fig.id)
        self.assertIs(figures.by_id("cumulative_time"),
                      next(f for f in figures.FIGURES if f.id == "cumulative_time"))
        with self.assertRaises(KeyError):
            figures.by_id("does_not_exist")

    def test_cumulative_time_renders(self):
        frame = _frame(_rows(**SCAN), _rows(**A3I), _rows(**ADKD))
        with tempfile.TemporaryDirectory() as t:
            written = pr.render_figures(frame, Path(t), 0.01, 4, only="cumulative_time")
            self.assertTrue(written)
            for p in written:
                self.assertTrue(p.exists() and p.stat().st_size > 0)

    def test_cold_mem_mismatch_raises(self):
        # one facet, two mem values -> not comparable.
        frame = _frame(_rows(**SCAN), _rows(**dict(A3I, mem="unbounded")))
        with self.assertRaises(ValueError):
            figures._method_slice(frame, 0.01, 4)

    def test_capped_and_uncapped_clip_to_shared_prefix(self):
        frame = _frame(_rows(**SCAN, n=5, nq=5),
                       _rows(**A3I, n=3, nq=3))
        clipped = aggregate.clip_to_shared_prefix(frame)
        self.assertEqual(int(clipped["query_ordinal"].max()), 2)

    def test_achieved_vs_targeted_quantiles_render(self):
        # Two eb cells of a3i with non-exact estimates around a truth of 100;
        # the figure draws the p50/p95 compliance lines and writes a file.
        a3i_01 = _rows(**dict(A3I, eb=0.01))
        a3i_05 = _rows(**dict(A3I, eb=0.05))
        frame = _frame(_rows(**SCAN), a3i_01, a3i_05)
        with tempfile.TemporaryDirectory() as t:
            written = pr.render_figures(frame, Path(t), 0.01, 4,
                                        only="achieved_vs_targeted_error")
            self.assertTrue(written)
            for p in written:
                self.assertTrue(p.exists() and p.stat().st_size > 0)

    def test_error_is_true_relative_and_zero_truth_excluded(self):
        # A sub-1 truth must divide by |truth|, not an absolute floor: truth
        # 0.2 with estimate 0.3 is a 50% error. A zero truth yields NaN (the
        # row is excluded from relative statistics, not scored absolutely).
        def one(method, measure, est, exact):
            row = dict(dataset="d", workload="w", method=method,
                       substrate="adaptive_kd" if method == "a3i_akd" else "n_a",
                       nm=4, mem="inmem", partition_size=1024, n=1, eb=0.01, run_id=0,
                       query_ordinal=0, aggregate="AVG", measure=measure,
                       estimate=est, ci_low=est, ci_high=est, exact=exact,
                       init_ms=0.0, cold=True, engine_build_version="v1",
                       measure_storage="eager")
            for c in COST_COLS:
                row[c] = 0
            return row
        frame = pd.DataFrame([
            one("scan", "small", 0.2, True), one("scan", "zero", 0.0, True),
            one("a3i_akd", "small", 0.3, False), one("a3i_akd", "zero", 0.1, False)])
        m = aggregate.with_error(frame).set_index(["method", "measure"])
        self.assertAlmostEqual(m.loc[("a3i_akd", "small"), "error"], 0.5)
        self.assertTrue(pd.isna(m.loc[("a3i_akd", "zero"), "error"]))

    def test_missing_cells_skip_then_strict(self):
        # single nm -> effect_of_nm has no sweep to draw.
        frame = _frame(_rows(**SCAN), _rows(**A3I))
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(
                pr.render_figures(frame, Path(t), 0.01, 4, only="effect_of_nm"), [])
            with self.assertRaises(figures.MissingCells):
                pr.render_figures(frame, Path(t), 0.01, 4, only="effect_of_nm",
                                  strict=True)


if __name__ == "__main__":
    unittest.main()
