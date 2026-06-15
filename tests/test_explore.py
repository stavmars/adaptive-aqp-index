#!/usr/bin/env python3
"""Tests for the exploratory plot driver: axis-role inference and the ambiguity
guard. No engine; tiny in-memory frames. Run directly (python3
tests/test_explore.py) or via ctest (label unit). Skips itself if pandas/
matplotlib are unavailable. Covers:
  (i)   substrate co-varies with method (excluded) and null `eb` is ignored;
  (ii)  a single varied axis renders the standard views;
  (iii) two varied axes raise loudly, and the driver skips them gracefully.
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
    from plotting.load import COST_COLS
    _HAVE_DEPS = True
except ImportError:                  # pandas / matplotlib not installed
    _HAVE_DEPS = False


def _rows(method, substrate, nm, eb, exact):
    """A few queries x two aggregates of one cell, with cost + accuracy columns."""
    out = []
    for q in range(3):
        for agg, meas, est in (("SUM", "m0", 10.0 + q), ("COUNT_STAR", "*", 100.0)):
            row = dict(dataset="d", workload="w", method=method, substrate=substrate,
                       nm=nm, mem="inmem", partition_size=1024, n=3, eb=eb, run_id=0,
                       query_ordinal=q, aggregate=agg, measure=meas,
                       estimate=est, ci_low=est, ci_high=est, exact=exact,
                       init_ms=0.0)
            for c in COST_COLS:
                row[c] = 1.0 if c in ("latency_ms", "measure_reads") else 0
            out.append(row)
    return out


def _frame(specs):
    rows = []
    for spec in specs:
        rows += _rows(**spec)
    return pd.DataFrame(rows)


SCAN = dict(method="scan", substrate="n_a", nm=4, eb=float("nan"), exact=True)
A3I = dict(method="a3i", substrate="adaptive_kd", nm=4, eb=0.01, exact=False)


@unittest.skipUnless(_HAVE_DEPS, "pandas/matplotlib not available")
class Explore(unittest.TestCase):
    def test_varying_axes_excludes_substrate_and_null_eb(self):
        # method varies; substrate co-varies (excluded); eb is NaN for the exact
        # method so it must not count as varying.
        f = _frame([SCAN, A3I])
        self.assertEqual(pr.varying_axes(f), ["method"])

    def test_single_axis_renders(self):
        f = _frame([SCAN, A3I])
        with tempfile.TemporaryDirectory() as t:
            written = pr.explore_facet(f, Path(t), eb=0.01)
            self.assertTrue(written)
            for p in written:
                self.assertTrue(p.exists() and p.stat().st_size > 0)

    def test_two_axes_fail_loudly(self):
        # method AND nm both vary -> not safely comparable.
        f = _frame([SCAN, dict(SCAN, nm=1), A3I])
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(pr.AmbiguousComparison):
                pr.explore_facet(f, Path(t), eb=0.01)

    def test_driver_skips_ambiguous_gracefully(self):
        f = _frame([dict(SCAN, nm=1), A3I])  # method + nm vary
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(pr.explore(f, Path(t), eb=0.01), [])


if __name__ == "__main__":
    unittest.main()
