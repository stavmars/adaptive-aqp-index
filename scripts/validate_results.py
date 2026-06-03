#!/usr/bin/env python3
"""Validate experiment results against the scan oracle.

A pure offline consumer of the files ``run_experiments.py`` produced: it never
invokes the engine and never regenerates ground truth. For each
``(dataset, workload)`` it treats the ``scan`` run's results as the exact oracle
and checks every other run against it:

  * **Exact aggregates** (a row's ``exact`` flag is true) must equal the oracle
    to tolerance: COUNT / COUNT_STAR to the bit, SUM / AVG within
    ``1e-6 * max(1, |oracle|)``, NaN matched by NaN. A mismatch is a hard
    failure (a correctness regression).
  * **Approximate aggregates** are scored: the empirical relative error vs the
    run's error bound and per-aggregate CI coverage (whether the oracle falls
    inside ``[ci_low, ci_high]``) are reported per row. Coverage is then *judged*
    per ``(measure, aggregate)`` -- never pooled across measures or aggregates,
    whose true coverage probabilities legitimately differ (a heavy-tail measure
    under-covers; a well-behaved one sits at nominal) -- and on the
    **replicate-run axis**: each run of a cell contributes one workload-average
    coverage for that ``(measure, aggregate)``, and the judgement is the mean
    over the R runs with a one-sided Student-t upper bound (R-1 df). The runs are
    the independent trials; the queries within a run are not (shared evolving
    index, heterogeneous rectangles), so a pooled ``Binomial`` over all
    ``query x run`` flags would understate the spread. When ``R == 1`` there is
    no over-run interval, so coverage falls back to a coarse single-run binomial
    trip-wire (``coverage_status`` is reported, never escalated). A
    ``(measure, aggregate)`` whose upper bound falls below nominal confidence is
    flagged ``low``: a warning by default, escalated to a failure by
    ``--strict-coverage`` (replicate-run basis only).

Every cell is always validated; failures and coverage warnings are collected and
reported at the end, never short-circuiting the run.

A same-query guard asserts that every compared run executed the *same* rectangle
per ``query_ordinal`` by comparing the ``query_rect`` column directly (not the
workload fingerprint). The oracle is required to exist and be complete (one row
per contiguous ``query_ordinal``); a missing or short oracle fails loudly.

Emits, per ``(dataset, workload)``:
  experiments/validation/<dataset>/<workload>/validation_summary.csv
  experiments/validation/<dataset>/<workload>/validation_details.csv

Usage:
    scripts/validate_results.py
    scripts/validate_results.py --results-root experiments/results
    scripts/validate_results.py --dataset synth10_1M --workload synth10_1M_clustered

Exits non-zero if any exact-equality check, same-query guard, or oracle
completeness check fails (and, under ``--strict-coverage``, on a coverage
shortfall).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import a3i_config

REPO_ROOT = Path(__file__).resolve().parent.parent

EXACT_TO_BIT = {"COUNT", "COUNT_STAR"}   # integer counts must match exactly
SUM_AVG_TOL = 1e-6                        # relative tolerance for SUM / AVG
BRACKET_TOL = 1e-9                        # relative slack for ci_low <= est <= ci_high

DEFAULT_COVERAGE_ALPHA = 0.01            # one-sided test level for CI coverage
MIN_COVERAGE_SAMPLES = 100              # below this, coverage is reported, not judged

# A query that finished sampling without exactifying converged; one that read its
# residual in full is exactified (or exact on the forced-exact path). All three
# imply the accuracy target was met. exhausted_unconverged means the residual was
# read in full yet the check still failed -- a contradiction for a correct
# implementation -- so it is a hard failure, never a statistical warning.
TARGET_MET_STATUS = {"exact", "converged", "exactified"}
EXACT_VALUED_STATUS = {"exact", "exactified"}

# runmeta stamps a result file must carry to be self-describing and traceable.
# (The memory cap is applied by the orchestrator's cgroup launcher wrapper, not
# recorded in the engine-written sidecar, so it is not among these.)
REQUIRED_RUNMETA_STAMPS = ("engine_build_version", "cold", "run_id",
                           "sampling_seed", "confidence", "num_measures",
                           "max_queries", "sort_gather_by_row_id")


# --- Student-t one-sided critical value (pure stdlib) ------------------------
# The replicate-run coverage judgement needs t_{1-alpha, R-1}. Python's stdlib
# has no t distribution, so we evaluate the t CDF via the regularized incomplete
# beta function (Lentz continued fraction) and invert it by bisection. This is
# the standard relation t_cdf(t; df) = 1 - 0.5 * I_x(df/2, 1/2) with
# x = df / (df + t^2) for t >= 0.

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    ib = _betai(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t >= 0.0 else 0.5 * ib


def student_t_ppf(p: float, df: float) -> float:
    """One-sided quantile: t such that student_t_cdf(t, df) == p."""
    lo, hi = -1.0e7, 1.0e7
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def coverage_status(covered: int, n: int, nominal: float, alpha: float) -> str:
    """Judge observed CI coverage against the nominal confidence.

    Each genuinely-sampled aggregate is one Bernoulli trial whose success
    probability equals the nominal confidence when the interval is well
    calibrated, so ``covered ~ Binomial(n, nominal)`` and the expected coverage
    fraction is the nominal confidence itself. We never flag a single uncovered
    interval; we flag only a *systemic* shortfall, judged by a one-sided
    lower-tail test (normal approximation with a continuity correction) so the
    decision accounts for binomial sampling noise rather than demanding the point
    estimate clear a fixed cutoff. Returns "n/a" (no sampled aggregates),
    "insufficient" (too few trials to judge), "ok", or "low"."""
    if n == 0:
        return "n/a"
    if n < MIN_COVERAGE_SAMPLES:
        return "insufficient"
    sd = math.sqrt(n * nominal * (1.0 - nominal))
    if sd == 0.0:
        return "ok"
    z = statistics.NormalDist().inv_cdf(1.0 - alpha)
    z_obs = (covered + 0.5 - n * nominal) / sd    # continuity-corrected
    return "low" if z_obs < -z else "ok"


def as_float(v) -> float:
    """A JSON estimate/bound -> float, with null and JSON-encoded NaN -> nan."""
    if v is None:
        return math.nan
    return float(v)


def both_nan(a: float, b: float) -> bool:
    return math.isnan(a) and math.isnan(b)


# --- reading one results file ------------------------------------------------

def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def explode(row: dict) -> tuple[int, list, list[dict]]:
    """A results row -> (query_ordinal, query_rect-as-[lower,upper], aggregates)."""
    ordinal = int(row["query_ordinal"])
    rect = json.loads(row["query_rect"])
    rect_key = [rect["lower"], rect["upper"]]
    aggregates = json.loads(row["aggregates"])
    return ordinal, rect_key, aggregates


def agg_key(entry: dict) -> tuple[str, str]:
    return (entry["aggregate"], entry["measure"])


# --- oracle ------------------------------------------------------------------

class Oracle:
    def __init__(self, csv_path: Path):
        self.path = csv_path
        self.by_ordinal: dict[int, dict] = {}   # ordinal -> {(agg,mea): estimate}
        self.rect: dict[int, list] = {}
        rows = read_rows(csv_path)
        for row in rows:
            ordinal, rect_key, aggregates = explode(row)
            self.rect[ordinal] = rect_key
            self.by_ordinal[ordinal] = {agg_key(e): as_float(e["estimate"])
                                        for e in aggregates}
        self.count = len(self.by_ordinal)

    def check_complete(self) -> None:
        expected = set(range(self.count))
        if set(self.by_ordinal) != expected:
            sys.exit(f"oracle {self.path} is incomplete: expected contiguous "
                     f"query_ordinal 0..{self.count - 1}, got "
                     f"{sorted(self.by_ordinal)}")


def runmeta_for(csv_path: Path) -> dict:
    meta = (csv_path.parent /
            csv_path.name.replace("qresults_", "runmeta_", 1).replace(".csv", ".json"))
    return json.loads(meta.read_text())


# --- locating runs -----------------------------------------------------------

def discover_dataset_workloads(results_root: Path) -> list[tuple[str, str]]:
    pairs = set()
    for csv_path in results_root.glob("*/*/*/*/qresults_*.csv"):
        # results/<dataset>/<workload>/<substrate>/<method>/qresults_*.csv
        dataset = csv_path.parents[3].name
        workload = csv_path.parents[2].name
        pairs.add((dataset, workload))
    return sorted(pairs)


def find_oracles(results_root: Path, dataset: str, workload: str) -> dict[int, Oracle]:
    """The scan oracle is produced once per measure count; key oracles by nm so a
    run is validated against the oracle that served the same measures."""
    scan_dir = results_root / dataset / workload / "n_a" / "scan"
    cands = sorted(scan_dir.glob("qresults_*.csv"))
    if not cands:
        sys.exit(f"no scan oracle for ({dataset}, {workload}): expected "
                 f"{scan_dir}/qresults_*.csv")
    oracles: dict[int, Oracle] = {}
    for csv_path in cands:
        nm = int(runmeta_for(csv_path)["num_measures"])
        if nm not in oracles:  # scan is exact and identical across str/mem/run
            oracles[nm] = Oracle(csv_path)
    return oracles


def run_files(results_root: Path, dataset: str, workload: str) -> list[Path]:
    out = []
    for csv_path in (results_root / dataset / workload).glob("*/*/qresults_*.csv"):
        if csv_path.parent.name != "scan":
            out.append(csv_path)
    return sorted(out)


# --- validating one run file -------------------------------------------------

def validate_run(csv_path: Path, oracles: dict[int, Oracle],
                 coverage_alpha: float) -> tuple[dict, list[dict], list[dict]]:
    runmeta = runmeta_for(csv_path)
    method = runmeta.get("method", csv_path.parent.name)
    substrate = runmeta.get("substrate", csv_path.parents[1].name)
    eb = float(runmeta.get("error_bound", 0.0) or 0.0)
    conf = float(runmeta.get("confidence", 0.0) or 0.0)
    nm = int(runmeta["num_measures"])
    if nm not in oracles:
        sys.exit(f"no scan oracle with num_measures={nm} for {csv_path}")
    oracle = oracles[nm]

    details: list[dict] = []
    approx_recs: list[dict] = []     # one per genuinely-sampled aggregate
    n_exact = n_exact_fail = 0
    n_approx = approx_within = covered = 0
    max_rel_err = 0.0
    guard_fail = 0

    # Build-inclusive cost. Per-query latency_ms times only the query; the
    # one-time index build (and any summary precompute) is timed once into
    # runmeta's init_ms. A fair total counts both, with init_ms as the query-0
    # offset of the cumulative curve: a full-build substrate pays a large
    # offset and cheap queries, a cracking substrate pays ~0 offset and folds
    # construction into the per-query latencies.
    init_ms = float(runmeta.get("init_ms", 0.0) or 0.0)
    query_ms_total = 0.0

    # Gate: the result file must be self-describing. A missing stamp is a hard
    # failure -- the run cannot be traced or reproduced without it.
    missing_stamps = [s for s in REQUIRED_RUNMETA_STAMPS if s not in runmeta]
    if missing_stamps:
        guard_fail += 1
        details.append({"query_ordinal": -1, "aggregate": "", "measure": "",
                        "kind": "runmeta", "ok": False,
                        "note": "missing stamps: " + ",".join(missing_stamps)})

    for row in read_rows(csv_path):
        ordinal, rect_key, aggregates = explode(row)
        query_ms_total += as_float(row.get("latency_ms")) if row.get("latency_ms") else 0.0
        # Same-query guard: identical rectangle per ordinal.
        if oracle.rect.get(ordinal) != rect_key:
            guard_fail += 1
            details.append({"query_ordinal": ordinal, "aggregate": "", "measure": "",
                            "kind": "guard", "ok": False,
                            "note": "query_rect differs from oracle"})
            continue

        # Gate: status must be internally consistent with the convergence flag.
        # A converged/exactified/exact row must report its target met; an
        # exhausted_unconverged (or unrecognized) status is a contradiction.
        status = row.get("status", "")
        target_met = str(row.get("target_satisfied", "")).lower() == "true"
        if status in TARGET_MET_STATUS:
            if not target_met:
                guard_fail += 1
                details.append({"query_ordinal": ordinal, "aggregate": "",
                                "measure": "", "kind": "status", "ok": False,
                                "note": f"status={status} but target_satisfied=false"})
        else:
            guard_fail += 1
            details.append({"query_ordinal": ordinal, "aggregate": "", "measure": "",
                            "kind": "status", "ok": False,
                            "note": f"unconverged/unknown status: {status!r}"})

        truth = oracle.by_ordinal[ordinal]
        row_all_exact = True
        for e in aggregates:
            k = agg_key(e)
            o = truth.get(k)
            if o is None and k not in truth:
                details.append({"query_ordinal": ordinal, "aggregate": k[0],
                                "measure": k[1], "kind": "missing", "ok": False,
                                "note": "aggregate absent from oracle"})
                n_exact_fail += 1
                continue
            est = as_float(e["estimate"])
            lo, hi = as_float(e.get("ci_low")), as_float(e.get("ci_high"))
            is_exact = bool(e.get("exact", False))
            row_all_exact = row_all_exact and is_exact

            # Gate: the reported interval must enclose its own point estimate.
            if not _interval_brackets(est, lo, hi):
                guard_fail += 1
                details.append({"query_ordinal": ordinal, "aggregate": k[0],
                                "measure": k[1], "kind": "interval", "ok": False,
                                "estimate": est,
                                "note": f"estimate {est} outside [{lo}, {hi}]"})

            if is_exact:
                n_exact += 1
                ok = _exact_ok(k[0], est, o)
                if not ok:
                    n_exact_fail += 1
                details.append({"query_ordinal": ordinal, "aggregate": k[0],
                                "measure": k[1], "kind": "exact", "ok": ok,
                                "estimate": est, "oracle": o})
            else:
                n_approx += 1
                rel = 0.0 if both_nan(est, o) else abs(est - o) / max(1.0, abs(o))
                within = both_nan(est, o) or rel <= eb
                cov = both_nan(est, o) or (
                    not math.isnan(lo) and not math.isnan(hi) and lo <= o <= hi)
                approx_within += int(within)
                covered += int(cov)
                max_rel_err = max(max_rel_err, 0.0 if math.isnan(rel) else rel)
                details.append({"query_ordinal": ordinal, "aggregate": k[0],
                                "measure": k[1], "kind": "approx", "ok": within,
                                "estimate": est, "oracle": o, "rel_err": rel,
                                "covered": cov})
                approx_recs.append({"ordinal": ordinal, "aggregate": k[0],
                                    "measure": k[1], "covered": bool(cov)})

        # Gate: an exact/exactified status implies every aggregate read in full.
        if status in EXACT_VALUED_STATUS and not row_all_exact:
            guard_fail += 1
            details.append({"query_ordinal": ordinal, "aggregate": "", "measure": "",
                            "kind": "exact_consistency", "ok": False,
                            "note": f"status={status} but an aggregate is not exact"})

    summary = {
        "dataset": csv_path.parents[3].name,
        "workload": csv_path.parents[2].name,
        "method": method, "substrate": substrate,
        "num_measures": runmeta.get("num_measures"),
        "sort_gather_by_row_id": runmeta.get("sort_gather_by_row_id"),
        "measure_storage": runmeta.get("measure_storage"),
        "error_bound": eb, "confidence": conf, "run_id": runmeta.get("run_id"),
        "exact_checked": n_exact, "exact_failed": n_exact_fail,
        "approx_checked": n_approx,
        "approx_within_eb_frac": (approx_within / n_approx) if n_approx else "",
        "coverage_frac": (covered / n_approx) if n_approx else "",
        "max_rel_err": max_rel_err if n_approx else "",
        "guard_failed": guard_fail,
        "init_ms": init_ms,
        "query_ms_total": query_ms_total,
        "cumulative_ms": init_ms + query_ms_total,
        "status": "pass" if (n_exact_fail == 0 and guard_fail == 0) else "FAIL",
    }
    return summary, details, approx_recs


def judge_coverage(runs: list[tuple[int, int]], positions: dict[int, list[bool]],
                   nominal: float, alpha: float) -> dict:
    """Judge CI coverage of one ``(measure, aggregate)`` across replicate runs.

    ``runs`` is one ``(covered, n)`` pair per run (n = sampled aggregates of this
    kind in that run); ``positions`` maps a query ordinal to the covered flags
    observed across runs (for the zero-coverage structural view). The runs are
    the independent trials: with R >= 2 we test H0 (coverage >= nominal) against
    H1 (coverage < nominal) at level ``alpha`` and flag ``low`` only when the
    one-sided **upper** Student-t (R-1 df) confidence bound on the mean still
    falls below nominal -- i.e. only on confident under-coverage. (Flagging on
    the *lower* bound would be the wrong tail: it dips below nominal even for a
    perfectly calibrated cell, so it would over-flag.) With R == 1 it falls back
    to a coarse single-run binomial trip-wire. A zero-coverage position is one
    sampled in every run yet covered in none."""
    runs = [(c, n) for (c, n) in runs if n > 0]
    R = len(runs)
    zero_positions = sum(1 for flags in positions.values()
                         if len(flags) == R and R > 0 and not any(flags))
    if R == 0:
        return {"n_runs": 0, "mean_coverage": "", "coverage_upper": "",
                "coverage_status": "n/a", "basis": "", "zero_coverage_positions": 0}
    if R == 1:
        c, n = runs[0]
        return {"n_runs": 1, "mean_coverage": c / n, "coverage_upper": "",
                "coverage_status": coverage_status(c, n, nominal, alpha),
                "basis": "single_run_binomial", "zero_coverage_positions": zero_positions}
    fracs = [c / n for c, n in runs]
    mean = statistics.fmean(fracs)
    total = sum(n for _, n in runs)
    if total < MIN_COVERAGE_SAMPLES:
        return {"n_runs": R, "mean_coverage": mean, "coverage_upper": "",
                "coverage_status": "insufficient", "basis": "runs_t",
                "zero_coverage_positions": zero_positions}
    s = statistics.stdev(fracs)
    upper = mean + student_t_ppf(1.0 - alpha, R - 1) * s / math.sqrt(R)
    return {"n_runs": R, "mean_coverage": mean, "coverage_upper": upper,
            "coverage_status": "low" if upper < nominal else "ok",
            "basis": "runs_t", "zero_coverage_positions": zero_positions}


def cell_key_of(csv_path: Path) -> tuple[str, str]:
    """Group replicate runs: everything but the ``_run<R>`` filename suffix."""
    stem = csv_path.name[len("qresults_"):-len(".csv")]
    return (str(csv_path.parent), stem.rsplit("_run", 1)[0])


def _exact_ok(aggregate: str, est: float, oracle: float) -> bool:
    if both_nan(est, oracle):
        return True
    if math.isnan(est) or math.isnan(oracle):
        return False
    if aggregate in EXACT_TO_BIT:
        return est == oracle
    return abs(est - oracle) <= SUM_AVG_TOL * max(1.0, abs(oracle))


def _interval_brackets(est: float, lo: float, hi: float) -> bool:
    """True if the reported interval encloses its own point estimate.

    Unverifiable rows (any of the three not finite, e.g. an empty-box AVG that
    serializes as null) are skipped rather than failed. A small relative slack
    absorbs floating-point rounding in delta-method bounds.
    """
    if math.isnan(est) or math.isnan(lo) or math.isnan(hi):
        return True
    tol = BRACKET_TOL * max(1.0, abs(est))
    return lo - tol <= est <= hi + tol


# --- output ------------------------------------------------------------------

SUMMARY_COLS = ["dataset", "workload", "method", "substrate", "num_measures",
                "sort_gather_by_row_id", "measure_storage",
                "error_bound", "confidence", "run_id", "exact_checked",
                "exact_failed", "approx_checked", "approx_within_eb_frac",
                "coverage_frac", "max_rel_err", "guard_failed",
                "init_ms", "query_ms_total", "cumulative_ms", "status"]
COVERAGE_COLS = ["dataset", "workload", "method", "substrate", "num_measures",
                 "sort_gather_by_row_id", "measure_storage",
                 "error_bound", "confidence", "aggregate", "measure", "n_runs",
                 "mean_coverage", "coverage_upper", "coverage_status",
                 "zero_coverage_positions", "basis"]
DETAIL_COLS = ["query_ordinal", "aggregate", "measure", "kind", "ok",
               "estimate", "oracle", "rel_err", "covered", "note"]


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate A3I results vs the scan oracle.")
    ap.add_argument("--results-root", default=None,
                    help="results root (default: experiments/results)")
    ap.add_argument("--validation-root", default=None,
                    help="output root (default: experiments/validation)")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--coverage-alpha", type=float, default=DEFAULT_COVERAGE_ALPHA,
                    help="one-sided test level for the CI-coverage shortfall "
                         "check (default: %(default)s)")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="treat a coverage shortfall as a failure (non-zero exit) "
                         "instead of a warning")
    args = ap.parse_args()

    results_root = a3i_config.results_root(args.results_root)
    validation_root = Path(args.validation_root) if args.validation_root else REPO_ROOT / "experiments" / "validation"

    pairs = discover_dataset_workloads(results_root)
    if args.dataset:
        pairs = [(d, w) for (d, w) in pairs if d == args.dataset]
    if args.workload:
        pairs = [(d, w) for (d, w) in pairs if w == args.workload]
    if not pairs:
        sys.exit(f"no results found under {results_root}")

    total_fail = 0
    total_warn = 0
    total_cov_fail = 0
    for dataset, workload in pairs:
        oracles = find_oracles(results_root, dataset, workload)
        for oracle in oracles.values():
            oracle.check_complete()

        summaries: list[dict] = []
        details: list[dict] = []
        coverage_rows: list[dict] = []

        # Group replicate runs into cells (same cell, differing only by run_id).
        cells: dict[tuple[str, str], list[Path]] = defaultdict(list)
        for csv_path in run_files(results_root, dataset, workload):
            cells[cell_key_of(csv_path)].append(csv_path)

        for _, paths in sorted(cells.items()):
            # Replicate runs are pooled for the coverage judgement, so they must
            # share the gather order: results built sorted and unsorted measure
            # different code paths and must never be combined. A mixed cell is a
            # collection error, not a statistical one, so fail hard.
            gather_orders = {bool(runmeta_for(p).get("sort_gather_by_row_id"))
                             for p in paths}
            if len(gather_orders) > 1:
                sys.exit(f"cell {paths[0].parent} mixes sort_gather_by_row_id "
                         f"across replicate runs ({sorted(gather_orders)}); "
                         f"these results cannot be pooled")
            per_run_ma: list[dict[tuple[str, str], list[int]]] = []
            positions: dict[tuple[str, str], dict[int, list[bool]]] = defaultdict(
                lambda: defaultdict(list))
            cell_meta: dict | None = None
            for csv_path in sorted(paths):
                summary, det, approx = validate_run(csv_path, oracles,
                                                     args.coverage_alpha)
                summaries.append(summary)
                for d in det:
                    d["method"] = summary["method"]
                    details.append(d)
                if summary["status"] == "FAIL":
                    total_fail += 1
                    print(f"FAIL: {dataset}/{workload} {summary['method']} "
                          f"(exact_failed={summary['exact_failed']}, "
                          f"guard_failed={summary['guard_failed']})", file=sys.stderr)
                ma_counts: dict[tuple[str, str], list[int]] = defaultdict(
                    lambda: [0, 0])
                for rec in approx:
                    ma = (rec["measure"], rec["aggregate"])
                    ma_counts[ma][0] += int(rec["covered"])
                    ma_counts[ma][1] += 1
                    positions[ma][rec["ordinal"]].append(bool(rec["covered"]))
                per_run_ma.append(ma_counts)
                cell_meta = summary

            if cell_meta is None:
                continue
            nominal = cell_meta["confidence"]
            all_ma: set[tuple[str, str]] = set()
            for d in per_run_ma:
                all_ma |= set(d)
            for ma in sorted(all_ma):
                runs = [tuple(d[ma]) for d in per_run_ma if ma in d]
                judged = judge_coverage(runs, positions[ma], nominal,
                                        args.coverage_alpha)
                coverage_rows.append({
                    "dataset": dataset, "workload": workload,
                    "method": cell_meta["method"], "substrate": cell_meta["substrate"],
                    "num_measures": cell_meta["num_measures"],
                    "sort_gather_by_row_id": cell_meta["sort_gather_by_row_id"],
                    "measure_storage": cell_meta["measure_storage"],
                    "error_bound": cell_meta["error_bound"], "confidence": nominal,
                    "measure": ma[0], "aggregate": ma[1], **judged})
                if judged["coverage_status"] == "low":
                    total_warn += 1
                    escalate = args.strict_coverage and judged["basis"] == "runs_t"
                    if escalate:
                        total_cov_fail += 1
                    tag = "FAIL (coverage)" if escalate else "WARN (coverage)"
                    mean = judged["mean_coverage"]
                    print(f"{tag}: {dataset}/{workload} {cell_meta['method']} "
                          f"{ma[1]}({ma[0]}) mean coverage={mean:.3f} "
                          f"< confidence={nominal:.3f} "
                          f"(R={judged['n_runs']}, basis={judged['basis']})",
                          file=sys.stderr)

        out_dir = validation_root / dataset / workload
        write_csv(out_dir / "validation_summary.csv", SUMMARY_COLS, summaries)
        write_csv(out_dir / "validation_coverage.csv", COVERAGE_COLS, coverage_rows)
        write_csv(out_dir / "validation_details.csv", DETAIL_COLS, details)
        print(f"{dataset}/{workload}: {len(summaries)} runs validated "
              f"-> {out_dir}/validation_summary.csv")

    if total_fail:
        print(f"\n{total_fail} run(s) FAILED exact-equality / same-query checks",
              file=sys.stderr)
    if total_warn:
        kind = "warned on" if not args.strict_coverage else "flagged for"
        print(f"{total_warn} (measure, aggregate) cell(s) {kind} CI coverage "
              f"below nominal confidence", file=sys.stderr)
    if total_fail or total_cov_fail:
        return 1
    print("\nall exact checks passed"
          + ("" if not total_warn else "; coverage warnings reported above"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
