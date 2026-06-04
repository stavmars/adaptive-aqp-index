"""Layer 2 -- the runs->queries reductions, the oracle join, and per-cell
summaries, over the long frame from `load.py`.

Reductions are fixed *per metric*:
  * cost (latency, measure_reads, ...): combine the R replicate runs by per-query
    **median**, then reduce across queries (sum for cumulative, percentiles for
    the distribution);
  * coverage: the fraction of intervals that bracket the truth (a probability;
    over multiple runs it is a mean).

The **cost-collapse rule** is enforced first: cost columns are per
`(cell-run, query)`, so they are de-duplicated to that grain before any sum/
median -- aggregating over the exploded `(aggregate, measure)` frame would
multiply a cost by the per-query aggregate x measure count.
"""
from __future__ import annotations

import pandas as pd

from .load import CELL_KEYS, COST_COLS

# Cell identity minus the replicate axis: the unit a figure/table compares.
METHOD_KEYS = tuple(k for k in CELL_KEYS if k != "run_id")
_ORACLE_KEYS = ["dataset", "workload", "query_ordinal", "aggregate", "measure"]


def per_query_cost(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per `(cell-run, query)` with the cost columns.

    This is the cost-collapse rule: it drops the exploded accuracy rows so a cost
    metric is counted once per query, not once per `(aggregate, measure)`.
    """
    cols = list(CELL_KEYS) + ["query_ordinal", "init_ms"] + list(COST_COLS)
    return (frame[cols]
            .drop_duplicates(subset=list(CELL_KEYS) + ["query_ordinal"])
            .reset_index(drop=True))


def cost_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per method-cell (runs reduced): init, cumulative time, latency
    percentiles, total measure reads. Latency is per-query median over runs,
    then summed / percentiled over queries."""
    pqc = per_query_cost(frame)
    # median over the replicate runs at each query
    pq = (pqc.groupby(list(METHOD_KEYS) + ["query_ordinal"], dropna=False)
             .median(numeric_only=True).reset_index())
    g = pq.groupby(list(METHOD_KEYS), dropna=False)
    out = g.agg(
        total_latency_ms=("latency_ms", "sum"),
        lat_p50=("latency_ms", lambda s: s.quantile(0.50)),
        lat_p95=("latency_ms", lambda s: s.quantile(0.95)),
        lat_p99=("latency_ms", lambda s: s.quantile(0.99)),
        total_reads=("measure_reads", "sum"),
        init_ms=("init_ms", "median"),
    ).reset_index()
    out["cum_ms"] = out["init_ms"] + out["total_latency_ms"]
    return out


def with_error(frame: pd.DataFrame) -> pd.DataFrame:
    """Join every row to the `scan` oracle truth at the same
    `(dataset, workload, query, aggregate, measure)` and add `error` (relative,
    floored at 1) and `covered` (truth within `[ci_low, ci_high]`, only for
    non-exact rows with a finite interval; NaN otherwise). The oracle is matched
    on the join key only -- it is independent of mem/str/run/eb."""
    oracle = (frame[frame["method"] == "scan"]
              [_ORACLE_KEYS + ["estimate"]]
              .drop_duplicates(subset=_ORACLE_KEYS)
              .rename(columns={"estimate": "truth"}))
    m = frame.merge(oracle, on=_ORACLE_KEYS, how="left")
    est = pd.to_numeric(m["estimate"], errors="coerce")
    truth = pd.to_numeric(m["truth"], errors="coerce")
    lo = pd.to_numeric(m["ci_low"], errors="coerce")
    hi = pd.to_numeric(m["ci_high"], errors="coerce")
    m["error"] = (est - truth).abs() / truth.abs().clip(lower=1.0)
    covered = (lo <= truth) & (truth <= hi)
    # coverage is only meaningful for genuinely-sampled (non-exact) intervals
    m["covered"] = covered.where(~m["exact"] & lo.notna() & hi.notna() & truth.notna())
    return m


def accuracy_summary(frame: pd.DataFrame, eb: float) -> pd.DataFrame:
    """Per method-cell: fraction of aggregates within `eb`, and CI coverage over
    sampled intervals."""
    m = with_error(frame)
    g = m.groupby(list(METHOD_KEYS), dropna=False)
    out = g.agg(
        within_eb=("error", lambda s: (s <= eb).mean()),
        coverage=("covered", "mean"),
        max_relerr=("error", "max"),
    ).reset_index()
    return out


def cell_summary(frame: pd.DataFrame, eb: float) -> pd.DataFrame:
    """One row per method-cell: cost + accuracy + speedup over `scan`."""
    cost = cost_summary(frame)
    acc = accuracy_summary(frame, eb)
    out = cost.merge(acc, on=list(METHOD_KEYS), how="left")
    # speedup = scan cumulative / this method's cumulative, within (dataset, workload)
    scan = (out[out["method"] == "scan"][["dataset", "workload", "cum_ms"]]
            .rename(columns={"cum_ms": "scan_cum_ms"}))
    out = out.merge(scan, on=["dataset", "workload"], how="left")
    out["speedup_vs_scan"] = out["scan_cum_ms"] / out["cum_ms"]
    return out


def shared_or_raise(frame: pd.DataFrame, axis: str) -> object:
    """Comparison-safety: every cell in `frame` must share `axis` (e.g. `cold`,
    `mem`) or the comparison is confounded -- raise. Returns the shared value."""
    vals = set(frame[axis].dropna().unique().tolist())
    if len(vals) > 1:
        raise ValueError(f"comparison mixes {axis}={sorted(vals)}; not comparable")
    return next(iter(vals), None)
