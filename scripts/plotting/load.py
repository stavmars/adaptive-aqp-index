"""Layer 1 -- discover result files and assemble one tidy long frame.

This is the only place that touches raw `qresults`/`runmeta` files. It discovers
cells by the result-path convention, validates each file, explodes the
`aggregates` JSON, and returns a single long DataFrame at the finest grain:
one row per `(cell-run, query_ordinal, aggregate, measure)`.

Two grains live side by side in that frame (see the data model): *cost/work*
columns (`latency_ms`, `measure_reads`, ...) are per `(cell-run, query)` and are
constant across a query's aggregate/measure rows; *accuracy* columns
(`estimate`, `ci_low`, ...) vary per `(aggregate, measure)`. Cell-level scalars
(`init_ms`, `cold`, `mem`, `measure_storage`, `engine_build_version`) come from
the sidecar and are denormalized onto every row.

Failure policy: *corrupt* input fails loudly (`LoadError`) -- a malformed/short
file, a header that does not match the schema, a missing runmeta stamp, or a
duplicate `(cell, run)` -- because silently analyzing the wrong or incomplete
data is the most dangerous plotting bug. *Missing* cells are simply absent from
the frame (discovery returns what exists), so a partial matrix loads fine.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# The frozen results header (must match cell_runner.cpp `kHeader`). A file whose
# header differs is rejected rather than silently misread.
EXPECTED_HEADER = (
    "query_ordinal,method,substrate,dataset,workload,query_rect,aggregates,"
    "target_satisfied,status,exactify_cause,pre_exactification_error_bound,"
    "sampling_seed,latency_ms,measure_reads,sampled_rows,"
    "exactified_rows,partitions_touched,partitions_split,exact_contributors,"
    "reusable_strata,query_local_strata,query_local_exact_contributors,"
    "summary_reuse_hits,adaptive_rounds"
).split(",")

# Runmeta stamps Layer 1 requires for a result file to be self-describing.
# (The `mem` cap is recovered from the cell path, not a runmeta key -- so it is
# not required here; `measure_storage` is recorded but not required for loading.)
REQUIRED_RUNMETA = ("engine_build_version", "cold", "max_queries")

# Per-query cost/work columns (constant across a query's aggregate/measure rows).
COST_COLS = ("latency_ms", "measure_reads", "sampled_rows", "exactified_rows",
             "partitions_touched", "partitions_split", "exact_contributors",
             "reusable_strata", "query_local_strata",
             "query_local_exact_contributors", "summary_reuse_hits",
             "adaptive_rounds")

# Cell identity (one cell-run = one qresults file). `n` (query count) is part of
# identity so runs with different `max_queries` are distinct cells, not pooled.
CELL_KEYS = ("dataset", "workload", "method", "substrate",
             "nm", "mem", "str", "n", "eb", "run_id")


class LoadError(Exception):
    """Corrupt or inconsistent result input -- never swallowed."""


_KEY_RE = re.compile(
    r"qresults_(?P<key>.+)_run(?P<run>\d+)\.csv$")


def _parse_axes(key: str) -> dict:
    """Pull nm / mem / str / n / eb out of a cell key like
    `err0.01_mcols2_memINMEM_n100_str1024` (order-insensitive)."""
    ax: dict = {"nm": None, "mem": None, "str": None, "n": None, "eb": None}
    for tok in key.split("_"):
        if tok.startswith("mcols"):
            ax["nm"] = int(tok[5:])
        elif tok.startswith("mem"):
            ax["mem"] = tok[3:]
        elif tok.startswith("str"):
            ax["str"] = int(tok[3:])
        elif tok.startswith("n") and tok[1:].isdigit():
            ax["n"] = int(tok[1:])
        elif tok.startswith("err"):
            ax["eb"] = float(tok[3:])
    return ax


def discover(results_root) -> list[Path]:
    """Every `qresults_*.csv` under the results root, by path convention
    `<dataset>/<workload>/<substrate>/<method>/qresults_*.csv`. Sorted for
    determinism."""
    root = Path(results_root)
    return sorted(root.glob("*/*/*/*/qresults_*.csv"))


def _read_runmeta(qpath: Path) -> dict:
    mpath = qpath.with_name(qpath.name.replace("qresults_", "runmeta_")
                            .replace(".csv", ".json"))
    try:
        meta = json.loads(mpath.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise LoadError(f"{qpath}: missing/unreadable runmeta sidecar: {e}")
    missing = [s for s in REQUIRED_RUNMETA if s not in meta]
    if missing:
        raise LoadError(f"{mpath}: runmeta missing required stamp(s): {missing}")
    return meta


def _cell_rows(qpath: Path) -> list[dict]:
    # Validate the header before the typed parse: a misaligned header makes
    # pandas raise an opaque dtype error, so check it up front and surface a
    # clear LoadError instead.
    with open(qpath) as f:
        header = f.readline().rstrip("\r\n").split(",")
    if header != EXPECTED_HEADER:
        raise LoadError(f"{qpath}: header does not match schema (got {header})")
    try:
        df = pd.read_csv(qpath, dtype={"query_ordinal": "int64"})
    except (ValueError, pd.errors.ParserError) as e:
        raise LoadError(f"{qpath}: parse error: {e}")
    if df.empty:
        raise LoadError(f"{qpath}: no query rows")
    # one row per query_ordinal, contiguous 0..n-1
    qo = sorted(df["query_ordinal"].tolist())
    if qo != list(range(len(qo))) or len(qo) != df["query_ordinal"].nunique():
        raise LoadError(f"{qpath}: query_ordinal not one-per-row contiguous "
                        f"(have {len(qo)} rows, {df['query_ordinal'].nunique()} distinct)")

    meta = _read_runmeta(qpath)
    p = qpath.parts
    dataset, workload, substrate, method = p[-5], p[-4], p[-3], p[-2]
    m = _KEY_RE.search(qpath.name)
    ax = _parse_axes(m.group("key"))
    run_id = int(m.group("run"))

    ident = dict(dataset=dataset, workload=workload, method=method,
                 substrate=substrate, nm=ax["nm"], mem=ax["mem"],
                 str=ax["str"], n=ax["n"], eb=ax["eb"], run_id=run_id,
                 init_ms=float(meta.get("init_ms", 0.0)),
                 cold=bool(meta.get("cold")),
                 measure_storage=meta.get("measure_storage"),
                 engine_build_version=meta.get("engine_build_version"))

    out = []
    for _, row in df.iterrows():
        base = {**ident, "query_ordinal": int(row["query_ordinal"]),
                "status": row["status"],
                "target_satisfied": bool(row["target_satisfied"])}
        for c in COST_COLS:
            base[c] = row[c]
        aggs = json.loads(row["aggregates"])
        if not aggs:
            raise LoadError(f"{qpath}: empty aggregates cell at q={row['query_ordinal']}")
        for a in aggs:
            out.append({**base, "aggregate": a["aggregate"], "measure": a["measure"],
                        "estimate": a["estimate"], "ci_low": a["ci_low"],
                        "ci_high": a["ci_high"],
                        "relative_half_width": a.get("relative_half_width"),
                        "exact": bool(a["exact"])})
    return out


def load_frame(results_root, datasets=None, workloads=None) -> pd.DataFrame:
    """Discover + validate every result cell into one tidy long frame.

    `datasets`/`workloads` optionally restrict by name. Raises `LoadError` on any
    corrupt file or a duplicate `(cell, run)`; missing cells are simply absent.
    """
    rows: list[dict] = []
    seen: set = set()
    for qpath in discover(results_root):
        p = qpath.parts
        if datasets and p[-5] not in datasets:
            continue
        if workloads and p[-4] not in workloads:
            continue
        cell_rows = _cell_rows(qpath)
        ident = tuple(cell_rows[0][k] for k in CELL_KEYS)
        if ident in seen:
            raise LoadError(f"duplicate cell-run {ident} (second file {qpath})")
        seen.add(ident)
        rows.extend(cell_rows)
    if not rows:
        return pd.DataFrame(columns=list(CELL_KEYS) + ["query_ordinal",
                            "aggregate", "measure", "estimate", "ci_low",
                            "ci_high", "exact", *COST_COLS])
    return pd.DataFrame(rows)
