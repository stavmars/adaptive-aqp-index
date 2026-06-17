#!/usr/bin/env python3
"""Hermetic end-to-end smoke test for the experiment pipeline.

In a self-contained temporary tree (no sudo, no page-cache drop) it:

  1. prepares a tiny synthetic dataset (generator path) via prepare_dataset.py;
  2. writes a minimal workload catalog + plan referencing that dataset;
  3. runs scripts/run_experiments.py (--warm) to produce real result cells for
     an exact ladder plus the accuracy-aware method;
  4. runs scripts/validate_results.py over those cells and asserts the exact
     runs equal the scan oracle and the same-query guard holds.

It exercises the orchestrator and validator against artifacts the compiled
tools actually wrote -- not mocks. Run directly (python3 tests/
test_experiments_smoke.py) or via ctest (label m_experiments). Uses only the
Python standard library + PyYAML.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def prepare_dataset(work: Path) -> tuple[Path, str]:
    """Prepare a tiny 2-dim, 2-measure synthetic dataset; return (prepared_root, id)."""
    cfg = work / "tiny.yaml"
    cfg.write_text(
        "dataset_id: synth_tiny\n"
        "source:\n"
        "  generator:\n"
        "    seed: 1\n"
        "    rows: 4000\n"
        "    columns:\n"
        "      - {name: col0, dist: uniform, p0: 0, p1: 1000}\n"
        "      - {name: col1, dist: uniform, p0: 0, p1: 1000}\n"
        "      - {name: col2, dist: uniform, p0: 0, p1: 1000}\n"
        "      - {name: col3, dist: normal, p0: 50, p1: 15}\n"
        "dimensions:\n"
        "  - {name: col0, bounds: [0, 1000]}\n"
        "  - {name: col1, bounds: [0, 1000]}\n"
        "measures: [col2, col3]\n"
    )
    prepared = work / "prepared"
    raw = work / "raw"
    run([sys.executable, str(SCRIPTS / "prepare_dataset.py"), str(cfg),
         "--prepared-root", str(prepared), "--parquet-dir", str(raw)])
    assert (prepared / "synth_tiny" / "manifest.json").is_file()
    return prepared, "synth_tiny"


def write_plan_tree(work: Path, dataset_id: str) -> tuple[Path, Path]:
    """Write a temp plan dir (_defaults + smoke plan) and workload catalog dir."""
    plans = work / "plans"
    plans.mkdir()
    (plans / "_defaults.yaml").write_text(
        "runs: [scan, kd, kd_agg, a3i_akd]\n"
        f"workloads: [{dataset_id}_clustered]\n"
        "nm: [1, 2]\n"
        "eb: [0.05]\n"
        "partition_size: [1024]\n"
        "mem: [unbounded]\n"
        "run_id: [0, 1]\n"
        "run_id_by_method: {scan: [0]}\n"   # scan once; other methods twice
        "max_queries: null\n"
        "confidence: 0.95\n"
    )
    (plans / "smoke.yaml").write_text("plan_id: smoke\n")

    wlcfg = work / "workload_configs"
    wlcfg.mkdir()
    (wlcfg / "tiny.yaml").write_text(
        "workloads:\n"
        f"  - workload_id: {dataset_id}_clustered\n"
        f"    dataset_id: {dataset_id}\n"
        "    family: clustered\n"
        "    extent_mode: closed_form\n"
        "    selectivity: 0.1\n"
        "    focus: [500, 500]\n"
        "    seed: 0\n"
        "    count: 25\n"
    )
    return plans, wlcfg


def run_experiments(work: Path, prepared: Path, plans: Path, wlcfg: Path) -> Path:
    results = work / "results"
    workloads = work / "workloads"
    run([sys.executable, str(SCRIPTS / "run_experiments.py"),
         "--plan", "smoke",
         "--plans-dir", str(plans),
         "--workload-config-dir", str(wlcfg),
         "--prepared-root", str(prepared),
         "--results-root", str(results),
         "--workloads-dir", str(workloads),
         "--warm"])
    return results


def assert_paths_and_idempotency(work: Path, prepared: Path, plans: Path,
                                 wlcfg: Path, results: Path, dataset_id: str) -> None:
    wl = f"{dataset_id}_clustered"
    # The exact ladder lands under its substrate; a3i under adaptive_kd.
    scan = list((results / dataset_id / wl / "n_a" / "scan").glob("qresults_*.csv"))
    kd = list((results / dataset_id / wl / "static_kd" / "kd").glob("qresults_*.csv"))
    a3i = list((results / dataset_id / wl / "adaptive_kd" / "a3i_akd").glob("qresults_*.csv"))
    assert scan, "missing scan oracle"
    assert kd, "missing kd run"
    assert a3i, "missing a3i run"
    # Exact runs carry no err in the key; approximate a3i carries err.
    assert "err" not in scan[0].name and "err" not in kd[0].name, (scan[0].name, kd[0].name)
    assert "err0.05" in a3i[0].name, a3i[0].name

    # Per-method run override: scan runs once (run0 only); other methods get the
    # default run_id [0, 1], so a3i has a run1.
    assert all("_run0.csv" in p.name for p in scan), [p.name for p in scan]
    assert any("_run1.csv" in p.name for p in a3i), [p.name for p in a3i]

    # Idempotency: a second run with no --force re-runs nothing.
    again = run([sys.executable, str(SCRIPTS / "run_experiments.py"),
                 "--plan", "smoke", "--plans-dir", str(plans),
                 "--workload-config-dir", str(wlcfg), "--prepared-root", str(prepared),
                 "--results-root", str(results), "--workloads-dir", str(work / "workloads"),
                 "--warm"])
    assert "0 run" in again.stdout, again.stdout

    # Completeness report: every expected cell is present on disk.
    rep = run([sys.executable, str(SCRIPTS / "run_experiments.py"),
               "--plan", "smoke", "--plans-dir", str(plans),
               "--workload-config-dir", str(wlcfg), "--prepared-root", str(prepared),
               "--results-root", str(results), "--workloads-dir", str(work / "workloads"),
               "--report"])
    assert "0 missing" in rep.stdout, rep.stdout
    # And a durable per-cell run log was written.
    assert list((results / "_runlog").glob("smoke_*.jsonl")), "no _runlog written"
    print("paths + idempotency + report: OK")


def validate(work: Path, results: Path, dataset_id: str) -> None:
    validation = work / "validation"
    out = run([sys.executable, str(SCRIPTS / "validate_results.py"),
               "--results-root", str(results),
               "--validation-root", str(validation)])
    assert "all exact checks passed" in out.stdout, out.stdout

    wl = f"{dataset_id}_clustered"
    summ = validation / dataset_id / wl / "validation_summary.csv"
    rows = list(csv.DictReader(summ.open()))
    methods = {r["method"]: r for r in rows}
    # kd / kd_agg are exact: every checked aggregate must pass, none fail.
    for m in ("kd", "kd_agg"):
        assert m in methods, (m, list(methods))
        assert int(methods[m]["exact_failed"]) == 0, methods[m]
        assert int(methods[m]["guard_failed"]) == 0, methods[m]
        assert methods[m]["status"] == "pass", methods[m]
        assert int(methods[m]["exact_checked"]) > 0, methods[m]
    # a3i ran approximate aggregates that were scored (coverage recorded).
    assert "a3i_akd" in methods, list(methods)
    assert methods["a3i_akd"]["status"] == "pass", methods["a3i_akd"]
    # Build-inclusive cost columns: cumulative = init + query time, all present
    # and self-consistent for every method.
    for m, r in methods.items():
        init = float(r["init_ms"])
        qsum = float(r["query_ms_total"])
        cum = float(r["cumulative_ms"])
        assert init >= 0.0 and qsum >= 0.0, (m, r)
        assert abs(cum - (init + qsum)) <= 1e-6, (m, r)
    print("validation summary: OK")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a3i_exp_smoke_") as d:
        work = Path(d)
        prepared, dataset_id = prepare_dataset(work)
        plans, wlcfg = write_plan_tree(work, dataset_id)
        results = run_experiments(work, prepared, plans, wlcfg)
        assert_paths_and_idempotency(work, prepared, plans, wlcfg, results, dataset_id)
        validate(work, results, dataset_id)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
