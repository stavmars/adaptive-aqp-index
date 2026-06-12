#!/usr/bin/env python3
"""Run an experiment plan: resolve axes to cells, then drive ``a3i_run`` once
per cell.

A *plan* (``experiments/plans/<id>.yaml``) selects values on a subset of the
experimental axes; every unstated axis inherits from
``experiments/plans/_defaults.yaml``. The legality-pruned Cartesian product of
those values (plus ``run_id``) is the cell set. Each cell is one fresh
``a3i_run`` process writing a results CSV and a JSON sidecar under a
deterministic path that encodes exactly the result-affecting axes, so two plans
that share a cell resolve to the same file and the second run skips it.

Pipeline per plan:
  1. Load the plan over the defaults and the workload catalog
     (``configs/workloads/*.yaml``).
  2. For each selected workload, materialise its CSV with ``generate_workload``
     (idempotent by fingerprint), reading the dataset's prepared manifest.
  3. Enumerate legal cells (method x substrate legality; the error bound applies
     to approximate runs only; ``nm`` values above a dataset's measure count are
     pruned).
  4. For each cell, skip when its outputs already exist (unless ``--force`` /
     ``--stale``); otherwise evict the cell's column files from the page cache
     so they are re-read from disk (cold by default; ``--warm`` skips it), run
     ``a3i_run`` under the cell's memory budget (a cgroup launcher when one is
     set), and atomically publish the outputs. Eviction is unprivileged
     (``posix_fadvise(DONTNEED)`` over exactly the dataset's columns); an
     optional ``--drop-caches-cmd`` hook additionally flushes the whole-machine
     page cache for a stricter cold baseline.
  5. Append every cell's disposition to a durable per-run log under
     ``<results>/_runlog/`` (one line per cell, flushed as it finishes, so an
     external kill still leaves a record); ``--report`` later diffs the plan's
     expected cells against what is on disk.

Usage:
    scripts/run_experiments.py --plan headline
    scripts/run_experiments.py --plan eb_sweep nm_sweep --dry-run
    scripts/run_experiments.py --all                 # every plan but the dev ones
    scripts/run_experiments.py --all --report        # what is complete vs missing
    scripts/run_experiments.py --plan headline --force --warm

Run with --help for the full option list. The C++ build itself never depends on
Python; this driver only orchestrates the compiled tools.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("PyYAML is required: pip install pyyaml")

import a3i_config  # loads .env and resolves the shared roots (flag > env > .env > default)

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANS_DIR = REPO_ROOT / "experiments" / "plans"
WORKLOAD_CONFIG_DIR = REPO_ROOT / "configs" / "workloads"

AXIS_KEYS = ("runs", "workloads", "nm", "eb", "str", "mem", "run_id",
             "run_id_by_method", "max_queries", "confidence")


# --- tool resolution ---------------------------------------------------------

def is_debug_build(build_dir: Path) -> bool:
    try:
        for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
            if line.startswith("CMAKE_BUILD_TYPE:"):
                return line.split("=", 1)[1].strip().lower() == "debug"
    except OSError:
        pass
    return False


def find_tool(name: str) -> Path:
    # Prefer an explicit A3I_BUILD_DIR, then the optimized `build-release` tree,
    # then `build` (which the project configures Debug for tests). Preferring
    # build-release means a timing run defaults to the optimized binary even
    # when A3I_BUILD_DIR is unset, and a Debug fallback is warned about below.
    candidates = []
    env_build = os.environ.get("A3I_BUILD_DIR")
    if env_build:
        candidates.append(Path(env_build) / "apps" / name)
        candidates.append(Path(env_build) / "tools" / name)
    for tree in ("build-release", "build"):
        candidates.append(REPO_ROOT / tree / "apps" / name)
        candidates.append(REPO_ROOT / tree / "tools" / name)
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            # cand is <tree>/<apps|tools>/<name>; the tree is two levels up.
            if is_debug_build(cand.parent.parent):
                print(f"warning: using a Debug build at {cand} -- it is "
                      f"unoptimized and several times slower; build a Release "
                      f"tree (cmake -S . -B build-release "
                      f"-DCMAKE_BUILD_TYPE=Release) and/or set A3I_BUILD_DIR "
                      f"for representative timings.", file=sys.stderr)
            return cand
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    searched = "\n  ".join(str(c) for c in candidates)
    sys.exit(f"{name} binary not found. Build the project first "
             f"(cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release && "
             f"cmake --build build-release), or set A3I_BUILD_DIR.\n"
             f"Searched:\n  " + searched)


def engine_version(a3i_run: Path) -> str:
    out = subprocess.run([str(a3i_run), "--version"], capture_output=True, text=True)
    return out.stdout.strip()


def describe_methods(a3i_run: Path) -> tuple[dict[str, str], set[str]]:
    """Read the method catalog from the engine so the method->substrate and
    approximate-method mappings have a single source of truth (the C++
    ``resolve_method``). Returns ``(substrate_by_method, approximate_methods)``.
    """
    out = subprocess.run([str(a3i_run), "--describe-methods"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"a3i_run --describe-methods failed:\n{out.stderr}")
    catalog = json.loads(out.stdout)
    substrate = {m["method"]: m["substrate"] for m in catalog}
    approximate = {m["method"] for m in catalog if m["approx"]}
    return substrate, approximate


# --- plan / catalog loading --------------------------------------------------

def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_plan(plan_id: str, plans_dir: Path) -> dict:
    defaults = load_yaml(plans_dir / "_defaults.yaml")
    plan_path = plans_dir / f"{plan_id}.yaml"
    if not plan_path.is_file():
        sys.exit(f"plan not found: {plan_path}")
    plan = load_yaml(plan_path)
    resolved = dict(defaults)
    for key in AXIS_KEYS:
        if key in plan:
            resolved[key] = plan[key]
    return resolved


def load_workload_catalog(config_dir: Path) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        doc = load_yaml(path)
        for entry in doc.get("workloads", []):
            wid = entry["workload_id"]
            if wid in catalog:
                sys.exit(f"duplicate workload_id '{wid}' (in {path})")
            catalog[wid] = entry
    return catalog


# --- key / path helpers ------------------------------------------------------

def mem_is_in_memory(mem: str) -> bool:
    """The `inmem` setting loads the measure columns fully resident (eager) at
    open time instead of memory-mapping them. Like `unbounded` it is uncapped;
    unlike it, the run touches no page-fault path. Mutually exclusive with a cap
    (resident columns are anonymous memory, not reclaimable page cache)."""
    return mem in ("inmem", "INMEM")


def mem_tag(mem: str) -> str:
    if mem in ("unbounded", "UNB", None):
        return "UNB"
    if mem_is_in_memory(mem):
        return "INMEM"
    # e.g. "16GiB" -> "16G", "8GiB" -> "8G", "512MiB" -> "512M".
    return mem.replace("iB", "").replace("B", "").upper()


def mem_bytes(mem: str) -> int | None:
    # No cgroup cap for either uncapped setting (unbounded mmap, or inmem
    # resident); only a byte budget like "16GiB" wraps the run in a launcher.
    if mem in ("unbounded", "UNB", None) or mem_is_in_memory(mem):
        return None
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    s = mem.replace("iB", "").replace("B", "").strip().upper()
    if s and s[-1] in units:
        return int(float(s[:-1]) * units[s[-1]])
    return int(s)


def cell_key(nm: int, mem: str, strv: int, q: int, eb: float, approx: bool) -> str:
    parts = [f"mcols{nm}", f"mem{mem_tag(mem)}", f"str{strv}", f"n{q}"]
    if approx:
        parts.append(f"err{eb:g}")
    return "_".join(sorted(parts))


def substrate_dir(substrate: str) -> str:
    return substrate.replace("/", "_")


def cell_paths(results_root: Path, dataset: str, workload: str, substrate: str,
               method: str, key: str, run_id: int) -> tuple[Path, Path]:
    d = results_root / dataset / workload / substrate_dir(substrate) / method
    return (d / f"qresults_{key}_run{run_id}.csv",
            d / f"runmeta_{key}_run{run_id}.json")


# --- workload materialisation ------------------------------------------------

def manifest_path(prepared_root: Path, dataset_id: str) -> Path:
    return prepared_root / dataset_id / "manifest.json"


def ensure_workload(gen_tool: Path, scenario: dict, manifest: Path,
                    workloads_dir: Path) -> Path:
    cmd = [str(gen_tool),
           "--manifest", str(manifest),
           "--out-dir", str(workloads_dir),
           "--dataset", scenario["dataset_id"],
           "--name", scenario["workload_id"],
           "--family", scenario["family"],
           "--extent", scenario["extent_mode"],
           "--selectivity", repr(float(scenario["selectivity"])),
           "--seed", str(scenario.get("seed", 0)),
           "--count", str(scenario.get("count", 500))]
    if scenario.get("focus"):
        cmd += ["--focus", ",".join(repr(float(x)) for x in scenario["focus"])]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return workloads_dir / f"{scenario['workload_id']}.csv"


def workload_query_count(csv_path: Path) -> int:
    # The CSV is: fingerprint line, header line, then one row per query.
    with csv_path.open() as f:
        return max(0, sum(1 for _ in f) - 2)


# --- cell enumeration --------------------------------------------------------

class Cell:
    __slots__ = ("method", "substrate", "dataset", "workload", "nm", "eb",
                 "strv", "mem", "run_id", "max_queries", "confidence",
                 "approx", "key", "qcount")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def enumerate_cells(plan: dict, catalog: dict, prepared_root: Path,
                    workloads_dir: Path, gen_tool: Path,
                    run_substrate: dict[str, str],
                    approximate_runs: set[str]) -> list[Cell]:
    cells: list[Cell] = []
    measure_counts: dict[str, int] = {}
    qcounts: dict[str, int] = {}

    # Per-method run-count override: a listed method uses its own run_id list,
    # any other method falls back to the plan's `run_id`. (Lets slow/
    # deterministic baselines run once while sampling methods get more seeds.)
    per_method_runs = plan.get("run_id_by_method") or {}
    for m in per_method_runs:
        if m not in run_substrate:
            sys.exit(f"run_id_by_method names unknown method '{m}'")

    for wid in plan["workloads"]:
        if wid not in catalog:
            sys.exit(f"plan references unknown workload '{wid}'")
        scenario = catalog[wid]
        dataset = scenario["dataset_id"]
        man = manifest_path(prepared_root, dataset)
        if not man.is_file():
            sys.exit(f"prepared dataset missing for workload '{wid}': {man}\n"
                     f"Prepare it first (scripts/prepare_dataset.py).")
        if dataset not in measure_counts:
            measure_counts[dataset] = len(json.loads(man.read_text())["measures"])
        csv_path = ensure_workload(gen_tool, scenario, man, workloads_dir)
        qcounts[wid] = workload_query_count(csv_path)

        for method in plan["runs"]:
            if method not in run_substrate:
                sys.exit(f"unknown run/method '{method}' in plan")
            substrate = run_substrate[method]
            approx = method in approximate_runs
            run_ids = per_method_runs.get(method, plan["run_id"])
            for nm in plan["nm"]:
                if nm > measure_counts[dataset]:
                    continue  # dataset has fewer measures than this nm
                for strv in plan["str"]:
                    for mem in plan["mem"]:
                        for run_id in run_ids:
                            mq = plan["max_queries"]
                            qcount = qcounts[wid] if mq in (None, 0) else min(mq, qcounts[wid])
                            # Approximate runs fan out over eb; exact runs emit
                            # once with no err component in the key.
                            eb_values = plan["eb"] if approx else [None]
                            for eb in eb_values:
                                key = cell_key(nm, mem, strv, qcount,
                                               eb if eb is not None else 0.0, approx)
                                cells.append(Cell(
                                    method=method, substrate=substrate,
                                    dataset=dataset, workload=wid, nm=nm,
                                    eb=eb, strv=strv, mem=mem, run_id=run_id,
                                    max_queries=mq, confidence=plan["confidence"],
                                    approx=approx, key=key, qcount=qcount))
    return cells


# --- execution ---------------------------------------------------------------

def cell_is_fresh(runmeta: Path, current_version: str, stale: bool) -> bool:
    """A cell counts as up to date when its outputs exist and (under --stale)
    were produced by the current engine build."""
    if not runmeta.is_file():
        return False
    if not stale:
        return True
    try:
        recorded = json.loads(runmeta.read_text()).get("engine_build_version")
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == current_version


def drop_caches(hook: str | None) -> bool:
    """Run the configured page-cache drop hook. Returns True when the cache was
    actually dropped (so the run is genuinely cold)."""
    if not hook:
        return False
    subprocess.run(hook, shell=True, check=True)
    return True


def column_files(manifest: Path) -> list[Path]:
    """The on-disk column files (dimensions + measures) a cell will read.

    These memory-mapped columns are the per-query cost the experiment measures;
    evicting exactly them before a cell makes its reads cold without disturbing
    the rest of the machine. Resolved relative to the manifest's directory."""
    try:
        m = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    base = manifest.parent
    cols = list(m.get("dimensions", [])) + list(m.get("measures", []))
    return [base / c["file"] for c in cols if c.get("file")]


def evict_from_cache(paths: list[Path]) -> bool:
    """Drop the given files from the OS page cache without privilege.

    ``posix_fadvise(POSIX_FADV_DONTNEED)`` releases the clean, non-resident
    pages backing each file, so the next process re-reads them from disk. Only
    the named files are touched -- no global flush, no effect on other users'
    working sets -- which is the right granularity for measuring the cold-read
    cost of the dataset under test. Returns True if at least one file was
    advised. Best-effort: unreadable files are skipped.
    """
    advised = False
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            advised = True
        except OSError:
            pass
        finally:
            os.close(fd)
    return advised


# Default cgroup launcher: a transient systemd scope whose memory controller
# charges page cache (including the memory-mapped columns) against the cap and
# forbids swap, so a run that needs more resident memory than the cap is killed
# rather than silently spilling to disk-backed cache. The `--user` form goes
# through the caller's own systemd user manager and needs no privilege where
# the memory controller is delegated to the user (the common cgroup-v2 case);
# a system-wide `systemd-run --scope` (without `--user`) needs root/polkit.
# Override either way with --mem-launcher / A3I_MEM_LAUNCH.
DEFAULT_MEM_LAUNCHER = (
    "systemd-run --user --scope -q -p MemoryMax={bytes} -p MemorySwapMax=0 --")


def render_mem_launcher(template: str | None, cap: int) -> list[str]:
    """Expand the memory-launcher template for a byte cap into argv tokens.

    The template must contain ``{bytes}``. A capped run without a working
    launcher cannot be enforced honestly, so the caller validates availability
    up front; here we only substitute and tokenize."""
    if not template:
        sys.exit("a memory-capped cell was requested but no --mem-launcher is "
                 "configured (set --mem-launcher or A3I_MEM_LAUNCH)")
    return shlex.split(template.format(bytes=cap))


def run_cell(a3i_run: Path, cell: Cell, manifest: Path, workload_csv: Path,
             qresults: Path, runmeta: Path, cold: bool,
             mem_launcher: list[str] | None, sort_gather: str = "auto") -> None:
    qresults.parent.mkdir(parents=True, exist_ok=True)
    tmp_q = qresults.with_suffix(qresults.suffix + ".tmp")
    tmp_m = runmeta.with_suffix(runmeta.suffix + ".tmp")
    cmd = [str(a3i_run),
           "--manifest", str(manifest),
           "--workload", str(workload_csv),
           "--method", cell.method,
           "--qresults", str(tmp_q),
           "--runmeta", str(tmp_m),
           "--dataset", cell.dataset,
           "--workload-name", cell.workload,
           "--num-measures", str(cell.nm),
           "--confidence", repr(float(cell.confidence)),
           "--refinement-threshold", str(cell.strv),
           "--run-id", str(cell.run_id),
           "--cold", "true" if cold else "false"]
    if cell.eb is not None:
        cmd += ["--error-bound", repr(float(cell.eb))]
    if cell.max_queries not in (None, 0):
        cmd += ["--max-queries", str(cell.max_queries)]

    # Storage backing is derived from the mem axis: `inmem` loads measures
    # resident, every other setting memory-maps them. The `mem<MemTag>` cell-key
    # segment keeps the two backings in separate cells, so they are never pooled.
    if mem_is_in_memory(cell.mem):
        cmd += ["--in-memory"]

    # Gather sorting: 'auto' ties it to the storage mode -- off when measures
    # are resident (the sort buys nothing and costs a per-round sort), on when
    # they are memory-mapped (ascending row ids keep the fault stream
    # near-sequential). 'on'/'off' force it. Recorded in runmeta; the validator
    # refuses to pool cells that differ on it.
    if sort_gather == "off" or (sort_gather == "auto" and
                                mem_is_in_memory(cell.mem)):
        cmd += ["--no-sort-gather"]

    cap = mem_bytes(cell.mem)
    if cap is not None:
        # The cap is enforced by a cgroup launcher so that page cache for the
        # memory-mapped columns is charged against the limit, not just the
        # process address space.
        cmd = render_mem_launcher(mem_launcher, cap) + cmd

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        tmp_q.unlink(missing_ok=True)
        tmp_m.unlink(missing_ok=True)
        raise
    os.replace(tmp_q, qresults)
    os.replace(tmp_m, runmeta)


# --- run log (durable, append-as-you-go) + completeness report ---------------

def _runlog_path(results_root: Path, plan: str, utc: str) -> Path:
    """Per-run append-only log under `<results>/_runlog/<plan>_<stamp>.jsonl`.

    Written one line per cell as it is dispositioned (not once at the end), so a
    run killed externally still leaves a record of every cell attempted."""
    d = results_root / "_runlog"
    d.mkdir(parents=True, exist_ok=True)
    stamp = utc.replace(":", "").replace("-", "")
    return d / f"{plan}_{stamp}.jsonl"


def _latest_runlog(results_root: Path, plan: str) -> Path | None:
    d = results_root / "_runlog"
    cands = sorted(d.glob(f"{plan}_*.jsonl")) if d.is_dir() else []
    return cands[-1] if cands else None


def report_completeness(cells, results_root: Path, plan: str) -> int:
    """Compare the plan's expected cells against the `qresults` files on disk
    (atomic writes => a present file is a complete cell, so this is robust to any
    termination). Missing cells are labelled from the latest run log when it
    explains them (error/oom) and `never-run` otherwise. Returns 1 if incomplete.
    """
    present, missing = [], []
    for cell in cells:
        q, _ = cell_paths(results_root, cell.dataset, cell.workload,
                          cell.substrate, cell.method, cell.key, cell.run_id)
        rel = str(q.relative_to(results_root))
        (present if q.is_file() else missing).append(rel)

    why: dict[str, str] = {}
    lg = _latest_runlog(results_root, plan)
    if lg:
        for line in lg.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "path" in rec:
                why[rec["path"]] = rec.get("cause") or rec.get("status") or ""

    print(f"plan '{plan}': {len(cells)} expected, {len(present)} present, "
          f"{len(missing)} missing")
    for rel in missing:
        print(f"  MISSING ({why.get(rel) or 'never-run'}): {rel}")
    if lg:
        print(f"(missing reasons cross-referenced from {lg})")
    return 1 if missing else 0


# Dev/test plans excluded from --all; run them explicitly by name.
DEV_PLANS = {"_defaults", "quicktest", "smoke"}


def _run_plan(plan_id, cells, args, results_root, prepared_root, workloads_dir,
              a3i_run, version, hook):
    """Run (or skip) every cell of one plan, appending each disposition to that
    plan's durable run log as it finishes. Returns (done, skipped, errored)."""
    done = skipped = errored = 0
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    runlog = open(_runlog_path(results_root, plan_id, utc), "w")

    def logline(rec: dict) -> None:
        runlog.write(json.dumps(rec) + "\n")
        runlog.flush()

    try:
        logline({"plan": plan_id, "engine_build_version": version,
                 "utc": utc, "total_cells": len(cells)})
        for cell in cells:
            qresults, runmeta = cell_paths(results_root, cell.dataset, cell.workload,
                                           cell.substrate, cell.method, cell.key,
                                           cell.run_id)
            rel = str(qresults.relative_to(results_root))
            if not args.force and cell_is_fresh(runmeta, version, args.stale):
                skipped += 1
                logline({"path": rel, "status": "skipped"})
                continue
            manifest = manifest_path(prepared_root, cell.dataset)
            workload_csv = workloads_dir / f"{cell.workload}.csv"
            # Cold by default: evict this cell's columns (unprivileged), and also
            # flush the whole machine when a hook is configured. --warm skips both.
            cold = False
            if not args.warm:
                evicted = evict_from_cache(column_files(manifest))
                dropped = drop_caches(hook)
                cold = evicted or dropped
            try:
                run_cell(a3i_run, cell, manifest, workload_csv, qresults, runmeta,
                         cold, args.mem_launcher, sort_gather=args.sort_gather)
                done += 1
                print(f"ran: {cell.method:14s} {rel}")
                logline({"path": rel, "status": "done"})
            except subprocess.CalledProcessError as e:
                errored += 1
                cause = "oom" if mem_bytes(cell.mem) is not None else "error"
                print(f"FAILED ({cause}): {cell.method:14s} {rel}\n{e.stderr}",
                      file=sys.stderr)
                logline({"path": rel, "status": "error", "cause": cause})
    finally:
        runlog.close()

    print(f"plan '{plan_id}': {len(cells)} cells "
          f"({done} run, {skipped} skipped, {errored} error)  "
          f"log: {_runlog_path(results_root, plan_id, utc)}")
    return done, skipped, errored


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one or more A3I experiment plans.")
    ap.add_argument("--plan", nargs="+",
                    help="one or more plan ids under experiments/plans/")
    ap.add_argument("--all", action="store_true", dest="all_plans",
                    help="run every plan except _defaults/quicktest/smoke")
    ap.add_argument("--plans-dir", default=None,
                    help="plan directory (default: experiments/plans)")
    ap.add_argument("--workload-config-dir", default=None,
                    help="workload catalog directory (default: configs/workloads)")
    ap.add_argument("--prepared-root", default=None,
                    help="prepared-dataset root (default: data/prepared)")
    ap.add_argument("--results-root", default=None,
                    help="output root (default: experiments/results)")
    ap.add_argument("--workloads-dir", default=None,
                    help="workload-CSV dir (default: experiments/workloads)")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate cells and print dispositions; run nothing")
    ap.add_argument("--report", action="store_true",
                    help="report which expected cells are present vs missing on "
                         "disk (with the reason from the run log) and exit; "
                         "run nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells whose outputs already exist")
    ap.add_argument("--stale", action="store_true",
                    help="re-run cells produced by a different engine build")
    ap.add_argument("--warm", action="store_true",
                    help="skip cache eviction entirely; records cold=false "
                         "(dev only -- columns may be served from a warm cache)")
    ap.add_argument("--sort-gather", choices=["auto", "on", "off"],
                    default="auto",
                    help="measure-gather sort policy. 'auto' (default): OFF for "
                         "in-memory (inmem) cells, where resident columns make "
                         "the sort pure overhead, and ON for memory-mapped "
                         "cells, where ascending row ids keep the page-fault "
                         "stream near-sequential. 'on'/'off' force one policy "
                         "for every cell. The choice is recorded in runmeta; "
                         "the validator refuses to pool cells that differ on it.")
    ap.add_argument("--drop-caches-cmd", default=os.environ.get("A3I_DROP_CACHES_CMD"),
                    help="optional shell hook run before each cell to flush the "
                         "WHOLE-MACHINE page cache (e.g. a sudo-granted helper "
                         "writing vm.drop_caches), in addition to the built-in "
                         "per-dataset eviction; for a stricter cold baseline")
    ap.add_argument("--mem-launcher",
                    default=os.environ.get("A3I_MEM_LAUNCH", DEFAULT_MEM_LAUNCHER),
                    help="argv template wrapping each memory-capped run in a "
                         "cgroup; must contain {bytes} (default: a transient "
                         "systemd scope with MemoryMax)")
    args = ap.parse_args()
    if bool(args.plan) == bool(args.all_plans):
        ap.error("specify exactly one of --plan <id...> or --all")

    prepared_root = a3i_config.prepared_root(args.prepared_root)
    results_root  = a3i_config.results_root(args.results_root)
    workloads_dir = a3i_config.workloads_dir(args.workloads_dir)
    workloads_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    a3i_run = find_tool("a3i_run")
    gen_tool = find_tool("generate_workload")
    version = engine_version(a3i_run)
    run_substrate, approximate_runs = describe_methods(a3i_run)

    plans_dir = Path(args.plans_dir) if args.plans_dir else PLANS_DIR
    config_dir = Path(args.workload_config_dir) if args.workload_config_dir else WORKLOAD_CONFIG_DIR
    catalog = load_workload_catalog(config_dir)

    if args.all_plans:
        plan_ids = sorted(p.stem for p in plans_dir.glob("*.yaml")
                          if p.stem not in DEV_PLANS)
        if not plan_ids:
            sys.exit(f"no plans found under {plans_dir}")
    else:
        plan_ids = args.plan

    # Cold by default: before each cell the dataset's columns are evicted from the
    # page cache (unprivileged posix_fadvise), so reads come from disk not a warm
    # cache; an optional --drop-caches-cmd also flushes the whole machine. --warm
    # skips eviction (records cold=false).
    hook = None if args.warm else args.drop_caches_cmd

    # Plans run in sequence; cells already on disk are skipped, so cells shared
    # across plans are computed once (cross-plan dedup is automatic).
    tot = [0, 0, 0]   # done, skipped, errored
    report_rc = 0
    for plan_id in plan_ids:
        plan = load_plan(plan_id, plans_dir)
        cells = enumerate_cells(plan, catalog, prepared_root, workloads_dir,
                                gen_tool, run_substrate, approximate_runs)
        if args.report:
            report_rc |= report_completeness(cells, results_root, plan_id)
            continue
        if args.dry_run:
            would = 0
            for cell in cells:
                qresults, runmeta = cell_paths(results_root, cell.dataset, cell.workload,
                                               cell.substrate, cell.method, cell.key,
                                               cell.run_id)
                if not args.force and cell_is_fresh(runmeta, version, args.stale):
                    continue
                print(f"would run: {cell.method:14s} "
                      f"{qresults.relative_to(results_root)}")
                would += 1
            print(f"plan '{plan_id}': {len(cells)} cells "
                  f"({would} would-run, {len(cells) - would} present)")
            continue
        d, s, e = _run_plan(plan_id, cells, args, results_root, prepared_root,
                            workloads_dir, a3i_run, version, hook)
        tot[0] += d; tot[1] += s; tot[2] += e

    if args.report or args.dry_run:
        return report_rc
    if len(plan_ids) > 1:
        print(f"\nALL {len(plan_ids)} plans: {tot[0]} run, {tot[1]} skipped, "
              f"{tot[2]} error")
    return 1 if tot[2] else 0


if __name__ == "__main__":
    raise SystemExit(main())
