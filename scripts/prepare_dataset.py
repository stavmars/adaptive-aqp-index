#!/usr/bin/env python3
"""Prepare A3I datasets from committed configs.

Drives the offline data-preparation pipeline for one, several, or all dataset
configs under ``configs/datasets/``. For each config it:

  1. ensures the dataset's Parquet exists -- a ``parquet:`` source points at a
     file produced once by ``tools/csv_to_parquet``; a ``generator:`` source is
     produced on demand by ``generate_dataset``;
  2. converts Parquet -> binary columns via ``convert_parquet_to_columns``,
     applying the dataset's ``drop_if`` predicates and dimension bounds;
  3. verifies the prepared dataset -- the manifest resolves, every column file
     is the right size, and (unless ``--verify structural``) the recomputed
     global stats match the manifest.

Preparation is idempotent: a dataset whose prepared ``manifest.json`` already
records the current source Parquet's ``(source_bytes, source_mtime)`` is left
untouched unless ``--force`` is passed. A ``generator:`` Parquet that already
exists is likewise reused unless ``--force``.

Usage:
    scripts/prepare_dataset.py <config.yaml> [<config.yaml> ...]
    scripts/prepare_dataset.py --all
    scripts/prepare_dataset.py configs/datasets/taxi.yaml --max-rows 100000

Options:
    --all                 Prepare every configs/datasets/*.yaml.
    --prepared-root DIR   Output root for prepared datasets (default: data/prepared,
                          or $A3I_PREPARED_ROOT). Point this at a fast disk.
    --parquet-dir DIR     Root for Parquet files (default: data/raw, or
                          $A3I_PARQUET_DIR). Generator Parquet is written here;
                          a relative ``parquet:`` source path is resolved here.
    --max-rows N          Single-config only: prepare the first N surviving rows
                          as a subset dataset "<id>_<N>" (records parent + max_rows).
    --force               Re-generate and re-convert even if up to date.
    --verify {full,structural,none}
                          full (default) recomputes global stats from the binary
                          columns; structural only checks file sizes/manifest;
                          none skips verification.
    --bin-dir DIR         Directory holding the built tools (default: build/tools).

Environment:
    A3I_BUILD_DIR         Build directory whose tools/ holds the executables
                          (overridden by --bin-dir).
    A3I_PREPARED_ROOT     Default for --prepared-root.
    A3I_PARQUET_DIR       Default for --parquet-dir.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("PyYAML is required: pip install pyyaml")

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- tool resolution ---------------------------------------------------------

def find_tool(name: str, bin_dir: Path | None) -> Path:
    candidates = []
    if bin_dir:
        candidates.append(bin_dir / name)
    env_build = os.environ.get("A3I_BUILD_DIR")
    if env_build:
        candidates.append(Path(env_build) / "tools" / name)
    candidates.append(REPO_ROOT / "build" / "tools" / name)
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    searched = "\n  ".join(str(c) for c in candidates)
    sys.exit(
        f"{name} binary not found. Build the project first (cmake --build build), "
        f"or pass --bin-dir / set A3I_BUILD_DIR.\nSearched:\n  " + searched
    )


# --- config helpers ----------------------------------------------------------

def resolve_path(p: str) -> Path:
    """Resolve a config-relative path against the repo root if not absolute."""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def source_kind(cfg: dict) -> str:
    src = cfg.get("source") or {}
    if "generator" in src:
        return "generator"
    if "parquet" in src:
        return "parquet"
    sys.exit(f"config '{cfg.get('dataset_id')}': source must be 'generator' or 'parquet'")


def source_parquet_path(cfg: dict, parquet_dir: Path) -> Path:
    """Locate a dataset's Parquet file.

    A ``generator:`` source is written to ``<parquet_dir>/<id>.parquet``. A
    ``parquet:`` source uses the config's path verbatim when absolute, else
    resolves it under ``parquet_dir`` so a single ``--parquet-dir`` (or
    ``$A3I_PARQUET_DIR``) relocates every real dataset to another disk.
    """
    if source_kind(cfg) == "parquet":
        p = Path(cfg["source"]["parquet"])
        return p if p.is_absolute() else (parquet_dir / p)
    return parquet_dir / f"{cfg['dataset_id']}.parquet"


def dim_args(cfg: dict) -> list[str]:
    args = []
    for d in cfg.get("dimensions", []):
        lo, hi = d["bounds"]
        args += ["--dimension", f"{d['name']}:{lo}:{hi}"]
    return args


def measure_args(cfg: dict) -> list[str]:
    args = []
    for m in cfg.get("measures", []):
        args += ["--measure", str(m)]
    return args


def dropif_args(cfg: dict) -> list[str]:
    args = []
    for pred in cfg.get("drop_if", []) or []:
        # Passed verbatim: the converter parses "<name> <op> <value>" and trims
        # surrounding whitespace, so a column name with spaces is preserved.
        args += ["--drop-if", str(pred)]
    return args


# --- pipeline steps ----------------------------------------------------------

def ensure_parquet(cfg: dict, parquet: Path, gen_tool: Path, force: bool) -> None:
    kind = source_kind(cfg)
    if kind == "parquet":
        if not parquet.is_file():
            sys.exit(
                f"[{cfg['dataset_id']}] Parquet not found: {parquet}\n"
                "  Produce it once with tools/csv_to_parquet (see docs/DATA_PREP.md)."
            )
        return
    # generator
    if parquet.is_file() and not force:
        print(f"[{cfg['dataset_id']}] parquet exists, reusing: {parquet}")
        return
    gen = cfg["source"]["generator"]
    parquet.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(gen_tool), "--output", str(parquet),
           "--seed", str(gen["seed"]), "--rows", str(gen["rows"])]
    for col in gen["columns"]:
        cmd += ["--column", f"{col['name']}:{col['dist']}:{col['p0']}:{col['p1']}"]
    cmd.append("--overwrite")
    print(f"[{cfg['dataset_id']}] generating parquet ({gen['rows']} rows)...")
    subprocess.run(cmd, check=True)


def up_to_date(out_dir: Path, parquet: Path) -> bool:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file() or not parquet.is_file():
        return False
    try:
        m = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    st = parquet.stat()
    return (int(m.get("source_bytes", -1)) == st.st_size
            and int(m.get("source_mtime", -1)) == int(st.st_mtime))


def convert(cfg: dict, parquet: Path, out_dir: Path, conv_tool: Path,
            dataset_id: str, max_rows: int | None, parent_id: str | None) -> None:
    cmd = [str(conv_tool), "--input", str(parquet),
           "--dataset-id", dataset_id, "--output-dir", str(out_dir)]
    cmd += dim_args(cfg) + measure_args(cfg) + dropif_args(cfg)
    if cfg.get("null_string") is not None:
        # null_string handling lives in csv_to_parquet (text->Parquet); by the
        # time we read Parquet, "X" is already a null. Recorded here for clarity
        # only -- the converter resolves nulls from the Parquet directly.
        pass
    if max_rows is not None:
        cmd += ["--max-rows", str(max_rows)]
        if parent_id:
            cmd += ["--parent-dataset-id", parent_id]
    cmd.append("--overwrite")
    print(f"[{dataset_id}] converting -> {out_dir}")
    subprocess.run(cmd, check=True)


# --- verification ------------------------------------------------------------

def _read_column(path: Path) -> array.array:
    a = array.array("d")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":  # manifest columns are little-endian float64
        a.byteswap()
    return a


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def verify(out_dir: Path, mode: str) -> None:
    if mode == "none":
        return
    manifest_path = out_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    row_count = int(m["row_count"])
    base = manifest_path.parent

    cols = list(m.get("dimensions", [])) + list(m.get("measures", []))
    for c in cols:
        f = base / c["file"]
        if not f.is_file():
            sys.exit(f"verify: missing column file {f}")
        expected = row_count * 8
        actual = f.stat().st_size
        if actual != expected:
            sys.exit(f"verify: {f} is {actual} bytes, expected {expected}")

    if mode == "structural":
        print(f"verify(structural): {out_dir} OK ({row_count} rows, {len(cols)} cols)")
        return

    # full: recompute global stats from the binary columns and match the manifest.
    for d in m.get("dimensions", []):
        vals = _read_column(base / d["file"])
        finite = [v for v in vals if not math.isnan(v)]
        if finite:
            if not _close(min(finite), d["min"]) or not _close(max(finite), d["max"]):
                sys.exit(f"verify: dim {d['name']} min/max mismatch")

    for mz in m.get("measures", []):
        vals = _read_column(base / mz["file"])
        g = mz["global"]
        nan_n = sum(1 for v in vals if math.isnan(v))
        finite = [v for v in vals if not math.isnan(v)]
        s = math.fsum(finite)
        ssq = math.fsum(v * v for v in finite)
        if int(g["nan_count"]) != nan_n or int(g["non_nan_count"]) != len(finite):
            sys.exit(f"verify: measure {mz['name']} count mismatch")
        if not _close(s, g["sum"]) or not _close(ssq, g["sum_sq"]):
            sys.exit(f"verify: measure {mz['name']} sum/sum_sq mismatch")
        if finite and (not _close(min(finite), g["min"]) or not _close(max(finite), g["max"])):
            sys.exit(f"verify: measure {mz['name']} min/max mismatch")

    print(f"verify(full): {out_dir} OK ({row_count} rows, {len(cols)} cols)")


# --- driver ------------------------------------------------------------------

def prepare_one(config_path: Path, args, gen_tool: Path, conv_tool: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    if not cfg or "dataset_id" not in cfg:
        sys.exit(f"{config_path}: missing dataset_id")
    base_id = cfg["dataset_id"]
    parquet_dir = resolve_path(args.parquet_dir)
    prepared_root = resolve_path(args.prepared_root)
    parquet = source_parquet_path(cfg, parquet_dir)

    # Subset naming for --max-rows (single-config only).
    if args.max_rows is not None:
        dataset_id = f"{base_id}_{args.max_rows}"
        parent_id = base_id
    else:
        dataset_id = base_id
        parent_id = None

    out_dir = (resolve_path(cfg["output_dir"]) if cfg.get("output_dir")
               else prepared_root / dataset_id)

    ensure_parquet(cfg, parquet, gen_tool, args.force)

    if not args.force and args.max_rows is None and up_to_date(out_dir, parquet):
        print(f"[{dataset_id}] up to date, skipping (pass --force to rebuild)")
    else:
        convert(cfg, parquet, out_dir, conv_tool, dataset_id, args.max_rows, parent_id)

    verify(out_dir, args.verify)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare A3I datasets from configs/datasets/*.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("configs", nargs="*", help="dataset config YAML paths")
    ap.add_argument("--all", action="store_true",
                    help="prepare every configs/datasets/*.yaml")
    ap.add_argument("--prepared-root",
                    default=os.environ.get("A3I_PREPARED_ROOT", "data/prepared"))
    ap.add_argument("--parquet-dir",
                    default=os.environ.get("A3I_PARQUET_DIR", "data/raw"))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", choices=["full", "structural", "none"], default="full")
    ap.add_argument("--bin-dir", default=None)
    args = ap.parse_args()

    if args.all:
        if args.configs:
            ap.error("pass either config paths or --all, not both")
        configs = sorted((REPO_ROOT / "configs" / "datasets").glob("*.yaml"))
        if not configs:
            sys.exit("no configs found under configs/datasets/")
    else:
        if not args.configs:
            ap.error("provide one or more config paths, or --all")
        configs = [resolve_path(c) for c in args.configs]

    if args.max_rows is not None and len(configs) != 1:
        ap.error("--max-rows applies to a single config only")

    bin_dir = Path(args.bin_dir).resolve() if args.bin_dir else None
    gen_tool = find_tool("generate_dataset", bin_dir)
    conv_tool = find_tool("convert_parquet_to_columns", bin_dir)

    for cfg_path in configs:
        if not cfg_path.is_file():
            sys.exit(f"config not found: {cfg_path}")
        prepare_one(cfg_path, args, gen_tool, conv_tool)

    print(f"done: {len(configs)} dataset(s) prepared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
