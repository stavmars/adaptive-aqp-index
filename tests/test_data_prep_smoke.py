#!/usr/bin/env python3
"""Script-level smoke test for the dataset preparation pipeline.

Exercises scripts/prepare_dataset.py end to end on tiny fixtures, independently
of the C++ converter's own stat code:

  * a real (parquet:) dataset built from a hand-written CSV via csv_to_parquet,
    checking the drop_if + bounds survival rule, null_string -> NaN, measure
    order, and that the manifest's recomputed global stats match a plain-Python
    re-derivation from the CSV;
  * idempotency: a re-run is a no-op unless --force;
  * a --max-rows subset records parent_dataset_id + max_rows;
  * a generator (generator:) dataset prepares and verifies.

Run directly (python3 tests/test_data_prep_smoke.py) or via ctest
(label m_data_prep). Uses only the Python standard library + PyYAML.
"""
from __future__ import annotations

import array
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def tools_dir() -> Path:
    env = os.environ.get("A3I_BUILD_DIR")
    cands = []
    if env:
        cands.append(Path(env) / "tools")
    cands.append(REPO_ROOT / "build" / "tools")
    for c in cands:
        if (c / "convert_parquet_to_columns").is_file():
            return c
    sys.exit("built tools not found; build the project first (cmake --build build)")


TOOLS = tools_dir()


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def read_col(path: Path) -> list[float]:
    a = array.array("d")
    a.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        a.byteswap()
    return list(a)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def prepare(args: list[str]) -> subprocess.CompletedProcess:
    return run([sys.executable, str(SCRIPTS / "prepare_dataset.py"), *args])


# --- real (parquet:) path with drop_if + bounds + null_string ----------------

def test_real_path(work: Path) -> None:
    csv = work / "src.csv"
    csv.write_text(
        "x,m\n"
        "10,5\n"     # survives
        "50,X\n"     # survives, m -> NaN
        "10,-3\n"    # dropped: m < 0
        "200,7\n"    # dropped: x out of bounds
        "-5,2\n"     # dropped: x out of bounds
        "30,8\n"     # survives
    )
    parquet = work / "src.parquet"
    run([str(TOOLS / "csv_to_parquet"), "--input", str(csv), "--output", str(parquet),
         "--has-header", "--delimiter", ",", "--null-string", "X", "--overwrite"])

    cfg = work / "real.yaml"
    cfg.write_text(
        "dataset_id: real_tiny\n"
        "source:\n"
        f"  parquet: {parquet}\n"
        "dimensions:\n"
        "  - {name: x, bounds: [0, 100]}\n"
        "measures: [m]\n"
        "null_string: \"X\"\n"
        "drop_if:\n"
        "  - \"m < 0\"\n"
    )
    prepared = work / "prepared"
    prepare([str(cfg), "--prepared-root", str(prepared)])

    out = prepared / "real_tiny"
    man = json.loads((out / "manifest.json").read_text())

    # Independent re-derivation from the CSV (not via the converter).
    survivors_x, survivors_m = [], []
    for line in csv.read_text().splitlines()[1:]:
        xs, ms = line.split(",")
        x = float(xs)
        m = math.nan if ms == "X" else float(ms)
        if not (0 <= x <= 100):
            continue
        if not math.isnan(m) and m < 0:
            continue
        survivors_x.append(x)
        survivors_m.append(m)

    assert man["row_count"] == len(survivors_x) == 3, man["row_count"]
    assert man["null_encoding"] == "NaN"
    assert man["applied_drop_if"] == ["m < 0"], man["applied_drop_if"]
    assert [mz["name"] for mz in man["measures"]] == ["m"]
    assert man["domain_bounds"]["low"] == [0.0] and man["domain_bounds"]["high"] == [100.0]

    base = out
    x_col = read_col(base / man["dimensions"][0]["file"])
    m_col = read_col(base / man["measures"][0]["file"])
    assert x_col == survivors_x, (x_col, survivors_x)
    assert [("nan" if math.isnan(v) else v) for v in m_col] == \
           [("nan" if math.isnan(v) else v) for v in survivors_m]

    finite = [v for v in survivors_m if not math.isnan(v)]
    g = man["measures"][0]["global"]
    assert g["non_nan_count"] == len(finite) == 2
    assert g["nan_count"] == 1
    assert approx(g["sum"], math.fsum(finite))
    assert approx(g["sum_sq"], math.fsum(v * v for v in finite))
    assert approx(g["min"], min(finite)) and approx(g["max"], max(finite))
    print("real-path survival/null/stats: OK")

    # Idempotency: second run is a no-op; --force rebuilds.
    again = prepare([str(cfg), "--prepared-root", str(prepared)])
    assert "up to date" in again.stdout, again.stdout
    forced = prepare([str(cfg), "--prepared-root", str(prepared), "--force"])
    assert "converting" in forced.stdout, forced.stdout
    print("idempotency / --force: OK")

    # --max-rows subset records parent + cap.
    prepare([str(cfg), "--prepared-root", str(prepared), "--max-rows", "2"])
    sub = json.loads((prepared / "real_tiny_2" / "manifest.json").read_text())
    assert sub["dataset_id"] == "real_tiny_2"
    assert sub["row_count"] == 2
    assert sub["parent_dataset_id"] == "real_tiny"
    assert sub["max_rows"] == 2
    print("--max-rows subset: OK")


# --- generator (generator:) path ---------------------------------------------

def test_generator_path(work: Path) -> None:
    cfg = work / "gen.yaml"
    cfg.write_text(
        "dataset_id: gen_tiny\n"
        "source:\n"
        "  generator:\n"
        "    seed: 7\n"
        "    rows: 500\n"
        "    columns:\n"
        "      - {name: col0, dist: uniform, p0: 0, p1: 1000}\n"
        "      - {name: col1, dist: uniform, p0: 0, p1: 1000}\n"
        "      - {name: col2, dist: normal, p0: 50, p1: 15}\n"
        "dimensions:\n"
        "  - {name: col0, bounds: [0, 1000]}\n"
        "  - {name: col1, bounds: [0, 1000]}\n"
        "measures: [col2]\n"
    )
    prepared = work / "prepared"
    raw = work / "raw"
    res = prepare([str(cfg), "--prepared-root", str(prepared), "--parquet-dir", str(raw)])
    assert "generating parquet" in res.stdout, res.stdout
    man = json.loads((prepared / "gen_tiny" / "manifest.json").read_text())
    assert man["row_count"] == 500
    assert man["source_bytes"] > 0 and man["source_mtime"] > 0
    print("generator-path: OK")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a3i_mdata_smoke_") as d:
        work = Path(d)
        test_real_path(work)
        test_generator_path(work)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
