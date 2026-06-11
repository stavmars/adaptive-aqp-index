#!/usr/bin/env python3
"""Produce a row-shuffled copy of a dataset's Parquet file.

Source exports often carry a strong row order (by time, location, or
catalogue), which leaks into any row-order-sensitive measurement. Shuffling
the file once at onboarding removes that variable for every consumer.

The shuffle is deterministic and avoids a global sort: a streaming pass
scatters rows into hash-assigned buckets, then each bucket is permuted in
memory with a seeded generator and appended to the output. The same
(input, seed, buckets) always yields the same file.

Usage:
    scripts/shuffle_parquet.py INPUT.parquet OUTPUT.parquet [--seed N]
                               [--buckets N] [--threads N]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deterministically row-shuffle a Parquet file.")
    ap.add_argument("input", type=Path, help="source Parquet (unchanged)")
    ap.add_argument("output", type=Path, help="shuffled Parquet to write")
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed; same (input, seed, buckets) -> same "
                         "output (default: %(default)s)")
    ap.add_argument("--buckets", type=int, default=256,
                    help="scatter buckets; each bucket (~rows/buckets) must "
                         "fit in memory when read back (default: %(default)s)")
    ap.add_argument("--threads", type=int, default=None,
                    help="DuckDB threads for the scatter pass")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"input not found: {args.input}")
    if args.output.exists() and not args.force:
        sys.exit(f"output exists (use --force to overwrite): {args.output}")
    if args.output.resolve() == args.input.resolve():
        sys.exit("output must differ from input")

    import duckdb
    import numpy as np
    import pyarrow.parquet as pq

    scatter_dir = args.output.parent / (args.output.stem + ".scatter_tmp")
    if scatter_dir.exists():
        shutil.rmtree(scatter_dir)

    con = duckdb.connect()
    if args.threads:
        con.execute(f"SET threads={int(args.threads)}")

    src = str(args.input).replace("'", "''")
    t0 = time.time()
    n_in = con.execute(
        f"SELECT count(*) FROM read_parquet('{src}')").fetchone()[0]
    print(f"input rows: {n_in:,}", flush=True)

    # Phase 1: streaming scatter. file_row_number gives every row a stable
    # ordinal in the source, so the bucket assignment is a deterministic
    # function of (row, seed). PARTITION_BY keeps one writer per bucket and
    # never sorts; the partition column lives in the directory name, not in
    # the bucket files.
    B = int(args.buckets)
    con.execute(f"""
        COPY (
            SELECT *,
                   hash(file_row_number + {int(args.seed)}) % {B} AS __bucket
            FROM read_parquet('{src}', file_row_number=true)
        ) TO '{scatter_dir}' (FORMAT PARQUET, PARTITION_BY (__bucket))
    """)
    print(f"scatter pass done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # Phase 2: read the buckets back in fixed order, permute each in memory
    # with a per-bucket seeded generator, and append to one output file.
    writer = None
    n_out = 0
    try:
        for b in range(B):
            bdir = scatter_dir / f"__bucket={b}"
            if not bdir.is_dir():
                continue  # an empty bucket writes no directory
            # The scatter pass appends to bucket files in a thread-dependent
            # order, so restore a canonical order first: sort the bucket by
            # the rows' original file position (kept through the scatter for
            # exactly this purpose), then apply the seeded permutation. This
            # makes the output a pure function of (input, seed, buckets).
            table = pq.read_table(bdir)
            table = table.sort_by("file_row_number")
            table = table.drop_columns(["file_row_number"])
            rng = np.random.RandomState(int(args.seed) * 1_000_003 + b)
            table = table.take(rng.permutation(table.num_rows))
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema,
                                          compression="zstd")
            writer.write_table(table)
            n_out += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    shutil.rmtree(scatter_dir)

    if n_out != n_in:
        sys.exit(f"row count mismatch: wrote {n_out:,}, expected {n_in:,}")
    print(f"wrote {args.output} ({n_out:,} rows, seed={args.seed}, "
          f"buckets={B}, {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
