#!/usr/bin/env python3
"""
Download the Gaia DR3 spatial benchmark as CSV via the ESA Gaia Archive TAP
service.

Prerequisites:
  1. pip install astroquery astropy pandas
  2. No account needed (public data); --login keeps async results longer.

Strategy:
  - Deterministic subsample via MOD(source_id, N) = 0.
  - Partitioned into 1-degree RA strips (360 async TAP jobs) to avoid timeouts.
  - Each strip saved as a headerless, comma-delimited CSV chunk.
  - Chunks concatenated into the final output file.
  - NULLs written as empty fields (read downstream as NaN).

Columns (14, no source_id -- a 19-digit long that does not fit float32):
  #  Column                     Description
  0  ra                         Right Ascension (degrees)
  1  dec                        Declination (degrees)
  2  phot_g_mean_mag            G-band magnitude
  3  phot_bp_mean_mag           Blue photometer magnitude
  4  phot_rp_mean_mag           Red photometer magnitude
  5  parallax                   Parallax (mas)
  6  pmra                       Proper motion in RA
  7  pmdec                      Proper motion in Dec
  8  ruwe                       Astrometric quality (renormalised unit weight error)
  9  phot_g_mean_flux           G-band flux
  10 phot_g_mean_flux_error     G-band flux uncertainty
  11 phot_bp_mean_flux          BP flux
  12 phot_rp_mean_flux          RP flux
  13 astrometric_excess_noise   Astrometric residual


Sampling rule: MOD(source_id, 10) = 0 yields a ~10% sample (~180M rows).
  - Preserves the spatial distribution (source_id encodes HEALPix in high bits).
  - Deterministic and reproducible for any Gaia DR3 mirror.

Estimated CSV sizes (~160 bytes/row average):
  --sample-mod  5  (20%): ~362M rows -> ~58 GB
  --sample-mod 10  (10%): ~181M rows -> ~29 GB   (default)
  --sample-mod 20   (5%):  ~90M rows -> ~14.5 GB
  --sample-mod 50   (2%):  ~36M rows -> ~5.8 GB
  --sample-mod 100  (1%):  ~18M rows -> ~2.9 GB

After download, shuffle the data rows once while keeping the header (e.g.
`{ head -1 in.csv; tail -n +2 in.csv | shuf; } > out.csv`) and run
tools/csv_to_parquet --has-header to produce the Parquet.

Usage:
  python download_gaia_benchmark.py                   # 10% sample (default)
  python download_gaia_benchmark.py --sample-mod 20   # 5% sample
  python download_gaia_benchmark.py --sample-mod 100  # 1% sample
  python download_gaia_benchmark.py --login           # authenticated mode
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

try:
    from astroquery.gaia import Gaia
    import pandas as pd  # noqa: F401  (used via DataFrame.to_csv)
except ImportError:
    print("ERROR: Required packages not found.")
    print("Install with:  pip install astroquery astropy pandas")
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning, module='astropy')


# 14 columns (no source_id), ordered so truncation from the end stays valid.
# This same list is written as the CSV header row (see concatenate_chunks).
COLUMN_NAMES = [
    "ra", "dec",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "parallax", "pmra", "pmdec", "ruwe",
    "phot_g_mean_flux", "phot_g_mean_flux_error",
    "phot_bp_mean_flux", "phot_rp_mean_flux",
    "astrometric_excess_noise",
]
COLUMNS = ", ".join(COLUMN_NAMES)

QUERY_TEMPLATE = (
    "SELECT {columns} "
    "FROM gaiadr3.gaia_source "
    "WHERE ra >= {ra_lo} AND ra < {ra_hi} "
    "AND MOD(source_id, {modulus}) = 0"
)


def submit_and_download_strip(ra_lo, ra_hi, modulus, output_path, max_retries=3):
    """Submit one RA-strip query as an async TAP job and save as headerless CSV.

    Returns the number of rows written.
    """
    query = QUERY_TEMPLATE.format(
        columns=COLUMNS, ra_lo=ra_lo, ra_hi=ra_hi, modulus=modulus)

    tbl = None
    for attempt in range(max_retries):
        try:
            job = Gaia.launch_job_async(query)
            tbl = job.get_results()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"\n    Attempt {attempt + 1} failed ({e}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    n_rows = len(tbl)
    if n_rows == 0:
        Path(output_path).touch()
        return 0

    # Masked (NULL) values become NaN; written as empty CSV fields.
    df = tbl.to_pandas()
    df.to_csv(output_path, index=False, header=False, na_rep='')
    return n_rows


def count_lines(filepath):
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count


def concatenate_chunks(chunk_dir, output_file):
    chunk_files = sorted(chunk_dir.glob("strip_*.csv"))
    print(f"\nConcatenating {len(chunk_files)} chunks into {output_file} ...")
    total_rows = 0
    with open(output_file, 'wb') as out:
        # Header row first; the per-strip chunks are headerless data.
        out.write((",".join(COLUMN_NAMES) + "\n").encode("ascii"))
        for cf in chunk_files:
            if cf.stat().st_size > 0:
                with open(cf, 'rb') as inp:
                    out.write(inp.read())
                total_rows += count_lines(cf)
    total_size = output_file.stat().st_size
    print(f"Done. {total_rows:,} data rows (+1 header), "
          f"{total_size / 1e9:.1f} GB")
    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Download the Gaia DR3 spatial benchmark via TAP")
    parser.add_argument(
        "--ra-step", type=float, default=1.0,
        help="RA strip width in degrees (default: 1.0)")
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/gaia_dr3/chunks",
        help="Directory for intermediate chunk files")
    parser.add_argument(
        "--output-file", type=str,
        default="data/raw/gaia_dr3/gaia_dr3_benchmark.csv",
        help="Final concatenated CSV path")
    parser.add_argument(
        "--sample-mod", type=int, default=10,
        help="Modulus N for MOD(source_id, N)=0 sampling "
             "(10->10%%, 20->5%%, 100->1%%; default: 10)")
    parser.add_argument(
        "--login", action="store_true",
        help="Login to the Gaia archive (keeps async results longer)")
    parser.add_argument(
        "--resume-from", type=float, default=0.0,
        help="Resume from this RA value (skip earlier strips)")
    args = parser.parse_args()

    chunk_dir = Path(args.output_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    Gaia.ROW_LIMIT = -1  # remove the default sync row cap

    if args.login:
        Gaia.login()
        print("Authenticated to the Gaia archive.")
    else:
        print("Using public (anonymous) access. "
              "Add --login for persistent server-side results.")

    sample_pct = 100.0 / args.sample_mod
    print(f"\nDownloading gaiadr3.gaia_source "
          f"({sample_pct:.1f}% sample, MOD(source_id, {args.sample_mod}) = 0)")
    print(f"RA strip width: {args.ra_step} deg  |  Chunks -> {chunk_dir}/\n")

    ra = args.resume_from
    session_rows = 0
    skipped = 0
    strip_idx = int(ra / args.ra_step)

    while ra < 360.0:
        ra_lo = ra
        ra_hi = min(ra + args.ra_step, 360.0)
        chunk_path = (chunk_dir /
                      f"strip_{strip_idx:04d}_ra{ra_lo:.1f}-{ra_hi:.1f}.csv")

        if chunk_path.exists() and chunk_path.stat().st_size > 100:
            skipped += 1
            print(f"  [{strip_idx:4d}] RA [{ra_lo:6.1f}, {ra_hi:6.1f}) "
                  f"-- exists, skipping")
            ra = ra_hi
            strip_idx += 1
            continue

        print(f"  [{strip_idx:4d}] RA [{ra_lo:6.1f}, {ra_hi:6.1f}) ",
              end="", flush=True)
        try:
            t0 = time.time()
            n = submit_and_download_strip(
                ra_lo, ra_hi, args.sample_mod, chunk_path)
            elapsed = time.time() - t0
            session_rows += n
            print(f"-> {n:,} rows in {elapsed:.0f}s  "
                  f"(session: {session_rows:,})")
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"    Resume with: --resume-from {ra_lo}")
            sys.exit(1)

        ra = ra_hi
        strip_idx += 1
        time.sleep(2)  # be polite to the archive

    print(f"\nAll strips downloaded. "
          f"Session: {session_rows:,} new rows, {skipped} skipped.")
    total = concatenate_chunks(chunk_dir, Path(args.output_file))
    print(f"\nFinal dataset: {total:,} rows")


if __name__ == "__main__":
    main()
