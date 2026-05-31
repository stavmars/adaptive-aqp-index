# Preparing the benchmark datasets

This document records how the datasets used in the experiments are prepared,
so the process can be reproduced. It covers the Parquet
step, the dataset configs under `configs/datasets/`, and the
`scripts/prepare_dataset.py` orchestrator. For the on-disk manifest and binary
column format consumed by the engine, see `DATA_FORMAT.md`.

## Overview

Every dataset has one source **Parquet file** — that file
*is* the dataset. Preparation always follows the same three steps:

1. **ensure the Parquet exists** — real datasets point at a Parquet (e.g., produced
   by `tools/csv_to_parquet`); synthetic datasets are generated on demand by
   the seeded `generate_dataset` tool;
2. **convert Parquet → binary columns** via `convert_parquet_to_columns`,
   applying the dataset's dimension bounds and `drop_if` predicates;
3. **verify** — the manifest resolves, column files are the right size, and the
   recomputed global stats match the manifest.

`scripts/prepare_dataset.py` performs steps 2–3 (and step 1 for generator
sources). In case of a source dataset in CSV format, a raw text → Parquet step (`csv_to_parquet`) is run once per real
dataset and is kept separate from the orchestrator.

## Build the tools first

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

This produces the executables the scripts call: `build/tools/csv_to_parquet`,
`build/tools/generate_dataset`, `build/tools/convert_parquet_to_columns`. The
scripts find them automatically under `build/tools`, or set `A3I_BUILD_DIR` (or
pass `--bin-dir`) for a non-default build location.

## Choosing where outputs land

The pipeline touches three kinds of files, with very different access patterns:

| what | env var | CLI flag | default | recommended storage |
| --- | --- | --- | --- | --- |
| prepared binary columns + manifests | `A3I_PREPARED_ROOT` | `--prepared-root` | `data/prepared` | **fast SSD** (query hot path: mmap gather) |
| generator Parquet root | `A3I_PARQUET_DIR` | `--parquet-dir` | `data/raw` | any disk (read once at convert time) |
| real datasets' Parquet | (in the config) | (in the config) | absolute path in each config | any disk; HDD is fine |

The prepared columns are the engine's hot path — every query mmaps and gathers
from them — so put `--prepared-root` on the fastest disk you have. Parquet is
only read once (to convert to binary) and later by the external baselines
(DuckDB/PilotDB) off the measured clock, so it can live on slower/larger
storage. Keep raw CSV/TSV sources wherever is convenient; they are read once.

Each real dataset's `source.parquet` is an **absolute path written in its
config**, so different datasets can live on different disks. `--parquet-dir`
(or `A3I_PARQUET_DIR`) only governs where the **generator** writes synthetic
Parquet (and resolves any relative `source.parquet`). For example, with a fast
SSD mounted for the hot binary columns and a larger disk for the Parquet:

```sh
export A3I_PARQUET_DIR=/path/on/large/disk/parquet    # synth Parquet output
export A3I_PREPARED_ROOT=/path/on/fast/ssd/prepared   # binary columns (fast SSD)
python3 scripts/prepare_dataset.py --all
```

## Dataset catalog

Configs live under `configs/datasets/<id>.yaml`. Each names its `dataset_id`, a
`source:` block, the dimension columns and their bounds, the ordered measure
columns, and optional `null_string` / `drop_if`.

| dataset | source | dims | measures |
| --- | --- | --- | --- |
| `synth10_1M` … `synth10_1B` | generator (seed 0; rows 1M…1B) | `col0,col1` | `col2…col9` (uniform[0,1000]) |
| `taxi` | parquet (real) | `pickup_lon,pickup_lat` | `fare_amount,total_amount,trip_distance,passenger_count` |
| `gaia_dr3` | parquet (real) | `ra,dec` | `parallax,phot_g_mean_mag,pmra,pmdec,phot_bp_mean_mag,phot_rp_mean_mag,ruwe,astrometric_excess_noise` |
| `ebird_us` | parquet (real) | `LONGITUDE,LATITUDE` | `DURATION MINUTES,EFFORT DISTANCE KM,NUMBER OBSERVERS,OBSERVATION COUNT` |

**Generator-produced vs real:** the `synth10_*` family is generated from its
config alone (no external file). `taxi`, `gaia_dr3`, and `ebird_us` are real and
require their Parquet to be produced first (see below). Measure order is
significant — the experiment layer's "number of measures" sweep takes the first
*k* measures in the listed order — so do not reorder them. Column names that
contain spaces (the eBird measures) are matched verbatim against the Parquet
schema and are quoted in the config.

### Per-synth seed/params

Every `synth10_*` config uses `seed: 0` and ten columns, each drawn uniformly
from `[0, 1000)`: `col0`/`col1` are the dimensions, `col2…col9` the
eight measures. Only the generated `rows` differs between sizes (1M, 50M, 100M,
300M, 500M, 1B). Generation is deterministic, so each size's Parquet is
reproducible from its config alone.

## Source datasets and subsets

The real datasets are public; the benchmark uses the following subsets.

- **taxi** — NYC Yellow Cab trip records for 2013–2014, merged and pre-cleaned
  into a single CSV with a header row. Dimensions are pickup longitude/latitude.
- **gaia_dr3** — a ~10% deterministic sample of `gaiadr3.gaia_source`
  (`MOD(source_id, 10) = 0`), shuffled once.
  `scripts/download_gaia_benchmark.py` pulls it from the ESA Gaia Archive TAP
  service in 1-degree RA strips and writes a header row; see its `--help` for
  sampling fractions and resume options. The 14 columns it emits, in order, are:
  `ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra,
  pmdec, ruwe, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux,
  phot_rp_mean_flux, astrometric_excess_noise`. (An older export made without a
  header row needs the header prepended once — see below.)
- **ebird_us** — the US eBird Basic Dataset filtered to continental US
  observations in 2023–2026. `scripts/filter_ebird_us.sh <input> <output>`
  applies the year + bounding-box filter while preserving the header row, so the
  native eBird column names carry through unchanged.

## Producing Parquet for the real datasets

Run `csv_to_parquet` once per real dataset, then point the config's `parquet:`
path at the output. The commands below write to `/path/to/parquet/<id>/` as a
stand-in — substitute the location where you keep that dataset's Parquet, and
set the config's `parquet:` line to the same path.

`csv_to_parquet` carries the source's header names through unchanged, so each
source must already have a header row whose names match the dataset config. Two
of the three do; gaia needs a one-time header prepend.

```sh
# taxi
build/tools/csv_to_parquet \
  --input  /path/to/yellow_tripdata_2013_2014_cleaned.csv \
  --output /path/to/parquet/taxi/yellow_tripdata_2013_2014_cleaned.parquet \
  --has-header --delimiter ,

# gaia_dr3
build/tools/csv_to_parquet \
  --input  /path/to/gaia_dr3_benchmark_shuffled_h.csv \
  --output /path/to/parquet/gaia_dr3/gaia_dr3_benchmark_shuffled.parquet \
  --has-header --delimiter ,

# ebird_us
build/tools/csv_to_parquet \
  --input  /path/to/ebd_US_2023_2026.txt \
  --output /path/to/parquet/ebird_us/ebd_US_2023_2026.parquet \
  --has-header --delimiter tab --null-string X
```

A headerless source converted with no `--has-header` still works, but its
columns get positional placeholder names (`col0`, `col1`, …) that will not
match a config — so add a real header first whenever the config names matter.

## Preparing datasets

```sh
# one dataset
python3 scripts/prepare_dataset.py configs/datasets/taxi.yaml

# several
python3 scripts/prepare_dataset.py configs/datasets/synth10_1M.yaml configs/datasets/taxi.yaml

# everything under configs/datasets/ in one batch
python3 scripts/prepare_dataset.py --all
```

Useful options:

- `--prepared-root DIR` — output root (default `data/prepared`, or
  `A3I_PREPARED_ROOT`); a dataset lands in
  `<root>/<id>/{manifest.json, columns/...}`.
- `--parquet-dir DIR` — Parquet root (default `data/raw`, or `A3I_PARQUET_DIR`);
  generator Parquet files are written/reused here, and any relative
  `source.parquet` resolves here (real configs use absolute paths).
- `--max-rows N` — single-config only: prepare the first N surviving rows as a
  subset dataset `<id>_<N>` (the manifest records `parent_dataset_id` + `max_rows`).
  Use this to make small real-data slices for tests/CI.
- `--force` — re-generate and re-convert even if up to date.
- `--verify {full,structural,none}` — `full` (default) recomputes global stats
  from the binary columns; `structural` only checks file sizes/manifest (use for
  very large datasets where a full Python re-read is impractical); `none` skips.

### Idempotency

Preparation is idempotent. A dataset whose prepared `manifest.json` already
records the current source Parquet's `(source_bytes, source_mtime)` is left
untouched unless `--force` is passed; a generator Parquet that already exists is
reused. So `--all` re-runs cheaply and only re-prepare what changed.

## Verifying a prepared dataset

`prepare_dataset.py` verifies as its last step. To re-verify without converting,
re-run the same command — an up-to-date dataset is skipped but still verified.
The smoke test `tests/test_data_prep_smoke.py` (CTest label `m_data_prep`)
exercises the whole flow on tiny fixtures, including the `drop_if`+bounds
survival rule and `null_string` → NaN, cross-checking the manifest stats against
an independent re-derivation.

## Disk footprint note

Each synthetic cardinality is materialized as its own full prepared dataset, so
`synth10_1B` alone is large (10 columns × 8 bytes × 1e9 ≈ 80 GB of binary plus
its Parquet). Prepare only the sizes you need, and point `--prepared-root` /
`--parquet-dir` at a disk with room.
