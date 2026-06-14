# Experiment Orchestration and Validation

This document describes the experiment pipeline for A3I, including workload and plan configuration, result structure, and validation workflow. It covers how to run large-scale experiments, interpret results, and ensure reproducibility and correctness.

## Overview

A3I experiments are orchestrated via Python scripts and YAML configuration files. The pipeline supports flexible axes (methods, error bounds, measure counts, memory, etc.),and idempotent result generation. All results are written in a CSV schema, with one row per query and JSON-encoded columns for query rectangles and aggregates.

## Key Components

- **Workload Catalogs** (`configs/workloads/*.yaml`): Define query workloads for each dataset, including random and clustered families, selectivity, focus, and count.
- **Experiment Plans** (`experiments/plans/*.yaml`): Specify experiment axes (methods, error bounds, measure counts, memory, etc.) and enumerate all cells to run.
- **Orchestration Script** (`scripts/run_experiments.py`): Drives the execution of all experiment cells, memory limits, and result manifest stamping.
- **Validation Script** (`scripts/validate_results.py`): Consumes result CSVs, checks query results, and emits validation summaries.


## Running Experiments

**Prerequisites.** Build the C++ tools first — the scripts shell out to
`a3i_run`, `generate_workload`, and the converters under `build/`:
```sh
cmake -S . -B build -G Ninja && cmake --build build
```

**Directory configuration.** Every script resolves the prepared-data, source-Parquet,
results, and workload roots in this order: **CLI flag > environment variable >
a repo-root `.env` file > repo-relative default** (`data/prepared`, `data/raw`,
`experiments/results`, `experiments/workloads`). Set machine-specific paths once
in a git-ignored `.env` (copy `.env.example`) instead of passing flags every
time:
```sh
# .env
A3I_PREPARED_ROOT=/path/to/prepared
A3I_PARQUET_DIR=/path/to/parquet
# A3I_RESULTS_ROOT / A3I_WORKLOADS_DIR default under the repo
```
The explicit `--prepared-root`/`--results-root` flags below are then optional and
override the `.env`/defaults when given.

1. **Prepare Datasets:**
   - Use `scripts/prepare_dataset.py` to generate or convert datasets into the prepared format.
2. **Configure Workloads and Plans:**
   - Edit or extend YAML files in `configs/workloads/` and `experiments/plans/` to define new workloads or experiment axes.
3. **Run Experiments:**
   - With roots set in `.env` (above), the common case is just the plan id:
     ```sh
     python3 scripts/run_experiments.py --plan <plan_id>
     ```
   - `--plan` accepts several ids, and `--all` runs every plan except the dev
     ones (`_defaults`, `quicktest`, `smoke`). Cells shared across plans are
     computed once (a cell already on disk is skipped), so `--all` runs the
     union of distinct cells, not the sum of per-plan counts:
     ```sh
     python3 scripts/run_experiments.py --plan eb_sweep nm_sweep
     python3 scripts/run_experiments.py --all
     ```
   - The fully-explicit form (flags override `.env`/defaults):
     ```sh
     python3 scripts/run_experiments.py --plan <plan_id> --plans-dir experiments/plans --workload-config-dir configs/workloads --prepared-root <prepared_dir> --results-root <results_dir>
     ```
   - Options: `--dry-run`, `--force`, `--stale`, `--warm`, `--drop-caches-cmd`, `--mem-launcher`, `--report`.
   - Results are written to `results/<dataset>/<workload>/<substrate_dir>/<method>/qresults_<key>_run<R>.csv` and a sibling `runmeta_...json`. Each cell's disposition is appended (as it finishes) to a durable per-run log under `results/_runlog/<plan>_<stamp>.jsonl`, so an interrupted run still leaves a record.
   - **Completeness:** `--report` (runs nothing) diffs the plan's expected cells against what is on disk and lists any missing ones, labelling the reason (`error`/`oom`/`never-run`) from the run log. Use it after a long run to confirm nothing silently dropped out.
4. **Validate Results:**
   - Run:
     ```sh
     python3 scripts/validate_results.py --results-root <results_dir> --validation-root <validation_dir>
     ```
   - Produces `validation_summary.csv` and `validation_details.csv` per (dataset, workload).


## Result Schema

One row per query, 28 columns. The authoritative header is `kHeader` in
`src/experiments/cell_runner.cpp`; the field semantics are documented on
`QueryMetrics` in `include/a3i/core/query.hpp`. Columns in order:

| # | Column | Meaning |
|---|--------|---------|
| 1 | `query_ordinal` | 0-based position of the query within its workload. |
| 2 | `method` | Run name (e.g. `a3i`, `scan`, `adkd_sampling`). |
| 3 | `substrate` | Substrate id (e.g. `adaptive_kd`), or `n/a`/`none` for the scan oracle. |
| 4 | `dataset` | Dataset label. |
| 5 | `workload` | Workload label. |
| 6 | `query_rect` | JSON `{ "lower": [...], "upper": [...] }` — the predicate rectangle. |
| 7 | `aggregates` | JSON array, one object per (aggregate, measure); see below. |
| 8 | `target_satisfied` | `true` iff every aggregate met its accuracy target. |
| 9 | `status` | `exact`, `converged`, `exactified`, or `exhausted_unconverged`. |
| 10 | `exactify_cause` | `none`, `cheaper_to_exactify`, or `gave_up` — why the residual was read in full. |
| 11 | `pre_exactification_error_bound` | Worst relative half-width just before exactification (0 if none). |
| 12 | `sampling_seed` | Seed material for the run (the run id). |
| 13 | `latency_ms` | Wall-clock of the query (excludes the one-time build; see `init_ms` in runmeta). |
| 14 | `measure_reads` | Total measure values gathered by row id (rows × measures): the I/O cost. |
| 15 | `sampled_rows` | Distinct rows read by without-replacement sampling. |
| 16 | `exactified_rows` | Distinct rows read by exhausting a stratum to exact (scan oracle: all qualifying rows). |
| 17 | `frontier_partitions` | The partitions the answer is actually assembled from — where the top-down walk stops, one group per region that is wholly inside the query or a boundary remainder. Not the total number of partitions in the index, nor everything the walk inspected; the four columns below are its breakdown and sum to it. |
| 18 | `partitions_refined` | How many boundary partitions were *cracked* for this query. |
| 19 | `exact_contributors` | Wholly-inside partitions already known exactly from an earlier query (read in full before); contribute with **no reads**. The strongest form of reuse. |
| 20 | `reusable_sampled_strata` | Wholly-inside partitions an earlier query *partially* sampled; the partial sample is reused and extended only if this query needs a tighter answer. Partial reuse. |
| 21 | `reusable_absent_strata` | Wholly-inside partitions nothing is known about yet (just created, or never sampled); sampled from scratch. No reuse. Over a session a partition migrates absent → sampled → exact, which is why later queries get cheaper. |
| 22 | `query_local_strata` | Boundary partitions the query only partially covers (not worth cracking further). Only the in-rectangle subset is used, and since that selection is specific to this rectangle, its samples are discarded afterward and cannot help a future query. Reuse does not apply here. |
| 23 | `adaptive_rounds` | Number of sampling/exactify rounds the engine loop ran. |
| 24 | `scan_path_rows` | On-disk only: distinct wanted rows the cost model served via the **sequential-scan** path (vs scattered gather). A decision count, not bytes. `0` for in-memory (eager) cells. |
| 25 | `gather_path_rows` | On-disk only: distinct wanted rows served via the **scattered-gather** path. `0` in-memory. |
| 26 | `scan_bytes_read` | On-disk only: device bytes the scan path actually moved across all measures. Far exceeds the wanted rows — a scan reads its whole `[min,max]` row span — so this, not column 24, is the real scan I/O cost. `0` in-memory. |
| 27 | `gather_bytes_read` | On-disk only: device bytes the gather path actually moved (whole pages, so it carries page read-amplification). `0` in-memory. |
| 28 | `round_paths` | On-disk only: JSON array, one object per round that read from disk — `{ "r": round, "sr": scan_rows, "gr": gather_rows, "sb": scan_bytes, "gb": gather_bytes }`. Reveals how many rounds scanned vs gathered (a query that scans in several rounds re-reads its span each time). `[]` in-memory. |

The four contributor counts (columns 19–22) are per **partition** (not per
(partition, measure)) and **sum to `frontier_partitions`**. Two identities hold
for every method and are checked by `validate_results.py`:

- `measure_reads == (sampled_rows + exactified_rows) × num_measures`
- `frontier_partitions == exact_contributors + reusable_sampled_strata + reusable_absent_strata + query_local_strata`

The access-path columns (24–28) are on-disk telemetry only (an eager/in-memory
store has no I/O path, so all five are `0`/`[]`). They are not checked by the
validator, but a third identity holds by construction:

- `scan_path_rows + gather_path_rows == sampled_rows + exactified_rows` (the wanted rows, split by access path). The `*_bytes_read` columns measure actual device traffic and are unrelated to this row identity; `(scan_bytes_read + gather_bytes_read) / (measure_reads × 8)` is the read amplification (≫ 1 on scattered data).

- **JSON Columns:**
  - `query_rect`: `{ "lower": [...], "upper": [...] }`
  - `aggregates`: Array of `{ aggregate, measure, estimate, ci_low, ci_high, relative_half_width, effective_df, exact }`
- **Manifest:** Each run writes a sibling `runmeta_*.json` with the fully-resolved configuration and environment details.

## Axes and Cell Key

- **Axes:** Methods, error bounds, measure counts, memory, stride, run ID, etc. Plans enumerate all combinations.
- **Cell Key:** Each result cell is uniquely identified by a sorted join of axes (measures, memory, stride, n, error bound if approximate).
- **Idempotency:** Re-running a cell with the same configuration and seed produces identical results.

## Memory Limits and Cold Cache

- **Memory Cap:** A cell may declare a memory budget (e.g. `mem16G`). The budget
  is enforced by wrapping each run in a cgroup launcher (`--mem-launcher`,
  default: a transient `systemd-run --scope` with `MemoryMax` and swap disabled).
  A cgroup is used rather than a process address-space limit because the columns
  are memory-mapped: their page cache must count against the budget, and only the
  cgroup memory controller charges file-backed page cache to the run. A run that
  exceeds its budget is killed and recorded as `status=error`; the
  rest of the matrix continues. Capping requires cgroup delegation or privilege,
  so a cell with a memory budget will refuse to run if no working launcher is
  configured. Unbudgeted cells run with no cap.
- **Cold Cache:** Runs are cold by default. Before each cell the dataset's
  column files are evicted from the page cache so the memory-mapped columns are
  read from disk rather than from a cache warmed by an earlier run; `runmeta`
  records `cold=true`. Eviction is unprivileged: the runner calls
  `posix_fadvise(POSIX_FADV_DONTNEED)` on exactly the columns the cell maps, so
  no `sudo` and no whole-machine flush are needed. For a stricter baseline that
  also clears unrelated pages, supply a whole-machine drop hook via
  `--drop-caches-cmd` or `A3I_DROP_CACHES_CMD` (typically a privileged helper
  that writes `vm.drop_caches`); it runs in addition to the per-dataset
  eviction. `--warm` skips eviction entirely and records `cold=false` (dev
  only).

## Validation

Every cell is validated; failures and warnings are collected and reported at the
end rather than stopping the run.

- **Exact Runs:** Compared against the scan oracle. `COUNT` and `COUNT(*)` must
  match bit-for-bit; `SUM` and `AVG` must agree within a small relative
  tolerance (`1e-6 * max(1, |oracle|)`).
- **Approximate Runs:** Each approximate aggregate is scored: the relative error
  against the oracle is checked against the run's error bound, and coverage is the
  fraction of aggregates whose reported `[ci_low, ci_high]` interval contains the
  oracle value (`approx_within_eb_frac`, `coverage_frac`, `max_rel_err`).
  Coverage is also judged: each sampled aggregate is one Bernoulli trial whose
  success probability is the run's nominal confidence when the interval is well
  calibrated, so the *expected* coverage fraction is that confidence (e.g. ~0.95).
  Because a finite run fluctuates around that expectation, a one-sided lower-tail
  binomial test (not a fixed cutoff) flags only a *systemic* shortfall, recorded
  as `coverage_status` (`ok` / `low` / `insufficient` / `n/a`). Tune the test
  level with `--coverage-alpha`.
- **Failure tiers:** Hard failures — an exact mismatch, a same-query guard
  violation, or a missing/incomplete oracle — are deterministic correctness
  regressions and always cause a non-zero exit. A `low` coverage status is
  statistical: it is reported as a **warning** by default, and `--strict-coverage`
  escalates it to a failing exit.
- **Same-Query Guard:** Ensures all methods answer the same set of queries (by
  rectangle); a divergent rectangle fails validation.

## Usage Example

```sh
python3 scripts/run_experiments.py --plan headline --prepared-root /data/prepared --results-root /data/results
python3 scripts/validate_results.py --results-root /data/results --validation-root /data/validation
```
