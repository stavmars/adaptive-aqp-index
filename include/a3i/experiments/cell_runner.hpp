// Running one experiment cell: a single (method, dataset, workload, run).
//
// A cell binds a prepared dataset and a workload of rectangles to one named
// method, replays every query through that method's engine (or the exact-scan
// oracle for the `scan` method), and writes two files: a results CSV with one
// row per query (the per-measure aggregate estimates and the query rectangle
// each packed into a single JSON column, so the column set is fixed regardless
// of measure count or dimensionality) and a JSON sidecar recording the resolved
// configuration and provenance. Downstream validation and analysis explode the
// JSON columns and join by query ordinal.
//
// A method name composes a substrate with a behavior:
//   scan          -> exact-scan oracle (no substrate)
//   kd            -> static_kd   + plain
//   kd_agg        -> static_kd   + aggregating
//   adkd          -> adaptive_kd + plain
//   adkd_agg      -> adaptive_kd + aggregating
//   adkd_sampling -> adaptive_kd + sampling
//   a3i           -> adaptive_kd + accuracy-aware
//
// The measure subset (`num_measures`, the first-k of the dataset's measures)
// is applied at the store boundary; the engine stays oblivious to it. The
// accuracy target (`error_bound`, `confidence`) is applied per query at run
// time — the workload rows carry only geometry.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "a3i/storage/binary_column_store.hpp"  // MeasureStorage

namespace a3i {

struct CellConfig {
    std::filesystem::path manifest_path;   ///< Prepared dataset manifest.json.
    std::filesystem::path workload_path;   ///< Workload rectangles CSV.
    std::string           method;          ///< One of the names listed above.
    std::filesystem::path qresults_path;   ///< Output results CSV.
    std::filesystem::path runmeta_path;    ///< Output JSON sidecar.

    /// Result-file labels. Empty `dataset_id` falls back to the manifest's id;
    /// empty `workload_name` falls back to the workload file stem.
    std::string dataset_id;
    std::string workload_name;

    /// First-k measures to serve; 0 means every measure in the manifest.
    std::size_t num_measures = 0;

    /// Per-query accuracy target. `error_bound <= 0` requests exact answers;
    /// the exact methods ignore it regardless.
    double error_bound = 0.0;
    double confidence  = 0.95;

    /// Substrate construction knobs.
    std::uint32_t partition_size       = 1024;
    bool          stochastic_cracking  = false;

    /// Sort each round's gathered row ids ascending before reading measures
    /// (gather locality). Recorded in the sidecar so results built under
    /// different gather orders are never silently aggregated or compared.
    bool          sort_gather_by_row_id = true;

    /// How the measure columns are backed: OnDisk (out-of-core, the default)
    /// or Eager (loaded fully resident, the in-memory regime). The
    /// orchestrator derives this from the `mem` axis (`inmem` => Eager); it is
    /// recorded in the sidecar for self-description. Eager is only valid
    /// uncapped.
    MeasureStorage measure_storage = MeasureStorage::OnDisk;

    /// Outer repetition index; also the recorded sampling seed and the seed
    /// material that makes each run's sampling draws distinct yet reproducible.
    std::uint64_t run_id = 0;

    /// Replay only the first `max_queries` of the workload; 0 means all.
    std::size_t max_queries = 0;

    /// Recorded in the sidecar; the page cache is dropped by the launcher, not
    /// here, so this only stamps whether the run was meant to be cold.
    bool cold = true;
};

struct CellReport {
    std::filesystem::path qresults_path;
    std::filesystem::path runmeta_path;
    std::size_t           queries_executed = 0;
    std::size_t           measures_served  = 0;
};

/// The results-CSV header (one line, no trailing newline). Downstream
/// validation and analysis depend on the exact column set and order.
const char* qresults_header();

/// One catalog entry per selectable method. `substrate` is the label written
/// to the results path and the `substrate` column (`n/a` for the scan oracle);
/// `approx` is true for the methods that consume the per-query error bound (so
/// the orchestrator fans those out over the eb axis and exact methods do not).
/// This is the single source of truth for the method set: tools and scripts
/// that need the method->substrate / approx mapping read it from here (via
/// `a3i_run --describe-methods`) rather than restating it.
struct MethodInfo {
    std::string name;
    std::string substrate;
    bool        approx = false;
};

const std::vector<MethodInfo>& method_catalog();

/// Run the cell described by `config`, writing the results CSV and the sidecar.
/// Throws std::invalid_argument on an unknown method or a malformed workload,
/// std::runtime_error on I/O failure.
CellReport run_cell(const CellConfig& config);

}  // namespace a3i
